# src/webapp.py

from __future__ import annotations

import io
import os
import time
import html
from typing import Optional, Iterable

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import LanceDBConfig, MinioConfig
from src.lancedb_manager import LanceDBManager
from src.search import MultiModalSearcher
from src.storage import MinIOClient
from src.cli_utils import mask

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
TABLE_DEFAULT = os.getenv("WEBAPP_TABLE", "multimodal")
SHOW_IMAGES = os.getenv("WEBAPP_SHOW_IMAGES", "1").lower() not in ("0", "false")
MAX_UPLOAD_BYTES = int(os.getenv("WEBAPP_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8MiB
# TTL for presigned URLs used in result previews/links
PRESIGN_TTL = int(os.getenv("WEBAPP_PRESIGN_TTL", "3600"))
ALLOW_LIDAR = os.getenv("WEBAPP_ALLOW_LIDAR", "1").lower() not in ("0", "false")

# Default metric for searches (web + api). Accepts l2 | cosine | dot
DEFAULT_METRIC = os.getenv("LANCEDB_METRIC", os.getenv("WEBAPP_DEFAULT_METRIC", "l2")).strip().lower()

# Query-time ANN tunables (defaults optimize recall without too much latency)
try:
    LANCEDB_NPROBES = int(os.getenv("LANCEDB_NPROBES", "32"))
except Exception:
    LANCEDB_NPROBES = 32
try:
    LANCEDB_REFINE = int(os.getenv("LANCEDB_REFINE_FACTOR", "50"))
except Exception:
    LANCEDB_REFINE = 50

app = FastAPI(title="LanceDB Search UI", version="1.2")
# Minimal CORS (adjust `WEBAPP_CORS_ORIGINS` in env for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WEBAPP_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Shared LanceDB manager at app startup for connection reuse
@app.on_event("startup")
def _startup_connect():
    try:
        lcfg = LanceDBConfig()
        mgr = LanceDBManager(lcfg)
        try:
            mgr.connect(lcfg.uri, lcfg.storage_options())
        except Exception:
            pass
        app.state.lance_mgr = mgr
    except Exception:
        pass

def _get_manager() -> LanceDBManager:
    mgr = getattr(app.state, "lance_mgr", None)
    if mgr is None:
        lcfg = LanceDBConfig()
        mgr = LanceDBManager(lcfg)
        try:
            mgr.connect(lcfg.uri, lcfg.storage_options())
        except Exception:
            pass
        app.state.lance_mgr = mgr
    return mgr

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _escape(s: str) -> str:
    return html.escape(s, quote=True)

def _snippet(text: str, n: int = 160) -> str:
    if not isinstance(text, str):
        return ""
    s = " ".join(text.split())
    return s[:n] + ("…" if len(s) > n else "")

def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    if not uri or not uri.startswith("s3://"):
        return None
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key

def _presign_with_client(path: str, s3: Optional[MinIOClient]) -> str | None:
    """Return a presigned HTTPS URL for s3://bucket/key using a provided client (or None)."""
    if not (SHOW_IMAGES and s3 and path and path.startswith("s3://")):
        return None
    parsed = _parse_s3_uri(path)
    if not parsed:
        return None
    b, k = parsed
    try:
        return s3.get_presigned_url(b, k, expires=PRESIGN_TTL)
    except Exception:
        return None

def _load_lidar_bytes(ext: str, blob: bytes) -> np.ndarray:
    """Lightweight LiDAR loader for queries (avoids heavy ingest imports).
    Supports .bin (float32 packed) and a best-effort .pcd ASCII fallback.
    """
    if ext == ".bin":
        try:
            arr = np.frombuffer(blob, dtype=np.float32)
            cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
            return arr.reshape(-1, cols)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
    if ext == ".pcd":
        try:
            text = blob.decode("utf-8", errors="ignore").strip().split()
            arr = np.asarray([float(x) for x in text], dtype=np.float32)
            cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
            return arr.reshape(-1, cols)[:, :3]
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)

def _index_summary(tbl: object) -> str:
    try:
        li = getattr(tbl, "list_indices", None) or getattr(tbl, "list_indexes", None)
        if callable(li):
            res = li()
            if isinstance(res, (list, tuple)):
                return "present" if len(res) > 0 else "none"
            return "present"
    except Exception:
        pass
    return "unknown"

def _ensure_len(blob: bytes) -> None:
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded file exceeds {MAX_UPLOAD_BYTES} bytes.")

def _vectorize_query(
    searcher: MultiModalSearcher,
    modality: str,
    q_text: Optional[str],
    q_path: Optional[str],
    img_blob: Optional[bytes],
    lidar_blob: Optional[bytes],
    lidar_ext: Optional[str],
) -> list[float]:
    if modality == "text":
        if not q_text:
            raise ValueError("Text query is empty.")
        return searcher.text_embedder.embed(q_text).tolist()

    if modality == "image":
        if img_blob:
            return searcher.image_embedder.embed(img_blob).tolist()
        if q_path:
            with open(q_path, "rb") as f:
                return searcher.image_embedder.embed(f.read()).tolist()
        raise ValueError("Provide an image file or a readable server path.")

    if modality == "lidar":
        if not ALLOW_LIDAR:
            raise ValueError("LiDAR queries are disabled.")
        if lidar_blob:
            pts = _load_lidar_bytes(lidar_ext or "", lidar_blob)
            return searcher.lidar_embedder.embed(pts).tolist()
        if q_path:
            _, ext = os.path.splitext(q_path)
            with open(q_path, "rb") as f:
                pts = _load_lidar_bytes(ext.lower(), f.read())
            return searcher.lidar_embedder.embed(pts).tolist()
        raise ValueError("Provide a LiDAR .pcd/.bin file or a readable server path.")

    raise ValueError(f"Unknown modality: {modality}")

def _results_table(df, s3: Optional[MinIOClient], metric: str) -> str:
    if df is None or len(df) == 0:
        return "<p>No results.</p>"
    m = _norm_metric(metric)
    dist_col = "distance" if m == "l2" else "similarity"
    rows_html = []
    for _, r in df.iterrows():
        rid = str(r.get("id", ""))
        mod = str(r.get("modality", ""))
        dval = r.get("_distance", "")
        if isinstance(dval, (int, float)):
            if m == "l2":
                dist = f"{float(dval):.4f}"
            else:
                sim = 1.0 - float(dval)
                dist = f"{sim:.4f}"
        else:
            dist = str(dval)
        text = r.get("text") or ""
        path = r.get("path", "")
        preview_html = ""

        # Previews
        if SHOW_IMAGES and path:
            url = _presign_with_client(path, s3)
            if url and mod == "image":
                preview_html = f"<div class='imgbox'><img loading='lazy' src='{url}' alt='preview'></div>"
            elif url and mod == "pdf":
                preview_html = f"<div class='imgbox' title='PDF'><a href='{url}' target='_blank' rel='noreferrer'>🧾 open</a></div>"

        # Linkify path if we got a URL; otherwise plain text
        url_for_path = _presign_with_client(path, s3)
        path_cell = (
            f"<a class='break' href='{url_for_path}' target='_blank' rel='noreferrer' title='{_escape(path)}'>{_escape(path)}</a>"
            if url_for_path
            else f"<span class='break' title='{_escape(path)}'>{_escape(path)}</span>"
        )

        rows_html.append(
            "<tr>"
            f"<td class='preview'>{preview_html}</td>"
            f"<td class='truncate' title='{_escape(rid)}'>{_escape(rid)}</td>"
            f"<td class='mod'>{_escape(mod)}</td>"
            f"<td class='dist'>{dist}</td>"
            f"<td class='truncate' title='{_escape(text)}'>{_snippet(text)}</td>"
            f"<td class='break'>{path_cell}</td>"
            "</tr>"
        )

    results_html = (
        "<table class='table'>"
        "<thead><tr>"
        "<th class='preview'>preview</th><th>id</th><th class='mod'>modality</th>"
        f"<th class='dist'>{dist_col}</th><th>snippet</th><th>path</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html) +
        "</tbody></table>"
    )
    return results_html

def _where_from_filter(f: str) -> Optional[str]:
    if f == "pdf":
        return "modality = 'pdf'"
    if f == "image":
        return "modality = 'image'"
    if f == "text":
        return "modality = 'text'"
    if f == "lidar":
        return "modality = 'lidar'"
    return None

def _get_rowcount(tbl) -> str:
    try:
        return str(len(tbl))
    except Exception:
        try:
            return str(tbl.count_rows())
        except Exception:
            return "?"

def _norm_metric(m: str) -> str:
    m = (m or "l2").strip().lower()
    if m in ("euclidean", "l2", "l2_distance"):
        return "l2"
    if m in ("cos", "cosine"):
        return "cosine"
    if m in ("dot", "ip", "inner_product", "dot_product"):
        # embeddings are L2-normalized → dot is equivalent to cosine
        return "cosine"
    return "l2"

def _metric_slider_defaults(metric: str) -> tuple[float, float, float, float, str]:
    """Return (min, max, step, default_value, label) for the metric threshold slider.
    Defaults are non-restrictive (allow all results).
    - l2: distance in [0, 2]; default 2.0 (max distance)
    - cosine/dot: similarity in [-1, 1]; default -1.0 (min similarity)
    """
    m = _norm_metric(metric)
    if m == "l2":
        return 0.0, 2.0, 0.01, 2.0, "Max L2 distance"
    # cosine (and dot): control similarity threshold
    return -1.0, 1.0, 0.01, -1.0, "Min cosine/dot similarity"

def _apply_metric_threshold(df, metric: str, val: Optional[float]):
    """Filter DataFrame rows by metric threshold (client-side).
    - For l2: keep rows with _distance <= val (if val provided)
    - For cosine/dot: interpret val as min similarity s; distance d = 1 - s; keep _distance <= d
    If val is None, no filtering.
    """
    try:
        if df is None or val is None:
            return df
        m = _norm_metric(metric)
        if m == "l2":
            return df[df.get("_distance", 0) <= float(val)]
        # cosine/dot: convert similarity threshold to distance cutoff
        dmax = 1.0 - float(val)
        return df[df.get("_distance", 0) <= dmax]
    except Exception:
        return df

# --------------------------------------------------------------------------------------
# HTML template (responsive, no template engine needed)
# --------------------------------------------------------------------------------------
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LanceDB Search</title>
<style>
:root{{--b:#111827;--m:#6b7280;--p:#2563eb;--bd:#e5e7eb;--bg:#f9fafb}}
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:var(--bg);color:var(--b)}}
.container{{max-width:1100px;margin:0 auto;padding:24px}}
.card{{background:#fff;border:1px solid var(--bd);border-radius:12px;padding:16px;margin-bottom:16px}}
h1{{font-size:1.5rem;margin:0 0 12px}}
h2{{font-size:1.1rem;margin:0 0 8px;color:var(--m)}}
.row{{display:grid;grid-template-columns:1fr;gap:12px}}
@media(min-width:900px){{.row{{grid-template-columns:2fr 1fr 1fr}}}}
input[type=text],select,button{{width:100%;padding:.6rem;border:1px solid var(--bd);border-radius:8px;background:#fff}}
button{{background:var(--p);border-color:var(--p);color:#fff;cursor:pointer}}
small{{color:var(--m)}}
.flex{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.badge{{display:inline-block;padding:.15rem .45rem;border:1px solid var(--bd);border-radius:6px;background:#fff;font-size:.85rem}}
.table{{width:100%;border-collapse:collapse;table-layout:fixed}}
thead th{{position:sticky;top:0;background:#fff}}
th,td{{border-bottom:1px solid var(--bd);padding:.6rem;text-align:left;vertical-align:top}}
th.preview,td.preview{{width:220px}}
th.mod,td.mod{{width:90px}}
th.dist,td.dist{{width:90px}}
.truncate{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.break{{word-break:break-all}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.err{{color:#b91c1c}}
.imgbox{{width:200px;height:140px;display:flex;align-items:center;justify-content:center;border:1px solid var(--bd);border-radius:8px;overflow:hidden;background:#fafafa}}
.imgbox img{{max-width:100%;max-height:100%;object-fit:cover}}
.kv{{display:grid;grid-template-columns:180px 1fr;gap:6px;margin-top:8px}}
.kv div{{padding:2px 0;border-bottom:1px dashed #eee}}
details>summary{{cursor:pointer;user-select:none;color:var(--m)}}

/* Segmented control for modality */
.seg{{display:flex; gap:6px; background:var(--bg); padding:4px; border:1px solid var(--bd); border-radius:8px}}
.seg-item{{position:relative}}
.seg-item input{{position:absolute; opacity:0; pointer-events:none}}
.seg-item span{{display:inline-block; padding:.45rem .7rem; border-radius:6px; border:1px solid transparent; color:var(--m); background:transparent}}
.seg-item input:checked + span{{background:var(--p); color:#fff; border-color:var(--p)}}
.hint{{color:var(--m); font-size:.82rem; margin-top:4px}}
.note{{color:var(--m); font-size:.9rem}}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>LanceDB Search</h1>
    <div class="flex">
      <div class="badge">Table: <b>{table}</b></div>
      <div class="badge">Rows: <b>{rowcount}</b></div>
      <div class="badge">Duration: <b>{elapsed_ms} ms</b></div>
      <div class="badge">Images: <b>{show_images}</b></div>
      <div class="badge">Metric: <b>{metric_name}</b></div>
      <div class="badge">Probes: <b>{nprobes}</b></div>
      <div class="badge">Refine: <b>{refine}</b></div>
      <div class="badge">Index: <b>{index_summary}</b></div>
    </div>
  </div>

  <div class="card">
    <h2>Query</h2>

    <form method="post" action="/search" enctype="multipart/form-data">
      <div class="row" style="margin-bottom:8px">
        <div>
          <label>Modality</label>
          <div class="seg" role="tablist" aria-label="Modality">
            <label class="seg-item">
              <input type="radio" name="modality" value="text" {m_text}><span>Text</span>
            </label>
            <label class="seg-item">
              <input type="radio" name="modality" value="image" {m_img}><span>Image</span>
            </label>
            {lidar_seg}
          </div>
          <div class="hint">Choose the kind of query you’ll send</div>
        </div>

        <div>
          <label>Top-K</label>
          <input type="number" name="topk" min="1" max="2000" step="1" value="{topk}">
          <div class="hint">Enter how many results to show</div>
        </div>

        <div>
          <label>Filter</label>
          <select name="filter">
            <option value="all" {f_all}>All</option>
            <option value="text" {f_text}>Text only</option>
            <option value="pdf" {f_pdf}>PDFs only</option>
            <option value="image" {f_image}>Images only</option>
            <option value="lidar" {f_lidar}>LiDAR only</option>
          </select>
          <div class="hint">Quick pre-filter</div>
        </div>
      </div>

      <div class="row" style="margin-bottom:8px">
        <div>
          <label>Metric</label>
          <select name="metric">
            <option value="l2" {m_l2}>Euclidean (L2)</option>
            <option value="cosine" {m_cos}>Cosine</option>
            <option value="dot" {m_dot}>Dot Product</option>
          </select>
          <div class="hint">Cosine/Dot recommended; embeddings are L2-normalized</div>
        </div>
      </div>

      <!-- Metric threshold -->
      <div class="row" style="margin-bottom:8px">
        <div>
          <label id="metric_val_label">{metric_label}</label>
          <div class="flex" style="gap:10px">
            <input type="range" name="metric_val" min="{m_min}" max="{m_max}" step="{m_step}" value="{m_val}" oninput="document.getElementById('metric_val_num').value=this.value">
            <input id="metric_val_num" type="number" name="metric_val_num" min="{m_min}" max="{m_max}" step="{m_step}" value="{m_val}" oninput="document.querySelector('input[name=metric_val]').value=this.value">
          </div>
          <div class="hint">Slider adapts to chosen metric</div>
        </div>
      </div>

      <!-- Text query -->
      <div class="mod-section" data-mod="text" style="margin-top:4px">
        <label>Text Query</label>
        <input type="text" name="q_text" placeholder="e.g. 'beethoven', 'invoice', 'lidar map'…" value="{q_text}">
        <div class="hint">Best for most searches</div>
      </div>

      <!-- Image query -->
      <div class="row mod-section" data-mod="image" style="margin-top:8px">
        <div>
          <label>Image File (upload)</label>
          <input type="file" name="image_file" accept="image/*">
          <div class="hint">We’ll embed the uploaded image</div>
        </div>
        <div>
          <label>Image Path (server-local)</label>
          <input type="text" name="image_path" placeholder="/path/to/image.jpg" value="{image_path}">
          <div class="hint">Optional, if the file already exists on this server</div>
        </div>
      </div>

      <!-- LiDAR query -->
      <div class="row mod-section" data-mod="lidar" style="{lidar_row}; margin-top:8px">
        <div>
          <label>LiDAR File (.pcd/.bin)</label>
          <input type="file" name="lidar_file" accept=".pcd,.bin,application/octet-stream">
          <div class="hint">Single frame point cloud</div>
        </div>
        <div>
          <label>LiDAR Path (server-local)</label>
          <input type="text" name="lidar_path" placeholder="/path/to/scan.pcd" value="{lidar_path}">
        </div>
      </div>

      <!-- Advanced -->
      <details style="margin-top:10px">
        <summary>Advanced filter (WHERE)</summary>
        <div style="margin-top:8px">
          <input type="text" name="where" placeholder="e.g. modality = 'pdf' AND path LIKE 's3://%reports%'" value="{where}">
          <div class="hint">SQL-style predicate evaluated server-side</div>
        </div>
      </details>

      <!-- Submit -->
      <div style="margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap">
        <button type="submit">Search</button>
        <span class="note">Tip: most users should stick with <b>Text</b> queries. Uploads capped at {max_bytes} bytes.</span>
      </div>
    </form>

    {error_html}
  </div>

  <div class="card">
    <h2>Results</h2>
    {results_html}
  </div>

  <div class="card">
    <h2>Config Snapshot</h2>
    <details>
      <summary>Show configuration</summary>
      <div class="kv mono" style="margin-top:8px">
        <div>URI</div><div>{lancedb_uri}</div>
        <div>S3 Endpoint</div><div>{endpoint}</div>
        <div>Region</div><div>{region}</div>
        <div>Buckets</div><div>{buckets}</div>
        <div>Access</div><div>{access}</div>
        <div>Secret</div><div>{secret}</div>
      </div>
    </details>
  </div>
</div>

<!-- Tiny script to toggle sections by modality and adapt metric slider -->
<script>
(function(){{
  const card = document.currentScript.parentElement;
  const form = card.querySelector('form');
  const metricSelect = form.querySelector('select[name="metric"]');
  const slider = form.querySelector('input[name="metric_val"]');
  const num = form.querySelector('#metric_val_num');
  function showSections(mod){{ 
    form.querySelectorAll('.mod-section').forEach(el=>{{ 
      const want = el.getAttribute('data-mod');
      el.style.display = (want===mod) ? '' : 'none';
    }});
  }}
  function currentMod(){{ 
    const el = form.querySelector('input[name="modality"]:checked');
    if (el) return el.value;
    const sel = form.querySelector('select[name="modality"]');
    return sel ? sel.value : 'text';
  }}
  function applyMetricMeta(m){{
    if(!slider || !num) return;
    if(m === 'l2'){{
      slider.min = '0.0'; slider.max = '2.0'; slider.step = '0.01';
      num.min = '0.0'; num.max = '2.0'; num.step = '0.01';
    }} else {{
      slider.min = '-1.0'; slider.max = '1.0'; slider.step = '0.01';
      num.min = '-1.0'; num.max = '1.0'; num.step = '0.01';
    }}
  }}
  form.addEventListener('change', e=>{{ 
    if(e.target && e.target.name==='modality'){{ showSections(e.target.value); }}
    if(e.target && e.target.name==='metric'){{ applyMetricMeta(e.target.value); }}
  }});
  // init
  showSections(currentMod());
  if(metricSelect) applyMetricMeta(metricSelect.value || 'l2');
}})();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(
    table: str = Query(default=TABLE_DEFAULT),
    filter: str = Query(default="all"),
    topk: int = Query(default=5),
    where: str = Query(default=""),
    modality: str = Query(default="text"),
    metric: str = Query(default=DEFAULT_METRIC),
    metric_val: float | None = Query(default=None),
):
    t0 = time.perf_counter()
    lcfg = LanceDBConfig()
    minio = MinioConfig()
    s3 = MinIOClient(minio) if SHOW_IMAGES else None

    # Prefer a shared connection/manager
    mgr = _get_manager()
    error_html = ""
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        tbl = None
        error_html = f"<p class='err'>Error opening table '{_escape(table)}': {_escape(str(e))}</p>"

    rowcount = _get_rowcount(tbl) if tbl else "n/a"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    results_html = "<p>Run a query to see results.</p>"

    m_min, m_max, m_step, m_def, m_label = _metric_slider_defaults(metric)
    m_val = metric_val if (metric_val is not None) else m_def

    html_out = PAGE.format(
        table=_escape(table), rowcount=rowcount, elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        metric_name=_escape(_norm_metric(metric)),
        nprobes=str(LANCEDB_NPROBES),
        refine=str(LANCEDB_REFINE),
        index_summary=_escape(_index_summary(tbl) if tbl else "n/a"),
        m_text="checked" if modality == "text" else "",
        m_img="checked" if modality == "image" else "",
        lidar_seg=(
            f"<label class='seg-item'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
            if ALLOW_LIDAR else ""
        ),
        topk=str(int(topk)),
        f_all="selected" if filter == "all" else "",
        f_text="selected" if filter == "text" else "",
        f_pdf="selected" if filter == "pdf" else "",
        f_image="selected" if filter == "image" else "",
        f_lidar="selected" if filter == "lidar" else "",
        m_l2="selected" if _norm_metric(metric) == "l2" else "",
        m_cos="selected" if _norm_metric(metric) == "cosine" else "",
        m_dot="selected" if metric.strip().lower() == "dot" else ("selected" if _norm_metric(metric) == "cosine" and metric.strip().lower() not in ("l2", "cosine") else ""),
        metric_label=_escape(m_label),
        m_min=str(m_min), m_max=str(m_max), m_step=str(m_step), m_val=str(m_val),
        where=_escape(where),
        q_text="",
        image_path="",
        lidar_path="",
        lidar_row="" if ALLOW_LIDAR else "display:none",
        max_bytes=MAX_UPLOAD_BYTES,
        results_html=results_html,
        error_html=error_html,
        lancedb_uri=_escape(lcfg.uri),
        endpoint=_escape(minio.endpoint or ""),
        region=_escape(minio.region or ""),
        buckets=_escape(minio.buckets or ""),
        access=_escape(mask(os.getenv("AWS_ACCESS_KEY_ID") or "")),
        secret=_escape(mask(os.getenv("AWS_SECRET_ACCESS_KEY") or "")),
    )
    return HTMLResponse(content=html_out)

@app.post("/search", response_class=HTMLResponse)
async def search(
    modality: str = Form(...),
    topk: int = Form(5),
    filter: str = Form("all"),
    where: str = Form(""),
    metric: str = Form(DEFAULT_METRIC),
    metric_val: str = Form(""),
    q_text: str = Form(""),
    image_path: str = Form(""),
    lidar_path: str = Form(""),
    image_file: UploadFile | None = File(default=None),
    lidar_file: UploadFile | None = File(default=None),
    table: str = Query(default=TABLE_DEFAULT),
):
    t0 = time.perf_counter()
    lcfg = LanceDBConfig()
    minio = MinioConfig()
    s3 = MinIOClient(minio) if SHOW_IMAGES else None
    mgr = _get_manager()

    error_html = ""
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        err = f"<p class='err'>Error opening table '{_escape(table)}': {_escape(str(e))}</p>"
        m_min, m_max, m_step, m_def, m_label = _metric_slider_defaults(metric)
        m_val = metric_val if (metric_val is not None) else m_def
        html_out = PAGE.format(
            table=_escape(table), rowcount="n/a", elapsed_ms=0, show_images="on" if SHOW_IMAGES else "off",
            metric_name=_escape(_norm_metric(metric)),
            nprobes=str(LANCEDB_NPROBES),
            refine=str(LANCEDB_REFINE),
            index_summary="n/a",
            m_text="checked" if modality == "text" else "",
            m_img="checked" if modality == "image" else "",
            lidar_seg=(
                f"<label class='seg-item'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
                if ALLOW_LIDAR else ""
            ),
            topk=str(int(topk)),
            f_all="selected" if filter == "all" else "",
            f_text="selected" if filter == "text" else "",
            f_pdf="selected" if filter == "pdf" else "",
            f_image="selected" if filter == "image" else "",
            f_lidar="selected" if filter == "lidar" else "",
            m_l2="selected" if _norm_metric(metric) == "l2" else "",
            m_cos="selected" if _norm_metric(metric) == "cosine" else "",
            m_dot="selected" if metric.strip().lower() == "dot" else ("selected" if _norm_metric(metric) == "cosine" and metric.strip().lower() not in ("l2", "cosine") else ""),
            metric_label=_escape(m_label),
            m_min=str(m_min), m_max=str(m_max), m_step=str(m_step), m_val=str(m_val),
            where=_escape(where),
            q_text=_escape(q_text),
            image_path=_escape(image_path),
            lidar_path=_escape(lidar_path),
            lidar_row="" if ALLOW_LIDAR else "display:none",
            max_bytes=MAX_UPLOAD_BYTES,
            results_html="<p>No results.</p>",
            error_html=err,
            lancedb_uri=_escape(lcfg.uri),
            endpoint=_escape(minio.endpoint or ""),
            region=_escape(minio.region or ""),
            buckets=_escape(minio.buckets or ""),
            access=_escape(mask(os.getenv("AWS_ACCESS_KEY_ID") or "")),
            secret=_escape(mask(os.getenv("AWS_SECRET_ACCESS_KEY") or "")),
        )
        return HTMLResponse(content=html_out)

    searcher = MultiModalSearcher(tbl)

    # Read uploads (bounded)
    img_blob, lidar_blob, lidar_ext = None, None, None
    try:
        if image_file and image_file.filename:
            blob = await image_file.read()
            _ensure_len(blob)
            img_blob = blob
        if ALLOW_LIDAR and lidar_file and lidar_file.filename:
            blob = await lidar_file.read()
            _ensure_len(blob)
            lidar_blob = blob
            _, lidar_ext = os.path.splitext(lidar_file.filename.lower())
    except ValueError as e:
        error_html = f"<p class='err'>{_escape(str(e))}</p>"

    # Create vector & search
    df = None
    if not error_html:
        try:
            vec = _vectorize_query(
                searcher, modality, q_text.strip() or None, image_path.strip() or None,
                img_blob, lidar_blob, lidar_ext
            )
        except Exception as e:
            error_html = f"<p class='err'>Vectorization error: {_escape(str(e))}</p>"
            vec = None

        if vec is not None:
            try:
                qobj = tbl.search(vec).metric(_norm_metric(metric))
                quick = _where_from_filter(filter)
                if quick:
                    qobj = qobj.where(quick)
                if where.strip():
                    qobj = qobj.where(where.strip())
                if LANCEDB_NPROBES and LANCEDB_NPROBES > 0:
                    qobj = qobj.nprobes(int(LANCEDB_NPROBES))
                if LANCEDB_REFINE and LANCEDB_REFINE > 0:
                    qobj = qobj.refine_factor(int(LANCEDB_REFINE))
                # prefetch more so thresholding still has enough rows
                prefetch = int(min(2000, max(int(topk), int(topk) * 5)))
                df = qobj.limit(prefetch).to_pandas()
                # apply metric threshold if provided
                try:
                    mv = float(metric_val) if metric_val is not None and str(metric_val).strip() != "" else None
                except Exception:
                    mv = None
                df = _apply_metric_threshold(df, metric, mv)
                df = df.head(int(topk))
            except Exception as e:
                error_html = f"<p class='err'>Search error: {_escape(str(e))}</p>"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    rowcount = _get_rowcount(tbl)
    results_html = _results_table(df, s3, metric) if df is not None else "<p>No results.</p>"

    # slider defaults for re-render
    m_min, m_max, m_step, m_def, m_label = _metric_slider_defaults(metric)
    try:
        mv_cur = float(metric_val) if metric_val is not None and str(metric_val).strip() != "" else m_def
    except Exception:
        mv_cur = m_def

    html_out = PAGE.format(
        table=_escape(table), rowcount=rowcount, elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        metric_name=_escape(_norm_metric(metric)),
        nprobes=str(LANCEDB_NPROBES),
        refine=str(LANCEDB_REFINE),
        index_summary=_escape(_index_summary(tbl)),
        m_text="checked" if modality == "text" else "",
        m_img="checked" if modality == "image" else "",
        lidar_seg=(
            f"<label class='seg-item'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
            if ALLOW_LIDAR else ""
        ),
        topk=str(int(topk)),
        f_all="selected" if filter == "all" else "",
        f_text="selected" if filter == "text" else "",
        f_pdf="selected" if filter == "pdf" else "",
        f_image="selected" if filter == "image" else "",
        f_lidar="selected" if filter == "lidar" else "",
        m_l2="selected" if _norm_metric(metric) == "l2" else "",
        m_cos="selected" if _norm_metric(metric) == "cosine" else "",
        m_dot="selected" if metric.strip().lower() == "dot" else ("selected" if _norm_metric(metric) == "cosine" and metric.strip().lower() not in ("l2", "cosine") else ""),
        metric_label=_escape(m_label),
        m_min=str(m_min), m_max=str(m_max), m_step=str(m_step), m_val=str(mv_cur),
        where=_escape(where),
        q_text=_escape(q_text),
        image_path=_escape(image_path),
        lidar_path=_escape(lidar_path),
        lidar_row="" if ALLOW_LIDAR else "display:none",
        max_bytes=MAX_UPLOAD_BYTES,
        results_html=results_html,
        error_html=error_html,
        lancedb_uri=_escape(lcfg.uri),
        endpoint=_escape(minio.endpoint or ""),
        region=_escape(minio.region or ""),
        buckets=_escape(minio.buckets or ""),
        access=_escape(mask(os.getenv("AWS_ACCESS_KEY_ID") or "")),
        secret=_escape(mask(os.getenv("AWS_SECRET_ACCESS_KEY") or "")),
    )
    return HTMLResponse(content=html_out)

# JSON API (unchanged)
@app.get("/api/search")
def api_search(
    q: str = Query(""),
    modality: str = Query("text"),
    topk: int = Query(5),
    where: str = Query(""),
    metric: str = Query(DEFAULT_METRIC),
    table: str = Query(default=TABLE_DEFAULT),
):
    lcfg = LanceDBConfig()
    mgr = LanceDBManager(lcfg)
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        return JSONResponse({"error": f"open table failed: {str(e)}"}, status_code=400)
    searcher = MultiModalSearcher(tbl)
    try:
        vec = _vectorize_query(searcher, modality, q, None, None, None, None)
        qobj = tbl.search(vec).metric(_norm_metric(metric))
        if where.strip():
            qobj = qobj.where(where.strip())
        if LANCEDB_NPROBES and LANCEDB_NPROBES > 0:
            qobj = qobj.nprobes(int(LANCEDB_NPROBES))
        if LANCEDB_REFINE and LANCEDB_REFINE > 0:
            qobj = qobj.refine_factor(int(LANCEDB_REFINE))
        df = qobj.limit(int(topk)).to_pandas()
    except Exception as e:
        return JSONResponse({"error": f"search failed: {str(e)}"}, status_code=400)
    return JSONResponse({"count": len(df), "rows": df.to_dict(orient="records")})

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "OK"
