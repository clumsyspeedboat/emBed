# src/webapp.py

from __future__ import annotations

import io
import os
import time
import html
import urllib.parse
from functools import lru_cache
from typing import Optional, Iterable

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
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


@lru_cache(maxsize=4)
def _blank_lidar_png(size: int = 256) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (size, size), color=240).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _lidar_preview_png(points: np.ndarray, *, size: int = 256) -> bytes:
    if points is None or len(points) == 0:
        return _blank_lidar_png(size)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return _blank_lidar_png(size)

    xy = pts[:, :2]
    xy = xy[~np.isnan(xy).any(axis=1)]
    if xy.size == 0:
        return _blank_lidar_png(size)

    try:
        low = np.percentile(xy, 1, axis=0)
        high = np.percentile(xy, 99, axis=0)
    except Exception:
        return _blank_lidar_png(size)

    span = np.maximum(high - low, 1e-3)
    norm = (xy - low) / span
    if norm.size == 0:
        return _blank_lidar_png(size)

    mask = ((norm >= 0.0) & (norm <= 1.0)).all(axis=1)
    clipped = norm[mask] if mask.any() else np.clip(norm, 0.0, 1.0)
    if clipped.size == 0:
        return _blank_lidar_png(size)

    xs = np.clip((clipped[:, 0] * (size - 1)).astype(np.int32), 0, size - 1)
    ys = np.clip(((1.0 - clipped[:, 1]) * (size - 1)).astype(np.int32), 0, size - 1)

    grid = np.zeros((size, size), dtype=np.float32)
    np.add.at(grid, (ys, xs), 1.0)
    grid = np.log1p(grid)
    gmax = float(grid.max())
    if gmax <= 0.0:
        return _blank_lidar_png(size)
    grid = grid / gmax
    array = np.clip(255.0 - (grid * 255.0), 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _fetch_lidar_blob(path: str) -> tuple[bytes, str]:
    if not path:
        raise FileNotFoundError("empty path")
    ext = os.path.splitext(path)[1].lower()
    if path.startswith("s3://"):
        parsed = _parse_s3_uri(path)
        if not parsed:
            raise FileNotFoundError("invalid s3 uri")
        bucket, key = parsed
        client = MinIOClient(MinioConfig())
        data = client.get_object_bytes(bucket, key)
        key_ext = os.path.splitext(key)[1].lower()
        return data, (key_ext or ext)
    with open(path, "rb") as f:
        return f.read(), ext


@lru_cache(maxsize=64)
def _cached_lidar_preview(path: str) -> bytes:
    data, ext = _fetch_lidar_blob(path)
    pts = _load_lidar_bytes(ext, data)
    return _lidar_preview_png(pts)

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
        url = _presign_with_client(path, s3)

        # Previews
        if SHOW_IMAGES and path:
            if url and mod == "image":
                preview_html = f"<div class='imgbox'><img loading='lazy' src='{url}' alt='preview'></div>"
            elif url and mod == "pdf":
                preview_html = f"<div class='imgbox' title='PDF'><a href='{url}' target='_blank' rel='noreferrer'>🧾 open</a></div>"
            elif mod == "lidar" and ALLOW_LIDAR:
                q_path = urllib.parse.quote_plus(path)
                preview_html = (
                    "<div class='imgbox'><img loading='lazy' alt='LiDAR preview' "
                    f"src='/preview/lidar?path={q_path}'></div>"
                )

        # Linkify path if we got a URL; otherwise plain text
        url_for_path = url
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
:root{{--bg:#0f172a;--surface:#f8fafc;--card:#ffffff;--accent:#2563eb;--accent-dark:#1d4ed8;--muted:#64748b;--border:#e2e8f0;--text:#0f172a;--shadow:0 24px 48px -28px rgba(15,23,42,.55)}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:radial-gradient(circle at top,#e0ecff 0%,#f8fafc 55%,#eef2ff 100%);color:var(--text)}}
.shell{{max-width:1200px;margin:0 auto;padding:32px 24px 80px}}
.masthead{{background:rgba(255,255,255,.92);backdrop-filter:blur(14px);border-radius:18px;padding:32px;border:1px solid rgba(226,232,240,.7);box-shadow:var(--shadow);display:flex;flex-direction:column;gap:24px}}
.masthead h1{{margin:0;font-size:2.05rem;letter-spacing:-.02em}}
.masthead p{{margin:0;color:var(--muted);font-size:1rem}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.stat{{padding:14px 16px;border-radius:14px;background:linear-gradient(135deg,rgba(37,99,235,.08),rgba(255,255,255,.9));border:1px solid rgba(37,99,235,.14);display:flex;flex-direction:column}}
.stat-label{{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}}
.stat-value{{margin-top:4px;font-size:1.05rem;font-weight:600;color:var(--text)}}
.panel{{margin-top:28px;background:var(--card);border-radius:18px;padding:28px;border:1px solid var(--border);box-shadow:var(--shadow);position:relative}}
.panel-head{{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:24px}}
.panel-head h2{{margin:0;font-size:1.35rem;letter-spacing:-.01em;color:var(--text)}}
.panel-head small{{color:var(--muted)}}
.form-grid{{display:grid;gap:28px}}
.form-main{{display:grid;gap:20px}}
.form-side{{display:grid;gap:20px}}
@media(min-width:960px){{.form-grid{{grid-template-columns:2fr 1fr;align-items:start}}.form-main{{grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}.form-main .full-row{{grid-column:1/-1}}}}
.segmented{{display:flex;gap:8px;padding:6px;background:var(--surface);border-radius:12px;border:1px solid var(--border)}}
.segmented label{{position:relative;flex:1;font-weight:600}}
.segmented input{{position:absolute;opacity:0;pointer-events:none}}
.segmented span{{display:block;padding:11px 14px;border-radius:10px;text-align:center;color:var(--muted);transition:all .18s ease}}
.segmented input:checked + span{{background:var(--accent);color:#fff;box-shadow:0 16px 24px -18px rgba(37,99,235,.75)}}
.field{{display:flex;flex-direction:column;gap:8px}}
.field label{{font-weight:600;color:var(--text);font-size:.95rem}}
.field input[type=text],.field input[type=number],.field input[type=file],.field select,.field textarea{{width:100%;padding:11px 12px;border-radius:11px;border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:.95rem;transition:border-color .15s ease,box-shadow .15s ease}}
.field input[type=text]:focus,.field input[type=number]:focus,.field input[type=file]:focus,.field select:focus,.field textarea:focus{{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.15)}}
.field input[type=file]{{padding:14px 12px;background:linear-gradient(95deg,rgba(243,246,255,.85),#fff)}}
.field .hint{{color:var(--muted);font-size:.82rem}}
.media-grid{{display:grid;gap:18px}}
@media(min-width:780px){{.media-grid.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.slider-row{{display:flex;gap:14px;align-items:center}}
.slider-row input[type=range]{{flex:1}}
.actions{{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin-top:12px}}
button[type=submit]{{display:inline-flex;align-items:center;gap:10px;padding:12px 24px;border:none;border-radius:12px;font-weight:600;font-size:1rem;background:linear-gradient(135deg,var(--accent),var(--accent-dark));color:#fff;cursor:pointer;box-shadow:0 20px 36px -20px rgba(37,99,235,.95);transition:transform .15s ease,box-shadow .15s ease}}
button[type=submit]:hover{{transform:translateY(-1px);box-shadow:0 28px 40px -24px rgba(37,99,235,1)}}
button[type=submit]:active{{transform:translateY(0)}}
.note{{color:var(--muted);font-size:.9rem;max-width:420px}}
.alert,.err{{margin-top:18px;padding:16px 18px;border-radius:14px;background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.28);color:#991b1b;font-weight:500}}
.results{{margin-top:28px;border-radius:20px;background:var(--card);border:1px solid var(--border);box-shadow:var(--shadow);overflow:hidden}}
.results-header{{padding:22px 28px;display:flex;justify-content:space-between;align-items:center;gap:16px;border-bottom:1px solid var(--border)}}
.results-header h2{{margin:0;font-size:1.3rem}}
.results-header span{{color:var(--muted);font-size:.82rem}}
.table-wrap{{overflow-x:auto}}
.table{{width:100%;border-collapse:collapse;min-width:820px}}
.table thead th{{position:sticky;top:0;background:var(--surface);padding:14px 16px;color:var(--muted);font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid var(--border)}}
.table td{{padding:14px 16px;border-bottom:1px solid var(--border);vertical-align:middle;font-size:.93rem;color:var(--text);line-height:1.4;word-break:break-word;white-space:normal}}
.table tr:hover{{background:rgba(37,99,235,.03)}}
.table th.preview,.table td.preview{{width:210px;vertical-align:middle}}
.table th.mod,.table td.mod{{width:100px;white-space:nowrap}}
.table th.dist,.table td.dist{{width:110px;white-space:nowrap}}
.table td a{{color:var(--accent);text-decoration:underline;word-break:break-all}}
.truncate{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block}}
.break{{word-break:break-word}}
.imgbox{{width:200px;height:132px;border-radius:12px;border:1px solid var(--border);background:var(--surface);display:flex;align-items:center;justify-content:center;overflow:hidden;box-shadow:inset 0 1px 0 rgba(255,255,255,.7)}}
.imgbox img{{max-width:100%;max-height:100%;object-fit:contain;display:block}}
.kv{{display:grid;grid-template-columns:220px 1fr;gap:14px;margin-top:18px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9rem;background:var(--surface);border-radius:14px;padding:20px;border:1px dashed var(--border);color:var(--text)}}
.kv div{{padding:2px 0}}
details summary{{cursor:pointer;color:var(--muted);font-weight:600}}
.table-wrap p{{padding:26px;color:var(--muted);margin:0}}
footer{{margin-top:42px;text-align:center;color:var(--muted);font-size:.85rem}}
@media(max-width:760px){{.masthead{{padding:24px}}.panel{{padding:24px}}.results-header{{padding:22px}}.table{{min-width:100%}}.kv{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main class="shell">
  <header class="masthead">
    <div class="masthead-body">
      <h1>LanceDB Search</h1>
      <p>Multimodal retrieval across documents, imagery, and LiDAR frames.</p>
    </div>
    <div class="stat-grid">
      <div class="stat"><span class="stat-label">Table</span><span class="stat-value">{table}</span></div>
      <div class="stat"><span class="stat-label">Rows</span><span class="stat-value">{rowcount}</span></div>
      <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">{elapsed_ms} ms</span></div>
      <div class="stat"><span class="stat-label">Images</span><span class="stat-value">{show_images}</span></div>
      <div class="stat"><span class="stat-label">Metric</span><span class="stat-value">{metric_name}</span></div>
      <div class="stat"><span class="stat-label">Probes</span><span class="stat-value">{nprobes}</span></div>
      <div class="stat"><span class="stat-label">Refine</span><span class="stat-value">{refine}</span></div>
      <div class="stat"><span class="stat-label">Index</span><span class="stat-value">{index_summary}</span></div>
    </div>
  </header>

  <section class="panel">
    <div class="panel-head">
      <h2>Build Query</h2>
      <small>Uploads capped at {max_bytes} bytes</small>
    </div>
    <form method="post" action="/search" enctype="multipart/form-data" class="form-grid" data-search-form>
      <div class="form-main">
        <div class="field full-row">
          <label>Modality</label>
          <div class="segmented" role="tablist" aria-label="Modality">
            <label class="seg-option">
              <input type="radio" name="modality" value="text" {m_text}><span>Text</span>
            </label>
            <label class="seg-option">
              <input type="radio" name="modality" value="image" {m_img}><span>Image</span>
            </label>
            {lidar_seg}
          </div>
          <span class="hint">Choose the embedding pipeline you want to query.</span>
        </div>

        <div class="field mod-section" data-mod="text">
          <label>Text Query</label>
          <input type="text" name="q_text" placeholder="e.g. 'warehouse inventory report'" value="{q_text}">
          <span class="hint">A sentence or short paragraph works best.</span>
        </div>

        <div class="media-grid two mod-section" data-mod="image">
          <div class="field">
            <label>Image File (upload)</label>
            <input type="file" name="image_file" accept="image/*">
            <span class="hint">Select a local file to embed on the fly.</span>
          </div>
          <div class="field">
            <label>Image Path (server-local)</label>
            <input type="text" name="image_path" placeholder="/path/to/image.jpg" value="{image_path}">
            <span class="hint">Optional: reference a file reachable from the server.</span>
          </div>
        </div>

        <div class="media-grid two mod-section" data-mod="lidar" style="{lidar_row}">
          <div class="field">
            <label>LiDAR File (.pcd/.bin)</label>
            <input type="file" name="lidar_file" accept=".pcd,.bin,application/octet-stream">
            <span class="hint">Upload a single frame point cloud for embedding.</span>
          </div>
          <div class="field">
            <label>LiDAR Path (server-local)</label>
            <input type="text" name="lidar_path" placeholder="/path/to/scan.pcd" value="{lidar_path}">
            <span class="hint">Use when the file already resides on the server.</span>
          </div>
        </div>

        <details class="full-row">
          <summary>Advanced filter (WHERE)</summary>
          <div class="field" style="margin-top:14px">
            <label class="hint" style="font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">SQL-like predicate</label>
            <input type="text" name="where" placeholder="e.g. modality = 'pdf' AND path LIKE 's3://%reports%'" value="{where}">
            <span class="hint">Translated to LanceDB <code>where</code> before scanning.</span>
          </div>
        </details>
      </div>

      <div class="form-side">
        <div class="field">
          <label>Top-K</label>
          <input type="number" name="topk" min="1" max="2000" step="1" value="{topk}">
          <span class="hint">Controls how many rows are rendered in the results.</span>
        </div>
        <div class="field">
          <label>Quick Filter</label>
          <select name="filter">
            <option value="all" {f_all}>All modalities</option>
            <option value="text" {f_text}>Text only</option>
            <option value="pdf" {f_pdf}>PDFs only</option>
            <option value="image" {f_image}>Images only</option>
            <option value="lidar" {f_lidar}>LiDAR only</option>
          </select>
          <span class="hint">Client-side shortcut before additional predicates.</span>
        </div>
        <div class="field">
          <label>Metric</label>
          <select name="metric">
            <option value="l2" {m_l2}>Euclidean (L2)</option>
            <option value="cosine" {m_cos}>Cosine</option>
            <option value="dot" {m_dot}>Dot Product</option>
          </select>
          <span class="hint">Cosine/Dot recommended; embeddings are L2-normalized.</span>
        </div>
        <div class="field">
          <label id="metric_val_label">{metric_label}</label>
          <div class="slider-row">
            <input type="range" name="metric_val" min="{m_min}" max="{m_max}" step="{m_step}" value="{m_val}" oninput="document.getElementById('metric_val_num').value=this.value">
            <input id="metric_val_num" type="number" name="metric_val_num" min="{m_min}" max="{m_max}" step="{m_step}" value="{m_val}" oninput="document.querySelector('input[name=metric_val]').value=this.value">
          </div>
          <span class="hint">Applies client-side after the search results are fetched.</span>
        </div>
        <div class="actions">
          <button type="submit">Run Search</button>
          <span class="note">Tip: start with text, then add image or LiDAR queries for visual validation.</span>
        </div>
      </div>
    </form>
    {error_html}
  </section>

  <section class="results">
    <div class="results-header">
      <h2>Results</h2>
      <span>Metric threshold and filters apply instantly—experiment freely.</span>
    </div>
    <div class="table-wrap">
      {results_html}
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Config Snapshot</h2>
      <small>Loaded from environment and shared with the frontend for transparency.</small>
    </div>
    <details open>
      <summary>Show configuration</summary>
      <div class="kv">
        <div>URI</div><div>{lancedb_uri}</div>
        <div>S3 Endpoint</div><div>{endpoint}</div>
        <div>Region</div><div>{region}</div>
        <div>Buckets</div><div>{buckets}</div>
        <div>Access</div><div>{access}</div>
        <div>Secret</div><div>{secret}</div>
      </div>
    </details>
  </section>

  <footer>Powered by LanceDB · FastAPI · MinIO</footer>
</main>

<script>
(function(){{
  const form = document.querySelector('form[data-search-form]');
  if (!form) return;
  const metricSelect = form.querySelector('select[name="metric"]');
  const slider = form.querySelector('input[name="metric_val"]');
  const num = form.querySelector('#metric_val_num');
  function showSections(mod){{
    form.querySelectorAll('.mod-section').forEach(el => {{
      const target = el.getAttribute('data-mod');
      el.style.display = (target === mod) ? '' : 'none';
    }});
  }}
  function currentMod(){{
    const el = form.querySelector('input[name="modality"]:checked');
    return el ? el.value : 'text';
  }}
  function applyMetricMeta(metric){{
    if (!slider || !num) return;
    if (metric === 'l2'){{
      slider.min = '0.0'; slider.max = '2.0'; slider.step = '0.01';
      num.min = '0.0'; num.max = '2.0'; num.step = '0.01';
    }} else {{
      slider.min = '-1.0'; slider.max = '1.0'; slider.step = '0.01';
      num.min = '-1.0'; num.max = '1.0'; num.step = '0.01';
    }}
  }}
  form.addEventListener('change', ev => {{
    if (ev.target && ev.target.name === 'modality'){{
      showSections(ev.target.value);
    }}
    if (ev.target && ev.target.name === 'metric'){{
      applyMetricMeta(ev.target.value);
    }}
  }});
  showSections(currentMod());
  if (metricSelect) applyMetricMeta(metricSelect.value || 'l2');
}})();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------


@app.get("/preview/lidar")
def lidar_preview(path: str = Query(..., description="LiDAR object path")):
    if not (ALLOW_LIDAR and SHOW_IMAGES):
        raise HTTPException(status_code=404)
    target = path.strip()
    if not target:
        raise HTTPException(status_code=404, detail="Path missing")
    try:
        png_bytes = _cached_lidar_preview(target)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LiDAR object not found")
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to render LiDAR preview")
    if not png_bytes:
        raise HTTPException(status_code=404, detail="Preview unavailable")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


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
            f"<label class='seg-option'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
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
                f"<label class='seg-option'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
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
            f"<label class='seg-option'><input type='radio' name='modality' value='lidar' {'checked' if modality=='lidar' else ''}><span>LiDAR</span></label>"
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
