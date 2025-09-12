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
SHOW_IMAGES   = os.getenv("WEBAPP_SHOW_IMAGES", "1").lower() not in ("0", "false")
MAX_UPLOAD_BYTES = int(os.getenv("WEBAPP_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8MiB
ALLOW_LIDAR   = True  # set False to hide LiDAR tab without code change

app = FastAPI(title="LanceDB Search UI", version="1.0")
# Minimal CORS (adjust `WEBAPP_CORS_ORIGINS` in env for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WEBAPP_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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

def _presign_if_image(path: str, s3: Optional[MinIOClient]) -> str | None:
    if not (SHOW_IMAGES and s3 and path and path.startswith("s3://")):
        return None
    parsed = _parse_s3_uri(path)
    if not parsed:
        return None
    b, k = parsed
    try:
        return s3.get_presigned_url(b, k, expires=3600)
    except Exception:
        return None

def _load_lidar_bytes(ext: str, blob: bytes) -> np.ndarray:
    # Minimal LiDAR loader matching ingest’s logic
    if ext == ".bin":
        arr  = np.frombuffer(blob, dtype=np.float32)
        cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
        return arr.reshape(-1, cols)
    elif ext == ".pcd":
        # basic fallback: treat as whitespace-separated floats; robust parser lives in ingest
        try:
            text = blob.decode("utf-8", errors="ignore").strip().split()
            arr  = np.asarray([float(x) for x in text], dtype=np.float32)
            cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
            return arr.reshape(-1, cols)[:, :3]
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)
    else:
        return np.zeros((0, 3), dtype=np.float32)

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

def _results_table(df, s3: Optional[MinIOClient]) -> str:
    if df is None or len(df) == 0:
        return "<p>No results.</p>"
    rows_html = []
    for _, r in df.iterrows():
        dist = r.get("_distance", "")
        if isinstance(dist, (int, float)):
            dist = f"{dist:.4f}"
        preview = ""
        if r.get("modality") == "image":
            url = _presign_if_image(r.get("path", ""), s3)
            if url:
                preview = f"<div class='imgbox'><img src='{_escape(url)}' alt='preview'></div>"
        rows_html.append(
            "<tr>"
            f"<td>{preview}</td>"
            f"<td><span class='badge mono'>{_escape(str(r.get('id','')))}</span></td>"
            f"<td>{_escape(str(r.get('modality','')))}</td>"
            f"<td>{_escape(str(dist))}</td>"
            f"<td>{_escape(_snippet(r.get('text') or ''))}</td>"
            f"<td class='mono'>{_escape(str(r.get('path','')))}</td>"
            "</tr>"
        )
    return (
        "<table class='table'>"
        "<tr><th>preview</th><th>id</th><th>modality</th>"
        "<th>distance</th><th>snippet</th><th>path</th></tr>"
        + "".join(rows_html)
        + "</table>"
    )

def _where_from_filter(f: str) -> Optional[str]:
    if f == "pdf":
        return "modality = 'pdf'"
    if f == "image":
        return "modality = 'image'"
    return None

def _get_rowcount(tbl) -> str:
    try:
        return str(len(tbl))
    except Exception:
        try:
            return str(tbl.count_rows())
        except Exception:
            return "?"

# --------------------------------------------------------------------------------------
# HTML template (responsive, no templates engine needed)
# --------------------------------------------------------------------------------------
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LanceDB Search</title>
<style>
:root{--b:#111827;--m:#6b7280;--p:#2563eb;--bd:#e5e7eb;--bg:#f9fafb}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:var(--bg);color:var(--b)}
.container{max-width:1100px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid var(--bd);border-radius:12px;padding:16px;margin-bottom:16px}
h1{font-size:1.5rem;margin:0 0 12px}
h2{font-size:1.1rem;margin:0 0 8px;color:var(--m)}
.row{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:800px){.row{grid-template-columns:2fr 1fr}}
input[type=text],select,button{width:100%;padding:.6rem;border:1px solid var(--bd);border-radius:8px;background:#fff}
button{background:var(--p);border-color:var(--p);color:#fff;cursor:pointer}
small{color:var(--m)}
.flex{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{display:inline-block;padding:.15rem .45rem;border:1px solid var(--bd);border-radius:6px;background:#fff;font-size:.85rem}
.table{width:100%;border-collapse:collapse}
th,td{border-bottom:1px solid var(--bd);padding:.6rem;text-align:left;vertical-align:top}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.err{color:#b91c1c}
.imgbox{width:200px;height:140px;display:flex;align-items:center;justify-content:center;border:1px solid var(--bd);border-radius:8px;overflow:hidden;background:#fafafa}
.imgbox img{max-width:100%;max-height:100%;object-fit:cover}
.kv{display:grid;grid-template-columns:180px 1fr;gap:6px;margin-top:8px}
.kv div{padding:2px 0;border-bottom:1px dashed #eee}
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
    </div>
  </div>

  <div class="card">
    <h2>Query</h2>
    <form method="post" action="/search" enctype="multipart/form-data" class="row">
      <div>
        <label>Modality</label>
        <select name="modality">
          <option value="text" {m_text}>Text</option>
          <option value="image" {m_img}>Image</option>
          {lidar_opt}
        </select>
      </div>
      <div>
        <label>Top-K</label>
        <select name="topk">
          <option{t3}>3</option><option{t5}>5</option><option{t10}>10</option><option{t20}>20</option>
        </select>
      </div>
      <div>
        <label>Filter (quick)</label>
        <select name="filter">
          <option value="all" {f_all}>All</option>
          <option value="pdf" {f_pdf}>PDFs only</option>
          <option value="image" {f_image}>Images only</option>
        </select>
      </div>
      <div>
        <label>Where (advanced)</label>
        <input type="text" name="where" placeholder="e.g. modality = 'pdf' AND path LIKE 's3://%reports%'" value="{where}">
      </div>

      <div style="grid-column:1/-1">
        <label>Text Query</label>
        <input type="text" name="q_text" placeholder="Search text..." value="{q_text}">
      </div>

      <div>
        <label>Image File (upload)</label>
        <input type="file" name="image_file" accept="image/*">
      </div>
      <div>
        <label>Image Path (server-local path)</label>
        <input type="text" name="image_path" placeholder="/path/to/image.jpg" value="{image_path}">
      </div>

      <div style="{lidar_row}">
        <label>LiDAR File (.pcd/.bin)</label>
        <input type="file" name="lidar_file" accept=".pcd,.bin,application/octet-stream">
      </div>
      <div style="{lidar_row}">
        <label>LiDAR Path (server-local)</label>
        <input type="text" name="lidar_path" placeholder="/path/to/scan.pcd" value="{lidar_path}">
      </div>

      <div style="grid-column:1/-1">
        <button type="submit">Search</button>
        <small>Tip: most users should stick with <b>Text</b> queries. Uploads capped at {max_bytes} bytes.</small>
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
    <div class="kv mono">
      <div>URI</div><div>{lancedb_uri}</div>
      <div>S3 Endpoint</div><div>{endpoint}</div>
      <div>Region</div><div>{region}</div>
      <div>Buckets</div><div>{buckets}</div>
      <div>Access</div><div>{access}</div>
      <div>Secret</div><div>{secret}</div>
    </div>
  </div>
</div>
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
):
    t0 = time.perf_counter()
    lcfg = LanceDBConfig()
    minio = MinioConfig()
    s3 = MinIOClient(minio) if SHOW_IMAGES else None

    mgr = LanceDBManager(lcfg)
    error_html = ""
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        tbl = None
        error_html = f"<p class='err'>Error opening table '{_escape(table)}': {_escape(str(e))}</p>"

    rowcount  = _get_rowcount(tbl) if tbl else "n/a"
    elapsed_ms= int((time.perf_counter() - t0) * 1000)
    results_html = "<p>Run a query to see results.</p>"

    html_out = PAGE.format(
        table=_escape(table), rowcount=rowcount, elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        m_text="selected" if modality == "text" else "",
        m_img="selected" if modality == "image" else "",
        lidar_opt=(f"<option value='lidar' {'selected' if modality=='lidar' else ''}>LiDAR</option>" if ALLOW_LIDAR else ""),
        t3=" selected" if int(topk) == 3 else "",
        t5=" selected" if int(topk) == 5 else "",
        t10=" selected" if int(topk) == 10 else "",
        t20=" selected" if int(topk) == 20 else "",
        f_all="selected" if filter == "all" else "",
        f_pdf="selected" if filter == "pdf" else "",
        f_image="selected" if filter == "image" else "",
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
    q_text: str = Form(""),
    image_path: str = Form(""),
    lidar_path: str = Form(""),
    image_file: UploadFile | None = File(default=None),
    lidar_file: UploadFile | None = File(default=None),
    table: str = Query(default=TABLE_DEFAULT),
):
    t0    = time.perf_counter()
    lcfg  = LanceDBConfig()
    minio = MinioConfig()
    s3    = MinIOClient(minio) if SHOW_IMAGES else None
    mgr   = LanceDBManager(lcfg)

    error_html = ""
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        err = f"<p class='err'>Error opening table '{_escape(table)}': {_escape(str(e))}</p>"
        html_out = PAGE.format(
            table=_escape(table), rowcount="n/a", elapsed_ms=0, show_images="on" if SHOW_IMAGES else "off",
            m_text="selected" if modality == "text" else "",
            m_img="selected" if modality == "image" else "",
            lidar_opt=(f"<option value='lidar' {'selected' if modality=='lidar' else ''}>LiDAR</option>" if ALLOW_LIDAR else ""),
            t3=" selected" if int(topk) == 3 else "",
            t5=" selected" if int(topk) == 5 else "",
            t10=" selected" if int(topk) == 10 else "",
            t20=" selected" if int(topk) == 20 else "",
            f_all="selected" if filter == "all" else "",
            f_pdf="selected" if filter == "pdf" else "",
            f_image="selected" if filter == "image" else "",
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

    # Create vector
    df  = None
    if not error_html:
        try:
            vec = _vectorize_query(searcher, modality, q_text.strip() or None,
                                   image_path.strip() or None, img_blob,
                                   lidar_blob, lidar_ext)
        except Exception as e:
            error_html = f"<p class='err'>Vectorization error: {_escape(str(e))}</p>"
            vec = None

        if vec is not None:
            try:
                qobj  = tbl.search(vec)
                quick = _where_from_filter(filter)
                if quick:
                    qobj = qobj.where(quick)
                if where.strip():
                    qobj = qobj.where(where.strip())
                df = qobj.limit(int(topk)).to_pandas()
            except Exception as e:
                error_html = f"<p class='err'>Search error: {_escape(str(e))}</p>"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    rowcount   = _get_rowcount(tbl)
    results_html = _results_table(df, s3) if df is not None else "<p>No results.</p>"

    html_out = PAGE.format(
        table=_escape(table), rowcount=rowcount, elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        m_text="selected" if modality == "text" else "",
        m_img="selected" if modality == "image" else "",
        lidar_opt=(f"<option value='lidar' {'selected' if modality=='lidar' else ''}>LiDAR</option>" if ALLOW_LIDAR else ""),
        t3=" selected" if int(topk) == 3 else "",
        t5=" selected" if int(topk) == 5 else "",
        t10=" selected" if int(topk) == 10 else "",
        t20=" selected" if int(topk) == 20 else "",
        f_all="selected" if filter == "all" else "",
        f_pdf="selected" if filter == "pdf" else "",
        f_image="selected" if filter == "image" else "",
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

# JSON API (optional, helpful for integration)
@app.get("/api/search")
def api_search(
    q: str        = Query(""),
    modality: str = Query("text"),
    topk: int     = Query(5),
    where: str    = Query(""),
    table: str    = Query(default=TABLE_DEFAULT),
):
    lcfg = LanceDBConfig()
    mgr  = LanceDBManager(lcfg)
    try:
        tbl = mgr.get_table(table)
    except Exception as e:
        return JSONResponse({"error": f"open table failed: {str(e)}"}, status_code=400)
    searcher = MultiModalSearcher(tbl)
    try:
        vec = _vectorize_query(searcher, modality, q, None, None, None, None)
        qobj = tbl.search(vec)
        if where.strip():
            qobj = qobj.where(where.strip())
        df = qobj.limit(int(topk)).to_pandas()
    except Exception as e:
        return JSONResponse({"error": f"search failed: {str(e)}"}, status_code=400)
    return JSONResponse({"count": len(df), "rows": df.to_dict(orient="records")})

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "OK"
