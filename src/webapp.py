from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.config import LanceDBConfig, MinioConfig
from src.lancedb_manager import LanceDBManager
from src.search import MultiModalSearcher
from src.storage import MinIOClient

TABLE_NAME = os.getenv("WEBAPP_TABLE", "test")
SHOW_IMAGES = os.getenv("WEBAPP_SHOW_IMAGES", "1") not in ("0", "false", "False")

app = FastAPI(title="LanceDB Search")

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>LanceDB Search</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:2rem;max-width:1100px}}
form{{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}}
input[type=text]{{flex:1;min-width:320px;padding:.6rem;border:1px solid #ddd;border-radius:.5rem}}
select,button,label{{padding:.55rem;border-radius:.5rem;border:1px solid #ddd;background:#fff}}
button{{border:1px solid #3b82f6;background:#3b82f6;color:#fff;cursor:pointer}}
small{{color:#666}}
table{{width:100%;border-collapse:collapse;margin-top:1rem}}
th,td{{border-bottom:1px solid #eee;padding:.6rem;text-align:left;vertical-align:top}}
.badge{{display:inline-block;padding:.1rem .5rem;border:1px solid #ddd;border-radius:.5rem;background:#fafafa}}
.info{{margin:.75rem 0;padding:.5rem .75rem;border:1px solid #e5e7eb;background:#f9fafb;border-radius:.5rem;color:#374151}}
.err{{color:#b91c1c}}
.imgbox{{width:180px;height:120px;display:flex;align-items:center;justify-content:center;border:1px solid #eee;border-radius:.5rem;overflow:hidden;background:#fafafa}}
.imgbox img{{max-width:100%;max-height:100%;object-fit:cover}}
</style>
</head>
<body>
<h2>Search</h2>
<div class="info">Using table <b>{table}</b> &middot; rows: <b>{rowcount}</b></div>
<form method="get" action="/">
  <input type="text" name="q" placeholder="Type what you're looking for (e.g. 'invoice', 'portrait', 'beethoven')" value="{q}">
  <select name="modality">
    <option value="text" {m_text}>Text query</option>
    <option value="image" {m_img}>Image file (local path)</option>
    <option value="lidar" {m_lidar}>LiDAR (demo)</option>
  </select>
  <select name="filter">
    <option value="all" {f_all}>All files</option>
    <option value="pdf" {f_pdf}>PDFs only</option>
    <option value="image" {f_image}>Images only</option>
  </select>
  <select name="topk">
    <option{t3}>3</option><option{t5}>5</option><option{t10}>10</option><option{t20}>20</option>
  </select>
  <button type="submit">Search</button>
</form>
<small>Tip: most users should leave “Text query” selected. “Image file” expects a local path this server can read.</small>
{results}
</body>
</html>"""

RESULTS_TPL = """<h3>Results</h3>
<table>
<tr><th>preview</th><th>id</th><th>modality</th><th>distance</th><th>snippet</th><th>path</th></tr>
{rows}
</table>"""

def _snippet(text: str, n: int = 140) -> str:
    if not text: return ""
    s = " ".join(text.split())
    return s[:n] + ("…" if len(s) > n else "")

def _where_from_filter(f: str) -> str|None:
    if f == "pdf": return "modality = 'pdf'"
    if f == "image": return "modality = 'image'"
    return None

def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    if not uri or not uri.startswith("s3://"): return None
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key

def _detect_vector_column(tbl) -> str | None:
    for cand in ("embedding", "vector", "features"):
        try:
            if cand in tbl.schema.names:
                return cand
        except Exception:
            pass
    try:
        import pyarrow as pa
        for field in tbl.schema:
            t = field.type
            if (pa.types.is_list(t) and pa.types.is_floating(t.value_type)) or \
               (pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type)):
                return field.name
    except Exception:
        pass
    return None

@app.get("/", response_class=HTMLResponse)
def index(q: str = "", modality: str = "text", filter: str = "all", topk: int = 5):
    lcfg = LanceDBConfig()
    mgr = LanceDBManager(lcfg)
    try:
        tbl = mgr.get_table(TABLE_NAME)
    except Exception as e:
        err = f"<p class='err'>Error opening table '{TABLE_NAME}': {e}</p>"
        html = HTML.format(table=TABLE_NAME, rowcount="n/a",
                           q=q.replace('"','&quot;'),
                           m_text="selected" if modality=="text" else "",
                           m_img="selected" if modality=="image" else "",
                           m_lidar="selected" if modality=="lidar" else "",
                           f_all="selected" if filter=="all" else "",
                           f_pdf="selected" if filter=="pdf" else "",
                           f_image="selected" if filter=="image" else "",
                           t3=" selected" if int(topk)==3 else "",
                           t5=" selected" if int(topk)==5 else "",
                           t10=" selected" if int(topk)==10 else "",
                           t20=" selected" if int(topk)==20 else "",
                           results=err)
        return html

    try:
        rowcount = len(tbl)
    except Exception:
        rowcount = "?"

    where = _where_from_filter(filter)

    html = HTML.format(table=TABLE_NAME, rowcount=rowcount,
                       q=q.replace('"','&quot;'),
                       m_text="selected" if modality=="text" else "",
                       m_img="selected" if modality=="image" else "",
                       m_lidar="selected" if modality=="lidar" else "",
                       f_all="selected" if filter=="all" else "",
                       f_pdf="selected" if filter=="pdf" else "",
                       f_image="selected" if filter=="image" else "",
                       t3=" selected" if int(topk)==3 else "",
                       t5=" selected" if int(topk)==5 else "",
                       t10=" selected" if int(topk)==10 else "",
                       t20=" selected" if int(topk)==20 else "",
                       results="")
    if not q:
        return html

    searcher = MultiModalSearcher(tbl)
    try:
        vec_col = _detect_vector_column(tbl)
        if modality == "text":
            vec = searcher.text_embedder.embed(q).tolist()
        elif modality == "image":
            with open(q, "rb") as f:
                vec = searcher.image_embedder.embed(f.read()).tolist()
        else:
            vec = searcher.lidar_embedder.embed([(0.0, 0.0, 0.0)]).tolist()

        qobj = tbl.search(vec) if vec_col is None else tbl.search(vec, vector_column_name=vec_col)
        if where:
            qobj = qobj.where(where)
        df = qobj.limit(int(topk)).to_pandas()
        df = df.drop_duplicates(subset=["id"], keep="first")

        if df is None or len(df) == 0:
            df = searcher.search(q, modality, top_k=int(topk), where=where)
    except Exception:
        df = searcher.search(q, modality, top_k=int(topk), where=where)

    minio = MinioConfig()
    s3 = MinIOClient(minio.endpoint, minio.access_key, minio.secret_key, minio.region) if SHOW_IMAGES else None

    rows = []
    for _, r in df.iterrows():
        dist = r.get("_distance", "")
        if isinstance(dist, (int, float)): dist = f"{dist:.4f}"

        preview_html = ""
        if SHOW_IMAGES and r.get("modality") == "image" and r.get("path"):
            parsed = _parse_s3_uri(r["path"])
            if parsed and s3 is not None:
                bucket, key = parsed
                try:
                    url = s3.get_presigned_url(bucket, key, expires=3600)
                    preview_html = f"<div class='imgbox'><img src='{url}' alt='preview'></div>"
                except Exception:
                    preview_html = ""

        rows.append(
            "<tr>"
            f"<td>{preview_html}</td>"
            f"<td><span class='badge'>{r.get('id','')}</span></td>"
            f"<td>{r.get('modality','')}</td>"
            f"<td>{dist}</td>"
            f"<td>{_snippet(r.get('text') or '')}</td>"
            f"<td>{r.get('path','')}</td>"
            "</tr>"
        )

    results_html = RESULTS_TPL.format(rows="".join(rows)) if rows else "<p>No results.</p>"
    return HTML.format(table=TABLE_NAME, rowcount=rowcount,
                       q=q.replace('"','&quot;'),
                       m_text="selected" if modality=="text" else "",
                       m_img="selected" if modality=="image" else "",
                       m_lidar="selected" if modality=="lidar" else "",
                       f_all="selected" if filter=="all" else "",
                       f_pdf="selected" if filter=="pdf" else "",
                       f_image="selected" if filter=="image" else "",
                       t3=" selected" if int(topk)==3 else "",
                       t5=" selected" if int(topk)==5 else "",
                       t10=" selected" if int(topk)==10 else "",
                       t20=" selected" if int(topk)==20 else "",
                       results=results_html)

@app.get("/health", response_class=HTMLResponse)
def health():
    return "OK"