from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import LanceDBConfig
from .lancedb_manager import LanceDBManager
from .search import MultiModalSearcher

TABLE_NAME = os.getenv("WEBAPP_TABLE", "demo")  # override if needed

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
<tr><th>id</th><th>modality</th><th>distance</th><th>snippet</th><th>path</th></tr>
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

def _detect_vector_column(tbl) -> str | None:
    # Prefer common names; fall back to scanning schema if available.
    for cand in ("embedding", "vector", "features"):
        try:
            if cand in tbl.schema.names:
                return cand
        except Exception:
            pass
    try:
        import pyarrow as pa  # already in requirements
        for field in tbl.schema:
            t = field.type
            # list<float> or fixed_size_list<float>
            if (pa.types.is_list(t) and pa.types.is_floating(t.value_type)) or \
               (pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type)):
                return field.name
    except Exception:
        pass
    return None

@app.get("/", response_class=HTMLResponse)
def index(q: str = "", modality: str = "text", filter: str = "all", topk: int = 5):
    # Connect & basic table info
    lcfg = LanceDBConfig()
    mgr = LanceDBManager(lcfg)
    try:
        tbl = mgr.get_table(TABLE_NAME)
    except Exception as e:
        err = f"<p class='err'>Error opening table '{TABLE_NAME}': {e}</p>"
        html = HTML.format(
            table=TABLE_NAME, rowcount="n/a",
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
            results=err
        )
        return html

    # Count rows (safe on your small set)
    try:
        rowcount = len(tbl.to_pandas())
    except Exception:
        rowcount = "?"

    where = _where_from_filter(filter)

    # Render shell first (no results yet)
    html = HTML.format(
        table=TABLE_NAME, rowcount=rowcount,
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
        results=""
    )
    if not q:
        return html

    # Run search with vector column detection & fallback
    searcher = MultiModalSearcher(tbl)
    try:
        vec_col = _detect_vector_column(tbl)
        if vec_col:
            try:
                qobj = tbl.search(searcher.text_embedder.embed(q).tolist()) if modality=="text" else tbl.search(searcher.image_embedder.embed(open(q,"rb").read()).tolist())
                # try parameterized variants for compatibility
                try:
                    qobj = tbl.search(searcher.text_embedder.embed(q).tolist(), vector_column_name=vec_col) if modality=="text" else \
                           tbl.search(searcher.image_embedder.embed(open(q,"rb").read()).tolist(), vector_column_name=vec_col)
                except TypeError:
                    qobj = tbl.search(searcher.text_embedder.embed(q).tolist(), vector_column=vec_col) if modality=="text" else \
                           tbl.search(searcher.image_embedder.embed(open(q,"rb").read()).tolist(), vector_column=vec_col)
                if where:
                    qobj = qobj.where(where)
                df = qobj.limit(int(topk)).to_pandas()
            except Exception:
                # Fall back to MultiModalSearcher (uses .search(vec) without explicit column)
                df = searcher.search(q, modality, topk=int(topk), where=where)
        else:
            df = searcher.search(q, modality, topk=int(topk), where=where)
    except Exception as e:
        return html.replace("{results}", f"<p class='err'>Search error: {e}</p>")

    # Render results
    rows = []
    for _, r in df.iterrows():
        dist = r.get("_distance", "")
        if isinstance(dist, (int, float)):
            dist = f"{dist:.4f}"
        rows.append(
            "<tr>"
            f"<td><span class='badge'>{(r.get('id',''))}</span></td>"
            f"<td>{(r.get('modality',''))}</td>"
            f"<td>{dist}</td>"
            f"<td>{_snippet(r.get('text') or '')}</td>"
            f"<td>{(r.get('path',''))}</td>"
            "</tr>"
        )
    results_html = RESULTS_TPL.format(rows="".join(rows)) if rows else "<p>No results.</p>"
    return HTML.format(
        table=TABLE_NAME, rowcount=rowcount,
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
        results=results_html
    )

@app.get("/health", response_class=HTMLResponse)
def health():
    return "OK"
