"""Purpose: implement the FastAPI application serving the search UI/API.
Why extend: add routes, auth, or caching behaviour.
How extend: register new FastAPI routers or dependency overrides within this module, ensuring tests cover the new endpoints.
"""
# src/web/app.py

from __future__ import annotations

import io
import os
import time
import html
import urllib.parse
from functools import lru_cache
from typing import Optional, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import LanceDBConfig, MinioConfig, WebAppConfig
from src.vectordb import LanceDBManager
from src.search import (
    SearchInputs,
    SearchService,
    load_lidar_points,
    normalize_modalities,
)
from src.storage import MinIOClient
from src.cli.utils import mask

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
web_cfg = WebAppConfig()
TABLE_DEFAULT = web_cfg.table
SHOW_IMAGES = web_cfg.show_images
MAX_UPLOAD_BYTES = web_cfg.max_upload_bytes
PRESIGN_TTL = web_cfg.presign_ttl
ALLOW_LIDAR = web_cfg.allow_lidar
DEFAULT_METRIC = web_cfg.default_metric
LANCEDB_NPROBES = web_cfg.lance_nprobes
LANCEDB_REFINE = web_cfg.lance_refine_factor

app = FastAPI(title="LanceDB Search UI", version="1.2")
# Minimal CORS (adjust `WEBAPP_CORS_ORIGINS` in env for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=web_cfg.cors_origins,
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
    try:
        return load_lidar_points(ext, blob)
    except Exception:
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


@lru_cache(maxsize=1)
def _available_tables() -> list[str]:
    try:
        mgr = _get_manager()
        tables = mgr.list_tables()
        return sorted(tables)
    except Exception:
        return []


def _normalize_tables(
    requested: Sequence[str] | None,
    available: Sequence[str],
    fallback: Optional[str],
) -> list[str]:
    avail_set = {name for name in available}
    result: list[str] = []
    if requested:
        for name in requested:
            if name in avail_set and name not in result:
                result.append(name)
    if not result and fallback and fallback in avail_set:
        result = [fallback]
    if not result and available:
        result = [available[0]]
    return result


def _render_table_options(available: Sequence[str], selected: Sequence[str]) -> str:
    if not available:
        return "<p class='table-empty'>No tables discovered.</p>"
    selected_set = set(selected)
    parts: list[str] = []
    for name in available:
        checked = "checked" if name in selected_set else ""
        parts.append(
            "<label class='table-option'>"
            f"<input type='checkbox' name='tables' value='{_escape(name)}' {checked} form='search-form'>"
            f"<span>{_escape(name)}</span>"
            "</label>"
        )
    return "".join(parts)


def _render_table_overview(meta: Sequence[tuple[str, str, str]], selected: Sequence[str]) -> str:
    if not meta:
        return "<p class='table-empty'>No tables discovered.</p>"
    selected_set = set(selected)
    rows = [
        "<table class='meta-table'>",
        "<thead><tr><th>table</th><th>rows</th><th>index</th></tr></thead><tbody>",
    ]
    for name, rows_val, index_val in meta:
        active = " active" if name in selected_set else ""
        pill = (
            "<span class='table-pill'>active</span>"
            if name in selected_set
            else ""
        )
        rows.append(
            f"<tr class='meta-row{active}'>"
            f"<td class='meta-name'>{_escape(name)} {pill}</td>"
            f"<td class='meta-rows'>{_escape(rows_val)}</td>"
            f"<td class='meta-index'>{_escape(index_val)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _describe_tables(
    available: Sequence[str],
    selected: Sequence[str],
) -> tuple[dict[str, object], str, str, list[tuple[str, str, str]]]:
    mgr = _get_manager()
    handles: dict[str, object] = {}
    row_parts: list[str] = []
    index_parts: list[str] = []
    meta: list[tuple[str, str, str]] = []

    selected_set = set(selected)
    for name in available:
        rows_val = "?"
        index_val = "?"
        try:
            tbl = mgr.get_table(name)
            rows_val = _get_rowcount(tbl)
            index_val = _index_summary(tbl)
            if name in selected_set:
                handles[name] = tbl
                row_parts.append(f"{name}: {rows_val}")
                index_parts.append(f"{name}: {index_val}")
        except Exception:
            if name in selected_set:
                row_parts.append(f"{name}: ?")
                index_parts.append(f"{name}: ?")
        meta.append((name, rows_val, index_val))

    row_summary = " • ".join(row_parts) if row_parts else "n/a"
    index_summary = " • ".join(index_parts) if index_parts else "n/a"
    return handles, row_summary, index_summary, meta


def _results_table(
    df,
    s3: Optional[MinIOClient],
    metric: str,
    default_table: Optional[str] = None,
) -> str:
    if df is None or len(df) == 0:
        return "<p>No results.</p>"
    m = _norm_metric(metric)
    dist_col = "distance" if m == "l2" else "similarity"
    rows_html = []
    for _, r in df.iterrows():
        rid = str(r.get("id", ""))
        table_name = str(r.get("table", default_table or ""))
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

        snippet_text = _escape(_snippet(text))
        text_title = _escape(text)

        rows_html.append(
            "<tr>"
            f"<td class='preview'>{preview_html}</td>"
            f"<td class='table'>{_escape(table_name)}</td>"
            f"<td class='id-cell' title='{_escape(rid)}'><span class='truncate'>{_escape(rid)}</span></td>"
            f"<td class='mod'>{_escape(mod)}</td>"
            f"<td class='dist'>{dist}</td>"
            f"<td class='snippet' title='{text_title}'><span class='snippet-text'>{snippet_text}</span></td>"
            f"<td class='break'>{path_cell}</td>"
            "</tr>"
        )

    results_html = (
        "<table class='table'>"
        "<thead><tr>"
        "<th class='preview'>preview</th><th class='table'>table</th><th>id</th><th class='mod'>modality</th>"
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

# --------------------------------------------------------------------------------------
# HTML template (responsive, no template engine needed)
# --------------------------------------------------------------------------------------
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LanceDB Search</title>
<style>
:root{{--bg:#f5f7fb;--bg-accent:#eef2ff;--card:#ffffff;--card-shadow:0 25px 60px -30px rgba(15,23,42,.25);--border:rgba(15,23,42,.08);--border-strong:rgba(15,23,42,.14);--text:#0f172a;--text-soft:#475569;--muted:#64748b;--accent:#0ea5e9;--accent-strong:#0284c7;--accent-soft:rgba(14,165,233,.12);--radius-lg:26px;--radius-md:18px;--radius-sm:12px;--code-bg:#f1f5f9}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:radial-gradient(circle at 20% -10%,rgba(14,165,233,.16),transparent 60%),radial-gradient(circle at 85% -15%,rgba(59,130,246,.18),transparent 55%),linear-gradient(180deg,var(--bg) 0%,#fcfdff 60%,#f8fafc 100%);color:var(--text);min-height:100vh}}
.shell{{max-width:1240px;margin:0 auto;padding:48px 28px 96px}}
.masthead{{background:linear-gradient(135deg,#fff 10%,#f3f7ff 90%);border-radius:var(--radius-lg);padding:44px;border:1px solid rgba(14,165,233,.12);box-shadow:0 45px 65px -50px rgba(14,116,144,.35);position:relative;overflow:hidden}}
.masthead h1{{margin:0;font-size:2.4rem;letter-spacing:-.02em}}
.masthead p{{margin:14px 0 0;color:var(--muted);font-size:1.05rem;max-width:620px}}
.masthead-controls{{display:flex;flex-direction:column;gap:12px;margin-top:26px}}
.table-select-label{{font-weight:600;color:var(--text);font-size:.95rem}}
.masthead-controls .hint{{color:var(--muted);font-size:.82rem}}
.stat-grid{{display:grid;gap:18px}}
.stat-grid-meta{{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}}
.stat-grid-run{{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}}
.stat{{display:flex;gap:18px;align-items:center;background:#ffffff;border-radius:var(--radius-md);padding:18px 20px;border:1px solid var(--border);box-shadow:0 16px 32px -28px rgba(15,23,42,.25)}}
.stat-icon{{display:inline-flex;width:42px;height:42px;border-radius:50%;background:var(--accent-soft);align-items:center;justify-content:center;font-size:1.2rem;color:var(--accent)}}
.stat-meta{{display:flex;flex-direction:column;gap:4px}}
.stat-label{{font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}}
.stat-value{{font-weight:600;font-size:1.12rem;color:var(--text)}}
.panel{{margin-top:40px;background:var(--card);border-radius:var(--radius-lg);padding:36px;border:1px solid var(--border);box-shadow:var(--card-shadow)}}
.panel-head{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:18px;margin-bottom:28px}}
.panel-head h2{{margin:0;font-size:1.65rem;letter-spacing:-.01em}}
.panel-head small{{color:var(--accent);font-size:.85rem;background:var(--accent-soft);padding:6px 14px;border-radius:999px;border:1px solid rgba(14,165,233,.18)}}
.form-grid{{display:grid;gap:32px}}
.form-main{{display:grid;gap:24px}}
.form-side{{display:grid;gap:24px}}
@media(min-width:960px){{.form-grid{{grid-template-columns:2fr 1fr;align-items:start}}.form-main{{grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}}.form-main .full-row{{grid-column:1/-1}}}}
.table-options{{display:flex;flex-wrap:wrap;gap:12px}}
.table-option{{display:inline-flex;align-items:center;gap:10px;padding:12px 16px;border:1px solid rgba(15,23,42,.1);border-radius:var(--radius-sm);background:#ffffff;box-shadow:0 14px 26px -22px rgba(15,23,42,.25);transition:border-color .15s ease,box-shadow .15s ease,transform .15s ease}}
.table-option:hover{{border-color:rgba(14,165,233,.35);box-shadow:0 18px 32px -22px rgba(14,165,233,.35);transform:translateY(-1px)}}
.table-option input{{accent-color:var(--accent)}}
.table-option span{{font-weight:600;color:var(--text)}}
.table-empty{{margin:0;color:var(--muted)}}
.table-overview{{margin-top:22px;overflow-x:auto}}
.meta-table{{width:100%;min-width:520px;border-collapse:collapse;border-radius:var(--radius-md);overflow:hidden;box-shadow:0 22px 45px -34px rgba(15,23,42,.3);border:1px solid rgba(15,23,42,.08);background:#fff}}
.meta-table thead th{{padding:14px 18px;color:var(--muted);font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;text-align:left;background:#f1f5f9}}
.meta-table tbody td{{padding:16px 18px;border-top:1px solid rgba(15,23,42,.05);font-size:.96rem;color:var(--text);background:#fff}}
.meta-row.active td{{background:linear-gradient(90deg,rgba(14,165,233,.12),transparent)}}
.meta-name{{font-weight:600;display:flex;align-items:center;gap:10px}}
.meta-rows,.meta-index{{white-space:nowrap}}
.table-pill{{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-size:.72rem;text-transform:uppercase;letter-spacing:.14em}}
.table-overview .table-empty{{padding:12px 0}}
.segmented{{display:flex;gap:10px;padding:10px;background:linear-gradient(135deg,#f8fbff,#eef5ff);border-radius:var(--radius-sm);border:1px solid rgba(14,165,233,.15)}}
.segmented label{{position:relative;flex:1;font-weight:600}}
.segmented input{{position:absolute;opacity:0;pointer-events:none}}
.segmented span{{display:block;padding:12px 16px;border-radius:var(--radius-sm);text-align:center;color:var(--muted);transition:all .18s ease;background:transparent}}
.segmented input:checked + span{{color:#0f172a;background:#fff;border:1px solid rgba(14,165,233,.4);box-shadow:0 18px 30px -22px rgba(14,165,233,.55)}}
.field{{display:flex;flex-direction:column;gap:10px}}
.field label{{font-weight:600;color:#0f172a;font-size:1rem}}
.field input[type=text],.field input[type=number],.field input[type=file],.field select,.field textarea{{width:100%;padding:14px 16px;border-radius:var(--radius-sm);border:1px solid rgba(15,23,42,.12);background:#fafcff;color:var(--text);font-size:.98rem;transition:border-color .18s ease,box-shadow .18s ease}}
.field textarea{{min-height:110px;resize:vertical}}
.field input:focus,.field select:focus,.field textarea:focus{{outline:none;border-color:var(--accent);box-shadow:0 0 0 4px rgba(14,165,233,.18)}}
.field input[type=file]{{background:#fff;border-style:dashed;color:var(--muted)}}
.field .hint{{color:var(--muted);font-size:.84rem}}
.media-grid{{display:grid;gap:22px}}
@media(min-width:780px){{.media-grid.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.slider-row{{display:flex;gap:18px;align-items:center}}
.slider-row input[type=range]{{flex:1;accent-color:var(--accent)}}
.slider-row input[type=number]{{width:120px}}
.actions{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin-top:12px}}
button[type=submit]{{display:inline-flex;align-items:center;gap:12px;padding:14px 30px;border:none;border-radius:999px;font-weight:600;font-size:1rem;color:#fff;background:linear-gradient(135deg,var(--accent) 0%,var(--accent-strong) 90%);box-shadow:0 28px 45px -24px rgba(14,165,233,.65);cursor:pointer;transition:transform .15s ease,box-shadow .15s ease}}
button[type=submit]:hover{{transform:translateY(-1px);box-shadow:0 34px 55px -20px rgba(14,165,233,.75)}}
button[type=submit]:active{{transform:translateY(0)}}
.note{{color:var(--muted);font-size:.88rem;max-width:420px}}
.alert,.err{{margin-top:22px;padding:18px 20px;border-radius:var(--radius-sm);background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.3);color:#b91c1c;font-weight:600}}
.results{{margin-top:40px;border-radius:var(--radius-lg);background:#fff;border:1px solid var(--border);box-shadow:var(--card-shadow)}}
.results-header{{padding:28px 32px;display:flex;flex-direction:column;gap:22px;border-bottom:1px solid var(--border-strong)}}
.results-header-main{{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between}}
.results-header-main h2{{margin:0;font-size:1.5rem}}
.results-header-main span{{color:var(--muted);font-size:.9rem}}
.table-wrap{{overflow-x:auto}}
.table{{width:100%;border-collapse:collapse;min-width:840px;table-layout:fixed}}
.table thead th{{position:sticky;top:0;background:#f1f5f9;padding:15px 18px;color:var(--muted);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid rgba(15,23,42,.08)}}
.table td{{padding:16px 18px;border-bottom:1px solid rgba(15,23,42,.06);vertical-align:middle;font-size:.95rem;color:var(--text);line-height:1.5;background:#fff;word-break:break-word}}
.table tr:hover td{{background:#f8fafc}}
.table th.preview,.table td.preview{{width:200px;vertical-align:middle}}
.table td.preview{{padding:12px;display:flex;align-items:center;justify-content:center}}
.table td.preview:empty::after{{content:"—";color:var(--muted);font-size:.85rem}}
.table th.table,.table td.table{{width:140px;white-space:nowrap}}
.table th.mod,.table td.mod{{width:110px;white-space:nowrap}}
.table th.dist,.table td.dist{{width:130px;white-space:nowrap}}
.table td.id-cell{{white-space:normal;vertical-align:middle}}
.table td.snippet{{white-space:normal;vertical-align:top}}
.table td.snippet .snippet-text{{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;line-height:1.48;max-height:4.4em;word-break:break-word}}
.table td a{{color:var(--accent);text-decoration:none;font-weight:500}}
.table td a:hover{{text-decoration:underline}}
.truncate{{display:block;white-space:normal;word-break:break-word}}
.break{{word-break:break-word}}
.imgbox{{width:100%;max-width:184px;height:120px;border-radius:18px;border:1px solid rgba(15,23,42,.08);background:#f8fbff;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.imgbox img{{width:100%;height:100%;object-fit:cover;display:block}}
.kv{{display:grid;grid-template-columns:210px 1fr;gap:18px;margin-top:24px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9rem;background:#f8fbff;border-radius:var(--radius-md);padding:24px;border:1px dashed rgba(14,165,233,.3);color:var(--text)}}
.kv div{{padding:2px 0}}
details summary{{cursor:pointer;color:var(--accent-strong);font-weight:600}}
.table-wrap p{{padding:28px;color:var(--muted);margin:0}}
footer{{margin-top:52px;text-align:center;color:var(--muted);font-size:.88rem}}
@media(max-width:760px){{.shell{{padding:36px 18px 80px}}.masthead{{padding:32px}}.panel{{padding:30px}}.results-header{{padding:24px}}.table{{min-width:100%}}.kv{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main class="shell">
  <header class="masthead">
    <div class="masthead-body">
      <h1>LanceDB Search</h1>
      <p>Multimodal retrieval across documents, imagery, and LiDAR frames.</p>
    </div>
    <div class="masthead-controls">
      <div class="table-select-label">Tables</div>
      <div class="table-options">
        {table_options}
      </div>
      <span class="hint">Select one or more tables to search.</span>
    </div>
    <div class="table-overview">
      {table_overview}
    </div>
  </header>

  <section class="panel">
    <div class="panel-head">
      <h2>Build Query</h2>
      <small>Uploads capped at {max_bytes} bytes</small>
    </div>
    <form method="post" action="/search" enctype="multipart/form-data" class="form-grid" data-search-form id="search-form">
      <div class="form-main">
        <div class="field full-row">
          <label>Modalities</label>
          <div class="segmented" role="group" aria-label="Modalities">
            <label class="seg-option">
              <input type="checkbox" name="modalities" value="text" {mod_text}><span>Text</span>
            </label>
            <label class="seg-option">
              <input type="checkbox" name="modalities" value="image" {mod_image}><span>Image</span>
            </label>
            {lidar_checkbox}
          </div>
          <span class="hint">Blend multiple modalities to tighten recall; at least one must remain selected.</span>
        </div>

        <div class="field mod-section" data-mod="text" style="{text_section_style}">
          <label>Text Query</label>
          <textarea name="q_text" rows="4" placeholder="e.g. 'warehouse inventory report'">{q_text}</textarea>
          <span class="hint">A sentence or short paragraph works best; longer context is supported too.</span>
        </div>

        <div class="media-grid two mod-section" data-mod="image" style="{image_section_style}">
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

        <div class="media-grid two mod-section" data-mod="lidar" style="{lidar_section_style}">
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
      <div class="results-header-main">
        <h2>Results</h2>
        <span>Metric threshold and filters apply instantly—experiment freely.</span>
      </div>
      <div class="stat-grid stat-grid-run">
        <div class="stat"><span class="stat-icon">⏱️</span><div class="stat-meta"><span class="stat-label">Duration</span><span class="stat-value">{elapsed_ms} ms</span></div></div>
        <div class="stat"><span class="stat-icon">📐</span><div class="stat-meta"><span class="stat-label">Metric</span><span class="stat-value">{metric_name}</span></div></div>
        <div class="stat"><span class="stat-icon">🧭</span><div class="stat-meta"><span class="stat-label">Probes</span><span class="stat-value">{nprobes}</span></div></div>
        <div class="stat"><span class="stat-icon">🔁</span><div class="stat-meta"><span class="stat-label">Refine</span><span class="stat-value">{refine}</span></div></div>
      </div>
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
  const label = form.querySelector('#metric_val_label');
  const tableInputs = Array.from(document.querySelectorAll('input[name="tables"]'));
  const modalityInputs = Array.from(form.querySelectorAll('input[name="modalities"]'));

  const METRIC_META = {{
    l2: {{ min: 0.0, max: 2.0, step: 0.01, defaultValue: 2.0, label: 'Max L2 distance' }},
    cosine: {{ min: -1.0, max: 1.0, step: 0.01, defaultValue: -1.0, label: 'Min cosine/dot similarity' }},
  }};

  function resolveMeta(metric){{
    const key = (metric || '').toLowerCase();
    if (key === 'l2' || key === 'euclidean' || key === 'l2_distance'){{
      return METRIC_META.l2;
    }}
    return METRIC_META.cosine;
  }}

  function ensureTablesSelected(){{
    if (tableInputs.length === 0) return;
    const chosen = tableInputs.filter(input => input.checked);
    if (chosen.length === 0){{
      tableInputs[0].checked = true;
    }}
  }}

  tableInputs.forEach(input => {{
    input.addEventListener('change', () => {{
      const chosen = tableInputs.filter(item => item.checked);
      if (chosen.length === 0){{
        input.checked = true;
      }}
    }});
  }});

  function ensureActiveModalities(){{
    const selected = modalityInputs.filter(input => input.checked);
    if (selected.length === 0 && modalityInputs.length > 0){{
      modalityInputs[0].checked = true;
      return [modalityInputs[0].value];
    }}
    return selected.map(input => input.value);
  }}

  function syncModalSections(){{
    const active = new Set(ensureActiveModalities());
    form.querySelectorAll('.mod-section').forEach(el => {{
      const target = el.getAttribute('data-mod');
      el.style.display = active.has(target) ? '' : 'none';
    }});
  }}

  function applyMetricMeta(metric, options = {{}}){{
    if (!slider || !num) return;
    const meta = resolveMeta(metric);
    const reset = Boolean(options.reset);

    slider.min = String(meta.min);
    slider.max = String(meta.max);
    slider.step = String(meta.step);
    num.min = slider.min;
    num.max = slider.max;
    num.step = slider.step;

    if (label){{
      label.textContent = meta.label;
    }}

    let current = parseFloat(slider.value);
    if (Number.isNaN(current)){{
      current = meta.defaultValue;
    }}
    if (reset || current < meta.min || current > meta.max){{
      current = meta.defaultValue;
    }}
    const nextValue = Math.min(meta.max, Math.max(meta.min, current));
    slider.value = String(nextValue);
    num.value = slider.value;
  }}

  form.addEventListener('change', ev => {{
    if (ev.target && ev.target.name === 'tables'){{
      ensureTablesSelected();
    }}
    if (ev.target && ev.target.name === 'modalities'){{
      syncModalSections();
    }}
    if (ev.target && ev.target.name === 'metric'){{
      applyMetricMeta(ev.target.value, {{ reset: true }});
    }}
  }});

  ensureTablesSelected();
  syncModalSections();
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
    tables: Optional[List[str]] = Query(default=None),
    filter: str = Query(default="all"),
    topk: int = Query(default=5),
    where: str = Query(default=""),
    modality: str = Query(default="text"),
    modalities: Optional[List[str]] = Query(default=None),
    metric: str = Query(default=DEFAULT_METRIC),
    metric_val: float | None = Query(default=None),
):
    t0 = time.perf_counter()
    lcfg = LanceDBConfig()
    minio = MinioConfig()
    s3 = MinIOClient(minio) if SHOW_IMAGES else None

    available_tables = _available_tables()
    selected_tables = _normalize_tables(tables or [table], available_tables, TABLE_DEFAULT)
    table_display = ", ".join(selected_tables) if selected_tables else "n/a"
    table_options_html = _render_table_options(available_tables, selected_tables)
    table_handles, row_summary, index_summary, meta_all = _describe_tables(available_tables, selected_tables)
    table_overview_html = _render_table_overview(meta_all, selected_tables)
    missing_tables = [name for name in selected_tables if name not in table_handles]

    selected_modalities = normalize_modalities(
        modalities,
        fallback=modality,
        allow_lidar=ALLOW_LIDAR,
    )
    mod_text = "checked" if "text" in selected_modalities else ""
    mod_image = "checked" if "image" in selected_modalities else ""
    if ALLOW_LIDAR:
        mod_lidar = "checked" if "lidar" in selected_modalities else ""
        lidar_checkbox = (
            f"<label class='seg-option'><input type='checkbox' name='modalities' value='lidar' {mod_lidar}><span>LiDAR</span></label>"
        )
    else:
        lidar_checkbox = ""

    text_section_style = "" if "text" in selected_modalities else "display:none"
    image_section_style = "" if "image" in selected_modalities else "display:none"
    if ALLOW_LIDAR and "lidar" in selected_modalities:
        lidar_section_style = ""
    elif ALLOW_LIDAR:
        lidar_section_style = "display:none"
    else:
        lidar_section_style = "display:none"

    error_html = ""
    if missing_tables:
        missing_text = ", ".join(_escape(name) for name in missing_tables)
        error_html = f"<p class='err'>Unable to open tables: {missing_text}</p>"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    results_html = "<p>Run a query to see results.</p>"

    m_min, m_max, m_step, m_def, m_label = _metric_slider_defaults(metric)
    m_val = metric_val if (metric_val is not None) else m_def

    html_out = PAGE.format(
        table=_escape(table_display),
        rowcount=_escape(row_summary),
        elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        metric_name=_escape(_norm_metric(metric)),
        nprobes=str(LANCEDB_NPROBES),
        refine=str(LANCEDB_REFINE),
        index_summary=_escape(index_summary),
        table_options=table_options_html,
        mod_text=mod_text,
        mod_image=mod_image,
        lidar_checkbox=lidar_checkbox,
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
        text_section_style=text_section_style,
        image_section_style=image_section_style,
        lidar_section_style=lidar_section_style,
        max_bytes=MAX_UPLOAD_BYTES,
        results_html=results_html,
        error_html=error_html,
        table_overview=table_overview_html,
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
    tables: Optional[List[str]] = Form(default=None),
    primary_table: Optional[str] = Form(default=None, alias="table"),
    modalities: Optional[List[str]] = Form(default=None),
    primary_modality: Optional[str] = Form(default=None, alias="modality"),
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

    available_tables = _available_tables()
    fallback_table = primary_table or table
    selected_tables = _normalize_tables(tables or ([fallback_table] if fallback_table else None), available_tables, TABLE_DEFAULT)
    table_display = ", ".join(selected_tables) if selected_tables else "n/a"
    table_options_html = _render_table_options(available_tables, selected_tables)
    table_handles, row_summary, index_summary, meta_all = _describe_tables(available_tables, selected_tables)
    table_overview_html = _render_table_overview(meta_all, selected_tables)
    missing_tables = [name for name in selected_tables if name not in table_handles]

    selected_modalities = normalize_modalities(
        modalities,
        fallback=primary_modality,
        allow_lidar=ALLOW_LIDAR,
    )
    mod_text = "checked" if "text" in selected_modalities else ""
    mod_image = "checked" if "image" in selected_modalities else ""
    if ALLOW_LIDAR:
        mod_lidar = "checked" if "lidar" in selected_modalities else ""
        lidar_checkbox = (
            f"<label class='seg-option'><input type='checkbox' name='modalities' value='lidar' {mod_lidar}><span>LiDAR</span></label>"
        )
    else:
        lidar_checkbox = ""

    text_section_style = "" if "text" in selected_modalities else "display:none"
    image_section_style = "" if "image" in selected_modalities else "display:none"
    if ALLOW_LIDAR and "lidar" in selected_modalities:
        lidar_section_style = ""
    elif ALLOW_LIDAR:
        lidar_section_style = "display:none"
    else:
        lidar_section_style = "display:none"

    error_messages: list[str] = []
    if missing_tables:
        missing_text = ", ".join(_escape(name) for name in missing_tables)
        error_messages.append(f"<p class='err'>Unable to open tables: {missing_text}</p>")

    upload_error = False
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
        upload_error = True
        error_messages.append(f"<p class='err'>{_escape(str(e))}</p>")

    inputs = SearchInputs(
        modalities=selected_modalities,
        text_query=q_text.strip() or None,
        image_path=image_path.strip() or None,
        image_blob=img_blob,
        lidar_path=lidar_path.strip() or None,
        lidar_blob=lidar_blob,
        lidar_ext=lidar_ext,
        allow_lidar=ALLOW_LIDAR,
    )

    metric_threshold = None
    try:
        metric_threshold = (
            float(metric_val)
            if metric_val is not None and str(metric_val).strip() != ""
            else None
        )
    except Exception:
        metric_threshold = None

    combined: list[pd.DataFrame] = []
    if table_handles and not upload_error:
        for name in selected_tables:
            tbl_obj = table_handles.get(name)
            if tbl_obj is None:
                continue
            service = SearchService(tbl_obj)
            try:
                quick = _where_from_filter(filter)
                prefetch = int(min(2000, max(int(topk), int(topk) * 5)))
                df_local = service.run(
                    inputs,
                    topk=int(topk),
                    filter_hint=quick,
                    where=where.strip() or None,
                    metric=metric,
                    nprobes=LANCEDB_NPROBES,
                    refine_factor=LANCEDB_REFINE,
                    metric_threshold=metric_threshold,
                    prefetch=prefetch,
                )
            except ValueError as e:
                error_messages.append(f"<p class='err'>Vectorization error: {_escape(str(e))}</p>")
                combined = []
                break
            except Exception as e:
                error_messages.append(f"<p class='err'>Search error on table '{_escape(name)}': {_escape(str(e))}</p>")
                combined = []
                break

            if df_local is not None and len(df_local) > 0:
                df_local = df_local.copy()
                df_local["table"] = name
                combined.append(df_local)
    elif not table_handles and not available_tables:
        error_messages.append("<p class='err'>No tables available.</p>")
    elif not table_handles:
        error_messages.append("<p class='err'>No valid tables selected.</p>")

    df = None
    if combined:
        df = pd.concat(combined, ignore_index=True)
        df = df.sort_values("_distance", kind="stable").head(int(topk)).reset_index(drop=True)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    results_html = _results_table(
        df,
        s3,
        metric,
        default_table=selected_tables[0] if selected_tables else None,
    ) if df is not None else "<p>No results.</p>"

    m_min, m_max, m_step, m_def, m_label = _metric_slider_defaults(metric)
    try:
        mv_cur = (
            float(metric_val)
            if metric_val is not None and str(metric_val).strip() != ""
            else m_def
        )
    except Exception:
        mv_cur = m_def

    error_html = "".join(error_messages)

    html_out = PAGE.format(
        table=_escape(table_display),
        rowcount=_escape(row_summary),
        elapsed_ms=elapsed_ms,
        show_images="on" if SHOW_IMAGES else "off",
        metric_name=_escape(_norm_metric(metric)),
        nprobes=str(LANCEDB_NPROBES),
        refine=str(LANCEDB_REFINE),
        index_summary=_escape(index_summary),
        table_options=table_options_html,
        mod_text=mod_text,
        mod_image=mod_image,
        lidar_checkbox=lidar_checkbox,
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
        text_section_style=text_section_style,
        image_section_style=image_section_style,
        lidar_section_style=lidar_section_style,
        max_bytes=MAX_UPLOAD_BYTES,
        results_html=results_html,
        error_html=error_html,
        table_overview=table_overview_html,
        lancedb_uri=_escape(lcfg.uri),
        endpoint=_escape(minio.endpoint or ""),
        region=_escape(minio.region or ""),
        buckets=_escape(minio.buckets or ""),
        access=_escape(mask(os.getenv("AWS_ACCESS_KEY_ID") or "")),
        secret=_escape(mask(os.getenv("AWS_SECRET_ACCESS_KEY") or "")),
    )
    return HTMLResponse(content=html_out)

@app.get("/api/search")
def api_search(
    q: str = Query(""),
    modality: str = Query("text"),
    topk: int = Query(5),
    where: str = Query(""),
    metric: str = Query(DEFAULT_METRIC),
    table: str = Query(default=TABLE_DEFAULT),
    tables: Optional[List[str]] = Query(default=None),
):
    lcfg = LanceDBConfig()
    mgr = LanceDBManager(lcfg)
    try:
        available = mgr.list_tables()
    except Exception:
        available = []
    selected_tables = _normalize_tables(tables or [table], available, table)

    table_handles: dict[str, object] = {}
    for name in selected_tables:
        try:
            table_handles[name] = mgr.get_table(name)
        except Exception:
            continue

    if not table_handles:
        return JSONResponse({"error": "no valid tables available"}, status_code=400)

    modalities_norm = normalize_modalities([modality], fallback=modality, allow_lidar=ALLOW_LIDAR)
    inputs = SearchInputs(
        modalities=modalities_norm,
        text_query=q if "text" in modalities_norm else None,
        image_path=q if "image" in modalities_norm else None,
        lidar_path=q if "lidar" in modalities_norm else None,
        allow_lidar=ALLOW_LIDAR,
    )

    prefetch = int(min(2000, max(int(topk), int(topk) * 5)))
    results: list[pd.DataFrame] = []
    for name in selected_tables:
        tbl = table_handles.get(name)
        if tbl is None:
            continue
        service = SearchService(tbl)
        try:
            df_local = service.run(
                inputs,
                topk=int(topk),
                filter_hint=None,
                where=where.strip() or None,
                metric=metric,
                nprobes=LANCEDB_NPROBES,
                refine_factor=LANCEDB_REFINE,
                metric_threshold=None,
                prefetch=prefetch,
            )
        except ValueError as e:
            return JSONResponse({"error": f"vectorize failed: {str(e)}", "table": name}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": f"search failed on table {name}: {str(e)}"}, status_code=400)
        if df_local is not None and len(df_local) > 0:
            df_local = df_local.copy()
            df_local["table"] = name
            results.append(df_local)

    if results:
        df = pd.concat(results, ignore_index=True)
        df = df.sort_values("_distance", kind="stable").head(int(topk)).reset_index(drop=True)
    else:
        df = pd.DataFrame()

    return JSONResponse({"count": int(len(df)), "rows": df.to_dict(orient="records")})

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "OK"
