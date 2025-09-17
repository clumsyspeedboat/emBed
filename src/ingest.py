"""
High-performance ingestion of texts, PDFs, images, and LiDAR from MinIO into LanceDB.

Design goals:
    - Scalable: concurrent S3 downloads; batched embedding; chunked writes.
    - Idempotent: stable primary key = object key; batch upsert to avoid duplicates.
    - Modular: dependency-injected embedders; no globals; helpers isolated for testability.
    - Efficient: one pass over keys, per-batch downloads; avoid DataFrame hops; no silent failures.

Modality support:
    - Text: .txt, .md (UTF-8 decode)
    - PDF: via pypdf (limited pages; char cap)
    - Images: .png, .jpg, .jpeg, .webp, .bmp, .jfif (raw bytes embedded)
    - LiDAR: .pcd (ASCII or binary) + .bin (float32 packed)

Output schema (one row per object):
    - id (str)          -> object key (primary key)
    - modality (str)    -> 'text' | 'pdf' | 'image' | 'lidar'
    - embedding (list[float])
    - text (str|None)   -> text content summary for text/pdf
    - image (None)
    - lidar (None)
    - path (str)        -> s3://bucket/key
"""

from __future__ import annotations

import io
import os
import warnings
import json
import time
import socket
import datetime as dt
from typing import Iterable, List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pandas import DataFrame
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.storage import MinIOClient
from src.lancedb_manager import LanceDBManager
from src.embedding_runtime import (
    get_text_embedder,
    get_image_embedder,
    get_lidar_embedder,
)

# -------- configuration --------

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
PDF_EXTS = {".pdf"}
LIDAR_EXTS = {".pcd", ".bin"}
DEFAULT_EXTS = TEXT_EXTS | IMAGE_EXTS | PDF_EXTS | LIDAR_EXTS

PDF_TEXT_CHAR_LIMIT = 50_000
DEFAULT_PDF_MAX_PAGES = 8

# parallelism / batching (env-tunable)
def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

DEFAULT_WORKERS = _get_int_env("INGEST_WORKERS", 16)
DEFAULT_EMBED_BATCH = _get_int_env("INGEST_EMBED_BATCH", 64)
DEFAULT_WRITE_CHUNK = _get_int_env("INGEST_WRITE_CHUNK", 5000)
# LiDAR batches are heavier; keep smaller by default, but allow override
LIDAR_BATCH_DIVISOR = max(1, _get_int_env("LIDAR_BATCH_DIVISOR", 8))  # lidar_batch = max(1, embed_batch // LIDAR_BATCH_DIVISOR)
LIDAR_EMBED_BATCH = _get_int_env("LIDAR_EMBED_BATCH", 0)  # 0 means derive via divisor

DEFAULT_PDF_MAX_PAGES = _get_int_env("INGEST_PDF_MAX_PAGES", 8)

# progress throttling (print less often)
INGEST_PROGRESS_EVERY = _get_int_env("INGEST_PROGRESS_EVERY", 0)  # 0 => use embed_batch/small_batch

# -------- small utilities --------

def _normalize_exts(exts: Optional[Iterable[str]]) -> Optional[set[str]]:
    if exts is None:
        return None
    out = set()
    for e in exts:
        e = e.lower()
        out.add(e if e.startswith(".") else f".{e}")
    return out

def _to_list_vec(v: np.ndarray) -> List[float]:
    """Ensure (D,) list regardless of (D,) or (1,D)."""
    arr = np.asarray(v)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype("float32").tolist()

def _s3_path(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"

def _batch(seq: Iterable, n: int):
    it = iter(seq)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            pass
        if not chunk:
            break
        yield chunk

# -------- I/O helpers --------

def _download_many(s3: MinIOClient, bucket: str, keys: List[str], workers: int) -> Dict[str, bytes | Exception]:
    out: Dict[str, bytes | Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(s3.get_object_bytes, bucket, k): k for k in keys}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                out[k] = fut.result()
            except Exception as e:
                out[k] = e
    return out

# -------- text / pdf decoding --------

def _extract_pdf_text(blob: bytes, max_pages: int) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reader = PdfReader(io.BytesIO(blob))
        except (PdfReadError, Exception):
            return ""
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    text = "\n".join(parts)
    return text[:PDF_TEXT_CHAR_LIMIT] if len(text) > PDF_TEXT_CHAR_LIMIT else text

def _decode_text_bytes(blob: bytes) -> str:
    try:
        return blob.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -------- LiDAR loaders (no open3d dependency) --------

def _pcd_parse_header(blob: bytes) -> dict:
    """Parse PCD header, return dict with keys and header_len."""
    header_lines = []
    pos = 0
    while True:
        nl = blob.find(b"\n", pos)
        if nl == -1:
            break
        line = blob[pos:nl].decode("utf-8", errors="ignore").strip()
        header_lines.append(line)
        pos = nl + 1
        if line.upper().startswith("DATA"):
            break
    meta = {"header_len": pos}
    for ln in header_lines:
        up = ln.upper()
        if up.startswith("FIELDS"):
            meta["FIELDS"] = ln.split()[1:]
        elif up.startswith("SIZE"):
            meta["SIZE"] = [int(x) for x in ln.split()[1:]]
        elif up.startswith("TYPE"):
            meta["TYPE"] = ln.split()[1:]
        elif up.startswith("COUNT"):
            meta["COUNT"] = [int(x) for x in ln.split()[1:]]
        elif up.startswith("WIDTH"):
            meta["WIDTH"] = int(ln.split()[1])
        elif up.startswith("HEIGHT"):
            meta["HEIGHT"] = int(ln.split()[1])
        elif up.startswith("POINTS"):
            try:
                meta["POINTS"] = int(ln.split()[1])
            except Exception:
                pass
        elif up.startswith("DATA"):
            meta["DATA"] = ln.split()[1].lower()
    return meta

def _pcd_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    type_map = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
    }
    fields_expanded = []
    for f, sz, tp, cnt in zip(fields, sizes, types, counts):
        if cnt == 1:
            fields_expanded.append((f, type_map.get((tp.upper(), sz), np.float32)))
        else:
            fields_expanded.append((f, (type_map.get((tp.upper(), sz), np.float32), cnt)))
    return np.dtype(fields_expanded)

def _pcd_to_numpy(blob: bytes) -> np.ndarray:
    """Return NxC float32 array with at least XYZ; include intensity if present."""
    meta = _pcd_parse_header(blob)
    data_mode = meta.get("DATA", "ascii")
    start = meta.get("header_len", 0)
    fields = meta.get("FIELDS", [])
    sizes = meta.get("SIZE", [4] * len(fields))
    types = meta.get("TYPE", ["F"] * len(fields))
    counts = meta.get("COUNT", [1] * len(fields))
    n_points = meta.get("POINTS")
    width = meta.get("WIDTH")
    height = meta.get("HEIGHT")

    def idx_of(name: str) -> int | None:
        try:
            return fields.index(name)
        except ValueError:
            return None

    ix_x, ix_y, ix_z = idx_of("x"), idx_of("y"), idx_of("z")
    ix_i = idx_of("intensity")

    if any(v is None for v in (ix_x, ix_y, ix_z)):
        # Fallback: parse as whitespace floats and reshape to (N, C)
        txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
        arr = np.asarray([float(x) for x in txt], dtype=np.float32)
        cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
        return arr.reshape(-1, cols)[:, :3].astype(np.float32)

    if data_mode == "ascii":
        pts = []
        for ln in blob[start:].decode("utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) < max(ix_x, ix_y, ix_z) + 1:
                continue
            x = float(parts[ix_x]); y = float(parts[ix_y]); z = float(parts[ix_z])
            if ix_i is not None and len(parts) > ix_i:
                try:
                    i = float(parts[ix_i]); pts.append((x, y, z, i)); continue
                except Exception:
                    pass
            pts.append((x, y, z))
        return np.asarray(pts, dtype=np.float32)

    if "binary" in data_mode:
        dt = _pcd_dtype(fields, sizes, types, counts)
        if n_points is not None:
            n = n_points
        elif width and height:
            n = int(width) * int(height)
        else:
            n = None
        buf = memoryview(blob)[start:]
        arr = np.frombuffer(buf, dtype=dt, count=n)
        out = np.zeros((arr.shape[0], 3 + (1 if ix_i is not None else 0)), dtype=np.float32)
        out[:, 0] = np.asarray(arr["x"], dtype=np.float32)
        out[:, 1] = np.asarray(arr["y"], dtype=np.float32)
        out[:, 2] = np.asarray(arr["z"], dtype=np.float32)
        if ix_i is not None and "intensity" in arr.dtype.names:
            out[:, 3] = np.asarray(arr["intensity"], dtype=np.float32)
        return out

    # Unknown mode → best-effort ASCII
    txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
    arr = np.asarray([float(x) for x in txt], dtype=np.float32)
    cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
    return arr.reshape(-1, cols)[:, :3].astype(np.float32)

def _load_lidar_bytes(ext: str, blob: bytes) -> np.ndarray:
    if ext == ".bin":
        arr = np.frombuffer(blob, dtype=np.float32)
        cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
        return arr.reshape(-1, cols)
    elif ext == ".pcd":
        return _pcd_to_numpy(blob)
    else:
        raise ValueError("Unsupported LiDAR extension")

# -------- writer (upsert) --------

class _UpsertWriter:
    """Batch upsert helper: creates table once (mode=overwrite optionally),
    then for each batch: delete ids in-batch from table and add rows.
    """
    def __init__(self, manager: LanceDBManager):
        self.mgr = manager
        self._opened: Dict[str, bool] = {}  # table_name -> created/opened

    @staticmethod
    def _escape_ids(ids: List[str]) -> List[str]:
        return [("'" + x.replace("'", "''") + "'") for x in ids]

    def ensure_table(self, table_name: str, rows: List[dict], mode: str) -> None:
        if self._opened.get(table_name):
            return
        if mode == "overwrite":
            self.mgr.create_table(table_name, rows, mode="overwrite")
        else:
            # append mode: create if missing
            try:
                _ = self.mgr.get_table(table_name)
            except Exception:
                # create with at least one row to get schema
                self.mgr.create_table(table_name, rows[:1], mode="overwrite")
        self._opened[table_name] = True

    def upsert_rows(self, table_name: str, rows: List[dict]) -> int:
        if not rows:
            return 0
        tbl = self.mgr.get_table(table_name)
        # prefer upsert if available (newer lancedb)
        up = getattr(tbl, "upsert", None)
        if callable(up):
            up(rows)
            return len(rows)
        # fallback: delete existing ids then add rows
        ids = [r["id"] for r in rows]
        CHUNK = 1000
        for j in range(0, len(ids), CHUNK):
            chunk = self._escape_ids(ids[j:j + CHUNK])
            tbl.delete(f"id IN ({','.join(chunk)})")
        tbl.add(rows)
        return len(rows)

# -------- main ingest --------

def ingest_s3_objects(
    manager: LanceDBManager,
    s3: MinIOClient,
    bucket: str,
    prefix: str,
    table_name: str,
    include_ext: Optional[Iterable[str]] = None,
    exclude_ext: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    dry_run: bool = False,
    mode: str = "overwrite",  # or "append"
    *,
    workers: int = DEFAULT_WORKERS,
    embed_batch: int = DEFAULT_EMBED_BATCH,
    write_chunk: int = DEFAULT_WRITE_CHUNK,
    pdf_max_pages: int = DEFAULT_PDF_MAX_PAGES,
    # dependency-injected embedders:
    text_embedder=None,
    image_embedder=None,
    lidar_embedder=None,
    # streaming options
    stream_list: bool = False,
    # optional per-run overrides
    progress_every: Optional[int] = None,
    lidar_embed_batch: Optional[int] = None,
    lidar_batch_divisor: Optional[int] = None,
) -> int:
    """Scan s3://bucket/prefix and write rows into `table_name`.
    Returns number of rows processed (planned if dry_run=True)."""

    # deps
    text_embedder = text_embedder or get_text_embedder()
    image_embedder = image_embedder or get_image_embedder()
    lidar_embedder = lidar_embedder or get_lidar_embedder()

    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()

    # (optional) silence TLS warnings for self-signed/HTTP endpoints
    try:
        mv = str(os.getenv("MINIO_VERIFY", "")).strip().lower()
        sup = str(os.getenv("MINIO_SUPPRESS_TLS_WARN", "")).strip().lower()
        if sup in ("1", "true", "yes") or mv in ("0", "false", "no"):
            import urllib3  # type: ignore
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    # 1) list keys (optionally streaming)
    print("Listing objects...")
    incl = _normalize_exts(include_ext) or DEFAULT_EXTS
    excl = _normalize_exts(exclude_ext) or set()

    def _filtered_keys_stream():
        count = 0
        for key in s3.iter_objects(bucket=bucket, prefix=prefix or ""):
            ext = os.path.splitext(key)[1].lower()
            if ext in excl:
                continue
            if ext in incl:
                yield key
                count += 1
                if max_files and count >= max_files:
                    return

    # compute local effective knobs
    eff_progress_every = INGEST_PROGRESS_EVERY if progress_every is None else int(progress_every)
    eff_lidar_embed_batch = None
    if lidar_embed_batch is not None and int(lidar_embed_batch) > 0:
        eff_lidar_embed_batch = int(lidar_embed_batch)
    eff_lidar_divisor = int(lidar_batch_divisor) if lidar_batch_divisor else LIDAR_BATCH_DIVISOR

    if stream_list:
        filtered_iter = _filtered_keys_stream()
        if dry_run:
            # stream count+sample
            n = 0
            samp: List[str] = []
            for k in filtered_iter:
                n += 1
                if len(samp) < 10:
                    samp.append(k)
            print(f"[DRY-RUN] Would ingest {n} objects into '{table_name}' from s3://{bucket}/{prefix}")
            for k in samp:
                print("  -", k)
            return n
        # proceed streaming
        writer = _UpsertWriter(manager)
        total_written = 0
        bidx = 0
        buf: List[str] = []
        for k in filtered_iter:
            buf.append(k)
            if len(buf) >= write_chunk:
                bidx += 1
                master_keys = buf
                buf = []
                print(f"\n--- Batch {bidx}/? ({len(master_keys)} files) ---")
                # 2) download per-batch
                blobs = _download_many(s3, bucket, master_keys, workers)
                # process the batch
                # (reuse existing per-modality logic)
                # inline the body via helper
                total_written += _process_one_batch(
                    bidx, master_keys, blobs, manager, writer, bucket, table_name,
                    embed_batch, write_chunk, pdf_max_pages, text_embedder, image_embedder, lidar_embedder,
                    eff_progress_every, eff_lidar_embed_batch, eff_lidar_divisor
                )
        if buf:
            bidx += 1
            master_keys = buf
            print(f"\n--- Batch {bidx}/? ({len(master_keys)} files) ---")
            blobs = _download_many(s3, bucket, master_keys, workers)
            total_written += _process_one_batch(
                bidx, master_keys, blobs, manager, writer, bucket, table_name,
                embed_batch, write_chunk, pdf_max_pages, text_embedder, image_embedder, lidar_embedder,
                eff_progress_every, eff_lidar_embed_batch, eff_lidar_divisor
            )
        # log and return
        duration_s = round(time.perf_counter() - t0, 3)
        print(f"\n[BUILD COMPLETE] table='{table_name}' rows={total_written} mode={mode} duration={duration_s}s")
        try:
            with open("build_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{table_name}",
                    "table": table_name, "bucket": bucket, "prefix": prefix,
                    "rows": total_written, "mode": mode,
                    "include_ext": sorted(list(incl)) if incl else None,
                    "exclude_ext": sorted(list(excl)) if excl else None,
                    "started_at": started_at.isoformat().replace("+00:00", "Z"),
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "duration_s": duration_s,
                    "host": socket.gethostname(),
                }) + "\n")
        except Exception:
            pass
        return total_written

    # non-streaming path (original behavior)
    keys = s3.list_objects(bucket=bucket, prefix=prefix or "")
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return 0

    filtered: List[str] = []
    for key in keys:
        ext = os.path.splitext(key)[1].lower()
        if ext in excl:
            continue
        if ext in incl:
            filtered.append(key)
    if max_files and max_files > 0:
        filtered = filtered[:max_files]

    total_to_process = len(filtered)
    if dry_run:
        by_ext: Dict[str, int] = {}
        for k in filtered:
            e = os.path.splitext(k)[1].lower()
            by_ext[e] = by_ext.get(e, 0) + 1
        print(f"[DRY-RUN] Would ingest {len(filtered)} objects into '{table_name}' from s3://{bucket}/{prefix}")
        if by_ext:
            print("[DRY-RUN] By extension:", ", ".join(f"{k}:{v}" for k, v in sorted(by_ext.items())))
        for k in filtered[:10]:
            print("  -", k)
        return len(filtered)

    writer = _UpsertWriter(manager)
    total_written = 0
    num_batches = (total_to_process + write_chunk - 1) // write_chunk

    for bidx, master_keys in enumerate(_batch(filtered, write_chunk), start=1):
        print(f"\n--- Batch {bidx}/{num_batches} ({len(master_keys)} files) ---")
        # 2) download per-batch
        blobs = _download_many(s3, bucket, master_keys, workers)

        # split by modality
        img_keys = []
        pdf_keys = []
        text_keys = []
        lidar_keys = []
        for k in master_keys:
            ext = os.path.splitext(k)[1].lower()
            if ext in IMAGE_EXTS:
                img_keys.append(k)
            elif ext in PDF_EXTS:
                pdf_keys.append(k)
            elif ext in TEXT_EXTS:
                text_keys.append(k)
            elif ext in LIDAR_EXTS:
                lidar_keys.append(k)

        rows: List[dict] = []
        seen_ids: set[str] = set()

        # 3) images (batched)
        if img_keys:
            good = [k for k in img_keys if isinstance(blobs.get(k), (bytes, bytearray))]
            processed = 0
            for chunk_keys in _batch(good, embed_batch):
                batch_blobs = [blobs[k] for k in chunk_keys]
                try:
                    vecs = image_embedder.embed(batch_blobs)  # (B,D)
                except Exception:
                    vecs = [image_embedder.embed(b) for b in batch_blobs]
                for k, v in zip(chunk_keys, vecs):
                    if k in seen_ids:
                        continue
                    rows.append({
                        "id": k, "modality": "image", "embedding": _to_list_vec(v),
                        "text": None, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                    })
                    seen_ids.add(k)
                processed += len(chunk_keys)
                prog_every = INGEST_PROGRESS_EVERY or embed_batch
                if processed % max(1, prog_every) == 0 or processed >= len(img_keys):
                    print(f"  images: {processed}/{len(img_keys)}")

        # 4) text files (batched)
        if text_keys:
            good_pairs: List[Tuple[str, str]] = []
            for k in text_keys:
                b = blobs.get(k)
                if isinstance(b, (bytes, bytearray)):
                    good_pairs.append((k, _decode_text_bytes(b)))
            processed = 0
            for chunk in _batch(good_pairs, embed_batch):
                texts = [t for _, t in chunk]
                try:
                    vecs = text_embedder.embed(texts)
                except Exception:
                    vecs = [text_embedder.embed(t) for t in texts]
                for (k, text), v in zip(chunk, vecs):
                    if k in seen_ids:
                        continue
                    rows.append({
                        "id": k, "modality": "text", "embedding": _to_list_vec(v),
                        "text": text, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                    })
                    seen_ids.add(k)
                processed += len(chunk)
                prog_every = INGEST_PROGRESS_EVERY or embed_batch
                if processed % max(1, prog_every) == 0 or processed >= len(good_pairs):
                    print(f"  text: {processed}/{len(good_pairs)}")

        # 5) PDFs (batched)
        if pdf_keys:
            pairs: List[Tuple[str, str]] = []
            for k in pdf_keys:
                b = blobs.get(k)
                if isinstance(b, (bytes, bytearray)):
                    txt = _extract_pdf_text(b, max_pages=pdf_max_pages)
                    pairs.append((k, txt if txt.strip() else ""))
            processed = 0
            for chunk in _batch(pairs, embed_batch):
                texts = [t for _, t in chunk]
                try:
                    vecs = text_embedder.embed(texts)
                except Exception:
                    vecs = [text_embedder.embed(t) for t in texts]
                for (k, text), v in zip(chunk, vecs):
                    if k in seen_ids:
                        continue
                    rows.append({
                        "id": k, "modality": "pdf", "embedding": _to_list_vec(v),
                        "text": text or "Empty PDF document", "image": None, "lidar": None, "path": _s3_path(bucket, k)
                    })
                    seen_ids.add(k)
                processed += len(chunk)
                prog_every = INGEST_PROGRESS_EVERY or embed_batch
                if processed % max(1, prog_every) == 0 or processed >= len(pairs):
                    print(f"  pdfs: {processed}/{len(pairs)}")

        # 6) LiDAR (batched)
        if lidar_keys:
            small_batch = LIDAR_EMBED_BATCH if LIDAR_EMBED_BATCH > 0 else max(1, embed_batch // LIDAR_BATCH_DIVISOR)
            good_lidar = [k for k in lidar_keys if isinstance(blobs.get(k), (bytes, bytearray))]
            processed = 0
            for chunk_keys in _batch(good_lidar, small_batch):
                pts_list = []
                valid_keys = []
                for k in chunk_keys:
                    ext = os.path.splitext(k)[1].lower()
                    try:
                        pts = _load_lidar_bytes(ext, blobs[k])
                        if pts.size > 0:
                            pts_list.append(pts)
                            valid_keys.append(k)
                    except Exception:
                        continue
                if not valid_keys:
                    continue
                try:
                    if hasattr(lidar_embedder, "embed_pointclouds"):
                        vecs = lidar_embedder.embed_pointclouds(pts_list)  # (B,D)
                    else:
                        vecs = [lidar_embedder.embed(pts) for pts in pts_list]
                except Exception:
                    vecs = [lidar_embedder.embed(pts) for pts in pts_list]
                for k, v in zip(valid_keys, vecs):
                    if k in seen_ids:
                        continue
                    rows.append({
                        "id": k, "modality": "lidar", "embedding": _to_list_vec(v),
                        "text": None, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                    })
                    seen_ids.add(k)
                processed += len(valid_keys)
                prog_every = INGEST_PROGRESS_EVERY or small_batch
                if processed % max(1, prog_every) == 0 or processed >= len(good_lidar):
                    print(f"  lidar: {processed}/{len(good_lidar)}")

        if not rows:
            print("  (no rows in this batch)")
            continue

        # First batch may create table depending on mode.
        if bidx == 1:
            writer.ensure_table(table_name, rows, mode)

        written = writer.upsert_rows(table_name, rows)
        total_written += written
        print(f"  wrote {written} rows (total {total_written}/{total_to_process})")

    duration_s = round(time.perf_counter() - t0, 3)
    finished_at = dt.datetime.now(dt.timezone.utc)
    log_rec = {
        "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{table_name}",
        "table": table_name,
        "bucket": bucket,
        "prefix": prefix,
        "rows": total_written,
        "mode": mode,
        "include_ext": sorted(list(incl)) if incl else None,
        "exclude_ext": sorted(list(excl)) if excl else None,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_s": duration_s,
        "host": socket.gethostname(),
    }

    print(f"\n[BUILD COMPLETE] table='{table_name}' rows={total_written} mode={mode} duration={duration_s}s")

    try:
        with open("build_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec) + "\n")
    except Exception:
        pass

    try:
        meta_tbl = None
        try:
            meta_tbl = manager.get_table("__build_log")
        except Exception:
            manager.create_table("__build_log", [log_rec], mode="overwrite")
            meta_tbl = manager.get_table("__build_log")
        if meta_tbl is not None:
            rid = log_rec["run_id"].replace("'", "''")
            meta_tbl.delete(f"run_id = '{rid}'")
            meta_tbl.add([log_rec])
    except Exception:
        pass

    return total_written


def _process_one_batch(
    bidx: int,
    master_keys: List[str],
    blobs: Dict[str, bytes | Exception],
    manager: LanceDBManager,
    writer: _UpsertWriter,
    bucket: str,
    table_name: str,
    embed_batch: int,
    write_chunk: int,
    pdf_max_pages: int,
    text_embedder,
    image_embedder,
    lidar_embedder,
    progress_every: int,
    lidar_embed_batch_eff: Optional[int],
    lidar_batch_divisor_eff: int,
) -> int:
    """Process one batch (shared by streaming/non-streaming paths). Returns rows written."""
    # split by modality
    img_keys: List[str] = []
    pdf_keys: List[str] = []
    text_keys: List[str] = []
    lidar_keys: List[str] = []
    for k in master_keys:
        ext = os.path.splitext(k)[1].lower()
        if ext in IMAGE_EXTS:
            img_keys.append(k)
        elif ext in PDF_EXTS:
            pdf_keys.append(k)
        elif ext in TEXT_EXTS:
            text_keys.append(k)
        elif ext in LIDAR_EXTS:
            lidar_keys.append(k)

    rows: List[dict] = []
    seen_ids: set[str] = set()

    # reuse same body as in main loop by copying the logic
    # images
    if img_keys:
        good = [k for k in img_keys if isinstance(blobs.get(k), (bytes, bytearray))]
        processed = 0
        for chunk_keys in _batch(good, embed_batch):
            batch_blobs = [blobs[k] for k in chunk_keys]
            try:
                vecs = image_embedder.embed(batch_blobs)
            except Exception:
                vecs = [image_embedder.embed(b) for b in batch_blobs]
            for k, v in zip(chunk_keys, vecs):
                if k in seen_ids:
                    continue
                rows.append({
                    "id": k, "modality": "image", "embedding": _to_list_vec(v),
                    "text": None, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                })
                seen_ids.add(k)
            processed += len(chunk_keys)
            prog_every = (progress_every if progress_every is not None else INGEST_PROGRESS_EVERY) or embed_batch
            if processed % max(1, prog_every) == 0 or processed >= len(img_keys):
                print(f"  images: {processed}/{len(img_keys)}")

    # text
    if text_keys:
        good_pairs: List[Tuple[str, str]] = []
        for k in text_keys:
            b = blobs.get(k)
            if isinstance(b, (bytes, bytearray)):
                good_pairs.append((k, _decode_text_bytes(b)))
        processed = 0
        for chunk in _batch(good_pairs, embed_batch):
            texts = [t for _, t in chunk]
            try:
                vecs = text_embedder.embed(texts)
            except Exception:
                vecs = [text_embedder.embed(t) for t in texts]
            for (k, text), v in zip(chunk, vecs):
                if k in seen_ids:
                    continue
                rows.append({
                    "id": k, "modality": "text", "embedding": _to_list_vec(v),
                    "text": text, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                })
                seen_ids.add(k)
            processed += len(chunk)
            prog_every = (progress_every if progress_every is not None else INGEST_PROGRESS_EVERY) or embed_batch
            if processed % max(1, prog_every) == 0 or processed >= len(good_pairs):
                print(f"  text: {processed}/{len(good_pairs)}")

    # pdfs
    if pdf_keys:
        pairs: List[Tuple[str, str]] = []
        for k in pdf_keys:
            b = blobs.get(k)
            if isinstance(b, (bytes, bytearray)):
                txt = _extract_pdf_text(b, max_pages=pdf_max_pages)
                pairs.append((k, txt if txt.strip() else ""))
        processed = 0
        for chunk in _batch(pairs, embed_batch):
            texts = [t for _, t in chunk]
            try:
                vecs = text_embedder.embed(texts)
            except Exception:
                vecs = [text_embedder.embed(t) for t in texts]
            for (k, text), v in zip(chunk, vecs):
                if k in seen_ids:
                    continue
                rows.append({
                    "id": k, "modality": "pdf", "embedding": _to_list_vec(v),
                    "text": text or "Empty PDF document", "image": None, "lidar": None, "path": _s3_path(bucket, k)
                })
                seen_ids.add(k)
            processed += len(chunk)
            prog_every = (progress_every if progress_every is not None else INGEST_PROGRESS_EVERY) or embed_batch
            if processed % max(1, prog_every) == 0 or processed >= len(pairs):
                print(f"  pdfs: {processed}/{len(pairs)}")

    # lidar
    if lidar_keys:
        small_batch = (
            int(lidar_embed_batch_eff)
            if (lidar_embed_batch_eff is not None and int(lidar_embed_batch_eff) > 0)
            else max(1, embed_batch // max(1, int(lidar_batch_divisor_eff)))
        )
        good_lidar = [k for k in lidar_keys if isinstance(blobs.get(k), (bytes, bytearray))]
        processed = 0
        for chunk_keys in _batch(good_lidar, small_batch):
            pts_list = []
            valid_keys = []
            for k in chunk_keys:
                ext = os.path.splitext(k)[1].lower()
                try:
                    pts = _load_lidar_bytes(ext, blobs[k])
                    if pts.size > 0:
                        pts_list.append(pts)
                        valid_keys.append(k)
                except Exception:
                    continue
            if not valid_keys:
                continue
            try:
                if hasattr(lidar_embedder, "embed_pointclouds"):
                    vecs = lidar_embedder.embed_pointclouds(pts_list)
                else:
                    vecs = [lidar_embedder.embed(pts) for pts in pts_list]
            except Exception:
                vecs = [lidar_embedder.embed(pts) for pts in pts_list]
            for k, v in zip(valid_keys, vecs):
                if k in seen_ids:
                    continue
                rows.append({
                    "id": k, "modality": "lidar", "embedding": _to_list_vec(v),
                    "text": None, "image": None, "lidar": None, "path": _s3_path(bucket, k)
                })
                seen_ids.add(k)
            processed += len(valid_keys)
            prog_every = (progress_every if progress_every is not None else INGEST_PROGRESS_EVERY) or small_batch
            if processed % max(1, prog_every) == 0 or processed >= len(good_lidar):
                print(f"  lidar: {processed}/{len(good_lidar)}")

    if not rows:
        print("  (no rows in this batch)")
        return 0

    if bidx == 1:
        writer.ensure_table(table_name, rows, mode="append")  # table should exist or be created
    written = writer.upsert_rows(table_name, rows)
    print(f"  wrote {written} rows (stream)")
    return written
