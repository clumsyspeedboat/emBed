"""Ingest images, PDFs, and LiDAR from MinIO into a LanceDB table.

Features:
- Dry-run (preview what would be ingested)
- Include/exclude extensions
- Max files limit
- Append or overwrite modes
- Idempotent ingestion (deduplicated by primary key `id`)
- Parallel S3 downloads, batched embedding, chunked writes
- Memory-efficient streaming for large buckets
"""

from __future__ import annotations

import io
import os
import warnings
from typing import List, Dict, Iterable, Optional
import time, socket, json, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pandas import DataFrame
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Corrected absolute imports
from src.storage import MinIOClient
from src.embedding import TextEmbedder, ImageEmbedder, LidarBEVEmbedder
from src.lancedb_manager import LanceDBManager

# Initialize embedders once (CPU/GPU auto-picked in embedding.py)
text_embedder = TextEmbedder()
img_embedder = ImageEmbedder()
lidar_embedder = LidarBEVEmbedder(img_embedder)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
PDF_EXTS = {".pdf"}
LIDAR_EXTS = {".pcd", ".bin"}
DEFAULT_EXTS = IMAGE_EXTS | PDF_EXTS | LIDAR_EXTS

PDF_TEXT_CHAR_LIMIT = 50_000

# Parallelism / batching defaults (tunable)
DEFAULT_WORKERS = 16
DEFAULT_EMBED_BATCH = 64
DEFAULT_WRITE_CHUNK = 5000
DEFAULT_PDF_MAX_PAGES = 8


def _extract_pdf_text(blob: bytes, max_pages: int = DEFAULT_PDF_MAX_PAGES) -> str:
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
    if len(text) > PDF_TEXT_CHAR_LIMIT:
        text = text[:PDF_TEXT_CHAR_LIMIT]
    return text


# -------- LiDAR loaders (no open3d) --------

def _pcd_parse_header(blob: bytes) -> dict:
    """Parse PCD header, return dict with keys: FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT, POINTS, DATA, header_len."""
    header_lines = []
    pos = 0
    # Read line by line until "DATA ..."
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
            meta["DATA"] = ln.split()[1].lower()  # 'ascii' or 'binary' / 'binary_compressed' (not supported)
    return meta


def _pcd_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    """Build numpy dtype for binary PCD."""
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
            # e.g., FIELDS normal SIZE 4 TYPE F COUNT 3
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

    # Identify indices we care about
    def idx_of(name: str) -> int | None:
        try:
            return fields.index(name)
        except ValueError:
            return None

    ix_x, ix_y, ix_z = idx_of("x"), idx_of("y"), idx_of("z")
    ix_i = idx_of("intensity")
    needed = [ix_x, ix_y, ix_z]
    if any(v is None for v in needed):
        # Fallback ASCII best-effort
        txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
        arr = np.asarray([float(x) for x in txt], dtype=np.float32)
        cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
        out = arr.reshape(-1, cols)[:, :3]
        return out.astype(np.float32)

    if data_mode == "ascii":
        txt = blob[start:].decode("utf-8", errors="ignore").strip().splitlines()
        pts = []
        for ln in txt:
            parts = ln.split()
            if len(parts) < max(ix_x, ix_y, ix_z) + 1:
                continue
            x = float(parts[ix_x])
            y = float(parts[ix_y])
            z = float(parts[ix_z])
            if ix_i is not None and len(parts) > ix_i:
                try:
                    i = float(parts[ix_i])
                    pts.append((x, y, z, i))
                    continue
                except Exception:
                    pass
            pts.append((x, y, z))
        return np.asarray(pts, dtype=np.float32)

    if "binary" in data_mode:  # supports 'binary' (not compressed)
        dt = _pcd_dtype(fields, sizes, types, counts)
        # Determine number of points
        if n_points is not None:
            n = n_points
        elif width and height:
            n = int(width) * int(height)
        else:
            n = None
        buf = memoryview(blob)[start:]
        arr = np.frombuffer(buf, dtype=dt, count=n)
        # Build output: x,y,z,(intensity?)
        out = np.zeros((arr.shape[0], 3 + (1 if ix_i is not None else 0)), dtype=np.float32)
        out[:, 0] = np.asarray(arr["x"], dtype=np.float32)
        out[:, 1] = np.asarray(arr["y"], dtype=np.float32)
        out[:, 2] = np.asarray(arr["z"], dtype=np.float32)
        if ix_i is not None and "intensity" in arr.dtype.names:
            out[:, 3] = np.asarray(arr["intensity"], dtype=np.float32)
        return out

    # Unknown mode → fallback ASCII best-effort
    txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
    arr = np.asarray([float(x) for x in txt], dtype=np.float32)
    cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
    out = arr.reshape(-1, cols)[:, :3]
    return out.astype(np.float32)


def _load_lidar_bytes(ext: str, blob: bytes) -> np.ndarray:
    if ext == ".bin":
        arr = np.frombuffer(blob, dtype=np.float32)
        cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
        return arr.reshape(-1, cols)
    elif ext == ".pcd":
        return _pcd_to_numpy(blob)
    else:
        raise ValueError("Unsupported LiDAR extension")


# -------- utils --------

def _normalize_exts(exts: Optional[Iterable[str]]) -> Optional[set[str]]:
    if exts is None:
        return None
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}


def _to_list_vec(vec: np.ndarray) -> List[float]:
    """Ensure (D,) list regardless of (D,) or (1,D)."""
    v = np.asarray(vec)
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    return v.astype("float32").tolist()


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
) -> int:
    """Scan s3://bucket/prefix and write rows into `table_name`.
    Returns number of rows processed (planned if dry_run=True)."""

    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()

    print("Listing objects in bucket...")
    keys = s3.list_objects(bucket=bucket, prefix=prefix or "")
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return 0

    incl = _normalize_exts(include_ext) or DEFAULT_EXTS
    excl = _normalize_exts(exclude_ext) or set()

    filtered = []
    for key in keys:
        _, ext = os.path.splitext(key)
        ext = ext.lower()
        if ext in excl:
            continue
        if ext in incl:
            filtered.append(key)

    if max_files is not None and max_files > 0:
        filtered = filtered[:max_files]
    
    total_to_process = len(filtered)
    print(f"Found {total_to_process} files to process.")

    if dry_run:
        by_ext: Dict[str, int] = {}
        for k in filtered:
            _, e = os.path.splitext(k)
            by_ext[e.lower()] = by_ext.get(e.lower(), 0) + 1
        print(f"[DRY-RUN] Would ingest {len(filtered)} objects into table '{table_name}' from s3://{bucket}/{prefix}")
        if by_ext:
            print("[DRY-RUN] By extension:", ", ".join(f"{k}:{v}" for k, v in sorted(by_ext.items())))
        print("[DRY-RUN] Examples:")
        for k in filtered[:10]:
            print(f"  - {k}")
        return len(filtered)
    
    all_rows: List[Dict] = []
    
    # Process all keys in master batches to conserve memory
    num_batches = (len(filtered) + write_chunk - 1) // write_chunk
    for i, master_batch_keys in enumerate(_batch(filtered, write_chunk)):
        print(f"\n--- Processing master batch {i+1} of {num_batches} ({len(master_batch_keys)} files) ---")
        
        # Download all blobs for this master batch at once
        blobs = _download_many(s3, bucket, master_batch_keys, workers)
        
        # Separate keys by modality for the current batch
        img_keys = [k for k in master_batch_keys if os.path.splitext(k)[1].lower() in IMAGE_EXTS]
        pdf_keys = [k for k in master_batch_keys if os.path.splitext(k)[1].lower() in PDF_EXTS]
        lidar_keys = [k for k in master_batch_keys if os.path.splitext(k)[1].lower() in LIDAR_EXTS]

        # ---- IMAGES (parallel dl + batched embed) ----
        if img_keys:
            img_processed_count = 0
            good_keys = [k for k in img_keys if isinstance(blobs.get(k), (bytes, bytearray))]
            for chunk_keys in _batch(good_keys, embed_batch):
                imgs = [blobs[k] for k in chunk_keys]
                try:
                    vecs = img_embedder.embed(imgs)  # (B, D)
                except Exception:
                    # fallback per-item to avoid losing the whole chunk
                    vecs = np.vstack([np.asarray(img_embedder.embed(x)).reshape(1, -1) for x in imgs])
                for k, v in zip(chunk_keys, vecs):
                    all_rows.append({
                        "id": k, "modality": "image", "embedding": _to_list_vec(v),
                        "text": None, "image": None, "lidar": None, "path": f"s3://{bucket}/{k}",
                    })
                img_processed_count += len(chunk_keys)
                print(f"  Image batch processed: {img_processed_count}/{len(img_keys)} images in this master batch.")

        # ---- PDFs (parallel dl + batched text embed) ----
        if pdf_keys:
            pdf_processed_count = 0
            pairs: List[tuple[str, str]] = []
            for k in pdf_keys:
                b = blobs.get(k)
                if isinstance(b, (bytes, bytearray)):
                    text = _extract_pdf_text(b, max_pages=pdf_max_pages)
                    pairs.append((k, text if text.strip() else "Empty PDF document"))
            for chunk in _batch(pairs, embed_batch):
                texts = [t for _, t in chunk]
                try:
                    vecs = text_embedder.embed(texts)  # (B, D)
                except Exception:
                    vecs = np.vstack([np.asarray(text_embedder.embed(t)).reshape(1, -1) for t in texts])
                for (k, text), v in zip(chunk, vecs):
                    all_rows.append({
                        "id": k, "modality": "pdf", "embedding": _to_list_vec(v),
                        "text": text, "image": None, "lidar": None, "path": f"s3://{bucket}/{k}",
                    })
                pdf_processed_count += len(chunk)
                print(f"  PDF batch processed: {pdf_processed_count}/{len(pdf_keys)} PDFs in this master batch.")

        # ---- LiDAR (parallel dl + small-batch embed loop) ----
        if lidar_keys:
            lidar_processed_count = 0
            small_batch = max(1, embed_batch // 8)  # e.g., 8 for embed_batch=64
            key_list = [k for k in lidar_keys if isinstance(blobs.get(k), (bytes, bytearray))]
            for chunk_keys in _batch(key_list, small_batch):
                pts_list = []
                valid = []
                for k in chunk_keys:
                    ext = os.path.splitext(k)[1].lower()
                    try:
                        pts = _load_lidar_bytes(ext, blobs[k])
                        if pts.size > 0:
                            pts_list.append(pts)
                            valid.append(k)
                    except Exception:
                        continue
                # Assuming lidar_embedder.embed does not support batching natively,
                # we embed one-by-one; if it does, replace with vecs = lidar_embedder.embed(pts_list)
                for k, pts in zip(valid, pts_list):
                    try:
                        v = lidar_embedder.embed(pts)
                        all_rows.append({
                            "id": k, "modality": "lidar", "embedding": _to_list_vec(v),
                            "text": None, "image": None, "lidar": None, "path": f"s3://{bucket}/{k}",
                        })
                        lidar_processed_count += 1
                    except Exception:
                        continue  # Skip failed embeddings
            print(f"  LiDAR processed: {lidar_processed_count}/{len(lidar_keys)} files in this master batch.")
        
        print(f"  Master batch {i+1} complete. Total items so far: {len(all_rows)}/{total_to_process}")


    if not all_rows:
        print("No processable objects found after filtering.")
        return 0

    print("\nDeduplicating and preparing data for database...")
    df = DataFrame(all_rows)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    rows = df.to_dict("records")

    print(f"\nWriting {len(rows)} rows to LanceDB...")
    # Write to LanceDB in chunks (fewer transactions)
    if mode == "append":
        try:
            tbl = manager.get_table(table_name)
            ids = [r["id"] for r in rows]
            CHUNK_DEL = 500
            for i in range(0, len(ids), CHUNK_DEL):
                chunk = ids[i:i + CHUNK_DEL]
                escaped = [("'" + x.replace("'", "''") + "'") for x in chunk]
                tbl.delete(f"id IN ({','.join(escaped)})")
            for i in range(0, len(rows), write_chunk):
                tbl.add(rows[i:i + write_chunk])
        except Exception:
            manager.create_table(table_name, rows[:write_chunk], mode="overwrite")
            tbl = manager.get_table(table_name)
            for i in range(write_chunk, len(rows), write_chunk):
                tbl.add(rows[i:i + write_chunk])
    else:
        # overwrite: 1st chunk creates, rest append
        manager.create_table(table_name, rows[:write_chunk], mode="overwrite")
        tbl = manager.get_table(table_name)
        for i in range(write_chunk, len(rows), write_chunk):
            tbl.add(rows[i:i + write_chunk])
            
    duration_s = round(time.perf_counter() - t0, 3)
    finished_at = dt.datetime.now(dt.timezone.utc)

    log_rec = {
        "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{table_name}",
        "table": table_name,
        "bucket": bucket,
        "prefix": prefix,
        "rows": len(rows),
        "mode": mode,
        "include_ext": sorted(list(include_ext)) if include_ext else None,
        "exclude_ext": sorted(list(exclude_ext)) if exclude_ext else None,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_s": duration_s,
        "host": socket.gethostname(),
    }

    print(f"\n[BUILD COMPLETE] table='{table_name}' rows={len(rows)} mode={mode} total_duration={duration_s}s")

    try:
        with open("build_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec) + "\n")
    except Exception: pass

    try:
        meta_tbl = None
        try: meta_tbl = manager.get_table("__build_log")
        except Exception:
            manager.create_table("__build_log", [log_rec], mode="overwrite")
            meta_tbl = manager.get_table("__build_log")
        if meta_tbl is not None:
            rid = log_rec["run_id"].replace("'", "''")
            meta_tbl.delete(f"run_id = '{rid}'")
            meta_tbl.add([log_rec])
    except Exception: pass

    print(f"\nSuccessfully ingested {len(rows)} objects into table '{table_name}' from s3://{bucket}/{prefix}")
    return len(rows)