"""Purpose: orchestrate multimodal ingestion from object storage into LanceDB using modular helpers.
Why extend: introduce new batching strategies, logging hooks, or modality handlers without rewriting the CLI.
How extend: modify `_process_one_batch` to recognise new extensions or add keyword arguments to `ingest_s3_objects` that wire through to the helper functions.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.embedding_runtime import get_image_embedder, get_lidar_embedder, get_text_embedder
from src.ingest.constants import (
    DEFAULT_EMBED_BATCH,
    DEFAULT_EXTS,
    DEFAULT_PDF_MAX_PAGES,
    DEFAULT_WORKERS,
    DEFAULT_WRITE_CHUNK,
    IMAGE_EXTS,
    INGEST_PROGRESS_EVERY,
    LIDAR_BATCH_DIVISOR,
    LIDAR_EMBED_BATCH,
    LIDAR_EXTS,
    PDF_EXTS,
    TEXT_EXTS,
    normalize_exts,
)
from src.ingest.io import download_many
from src.ingest.lidar import load_lidar_bytes
from src.ingest.text_pdf import decode_text_bytes, extract_pdf_text
from src.ingest.utils import batched, s3_path, to_list_vec
from src.ingest.writer import UpsertWriter
from src.storage import MinIOClient
from src.vectordb import LanceDBManager


def _effective_progress_every(progress_every: Optional[int], fallback_batch: int) -> int:
    base = INGEST_PROGRESS_EVERY if progress_every is None else int(progress_every)
    return base or fallback_batch


def _effective_lidar_batch(
    embed_batch: int,
    override: Optional[int] = None,
    divisor: Optional[int] = None,
) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    div = int(divisor) if divisor else LIDAR_BATCH_DIVISOR
    div = max(1, div)
    return max(1, embed_batch // div)


def _log_build(record: dict) -> None:
    try:
        with open("build_logs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _maybe_disable_tls_warnings() -> None:
    try:
        mv = str(os.getenv("MINIO_VERIFY", "")).strip().lower()
        sup = str(os.getenv("MINIO_SUPPRESS_TLS_WARN", "")).strip().lower()
        if sup in ("1", "true", "yes") or mv in ("0", "false", "no"):
            import urllib3  # type: ignore

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def _filtered_key_iter(
    s3: MinIOClient,
    bucket: str,
    prefix: str,
    include: set[str],
    exclude: set[str],
    max_files: Optional[int],
) -> Iterable[str]:
    count = 0
    for key in s3.iter_objects(bucket=bucket, prefix=prefix or ""):
        ext = os.path.splitext(key)[1].lower()
        if ext in exclude:
            continue
        if ext in include:
            yield key
            count += 1
            if max_files and count >= max_files:
                return


def _process_one_batch(
    bidx: int,
    master_keys: List[str],
    blobs: Dict[str, bytes | Exception],
    manager: LanceDBManager,
    writer: UpsertWriter,
    bucket: str,
    table_name: str,
    *,
    mode: str,
    embed_batch: int,
    pdf_max_pages: int,
    text_embedder,
    image_embedder,
    lidar_embedder,
    progress_every: Optional[int],
    lidar_batch_size: int,
) -> int:
    img_keys: List[str] = []
    pdf_keys: List[str] = []
    text_keys: List[str] = []
    lidar_keys: List[str] = []
    for key in master_keys:
        ext = os.path.splitext(key)[1].lower()
        if ext in IMAGE_EXTS:
            img_keys.append(key)
        elif ext in PDF_EXTS:
            pdf_keys.append(key)
        elif ext in TEXT_EXTS:
            text_keys.append(key)
        elif ext in LIDAR_EXTS:
            lidar_keys.append(key)

    rows: List[dict] = []
    seen_ids: set[str] = set()

    if img_keys:
        good = [k for k in img_keys if isinstance(blobs.get(k), (bytes, bytearray))]
        processed = 0
        for chunk_keys in batched(good, embed_batch):
            batch_blobs = [blobs[k] for k in chunk_keys]
            try:
                vecs = image_embedder.embed(batch_blobs)
            except Exception:
                vecs = [image_embedder.embed(b) for b in batch_blobs]
            for key, vec in zip(chunk_keys, vecs):
                if key in seen_ids:
                    continue
                rows.append(
                    {
                        "id": key,
                        "modality": "image",
                        "embedding": to_list_vec(vec),
                        "text": None,
                        "image": None,
                        "lidar": None,
                        "path": s3_path(bucket, key),
                    }
                )
                seen_ids.add(key)
            processed += len(chunk_keys)
            prog_every = _effective_progress_every(progress_every, embed_batch)
            if processed % max(1, prog_every) == 0 or processed >= len(good):
                print(f"  images: {processed}/{len(good)}")

    if text_keys:
        pairs: List[Tuple[str, str]] = []
        for key in text_keys:
            blob = blobs.get(key)
            if isinstance(blob, (bytes, bytearray)):
                pairs.append((key, decode_text_bytes(blob)))
        processed = 0
        for chunk in batched(pairs, embed_batch):
            texts = [t for _, t in chunk]
            try:
                vecs = text_embedder.embed(texts)
            except Exception:
                vecs = [text_embedder.embed(t) for t in texts]
            for (key, text_value), vec in zip(chunk, vecs):
                if key in seen_ids:
                    continue
                rows.append(
                    {
                        "id": key,
                        "modality": "text",
                        "embedding": to_list_vec(vec),
                        "text": text_value,
                        "image": None,
                        "lidar": None,
                        "path": s3_path(bucket, key),
                    }
                )
                seen_ids.add(key)
            processed += len(chunk)
            prog_every = _effective_progress_every(progress_every, embed_batch)
            if processed % max(1, prog_every) == 0 or processed >= len(pairs):
                print(f"  text: {processed}/{len(pairs)}")

    if pdf_keys:
        pairs = []
        for key in pdf_keys:
            blob = blobs.get(key)
            if isinstance(blob, (bytes, bytearray)):
                text_value = extract_pdf_text(blob, max_pages=pdf_max_pages)
                pairs.append((key, text_value if text_value.strip() else ""))
        processed = 0
        for chunk in batched(pairs, embed_batch):
            texts = [t for _, t in chunk]
            try:
                vecs = text_embedder.embed(texts)
            except Exception:
                vecs = [text_embedder.embed(t) for t in texts]
            for (key, text_value), vec in zip(chunk, vecs):
                if key in seen_ids:
                    continue
                rows.append(
                    {
                        "id": key,
                        "modality": "pdf",
                        "embedding": to_list_vec(vec),
                        "text": text_value or "Empty PDF document",
                        "image": None,
                        "lidar": None,
                        "path": s3_path(bucket, key),
                    }
                )
                seen_ids.add(key)
            processed += len(chunk)
            prog_every = _effective_progress_every(progress_every, embed_batch)
            if processed % max(1, prog_every) == 0 or processed >= len(pairs):
                print(f"  pdfs: {processed}/{len(pairs)}")

    if lidar_keys:
        good_lidar = [k for k in lidar_keys if isinstance(blobs.get(k), (bytes, bytearray))]
        processed = 0
        for chunk_keys in batched(good_lidar, lidar_batch_size):
            clouds: List[np.ndarray] = []
            valid_keys: List[str] = []
            for key in chunk_keys:
                ext = os.path.splitext(key)[1].lower()
                try:
                    pts = load_lidar_bytes(ext, blobs[key])
                    if pts.size > 0:
                        clouds.append(pts)
                        valid_keys.append(key)
                except Exception:
                    continue
            if not valid_keys:
                continue
            try:
                if hasattr(lidar_embedder, "embed_pointclouds"):
                    vecs = lidar_embedder.embed_pointclouds(clouds)
                else:
                    vecs = [lidar_embedder.embed(points) for points in clouds]
            except Exception:
                vecs = [lidar_embedder.embed(points) for points in clouds]
            for key, vec in zip(valid_keys, vecs):
                if key in seen_ids:
                    continue
                rows.append(
                    {
                        "id": key,
                        "modality": "lidar",
                        "embedding": to_list_vec(vec),
                        "text": None,
                        "image": None,
                        "lidar": None,
                        "path": s3_path(bucket, key),
                    }
                )
                seen_ids.add(key)
            processed += len(valid_keys)
            prog_every = _effective_progress_every(progress_every, lidar_batch_size)
            if processed % max(1, prog_every) == 0 or processed >= len(good_lidar):
                print(f"  lidar: {processed}/{len(good_lidar)}")

    if not rows:
        print("  (no rows in this batch)")
        return 0

    if bidx == 1:
        writer.ensure_table(table_name, rows, mode=mode)
    written = writer.upsert_rows(table_name, rows)
    print(f"  wrote {written} rows (stream)")
    return written


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
    mode: str = "overwrite",
    *,
    workers: int = DEFAULT_WORKERS,
    embed_batch: int = DEFAULT_EMBED_BATCH,
    write_chunk: int = DEFAULT_WRITE_CHUNK,
    pdf_max_pages: int = DEFAULT_PDF_MAX_PAGES,
    text_embedder=None,
    image_embedder=None,
    lidar_embedder=None,
    stream_list: bool = False,
    progress_every: Optional[int] = None,
    lidar_embed_batch: Optional[int] = None,
    lidar_batch_divisor: Optional[int] = None,
) -> int:
    text_embedder = text_embedder or get_text_embedder()
    image_embedder = image_embedder or get_image_embedder()
    lidar_embedder = lidar_embedder or get_lidar_embedder()

    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()
    _maybe_disable_tls_warnings()

    include = normalize_exts(include_ext) or DEFAULT_EXTS
    exclude = normalize_exts(exclude_ext) or set()

    override = lidar_embed_batch
    if override is None and LIDAR_EMBED_BATCH > 0:
        override = LIDAR_EMBED_BATCH
    lidar_batch_size = _effective_lidar_batch(embed_batch, override, lidar_batch_divisor)

    if stream_list:
        filtered_iter = _filtered_key_iter(s3, bucket, prefix, include, exclude, max_files)
        if dry_run:
            total = 0
            sample: List[str] = []
            for key in filtered_iter:
                total += 1
                if len(sample) < 10:
                    sample.append(key)
            print(f"[DRY-RUN] Would ingest {total} objects into '{table_name}' from s3://{bucket}/{prefix}")
            for key in sample:
                print("  -", key)
            return total

        writer = UpsertWriter(manager)
        total_written = 0
        batch_index = 0
        pending: List[str] = []
        for key in filtered_iter:
            pending.append(key)
            if len(pending) >= write_chunk:
                batch_index += 1
                master_keys = pending
                pending = []
                print(f"\n--- Batch {batch_index}/? ({len(master_keys)} files) ---")
                blobs = download_many(s3, bucket, master_keys, workers)
                total_written += _process_one_batch(
                    batch_index,
                    master_keys,
                    blobs,
                    manager,
                    writer,
                    bucket,
                    table_name,
                    mode=mode,
                    embed_batch=embed_batch,
                    pdf_max_pages=pdf_max_pages,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                    lidar_embedder=lidar_embedder,
                    progress_every=progress_every,
                    lidar_batch_size=lidar_batch_size,
                )
        if pending:
            batch_index += 1
            master_keys = pending
            print(f"\n--- Batch {batch_index}/? ({len(master_keys)} files) ---")
            blobs = download_many(s3, bucket, master_keys, workers)
            total_written += _process_one_batch(
                batch_index,
                master_keys,
                blobs,
                manager,
                writer,
                bucket,
                table_name,
                mode=mode,
                embed_batch=embed_batch,
                pdf_max_pages=pdf_max_pages,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
                lidar_embedder=lidar_embedder,
                progress_every=progress_every,
                lidar_batch_size=lidar_batch_size,
            )

        duration_s = round(time.perf_counter() - t0, 3)
        log_rec = {
            "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{table_name}",
            "table": table_name,
            "bucket": bucket,
            "prefix": prefix,
            "rows": total_written,
            "mode": mode,
            "include_ext": sorted(list(include)) if include else None,
            "exclude_ext": sorted(list(exclude)) if exclude else None,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_s": duration_s,
            "host": socket.gethostname(),
        }
        print(f"\n[BUILD COMPLETE] table='{table_name}' rows={total_written} mode={mode} duration={duration_s}s")
        _log_build(log_rec)
        try:
            meta_tbl = manager.get_table("__build_log")
        except Exception:
            manager.create_table("__build_log", [log_rec], mode="overwrite")
            meta_tbl = manager.get_table("__build_log")
        try:
            if meta_tbl is not None:
                rid = log_rec["run_id"].replace("'", "''")
                meta_tbl.delete(f"run_id = '{rid}'")
                meta_tbl.add([log_rec])
        except Exception:
            pass
        return total_written

    keys = s3.list_objects(bucket=bucket, prefix=prefix or "")
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return 0

    filtered: List[str] = []
    for key in keys:
        ext = os.path.splitext(key)[1].lower()
        if ext in exclude:
            continue
        if ext in include:
            filtered.append(key)
    if max_files and max_files > 0:
        filtered = filtered[:max_files]

    if dry_run:
        by_ext: Dict[str, int] = {}
        for key in filtered:
            ext = os.path.splitext(key)[1].lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
        print(f"[DRY-RUN] Would ingest {len(filtered)} objects into '{table_name}' from s3://{bucket}/{prefix}")
        if by_ext:
            print("[DRY-RUN] By extension:", ", ".join(f"{k}:{v}" for k, v in sorted(by_ext.items())))
        for key in filtered[:10]:
            print("  -", key)
        return len(filtered)

    writer = UpsertWriter(manager)
    total_written = 0
    num_batches = (len(filtered) + write_chunk - 1) // write_chunk

    for batch_index, master_keys in enumerate(batched(filtered, write_chunk), start=1):
        print(f"\n--- Batch {batch_index}/{num_batches} ({len(master_keys)} files) ---")
        blobs = download_many(s3, bucket, master_keys, workers)
        total_written += _process_one_batch(
            batch_index,
            master_keys,
            blobs,
            manager,
            writer,
            bucket,
            table_name,
            mode=mode,
            embed_batch=embed_batch,
            pdf_max_pages=pdf_max_pages,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            lidar_embedder=lidar_embedder,
            progress_every=progress_every,
            lidar_batch_size=lidar_batch_size,
        )

    duration_s = round(time.perf_counter() - t0, 3)
    finished_at = dt.datetime.now(dt.timezone.utc)
    log_rec = {
        "run_id": f"{started_at.strftime('%Y%m%dT%H%M%S')}-{table_name}",
        "table": table_name,
        "bucket": bucket,
        "prefix": prefix,
        "rows": total_written,
        "mode": mode,
        "include_ext": sorted(list(include)) if include else None,
        "exclude_ext": sorted(list(exclude)) if exclude else None,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_s": duration_s,
        "host": socket.gethostname(),
    }

    print(f"\n[BUILD COMPLETE] table='{table_name}' rows={total_written} mode={mode} duration={duration_s}s")
    _log_build(log_rec)

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


__all__ = ["ingest_s3_objects"]
