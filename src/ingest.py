"""Ingest images and PDFs from MinIO into a LanceDB table.

Features:
- Dry-run (preview what would be ingested)
- Include/exclude extensions
- Max files limit
- Append or overwrite modes
"""

from __future__ import annotations

import io
import os
import warnings
from typing import List, Dict, Iterable, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .storage import MinioClient
from .embedding import DummyTextEmbedder, DummyImageEmbedder
from .lancedb_manager import LanceDBManager

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
PDF_EXTS = {".pdf"}
DEFAULT_EXTS = IMAGE_EXTS | PDF_EXTS

PDF_TEXT_CHAR_LIMIT = 50_000


def _extract_pdf_text(blob: bytes) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reader = PdfReader(io.BytesIO(blob))
        except (PdfReadError, Exception):
            return ""
    parts: List[str] = []
    for page in reader.pages:
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


def _normalize_exts(exts: Optional[Iterable[str]]) -> Optional[set[str]]:
    if exts is None:
        return None
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}


def ingest_s3_objects(
    manager: LanceDBManager,
    s3: MinioClient,
    bucket: str,
    prefix: str,
    table_name: str,
    include_ext: Optional[Iterable[str]] = None,
    exclude_ext: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    dry_run: bool = False,
    mode: str = "overwrite",  # or "append"
) -> int:
    """Scan s3://bucket/prefix and write rows into `table_name`.
    Returns number of rows processed (planned if dry_run=True)."""

    keys = s3.list_objects(bucket, prefix or "")
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return 0

    incl = _normalize_exts(include_ext) or DEFAULT_EXTS
    excl = _normalize_exts(exclude_ext) or set()

    # Filter by extension
    filtered: List[str] = []
    for key in keys:
        _, ext = os.path.splitext(key)
        ext = ext.lower()
        if ext in excl:
            continue
        if ext in incl:
            filtered.append(key)

    if max_files is not None and max_files > 0:
        filtered = filtered[:max_files]

    if dry_run:
        # Preview summary
        by_ext: Dict[str, int] = {}
        for k in filtered:
            _, e = os.path.splitext(k)
            by_ext[e.lower()] = by_ext.get(e.lower(), 0) + 1
        print(f"[DRY-RUN] Would ingest {len(filtered)} objects into table '{table_name}' "
              f"from s3://{bucket}/{prefix}")
        if by_ext:
            print("[DRY-RUN] By extension:", ", ".join(f"{k}:{v}" for k, v in sorted(by_ext.items())))
        print("[DRY-RUN] Examples:")
        for k in filtered[:10]:
            print(f"  - {k}")
        return len(filtered)

    # Real ingestion
    rows: List[Dict] = []
    text_embedder = DummyTextEmbedder()
    img_embedder = DummyImageEmbedder()

    for key in filtered:
        _, ext = os.path.splitext(key)
        ext = ext.lower()

        if ext in IMAGE_EXTS:
            try:
                data = s3.get_object_bytes(bucket, key)
                emb = img_embedder.embed(data).tolist()
                rows.append({
                    "id": key,
                    "modality": "image",
                    "embedding": emb,
                    "text": None,
                    "image": None,
                    "lidar": None,
                    "path": f"s3://{bucket}/{key}",
                })
            except Exception as e:
                print(f"Skipping image {key}: {e}")

        elif ext in PDF_EXTS:
            try:
                data = s3.get_object_bytes(bucket, key)
                text = _extract_pdf_text(data)
                text_for_embedding = text if text.strip() else key
                emb = text_embedder.embed(text_for_embedding).tolist()
                rows.append({
                    "id": key,
                    "modality": "pdf",
                    "embedding": emb,
                    "text": text,
                    "image": None,
                    "lidar": None,
                    "path": f"s3://{bucket}/{key}",
                })
            except Exception as e:
                print(f"Skipping pdf {key}: {e}")

    if not rows:
        print(f"No supported objects (after filters) under s3://{bucket}/{prefix}")
        return 0

    if mode == "append":
        try:
            tbl = manager.get_table(table_name)
            tbl.add(rows)
        except Exception:
            manager.create_table_with_data(table_name, rows, mode="overwrite")
    else:
        manager.create_table_with_data(table_name, rows, mode="overwrite")

    print(f"Ingested {len(rows)} objects into table '{table_name}' from s3://{bucket}/{prefix}")
    return len(rows)
