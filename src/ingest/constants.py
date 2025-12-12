"""Purpose: centralise ingest-related constants and environment knobs used across the pipeline.
Why extend: add tunables for new modalities or throttling options without scattering env lookups.
How extend: define additional constants/functions here (e.g. `DEFAULT_AUDIO_EXTS`) and import them from pipeline components that need the new behaviour.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Set

TEXT_EXTS: Set[str] = {".txt", ".md"}
IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
PDF_EXTS: Set[str] = {".pdf"}
LIDAR_EXTS: Set[str] = {".pcd", ".bin"}
DEFAULT_EXTS: Set[str] = TEXT_EXTS | IMAGE_EXTS | PDF_EXTS | LIDAR_EXTS

PDF_TEXT_CHAR_LIMIT = 50_000


def get_int_env(name: str, default: int) -> int:
    """Safe int(env) helper with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def normalize_exts(exts: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if exts is None:
        return None
    norm: Set[str] = set()
    for ext in exts:
        e = ext.lower()
        norm.add(e if e.startswith(".") else f".{e}")
    return norm

from src.config.ingest import IngestConfig

INGEST_CONFIG = IngestConfig()
DEFAULT_WORKERS = INGEST_CONFIG.workers
DEFAULT_EMBED_BATCH = INGEST_CONFIG.embed_batch
DEFAULT_WRITE_CHUNK = INGEST_CONFIG.write_chunk
LIDAR_BATCH_DIVISOR = INGEST_CONFIG.lidar_batch_divisor
LIDAR_EMBED_BATCH = INGEST_CONFIG.lidar_embed_batch
DEFAULT_PDF_MAX_PAGES = INGEST_CONFIG.pdf_max_pages
INGEST_PROGRESS_EVERY = INGEST_CONFIG.progress_every
