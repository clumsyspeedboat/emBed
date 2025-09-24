"""Purpose: encapsulate ingestion tunables derived from environment variables.
Why extend: introduce new batching or filtering knobs without editing multiple ingestion modules.
How extend: add dataclass fields (e.g. `streaming_chunk_size`) and have the pipeline consume them when deriving defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

from src.config._env import ensure_dotenv_loaded

ensure_dotenv_loaded()


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass
class IngestConfig:
    workers: int = field(default_factory=lambda: _get_int_env("INGEST_WORKERS", 16))
    embed_batch: int = field(default_factory=lambda: _get_int_env("INGEST_EMBED_BATCH", 64))
    write_chunk: int = field(default_factory=lambda: _get_int_env("INGEST_WRITE_CHUNK", 5000))
    pdf_max_pages: int = field(default_factory=lambda: _get_int_env("INGEST_PDF_MAX_PAGES", 8))
    progress_every: int = field(default_factory=lambda: _get_int_env("INGEST_PROGRESS_EVERY", 0))
    lidar_batch_divisor: int = field(default_factory=lambda: max(1, _get_int_env("LIDAR_BATCH_DIVISOR", 8)))
    lidar_embed_batch: int = field(default_factory=lambda: _get_int_env("LIDAR_EMBED_BATCH", 0))


__all__ = ["IngestConfig"]
