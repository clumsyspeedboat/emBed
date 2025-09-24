"""Purpose: surface the ingestion pipeline and maintain legacy helpers under a stable import path.
Why extend: add new ingest entry points (e.g. local filesystem, message queues) while keeping existing consumers untouched.
How extend: import the new functions/classes here and list them in `__all__`, optionally mirroring old names for compatibility.
"""

from __future__ import annotations

from src.ingest.constants import (
    DEFAULT_EXTS,
    IMAGE_EXTS,
    LIDAR_EXTS,
    PDF_EXTS,
    TEXT_EXTS,
)
from src.ingest.lidar import load_lidar_bytes
from src.ingest.pipeline import ingest_s3_objects

_load_lidar_bytes = load_lidar_bytes

__all__ = [
    "ingest_s3_objects",
    "TEXT_EXTS",
    "IMAGE_EXTS",
    "PDF_EXTS",
    "LIDAR_EXTS",
    "DEFAULT_EXTS",
    "load_lidar_bytes",
    "_load_lidar_bytes",
]
