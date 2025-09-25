"""Purpose: expose search utilities and the LanceDB searcher via a stable package path.
Why extend: add new helpers (e.g. cache-aware services) without forcing callers to change imports.
How extend: append exports to `__all__` once helpers in sibling modules are ready for public use.
"""
from __future__ import annotations

from src.core.search import MultiModalSearcher
from .service import (
    SearchInputs,
    SearchService,
    apply_metric_threshold,
    load_lidar_points,
    normalize_modalities,
    vectorize_modalities,
)

__all__ = [
    "MultiModalSearcher",
    "SearchInputs",
    "SearchService",
    "normalize_modalities",
    "vectorize_modalities",
    "apply_metric_threshold",
    "load_lidar_points",
]
