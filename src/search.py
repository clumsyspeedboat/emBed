"""Purpose: compatibility shim exposing `MultiModalSearcher` at the legacy module path.
Why extend: keep external callers working while core search logic lives in `src.core.search`.
How extend: add new exports in `src.core.search` and re-import them here to surface additional helpers without breaking old imports.
"""
from __future__ import annotations

from src.core.search import MultiModalSearcher

__all__ = ["MultiModalSearcher"]
