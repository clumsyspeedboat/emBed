"""Purpose: compatibility shim for projects that still import `VectorDBManager` from the legacy module.
Why extend: keep re-exporting new abstract manager interfaces introduced in `src.vectordb`.
How extend: import the new class and include it in `__all__`, mirroring any renames made in the vectordb package.
"""

from __future__ import annotations

from src.vectordb.base import VectorDBManager

__all__ = ["VectorDBManager"]
