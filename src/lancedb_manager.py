"""Purpose: compatibility shim exposing the LanceDB manager for legacy imports and monkeypatch-friendly tests.
Why extend: maintain backwards compatibility while the real implementation evolves in `src.vectordb.lance`.
How extend: re-export new helpers or adapters from the vectordb package, e.g. `from src.vectordb.lance import LanceDBManager, LanceIndexTuner` and list them in `__all__`.
"""

from __future__ import annotations

from src.storage import MinIOClient
from src.vectordb.lance import LanceDBManager

__all__ = ["LanceDBManager", "MinIOClient"]
