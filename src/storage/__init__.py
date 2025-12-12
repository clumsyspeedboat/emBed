"""Purpose: expose storage client implementations via a tidy import surface.
Why extend: add clients for other object stores or caching layers.
How extend: import the new client class here and include it in `__all__`, then update callers to select the appropriate implementation.
"""
from __future__ import annotations

from src.storage.minio_client import MinIOClient

__all__ = ["MinIOClient"]
