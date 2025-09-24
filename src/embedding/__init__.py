"""Purpose: expose embedding classes and factory helpers via a single import.
Why extend: streamline access to new embedder implementations added under this package.
How extend: import the new classes here and add them to `__all__`, keeping default factory wiring consistent with `embedding_runtime`.
"""
from __future__ import annotations

from src.embedding.image import ImageEmbedder
from src.embedding.lidar import LidarBEVEmbedder
from src.embedding.text import TextEmbedder

from src.embedding.base import l2_normalize as _l2_normalize, l2_normalize, pick_device
from src.embedding.factory import get_embedding_model

__all__ = [
    "TextEmbedder",
    "ImageEmbedder",
    "LidarBEVEmbedder",
    "get_embedding_model",
    "l2_normalize",
    "pick_device",
    "_l2_normalize",
]
