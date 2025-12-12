"""Purpose: factory for runtime-selectable embedding backends (real vs. dummy).
Why extend: support new model families or caching strategies without touching callers.
How extend: add loader functions and branch in `_backend()` checks, for example wiring in an OpenAI API embedder guarded by `EMBEDDING_BACKEND=azure`.
"""

from __future__ import annotations

from src.embedding.runtime import (
    DummyImageEmbedder,
    DummyLidarEmbedder,
    DummyTextEmbedder,
    ImageBackend,
    ImageEmbedder,
    LidarEmbedder,
    LidarBEVEmbedder,
    TextBackend,
    TextEmbedder,
    get_image_embedder,
    get_lidar_embedder,
    get_text_embedder,
    reset_singletons_for_tests,
)

__all__ = [
    "TextEmbedder",
    "ImageEmbedder",
    "LidarBEVEmbedder",
    "TextBackend",
    "ImageBackend",
    "LidarEmbedder",
    "DummyTextEmbedder",
    "DummyImageEmbedder",
    "DummyLidarEmbedder",
    "get_text_embedder",
    "get_image_embedder",
    "get_lidar_embedder",
    "reset_singletons_for_tests",
]
