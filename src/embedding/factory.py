"""Purpose: expose a simple `get_embedding_model` helper keyed by modality.
Why extend: register new modalities or custom embedder variants.
How extend: add another `elif` branch returning the appropriate embedder instance, and update callers/tests accordingly.
"""
from __future__ import annotations

from typing import Optional

from src.embedding.image import ImageEmbedder
from src.embedding.lidar import LidarBEVEmbedder
from src.embedding.text import TextEmbedder

__all__ = ["get_embedding_model"]


def get_embedding_model(model_type: str, model_name: Optional[str] = None):
    if model_type == "text":
        return TextEmbedder(model_name)
    if model_type == "image":
        return ImageEmbedder(model_name)
    if model_type == "lidar":
        return LidarBEVEmbedder(ImageEmbedder(model_name))
    raise ValueError(f"Unknown model type: {model_type}")
