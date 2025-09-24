"""Runtime factory for embedding models with optional dummy backends."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol, cast

import numpy as np

from src.embedding import ImageEmbedder, LidarBEVEmbedder, TextEmbedder

__all__ = [
    "TextEmbedder",
    "ImageEmbedder",
    "LidarEmbedder",
    "DummyTextEmbedder",
    "DummyImageEmbedder",
    "DummyLidarEmbedder",
    "get_text_embedder",
    "get_image_embedder",
    "get_lidar_embedder",
    "reset_singletons_for_tests",
]


class TextBackend(Protocol):
    dim: int

    def embed(self, texts: Iterable[str]) -> np.ndarray: ...


class ImageBackend(Protocol):
    dim: int

    def embed(self, images: Iterable[bytes]) -> np.ndarray: ...


class LidarEmbedder(Protocol):
    dim: int

    def embed(self, clouds: Iterable[np.ndarray]) -> np.ndarray: ...


def _hash_to_unit_vec(data: bytes, dim: int) -> np.ndarray:
    """Convert arbitrary bytes into a deterministic unit vector."""
    digest = hashlib.sha256(data).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    return vec / (np.linalg.norm(vec) + 1e-9)


@dataclass
class DummyTextEmbedder:
    dim: int = 512

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([_hash_to_unit_vec(text.encode("utf-8"), self.dim) for text in texts])


@dataclass
class DummyImageEmbedder:
    dim: int = 512

    def embed(self, images: Iterable[bytes]) -> np.ndarray:
        return np.vstack([_hash_to_unit_vec(blob, self.dim) for blob in images])


@dataclass
class DummyLidarEmbedder:
    dim: int = 512

    def embed(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        rows = []
        for cloud in clouds:
            arr = np.asarray(cloud)
            arr32 = arr.astype(np.float32, copy=False).ravel()
            rows.append(_hash_to_unit_vec(arr32.tobytes(), self.dim))
        return np.vstack(rows)

    def embed_pointclouds(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        return self.embed(clouds)


_real_text: Optional[Any] = None
_real_image: Optional[Any] = None
_real_lidar: Optional["_RealLidarAdapter"] = None


def _load_real_text() -> Any:
    global _real_text
    if _real_text is None:
        _real_text = TextEmbedder()
    return _real_text


def _load_real_image() -> Any:
    global _real_image
    if _real_image is None:
        _real_image = ImageEmbedder()
    return _real_image


class _RealLidarAdapter:
    """Adapter to unify ``LidarBEVEmbedder`` under the ``LidarEmbedder`` protocol."""

    def __init__(self, real_lidar_obj: LidarBEVEmbedder) -> None:
        self._lidar = real_lidar_obj
        dummy = np.zeros((2, 3), dtype=np.float32)
        vec = self._lidar.embed(dummy)
        self.dim = int(vec.shape[-1])

    def embed_pointclouds(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        outs = []
        for cloud in clouds:
            outs.append(self._lidar.embed(np.asarray(cloud)))
        return np.vstack(outs)

    def embed(self, clouds: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(clouds, np.ndarray):
            return self.embed_pointclouds([clouds])
        return self.embed_pointclouds(clouds)


def _load_real_lidar() -> _RealLidarAdapter:
    global _real_lidar
    if _real_lidar is None:
        real_image_any: Any = _load_real_image()
        _real_lidar = _RealLidarAdapter(LidarBEVEmbedder(real_image_any))
    return _real_lidar


_text_singleton: Optional[TextBackend] = None
_image_singleton: Optional[ImageBackend] = None
_lidar_singleton: Optional[LidarEmbedder] = None


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "real").strip().lower()


def get_text_embedder() -> TextBackend:
    global _text_singleton
    if _text_singleton is not None:
        return _text_singleton
    if _backend() == "dummy":
        _text_singleton = DummyTextEmbedder()
    else:
        _text_singleton = cast(TextBackend, _load_real_text())
    return _text_singleton


def get_image_embedder() -> ImageBackend:
    global _image_singleton
    if _image_singleton is not None:
        return _image_singleton
    if _backend() == "dummy":
        _image_singleton = DummyImageEmbedder()
    else:
        _image_singleton = cast(ImageBackend, _load_real_image())
    return _image_singleton


def get_lidar_embedder() -> LidarEmbedder:
    global _lidar_singleton
    if _lidar_singleton is not None:
        return _lidar_singleton
    if _backend() == "dummy":
        _lidar_singleton = DummyLidarEmbedder()
    else:
        _lidar_singleton = _load_real_lidar()
    return _lidar_singleton


def reset_singletons_for_tests() -> None:
    global _text_singleton, _image_singleton, _lidar_singleton
    global _real_text, _real_image, _real_lidar
    _text_singleton = None
    _image_singleton = None
    _lidar_singleton = None
    _real_text = None
    _real_image = None
    _real_lidar = None
