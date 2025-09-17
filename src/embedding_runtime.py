"""Runtime factory for embedding models.

This module provides a simple abstraction over different embedding backends.
It exposes lazy-loaded singleton instances for text, image, and LiDAR embedders.
The backend used is determined by the ``EMBEDDING_BACKEND`` environment variable.

If ``EMBEDDING_BACKEND`` is ``"dummy"``, the factory returns deterministic
dummy embedders that hash their inputs to pseudo-random unit vectors. This
backend is extremely fast and requires no external dependencies. It is
primarily intended for unit testing and prototyping.

For any other value (or if the variable is unset), the factory wraps the
real embedders defined in :mod:`src.embedding`. These embedders perform
actual model inference via third-party libraries. To avoid costly imports
and initializations when the dummy backend is active, the real embedders are
loaded lazily.

The module also defines a ``reset_singletons_for_tests`` helper which clears
cached singletons between test runs, ensuring isolation.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Protocol, Iterable, Optional, Any, cast
import numpy as np


# ---------- Protocols ----------
class TextEmbedder(Protocol):
    dim: int
    def embed(self, texts: Iterable[str]) -> np.ndarray: ...


class ImageEmbedder(Protocol):
    dim: int
    def embed(self, images: Iterable[bytes]) -> np.ndarray: ...


class LidarEmbedder(Protocol):
    dim: int
    def embed(self, clouds: Iterable[np.ndarray]) -> np.ndarray: ...


# ---------- Helper ----------
def _hash_to_unit_vec(data: bytes, dim: int) -> np.ndarray:
    """Convert arbitrary bytes into a deterministic unit vector."""
    h = hashlib.sha256(data).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------- Dummy Embedders ----------
@dataclass
class DummyTextEmbedder:
    dim: int = 512

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        arr = [_hash_to_unit_vec(t.encode("utf-8"), self.dim) for t in texts]
        return np.vstack(arr)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return self.embed_text(texts)


@dataclass
class DummyImageEmbedder:
    dim: int = 512

    def embed_images(self, images: Iterable[bytes]) -> np.ndarray:
        arr = [_hash_to_unit_vec(b, self.dim) for b in images]
        return np.vstack(arr)

    def embed(self, images: Iterable[bytes]) -> np.ndarray:
        return self.embed_images(images)


@dataclass
class DummyLidarEmbedder:
    dim: int = 512

    def embed(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        rows = []
        for arr in clouds:
            a = np.asarray(arr)
            a32 = a.astype(np.float32, copy=False).ravel()
            rows.append(_hash_to_unit_vec(a32.tobytes(), self.dim))
        return np.vstack(rows)

    # Provide batch-friendly API symmetry with real adapter
    def embed_pointclouds(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        return self.embed(clouds)


# ---------- Real Embedders (lazy import) ----------
_real_text: Optional[Any] = None
_real_image: Optional[Any] = None
_real_lidar: Optional["_RealLidarAdapter"] = None


def _load_real_text() -> Any:
    global _real_text
    if _real_text is None:
        from src.embedding import TextEmbedder as RealTextEmbedder  # type: ignore[attr-defined]
        _real_text = RealTextEmbedder()
    return _real_text


def _load_real_image() -> Any:
    global _real_image
    if _real_image is None:
        from src.embedding import ImageEmbedder as RealImageEmbedder  # type: ignore[attr-defined]
        _real_image = RealImageEmbedder()
    return _real_image


class _RealLidarAdapter:
    """Adapter to unify your LidarBEVEmbedder under LidarEmbedder Protocol."""

    def __init__(self, real_lidar_obj):
        self._lidar = real_lidar_obj
        dummy = np.zeros((2, 3), dtype=np.float32)
        v = self._lidar.embed(dummy)
        self.dim = int(v.shape[-1])

    def embed_pointclouds(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        outs = []
        for c in clouds:
            outs.append(self._lidar.embed(np.asarray(c)))
        return np.vstack(outs)

     # Conform to LidarEmbedder Protocol: accept iterable or single array
    def embed(self, clouds: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(clouds, np.ndarray):
            return self.embed_pointclouds([clouds])
        return self.embed_pointclouds(clouds)


def _load_real_lidar() -> _RealLidarAdapter:
    global _real_lidar
    if _real_lidar is None:
        from src.embedding import LidarBEVEmbedder  # your existing class
        real_img_any: Any = _load_real_image()
        _real_lidar = _RealLidarAdapter(LidarBEVEmbedder(real_img_any))
    return _real_lidar


# ---------- Singletons ----------
_text_singleton: Optional[TextEmbedder] = None
_image_singleton: Optional[ImageEmbedder] = None
_lidar_singleton: Optional[LidarEmbedder] = None


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "real").strip().lower()


def get_text_embedder() -> TextEmbedder:
    global _text_singleton
    if _text_singleton is not None:
        return _text_singleton
    if _backend() == "dummy":
        _text_singleton = DummyTextEmbedder()
    else:
        _text_singleton = cast(TextEmbedder, _load_real_text())
    return _text_singleton


def get_image_embedder() -> ImageEmbedder:
    global _image_singleton
    if _image_singleton is not None:
        return _image_singleton
    if _backend() == "dummy":
        _image_singleton = DummyImageEmbedder()
    else:
        _image_singleton = cast(ImageEmbedder, _load_real_image())
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
