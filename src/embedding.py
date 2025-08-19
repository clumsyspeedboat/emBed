"""Dummy embedding utilities for text, images, and LiDAR."""

import hashlib
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, data) -> np.ndarray:
        ...


class DummyTextEmbedder(BaseEmbedder):
    def embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(16)


class DummyImageEmbedder(BaseEmbedder):
    def embed(self, img_bytes: bytes) -> np.ndarray:
        seed = int(hashlib.sha256(img_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(16)


class DummyLidarEmbedder(BaseEmbedder):
    def embed(self, points: list[tuple[float, float, float]]) -> np.ndarray:
        flat = ",".join([f"{x:.2f}" for p in points for x in p])
        seed = int(hashlib.sha256(flat.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(16)
