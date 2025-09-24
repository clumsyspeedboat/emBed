"""Purpose: convert LiDAR point clouds into bird's-eye-view images and embed them using the image encoder.
Why extend: support different projections or intensity channels for new sensors.
How extend: adjust `_pointcloud_to_bev` or parameterise the constructor with additional transformation knobs.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from src.embedding.image import ImageEmbedder
from src.exceptions import EmbeddingError

__all__ = ["LidarBEVEmbedder"]


class LidarBEVEmbedder:
    """Projects XYZ(+intensity) clouds to BEV images and embeds them with CLIP."""

    def __init__(
        self,
        image_embedder: ImageEmbedder,
        img_size: int = 224,
        meters: float = 50.0,
        px_per_m: float = 2.0,
    ) -> None:
        self.ie = image_embedder
        self.img_size = img_size
        self.meters = meters
        self.scale = px_per_m

    def _pointcloud_to_bev(self, pts: np.ndarray) -> Image.Image:
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("LiDAR points need shape (N, >=3) for (x,y,z[,intensity])")
        x, y = pts[:, 0], pts[:, 1]
        side = int(self.meters * 2 * self.scale)

        u = np.clip(((x + self.meters) * self.scale).astype(np.int32), 0, side - 1)
        v = np.clip(((y + self.meters) * self.scale).astype(np.int32), 0, side - 1)
        canvas = np.zeros((side, side), dtype=np.uint8)

        if pts.shape[1] >= 4:
            val = np.clip((pts[:, 3] * 255.0), 0, 255).astype(np.uint8)
        else:
            z = pts[:, 2]
            z_norm = (z - z.min()) / max(1e-6, (z.max() - z.min()))
            val = (z_norm * 255.0).astype(np.uint8)

        canvas[v, u] = np.maximum(canvas[v, u], val)
        img = Image.fromarray(canvas).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        return img

    def embed(self, clouds: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        try:
            if isinstance(clouds, np.ndarray):
                clouds = [clouds]
            images: List[Image.Image] = [self._pointcloud_to_bev(points) for points in clouds]
            feats = self.ie.embed(images)
            return feats
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed LiDAR data: {exc}") from exc
