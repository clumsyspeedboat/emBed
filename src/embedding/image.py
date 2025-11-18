"""Purpose: provide a CLIP-based image embedder with consistent API.
Why extend: experiment with different visual encoders or preprocessing pipelines.
How extend: modify model selection logic or override `_to_tensor`/`embed` to incorporate augmentations and alternative frameworks.
"""
from __future__ import annotations

import io
import os
from typing import Iterable, List, Optional

# Disable MKLDNN by default to avoid GELU primitive issues triggered on some hosts.
if "PYTORCH_ENABLE_MKLDNN" not in os.environ:
    os.environ["PYTORCH_ENABLE_MKLDNN"] = "0"

import numpy as np
import torch
from PIL import Image

from src.embedding.base import pick_device, l2_normalize
from src.exceptions import EmbeddingError

__all__ = ["ImageEmbedder"]

# Ensure the flag takes effect even if PyTorch was imported earlier.
try:
    if os.environ.get("PYTORCH_ENABLE_MKLDNN", "0").strip().lower() in {"0", "false", "no"}:
        torch.backends.mkldnn.enabled = False
except Exception:
    pass


class ImageEmbedder:
    """CLIP image encoder with unified ``.embed`` interface."""

    def __init__(self, model_spec: Optional[str] = None, device: Optional[str] = None) -> None:
        import open_clip

        self.device = device or pick_device()
        spec = model_spec or os.getenv("IMAGE_MODEL", "ViT-B-32#openai")
        name, _, ckpt = spec.partition("#")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=(ckpt or "openai"), device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(name)
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            feat = self.model.encode_image(dummy)
            self.dim = int(feat.shape[-1])

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def embed(self, images: Iterable[bytes | Image.Image]) -> np.ndarray:
        if isinstance(images, (bytes, Image.Image)):
            images = [images]
        try:
            batch: List[torch.Tensor] = []
            for item in images:
                if isinstance(item, bytes):
                    img = Image.open(io.BytesIO(item)).convert("RGB")
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    raise TypeError("ImageEmbedder expects bytes or PIL.Image.Image")
                batch.append(self._to_tensor(img))

            tensor = torch.cat(batch, dim=0)
            with torch.no_grad():
                feats = self.model.encode_image(tensor).float()
            feats = feats.cpu().numpy().astype("float32")
            return l2_normalize(feats)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed image: {exc}") from exc
