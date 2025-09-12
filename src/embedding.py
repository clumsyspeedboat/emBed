from __future__ import annotations

import io
import os
from typing import Iterable, Optional, List

import numpy as np
import torch
from PIL import Image

from src.exceptions import EmbeddingError


# ---------- Helpers ----------

def _pick_device() -> str:
    dev = os.getenv("DEVICE", "auto").lower()
    if dev == "cpu":
        return "cpu"
    if dev == "cuda" and torch.cuda.is_available():
        return "cuda"
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


# ---------- Text Embedder ----------

class TextEmbedder:
    """
    Text encoder with unified .embed(texts) API (returns (B, D) float32).

    If TEXT_MODEL starts with 'clip:', use open_clip text encoder
    so vectors are in the same space/dim as image embeddings.
    Otherwise, use sentence-transformers.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.device = device or _pick_device()
        self.model_name = model_name or os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        if self.model_name.startswith("clip:"):
            import open_clip
            spec = self.model_name.split("clip:", 1)[1]
            name, _, ckpt = spec.partition("#")
            self._backend = "clip"
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                name, pretrained=(ckpt or "openai"), device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(name)
            self.clip_model.eval()
            with torch.no_grad():
                toks = self.tokenizer(["dummy"]).to(self.device)
                feat = self.clip_model.encode_text(toks)
                self.dim = int(feat.shape[-1])
        else:
            from sentence_transformers import SentenceTransformer
            self._backend = "st"
            self.st_model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self.st_model.get_sentence_embedding_dimension()

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        try:
            if self._backend == "clip":
                toks = self.tokenizer(list(texts)).to(self.device)
                with torch.no_grad():
                    feats = self.clip_model.encode_text(toks).float()
                feats = feats.cpu().numpy().astype("float32")
                return _l2_normalize(feats)

            embs = self.st_model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype("float32")
            return _l2_normalize(embs)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e


# ---------- Image Embedder ----------

class ImageEmbedder:
    """
    CLIP image encoder (defaults to ViT-B/32#openai).
    Unified .embed(images) API; input = bytes or PIL.Image; output = (B, D) float32, L2-normalized.
    """
    def __init__(self, model_spec: Optional[str] = None, device: Optional[str] = None):
        import open_clip
        self.device = device or _pick_device()
        spec = model_spec or os.getenv("IMAGE_MODEL", "ViT-B-32#openai")
        name, _, ckpt = spec.partition("#")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=(ckpt or "openai"), device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(name)
        self.model.eval()
        # infer dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            feat = self.model.encode_image(dummy)
            self.dim = int(feat.shape[-1])

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def embed(self, images: Iterable[bytes | Image.Image]) -> np.ndarray:
        # Accept single item transparently
        if isinstance(images, (bytes, Image.Image)):
            images = [images]
        try:
            batch: List[torch.Tensor] = []
            for it in images:
                if isinstance(it, bytes):
                    img = Image.open(io.BytesIO(it)).convert("RGB")
                elif isinstance(it, Image.Image):
                    img = it.convert("RGB")
                else:
                    raise TypeError("ImageEmbedder expects bytes or PIL.Image.Image")
                batch.append(self._to_tensor(img))

            x = torch.cat(batch, dim=0)
            with torch.no_grad():
                feats = self.model.encode_image(x).float()
            feats = feats.cpu().numpy().astype("float32")
            return _l2_normalize(feats)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed image: {e}") from e


# ---------- LiDAR → BEV → Image Embedder ----------

class LidarBEVEmbedder:
    """
    Projects XYZ(+intensity) to a fixed-size BEV image and uses the image embedder.
    Unified .embed(clouds) API; accepts single np.ndarray or iterable of arrays; returns (B, D).
    """
    def __init__(self, image_embedder: ImageEmbedder, img_size: int = 224, meters: float = 50.0, px_per_m: float = 2.0):
        self.ie = image_embedder
        self.img_size = img_size
        self.meters = meters
        self.scale = px_per_m

    def _pointcloud_to_bev(self, pts: np.ndarray) -> Image.Image:
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("LiDAR points need shape (N, >=3) for (x,y,z[,intensity])")
        x, y = pts[:, 0], pts[:, 1]
        w = h = int(self.meters * 2 * self.scale)  # cover [-meters, +meters]

        # map meters→pixels (origin center)
        u = np.clip(((x + self.meters) * self.scale).astype(np.int32), 0, w - 1)
        v = np.clip(((y + self.meters) * self.scale).astype(np.int32), 0, h - 1)
        canvas = np.zeros((h, w), dtype=np.uint8)

        # intensity channel if present; else z as height proxy
        if pts.shape[1] >= 4:
            val = np.clip((pts[:, 3] * 255.0), 0, 255).astype(np.uint8)
        else:
            z = pts[:, 2]
            z_norm = (z - z.min()) / max(1e-6, (z.max() - z.min()))
            val = (z_norm * 255.0).astype(np.uint8)

        # draw: keep the max per pixel
        canvas[v, u] = np.maximum(canvas[v, u], val)
        img = Image.fromarray(canvas).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)

        return img

    def embed(self, clouds: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Accepts a single cloud (np.ndarray) or an iterable of clouds.
        Returns a (B, D) float32 ndarray.
        """
        try:
            if isinstance(clouds, np.ndarray):
                clouds = [clouds]
            images: List[Image.Image] = [self._pointcloud_to_bev(pts) for pts in clouds]
            feats = self.ie.embed(images)  # (B, D), already normalized
            return feats
        except Exception as e:
            raise EmbeddingError(f"Failed to embed LiDAR data: {e}") from e


# ---------- Factory ----------

def get_embedding_model(model_type: str, model_name: Optional[str] = None):
    if model_type == "text":
        return TextEmbedder(model_name)
    elif model_type == "image":
        return ImageEmbedder(model_name)
    elif model_type == "lidar":
        return LidarBEVEmbedder(ImageEmbedder(model_name))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
