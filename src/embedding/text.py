"""Purpose: implement the text embedder abstraction using sentence-transformers or OpenCLIP.
Why extend: plug in alternate NLP models or pre/post-processing steps.
How extend: adjust the constructor to detect new `TEXT_MODEL` prefixes and implement a branch in `embed()` handling those backends.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import torch

from src.embedding.base import pick_device, l2_normalize
from src.exceptions import EmbeddingError

__all__ = ["TextEmbedder"]


class TextEmbedder:
    """
    Text encoder with unified ``.embed(texts)`` API (returns ``(B, D)`` float32 arrays).

    If the ``TEXT_MODEL`` environment variable starts with ``clip:``, the embedder
    uses OpenCLIP to align text vectors with image embeddings. Otherwise it falls
    back to a sentence-transformers model.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = device or pick_device()
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
                return l2_normalize(feats)

            embs = self.st_model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype("float32")
            return l2_normalize(embs)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc
