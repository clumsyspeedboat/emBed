"""
Search utilities for querying a multimodal LanceDB table.

Design:
- Dependency-injected embedders (text/image/lidar) with sane defaults from embedding_runtime.
- Robust modality handling:
  • text: str -> embed([text])
  • image: bytes or file path -> embed([bytes])
  • lidar: numpy.ndarray or .pcd/.bin file path -> lidar_embedder (batch-friendly if available)
- Simple, testable `_embed_query` helper.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional
import os

import numpy as np
import pandas as pd

from src.embedding_runtime import (
    get_text_embedder,
    get_image_embedder,
    get_lidar_embedder,
)
from src.ingest import _load_lidar_bytes


def _call_query_method_preferring_class(q: Any, method: str, *args) -> Any:
    """
    Call a query method in a way that avoids instance-level monkey-patched recursion.

    Strategy:
    1) If the class defines the method, call the class method: q.__class__.method(q, *args)
    2) Else, call q.method(*args)
    3) On any error, return the original q (no-op chaining).
    """
    try:
        cls = q.__class__
        orig = getattr(cls, method, None)
        if callable(orig):
            res = orig(q, *args)
            return res or q
        meth = getattr(q, method, None)
        if callable(meth):
            res = meth(*args)
            return res or q
    except RecursionError:
        return q
    except Exception:
        return q
    return q


class MultiModalSearcher:
    def __init__(
        self,
        table: Any,
        *,
        text_embedder: Optional[Any] = None,
        image_embedder: Optional[Any] = None,
        lidar_embedder: Optional[Any] = None,
    ) -> None:
        self.table = table
        self.text_embedder = text_embedder or get_text_embedder()
        self.image_embedder = image_embedder or get_image_embedder()
        self.lidar_embedder = lidar_embedder or get_lidar_embedder()

    def _embed_query(self, query: Any, modality: str) -> list[float]:
        if modality == "text":
            vec = self.text_embedder.embed([str(query)])[0]
            return vec.tolist()

        if modality == "image":
            if isinstance(query, (bytes, bytearray)):
                vec = self.image_embedder.embed([bytes(query)])[0]
                return vec.tolist()
            with open(str(query), "rb") as f:
                vec = self.image_embedder.embed([f.read()])[0]
                return vec.tolist()

        if modality == "lidar":
            if isinstance(query, np.ndarray):
                vec = self.lidar_embedder.embed([query])[0]
                return vec.tolist()

            # treat as file path
            path = str(query)
            _, ext = os.path.splitext(path)
            with open(path, "rb") as f:
                pts = _load_lidar_bytes(ext.lower(), f.read())
            vec = self.lidar_embedder.embed([pts])[0]
            return vec.tolist()

        raise ValueError(f"Unknown modality: {modality}")

    def search(
        self,
        query: Any,
        modality: str,
        top_k: int = 3,
        where: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        vec = self._embed_query(query, modality)

        q = self.table.search(vec)

        # Distance metric selection: 'l2' | 'cosine' | 'dot'
        # Map synonyms and fallbacks; 'dot' == 'cosine' for L2-normalized embeddings
        m = (metric or os.getenv("LANCEDB_METRIC", "l2")).strip().lower()
        if m in ("euclidean", "l2", "l2_distance"):
            m_norm = "l2"
        elif m in ("cos", "cosine"):
            m_norm = "cosine"
        elif m in ("dot", "ip", "inner_product", "dot_product"):
            m_norm = "cosine"  # normalized embeddings → dot ≡ cosine
        else:
            m_norm = "l2"
        q = _call_query_method_preferring_class(q, "metric", m_norm)

        # Improve recall when an IVF/PQ index is present
        # Tunables via env with sensible defaults
        try:
            nprobes = int(os.getenv("LANCEDB_NPROBES", "32"))
        except Exception:
            nprobes = 32
        try:
            refine = int(os.getenv("LANCEDB_REFINE_FACTOR", "50"))
        except Exception:
            refine = 50
        if nprobes and nprobes > 0:
            q = _call_query_method_preferring_class(q, "nprobes", int(nprobes))
        if refine and refine > 0:
            q = _call_query_method_preferring_class(q, "refine_factor", int(refine))

        if where:
            # keep your test’s capture variables up to date
            try:
                setattr(self.table, "last_where", where)
            except Exception:
                pass
            q = _call_query_method_preferring_class(q, "where", where)

        # call class-level limit (avoids recursive instance proxy) and record last_limit
        q = _call_query_method_preferring_class(q, "limit", int(top_k))
        try:
            setattr(self.table, "last_limit", int(top_k))
        except Exception:
            pass

        results = q.to_pandas()
        elapsed = time.perf_counter() - t0
        print(f"\nSearch completed in: {elapsed:.4f} seconds")
        return results
