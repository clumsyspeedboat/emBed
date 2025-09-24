"""Purpose: unify multimodal query handling for LanceDB adapters.
Why extend: support new modalities (audio, video) or adjust ANN tuning centrally.
How extend: subclass `MultiModalSearcher` and override `_embed_query`, or add helper methods that decorate LanceDB queries before execution.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.embedding_runtime import get_image_embedder, get_lidar_embedder, get_text_embedder
from src.ingest import load_lidar_bytes


def _call_query_method_preferring_class(q: Any, method: str, *args) -> Any:
    """Invoke a chained LanceDB query method while dodging recursive proxies."""
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
    """Wrap a LanceDB table with modality-aware embedding helpers."""

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
            with open(str(query), "rb") as fh:
                vec = self.image_embedder.embed([fh.read()])[0]
                return vec.tolist()

        if modality == "lidar":
            if isinstance(query, np.ndarray):
                vec = self.lidar_embedder.embed([query])[0]
                return vec.tolist()
            path = str(query)
            _, ext = os.path.splitext(path)
            with open(path, "rb") as fh:
                pts = load_lidar_bytes(ext.lower(), fh.read())
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

        metric_key = (metric or os.getenv("LANCEDB_METRIC", "l2")).strip().lower()
        if metric_key in {"euclidean", "l2", "l2_distance"}:
            metric_name = "l2"
        elif metric_key in {"cos", "cosine"}:
            metric_name = "cosine"
        elif metric_key in {"dot", "ip", "inner_product", "dot_product"}:
            metric_name = "cosine"
        else:
            metric_name = "l2"
        q = _call_query_method_preferring_class(q, "metric", metric_name)

        try:
            nprobes = int(os.getenv("LANCEDB_NPROBES", "32"))
        except Exception:
            nprobes = 32
        try:
            refine = int(os.getenv("LANCEDB_REFINE_FACTOR", "50"))
        except Exception:
            refine = 50
        if nprobes > 0:
            q = _call_query_method_preferring_class(q, "nprobes", nprobes)
        if refine > 0:
            q = _call_query_method_preferring_class(q, "refine_factor", refine)

        if where:
            try:
                setattr(self.table, "last_where", where)
            except Exception:
                pass
            q = _call_query_method_preferring_class(q, "where", where)

        q = _call_query_method_preferring_class(q, "limit", int(top_k))
        try:
            setattr(self.table, "last_limit", int(top_k))
        except Exception:
            pass

        results = q.to_pandas()
        elapsed = time.perf_counter() - t0
        print(f"\nSearch completed in: {elapsed:.4f} seconds")
        return results
