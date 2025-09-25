"""Search runtime helpers shared across app surfaces."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.search import MultiModalSearcher

__all__ = [
    "SearchInputs",
    "SearchService",
    "normalize_modalities",
    "vectorize_modalities",
    "apply_metric_threshold",
    "load_lidar_points",
]


@dataclass()
class SearchInputs:
    """Container for modality-specific query payloads."""

    modalities: Sequence[str]
    text_query: Optional[str] = None
    image_path: Optional[str] = None
    image_blob: Optional[bytes] = None
    lidar_path: Optional[str] = None
    lidar_blob: Optional[bytes] = None
    lidar_ext: Optional[str] = None
    allow_lidar: bool = True


class SearchService:
    """Bridge UI payloads with the LanceDB searcher."""

    def __init__(self, table: any, *, searcher: Optional[MultiModalSearcher] = None) -> None:
        self.table = table
        self.searcher = searcher or MultiModalSearcher(table)

    def vectorize(self, inputs: SearchInputs) -> list[float]:
        return vectorize_modalities(self.searcher, inputs)

    def run(
        self,
        inputs: SearchInputs,
        *,
        topk: int,
        filter_hint: Optional[str],
        where: Optional[str],
        metric: str,
        nprobes: int,
        refine_factor: int,
        metric_threshold: Optional[float],
        prefetch: int,
    ) -> pd.DataFrame:
        vector = self.vectorize(inputs)
        q = self.table.search(vector)
        q = _apply_metric(q, metric)
        if nprobes > 0:
            q = _call_query_method_preferring_class(q, "nprobes", int(nprobes))
        if refine_factor > 0:
            q = _call_query_method_preferring_class(q, "refine_factor", int(refine_factor))
        if filter_hint:
            q = _call_query_method_preferring_class(q, "where", filter_hint)
        if where:
            q = _call_query_method_preferring_class(q, "where", where)
        q = _call_query_method_preferring_class(q, "limit", prefetch)
        df = q.to_pandas()
        df = apply_metric_threshold(df, metric, metric_threshold)
        return df.head(int(topk))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_modalities(
    modalities: Iterable[str] | None,
    *,
    fallback: Optional[str],
    allow_lidar: bool,
) -> list[str]:
    allowed = {"text", "image"}
    if allow_lidar:
        allowed.add("lidar")

    result: list[str] = []
    seen: set[str] = set()

    def _try_add(value: Optional[str]) -> None:
        if not value:
            return
        key = value.strip().lower()
        if not key or key in seen or key not in allowed:
            return
        seen.add(key)
        result.append(key)

    if modalities is not None:
        for item in modalities:
            if isinstance(item, str):
                _try_add(item)

    if not result and fallback:
        _try_add(fallback)
    if not result:
        _try_add("text")
    return result


class VectorizationError(ValueError):
    pass


def load_lidar_points(ext: str, blob: bytes) -> np.ndarray:
    ext = (ext or "").lower()
    if ext == ".bin":
        try:
            arr = np.frombuffer(blob, dtype=np.float32)
            cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
            return arr.reshape(-1, cols)
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorizationError("Unable to parse LiDAR .bin payload") from exc
    if ext == ".pcd":
        try:
            text = blob.decode("utf-8", errors="ignore").strip().split()
            arr = np.asarray([float(x) for x in text], dtype=np.float32)
            cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
            return arr.reshape(-1, cols)[:, :3]
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorizationError("Unable to parse LiDAR .pcd payload") from exc
    raise VectorizationError(f"Unsupported LiDAR extension: {ext or 'unknown'}")


def _flatten(vec: any, label: str) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1).tolist()
    raise VectorizationError(f"Embedding for modality '{label}' has unexpected shape {arr.shape}.")


def _vector_for_modality(searcher: MultiModalSearcher, modality: str, inputs: SearchInputs) -> list[float]:
    if modality == "text":
        if not inputs.text_query:
            raise VectorizationError("Text query is empty.")
        return _flatten(searcher.text_embedder.embed(inputs.text_query), "text")

    if modality == "image":
        if inputs.image_blob:
            return _flatten(searcher.image_embedder.embed(inputs.image_blob), "image")
        if inputs.image_path:
            with open(inputs.image_path, "rb") as fh:
                return _flatten(searcher.image_embedder.embed(fh.read()), "image")
        raise VectorizationError("Provide an image file or a readable server path.")

    if modality == "lidar":
        if not inputs.allow_lidar:
            raise VectorizationError("LiDAR queries are disabled.")
        if inputs.lidar_blob:
            pts = load_lidar_points(inputs.lidar_ext or "", inputs.lidar_blob)
            return _flatten(searcher.lidar_embedder.embed(pts), "lidar")
        if inputs.lidar_path:
            _, ext = os.path.splitext(inputs.lidar_path)
            with open(inputs.lidar_path, "rb") as fh:
                pts = load_lidar_points(ext, fh.read())
            return _flatten(searcher.lidar_embedder.embed(pts), "lidar")
        raise VectorizationError("Provide a LiDAR .pcd/.bin file or a readable server path.")

    raise VectorizationError(f"Unknown modality: {modality}")


def vectorize_modalities(searcher: MultiModalSearcher, inputs: SearchInputs) -> list[float]:
    vectors: list[np.ndarray] = []
    for mod in inputs.modalities:
        vec = _vector_for_modality(searcher, mod, inputs)
        arr = np.asarray(vec, dtype=np.float32)
        vectors.append(arr)

    if not vectors:
        raise VectorizationError("Select at least one modality to run a search.")
    if len(vectors) == 1:
        return vectors[0].tolist()

    stacked = np.vstack(vectors)
    mean_vec = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0.0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32).tolist()


def apply_metric_threshold(df: Optional[pd.DataFrame], metric: str, value: Optional[float]) -> Optional[pd.DataFrame]:
    if df is None or value is None:
        return df
    try:
        metric_key = (metric or "l2").strip().lower()
        if metric_key in {"euclidean", "l2", "l2_distance"}:
            return df[df.get("_distance", 0) <= float(value)]
        distance_cutoff = 1.0 - float(value)
        return df[df.get("_distance", 0) <= distance_cutoff]
    except Exception:
        return df


# LanceDB chaining helpers (mirrors src.core.search)


def _call_query_method_preferring_class(q: any, method: str, *args):
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


def _apply_metric(q: any, metric: str):
    metric_key = (metric or "l2").strip().lower()
    if metric_key in {"euclidean", "l2", "l2_distance"}:
        name = "l2"
    elif metric_key in {"cos", "cosine"}:
        name = "cosine"
    elif metric_key in {"dot", "ip", "inner_product", "dot_product"}:
        name = "cosine"
    else:
        name = "l2"
    return _call_query_method_preferring_class(q, "metric", name)
