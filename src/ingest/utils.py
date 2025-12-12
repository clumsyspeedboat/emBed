"""Purpose: offer small helpers shared across ingestion modules (batching, vector conversions, S3 path formatting).
Why extend: reuse common math/iteration utilities when adding new ingestion flows without duplicating logic.
How extend: add new helpers (e.g. `chunk_dict`) here and import them from pipeline modules as needed.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List

import numpy as np


def to_list_vec(array: np.ndarray) -> List[float]:
    """Ensure numpy embeddings are returned as a 1-D float list."""
    arr = np.asarray(array)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype("float32").tolist()


def s3_path(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def batched(seq: Iterable, n: int) -> Iterator[list]:
    it = iter(seq)
    while True:
        chunk: list = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            pass
        if not chunk:
            break
        yield chunk
