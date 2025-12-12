"""Purpose: wrap concurrent S3 downloads for reuse by both streaming and buffered ingestion paths.
Why extend: support alternative transports (e.g. async clients) or retry policies.
How extend: add new functions such as `download_with_retries` and switch pipeline calls to the appropriate helper.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.storage import MinIOClient


def download_many(s3: MinIOClient, bucket: str, keys: List[str], workers: int) -> Dict[str, bytes | Exception]:
    out: Dict[str, bytes | Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(s3.get_object_bytes, bucket, k): k for k in keys}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                out[key] = fut.result()
            except Exception as exc:
                out[key] = exc
    return out
