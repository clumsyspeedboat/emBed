"""
Service container for runtime dependencies (S3, LanceDB, embedders).

Why:
- Eliminate globals and ad-hoc wiring.
- Enable easy swapping (real vs. fakes) in unit tests.
- Keep construction lazy, cached, and minimal.

What:
- `ServiceContainer` wires:
    • `s3`       -> MinIOClient(MinioConfig)  [lazy, cached_property]
    • `vectordb` -> LanceDBManager(LanceDBConfig)  [lazy, cached_property]
    • `text_embedder()` / `image_embedder()` / `lidar_embedder()` -> from embedding_runtime
- `default_container()` returns a ready container using env-driven configs.

How to test:
- Monkeypatch MinIOClient/LanceDBManager with fakes; assert correct wiring + caching.
- Set EMBEDDING_BACKEND=dummy; assert singleton embedders are exposed and functional.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

from src.config import MinioConfig, LanceDBConfig
from src.storage import MinIOClient
from src.lancedb_manager import LanceDBManager
from src.embedding_runtime import (
    get_text_embedder,
    get_image_embedder,
    get_lidar_embedder,
)


@dataclass
class ServiceContainer:
    """
    Minimal dependency-injection container.

    - s3 / vectordb are constructed lazily and cached per container instance.
    - embedders are provided via embedding_runtime lazy singletons (fast dummy backend available).
    """
    minio_cfg: MinioConfig
    lancedb_cfg: LanceDBConfig

    @functools.cached_property
    def s3(self) -> MinIOClient:
        return MinIOClient(self.minio_cfg)

    @functools.cached_property
    def vectordb(self) -> LanceDBManager:
        return LanceDBManager(self.lancedb_cfg)

    # Embedders (methods to avoid caching issues across tests/process boundaries)
    def text_embedder(self):
        return get_text_embedder()

    def image_embedder(self):
        return get_image_embedder()

    def lidar_embedder(self):
        return get_lidar_embedder()


def default_container(
    minio_cfg: Optional[MinioConfig] = None,
    lancedb_cfg: Optional[LanceDBConfig] = None,
) -> ServiceContainer:
    return ServiceContainer(minio_cfg or MinioConfig(), lancedb_cfg or LanceDBConfig())
