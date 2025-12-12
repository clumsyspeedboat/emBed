"""Purpose: provide a reusable service registry for wiring runtime dependencies.
Why extend: register alternative storage/vector backends or instrument service creation in one place.
How extend: subclass `ServiceRegistry` or pass overrides when instantiating `ServiceContainer`; e.g. provide a custom `s3` factory for Azure Blob storage.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import LanceDBConfig, MinioConfig
from src.storage import MinIOClient
from src.vectordb import LanceDBManager


class ServiceRegistry:
    """Minimal registry supporting overrides and lazy factory evaluation."""

    def __init__(self, *, overrides: Optional[Dict[str, Any]] = None) -> None:
        self._overrides: Dict[str, Any] = overrides.copy() if overrides else {}

    def set(self, name: str, value: Any) -> None:
        self._overrides[name] = value

    def _resolve(self, name: str, factory) -> Any:
        value = self._overrides.get(name, NotImplemented)
        if value is NotImplemented:
            value = factory()
            self._overrides[name] = value
        elif callable(value):
            value = value()
            self._overrides[name] = value
        return value


class ServiceContainer(ServiceRegistry):
    """Default runtime wiring for S3, LanceDB, and embedding factories."""

    def __init__(
        self,
        minio_cfg: Optional[MinioConfig] = None,
        lancedb_cfg: Optional[LanceDBConfig] = None,
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(overrides=overrides)
        self.minio_cfg = minio_cfg or MinioConfig()
        self.lancedb_cfg = lancedb_cfg or LanceDBConfig()

    @property
    def s3(self) -> MinIOClient:
        return self._resolve("s3", lambda: MinIOClient(self.minio_cfg))

    @property
    def vectordb(self) -> LanceDBManager:
        return self._resolve("vectordb", lambda: LanceDBManager(self.lancedb_cfg))

    def text_embedder(self):
        from src.embedding_runtime import get_text_embedder
        return self._resolve("text_embedder", get_text_embedder)

    def image_embedder(self):
        from src.embedding_runtime import get_image_embedder
        return self._resolve("image_embedder", get_image_embedder)

    def lidar_embedder(self):
        from src.embedding_runtime import get_lidar_embedder
        return self._resolve("lidar_embedder", get_lidar_embedder)


def default_container(
    minio_cfg: Optional[MinioConfig] = None,
    lancedb_cfg: Optional[LanceDBConfig] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ServiceContainer:
    return ServiceContainer(minio_cfg=minio_cfg, lancedb_cfg=lancedb_cfg, overrides=overrides)


__all__ = ["ServiceRegistry", "ServiceContainer", "default_container"]
