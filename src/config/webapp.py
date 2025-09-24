"""Purpose: capture FastAPI webapp settings resolved from environment variables.
Why extend: add strongly-typed accessors for new UI/API toggles without sprinkling `os.getenv` throughout the app.
How extend: declare additional dataclass fields (e.g. `auth_provider: str`) and update the web layer to consume them instead of reading the environment directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from src.config._env import ensure_dotenv_loaded
from src.config.utils import as_bool, strip_quotes

ensure_dotenv_loaded()

def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class WebAppConfig:
    table: str = field(default_factory=lambda: os.getenv("WEBAPP_TABLE", "multimodal"))
    show_images: bool = field(default_factory=lambda: as_bool(os.getenv("WEBAPP_SHOW_IMAGES", "1"), True))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("WEBAPP_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))))
    presign_ttl: int = field(default_factory=lambda: int(os.getenv("WEBAPP_PRESIGN_TTL", "3600")))
    allow_lidar: bool = field(default_factory=lambda: as_bool(os.getenv("WEBAPP_ALLOW_LIDAR", "1"), True))
    default_metric: str = field(default_factory=lambda: os.getenv("WEBAPP_DEFAULT_METRIC", os.getenv("LANCEDB_METRIC", "l2")).strip().lower())
    cors_origins: List[str] = field(default_factory=lambda: _split_csv(os.getenv("WEBAPP_CORS_ORIGINS", "*")))
    lance_nprobes: int = field(default_factory=lambda: int(os.getenv("LANCEDB_NPROBES", "32")))
    lance_refine_factor: int = field(default_factory=lambda: int(os.getenv("LANCEDB_REFINE_FACTOR", "50")))

    def __post_init__(self) -> None:
        ensure_dotenv_loaded()
        self.table = strip_quotes(self.table) or "multimodal"
        self.default_metric = self.default_metric or "l2"
        if not self.cors_origins:
            self.cors_origins = ["*"]


__all__ = ["WebAppConfig"]
