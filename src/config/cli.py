"""Purpose: capture CLI-specific defaults derived from environment variables.
Why extend: centralise new CLI toggles or defaults without scattering `os.getenv` lookups in command handlers.
How extend: add dataclass fields (e.g. `auto_index_default`) and update the CLI to consume them when parsing arguments or rendering config output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from src.config._env import ensure_dotenv_loaded

ensure_dotenv_loaded()


@dataclass
class CLIConfig:
    default_metric: str = field(default_factory=lambda: os.getenv("LANCEDB_METRIC", "l2").strip().lower())
    snapshot_keys: List[str] = field(default_factory=lambda: [
        "MINIO_ENDPOINT",
        "MINIO_REGION",
        "MINIO_VERIFY",
        "MINIO_ALLOW_HTTP",
        "MINIO_FORCE_PATH_STYLE",
        "MINIO_VIRTUAL_HOSTED_STYLE",
        "MINIO_BUCKETS",
        "LANCEDB_URI",
        "LANCEDB_ALLOW_HTTP",
        "AWS_ENDPOINT",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ])

    def __post_init__(self) -> None:
        if not self.default_metric:
            self.default_metric = "l2"


__all__ = ["CLIConfig"]
