"""Purpose: make configuration objects (MinIO, LanceDB, Webapp, CLI, Ingest) available at a stable import location.
Why extend: add new configuration dataclasses for alternative backends or surfaces without changing consumer code.
How extend: import the new dataclass here and list it in `__all__`, ensuring `. _env` loads shared state first.
"""
from __future__ import annotations

from src.config._env import ensure_dotenv_loaded
from src.config.lancedb import LanceDBConfig
from src.config.webapp import WebAppConfig
from src.config.cli import CLIConfig
from src.config.ingest import IngestConfig
from src.config.minio import MinioConfig

ensure_dotenv_loaded()

__all__ = ["MinioConfig", "LanceDBConfig", "WebAppConfig", "CLIConfig", "IngestConfig", "ensure_dotenv_loaded"]
