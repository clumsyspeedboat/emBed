"""Purpose: encapsulate LanceDB configuration and storage options for S3-compatible backends.
Why extend: support new auth flows or regional overrides.
How extend: add dataclass fields and update `storage_options()` to surface the extra parameters the LanceDB client expects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.config._env import ensure_dotenv_loaded
from src.config.utils import strip_quotes

__all__ = ["LanceDBConfig"]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class LanceDBConfig:
    uri: str | None = None
    allow_http_env: str | None = None
    s3_region: str | None = None
    s3_endpoint: str | None = None

    def __post_init__(self) -> None:
        ensure_dotenv_loaded()

        self.uri = strip_quotes(self.uri or _env("LANCEDB_URI", "data/lancedb"))
        if self.allow_http_env is None:
            self.allow_http_env = _env("LANCEDB_ALLOW_HTTP", "0")
        if self.s3_region is None:
            self.s3_region = _env("AWS_DEFAULT_REGION", _env("MINIO_REGION", "us-east-1"))
        if self.s3_endpoint is None:
            self.s3_endpoint = strip_quotes(_env("AWS_ENDPOINT", _env("MINIO_ENDPOINT", "")))

    # ----------------------------------------------------------------------------------
    # Storage options
    # ----------------------------------------------------------------------------------
    def storage_options(self) -> dict:
        endpoint = self.s3_endpoint or ""
        region = self.s3_region or "us-east-1"
        access = _env("MINIO_ACCESS_KEY") or _env("AWS_ACCESS_KEY_ID") or ""
        secret = _env("MINIO_SECRET_KEY") or _env("AWS_SECRET_ACCESS_KEY") or ""
        session = _env("MINIO_SESSION_TOKEN") or _env("AWS_SESSION_TOKEN") or ""
        allow_http_candidates = (
            (self.allow_http_env or ""),
            _env("MINIO_ALLOW_HTTP", ""),
            _env("AWS_ALLOW_HTTP", ""),
        )
        allow_http = any(str(val).lower() in ("1", "true", "yes") for val in allow_http_candidates)
        force_path = _env("MINIO_FORCE_PATH_STYLE", "true").lower() in ("1", "true", "yes")

        def stringify(value: object) -> str:
            if isinstance(value, bool):
                return "true" if value else "false"
            return "" if value is None else str(value)

        options: dict[str, str] = {
            "aws_region": stringify(region),
            "aws_endpoint": stringify(endpoint),
            "aws_access_key_id": stringify(access),
            "aws_secret_access_key": stringify(secret),
            "aws_s3_force_path_style": stringify(force_path),
            "region": stringify(region),
            "endpoint": stringify(endpoint),
        }
        if session:
            options["aws_session_token"] = stringify(session)
        if endpoint.startswith("http://") and allow_http:
            options["aws_allow_http"] = "true"
            options["allow_http"] = "true"
        return options
