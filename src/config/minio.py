"""Purpose: encapsulate MinIO/S3 configuration and boto client options.
Why extend: expose new tunables (timeouts, addressing modes) or validation hooks.
How extend: extend the dataclass fields/post-init logic and update `to_boto_kwargs()` accordingly; document new env vars in README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.config._env import ensure_dotenv_loaded
from src.config.utils import strip_quotes, as_bool

__all__ = ["MinioConfig"]


@dataclass
class MinioConfig:
    endpoint: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    buckets: str | None = None

    verify: bool | None = None
    allow_http: bool | None = None
    force_path_style: bool | None = None
    virtual_hosted_style: bool | None = None

    def __post_init__(self) -> None:
        ensure_dotenv_loaded()

        self.endpoint = strip_quotes(self.endpoint or os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
        self.access_key = self.access_key or os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key = self.secret_key or os.getenv("MINIO_SECRET_KEY", "")
        self.session_token = self.session_token or os.getenv("MINIO_SESSION_TOKEN", "")
        self.region = self.region or os.getenv("MINIO_REGION", "us-east-1")
        self.buckets = self.buckets or os.getenv("MINIO_BUCKETS", "test-bucket")

        verify_env = os.getenv("MINIO_VERIFY") if self.verify is None else str(self.verify)
        allow_http_env = os.getenv("MINIO_ALLOW_HTTP") if self.allow_http is None else str(self.allow_http)
        force_path_style_env = (
            os.getenv("MINIO_FORCE_PATH_STYLE") if self.force_path_style is None else str(self.force_path_style)
        )
        virtual_hosted_env = (
            os.getenv("MINIO_VIRTUAL_HOSTED_STYLE")
            if self.virtual_hosted_style is None else str(self.virtual_hosted_style)
        )

        self.verify = as_bool(verify_env, True)
        self.allow_http = as_bool(allow_http_env, False)
        self.force_path_style = as_bool(force_path_style_env, True)
        self.virtual_hosted_style = as_bool(virtual_hosted_env, False)

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _normalized_endpoint(self) -> str:
        endpoint = self.endpoint or ""
        if not endpoint.startswith(("http://", "https://")):
            endpoint = ("http://" if self.allow_http else "https://") + endpoint
        return endpoint

    def to_boto_kwargs(self) -> dict:
        from botocore.config import Config as BotoConfig

        endpoint_url = self._normalized_endpoint()
        addressing_style = "virtual" if self.virtual_hosted_style else "path"

        cfg = BotoConfig(
            max_pool_connections=256,
            retries={"max_attempts": 8, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
        )

        if endpoint_url.startswith("http://") and not self.allow_http:
            raise ValueError("HTTP endpoint used but MINIO_ALLOW_HTTP is not enabled.")

        return {
            "endpoint_url": endpoint_url,
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "aws_session_token": (self.session_token or None),
            "region_name": self.region,
            "use_ssl": endpoint_url.startswith("https://"),
            "verify": bool(self.verify),
            "config": cfg,
        }
