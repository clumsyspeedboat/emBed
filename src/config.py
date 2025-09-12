# src/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
except ImportError:
    # dotenv not installed, ignore
    pass

def _strip_quotes(s: str | None) -> str | None:
    if not s: return s
    if (s[0] == s[-1] and s[0] in ('"', "'")): return s[1:-1]
    return s

def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None: return default
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")

@dataclass
class MinioConfig:
    # defer env reads to __post_init__
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
        self.endpoint = _strip_quotes(self.endpoint or os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
        self.access_key = self.access_key or os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key = self.secret_key or os.getenv("MINIO_SECRET_KEY", "")
        self.session_token = self.session_token or os.getenv("MINIO_SESSION_TOKEN", "")
        self.region = self.region or os.getenv("MINIO_REGION", "us-east-1")
        self.buckets = self.buckets or os.getenv("MINIO_BUCKETS", "test-bucket")

        self.verify = _as_bool(os.getenv("MINIO_VERIFY") if self.verify is None else str(self.verify), True)
        self.allow_http = _as_bool(os.getenv("MINIO_ALLOW_HTTP") if self.allow_http is None else str(self.allow_http), False)
        self.force_path_style = _as_bool(os.getenv("MINIO_FORCE_PATH_STYLE") if self.force_path_style is None else str(self.force_path_style), True)
        self.virtual_hosted_style = _as_bool(os.getenv("MINIO_VIRTUAL_HOSTED_STYLE") if self.virtual_hosted_style is None else str(self.virtual_hosted_style), False)

    def _normalized_endpoint(self) -> str:
        ep = self.endpoint or ""
        if not ep.startswith(("http://", "https://")):
            ep = ("http://" if self.allow_http else "https://") + ep
        return ep

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

@dataclass
class LanceDBConfig:
    uri: str | None = None
    allow_http_env: str | None = None
    s3_region: str | None = None
    s3_endpoint: str | None = None

    def __post_init__(self) -> None:
        self.uri = _strip_quotes(self.uri or os.getenv("LANCEDB_URI", "data/lancedb"))
        self.allow_http_env = os.getenv("LANCEDB_ALLOW_HTTP", "0") if self.allow_http_env is None else self.allow_http_env
        self.s3_region = os.getenv("AWS_DEFAULT_REGION", os.getenv("MINIO_REGION", "us-east-1")) if self.s3_region is None else self.s3_region
        self.s3_endpoint = _strip_quotes(self.s3_endpoint or os.getenv("AWS_ENDPOINT", os.getenv("MINIO_ENDPOINT", "")))

    def storage_options(self) -> dict:
        import os
        endpoint = self.s3_endpoint  # e.g., "https://10.16.246.151:9000"
        region = (self.s3_region or "us-east-1")
        access = os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or ""
        secret = os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or ""
        session = os.getenv("MINIO_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN") or ""
        allow_http_env = (os.getenv("MINIO_ALLOW_HTTP", "") or os.getenv("AWS_ALLOW_HTTP", ""))
        allow_http = allow_http_env.lower() in ("1", "true", "yes")
        force_path = (os.getenv("MINIO_FORCE_PATH_STYLE", "true").lower() in ("1", "true", "yes"))

        def s(v: object) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            return "" if v is None else str(v)

        opts = {
            # object_store prefers these aws_* keys
            "aws_region": s(region),
            "aws_endpoint": s(endpoint),
            "aws_access_key_id": s(access),
            "aws_secret_access_key": s(secret),
            "aws_s3_force_path_style": s(force_path),
            # keep legacy keys too (harmless)
            "region": s(region),
            "endpoint": s(endpoint),
        }
        if session:
            opts["aws_session_token"] = s(session)
        # Only relevant for http:// endpoints
        if endpoint.startswith("http://") and allow_http:
            opts["aws_allow_http"] = "true"

        return opts
