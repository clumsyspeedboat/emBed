"""Utilities for working with MinIO/S3 object storage."""
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
# Corrected absolute import
from src.config import MinioConfig

# (The rest of the file is unchanged)
def _env(*keys: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value among keys."""
    for k in keys:
        v = os.environ.get(k)
        if v is not None and v != "":
            return v
    return default


class MinIOClient:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        config: MinioConfig | None = None,
        session_token: str | None = None,
    ):
        """
        Accepts explicit args, a MinioConfig, or falls back to environment variables.

        Env keys checked:
          - Endpoint:  AWS_ENDPOINT, S3_ENDPOINT, MINIO_ENDPOINT
          - Access:    AWS_ACCESS_KEY_ID, MINIO_ACCESS_KEY
          - Secret:    AWS_SECRET_ACCESS_KEY, MINIO_SECRET_KEY
          - Region:    AWS_DEFAULT_REGION, MINIO_REGION  (default: us-east-1)
          - Token:     AWS_SESSION_TOKEN (optional)
        """

        # If a MinioConfig object is supplied in 'config', prefer it as base
        if isinstance(config, MinioConfig):
            endpoint = endpoint or config.endpoint
            access_key = access_key or config.access_key
            secret_key = secret_key or config.secret_key
            region = region or (config.region or None)

        # Also support passing a MinioConfig in the 'endpoint' slot (legacy call sites)
        if isinstance(endpoint, MinioConfig):
            cfg = endpoint
            endpoint, access_key, secret_key, region = (
                cfg.endpoint,
                cfg.access_key,
                cfg.secret_key,
                (cfg.region or None),
            )

        # Prefer explicit args, else environment
        self.endpoint = endpoint or _env("AWS_ENDPOINT", "S3_ENDPOINT", "MINIO_ENDPOINT")
        self.access_key = access_key or _env("AWS_ACCESS_KEY_ID", "MINIO_ACCESS_KEY")
        self.secret_key = secret_key or _env("AWS_SECRET_ACCESS_KEY", "MINIO_SECRET_KEY")
        self.region = region or _env("AWS_DEFAULT_REGION", "MINIO_REGION", default="us-east-1")
        self.session_token = session_token or _env("AWS_SESSION_TOKEN")

        if not self.endpoint or not self.access_key or not self.secret_key:
            raise ValueError(
                "MinioClient requires endpoint, access_key, and secret_key "
                "(set MINIO_* or AWS_* environment variables)."
            )

        # Normalize endpoint URL
        if not (self.endpoint.startswith("http://") or self.endpoint.startswith("https://")):
            self.endpoint = "https://" + self.endpoint

        use_https = self.endpoint.startswith("https://")

        cfg = Config(
            max_pool_connections=256,
            retries={"max_attempts": 8, "mode": "standard"},
            connect_timeout=5,
            read_timeout=60,
            signature_version="s3v4",
        )

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token,
            region_name=self.region,
            use_ssl=use_https,
            config=cfg,
        )

    # -------- convenience methods --------

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)

    def upload_file(self, file_path: str, bucket: str, key: str) -> None:
        self.ensure_bucket(bucket)
        self._client.upload_file(file_path, bucket, key)

    def download_file(self, bucket: str, key: str, file_path: str) -> None:
        self._client.download_file(bucket, key, file_path)

    def list_objects(self, bucket: str, prefix: str = "", limit: int | None = None) -> list[str]:
        """List keys under bucket/prefix with full pagination, optional limit."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix

        total = 0
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                total += 1
                if limit is not None and total >= limit:
                    return keys
        return keys

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        resp = self._client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def get_presigned_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
