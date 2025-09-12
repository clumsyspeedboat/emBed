# src/storage.py
"""Utilities for working with MinIO/S3 object storage."""
from __future__ import annotations
import os
import boto3
from botocore.exceptions import ClientError
from typing import Iterable
from src.config import MinioConfig

class MinIOClient:
    def __init__(
        self,
        config: MinioConfig | None = None,
        *,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        region: str | None = None,
        verify: bool | None = None,
        allow_http: bool | None = None,
        force_path_style: bool | None = None,
        virtual_hosted_style: bool | None = None,
    ):
        """Preferred: pass a MinioConfig. Explicit kwargs override the config."""
        base = config or MinioConfig()
        eff = MinioConfig(
            endpoint=endpoint or base.endpoint,
            access_key=access_key or base.access_key,
            secret_key=secret_key or base.secret_key,
            session_token=session_token or base.session_token,
            region=region or base.region,
            buckets=base.buckets,
            verify=base.verify if verify is None else verify,
            allow_http=base.allow_http if allow_http is None else allow_http,
            force_path_style=base.force_path_style if force_path_style is None else force_path_style,
            virtual_hosted_style=base.virtual_hosted_style if virtual_hosted_style is None else virtual_hosted_style,
        )
        kwargs = eff.to_boto_kwargs()
        self._client = boto3.client("s3", **kwargs)

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

    def iter_objects(self, bucket: str, prefix: str = "") -> Iterable[str]:
        """Yield keys under bucket/prefix."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        resp = self._client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def get_presigned_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects under the given prefix. Returns count."""
        count = 0
        keys = [{"Key": k} for k in self.iter_objects(bucket, prefix)]
        for i in range(0, len(keys), 1000):
            chunk = keys[i:i + 1000]
            if not chunk:
                break
            self._client.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
            count += len(chunk)
        return count

    def upload_dir(self, local_dir: str, bucket: str, prefix: str) -> int:
        """Recursively upload a directory to S3, preserving relative paths."""
        uploaded = 0
        for root, _, files in os.walk(local_dir):
            for fname in files:
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, start=local_dir).replace("\\", "/")
                key = f"{prefix}/{rel_path}" if prefix else rel_path
                self._client.upload_file(src_path, bucket, key)
                uploaded += 1
        return uploaded

    def delete_objects(self, bucket: str, keys: Iterable[str]) -> int:
        """Delete a collection of objects in chunks of 1000."""
        keys_list = list(keys)
        count = 0
        for i in range(0, len(keys_list), 1000):
            chunk = keys_list[i:i + 1000]
            self._client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in chunk]})
            count += len(chunk)
        return count
