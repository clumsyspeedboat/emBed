"""Utilities for working with MinIO/S3 object storage."""
import boto3
from botocore.exceptions import ClientError
from .config import MinioConfig

class MinioClient:
    def __init__(self, config: MinioConfig) -> None:
        self._config = config
        self._client = boto3.client("s3", **config.to_boto_kwargs())
        from botocore.config import Config
        self._client = boto3.client(
            "s3",
            config=Config(s3={'addressing_style': 'path'}),
            **config.to_boto_kwargs()
        )

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

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        resp = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [c["Key"] for c in resp.get("Contents", [])]

    def get_object_bytes(self, bucket: str, key: str) -> bytes:
        resp = self._client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
