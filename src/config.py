import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _strip_quotes(s: str) -> str:
    if s and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

@dataclass
class MinioConfig:
    endpoint: str = _strip_quotes(os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "")
    region: str = os.getenv("MINIO_REGION", "us-east-1")
    buckets: str = os.getenv("MINIO_BUCKETS", "test-bucket")
    def to_boto_kwargs(self) -> dict:
        return {
            "endpoint_url": self.endpoint,
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "region_name": self.region,
        }

@dataclass
class LanceDBConfig:
    uri: str = _strip_quotes(os.getenv("LANCEDB_URI", "data/lancedb"))
    allow_http_env: str = os.getenv("LANCEDB_ALLOW_HTTP", "0")
    s3_region: str = os.getenv("AWS_DEFAULT_REGION", os.getenv("MINIO_REGION", "us-east-1"))
    s3_endpoint: str = _strip_quotes(os.getenv("AWS_ENDPOINT", os.getenv("MINIO_ENDPOINT", "")))

    def _export_aws_env_from_minio(self) -> None:
        # Fill AWS_* from MINIO_* if AWS_* not provided
        ak = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ACCESS_KEY")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_SECRET_KEY")
        rg = os.getenv("AWS_DEFAULT_REGION") or os.getenv("MINIO_REGION") or "us-east-1"
        ep = os.getenv("AWS_ENDPOINT") or os.getenv("MINIO_ENDPOINT") or ""
        if ak: os.environ["AWS_ACCESS_KEY_ID"] = ak
        if sk: os.environ["AWS_SECRET_ACCESS_KEY"] = sk
        if rg: os.environ["AWS_DEFAULT_REGION"] = rg
        if ep: os.environ["AWS_ENDPOINT"] = ep
        if ep.startswith("http://") and self.allow_http_env not in ("0", "false", "False"):
            os.environ["ALLOW_HTTP"] = "true"

    def storage_options(self) -> dict:
        if self.uri.startswith("s3://"):
            self._export_aws_env_from_minio()
            opts = {"region": self.s3_region}
            if self.s3_endpoint:
                opts["endpoint"] = self.s3_endpoint
                if self.s3_endpoint.startswith("http://") and self.allow_http_env not in ("0", "false", "False"):
                    opts["allow_http"] = "true"  # must be string
            return opts
        return {}
