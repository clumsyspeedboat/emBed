import os
import pytest
from src.config import MinioConfig, LanceDBConfig

def test_minio_config_defaults(monkeypatch):
    """Tests that MinioConfig loads default values correctly when no env vars are set."""
    # Temporarily remove environment variables that might interfere with the test
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_REGION", raising=False)
    monkeypatch.delenv("MINIO_BUCKETS", raising=False)

    config = MinioConfig()
    assert config.endpoint == "http://localhost:9000"
    assert config.region == "us-east-1"
    assert config.buckets == "test-bucket"

def test_minio_config_env_vars(monkeypatch):
    """Tests that MinioConfig correctly loads variables from the environment."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://test-minio:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test_access_key")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test_secret_key")
    
    config = MinioConfig()
    assert config.endpoint == "http://test-minio:9000"
    assert config.access_key == "test_access_key"
    assert config.secret_key == "test_secret_key"

def test_lancedb_config_s3_storage_options(monkeypatch):
    """Tests that LanceDB storage options are correctly generated for S3."""
    monkeypatch.setenv("LANCEDB_URI", "s3://my-bucket/lancedb")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("AWS_ENDPOINT", "http://test-s3:9000")
    
    config = LanceDBConfig()
    options = config.storage_options()
    
    assert config.uri == "s3://my-bucket/lancedb"
    assert options["region"] == "eu-central-1"
    assert options["endpoint"] == "http://test-s3:9000"