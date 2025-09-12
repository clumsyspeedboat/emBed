import os
import pytest
from src.config import MinioConfig, LanceDBConfig

def test_minio_config_http_disallowed_by_default(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "10.0.0.5:9000")
    monkeypatch.delenv("MINIO_ALLOW_HTTP", raising=False)
    kw = MinioConfig().to_boto_kwargs()
    assert kw["endpoint_url"].startswith("https://")

def test_minio_config_http_allowed(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "10.0.0.5:9000")
    monkeypatch.setenv("MINIO_ALLOW_HTTP", "true")
    kw = MinioConfig().to_boto_kwargs()
    assert kw["endpoint_url"].startswith("http://")
    # verify is honored but irrelevant for http
    assert "verify" in kw

def test_minio_config_https_self_signed(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "https://10.0.0.5:9000")
    monkeypatch.setenv("MINIO_VERIFY", "false")
    kw = MinioConfig().to_boto_kwargs()
    assert kw["endpoint_url"].startswith("https://")
    assert kw["verify"] is False

def test_addressing_style_default_path(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "https://10.0.0.5:9000")
    m = MinioConfig()
    kw = m.to_boto_kwargs()
    # Botocore Config object is opaque; check repr for addressing_style
    cfg = kw["config"]
    # Botocore exposes s3 options as a dict attribute on the Config object
    assert getattr(cfg, "s3", {}).get("addressing_style") == "path"

def test_lancedb_storage_options_allow_http(monkeypatch):
    monkeypatch.setenv("LANCEDB_URI", "s3://bucket/prefix")
    monkeypatch.setenv("AWS_ENDPOINT", "http://10.0.0.5:9000")
    monkeypatch.setenv("LANCEDB_ALLOW_HTTP", "1")
    opts = LanceDBConfig().storage_options()
    assert opts["endpoint"].startswith("http://")
    assert opts.get("allow_http") == "true"
