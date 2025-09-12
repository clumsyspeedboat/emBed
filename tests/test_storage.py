import types
import os
import builtins
import pytest

from unittest.mock import MagicMock
from src.storage import MinIOClient

class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages
    def paginate(self, **kwargs):
        return self._pages

class _FakeS3:
    def __init__(self, pages):
        self.pages = pages
        self.uploads = []
        self.deletes = []
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self.pages)
    def upload_file(self, src, bucket, key):
        self.uploads.append((src, bucket, key))
    def delete_objects(self, Bucket, Delete):
        self.deletes.append((Bucket, Delete))
    def head_bucket(self, Bucket):
        pass
    def create_bucket(self, Bucket):
        pass
    def get_object(self, Bucket, Key):
        return {"Body": types.SimpleNamespace(read=lambda: b"data")}
    def download_file(self, Bucket, Key, Filename):
        with open(Filename, "wb") as f:
            f.write(b"ok")
    def generate_presigned_url(self, *a, **k):
        return "https://example"

def test_iter_and_list_objects(monkeypatch, tmp_path):
    pages = [
        {"Contents": [{"Key": "a/x"}, {"Key": "a/y"}]},
        {"Contents": [{"Key": "a/z"}]},
    ]
    fake = _FakeS3(pages)
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)

    s3 = MinIOClient()
    keys = list(s3.iter_objects("bucket", "a/"))
    assert keys == ["a/x", "a/y", "a/z"]

    keys2 = s3.list_objects("bucket", "a/", limit=2)
    assert keys2 == ["a/x", "a/y"]

def test_delete_prefix_chunks(monkeypatch):
    pages = [{"Contents": [{"Key": f"k/{i}"} for i in range(2505)]}]
    fake = _FakeS3(pages)
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    s3 = MinIOClient()
    deleted = s3.delete_prefix("b", "k/")
    # 2505 keys -> 3 calls: 1000 + 1000 + 505
    assert deleted == 2505
    assert len(fake.deletes) == 3

def test_upload_dir(monkeypatch, tmp_path):
    pages = [{"Contents": []}]
    fake = _FakeS3(pages)
    monkeypatch.setattr("boto3.client", lambda *a, **k: fake)
    d = tmp_path / "dir"
    (d / "x/y").mkdir(parents=True)
    (d / "x/y/f.txt").write_text("hi")
    (d / "g.bin").write_bytes(b"ok")
    s3 = MinIOClient()
    cnt = s3.upload_dir(str(d), "buck", "pref")
    assert cnt == 2
    keys = sorted([k for _, _, k in fake.uploads])
    assert keys == ["pref/g.bin", "pref/x/y/f.txt"]


@pytest.fixture(autouse=True)
def _minio_env(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ALLOW_HTTP", "true")
    monkeypatch.setenv("MINIO_REGION", "us-east-1")
    # optional but common in tests
    monkeypatch.setenv("MINIO_VERIFY", "false")