import os
import numpy as np
from src.services import default_container
from src.embedding_runtime import reset_singletons_for_tests


# Fake S3 and LanceDB classes for unit tests.
class _FakeS3:
    def __init__(self, cfg):
        self.cfg = cfg  # store config to verify injection


class _FakeVDB:
    def __init__(self, cfg):
        self.cfg = cfg


def setup_function(_):
    reset_singletons_for_tests()


def test_container_wires_configs_and_caches(monkeypatch):
    # Use dummy backend for speed
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")

    # Set a custom endpoint for verification
    monkeypatch.setenv("MINIO_ENDPOINT", "https://10.0.0.5:9000")

    # Patch out the real clients in services.py
    import src.services as svc_mod
    monkeypatch.setattr(svc_mod, "MinIOClient", _FakeS3, raising=False)
    monkeypatch.setattr(svc_mod, "LanceDBManager", _FakeVDB, raising=False)

    c = default_container()

    s3a = c.s3
    s3b = c.s3
    v1 = c.vectordb
    v2 = c.vectordb
    # cached properties reuse the same instances
    assert s3a is s3b
    assert v1 is v2

    # fakes received configuration objects
    assert hasattr(s3a, "cfg")
    assert hasattr(v1, "cfg")
    assert str(s3a.cfg.endpoint).startswith("https://")


def test_container_exposes_embedders(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    c = default_container()

    t1, t2 = c.text_embedder(), c.text_embedder()
    i1, i2 = c.image_embedder(), c.image_embedder()
    l1, l2 = c.lidar_embedder(), c.lidar_embedder()

    assert t1 is t2
    assert i1 is i2
    assert l1 is l2

    # check that embedder methods exist and return valid shapes
    tv = t1.embed(["ok"])
    iv = i1.embed([b"ok"])
    lv = l1.embed_pointclouds([np.zeros((2, 3), dtype=np.float32)])
    assert tv.shape == (1, t1.dim)
    assert iv.shape == (1, i1.dim)
    assert lv.shape == (1, l1.dim)
