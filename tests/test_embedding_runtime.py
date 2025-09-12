"""
Unit tests for src.embedding_runtime.

Covers:
- Dummy backends (text/image/lidar) are deterministic and unit-normalized
- Lazy singletons are reused
- Real embedding module is not imported when backend=dummy
"""

import sys
import numpy as np

from src.embedding_runtime import (
    get_text_embedder,
    get_image_embedder,
    reset_singletons_for_tests,
)


def setup_function(_):
    """Reset singletons before each test for isolation."""
    reset_singletons_for_tests()


# -------------------- TEXT --------------------

def test_dummy_backend_text_is_deterministic(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    e = get_text_embedder()
    v1 = e.embed(["hello"])[0]
    v2 = e.embed(["hello"])[0]
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6


# -------------------- IMAGE --------------------

def test_dummy_backend_image_is_deterministic(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    e = get_image_embedder()
    img = b"\x89PNG\r\n\x1a\n" + b"abc123"  # arbitrary bytes
    v1 = e.embed([img])[0]
    v2 = e.embed([img])[0]
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6


# -------------------- SINGLETONS --------------------

def test_lazy_singletons_text_image(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    t1 = get_text_embedder()
    t2 = get_text_embedder()
    i1 = get_image_embedder()
    i2 = get_image_embedder()
    assert t1 is t2
    assert i1 is i2


def test_real_backend_is_lazy_not_imported_when_dummy(monkeypatch):
    """Ensure heavy src.embedding is not imported for dummy backend."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    sys.modules.pop("src.embedding", None)
    sys.modules.pop("embedding", None)
    _ = get_text_embedder()
    _ = get_image_embedder()
    assert "src.embedding" not in sys.modules
    assert "embedding" not in sys.modules


# -------------------- LIDAR --------------------

def test_dummy_backend_lidar_is_deterministic(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    reset_singletons_for_tests()
    from src.embedding_runtime import get_lidar_embedder  # import here to avoid circulars in older pytest
    e = get_lidar_embedder()

    # Same values, different dtypes → identical vector
    pc1 = np.array([[0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0]], dtype=np.float64)
    pc2 = np.array([[0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0]], dtype=np.float32)

    v1 = e.embed_pointclouds([pc1])[0]
    v2 = e.embed_pointclouds([pc2])[0]
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6


def test_lidar_singleton_reuse(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")
    reset_singletons_for_tests()
    from src.embedding_runtime import get_lidar_embedder
    l1 = get_lidar_embedder()
    l2 = get_lidar_embedder()
    assert l1 is l2
