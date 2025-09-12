import numpy as np

from src.embedding import LidarBEVEmbedder, _l2_normalize


class _FakeImageEmbedder:
    """Deterministic, fast fake image embedder with .embed(images) -> (B, D)."""
    def __init__(self, dim=8):
        self.dim = dim
    def embed(self, images):
        # Handle iterable vs list (no I/O)
        if not isinstance(images, list):
            images = list(images)
        b = len(images)
        vec = np.zeros((b, self.dim), dtype=np.float32)
        vec[:, 0] = 1.0  # unit vector along first dim
        return vec


def test_l2_normalize_basic():
    v = np.array([[3.0, 4.0]], dtype=np.float32)
    out = _l2_normalize(v)
    assert np.allclose(out, np.array([[0.6, 0.8]], dtype=np.float32), atol=1e-6)


def test_lidar_bev_single_and_batch():
    ie = _FakeImageEmbedder(dim=8)
    bev = LidarBEVEmbedder(ie, img_size=64, meters=10.0, px_per_m=2.0)

    # Single cloud (two points)
    cloud1 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0]], dtype=np.float32)
    emb1 = bev.embed(cloud1)
    assert emb1.shape == (1, 8)
    assert np.allclose(emb1[0, 0], 1.0)

    # Batch of two clouds
    cloud2 = np.array([[-2.0, 0.5, 0.2],
                       [ 3.0, -1.5, 2.2]], dtype=np.float32)
    emb2 = bev.embed([cloud1, cloud2])
    assert emb2.shape == (2, 8)
    assert np.allclose(emb2[:, 0], 1.0)
