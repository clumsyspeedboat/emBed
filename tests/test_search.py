import os
import numpy as np
import pandas as pd

from src.search import MultiModalSearcher
from src.embedding_runtime import (
    reset_singletons_for_tests,
    DummyTextEmbedder,
    DummyImageEmbedder,
    DummyLidarEmbedder,
)

# Minimal fake table/query to capture search vectors and parameters
class _Query:
    def __init__(self, df):
        self._df = df
        self.last_where = None
        self.last_limit = None
    def where(self, expr):
        self.last_where = expr
        return self
    def limit(self, k):
        self.last_limit = int(k)
        return self
    def to_pandas(self):
        return self._df

class _FakeTable:
    def __init__(self):
        self.last_vec = None
        self.last_where = None
        self.last_limit = None
    def search(self, vec):
        # capture vector passed to the DB
        self.last_vec = vec
        q = _Query(pd.DataFrame([{"id":"x","score":0.0}]))
        def proxy_where(expr):
            q.where(expr)
            self.last_where = expr
            return q
        def proxy_limit(k):
            q.limit(k)
            self.last_limit = int(k)
            return q
        q.where = proxy_where  # type: ignore
        q.limit = proxy_limit  # type: ignore
        return q

def test_search_text_image_lidar(tmp_path):
    reset_singletons_for_tests()
    table = _FakeTable()
    s = MultiModalSearcher(
        table,
        text_embedder=DummyTextEmbedder(dim=8),
        image_embedder=DummyImageEmbedder(dim=8),
        lidar_embedder=DummyLidarEmbedder(dim=8),
    )

    # text
    df_t = s.search("hello world", "text", top_k=2)
    assert isinstance(df_t, pd.DataFrame)
    assert isinstance(table.last_vec, list) and len(table.last_vec) == 8

    # image from file path
    img_path = tmp_path / "q.jpg"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nabc")
    df_i = s.search(str(img_path), "image", top_k=2)
    assert isinstance(df_i, pd.DataFrame)
    assert isinstance(table.last_vec, list) and len(table.last_vec) == 8

    # lidar from ndarray
    pts = np.array([[0,0,0],[1,2,3]], dtype=np.float32)
    df_l = s.search(pts, "lidar", top_k=4)
    assert isinstance(df_l, pd.DataFrame)
    assert isinstance(table.last_vec, list) and len(table.last_vec) == 8
    assert table.last_limit == 4  # top_k respected

def test_where_clause_and_bytes_image(tmp_path):
    reset_singletons_for_tests()
    table = _FakeTable()
    s = MultiModalSearcher(
        table,
        text_embedder=DummyTextEmbedder(dim=6),
        image_embedder=DummyImageEmbedder(dim=6),
        lidar_embedder=DummyLidarEmbedder(dim=6),
    )

    # image as bytes
    img_bytes = b"\x89PNG\r\n\x1a\nxyz"
    df = s.search(img_bytes, "image", top_k=5, where="modality = 'image'")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(table.last_vec, list) and len(table.last_vec) == 6
    assert table.last_where == "modality = 'image'"
    assert table.last_limit == 5
