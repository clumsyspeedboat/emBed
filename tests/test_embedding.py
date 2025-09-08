import pytest
import numpy as np
from unittest.mock import MagicMock
from src.embedding import _l2_normalize, get_embedding_model, TextEmbedder

def test_l2_normalize():
    """Tests that the L2 normalization works correctly."""
    vec = np.array([[1, 2, 3]], dtype=np.float32)
    normalized_vec = _l2_normalize(vec)
    norm = np.linalg.norm(normalized_vec)
    assert np.isclose(norm, 1.0)

def test_get_embedding_model_factory(monkeypatch):
    """Tests the embedding model factory."""
    # Mock the heavy dependencies to prevent them from being imported
    monkeypatch.setattr("src.embedding.TextEmbedder", MagicMock())
    monkeypatch.setattr("src.embedding.ImageEmbedder", MagicMock())
    
    text_model = get_embedding_model("text")
    assert isinstance(text_model, MagicMock)
    
    image_model = get_embedding_model("image")
    assert isinstance(image_model, MagicMock)
    
    with pytest.raises(ValueError):
        get_embedding_model("unknown_type")