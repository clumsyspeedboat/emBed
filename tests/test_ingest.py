import pytest
import numpy as np
from unittest.mock import MagicMock
from src.ingest import ingest_s3_objects

# A valid 1x1 pixel PNG image in bytes
MINIMAL_PNG_BYTES = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

@pytest.fixture
def mock_s3_client():
    client = MagicMock()
    client.list_objects.return_value = ["test.jpg"]
    client.get_object_bytes.return_value = MINIMAL_PNG_BYTES
    return client

@pytest.fixture
def mock_lancedb_manager():
    manager = MagicMock()
    manager.get_table.return_value = MagicMock()
    return manager

def test_ingest_s3_objects(mock_s3_client, mock_lancedb_manager, monkeypatch):
    """Tests the main ingest flow with mocked dependencies."""
    # Mock the embedding models to avoid loading them and to control their output
    mock_embedder = MagicMock()
    # Configure the mock to return a valid numpy array, as the real one would.
    # We are processing one image, so it should return one vector.
    mock_embedder.embed.return_value = np.zeros((1, 128))  # Example embedding shape

    monkeypatch.setattr("src.ingest.img_embedder", mock_embedder)
    monkeypatch.setattr("src.ingest.text_embedder", MagicMock())
    monkeypatch.setattr("src.ingest.lidar_embedder", MagicMock())
    
    ingest_s3_objects(
        manager=mock_lancedb_manager,
        s3=mock_s3_client,
        bucket="test-bucket",
        prefix="",
        table_name="test-table",
    )
    
    # Check that the manager was called to create the table.
    # Since rows will be generated now, this assertion will pass.
    mock_lancedb_manager.create_table_with_data.assert_called_once()