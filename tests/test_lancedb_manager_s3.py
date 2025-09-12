import os
import types
from unittest.mock import MagicMock
from src.lancedb_manager import LanceDBManager
from src.config import LanceDBConfig

class _FakeTable:
    def __init__(self, rows):
        self._rows = rows
        self.index_calls = []
    def create_index(self, **kwargs):
        self.index_calls.append(kwargs)
    def count_rows(self):
        return len(self._rows)
    def to_arrow(self):
        import pyarrow as pa
        emb = [r["embedding"] for r in self._rows]
        return pa.table({"embedding": pa.array(emb)})
    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._rows)

class _FakeDB:
    def __init__(self):
        self.tables = {}
    def create_table(self, name, data, mode, primary_key="id"):
        t = _FakeTable(data)
        self.tables[name] = t
        return t
    def open_table(self, name):
        return self.tables[name]

class _FakeS3Client:
    def __init__(self):
        self.deleted_prefixes = []
        self.uploads = []
    def delete_prefix(self, bucket, prefix):
        self.deleted_prefixes.append((bucket, prefix))
        return 0
    def upload_dir(self, local_dir, bucket, prefix):
        # simulate upload
        self.uploads.append((local_dir, bucket, prefix))
        return 3

def test_s3_create_upload_and_open(monkeypatch, tmp_path):
    # Patch lancedb.connect to return: first local DB, then remote DB
    db_local = _FakeDB()
    db_remote = _FakeDB()
    calls = {"n": 0}
    def _connect(uri, storage_options=None):
        calls["n"] += 1
        return db_local if calls["n"] == 1 else db_remote
    monkeypatch.setattr("lancedb.connect", _connect)

    # Patch MinIOClient used inside lancedb_manager
    fake_s3 = _FakeS3Client()
    monkeypatch.setattr("src.lancedb_manager.MinIOClient", lambda *a, **k: fake_s3)

    rows = [{"id": str(i), "embedding": [0.0]*32} for i in range(300)]
    cfg = LanceDBConfig(uri="s3://buck/prefix")
    m = LanceDBManager(cfg)
    tbl = m.create_table("t", rows, mode="overwrite")

    # local index created
    assert len(db_local.tables["t"].index_calls) == 1
    # s3 uploads invoked with proper prefix
    assert fake_s3.deleted_prefixes == [("buck", "prefix/t.lance")]
    assert len(fake_s3.uploads) == 1
    # returned table is remote
    assert tbl is db_remote.tables["t"]
