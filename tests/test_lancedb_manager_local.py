import types
import pytest
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
    def __init__(self, rows_ref):
        self.rows_ref = rows_ref
        self.tables = {}
    def create_table(self, name, data, mode, primary_key="id"):
        self.tables[name] = _FakeTable(data)
        self.rows_ref[:] = data  # expose for tests
        return self.tables[name]
    def open_table(self, name):
        return self.tables[name]

def test_local_create_and_index(monkeypatch):
    rows = [{"id": str(i), "embedding": [0.0]*32} for i in range(300)]
    # mock lancedb.connect
    rows_ref = []
    fake_db = _FakeDB(rows_ref)
    monkeypatch.setattr("lancedb.connect", lambda *a, **k: fake_db)

    cfg = LanceDBConfig(uri="memory://local")
    m = LanceDBManager(cfg)
    tbl = m.create_table("t", rows, mode="overwrite")
    # index should have been created (rows >= 256)
    assert len(tbl.index_calls) == 1
    # get_table reuses same db
    t2 = m.get_table("t")
    assert t2 is tbl
