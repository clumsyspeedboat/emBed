import sys
from types import SimpleNamespace

from src.storage import MinIOClient
from src.vectordb import lance as lance_mod


def test_resolve_minio_client_prefers_shim(monkeypatch):
    class FakeClient:
        pass

    shim = SimpleNamespace(MinIOClient=FakeClient)
    monkeypatch.setitem(sys.modules, "src.lancedb_manager", shim)
    try:
        assert lance_mod._resolve_minio_client_cls() is FakeClient
    finally:
        sys.modules.pop("src.lancedb_manager", None)


def test_resolve_minio_client_falls_back():
    sys.modules.pop("src.lancedb_manager", None)
    resolved = lance_mod._resolve_minio_client_cls()
    assert resolved is MinIOClient
