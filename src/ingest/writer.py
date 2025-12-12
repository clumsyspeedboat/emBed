"""Purpose: encapsulate table creation and batch upsert semantics for LanceDB.
Why extend: adapt to alternative vector stores or add instrumentation around upsert operations.
How extend: subclass `UpsertWriter` or introduce new methods (e.g. `ensure_schema`) and call them from the pipeline before ingestion.
"""
from __future__ import annotations

from typing import Dict, List

from src.vectordb import LanceDBManager


class UpsertWriter:
    """Create-or-open tables and batch upsert rows."""

    def __init__(self, manager: LanceDBManager):
        self.manager = manager
        self._opened: Dict[str, bool] = {}

    @staticmethod
    def _escape_ids(ids: List[str]) -> List[str]:
        return [("'" + x.replace("'", "''") + "'") for x in ids]

    def ensure_table(self, table_name: str, rows: List[dict], mode: str) -> None:
        if self._opened.get(table_name):
            return
        if mode == "overwrite":
            self.manager.create_table(table_name, rows, mode="overwrite")
        else:
            try:
                _ = self.manager.get_table(table_name)
            except Exception:
                seed = rows if rows else []
                self.manager.create_table(table_name, seed, mode="overwrite")
        self._opened[table_name] = True

    def upsert_rows(self, table_name: str, rows: List[dict]) -> int:
        if not rows:
            return 0
        table = self.manager.get_table(table_name)
        upsert = getattr(table, "upsert", None)
        if callable(upsert):
            upsert(rows)
            return len(rows)
        ids = [row["id"] for row in rows]
        chunk = 1000
        for start in range(0, len(ids), chunk):
            escaped = self._escape_ids(ids[start:start + chunk])
            table.delete(f"id IN ({','.join(escaped)})")
        table.add(rows)
        return len(rows)


__all__ = ["UpsertWriter"]
