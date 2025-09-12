# src/lancedb_manager.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Iterable, Optional

import lancedb
import pyarrow as pa

from src.config import LanceDBConfig
from src.exceptions import VectorDBError
from src.storage import MinIOClient
from src.vectordb_manager import VectorDBManager


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) for e.g. s3://bucket/lancedb -> ('bucket','lancedb')"""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected s3:// URI")
    s = s3_uri[5:]
    bucket, _, prefix = s.partition("/")
    return bucket, prefix.rstrip("/")


def _infer_dim_from_rows(rows: list[dict]) -> Optional[int]:
    try:
        if rows and isinstance(rows[0].get("embedding"), (list, tuple)):
            return len(rows[0]["embedding"])
    except Exception:
        pass
    return None


def _choose_index_params(dim: Optional[int], rows_count: int,
                         default_partitions: int = 256) -> tuple[int, int]:
    """Return (num_partitions, num_sub_vectors/m) based on data size and dim."""
    # partitions ~ rows/50 but capped
    num_partitions = min(default_partitions, max(1, rows_count // 50))
    m = 1
    if dim:
        for cand in (32, 16, 8, 4, 2, 1):
            if dim % cand == 0:
                m = cand
                break
    return num_partitions, m


class LanceDBManager(VectorDBManager):
    """LanceDB manager with S3-aware create/upload and unified index policy."""

    MIN_ROWS_FOR_INDEX = 256

    def __init__(self, config: LanceDBConfig):
        self.config = config
        self._db = None

    # ---------- connection ----------

    def connect(self, uri: str, storage_options: dict[str, Any]) -> None:
        """Initialize or refresh the underlying LanceDB connection."""
        self._db = lancedb.connect(uri=uri, storage_options=storage_options)

    def _ensure_connected(self) -> None:
        if self._db is None:
            self.connect(self.config.uri, self.config.storage_options())

    # ---------- table creation ----------

    def create_table(
        self,
        name: str,
        data: Iterable[dict[str, Any]],
        mode: str = "overwrite",
    ) -> Any:
        """Create table; for S3 URIs, build locally then upload with MinIOClient."""
        rows = list(data)

        # ---- Non-S3 path: create directly ----
        if not self.config.uri.startswith("s3://"):
            self._ensure_connected()
            # prefer PK if supported
            try:
                tbl = self._db.create_table(name, data=rows, mode=mode, primary_key="id")
            except TypeError:
                tbl = self._db.create_table(name, data=rows, mode=mode)

            # centralized index policy
            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(tbl, rows_count=len(rows), dim=dim)
            return tbl

        # ---- S3 path: build locally, (optionally) index, upload, then open remote ----
        bucket, base_prefix = _parse_s3_uri(self.config.uri)
        remote_prefix = f"{base_prefix}/{name}.lance" if base_prefix else f"{name}.lance"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_uri = os.path.join(tmpdir, "lancedb")
            db_local = lancedb.connect(local_uri)

            # local create (prefer PK)
            try:
                tbl_local = db_local.create_table(name, data=rows, mode=mode, primary_key="id")
            except TypeError:
                tbl_local = db_local.create_table(name, data=rows, mode=mode)

            # local index using unified policy
            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(tbl_local, rows_count=len(rows), dim=dim)

            # upload to S3 via storage adapter
            s3 = MinIOClient()
            if mode == "overwrite":
                s3.delete_prefix(bucket, remote_prefix)

            local_table_path = os.path.join(local_uri, f"{name}.lance")
            uploaded = s3.upload_dir(local_table_path, bucket, remote_prefix)
            print(f"Uploaded {uploaded} files to s3://{bucket}/{remote_prefix}")

        # reconnect to remote and open table (allow short S3 propagation)
        self.connect(self.config.uri, self.config.storage_options())

        last_err = None
        for attempt in range(8):
            try:
                return self._db.open_table(name)
            except Exception as e:
                last_err = e
                import time
                time.sleep(0.5)

        # --- LAST RESORT: register the table remotely (S3) using the same rows ---
        # Some MinIO/S3 setups don't expose the dataset to the DB registry immediately.
        # Creating the table on S3 directly registers it and makes open_table work.
        try:
            try:
                tbl_remote = self._db.create_table(name, data=rows, mode="create", primary_key="id")
            except TypeError:
                # older lancedb versions don't support primary_key kwarg
                tbl_remote = self._db.create_table(name, data=rows, mode="create")
            # apply the same index policy if the dataset is large enough
            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(tbl_remote, rows_count=len(rows), dim=dim)
            return tbl_remote
        except Exception as e2:
            raise VectorDBError(
                "Uploaded dataset to S3 and attempted to open/register it, "
                f"but failed. Uploaded prefix: s3://{bucket}/{remote_prefix}. "
                f"Last open error: {last_err}. Remote create error: {e2}"
            ) from e2


    def create_empty_table(
        self,
        name: str,
        schema: pa.Schema,
        mode: str = "create",
    ) -> Any:
        """Create empty table with a given schema. PK enforced if 'id' exists."""
        self._ensure_connected()
        kwargs = {"name": name, "schema": schema, "mode": mode}
        try:
            if "id" in schema.names:
                return self._db.create_table(primary_key="id", **kwargs)
        except TypeError:
            pass
        except Exception as e:
            raise VectorDBError(f"Failed to create empty table '{name}': {e}") from e
        return self._db.create_table(**kwargs)

    def get_table(self, name: str) -> Any:
        """Open an existing table by name."""
        self._ensure_connected()
        try:
            return self._db.open_table(name)
        except Exception as e:
            raise VectorDBError(f"Failed to open table '{name}': {e}") from e

    # ---------- index policy (centralized) ----------

    def _apply_index_to_table(self, table_obj: Any, rows_count: int, dim: Optional[int]) -> None:
        """Apply IVF-PQ index with policy based on dataset size and dim."""
        if rows_count < self.MIN_ROWS_FOR_INDEX:
            print(f"Skipping index creation: rows={rows_count} < {self.MIN_ROWS_FOR_INDEX}")
            return
        num_partitions, m = _choose_index_params(dim, rows_count)
        table_obj.create_index(
            metric="l2",
            num_partitions=num_partitions,
            num_sub_vectors=m,
            replace=True,
            vector_column_name="embedding",
        )

    def create_index(
        self,
        table_name: str,
        num_partitions: int = 256,
        num_sub_vectors: int = 96,
    ) -> None:
        """
        Public API: create an IVF-PQ index on an existing table.

        - If explicit num_partitions/num_sub_vectors are provided (not defaults),
        they are applied directly.
        - Otherwise, an adaptive policy chooses params based on dataset size and dim.
        """
        tbl = self.get_table(table_name)

        # --- Row count ---
        rows_count: int
        try:
            rows_count = tbl.count_rows()
        except AttributeError:
            rows_count = len(tbl.to_pandas())

        # --- Embedding dimension ---
        dim: int | None = None
        try:
            arr = tbl.to_arrow().column("embedding")
            if len(arr) > 0:
                first = arr[0].as_py()
                if isinstance(first, (list, tuple)):
                    dim = len(first)
        except Exception:
            df = tbl.to_pandas().head(1)
            if not df.empty and isinstance(df.iloc[0].get("embedding"), (list, tuple)):
                dim = len(df.iloc[0]["embedding"])

        # --- If user gave explicit params, honor them directly ---
        if (num_partitions, num_sub_vectors) != (256, 96):
            tbl.create_index(
                metric="l2",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                replace=True,
                vector_column_name="embedding",
            )
            return

        # --- Default adaptive policy ---
        self._apply_index_to_table(tbl, rows_count, dim)
