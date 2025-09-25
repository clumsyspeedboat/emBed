"""Purpose: provide the LanceDB manager implementation with S3 support and index policy.
Why extend: tune index heuristics, add observability, or support new LanceDB features.
How extend: modify helper functions (like `_choose_index_params`) or extend `LanceDBManager` methods, ensuring tests cover the new logic.
"""
from __future__ import annotations

import os
import tempfile
import time
from importlib import import_module
from typing import Any, Iterable, Optional

import lancedb
import pyarrow as pa

from src.config import LanceDBConfig
from src.exceptions import VectorDBError
from src.storage import MinIOClient
from src.vectordb.base import VectorDBManager

__all__ = ["LanceDBManager"]


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected s3:// URI")
    bucket, _, prefix = s3_uri[5:].partition("/")
    return bucket, prefix.rstrip("/")


def _infer_dim_from_rows(rows: list[dict]) -> Optional[int]:
    try:
        if rows and isinstance(rows[0].get("embedding"), (list, tuple)):
            return len(rows[0]["embedding"])
    except Exception:
        pass
    return None


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _resolve_minio_client_cls():
    """Return the active MinIO client class, preferring the legacy shim when patched."""
    try:
        shim = import_module("src.lancedb_manager")
        client = getattr(shim, "MinIOClient", MinIOClient)
        return client
    except Exception:
        return MinIOClient



def _choose_index_params(dim: Optional[int], rows_count: int, default_partitions: Optional[int] = None) -> tuple[int, int]:
    env_parts = os.getenv("INDEX_NUM_PARTITIONS")
    env_m = os.getenv("INDEX_M")

    if env_parts:
        try:
            num_partitions = max(1, int(env_parts))
        except Exception:
            num_partitions = max(1, (default_partitions or 256))
    else:
        import math

        est = int(4 * math.sqrt(max(1, rows_count)))
        min_p = _get_int_env("INDEX_MIN_PARTITIONS", 512)
        max_p = _get_int_env("INDEX_MAX_PARTITIONS", 8192)
        num_partitions = min(max_p, max(min_p, est))

    if env_m:
        try:
            m = max(1, int(env_m))
        except Exception:
            m = 1
    else:
        m = 1
        if dim and dim > 0:
            for candidate in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
                if dim % candidate == 0:
                    m = candidate
                    break

    return num_partitions, m


class LanceDBManager(VectorDBManager):
    """LanceDB manager with S3-aware create/upload and unified index policy."""

    MIN_ROWS_FOR_INDEX = _get_int_env("INDEX_MIN_ROWS", 256)

    def __init__(self, config: LanceDBConfig) -> None:
        self.config = config
        self._db = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self, uri: str, storage_options: dict[str, Any]) -> None:
        self._db = lancedb.connect(uri=uri, storage_options=storage_options)

    def _ensure_connected(self) -> None:
        if self._db is None:
            self.connect(self.config.uri, self.config.storage_options())

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    def create_table(self, name: str, data: Iterable[dict[str, Any]], mode: str = "overwrite") -> Any:
        rows = list(data)

        if not self.config.uri.startswith("s3://"):
            self._ensure_connected()
            try:
                table = self._db.create_table(name, data=rows, mode=mode, primary_key="id")
            except TypeError:
                table = self._db.create_table(name, data=rows, mode=mode)

            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(table, rows_count=len(rows), dim=dim)
            return table

        bucket, base_prefix = _parse_s3_uri(self.config.uri)
        remote_prefix = f"{base_prefix}/{name}.lance" if base_prefix else f"{name}.lance"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_uri = os.path.join(tmpdir, "lancedb")
            db_local = lancedb.connect(local_uri)

            try:
                tbl_local = db_local.create_table(name, data=rows, mode=mode, primary_key="id")
            except TypeError:
                tbl_local = db_local.create_table(name, data=rows, mode=mode)

            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(tbl_local, rows_count=len(rows), dim=dim)

            s3_cls = _resolve_minio_client_cls()
            s3 = s3_cls()
            if mode == "overwrite":
                s3.delete_prefix(bucket, remote_prefix)

            local_table_path = os.path.join(local_uri, f"{name}.lance")
            uploaded = s3.upload_dir(local_table_path, bucket, remote_prefix)
            print(f"Uploaded {uploaded} files to s3://{bucket}/{remote_prefix}")

        self.connect(self.config.uri, self.config.storage_options())

        last_err = None
        for _ in range(8):
            try:
                return self._db.open_table(name)
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        try:
            try:
                tbl_remote = self._db.create_table(name, data=rows, mode="create", primary_key="id")
            except TypeError:
                tbl_remote = self._db.create_table(name, data=rows, mode="create")
            if len(rows) >= self.MIN_ROWS_FOR_INDEX:
                dim = _infer_dim_from_rows(rows)
                self._apply_index_to_table(tbl_remote, rows_count=len(rows), dim=dim)
            return tbl_remote
        except Exception as exc_remote:
            raise VectorDBError(
                "Uploaded dataset to S3 and attempted to open/register it, but failed. "
                f"Uploaded prefix: s3://{bucket}/{remote_prefix}. Last open error: {last_err}. "
                f"Remote create error: {exc_remote}"
            ) from exc_remote

    def create_empty_table(self, name: str, schema: pa.Schema, mode: str = "create") -> Any:
        self._ensure_connected()
        kwargs = {"name": name, "schema": schema, "mode": mode}
        try:
            if "id" in schema.names:
                return self._db.create_table(primary_key="id", **kwargs)
        except TypeError:
            pass
        except Exception as exc:
            raise VectorDBError(f"Failed to create empty table '{name}': {exc}") from exc
        return self._db.create_table(**kwargs)

    def get_table(self, name: str) -> Any:
        self._ensure_connected()
        try:
            return self._db.open_table(name)
        except Exception as exc:
            raise VectorDBError(f"Failed to open table '{name}': {exc}") from exc

    def list_tables(self) -> list[str]:
        """Return the list of table names available in the connected LanceDB."""
        self._ensure_connected()
        try:
            names = self._db.table_names()
        except AttributeError:
            try:
                names = self._db.list_tables()
            except AttributeError:
                names = []
        return list(names)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def _apply_index_to_table(
        self,
        table_obj: Any,
        rows_count: int,
        dim: Optional[int],
        *,
        metric: Optional[str] = None,
        num_bits: Optional[int] = None,
        use_opq: Optional[bool] = None,
        max_train_rows: Optional[int] = None,
    ) -> None:
        if rows_count < self.MIN_ROWS_FOR_INDEX:
            print(f"Skipping index creation: rows={rows_count} < {self.MIN_ROWS_FOR_INDEX}")
            return
        num_partitions, m = _choose_index_params(dim, rows_count)

        metric_effective = (metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower()
        kwargs: dict[str, Any] = {
            "metric": metric_effective,
            "num_partitions": num_partitions,
            "num_sub_vectors": m,
            "replace": True,
            "vector_column_name": "embedding",
        }

        try:
            if num_bits is None:
                env_bits = os.getenv("INDEX_NUM_BITS")
                num_bits = int(env_bits) if env_bits else None
        except Exception:
            num_bits = None
        if num_bits is not None:
            kwargs["num_bits"] = int(num_bits)

        try:
            if use_opq is None:
                env_opq = os.getenv("INDEX_USE_OPQ")
                if env_opq is not None:
                    use_opq = str(env_opq).strip().lower() in ("1", "true", "yes")
        except Exception:
            use_opq = None
        if use_opq is not None:
            kwargs["use_opq"] = bool(use_opq)

        try:
            if max_train_rows is None:
                env_train = os.getenv("INDEX_MAX_TRAIN_ROWS")
                max_train_rows = int(env_train) if env_train else None
        except Exception:
            max_train_rows = None
        if max_train_rows is not None:
            kwargs["max_train_rows"] = int(max_train_rows)

        try:
            table_obj.create_index(**kwargs)
        except TypeError:
            table_obj.create_index(
                metric=metric_effective,
                num_partitions=num_partitions,
                num_sub_vectors=m,
                replace=True,
                vector_column_name="embedding",
            )

    def create_index(
        self,
        table_name: str,
        num_partitions: Optional[int] = None,
        num_sub_vectors: Optional[int] = None,
        *,
        metric: Optional[str] = None,
        num_bits: Optional[int] = None,
        use_opq: Optional[bool] = None,
        max_train_rows: Optional[int] = None,
        min_rows_for_index: Optional[int] = None,
    ) -> None:
        table = self.get_table(table_name)

        try:
            rows_count = table.count_rows()
        except AttributeError:
            rows_count = len(table.to_pandas())

        dim: Optional[int] = None
        try:
            arr = table.to_arrow().column("embedding")
            if len(arr) > 0:
                first = arr[0].as_py()
                if isinstance(first, (list, tuple)):
                    dim = len(first)
        except Exception:
            df = table.to_pandas().head(1)
            if not df.empty and isinstance(df.iloc[0].get("embedding"), (list, tuple)):
                dim = len(df.iloc[0]["embedding"])

        if isinstance(min_rows_for_index, int) and min_rows_for_index > 0:
            self.MIN_ROWS_FOR_INDEX = min_rows_for_index

        if num_partitions is not None or num_sub_vectors is not None:
            nparts, m = _choose_index_params(dim, rows_count)
            if num_partitions is not None:
                nparts = int(num_partitions)
            if num_sub_vectors is not None:
                m = int(num_sub_vectors)
            try:
                table.create_index(
                    metric=(metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower(),
                    num_partitions=nparts,
                    num_sub_vectors=m,
                    replace=True,
                    vector_column_name="embedding",
                )
            except TypeError:
                table.create_index(
                    metric=(metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower(),
                    num_partitions=nparts,
                    num_sub_vectors=m,
                    replace=True,
                    vector_column_name="embedding",
                )
            return

        self._apply_index_to_table(
            table,
            rows_count,
            dim,
            metric=metric,
            num_bits=num_bits,
            use_opq=use_opq,
            max_train_rows=max_train_rows,
        )
