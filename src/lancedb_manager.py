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


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _choose_index_params(
    dim: Optional[int], rows_count: int, default_partitions: Optional[int] = None
) -> tuple[int, int]:
    """Return (num_partitions, num_sub_vectors/m) based on data size and dim.

    Heuristic:
    - Partitions ~ 4 * sqrt(N), clamped to [512, 8192] by default (override via env).
    - m is the largest divisor of dim from the set {64, 48, 32, 24, 16, 12, 8, 4, 2, 1}.
    - Explicit overrides via env take precedence: INDEX_NUM_PARTITIONS, INDEX_M.
    """
    # explicit overrides
    env_parts = os.getenv("INDEX_NUM_PARTITIONS")
    env_m = os.getenv("INDEX_M")

    if env_parts:
        try:
            num_partitions = max(1, int(env_parts))
        except Exception:
            num_partitions = max(1, (default_partitions or 256))
    else:
        # heuristic based on dataset size
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
            for cand in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
                if dim % cand == 0:
                    m = cand
                    break

    return num_partitions, m


class LanceDBManager(VectorDBManager):
    """LanceDB manager with S3-aware create/upload and unified index policy."""

    # Minimum rows to attempt index training; override via INDEX_MIN_ROWS
    MIN_ROWS_FOR_INDEX = _get_int_env("INDEX_MIN_ROWS", 5000)

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
        """Apply IVF-PQ index with policy based on dataset size and dim."""
        if rows_count < self.MIN_ROWS_FOR_INDEX:
            print(f"Skipping index creation: rows={rows_count} < {self.MIN_ROWS_FOR_INDEX}")
            return
        num_partitions, m = _choose_index_params(dim, rows_count)

        # metric override (env or arg)
        metric_eff = (metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower()
        kwargs = dict(metric=metric_eff, num_partitions=num_partitions, num_sub_vectors=m, replace=True, vector_column_name="embedding")

        # optional params if supported by installed lancedb version
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
                use_opq = (str(env_opq).strip().lower() in ("1", "true", "yes")) if env_opq is not None else None
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
            # fallback without optional kwargs
            table_obj.create_index(
                metric=metric_eff,
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

        # Allow caller to override min rows for this call
        if isinstance(min_rows_for_index, int) and min_rows_for_index > 0:
            self.MIN_ROWS_FOR_INDEX = min_rows_for_index

        # --- If user gave explicit params, honor them directly ---
        if num_partitions is not None or num_sub_vectors is not None:
            nparts = int(num_partitions) if num_partitions is not None else _choose_index_params(dim, rows_count)[0]
            m = int(num_sub_vectors) if num_sub_vectors is not None else _choose_index_params(dim, rows_count)[1]
            try:
                tbl.create_index(
                    metric=(metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower(),
                    num_partitions=nparts,
                    num_sub_vectors=m,
                    replace=True,
                    vector_column_name="embedding",
                )
            except TypeError:
                tbl.create_index(
                    metric=(metric or os.getenv("LANCEDB_INDEX_METRIC") or "l2").strip().lower(),
                    num_partitions=nparts,
                    num_sub_vectors=m,
                    replace=True,
                    vector_column_name="embedding",
                )
            return

        # --- Default adaptive policy ---
        self._apply_index_to_table(
            tbl,
            rows_count,
            dim,
            metric=metric,
            num_bits=num_bits,
            use_opq=use_opq,
            max_train_rows=max_train_rows,
        )
