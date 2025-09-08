from __future__ import annotations
from typing import Any, Iterable
import lancedb
import pyarrow as pa
from src.config import LanceDBConfig
from src.exceptions import VectorDBError

class LanceDBManager:
    """Thin wrapper around LanceDB connection and table lifecycle."""

    def __init__(self, config: LanceDBConfig):
        try:
            self._db = lancedb.connect(
                uri=config.uri,
                storage_options=config.storage_options(),
            )
        except Exception as e:
            raise VectorDBError(f"Failed to connect to LanceDB: {e}") from e

    def create_table(
        self,
        name: str,
        data: Iterable[dict[str, Any]],
        mode: str = "overwrite",
    ) -> Any:
        """Create table from records and enforce PK on 'id' when supported."""
        try:
            return self._db.create_table(
                name,
                data=data,
                mode=mode,
                primary_key="id",
            )
        except TypeError:
            # Older lancedb without primary_key argument
            return self._db.create_table(
                name,
                data=data,
                mode=mode,
            )
        except Exception as e:
            raise VectorDBError(f"Failed to create table '{name}': {e}") from e

    def create_empty_table(self, name: str, schema: pa.Schema, mode: str = "create") -> Any:
        """Create empty table with a given schema. PK enforced if 'id' exists."""
        kwargs = {"name": name, "schema": schema, "mode": mode}
        try:
            if "id" in schema.names:
                return self._db.create_table(primary_key="id", **kwargs)
        except TypeError:
            # primary_key not supported
            pass
        except Exception as e:
            raise VectorDBError(f"Failed to create empty table '{name}': {e}") from e
        return self._db.create_table(**kwargs)

    def get_table(self, name: str) -> Any:
        """Open an existing table by name."""
        try:
            return self._db.open_table(name)
        except Exception as e:
            raise VectorDBError(f"Failed to open table '{name}': {e}") from e
    
    def create_index(self, table_name: str, num_partitions=256, num_sub_vectors=96):
        """Creates an IVF-PQ index for the table."""
        print(f"Creating index for table '{table_name}'...")
        tbl = self.get_table(table_name)
        
        # The PQ stage requires a minimum number of rows (default 256) to train.
        # For small tables, an index is not necessary and cannot be built.
        MIN_ROWS_FOR_INDEX = 256 
        if tbl.count_rows() < MIN_ROWS_FOR_INDEX:
            print(f"  Warning: Table '{table_name}' has only {tbl.count_rows()} rows.")
            print(f"  Skipping index creation as it requires at least {MIN_ROWS_FOR_INDEX} rows.")
            return

        # Dynamically adjust for small tables that are large enough for PQ
        actual_num_partitions = num_partitions
        if tbl.count_rows() < num_partitions:
            actual_num_partitions = max(1, tbl.count_rows() // 2)
            print(f"  Warning: Dataset is too small for {num_partitions} partitions.")
            print(f"  Adjusting num_partitions to {actual_num_partitions}.")

        # This is the core command to build the index
        tbl.create_index(
            metric="l2", # Or "cosine"
            num_partitions=actual_num_partitions, # How many coarse clusters to create
            num_sub_vectors=num_sub_vectors, # How many fine-grained clusters within each coarse cluster
            replace=True,
            vector_column_name="embedding" 
        )
        print("Index creation complete.")

