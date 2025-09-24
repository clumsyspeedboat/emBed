"""Purpose: declare the abstract interface all vector DB managers must implement.
Why extend: capture new capabilities (e.g. streaming inserts) that every manager should provide.
How extend: add abstract methods or refine signatures, then update each concrete manager and its tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import pyarrow as pa

__all__ = ["VectorDBManager"]


class VectorDBManager(ABC):
    @abstractmethod
    def connect(self, uri: str, storage_options: dict[str, Any]) -> None:
        """Connect to the vector database and store connection in the instance."""
        raise NotImplementedError

    @abstractmethod
    def create_table(self, name: str, data: Iterable[dict[str, Any]], mode: str = "overwrite") -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_empty_table(self, name: str, schema: pa.Schema, mode: str = "create") -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_table(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_index(
        self,
        table_name: str,
        num_partitions: int = 256,
        num_sub_vectors: int = 96,
        **kwargs: Any,
    ) -> None:
        """Create an index based on the manager's indexing policy."""
        raise NotImplementedError
