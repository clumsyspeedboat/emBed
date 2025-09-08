from abc import ABC, abstractmethod
from typing import Any, Iterable
import pyarrow as pa

class VectorDBManager(ABC):
    @abstractmethod
    def connect(self, uri: str, storage_options: dict) -> None:
        """Connect to the vector database."""
        pass

    @abstractmethod
    def create_table(self, name: str, data: Iterable[dict[str, Any]], mode: str = "overwrite") -> Any:
        """Create a table with data."""
        pass

    @abstractmethod
    def create_empty_table(self, name: str, schema: pa.Schema, mode: str = "create") -> Any:
        """Create an empty table with a given schema."""
        pass

    @abstractmethod
    def get_table(self, name: str) -> Any:
        """Open an existing table by name."""
        pass