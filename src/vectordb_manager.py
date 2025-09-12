from abc import ABC, abstractmethod
from typing import Any, Iterable
import pyarrow as pa

class VectorDBManager(ABC):
    @abstractmethod
    def connect(self, uri: str, storage_options: dict[str, Any]) -> None:
        """Connect to the vector database and store connection in instance."""
        pass

    @abstractmethod
    def create_table(self, name: str, data: Iterable[dict[str, Any]], mode: str = "overwrite") -> Any:
        pass

    @abstractmethod
    def create_empty_table(self, name: str, schema: pa.Schema, mode: str = "create") -> Any:
        pass

    @abstractmethod
    def get_table(self, name: str) -> Any:
        pass

    @abstractmethod
    def create_index(self, table_name: str, num_partitions: int = 256, num_sub_vectors: int = 96) -> None:
        """Create an index based on the manager's indexing policy."""
        pass
