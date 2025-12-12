"""Purpose: expose the high-level public API for configuration and search helpers.
Why extend: add new convenience exports so downstream code can import from `src` without deep knowledge of subpackages.
How extend: import additional classes/functions here once they stabilise, e.g. `from src.core.search import MultiModalSearcherV2` and append to `__all__`.
"""

from .config import MinioConfig, LanceDBConfig
from .lancedb_manager import LanceDBManager
from .search import MultiModalSearcher
__all__ = [
    "MinioConfig",
    "LanceDBConfig",
    "LanceDBManager",
    "MultiModalSearcher",
]
