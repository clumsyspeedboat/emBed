"""Purpose: namespace for runtime primitives that power the CLI and web surfaces.
Why extend: group new cross-cutting helpers (search, ranking, routing) without crowding the top-level package.
How extend: add new modules (e.g. `rerank.py`) and expose their key classes/functions here for convenient imports.
"""

from src.core.search import MultiModalSearcher
from src.core.services import ServiceRegistry, ServiceContainer, default_container

__all__ = ["MultiModalSearcher", "ServiceRegistry", "ServiceContainer", "default_container"]
