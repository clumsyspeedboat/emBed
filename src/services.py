"""Purpose: compatibility shim exposing the service container from `src.core.services`.
Why extend: keep legacy imports (`from src import services`) functioning while new code targets `src.core`.
How extend: re-export additional helpers from `src.core.services` as they are added.
"""
from __future__ import annotations

from src.core.services import ServiceContainer, ServiceRegistry, default_container
from src.storage import MinIOClient
from src.vectordb import LanceDBManager

__all__ = ["ServiceRegistry", "ServiceContainer", "default_container", "MinIOClient", "LanceDBManager"]
