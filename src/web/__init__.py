"""Purpose: expose the FastAPI app for ASGI servers.
Why extend: register additional routers or dependencies while keeping the import path stable.
How extend: import new routers/middleware from `src.web` submodules and append them to `__all__` or mount them on `app`.
"""
from __future__ import annotations

from src.web.app import app

__all__ = ["app"]
