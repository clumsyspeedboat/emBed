"""Purpose: compatibility wrapper exposing the FastAPI app at the historical module path.
Why extend: advertise additional FastAPI routers or life-cycle hooks while the main implementation lives in `src.web`.
How extend: import new callables from `src.web` (e.g. `from src.web.router import api_router`) and re-export them here once they stabilise.
"""

from __future__ import annotations

from src.web.app import app

__all__ = ["app"]
