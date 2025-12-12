"""Purpose: lazily load the project `.env` file for any module that relies on environment configuration.
Why extend: customise env discovery or support multiple .env locations.
How extend: adjust `ensure_dotenv_loaded()` to handle new search paths or caching semantics, then call it from other config modules as needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = ["ensure_dotenv_loaded"]

_DOTENV_LOADED: bool = False
_ENV_PATH: Final[Path] = Path(__file__).resolve().parents[2] / ".env"


def ensure_dotenv_loaded() -> None:
    """Load the project .env file exactly once if python-dotenv is available."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True
        return

    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=False)
    _DOTENV_LOADED = True


# Load on import for backwards compatibility with the original module.
ensure_dotenv_loaded()
