"""Purpose: house lightweight helpers (quote stripping, boolean parsing) used by config dataclasses.
Why extend: avoid duplicating parsing logic when adding new configuration types.
How extend: add helper functions and import them within config modules that need the shared behaviour.
"""
from __future__ import annotations

__all__ = ["strip_quotes", "as_bool"]


def strip_quotes(value: str | None) -> str | None:
    """Remove matching single/double quotes around a string value."""
    if not value:
        return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def as_bool(value: str | None, default: bool = False) -> bool:
    """Interpret common string representations as booleans."""
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}
