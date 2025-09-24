"""Purpose: expose the CLI entrypoints (`main`/`run`) as an import-friendly interface.
Why extend: allow programmatic invocation from notebooks or orchestration scripts.
How extend: wrap `main` with logging/telemetry or surface additional convenience functions used by external automation.
"""
from __future__ import annotations

from src.cli.app import main

__all__ = ["main", "run"]


def run() -> None:
    """Execute the CLI dispatcher."""
    main()
