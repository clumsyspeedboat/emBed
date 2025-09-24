"""Purpose: canonical CLI entry point (`python -m src.main`).
Why extend: hook additional startup behaviour (logging, profiling) while keeping the CLI module importable.
How extend: customise `run()` or wrap the import from `src.cli` with extra guards, e.g. initialise Sentry before delegating to `src.cli.run()`.
"""

from __future__ import annotations

from src.cli import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
