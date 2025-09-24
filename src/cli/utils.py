"""Purpose: provide rich/console-friendly helpers reused across CLI commands.
Why extend: support additional output patterns (spinners, tables) without cluttering command functions.
How extend: add new helper functions or context managers here and exercise them via CLI commands to ensure consistent UX.
"""
from __future__ import annotations

import json
import os
import textwrap
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional, Sequence

_DISABLE_RICH = os.getenv("FORCE_PLAIN_CLI") or os.getenv("NO_COLOR")
try:
    if _DISABLE_RICH:
        raise RuntimeError("Rich disabled by env")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.progress import track as _rich_track
    _RICH = True
    _console = Console()
except Exception:
    _RICH = False
    _console = None
    box = None  # type: ignore

__all__ = [
    "mask",
    "format_bytes",
    "echo",
    "info",
    "warn",
    "error",
    "success",
    "heading",
    "panel",
    "kv_table",
    "simple_table",
    "jprint",
    "status",
    "track",
    "env_snapshot",
]


def mask(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def format_bytes(num_bytes: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < step or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= step
    return f"{num_bytes}B"


def echo(message: str = "") -> None:
    if _RICH:
        _console.print(message)
    else:
        print(message)


def info(message: str) -> None:
    echo(f"[bold cyan]INFO[/]: {message}" if _RICH else f"INFO: {message}")


def warn(message: str) -> None:
    echo(f"[bold yellow]WARN[/]: {message}" if _RICH else f"WARN: {message}")


def error(message: str) -> None:
    echo(f"[bold red]ERROR[/]: {message}" if _RICH else f"ERROR: {message}")


def success(message: str) -> None:
    echo(f"[bold green]OK[/]: {message}" if _RICH else f"OK: {message}")


def heading(title: str) -> None:
    if _RICH:
        _console.rule(f"[bold]{title}")
    else:
        print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def panel(text: str, title: str = "") -> None:
    body = textwrap.dedent(text).strip()
    if _RICH:
        _console.print(Panel.fit(body, title=title))
    else:
        if title:
            print(f"\n[{title}]")
        print(body)


def kv_table(title: str, rows: Mapping[str, Any]) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY)  # type: ignore[arg-type]
        t.add_column("Key", style="bold cyan", no_wrap=True)
        t.add_column("Value", style="white")
        for key, value in rows.items():
            t.add_row(str(key), _stringify(value))
        _console.print(t)
    else:
        print(f"\n{title}:")
        for key, value in rows.items():
            print(f"  - {key}: {_stringify(value)}")


def simple_table(title: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY)  # type: ignore[arg-type]
        for column in columns:
            t.add_column(str(column))
        for row in rows:
            t.add_row(*[_stringify(value) for value in row])
        _console.print(t)
    else:
        print(f"\n{title}:")
        cols = list(columns)
        print("\t".join(cols))
        for row in rows:
            print("\t".join(_stringify(value) for value in row))


def jprint(obj: Any, title: str = "") -> None:
    rendered = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    if title:
        heading(title)
    echo(rendered)


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


@contextmanager
def status(message: str):
    if _RICH:
        with _console.status(message):
            yield
    else:
        print(f"{message} ...")
        yield


def track(iterable: Iterable[Any], total: Optional[int] = None, description: str = "") -> Iterable[Any]:
    if _RICH:
        return _rich_track(iterable, total=total, description=description)
    return iterable


_SECRET_HINTS: Sequence[str] = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PWD",
    "PASSWORD",
    "CREDENTIAL",
    "SESSION",
    "ACCESS",
)


def env_snapshot(keys: Iterable[str], title: str = "Environment") -> None:
    rows: dict[str, str] = {}
    for key in keys:
        value = os.getenv(key, "")
        if any(hint in key.upper() for hint in _SECRET_HINTS):
            rows[key] = mask(value)
        else:
            rows[key] = value
    kv_table(title, rows)
