# src/cli_utils.py
"""Small helpers for clean CLI output (Rich optional).

- Rich auto-detect (disable with env: NO_COLOR=1 or FORCE_PLAIN_CLI=1)
- Safe masking for secrets
- echo()/info()/warn()/error()/success() unified printing
- heading(), panel(), kv_table(), simple_table()
- status() context manager (spinner with Rich; no-op fallback)
- track() progress wrapper (Rich track fallback to plain iterable)
- jprint() pretty JSON, format_bytes()
- env_snapshot() to display selected env vars with masking
"""
from __future__ import annotations

import json
import os
import textwrap
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional, Sequence

# ---------- Rich detection ----------
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


# ---------- Core formatting ----------
def mask(s: Optional[str]) -> str:
    """Mask secret-ish strings but keep short context (ab..yz)."""
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def format_bytes(n: int) -> str:
    """Human friendly bytes."""
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < step or u == units[-1]:
            return f"{v:.1f}{u}"
        v /= step
    return f"{n}B"


# ---------- Printing helpers ----------
def echo(msg: str = "") -> None:
    if _RICH:
        _console.print(msg)
    else:
        print(msg)


def info(msg: str) -> None:
    echo(f"[bold cyan]INFO[/]: {msg}" if _RICH else f"INFO: {msg}")


def warn(msg: str) -> None:
    echo(f"[bold yellow]WARN[/]: {msg}" if _RICH else f"WARN: {msg}")


def error(msg: str) -> None:
    echo(f"[bold red]ERROR[/]: {msg}" if _RICH else f"ERROR: {msg}")


def success(msg: str) -> None:
    echo(f"[bold green]OK[/]: {msg}" if _RICH else f"OK: {msg}")


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
        for k, v in rows.items():
            t.add_row(str(k), _stringify(v))
        _console.print(t)
    else:
        print(f"\n{title}:")
        for k, v in rows.items():
            print(f"  - {k}: {_stringify(v)}")


def simple_table(title: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY)  # type: ignore[arg-type]
        for c in columns:
            t.add_column(str(c))
        for r in rows:
            t.add_row(*[ _stringify(x) for x in r ])
        _console.print(t)
    else:
        print(f"\n{title}:")
        cols = list(columns)
        print("\t".join(cols))
        for r in rows:
            print("\t".join(_stringify(x) for x in r))


def jprint(obj: Any, title: str = "") -> None:
    """Pretty-print JSON-like structures."""
    s = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    if title:
        heading(title)
    echo(s)


def _stringify(x: Any) -> str:
    if isinstance(x, (dict, list, tuple)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


# ---------- Progress & status ----------
@contextmanager
def status(message: str):
    """Display transient status/spinner (Rich) or simple message (plain)."""
    if _RICH:
        with _console.status(message):
            yield
    else:
        print(f"{message} ...")
        yield


def track(iterable: Iterable[Any], total: Optional[int] = None, description: str = "") -> Iterable[Any]:
    """Progress wrapper around an iterable."""
    if _RICH:
        return _rich_track(iterable, total=total, description=description)
    return iterable


# ---------- Env snapshot ----------
_SECRET_HINTS: Sequence[str] = (
    "KEY", "SECRET", "TOKEN", "PWD", "PASSWORD", "CREDENTIAL", "SESSION", "ACCESS"
)

def env_snapshot(keys: Iterable[str], title: str = "Environment") -> None:
    """Print selected env vars; mask if they look sensitive."""
    rows: dict[str, str] = {}
    for k in keys:
        v = os.getenv(k, "")
        if any(h in k.upper() for h in _SECRET_HINTS):
            rows[k] = mask(v)
        else:
            rows[k] = v
    kv_table(title, rows)
