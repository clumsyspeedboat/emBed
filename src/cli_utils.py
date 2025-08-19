"""Small helpers for clean CLI output."""
from __future__ import annotations

import os
import textwrap
from typing import Iterable, Mapping, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _RICH = True
    _console = Console()
except Exception:
    _RICH = False
    _console = None


def mask(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def echo(msg: str) -> None:
    if _RICH:
        _console.print(msg)
    else:
        print(msg)


def heading(title: str) -> None:
    if _RICH:
        _console.rule(f"[bold]{title}")
    else:
        print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def kv_table(title: str, rows: Mapping[str, str]) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY)
        t.add_column("Key", style="bold cyan", no_wrap=True)
        t.add_column("Value", style="white")
        for k, v in rows.items():
            t.add_row(str(k), str(v))
        _console.print(t)
    else:
        print(f"\n{title}:")
        for k, v in rows.items():
            print(f"  - {k}: {v}")


def simple_table(title: str, columns: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY)
        for c in columns:
            t.add_column(str(c))
        for r in rows:
            t.add_row(*[str(x) for x in r])
        _console.print(t)
    else:
        print(f"\n{title}:")
        print("\t".join(columns))
        for r in rows:
            print("\t".join(str(x) for x in r))


def panel(text: str, title: str = "") -> None:
    if _RICH:
        _console.print(Panel.fit(textwrap.dedent(text).strip(), title=title))
    else:
        if title:
            print(f"\n[{title}]")
        print(textwrap.dedent(text).strip())
