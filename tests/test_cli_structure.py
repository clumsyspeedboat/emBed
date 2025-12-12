import argparse

from src.cli.app import build_parser
from src.cli import run


def _get_subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    names: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            names.update(action.choices.keys())
    return names


def test_cli_exposes_expected_commands():
    parser = build_parser()
    commands = _get_subcommand_names(parser)
    expected = {
        "config",
        "health",
        "ls-s3",
        "ingest-s3",
        "create-index",
        "search",
        "peek",
        "stats",
        "reset",
    }
    assert expected.issubset(commands)


def test_cli_parses_minimal_invocations():
    parser = build_parser()
    args = parser.parse_args(["config"])
    assert args.cmd == "config"

    args = parser.parse_args(["search", "--query", "hello"])
    assert args.cmd == "search"
    assert args.query == "hello"


def test_cli_run_is_callable(monkeypatch):
    calls = {"n": 0}

    from src import cli as cli_module

    def fake_main():
        calls["n"] += 1

    monkeypatch.setattr(cli_module, "main", fake_main, raising=False)
    run()
    assert calls["n"] == 1
