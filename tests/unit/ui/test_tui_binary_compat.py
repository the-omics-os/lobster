"""A lobster-tui binary from the cache or PATH must match the installed lobster-ai."""

import sys
import types

import pytest

from lobster.cli_internal import tui_binary_compat
from lobster.cli_internal.go_tui_launcher import find_tui_binary_fast
from lobster.ui.bridge.binary_finder import find_tui_binary
from lobster.version import __version__

MAJOR_MINOR = ".".join(__version__.split(".")[:2])
FINDERS = [find_tui_binary, find_tui_binary_fast]


def _fake_binary(path, version_line, counter=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = f'echo x >> "{counter}"\n' if counter else ""
    path.write_text(f'#!/bin/sh\n{count}echo "{version_line}"\n')
    path.chmod(0o755)
    return str(path)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(tui_binary_compat, "_hinted", False)
    monkeypatch.setattr(tui_binary_compat, "_rejected", {})
    # Shell-script fakes start slowly on a loaded machine; the 3 s limit is for real use.
    monkeypatch.setattr(tui_binary_compat, "_PROBE_TIMEOUT", 30)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LOBSTER_TUI_BINARY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    # Source install: the wheel package is importable but ships no binary.
    monkeypatch.setitem(
        sys.modules,
        "lobster_ai_tui",
        types.SimpleNamespace(__file__=str(tmp_path / "pkg" / "__init__.py")),
    )
    return tmp_path


@pytest.mark.parametrize("finder", FINDERS)
def test_stale_cache_binary_is_skipped(finder, tmp_path):
    _fake_binary(tmp_path / "home/.cache/lobster/bin/lobster-tui", "lobster-tui 0.1.0")
    assert finder() is None


@pytest.mark.parametrize("finder", FINDERS)
def test_matching_cache_binary_is_used(finder, tmp_path):
    path = _fake_binary(
        tmp_path / "home/.cache/lobster/bin/lobster-tui",
        f"lobster-tui {MAJOR_MINOR}.9",
    )
    assert finder() == path


@pytest.mark.parametrize("finder", FINDERS)
def test_path_binary_checked_too(finder, tmp_path, monkeypatch):
    _fake_binary(tmp_path / "stale/lobster-tui", "lobster-tui 0.1.0")
    monkeypatch.setenv("PATH", str(tmp_path / "stale"))
    assert finder() is None
    path = _fake_binary(tmp_path / "good/lobster-tui", f"lobster-tui {MAJOR_MINOR}.0")
    monkeypatch.setenv("PATH", str(tmp_path / "good"))
    assert finder() == path


@pytest.mark.parametrize("finder", FINDERS)
def test_unparseable_version_output_is_skipped(finder, tmp_path):
    _fake_binary(tmp_path / "home/.cache/lobster/bin/lobster-tui", "something else")
    assert finder() is None


@pytest.mark.parametrize("finder", FINDERS)
def test_explicit_override_is_trusted_without_version_check(
    finder, tmp_path, monkeypatch
):
    path = _fake_binary(tmp_path / "custom-tui", "lobster-tui 0.1.0")
    monkeypatch.setenv("LOBSTER_TUI_BINARY", path)
    assert finder() == path


def test_version_check_runs_once_per_path(tmp_path):
    counter = tmp_path / "calls"
    _fake_binary(
        tmp_path / "home/.cache/lobster/bin/lobster-tui", "lobster-tui 0.1.0", counter
    )
    find_tui_binary()
    find_tui_binary_fast()
    assert counter.read_text().count("x") == 1


def test_hint_printed_once_when_only_stale_binary_exists(tmp_path, capsys):
    _fake_binary(tmp_path / "home/.cache/lobster/bin/lobster-tui", "lobster-tui 0.1.0")
    find_tui_binary()
    find_tui_binary_fast()
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and "0.1" in err
