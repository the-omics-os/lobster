"""A bad LOBSTER_RESPONSE_STYLE is one clean line and exit code 2 at every CLI entry."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal import response_style

BAD = "LOBSTER_RESPONSE_STYLE='bogus' is not valid; expected one of chat, run"


@pytest.mark.parametrize("value", [None, "", "  ", "chat", "run", "RUN", " Chat "])
def test_valid_or_unset_values_pass(monkeypatch, value):
    monkeypatch.delenv("LOBSTER_RESPONSE_STYLE", raising=False)
    if value is not None:
        monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", value)
    response_style.check_env()


def test_a_bad_value_names_the_variable_the_value_and_the_valid_values(
    monkeypatch, capsys
):
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "bogus")
    with pytest.raises(typer.Exit) as exit_:
        response_style.check_env()
    assert exit_.value.exit_code == 2
    out, err = capsys.readouterr()
    assert out == "" and err == f"lobster: {BAD}\n"


def test_forced_sets_the_style_for_the_block_and_puts_the_old_value_back(monkeypatch):
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "bogus")
    with response_style.forced("run"):
        assert os.environ["LOBSTER_RESPONSE_STYLE"] == "run"
    assert os.environ["LOBSTER_RESPONSE_STYLE"] == "bogus"
    monkeypatch.delenv("LOBSTER_RESPONSE_STYLE")
    with pytest.raises(RuntimeError):
        with response_style.forced("run"):
            raise RuntimeError("x")
    assert "LOBSTER_RESPONSE_STYLE" not in os.environ


@pytest.mark.parametrize(
    "argv",
    [
        ["query", "hi"],
        ["chat"],
        ["config", "show"],
        ["agent", "--stdio"],
        ["mcp", "--stdio"],
    ],
)
def test_every_entry_fails_cleanly(monkeypatch, argv):
    from lobster.cli import app

    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "bogus")
    monkeypatch.setenv("LOBSTER_SKIP_UPDATE_CHECK", "1")
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert result.stderr == f"lobster: {BAD}\n"
    assert "Traceback" not in result.stderr


def test_mcp_server_start_fails_on_stderr_and_leaves_stdout_untouched():
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LOBSTER_RESPONSE_STYLE": "bogus",
    }
    proc = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "mcp", "--stdio"],
        input='{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr.strip().splitlines()[-1] == f"lobster: {BAD}"
    assert "Traceback" not in proc.stderr
