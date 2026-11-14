"""``lobster commands --json``: the lobster.commands/1 manifest (Phase 0 B1)."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lobster.cli import app
from lobster.cli_internal.command_registry import build_manifest

SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "lobster"
    / "schemas"
    / "lobster.commands.v1.schema.json"
)


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(app)


def _by_name(manifest):
    return {c["name"]: c for c in manifest["commands"]}


def test_manifest_validates_against_schema(manifest):
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)


def test_manifest_lists_core_verbs(manifest):
    names = _by_name(manifest)
    for name in (
        "chat",
        "query",
        "agent",
        "agents list",
        "config show",
        "commands",
        "sessions list",
    ):
        assert "cli" in names[name]["surfaces"], name
    assert names["/help"]["surfaces"] == ["slash"]
    assert "apply-init-result" not in names  # hidden commands stay out
    assert manifest["agent_interface"] == "lobster agent --stdio"


def test_every_cli_handler_dispatches_to_a_callable(manifest):
    for command in manifest["commands"]:
        if "cli" not in command["surfaces"]:
            assert command["handler"] == f"slash:{command['name']}"
            continue
        module, _, qualname = command["handler"].partition(":")
        target = importlib.import_module(module)
        for part in qualname.split("."):
            target = getattr(target, part)
        assert callable(target), command["handler"]


def test_chat_args(manifest):
    chat = _by_name(manifest)["chat"]
    arg_names = {a["name"] for a in chat["args"]}
    assert {"workspace", "session-id", "provider", "model"} <= arg_names
    assert {a["name"]: a["type"] for a in chat["args"]}["workspace"] == "path"


def test_manifest_is_deterministic(manifest):
    names = [c["name"] for c in manifest["commands"]]
    assert names == sorted(names)
    assert build_manifest(app) == manifest


def test_cli_prints_manifest_json():
    result = subprocess.run(
        [sys.executable, "-m", "lobster", "commands", "--json"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema_version"] == "lobster.commands/1"


def test_manifest_is_locale_independent():
    import os
    import subprocess  # nosec B404
    import sys

    outputs = {}
    for lang in ("en", "zh"):
        env = dict(os.environ, LOBSTER_LANG=lang)
        result = subprocess.run(  # nosec B603
            [sys.executable, "-m", "lobster", "commands", "--json"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        outputs[lang] = result.stdout
    assert outputs["zh"] == outputs["en"]


def test_manifest_reexec_runs_at_most_once(monkeypatch):
    """With the marker set, a non-English UI never spawns another process."""
    import subprocess

    from typer.testing import CliRunner

    from lobster.cli import app
    from lobster.cli_internal.commands.light import manifest_commands

    def no_spawn(*args, **kwargs):
        raise AssertionError("commands --json re-ran itself twice")

    monkeypatch.setattr(subprocess, "run", no_spawn)
    monkeypatch.setattr("lobster.i18n.current_lang", lambda: "zh")
    monkeypatch.setenv(manifest_commands._REEXEC_MARKER, str(os.getppid()))
    result = CliRunner().invoke(app, ["commands", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema_version"] == "lobster.commands/1"


def test_inherited_reexec_marker_is_ignored(monkeypatch):
    """A marker not set by our parent process still re-runs in English."""
    import subprocess

    from typer.testing import CliRunner

    from lobster.cli import app
    from lobster.cli_internal.commands.light import manifest_commands

    spawned = []

    def fake_run(argv, env, **kwargs):
        spawned.append(env[manifest_commands._REEXEC_MARKER])
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("lobster.i18n.current_lang", lambda: "zh")
    # The stale marker is not a pid string: "1" would match a parent pid of 1
    # (init, as in some containers) and the test would fail there.
    monkeypatch.setattr(os, "getppid", lambda: 1)
    monkeypatch.setenv(manifest_commands._REEXEC_MARKER, "stale-marker")
    result = CliRunner().invoke(app, ["commands", "--json"])
    assert result.exit_code == 0, result.output
    assert spawned == [str(os.getpid())]


@pytest.mark.parametrize(
    "click_name, expected",
    [("integer range", "integer"), ("float range", "number")],
)
def test_bounded_numbers_keep_their_type(click_name, expected):
    # An option with min/max is a click range type; it is still a number.
    from types import SimpleNamespace

    from lobster.cli_internal.command_registry import _arg_type

    param = SimpleNamespace(
        type_descriptor=None,
        type=SimpleNamespace(name=click_name, choices=None),
        is_flag=False,
    )
    assert _arg_type(param) == (expected, None)
