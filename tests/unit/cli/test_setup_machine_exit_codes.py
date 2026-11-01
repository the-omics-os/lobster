"""Machine CLI error exits, without model, install, or kernel calls."""

import importlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.no_auto_config


@pytest.fixture(autouse=True)
def isolated_command(monkeypatch, tmp_path):
    # The circular-import guard removes CLI modules from sys.modules. Patch the
    # call-time objects used by the real Typer entry, not collected stale refs.
    session_infra = importlib.import_module(
        "lobster.cli_internal.commands.heavy.session_infra"
    )
    slash_commands = importlib.import_module(
        "lobster.cli_internal.commands.heavy.slash_commands"
    )
    output_adapter = importlib.import_module(
        "lobster.cli_internal.commands.output_adapter"
    )
    workspace = importlib.import_module("lobster.core.workspace")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(slash_commands.console, "_file", slash_commands.console._file)
    manager = SimpleNamespace(
        provenance=SimpleNamespace(activities=[object()]),
        list_notebooks=lambda: [
            {"name": "workflow", "n_steps": 1, "filename": "workflow.ipynb"}
        ],
    )
    client = SimpleNamespace(workspace_path=tmp_path, data_manager=manager)
    monkeypatch.setattr(session_infra, "CommandClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        workspace, "resolve_workspace", lambda *args, **kwargs: tmp_path
    )
    return output_adapter


def invoke(command, json_output):
    from lobster.cli import app

    return CliRunner().invoke(
        app, ["command", command, *(["--json"] if json_output else [])]
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_unknown_machine_command_exits_nonzero(json_output):
    result = invoke("definitely-unknown", json_output)
    assert result.exit_code == 1, result.output
    if json_output:
        assert json.loads(result.stdout)["success"] is False
    else:
        assert "Unknown command" in result.output


@pytest.mark.parametrize("json_output", [False, True])
def test_machine_pipeline_eof_is_not_success(
    json_output, monkeypatch, isolated_command
):
    def eof(*args, **kwargs):
        raise EOFError("synthetic stdin EOF")

    monkeypatch.setattr(isolated_command.ConsoleOutputAdapter, "prompt", eof)
    monkeypatch.setattr(isolated_command.JsonOutputAdapter, "prompt", eof)
    result = invoke("pipeline run", json_output)
    assert "synthetic stdin EOF" in result.output
    assert result.exit_code == 1, result.output
    if json_output:
        envelope = json.loads(result.stdout)
        assert envelope["success"] is False
        assert "Execution error:" in envelope["error"]


@pytest.mark.parametrize(
    "summary,expected_exit",
    [
        ("Notebook executed successfully in 0.1s", 0),
        ("Execution failed: synthetic failure", 1),
        ("Notebook not found: synthetic file", 1),
    ],
)
def test_machine_pipeline_classifies_shared_helper_summary(
    summary, expected_exit, monkeypatch
):
    commands = importlib.import_module("lobster.cli_internal.commands")
    monkeypatch.setattr(commands, "pipeline_run", lambda *args, **kwargs: summary)
    result = invoke("pipeline run", True)
    assert result.exit_code == expected_exit, result.output
    envelope = json.loads(result.stdout)
    assert envelope["success"] is (expected_exit == 0)
    assert envelope.get("summary") == (summary if expected_exit == 0 else None)
    if expected_exit:
        assert envelope["error"] == summary


@pytest.mark.parametrize("json_output", [False, True])
def test_valid_machine_help_remains_success(json_output):
    result = invoke("help", json_output)
    assert result.exit_code == 0, result.output
    if json_output:
        assert json.loads(result.stdout)["success"] is True
