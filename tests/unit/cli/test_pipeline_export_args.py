"""`pipeline export` argument handling in chat and in `lobster command`.

Covers: a mistyped option is refused instead of becoming the notebook name;
an existing notebook is not replaced without --force; `lobster command`
passes its arguments through and exits nonzero when the export fails; the
chat path keeps the name's capitalization.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import typer

from lobster.cli_internal.commands.heavy import slash_commands
from lobster.cli_internal.commands.light.pipeline_commands import (
    parse_export_args,
    pipeline_export,
)


@pytest.fixture(autouse=True)
def preserve_shared_console_target(monkeypatch):
    # command_cmd_impl's file getter/setter round-trip can pin pytest's
    # temporary capture stream. Restore Rich's underlying dynamic target at
    # the test boundary so export tests do not contaminate later setup tests.
    monkeypatch.setattr(slash_commands.console, "_file", slash_commands.console._file)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], (None, None, "successful", False)),
        (["nb"], ("nb", None, "successful", False)),
        (["nb", "my", "run"], ("nb", "my run", "successful", False)),
        (["--replay", "nb"], ("nb", None, "replay", False)),
        (["nb", "--force", "x"], ("nb", "x", "successful", True)),
        (["nb", "-", "x"], ("nb", "- x", "successful", False)),
    ],
)
def test_parse_export_args(args, expected):
    assert parse_export_args(args) == expected


@pytest.mark.parametrize("bad", ["--foo", "--Replay", "-f"])
def test_parse_export_args_refuses_unknown_options(bad):
    with pytest.raises(ValueError, match="Unknown option"):
        parse_export_args(["nb", bad])


def _client(tmp_path):
    client = Mock()
    client.data_manager.workspace_path = tmp_path
    client.data_manager.export_notebook.return_value = tmp_path / "nb.ipynb"
    return client


def test_chat_refuses_unknown_option_instead_of_using_it_as_name():
    with patch.object(slash_commands, "pipeline_export") as export:
        summary = slash_commands._execute_command(
            "/pipeline export --foo", Mock(), output=Mock()
        )
    export.assert_not_called()
    assert summary.startswith("Export failed: Unknown option: --foo")


def test_chat_keeps_the_name_case():
    with patch.object(slash_commands, "pipeline_export") as export:
        slash_commands._execute_command(
            "/pipeline export my_nb first run",
            Mock(),
            original_command="/pipeline export My_NB First run",
            output=Mock(),
        )
    args, kwargs = export.call_args
    assert args[2:] == ("My_NB", "First run")
    assert kwargs == {"mode": "successful", "force": False}


def test_export_refuses_to_replace_an_existing_notebook(tmp_path):
    (tmp_path / "notebooks").mkdir()
    existing = tmp_path / "notebooks" / "nb.ipynb"
    existing.write_text("user edits")
    client = _client(tmp_path)

    summary = pipeline_export(client, Mock(), "nb", "")

    client.data_manager.export_notebook.assert_not_called()
    assert summary.startswith("Export failed: notebook already exists")
    assert existing.read_text() == "user edits"


def test_export_replaces_an_existing_notebook_with_force(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "nb.ipynb").write_text("old")
    client = _client(tmp_path)

    summary = pipeline_export(client, Mock(), "nb", "", force=True)

    client.data_manager.export_notebook.assert_called_once_with(
        "nb", "", filter_strategy="successful"
    )
    assert summary.startswith("Exported notebook:")


def _command_client(tmp_path, activities):
    dm = SimpleNamespace(
        provenance=SimpleNamespace(activities=activities),
        workspace_path=tmp_path,
        export_notebook=Mock(return_value=tmp_path / "notebooks" / "x.ipynb"),
    )
    return SimpleNamespace(data_manager=dm, workspace_path=tmp_path)


def _run_command(tmp_path, cmd, activities, json_output=False):
    client = _command_client(tmp_path, activities)
    with patch(
        "lobster.cli_internal.commands.heavy.session_infra.CommandClient",
        return_value=client,
    ):
        slash_commands.command_cmd_impl(
            cmd, workspace=tmp_path, json_output=json_output
        )
    return client


def test_command_passes_its_arguments_through(tmp_path):
    client = _run_command(
        tmp_path, "pipeline export My_NB first run --replay", [{"type": "x"}]
    )
    client.data_manager.export_notebook.assert_called_once_with(
        "My_NB", "first run", filter_strategy="replay"
    )


def test_command_keeps_defaults_without_arguments(tmp_path):
    client = _run_command(tmp_path, "pipeline export", [{"type": "x"}])
    client.data_manager.export_notebook.assert_called_once_with(
        "analysis_workflow", "", filter_strategy="successful"
    )


def test_command_without_history_exits_nonzero(tmp_path):
    with pytest.raises(typer.Exit) as exc:
        _run_command(tmp_path, "pipeline export", [])
    assert exc.value.exit_code == 1


def test_command_failed_export_exits_nonzero(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "analysis_workflow.ipynb").write_text("old")
    with pytest.raises(typer.Exit) as exc:
        _run_command(tmp_path, "pipeline export", [{"type": "x"}])
    assert exc.value.exit_code == 1


def test_command_unknown_option_exits_nonzero_with_json(tmp_path, capsys):
    with pytest.raises(typer.Exit) as exc:
        _run_command(
            tmp_path, "pipeline export --foo", [{"type": "x"}], json_output=True
        )
    assert exc.value.exit_code == 1
    envelope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert envelope["success"] is False
    assert "Unknown option: --foo" in envelope["error"]
