import typer
import pytest

from lobster.cli_internal.commands.heavy import slash_commands


class _StubCommandClient:
    def __init__(self, workspace_path, session_id=None):
        self.workspace_path = workspace_path
        self.session_id = session_id
        self.data_manager = object()


class _StubConsoleOutputAdapter:
    def __init__(self, console):
        self.console = console
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((message, style))


class _StubJsonOutputAdapter:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((message, style))

    def to_dict(self):
        return {"messages": self.messages}


def test_command_cmd_impl_allows_no_session_when_none_requested(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.CommandClient",
        lambda workspace_path, session_id=None: captured.setdefault(
            "client", _StubCommandClient(workspace_path, session_id)
        ),
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands.ConsoleOutputAdapter",
        _StubConsoleOutputAdapter,
    )
    monkeypatch.setattr(
        slash_commands,
        "_dispatch_command",
        lambda cmd_str, client, output: None,
    )

    slash_commands.command_cmd_impl(
        cmd="status",
        workspace=tmp_path,
        session_id=None,
        json_output=False,
    )

    assert captured["client"].session_id is None


def test_command_cmd_impl_latest_requires_existing_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.CommandClient",
        _StubCommandClient,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands.ConsoleOutputAdapter",
        _StubConsoleOutputAdapter,
    )

    with pytest.raises(typer.Exit) as exc_info:
        slash_commands.command_cmd_impl(
            cmd="status",
            workspace=tmp_path,
            session_id="latest",
            json_output=False,
        )

    assert exc_info.value.exit_code == 1


def test_command_cmd_impl_does_not_wrap_typer_exit_in_json_mode(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.CommandClient",
        _StubCommandClient,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.JsonOutputAdapter",
        _StubJsonOutputAdapter,
    )

    def _raise_exit(cmd_str, client, output):
        raise typer.Exit(7)

    monkeypatch.setattr(
        slash_commands,
        "_dispatch_command",
        _raise_exit,
    )

    with pytest.raises(typer.Exit) as exc_info:
        slash_commands.command_cmd_impl(
            cmd="status",
            workspace=tmp_path,
            session_id="session_abc",
            json_output=True,
        )

    assert exc_info.value.exit_code == 7
    assert capsys.readouterr().out == ""


def test_command_cmd_impl_empty_command_returns_json_error(capsys):
    with pytest.raises(typer.Exit) as exc_info:
        slash_commands.command_cmd_impl(
            cmd="",
            workspace=None,
            session_id=None,
            json_output=True,
        )

    assert exc_info.value.exit_code == 1
    assert '"error": "No command specified."' in capsys.readouterr().out


def test_metadata_command_impl_clear_uses_adapter_messages(monkeypatch, tmp_path):
    captured = {}

    class _CapturingConsoleOutputAdapter(_StubConsoleOutputAdapter):
        def __init__(self, console):
            super().__init__(console)
            captured["output"] = self

    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=False: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.core.client.AgentClient",
        lambda workspace_path: object(),
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.ConsoleOutputAdapter",
        _CapturingConsoleOutputAdapter,
    )

    slash_commands.metadata_command_impl(subcommand="clear", workspace=tmp_path)

    assert captured["output"].messages == [
        ("Usage: lobster metadata clear [exports|all]", "warning"),
        (
            "  lobster metadata clear         # Clear metadata (memory + workspace/metadata/)",
            "info",
        ),
        (
            "  lobster metadata clear exports # Clear export files only",
            "info",
        ),
        (
            "  lobster metadata clear all     # Clear ALL metadata",
            "info",
        ),
    ]
