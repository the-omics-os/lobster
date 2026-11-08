import json

import pytest
import typer

from lobster.cli_internal.commands.heavy import slash_commands


@pytest.fixture(autouse=True)
def preserve_shared_console_target(monkeypatch):
    # Rich's file getter resolves a dynamic None target to the active capture
    # stream. command_cmd_impl restores through its setter, pinning that stream.
    # Preserve the underlying target across each test, not a soon-stale getter
    # result, so a JSON-command test cannot leak capture state to later modules.
    monkeypatch.setattr(slash_commands.console, "_file", slash_commands.console._file)


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


def test_command_cmd_impl_help_skips_auto_session_resolution(
    tmp_path, monkeypatch, capsys
):
    captured = {}
    sessions_dir = tmp_path / ".lobster" / "sessions" / "session_recent"
    sessions_dir.mkdir(parents=True)

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

    slash_commands.command_cmd_impl(
        cmd="/help",
        workspace=tmp_path,
        session_id=None,
        json_output=True,
    )

    payload = json.loads(capsys.readouterr().out)

    assert captured["client"].session_id is None
    assert payload["success"] is True
    assert any(
        block.get("title") == "Core Commands"
        for block in payload["data"]["blocks"]
        if block["kind"] == "table"
    )


def test_command_cmd_impl_query_help_profile_renders_query_subset(
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

    slash_commands.command_cmd_impl(
        cmd="/help",
        workspace=tmp_path,
        session_id=None,
        json_output=True,
        help_profile="query",
    )

    payload = json.loads(capsys.readouterr().out)
    titles = [
        block.get("title")
        for block in payload["data"]["blocks"]
        if block["kind"] == "table"
    ]

    assert payload["success"] is True
    assert titles == ["Query-Compatible Commands"]
    assert all(title != "Admin & UI Commands" for title in titles)


class _Stdin:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def _route_metadata(monkeypatch, tmp_path, tty=True, **kwargs):
    """Run metadata_command_impl with every metadata view stubbed; return the calls."""
    calls = []
    monkeypatch.setattr("sys.stdin", _Stdin(tty))
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=False: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.core.client.AgentClient", lambda workspace_path: object()
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.ConsoleOutputAdapter",
        _StubConsoleOutputAdapter,
    )
    for name in (
        "metadata_overview",
        "metadata_publications",
        "metadata_samples",
        "metadata_workspace",
        "metadata_exports",
        "metadata_list",
        "metadata_clear",
        "metadata_clear_exports",
        "metadata_clear_all",
    ):
        monkeypatch.setattr(
            f"lobster.cli_internal.commands.{name}",
            lambda *a, _name=name, **k: calls.append(_name),
        )
    slash_commands.metadata_command_impl(workspace=tmp_path, **kwargs)
    return calls


@pytest.mark.parametrize("subcommand", [None, "overview"])
def test_metadata_overview_is_reachable_by_name(monkeypatch, tmp_path, subcommand):
    assert _route_metadata(monkeypatch, tmp_path, subcommand=subcommand) == [
        "metadata_overview"
    ]


@pytest.mark.parametrize(
    "target,expected",
    [
        (None, "metadata_clear"),
        ("exports", "metadata_clear_exports"),
        ("all", "metadata_clear_all"),
    ],
)
def test_metadata_clear_routes_to_the_documented_target(
    monkeypatch, tmp_path, target, expected
):
    calls = _route_metadata(monkeypatch, tmp_path, subcommand="clear", target=target)
    assert calls == [expected]


@pytest.mark.parametrize(
    "subcommand,target", [("clear", "bogus"), ("samples", "extra"), (None, "x")]
)
def test_metadata_rejects_a_second_word_it_does_not_take(
    monkeypatch, tmp_path, subcommand, target
):
    with pytest.raises(typer.Exit) as exc_info:
        _route_metadata(monkeypatch, tmp_path, subcommand=subcommand, target=target)
    assert exc_info.value.exit_code == 2


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["metadata", "overview"], {"subcommand": "overview", "target": None}),
        (
            ["metadata", "clear", "exports"],
            {"subcommand": "clear", "target": "exports"},
        ),
        (["metadata", "clear", "all"], {"subcommand": "clear", "target": "all"}),
    ],
)
def test_metadata_cli_parses_the_documented_forms(monkeypatch, argv, expected):
    from typer.testing import CliRunner

    from lobster.cli import app

    seen = {}
    # Patch by dotted path: other tests may re-import the module.
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.slash_commands.metadata_command_impl",
        lambda **kwargs: seen.update(kwargs),
    )
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert {k: seen[k] for k in expected} == expected


def _clear_workspace(monkeypatch, tmp_path, tty, answer):
    """A workspace with one metadata file and one export, a fake client and a
    real console adapter whose confirmation returns ``answer``."""
    from types import SimpleNamespace

    (tmp_path / "metadata").mkdir()
    (tmp_path / "exports").mkdir()
    meta = tmp_path / "metadata" / "probe.json"
    export = tmp_path / "exports" / "probe.csv"
    meta.write_text("{}")
    export.write_text("a,b")
    store = {"GSE1": {}}
    client = SimpleNamespace(
        data_manager=SimpleNamespace(
            metadata_store=store, current_metadata={}, workspace_path=str(tmp_path)
        )
    )
    created = []
    monkeypatch.setattr("sys.stdin", _Stdin(tty))
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=False: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.core.client.AgentClient",
        lambda workspace_path: created.append(workspace_path) or client,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.output_adapter.ConsoleOutputAdapter.confirm",
        lambda self, question: answer,
    )
    return meta, export, store, created


@pytest.mark.parametrize("target", [None, "exports", "all"])
def test_metadata_clear_refuses_without_a_terminal(monkeypatch, tmp_path, target):
    meta, export, store, created = _clear_workspace(
        monkeypatch, tmp_path, tty=False, answer=True
    )
    printed = []
    monkeypatch.setattr(
        "lobster.cli_internal.commands.output_adapter.ConsoleOutputAdapter.print",
        lambda self, message, style=None: printed.append(message),
    )

    with pytest.raises(typer.Exit) as exc_info:
        slash_commands.metadata_command_impl(
            subcommand="clear", target=target, workspace=tmp_path
        )

    assert exc_info.value.exit_code == 1
    assert printed == [
        "lobster metadata clear deletes files and asks for confirmation first. "
        "Run it in a terminal.",
        "Nothing was deleted.",
    ]
    assert created == []
    assert meta.exists() and export.exists() and store == {"GSE1": {}}


@pytest.mark.parametrize("target", [None, "exports", "all"])
def test_metadata_clear_cancel_deletes_nothing(monkeypatch, tmp_path, target):
    meta, export, store, _ = _clear_workspace(
        monkeypatch, tmp_path, tty=True, answer=False
    )

    slash_commands.metadata_command_impl(
        subcommand="clear", target=target, workspace=tmp_path
    )

    assert meta.exists() and export.exists() and store == {"GSE1": {}}


@pytest.mark.parametrize(
    "target,meta_left,export_left",
    [(None, False, True), ("exports", True, False), ("all", False, False)],
)
def test_metadata_clear_confirmed_deletes_only_its_target(
    monkeypatch, tmp_path, target, meta_left, export_left
):
    meta, export, _, _ = _clear_workspace(monkeypatch, tmp_path, tty=True, answer=True)

    slash_commands.metadata_command_impl(
        subcommand="clear", target=target, workspace=tmp_path
    )

    assert (meta.exists(), export.exists()) == (meta_left, export_left)
