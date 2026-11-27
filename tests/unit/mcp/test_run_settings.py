"""The overview shows the provider, model and key a run of this build really
uses, which depends on the folder the server process runs in: the project's
own settings reach a run only when the server runs in the project folder and
no global file or LOBSTER_LLM_PROVIDER takes precedence. Otherwise one field
says so and gives the fix, and a run that then finds no provider says the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_overview
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
)
from lobster.config import global_config
from lobster.config.workspace_config import WorkspaceProviderConfig
from lobster.core import projects
from tests.unit.mcp.test_run import (  # noqa: F401
    COMPLETE,
    call,
    data,
    engine,
    wait_terminal,
)

# conftest's autouse provider fixture would set LOBSTER_LLM_PROVIDER=anthropic
pytestmark = pytest.mark.no_auto_config

EXE = "/opt/lobster/bin/lobster"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(global_config, "CONFIG_DIR", home / ".config" / "lobster")
    monkeypatch.setattr(
        global_config,
        "get_global_credentials_path",
        lambda: home / ".config" / "lobster" / "credentials.env",
    )
    for name in ("LOBSTER_LLM_PROVIDER", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set())
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(mcp_overview, "_lobster_path", lambda: EXE)


@pytest.fixture
def project(tmp_path):
    made, _ = projects.init_project(tmp_path / "study", name="Study")
    WorkspaceProviderConfig(global_provider="gemini", gemini_model="gemini-x").save(
        made.workspace
    )
    return made


def overview(start):
    return call(McpServer(ThreadRunner(), start=start), "lobster_overview", {})


def test_in_the_project_folder_the_project_settings_are_used(monkeypatch, project):
    monkeypatch.chdir(project.root)
    seen = overview(project.root)
    assert (seen["provider"], seen["provider_source"]) == ("gemini", "project")
    assert (seen["model"], seen["model_source"]) == ("gemini-x", "project")
    assert "project_settings_not_applied" not in seen
    assert "not_yet_applied" not in (seen["key"] or {})


def test_started_elsewhere_the_overview_says_what_a_run_uses_and_the_fix(
    monkeypatch, project, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    seen = overview(project.root)
    assert (seen["provider"], seen["provider_source"]) == ("gemini", "project")
    note = seen["project_settings_not_applied"]
    assert str(elsewhere) in note and str(project.root) in note
    assert "the run in progress takes" in note
    assert "the next run starts in the project folder" in note
    assert "key" in note and "provider" not in note


def test_without_its_own_provider_started_elsewhere_the_note_names_the_provider(
    monkeypatch, tmp_path
):
    made, _ = projects.init_project(tmp_path / "plain", name="Plain")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    note = overview(made.root)["project_settings_not_applied"]
    assert "key" in note and "provider" in note


def test_the_project_provider_and_model_win_over_a_global_file(monkeypatch, project):
    monkeypatch.chdir(project.root)
    global_config.GlobalProviderConfig(
        default_provider="gemini", gemini_default_model="gemini-global"
    ).save()
    seen = overview(project.root)
    assert (seen["provider"], seen["provider_source"]) == ("gemini", "project")
    assert (seen["model"], seen["model_source"]) == ("gemini-x", "project")
    assert "project_settings_not_applied" not in seen


def test_without_its_own_provider_the_global_file_is_used(monkeypatch, tmp_path):
    made, _ = projects.init_project(tmp_path / "plain", name="Plain")
    monkeypatch.chdir(made.root)
    global_config.GlobalProviderConfig(default_provider="ollama").save()
    seen = overview(made.root)
    assert (seen["provider"], seen["provider_source"]) == ("ollama", "global file")
    assert "project_settings_not_applied" not in seen


def test_a_run_from_elsewhere_says_so_and_how_to_fix_it(engine, project, data):
    # the engine fixture runs the server in a folder of its own
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(code="missing_credentials", title="No API key found")
    )
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(server, "lobster_run", {"question": "q", "input_path": str(data)})
    assert "the next run starts" in run["project_settings_not_applied"]
    done = wait_terminal(server, run["run_id"])
    message = done["error"]["message"]
    assert message.startswith("No API key found. ")
    assert "started outside the project" in message
    assert mcp_overview.lobster_command("init", "--global") in message


@pytest.mark.parametrize("provider", ["omics-os", "anthropic"])
def test_a_run_without_a_key_ignores_the_omics_os_cli_sign_in(
    engine, tmp_path, data, monkeypatch, provider
):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", provider)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    home = tmp_path / "home"
    sign_in = home / ".config" / "omics-os" / "credentials.json"
    sign_in.parent.mkdir(parents=True)
    sign_in.write_text(json.dumps({"auth_mode": "oauth", "access_token": "omc_000"}))
    before = sign_in.read_bytes()
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(
            code="missing_credentials",
            title=f"Missing credentials for provider '{provider}'",
        )
    )
    server = McpServer(ThreadRunner())
    run = call(
        server,
        "lobster_run",
        {"question": "q", "input_path": str(data), "output_dir": str(tmp_path / "o")},
    )
    message = wait_terminal(server, run["run_id"])["error"]["message"]
    assert message.startswith(f"Missing credentials for provider '{provider}'. ")
    assert mcp_overview.key_fix() in message
    assert "omc_000" not in message
    assert sign_in.read_bytes() == before


def test_a_run_without_a_provider_in_no_project_gets_the_fix(engine, tmp_path, data):
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(
            code="provider_not_configured", title="No provider configured"
        )
    )
    server = McpServer(ThreadRunner())
    run = call(
        server,
        "lobster_run",
        {"question": "q", "input_path": str(data), "output_dir": str(tmp_path / "o")},
    )
    message = wait_terminal(server, run["run_id"])["error"]["message"]
    assert "LOBSTER_LLM_PROVIDER" in message and "outside the project" not in message


def test_a_startup_failure_says_what_failed(engine, tmp_path, data):
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(
            code="startup_failed",
            title="Startup failed",
            detail_lines=("Install Ollama Cloud support: uv pip install 'x[openai]'",),
            fix_lines=("Run again with --debug for a full traceback.",),
        )
    )
    server = McpServer(ThreadRunner())
    run = call(
        server,
        "lobster_run",
        {"question": "q", "input_path": str(data), "output_dir": str(tmp_path / "o")},
    )
    message = wait_terminal(server, run["run_id"])["error"]["message"]
    assert message == (
        "Startup failed: Install Ollama Cloud support: uv pip install 'x[openai]'"
    )


def test_a_startup_failure_without_the_key_gives_the_two_fixes(
    engine, project, data, monkeypatch
):
    # The engine names a provider's missing key as a plain startup failure.
    monkeypatch.chdir(project.root)
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(
            code="startup_failed",
            title="Startup failed",
            detail_lines=("Set GOOGLE_API_KEY to use Gemini",),
        )
    )
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(server, "lobster_run", {"question": "q", "input_path": str(data)})
    message = wait_terminal(server, run["run_id"])["error"]["message"]
    assert message.startswith("Startup failed: Set GOOGLE_API_KEY to use Gemini. ")
    names = mcp_overview.key_names("gemini")
    assert f"No {' and '.join(names)} " in message
    assert mcp_overview.key_fix(names) in message


# start lines, expert names, start text, links, waits


def test_a_global_provider_without_a_key_variable_gives_the_plain_line(tmp_path):
    global_config.GlobalProviderConfig(default_provider="ollama").save()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert overview(empty)["start_lines"] == [f"{EXE} init {empty}"]


def test_expert_names_from_the_settings_file_are_filtered(monkeypatch, project):
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    WorkspaceAgentConfig(
        enabled_agents=["research_agent", "bad\u001b[31m", "a b", "x;y"]
    ).save(project.workspace)
    assert overview(project.root)["experts"]["enabled"] == ["research_agent"]


def test_the_start_text_names_a_broken_project_file(project):
    import json

    from lobster.cli_internal import mcp_contract

    (project.workspace / "project.json").write_text("{bad")
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }
    )
    text = McpServer(ThreadRunner(), start=project.root).handle_line(line)["result"][
        "instructions"
    ]
    assert "project.json" in text and "cannot be read" in text
    assert len(text) <= mcp_contract.INSTRUCTIONS_MAX


def test_an_output_dir_through_a_linked_runs_folder_is_refused(project, data, tmp_path):
    from tests.unit.mcp.test_run import Gate

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (project.root / "runs").symlink_to(elsewhere)
    gate = Gate()
    refused = call(
        McpServer(ThreadRunner(gate), start=project.root),
        "lobster_run",
        {
            "question": "q",
            "input_path": str(data),
            "output_dir": str(project.root / "runs" / "x"),
        },
    )["error_text"]
    assert refused.startswith("path_not_allowed:") and "link" in refused
    assert gate.calls == [] and list(elsewhere.iterdir()) == []


PLANTED = "sk-planted-not-copied-0123"


def test_the_run_session_gets_the_whole_project_provider_file_and_no_key(
    engine, project, data, monkeypatch
):
    """Model and per-expert maps come along; a key-looking value planted in the
    project's provider file, its .env and the environment reaches no file of
    the run and no reply."""
    import json

    path = project.workspace / "provider_config.json"
    planted = json.loads(path.read_text())
    planted.update(
        api_key=PLANTED,
        per_agent_providers={"research_agent": "gemini"},
        per_agent_models={"research_agent": "gemini-y"},
    )
    path.write_text(json.dumps(planted))
    (project.root / ".env").write_text(f"GOOGLE_API_KEY={PLANTED}\n")
    monkeypatch.setenv("GEMINI_API_KEY", PLANTED)
    monkeypatch.chdir(project.root)
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    replies = [call(server, "lobster_overview", {})]
    replies.append(
        call(
            server,
            "lobster_run",
            {"question": "q", "input_path": str(data), "plan": False},
        )
    )
    replies.append(wait_terminal(server, replies[-1]["run_id"]))
    assert replies[-1]["status"] == "completed"
    out = Path(replies[1]["output_dir"])
    copied = WorkspaceProviderConfig.load(out / ".lobster_workspace")
    assert (copied.global_provider, copied.gemini_model) == ("gemini", "gemini-x")
    assert copied.per_agent_models == {"research_agent": "gemini-y"}
    assert copied.per_agent_providers == {"research_agent": "gemini"}
    assert PLANTED not in json.dumps(replies)
    for file in out.rglob("*"):
        if file.is_file():
            assert PLANTED.encode() not in file.read_bytes(), file


def test_a_restored_run_keeps_the_provider_file_of_its_first_run(
    engine, project, data, monkeypatch
):
    from tests.unit.mcp.test_continue import go_on
    from tests.unit.mcp.test_restore import SESSION_FILE

    monkeypatch.chdir(project.root)
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {SESSION_FILE: b"{}"}
    first = ThreadRunner()
    server = McpServer(first, start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    out = Path(run["output_dir"])
    first.stop(30)
    WorkspaceProviderConfig(global_provider="ollama").save(project.workspace)
    script["files"] = {}
    second = McpServer(ThreadRunner(), start=project.root)
    reply = go_on(second, "next", output_dir=str(out))
    assert wait_terminal(second, reply["run_id"])["status"] == "completed"
    session = out / ".lobster_workspace"
    assert WorkspaceProviderConfig.load(session).global_provider == "gemini"


@pytest.mark.parametrize(
    "failure, said",
    [
        (OSError(28, "No space left on device"), "No space left on device"),
        (ValueError("model_dump failed"), "model_dump failed"),
    ],
    ids=["disk full", "any other error"],
)
def test_a_copy_that_fails_fails_the_run_and_leaves_no_folder(
    engine, project, data, monkeypatch, failure, said
):
    def refuse(self, workspace_path):
        raise failure

    monkeypatch.setattr(WorkspaceProviderConfig, "save", refuse)
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    error = wait_terminal(server, run["run_id"])["error"]
    assert error["code"] == "project_settings_not_copied"
    assert said in error["message"]
    assert not (Path(run["output_dir"]) / ".lobster_workspace").exists()


@pytest.mark.parametrize("text", ['{"global_provider": "gem', '{"global_provider": 7}'])
def test_a_project_provider_file_that_cannot_be_read_stops_the_run(
    engine, project, data, text
):
    """Never a run on the global or shell provider instead, without a word."""
    (project.workspace / "provider_config.json").write_text(text)
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    error = wait_terminal(server, run["run_id"])["error"]
    assert error["code"] == "project_settings_not_copied"
    assert str(project.workspace / "provider_config.json") in error["message"]
    assert not (Path(run["output_dir"]) / ".lobster_workspace").exists()


def test_the_overview_lists_settings_files_a_run_would_refuse(monkeypatch, project):
    monkeypatch.chdir(project.root)
    (project.workspace / "provider_config.json").write_text('{"global_provider": 7}')
    (project.workspace / "config.toml").write_text('enabled = "research_agent"\n')
    assert overview(project.root)["unreadable"] == [
        str(project.workspace / "provider_config.json"),
        str(project.workspace / "config.toml"),
    ]


# a finished run names the provider and model it used


def result_schema():
    import json

    return json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "lobster/schemas/lobster.result.v1.schema.json"
        ).read_text()
    )


def test_a_finished_run_and_its_continued_turn_name_the_provider_and_model(
    engine, project, data, monkeypatch
):
    import json

    from jsonschema import Draft202012Validator

    from tests.unit.mcp.test_continue import go_on
    from tests.unit.mcp.test_restore import SESSION_FILE

    monkeypatch.chdir(project.root)
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {SESSION_FILE: b"{}"}
    first = ThreadRunner()
    server = McpServer(first, start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    done = wait_terminal(server, run["run_id"])
    assert (done["provider"], done["model"]) == ("gemini", "gemini-x")
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["engine"]["provider"] == "gemini"
    assert record["engine"]["model"] == "gemini-x"
    Draft202012Validator(result_schema()).validate(record)
    # After a restart, with the project now set to another provider, the
    # continued turn still runs on, and names, the first run's settings.
    first.stop(30)
    WorkspaceProviderConfig(global_provider="ollama").save(project.workspace)
    script["files"] = {}
    second = McpServer(ThreadRunner(), start=project.root)
    reply = go_on(second, "next", output_dir=run["output_dir"])
    turn = wait_terminal(second, reply["run_id"])
    assert (turn["provider"], turn["model"]) == ("gemini", "gemini-x")
    later = json.loads(Path(turn["result_path"]).read_text())
    assert (later["engine"]["provider"], later["engine"]["model"]) == (
        "gemini",
        "gemini-x",
    )


def test_a_failed_run_names_them_too(engine, project, data, monkeypatch):
    monkeypatch.chdir(project.root)
    _opened, script = engine
    script["events"] = [{"type": "error", "error": "boom", "error_type": "X"}]
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "failed"
    assert (done["provider"], done["model"]) == ("gemini", "gemini-x")


def test_a_replayed_run_names_no_provider(engine, project, data, monkeypatch):
    from lobster.config.providers.replay_provider import REPLAY_ENV

    monkeypatch.chdir(project.root)
    monkeypatch.setenv(REPLAY_ENV, "/nowhere.json")
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(
        server, "lobster_run", {"question": "q", "input_path": str(data), "plan": False}
    )
    done = wait_terminal(server, run["run_id"])
    assert done["model_source"] == "replay"
    assert "provider" not in done and "model" not in done
