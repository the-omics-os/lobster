"""``lobster_overview``: the live state of the server's project in one call.

Provider, model, whether the key is there (never any part of it), the project,
its experts, its data files and past runs, and the command lines of what MCP
does not do. Read again on every call; nothing of the engine is imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from lobster.cli_internal import mcp_contract, mcp_overview
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.config import credentials, global_config
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.config.workspace_config import WorkspaceProviderConfig
from lobster.core import projects
from tests.unit.mcp.test_run import call

# conftest's autouse provider fixture would set LOBSTER_LLM_PROVIDER=anthropic
pytestmark = pytest.mark.no_auto_config

EXE = "/opt/lobster env/bin/lobster"  # a path with a space: printed quoted
SHOWN = "'/opt/lobster env/bin/lobster'"
KEY = "OLLAMA_API_KEY"
SECRET = "sk-overview-0123456789"
INSTALLED = {"research_agent", "data_expert_agent", "transcriptomics_expert"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setattr(
        global_config,
        "get_global_credentials_path",
        lambda: tmp_path / "home" / ".config" / "lobster" / "credentials.env",
    )
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "home" / "omics-os.json"
    )
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set(INSTALLED))
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(mcp_overview, "_lobster_path", lambda: EXE)


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "pbmc"
    root.mkdir()
    monkeypatch.chdir(root)  # a run takes the settings of the server's folder
    made, _ = projects.init_project(root, name="PBMC study")
    WorkspaceProviderConfig(
        global_provider="ollama-cloud", ollama_cloud_model="glm-5.2"
    ).save(made.workspace)
    WorkspaceAgentConfig(
        enabled_agents=["research_agent", "transcriptomics_expert"]
    ).save(made.workspace)
    return made


def overview(start):
    return call(McpServer(ThreadRunner(), start=start), "lobster_overview", {})


def test_the_overview_is_the_first_tool_read_only_without_arguments():
    tools = mcp_contract.list_tools()
    first = tools[0]
    assert first["name"] == "lobster_overview"
    assert first["inputSchema"]["properties"] == {}
    assert first["annotations"]["readOnlyHint"] is True
    for line in ("lobster init <folder>", "lobster agents install <package>"):
        assert line in first["description"]


def test_without_a_project_it_says_how_to_start_one(tmp_path):
    study = tmp_path / "study"  # tmp_path holds HOME: no project can go there
    study.mkdir()
    seen = overview(study)
    assert seen["project"] is None
    assert f"{SHOWN} init {study}" in seen["next"]
    assert seen["commands"] == {
        "start_project": f"{SHOWN} init <folder>",
        "install_experts": f"{SHOWN} agents install <package>",
    }


def test_a_project_made_after_the_server_started_is_seen_without_a_restart(
    tmp_path, monkeypatch
):
    study = tmp_path / "study"
    study.mkdir()
    server = McpServer(ThreadRunner(), start=study)
    assert call(server, "lobster_overview", {})["project"] is None
    projects.init_project(study, name="late")
    monkeypatch.setattr(
        mcp_overview, "loaded_experts", lambda: INSTALL_LATE_BASE
    )  # experts were loaded before the install below
    monkeypatch.setattr(
        mcp_overview,
        "installed_experts",
        lambda: INSTALL_LATE_BASE | {"proteomics_expert"},
    )
    seen = call(server, "lobster_overview", {})
    assert seen["project"]["name"] == "late"
    assert "proteomics_expert" in seen["restart_needed"]


INSTALL_LATE_BASE = frozenset(INSTALLED)


def test_a_project_is_found_from_a_folder_inside_it(project):
    inside = project.root / "notes"
    inside.mkdir()
    seen = overview(inside)
    assert seen["project"] == {
        "id": project.project_id,
        "name": "PBMC study",
        "folder": str(project.root),
        "runs_folder": str(project.root / "runs"),
    }
    assert (seen["provider"], seen["model"]) == ("ollama-cloud", "glm-5.2")
    assert seen["experts"]["enabled"] == ["research_agent", "transcriptomics_expert"]
    assert seen["experts"]["installed"] == sorted(INSTALLED)


@pytest.mark.parametrize(
    "where, found_in",
    [
        ("ws", "{ws}/.env"),
        ("root", "{root}/.env"),
        ("shell", "environment"),
        ("global", "global config"),
    ],
)
def test_the_key_is_reported_present_with_where_it_was_found(
    project, monkeypatch, tmp_path, capfd, where, found_in
):
    line = f"{KEY}={SECRET}\n"
    if where == "ws":
        (project.workspace / ".env").write_text(line)
    elif where == "root":
        (project.root / ".env").write_text(line)
    elif where == "shell":
        monkeypatch.setenv(KEY, SECRET)
    else:
        path = global_config.get_global_credentials_path()
        path.parent.mkdir(parents=True)
        path.write_text(line)
    seen = overview(project.root)
    expected = found_in.format(ws=project.workspace, root=project.root)
    assert seen["key"]["names"] == [KEY]
    assert seen["key"]["present"] is True
    assert seen["key"]["found_in"] == expected
    assert SECRET not in json.dumps(seen) + "".join(capfd.readouterr())


def test_next_leads_with_a_missing_key(project, monkeypatch):
    seen = overview(project.root)
    assert seen["next"].startswith(
        f"No {KEY} was found for {seen['provider']}, and a run fails at startup "
        "without it (unless the provider signs in another way; see key.limit). "
        "key.fix says how to set it. Ask lobster_run"
    )
    monkeypatch.setenv(KEY, SECRET)
    assert overview(project.root)["next"].startswith("Ask lobster_run")


LIVE = {
    "auth_mode": "oauth",
    "access_token": "omk_planted_000",
    "token_expiry": "2999-01-01T00:00:00+00:00",
}


@pytest.mark.parametrize("saved", [{"api_key": "omk_planted_000"}, LIVE])
def test_a_saved_omics_os_login_counts_as_the_key(project, monkeypatch, saved):
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    WorkspaceProviderConfig(global_provider="omics-os").save(project.workspace)
    seen = overview(project.root)
    assert seen["key"]["present"] is False
    assert seen["next"].startswith("No OMICS_OS_API_KEY was found for omics-os")
    credentials.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    credentials.CREDENTIALS_FILE.write_text(json.dumps(saved))
    assert credentials.get_api_key() == "omk_planted_000"  # what the run gets
    seen = overview(project.root)
    assert seen["key"]["present"] is True
    assert seen["key"]["found_in"] == "omics-os login"
    assert seen["next"].startswith("Ask lobster_run")
    assert "omk_planted_000" not in json.dumps(seen)


EXPIRED = {
    "auth_mode": "oauth",
    "access_token": "omc_expired_000",
    "token_expiry": "2026-01-01T00:00:00+00:00",
}


@pytest.mark.parametrize("refresh", [None, "omr_dead_000"])
def test_an_expired_omics_os_login_is_not_the_key(project, monkeypatch, refresh):
    # The run reads a saved login through credentials.get_api_key: an expired
    # token that cannot be refreshed is no key, and the run fails at startup
    # with missing_credentials. The overview must say the same.
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "refresh_token", lambda: None)
    WorkspaceProviderConfig(global_provider="omics-os").save(project.workspace)
    saved = {**EXPIRED, **({"refresh_token": refresh} if refresh else {})}
    credentials.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    credentials.CREDENTIALS_FILE.write_text(json.dumps(saved))
    assert credentials.get_api_key() is None  # what the run gets
    seen = overview(project.root)
    assert seen["key"]["present"] is False
    assert seen["key"]["found_in"] is None
    assert seen["key"]["login"] == "expired"
    assert "cloud login" in seen["key"]["fix"]
    assert seen["next"].startswith("No OMICS_OS_API_KEY was found for omics-os")
    assert "omc_expired_000" not in json.dumps(seen)


def test_a_token_without_its_oauth_mode_is_not_the_key(project, monkeypatch):
    # The provider reads access_token only in OAuth mode (get_api_key).
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    WorkspaceProviderConfig(global_provider="omics-os").save(project.workspace)
    credentials.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    credentials.CREDENTIALS_FILE.write_text(json.dumps({"access_token": "omk_0"}))
    assert credentials.get_api_key() is None
    seen = overview(project.root)
    assert (seen["key"]["present"], seen["key"]["login"]) == (False, "unusable")


def test_an_expired_login_the_run_can_refresh_counts_as_the_key(project, monkeypatch):
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "refresh_token", lambda: "omc_fresh_000")
    WorkspaceProviderConfig(global_provider="omics-os").save(project.workspace)
    credentials.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    credentials.CREDENTIALS_FILE.write_text(
        json.dumps({**EXPIRED, "refresh_token": "omr_live_000"})
    )
    seen = overview(project.root)
    assert seen["key"]["present"] is True
    assert seen["key"]["found_in"] == "omics-os login"
    assert "login" not in seen["key"]
    assert "omc_fresh_000" not in json.dumps(seen)


def test_both_env_files_of_a_project_are_read(project):
    # As the run reads them: a workspace .env without the key does not hide it.
    (project.workspace / ".env").write_text("OTHER=1\n")
    (project.root / ".env").write_text(f"{KEY}={SECRET}\n")
    seen = overview(project.root)
    assert seen["key"]["present"] is True
    assert seen["key"]["found_in"] == str(project.root / ".env")


def test_a_server_started_elsewhere_says_the_projects_key_is_not_used(
    project, tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text(f"{KEY}={SECRET}\n")
    monkeypatch.chdir(elsewhere)
    seen = overview(project.root)
    assert seen["provider"] == "ollama-cloud"
    assert seen["key"]["present"] is True  # from the .env of the server's folder
    assert str(elsewhere) in seen["project_settings_not_applied"]
    assert SECRET not in json.dumps(seen)


def test_a_provider_without_a_key_says_none_is_needed(project):
    WorkspaceProviderConfig(global_provider="ollama", ollama_model="llama3").save(
        project.workspace
    )
    seen = overview(project.root)
    assert seen["key"] == {"needed": False}


def test_experts_the_project_lacks_are_listed_with_the_install_line(project):
    seen = overview(project.root)
    installable = {p["package"]: p for p in seen["experts"]["installable"]}
    assert "lobster-research" not in installable  # all of it installed
    proteomics = installable["lobster-proteomics"]
    assert proteomics["install"] == f"{SHOWN} agents install lobster-proteomics"
    assert "proteomics_expert" in proteomics["experts"]
    # one of its experts is installed: only the missing ones are listed
    transcriptomics = installable["lobster-transcriptomics"]
    assert "transcriptomics_expert" not in transcriptomics["experts"]


def test_installed_experts_are_read_again_on_every_call(project, monkeypatch):
    server = McpServer(ThreadRunner(), start=project.root)
    assert (
        "proteomics_expert"
        not in call(server, "lobster_overview", {})["experts"]["installed"]
    )
    monkeypatch.setattr(
        mcp_overview, "installed_experts", lambda: INSTALLED | {"proteomics_expert"}
    )
    again = call(server, "lobster_overview", {})
    assert "proteomics_expert" in again["experts"]["installed"]
    assert "restart_needed" not in again  # nothing loaded yet: a run reads them


def test_an_expert_installed_after_they_were_loaded_needs_a_restart(
    project, monkeypatch
):
    monkeypatch.setattr(
        mcp_overview, "loaded_experts", lambda: INSTALLED - {"transcriptomics_expert"}
    )
    seen = overview(project.root)
    assert "transcriptomics_expert" in seen["restart_needed"]
    assert "restart the Lobster MCP server" in seen["restart_needed"]


def test_data_files_and_runs_are_listed_bounded(project):
    (project.root / "data").mkdir()
    for n in range(25):
        (project.root / "data" / f"sample_{n:02}.h5ad").write_bytes(b"x")
    (project.root / "counts.csv").write_text("a,b\n")
    (project.root / "notes.md").write_text("not data")
    (project.workspace / "inside.h5ad").write_bytes(b"x")  # the session's own
    for n in range(3):
        (project.root / "runs" / f"run_{n}").mkdir(parents=True)
    (project.root / "runs" / "run_0" / "out.h5ad").write_bytes(b"x")
    seen = overview(project.root)
    assert seen["data_files_total"] == 26
    assert len(seen["data_files"]) == mcp_overview.LISTED_MAX
    assert seen["data_files"][0] == "counts.csv"
    assert all(
        ".lobster_workspace" not in f and "runs/" not in f for f in seen["data_files"]
    )
    assert seen["runs"] == ["run_0", "run_1", "run_2"] and seen["runs_total"] == 3


def test_the_overview_imports_nothing_of_the_engine(project):
    code = (
        "import json, sys\n"
        "from lobster.cli_internal.mcp_stdio import McpServer, FakeRunner\n"
        f"server = McpServer(FakeRunner(), start={str(project.root)!r})\n"
        "line = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',"
        " 'params': {'name': 'lobster_overview', 'arguments': {}}})\n"
        "reply = server.handle_line(line)\n"
        "assert 'isError' not in reply['result'], reply\n"
        "heavy = sorted(m for m in sys.modules if m.startswith(("
        "'langchain', 'langgraph', 'lobster.agents', 'lobster.core.client',"
        " 'lobster.core.component_registry', 'lobster.cli_internal.commands.heavy')))\n"
        "print(','.join(heavy))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""


def test_a_server_started_elsewhere_finds_the_project_key(
    project, tmp_path, monkeypatch
):
    """The key the overview reports in the project folder's .env reaches the
    run's engine, whatever folder the server was started in."""
    from lobster.cli_internal import mcp_stdio
    from lobster.config.settings import settings
    from lobster.core.config_resolver import ConfigResolver

    seen = {}

    def resolver(*args, **kwargs):
        import os

        seen["key"] = os.environ.get(KEY)
        raise RuntimeError("stop after the key was read")

    monkeypatch.setattr(ConfigResolver, "get_instance", resolver)
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: "")
    (project.root / ".env").write_text(f"{KEY}={SECRET}\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    try:
        with pytest.raises(RuntimeError):
            mcp_stdio._build_client(tmp_path / "session", project.workspace, None)
    finally:
        settings.reload_credentials(tmp_path / "nowhere")  # undo the project's keys
    assert seen["key"] == SECRET


@pytest.mark.parametrize("found", ["argv", "beside_python", "none"])
def test_the_command_lines_name_the_lobster_of_this_install(
    monkeypatch, tmp_path, found
):
    monkeypatch.undo()  # the real lookup, not the fixture's stand-in
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("")
    monkeypatch.setattr(sys, "executable", str(python))
    script = tmp_path / "elsewhere" / "lobster"
    script.parent.mkdir()
    if found == "argv":
        script.write_text("#!/bin/sh\n")
        script.chmod(0o755)
        monkeypatch.setattr(sys, "argv", [str(script), "mcp", "--stdio"])
        expected = str(script)
    else:
        monkeypatch.setattr(sys, "argv", [str(tmp_path / "x" / "__main__.py")])
        expected = "lobster"
        if found == "beside_python":
            (bin_dir / "lobster").write_text("#!/bin/sh\n")
            (bin_dir / "lobster").chmod(0o755)
            expected = str(bin_dir / "lobster")
    assert mcp_overview.lobster_command() == expected


def test_the_experts_say_a_list_replaces_the_team(project):
    seen = overview(project.root)
    assert seen["experts"]["note"] == mcp_overview.TEAM_NOTE
    assert "not_yet_applied" not in seen["experts"]


@pytest.mark.parametrize("bad", ["unreadable", "not_utf8"])
def test_an_env_file_that_cannot_be_read_is_named_not_an_error(project, bad):
    env = project.root / ".env"
    if bad == "unreadable":
        env.write_text(f"{KEY}={SECRET}\n")
        env.chmod(0)
    else:
        env.write_bytes(b"\xff\xfe\x00" + KEY.encode())
    try:
        seen = overview(project.root)
    finally:
        env.chmod(0o600)
    assert seen["key"]["present"] is False and seen["key"]["found_in"] is None
    assert seen["key"]["unreadable"] == [str(env)]


def test_a_provider_config_that_cannot_be_read_is_named(project):
    (project.workspace / "provider_config.json").write_text("{not json")
    seen = overview(project.root)
    assert str(project.workspace / "provider_config.json") in seen["unreadable"]


def test_a_broken_project_file_is_named_not_hidden(tmp_path):
    root = tmp_path / "broken"
    projects.init_project(root, name="b")
    (root / ".lobster_workspace" / "project.json").write_text("{bad")
    seen = overview(root)
    assert seen["project"] is None
    assert f"project.json in {root} cannot be read" in seen["next"]
    assert "init" not in seen["next"]


# what a run will use -----------------------------------------------------------


@pytest.fixture
def global_dir(monkeypatch, tmp_path):
    folder = tmp_path / "home" / ".config" / "lobster"
    monkeypatch.setattr(global_config, "CONFIG_DIR", folder)
    monkeypatch.delenv("LOBSTER_LLM_PROVIDER", raising=False)
    return folder


@pytest.fixture
def identity_only(tmp_path):
    made, _ = projects.init_project(tmp_path / "bare", name="Bare")
    return made


def test_a_project_without_settings_shows_what_the_global_file_gives(
    global_dir, identity_only
):
    global_config.GlobalProviderConfig(
        default_provider="ollama-cloud", ollama_cloud_default_model="glm-5.2"
    ).save()
    seen = overview(identity_only.root)
    assert (seen["provider"], seen["provider_source"]) == (
        "ollama-cloud",
        "global file",
    )
    assert (seen["model"], seen["model_source"]) == ("glm-5.2", "global file")
    assert seen["key"]["names"] == [KEY]
    assert seen["experts"]["enabled"] == sorted(INSTALLED)
    assert seen["experts"]["source"] == "every installed expert"


def test_a_provider_from_the_environment_is_shown_with_its_source(
    monkeypatch, global_dir, identity_only
):
    # a shell export or a host config's env block: the server cannot tell
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "ollama")
    seen = overview(identity_only.root)
    assert (seen["provider"], seen["provider_source"]) == ("ollama", "environment")
    assert seen["model_source"] in ("provider default", None)


def test_project_settings_are_used_and_say_so(global_dir, project):
    seen = overview(project.root)
    assert (seen["provider"], seen["provider_source"]) == ("ollama-cloud", "project")
    assert (seen["model"], seen["model_source"]) == ("glm-5.2", "project")
    assert seen["experts"]["source"] == "project"
    assert "project_settings_not_applied" not in seen


def test_the_project_provider_wins_over_a_global_file(global_dir, project):
    global_config.GlobalProviderConfig(default_provider="ollama").save()
    seen = overview(project.root)
    assert (seen["provider"], seen["provider_source"]) == ("ollama-cloud", "project")
    assert "project_settings_not_applied" not in seen


def test_nothing_set_anywhere_is_null_with_no_source(global_dir, identity_only):
    seen = overview(identity_only.root)
    assert seen["provider"] is None and seen["provider_source"] is None
    assert seen["model"] is None and seen["model_source"] is None
    assert seen["key"] is None


def test_a_missing_key_says_where_to_put_it(project):
    seen = overview(project.root)
    key = seen["key"]
    assert key["present"] is False
    assert key["fix"] == mcp_overview.key_fix([KEY])
    assert KEY in key["fix"] and "env block" in key["fix"]
    assert "MCP config" in key["fix"]
    assert mcp_overview.lobster_command("init", "--global") in key["fix"]
    assert SECRET not in json.dumps(seen)


def test_a_present_key_needs_no_fix(monkeypatch, project):
    monkeypatch.setenv(KEY, SECRET)
    assert "fix" not in overview(project.root)["key"]


def test_key_limit_names_only_sign_ins_lobster_reads():
    from lobster.cli_internal.mcp_overview import KEY_LIMIT

    # Bedrock reads only AWS_BEDROCK_* keys; an AWS profile does not sign it in.
    assert "AWS profile" not in KEY_LIMIT
    assert "lobster cloud login" in KEY_LIMIT


def test_key_presence_reads_the_project_env_beside_a_lamin_workspace_env(
    tmp_path, monkeypatch
):
    from lobster.cli_internal.mcp_overview import key_presence

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("lobster.config.global_config.CONFIG_DIR", tmp_path / "g")
    ws = tmp_path / "p" / ".lobster_workspace"
    ws.mkdir(parents=True)
    (ws / ".env").write_text("LOBSTER_LAMIN=1\n")  # lobster init --lamin
    (tmp_path / "p" / ".env").write_text("ANTHROPIC_API_KEY=sk-x\n")

    key = key_presence("anthropic", ws)

    assert key["present"] is True
    assert key["found_in"] == str(tmp_path / "p" / ".env")
