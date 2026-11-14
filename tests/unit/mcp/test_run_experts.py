"""The experts a run uses: the run's ``experts`` list, else the project's list,
goes into the run's own session folder as its config.toml, which the engine
reads to build the team. A list replaces the whole team; the child experts of
a listed expert load with it. A file that does not load back as written, a
project file that cannot be read, a listed expert the license tier removes or
one that is not installed stop the run before it starts: never a run with
every expert or a silently smaller team. A run given a file whose list lacks
data_expert_agent, the expert that loads files, is refused before it starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_contract, mcp_overview, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.core import projects
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    call,
    data,
    engine,
    wait_terminal,
)

INSTALLED = {"research_agent", "data_expert_agent", "transcriptomics_expert"}
CHILDREN = {"transcriptomics_expert": ["annotation_expert", "de_analysis_expert"]}
PLANTED = "sk-planted-not-copied-0123"


@pytest.fixture
def team(monkeypatch):
    """Installed experts, their children and the tier filter, as the engine's
    registry would give them; ``denied`` holds names the tier removes."""
    denied = set()
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set(INSTALLED))
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(
        mcp_overview, "_lobster_path", lambda: "/opt/lobster/bin/lobster"
    )

    def expert_team(names):
        loaded = set(names)
        for name in names:
            loaded.update(CHILDREN.get(name, []))
        return sorted(loaded), sorted(n for n in names if n in denied)

    monkeypatch.setattr(mcp_stdio, "_expert_team", expert_team)
    return denied


@pytest.fixture
def project(tmp_path, monkeypatch):
    made, _ = projects.init_project(tmp_path / "study", name="Study")
    monkeypatch.chdir(made.root)
    return made


def run(server, data, **extra):
    """A run to its end; *data* None runs without a file, as a team without
    data_expert_agent must."""
    given = {"input_path": str(data)} if data is not None else {}
    started = call(
        server, "lobster_run", {"question": "q", "plan": False, **given, **extra}
    )
    return started, wait_terminal(server, started["run_id"])


def session_config(started) -> Path:
    return Path(started["output_dir"]) / ".lobster_workspace" / "config.toml"


def test_the_run_list_goes_into_the_session_and_the_status_names_the_team(
    engine, team, project, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, None, experts=["transcriptomics_expert"])
    assert done["status"] == "completed"
    written = WorkspaceAgentConfig.load(session_config(started).parent)
    assert written.enabled_agents == ["transcriptomics_expert"]
    assert done["experts_loaded"] == [
        "annotation_expert",
        "de_analysis_expert",
        "transcriptomics_expert",
    ]
    assert "not_yet_applied" not in started


def test_without_a_run_list_the_project_list_and_its_settings_go_in(
    engine, team, project, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    (project.workspace / "config.toml").write_text(
        'config_version = "1.0"\n'
        'enabled = ["research_agent", "data_expert_agent"]\n'
        "[agent_settings.research_agent]\n"
        'model = "gemini-y"\n'
    )
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, data)
    written = WorkspaceAgentConfig.load(session_config(started).parent)
    assert written.enabled_agents == ["research_agent", "data_expert_agent"]
    assert written.agent_settings["research_agent"].model == "gemini-y"
    assert done["experts_loaded"] == ["data_expert_agent", "research_agent"]


def test_with_no_list_anywhere_nothing_is_written(engine, team, project, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    started, done = run(McpServer(ThreadRunner(), start=project.root), data)
    assert done["status"] == "completed"
    assert not session_config(started).exists()
    assert "experts_loaded" not in done


def test_a_project_list_with_an_expert_not_installed_is_refused_by_name(
    engine, team, project, data
):
    WorkspaceAgentConfig(enabled_agents=["research_agent", "proteomics_expert"]).save(
        project.workspace
    )
    refused = call(
        McpServer(ThreadRunner(), start=project.root),
        "lobster_run",
        {"question": "q", "input_path": str(data)},
    )["error_text"]
    assert refused.startswith("agent_not_available:")
    assert "proteomics_expert" in refused and "lobster agents install" in refused


def test_a_project_file_that_cannot_be_read_stops_the_run(engine, team, project, data):
    (project.workspace / "config.toml").write_text('enabled = ["research_agent"\n')
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, None, experts=["research_agent"])
    assert done["error"]["code"] == "project_settings_unreadable"
    assert str(project.workspace / "config.toml") in done["error"]["message"]
    assert not session_config(started).parent.exists()


def test_a_written_file_that_does_not_load_back_stops_the_run(
    engine, team, project, data, monkeypatch
):
    def broken(self, workspace_path):
        path = Path(workspace_path) / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enabled = [\n")
        return path

    monkeypatch.setattr(WorkspaceAgentConfig, "save", broken)
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, None, experts=["research_agent"])
    assert done["error"]["code"] == "expert_list_not_applied"
    assert not session_config(started).parent.exists()


def test_a_listed_expert_the_license_tier_removes_stops_the_run(
    engine, team, project, data
):
    team.add("transcriptomics_expert")
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, None, experts=["transcriptomics_expert"])
    assert done["error"]["code"] == "expert_not_in_tier"
    assert "transcriptomics_expert" in done["error"]["message"]


def test_nothing_that_looks_like_a_key_is_carried(engine, team, project, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    (project.workspace / "config.toml").write_text(
        'config_version = "1.0"\n'
        'enabled = ["research_agent", "data_expert_agent"]\n'
        f'api_key = "{PLANTED}"\n'
        "[provider]\n"
        'default = "gemini"\n'
        f'api_key = "{PLANTED}"\n'
        "[agent_settings.research_agent]\n"
        'model = "gemini-y"\n'
        f'api_key = "{PLANTED}"\n'
    )
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, data)
    assert PLANTED not in json.dumps([started, done])
    for file in Path(started["output_dir"]).rglob("*"):
        if file.is_file():
            assert PLANTED.encode() not in file.read_bytes(), file


def test_the_texts_say_a_list_replaces_the_team_and_who_loads_files(team, project):
    run_tool = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"]
    about = " ".join(
        run_tool["inputSchema"]["properties"]["experts"]["description"].split()
    )
    assert "replaces the whole team" in about and "data_expert_agent" in about
    seen = call(McpServer(ThreadRunner(), start=project.root), "lobster_overview", {})
    assert "replaces the whole team" in seen["experts"]["note"]
    assert "not_yet_applied" not in seen["experts"]
    (cont,) = [t for t in mcp_contract.list_tools() if t["name"] == "lobster_continue"]
    assert "keeps the provider, model and experts of its first run" in " ".join(
        cont["description"].split()
    )


def test_the_real_registry_names_children_and_the_tier():
    """The engine's own registry and tier filter, once (slow: loads every
    installed expert)."""
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    parent = next((n for n, c in agents.items() if c.child_agents), None)
    if parent is None:
        pytest.skip("no installed expert has child experts")
    loaded, removed = mcp_stdio._expert_team([parent])
    assert parent in loaded and removed in ([], [parent])
    for child in agents[parent].child_agents:
        assert (child in loaded) == (child in agents)


def test_a_table_of_keys_the_schema_ignores_does_not_stop_the_run(
    engine, team, project, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    (project.workspace / "config.toml").write_text(
        'enabled = ["research_agent", "data_expert_agent"]\n'
        '[provider]\nsomething_else = "x"\n'
    )
    started, done = run(McpServer(ThreadRunner(), start=project.root), data)
    assert done["status"] == "completed", done
    written = WorkspaceAgentConfig.load(session_config(started).parent)
    assert written.enabled_agents == ["research_agent", "data_expert_agent"]


def test_a_continued_turn_names_its_team_too(engine, team, project, data):
    from tests.unit.mcp.test_continue import go_on

    opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    started, done = run(server, None, experts=["transcriptomics_expert"])
    opened[-1]["client"].events[:] = [COMPLETE]
    turn = wait_terminal(server, go_on(server, run_id=started["run_id"])["run_id"])
    assert turn["status"] == "completed"
    assert turn["experts_loaded"] == done["experts_loaded"]


def test_the_overview_says_a_run_refuses_until_a_file_is_repaired(team, project):
    (project.workspace / "config.toml").write_text("enabled = [\n")
    seen = call(McpServer(ThreadRunner(), start=project.root), "lobster_overview", {})
    assert seen["next"].startswith(
        f"A run refuses until {project.workspace / 'config.toml'} is repaired"
    )


# a file to analyse needs the expert that loads files


def test_a_run_list_without_the_data_expert_refuses_a_file(engine, team, project, data):
    server = McpServer(ThreadRunner(), start=project.root)
    text = call(
        server,
        "lobster_run",
        {"question": "q", "input_path": str(data), "experts": ["research_agent"]},
    )["error_text"]
    assert text.startswith(f"data_expert_needed: the file at {data} would not be ")
    assert "Add data_expert_agent to experts" in text
    assert not list((project.root / "runs").glob("*"))


def test_a_project_list_without_the_data_expert_refuses_a_file_with_the_init_line(
    engine, team, project, data
):
    (project.workspace / "config.toml").write_text('enabled = ["research_agent"]\n')
    text = call(
        McpServer(ThreadRunner(), start=project.root),
        "lobster_run",
        {"question": "q", "input_path": str(data)},
    )["error_text"]
    assert text.startswith("data_expert_needed: ")
    again = mcp_overview.lobster_command(
        "init", str(project.root), "--agents", "research_agent,data_expert_agent"
    )
    assert again in text


@pytest.mark.parametrize("listed", ["run", "project"])
def test_without_the_data_expert_installed_the_refusal_names_the_install_line(
    engine, team, project, data, monkeypatch, listed
):
    monkeypatch.setattr(
        mcp_overview,
        "installed_experts",
        lambda: set(INSTALLED) - {"data_expert_agent"},
    )
    args = {"question": "q", "input_path": str(data)}
    if listed == "run":
        args["experts"] = ["research_agent"]
    else:
        (project.workspace / "config.toml").write_text('enabled = ["research_agent"]\n')
    text = call(McpServer(ThreadRunner(), start=project.root), "lobster_run", args)[
        "error_text"
    ]
    assert text.startswith("data_expert_needed: ")
    assert mcp_overview.install_line("lobster-research") in text
    assert "restart the Lobster MCP server" in text
    assert "--agents" not in text and "Add data_expert_agent" not in text


@pytest.mark.parametrize(
    "experts", [None, ["data_expert_agent", "transcriptomics_expert"]]
)
def test_no_list_or_a_list_with_the_data_expert_takes_a_file(
    engine, team, project, data, experts
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    extra = {"experts": experts} if experts else {}
    _started, done = run(McpServer(ThreadRunner(), start=project.root), data, **extra)
    assert done["status"] == "completed"


def test_a_list_without_the_data_expert_runs_without_a_file(engine, team, project):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    started = call(
        server,
        "lobster_run",
        {"question": "q", "plan": False, "experts": ["research_agent"]},
    )
    assert wait_terminal(server, started["run_id"])["status"] == "completed"


def test_the_texts_name_the_rule():
    run = next(t for t in mcp_contract.TOOLS if t["name"] == "lobster_run")
    said = run["inputSchema"]["properties"]["experts"]["description"]
    assert "with input_path, a list without it is refused" in said
    assert "input_path whose list lacks it is refused" in mcp_overview.TEAM_NOTE
