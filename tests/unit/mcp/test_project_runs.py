"""A run inside a Lobster project: its folder, its settings, its experts.

Without ``output_dir`` a run gets a new folder in the project's runs folder.
Any run whose folder is inside a project is built from that project's settings,
may name only experts the project has, and a folder the server made there is
recorded so that ``lobster project delete`` removes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_overview
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.core import projects
from tests.unit.mcp.test_continue import go_on
from tests.unit.mcp.test_experts import built  # noqa: F401  (fixture)
from tests.unit.mcp.test_publish import H5AD, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_restore import SESSION_FILE
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    QUESTION,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

STAMP = "20261113T200000Z"
INSTALLED = {"research_agent", "data_expert_agent", "transcriptomics_expert"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mcp_lock, "_stamp", lambda: STAMP)
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set(INSTALLED))
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(
        mcp_overview, "_lobster_path", lambda: "/opt/lobster/bin/lobster"
    )


@pytest.fixture
def project(tmp_path):
    made, _ = projects.init_project(tmp_path / "pbmc", name="PBMC")
    WorkspaceAgentConfig(enabled_agents=["research_agent", "data_expert_agent"]).save(
        made.workspace
    )
    return made


def created(project):
    path = project.workspace / projects.CREATED_PATHS_FILE
    return json.loads(path.read_text())["paths"] if path.exists() else []


def ask(server, data, **extra):
    return call(
        server, "lobster_run", {"question": QUESTION, "input_path": str(data), **extra}
    )


def test_output_dir_is_optional_in_the_schema():
    from lobster.cli_internal import mcp_contract

    run = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"]
    assert run["inputSchema"]["required"] == ["question"]
    assert (
        "runs folder" in run["inputSchema"]["properties"]["output_dir"]["description"]
    )


def test_without_output_dir_and_without_a_project_the_run_is_refused(tmp_path, data):
    gate = Gate()
    study = tmp_path / "study"  # tmp_path holds HOME: no project can go there
    study.mkdir()
    server = McpServer(ThreadRunner(gate), start=study)
    refused = ask(server, data)["error_text"]
    assert refused.startswith("bad_argument:")
    assert f"/opt/lobster/bin/lobster init {study}" in refused
    assert "output_dir" in refused
    assert gate.calls == []


def test_without_output_dir_a_run_gets_a_new_folder_in_the_runs_folder(
    engine, built, project, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    run = ask(server, data)
    out = project.root / "runs" / STAMP
    assert run["output_dir"] == str(out)
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "completed"
    assert Path(done["result_path"]).parent == out
    assert built[0][1] == project.workspace
    assert created(project) == [f"runs/{STAMP}"]

    second = ask(server, data)  # same second: the name is not reused
    assert second["output_dir"] == str(project.root / "runs" / f"{STAMP}-2")
    wait_terminal(server, second["run_id"])


def test_a_folder_the_host_names_in_the_project_is_built_from_it_and_recorded(
    engine, built, project, data, tmp_path
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=tmp_path)  # started elsewhere
    out = project.root / "runs" / "clustering"
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert built[0][1] == project.workspace
    assert created(project) == ["runs/clustering"]


def test_a_folder_that_existed_is_not_recorded(engine, built, project, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    out = project.root / "runs" / "mine"
    out.mkdir(parents=True)
    server = McpServer(ThreadRunner(), start=project.root)
    run = call(server, "lobster_run", run_args(data, out))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert created(project) == []


def test_a_folder_outside_any_project_has_no_project_settings(
    engine, built, data, tmp_path
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=tmp_path)
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert built[0][1] is None


def test_an_expert_the_project_lacks_is_refused_with_the_line_that_adds_it(
    project, data
):
    gate = Gate()
    server = McpServer(ThreadRunner(gate), start=project.root)
    refused = ask(server, data, experts=["research_agent", "transcriptomics_expert"])[
        "error_text"
    ]
    assert refused.startswith("agent_not_available:")
    assert "transcriptomics_expert" in refused
    again = (
        f"/opt/lobster/bin/lobster init {project.root} "
        "--agents research_agent,data_expert_agent,transcriptomics_expert"
    )
    assert f"run this line in a shell: {again}" in refused
    assert "keeps its id" in refused and "agents install" not in refused
    assert gate.calls == [] and not (project.root / "runs").exists()


def test_a_project_that_lists_no_experts_allows_every_installed_one(
    engine, built, project, data
):
    WorkspaceAgentConfig(enabled_agents=[]).save(project.workspace)
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    run = ask(server, data, experts=["data_expert_agent", "transcriptomics_expert"])
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    assert built[0][2] == ("data_expert_agent", "transcriptomics_expert")


def test_a_continue_after_a_restart_is_built_from_the_project_again(
    engine, scan, built, project, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD, SESSION_FILE: b"{}"}
    first = McpServer(ThreadRunner(), start=project.root)
    run = ask(first, data)
    assert wait_terminal(first, run["run_id"])["status"] == "completed"
    first.runner.stop(5)

    script["files"] = {}
    second = McpServer(ThreadRunner())
    turn = go_on(second, output_dir=run["output_dir"])
    assert wait_terminal(second, turn["run_id"])["status"] == "completed"
    assert [b[1] for b in built] == [project.workspace] * 2


@pytest.mark.parametrize(
    "where",
    ["runs/a/b", "runs", "notes/run", ".lobster_workspace/x", "data"],
)
def test_in_a_project_a_run_goes_one_level_into_its_runs_folder(project, data, where):
    gate = Gate()
    out = project.root / where
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", run_args(data, out))[
        "error_text"
    ]
    assert refused.startswith("path_not_allowed:")
    assert str(project.root / "runs" / "<name>") in refused
    assert gate.calls == [] and not out.exists()
    assert created(project) == []


def test_a_runs_folder_that_is_a_link_is_not_used(project, data, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (project.root / "runs").symlink_to(elsewhere)
    gate = Gate()
    refused = ask(McpServer(ThreadRunner(gate), start=project.root), data)
    assert refused["error_text"].startswith("path_not_allowed:")
    assert "link" in refused["error_text"]
    assert gate.calls == [] and list(elsewhere.iterdir()) == []


def test_the_reply_in_a_project_has_nothing_left_unapplied(engine, project, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner(), start=project.root)
    run = ask(server, data)
    assert "not_yet_applied" not in run
    wait_terminal(server, run["run_id"])
