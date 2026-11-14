"""A run names the Lobster experts it may use (``experts``); a continue keeps them.

A name that is not installed is refused before any work with the exact command
that installs it; one installed after the server loaded its experts is refused
with the plain statement that the server must be restarted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_contract, mcp_overview, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_continue import go_on
from tests.unit.mcp.test_publish import H5AD, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_restore import SESSION_FILE
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

INSTALLED = {"research_agent", "data_expert_agent", "transcriptomics_expert"}


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set(INSTALLED))
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(
        mcp_overview, "_lobster_path", lambda: "/opt/lobster/bin/lobster"
    )


@pytest.fixture
def built(monkeypatch):
    """Every client build, as (session folder, project workspace, experts)."""
    calls = []
    real = mcp_stdio._build_client

    def spy(workspace, project_ws, experts, **kwargs):
        calls.append((Path(workspace), project_ws, experts))
        return real(workspace, project_ws, experts, **kwargs)

    monkeypatch.setattr(mcp_stdio, "_build_client", spy)
    return calls


def test_run_lists_experts_in_its_schema():
    run = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"]
    experts = run["inputSchema"]["properties"]["experts"]
    assert experts["type"] == "array" and experts["items"]["type"] == "string"
    assert experts["minItems"] == 1
    assert "experts" not in run["inputSchema"]["required"]
    assert "lobster_overview" in experts["description"]


@pytest.mark.parametrize(
    "experts, said",
    [
        ([], "experts"),
        ("research_agent", "experts"),
        ([1], "experts"),
        (["research_agent", "research_agent"], "experts"),
        ([""], "experts"),
    ],
)
def test_a_bad_experts_list_is_refused(installed, tmp_path, data, experts, said):
    gate = Gate()
    args = {**run_args(data, tmp_path / "out"), "experts": experts}
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", args)
    assert refused["error_text"].startswith("bad_argument:")
    assert said in refused["error_text"] and gate.calls == []
    assert not (tmp_path / "out").exists()


def test_an_expert_that_is_not_installed_is_refused_with_the_install_line(
    installed, tmp_path, data
):
    gate = Gate()
    args = {
        **run_args(data, tmp_path / "out"),
        "experts": ["research_agent", "proteomics_expert", "no_such_expert"],
    }
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", args)["error_text"]
    assert refused.startswith("agent_not_available:")
    assert "proteomics_expert" in refused and "no_such_expert" in refused
    assert "/opt/lobster/bin/lobster agents install lobster-proteomics" in refused
    # for the name no package provides
    assert "/opt/lobster/bin/lobster agents list" in refused
    assert gate.calls == [] and not (tmp_path / "out").exists()


def test_an_expert_installed_after_the_server_loaded_its_experts_needs_a_restart(
    monkeypatch, tmp_path, data
):
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set(INSTALLED))
    monkeypatch.setattr(
        mcp_overview, "loaded_experts", lambda: INSTALLED - {"transcriptomics_expert"}
    )
    gate = Gate()
    args = {**run_args(data, tmp_path / "out"), "experts": ["transcriptomics_expert"]}
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", args)["error_text"]
    assert refused.startswith("restart_needed:")
    assert "transcriptomics_expert" in refused and "restart" in refused
    assert gate.calls == []


def test_the_run_is_built_with_its_experts_and_records_them(
    engine, installed, built, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    experts = ["data_expert_agent", "transcriptomics_expert", "research_agent"]
    args = {**run_args(data, tmp_path / "out"), "experts": experts}
    run = call(server, "lobster_run", args)
    assert run["experts"] == experts
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "completed"
    assert built[0][2] == tuple(experts)
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["request"]["experts"] == experts


def test_without_experts_the_project_decides(engine, installed, built, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert "experts" not in run
    done = wait_terminal(server, run["run_id"])
    assert built[0][2] is None
    record = json.loads(Path(done["result_path"]).read_text())
    assert "experts" not in record["request"]


def test_a_continue_after_a_restart_keeps_the_runs_experts(
    engine, scan, installed, built, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    script["files"] = {"data/pbmc.h5ad": H5AD, SESSION_FILE: b"{}"}
    out = tmp_path / "out"
    first = McpServer(ThreadRunner())
    args = {
        **run_args(data, out),
        "experts": ["data_expert_agent", "transcriptomics_expert"],
    }
    run = call(first, "lobster_run", args)
    assert wait_terminal(first, run["run_id"])["status"] == "completed"
    first.runner.stop(5)

    script["files"] = {}  # the folder already holds the session
    second = McpServer(ThreadRunner())
    turn = go_on(second, output_dir=str(out))
    done = wait_terminal(second, turn["run_id"])
    assert done["status"] == "completed", done
    team = ["data_expert_agent", "transcriptomics_expert"]
    assert [b[2] for b in built] == [tuple(team)] * 2
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["request"]["experts"] == team


@pytest.mark.parametrize(
    "name", ["research_agent\u001b[31m", "a b", "x" * 81, "../research_agent"]
)
def test_a_name_that_is_not_an_expert_name_is_refused_without_echo(
    installed, tmp_path, data, name
):
    gate = Gate()
    args = {**run_args(data, tmp_path / "out"), "experts": [name]}
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", args)["error_text"]
    assert refused.startswith("bad_argument: experts")
    assert name not in refused and "\u001b" not in refused
    assert gate.calls == []


def test_the_list_applies_and_the_finished_run_names_its_team(
    engine, installed, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    args = {
        **run_args(data, tmp_path / "out"),
        "experts": ["data_expert_agent", "research_agent"],
    }
    run = call(server, "lobster_run", args)
    assert "not_yet_applied" not in run
    done = wait_terminal(server, run["run_id"])
    assert "research_agent" in done["experts_loaded"]
