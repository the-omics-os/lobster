"""Plan first, enforced: until the plan question is answered, only loading and
describing the input file runs.

With ``plan`` a first turn may load the file the host passed and describe it.
Every other tool call (custom code, analysis, QC, downloads, plots, a hand-off
to any expert but the loader) gets a refusal instead of running, and the plan
names what was refused. ``plan: false`` and continued turns are unchanged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from lobster.agents import graph
from tests.unit.mcp.replay_input import loading
from tests.unit.mcp.test_answer import McpProcess, _initialize
from tests.unit.mcp.test_run import data, run_args  # noqa: F401  (fixtures)


def _tool(name, delegate=False, coroutine=False):
    def ran(**kwargs) -> str:
        return f"{name} ran"

    async def ran_async(**kwargs) -> str:
        return f"{name} ran"

    metadata = {"categories": ["DELEGATE"] if delegate else ["ANALYZE"]}
    if coroutine:
        return StructuredTool.from_function(
            coroutine=ran_async, name=name, description=name, metadata=metadata
        )
    return StructuredTool.from_function(
        func=ran, name=name, description=name, metadata=metadata
    )


@pytest.fixture
def tools():
    named = {
        t.name: t
        for t in (
            _tool("handoff_to_data_expert_agent", delegate=True),
            _tool("handoff_to_transcriptomics_expert", delegate=True),
            _tool("load_modality"),
            _tool("get_modality_details"),
            _tool("assess_data_quality"),
            _tool("execute_custom_code", coroutine=True),
            _tool("ask_user", coroutine=True),
        )
    }
    agent = SimpleNamespace(
        nodes={"tools": SimpleNamespace(bound=SimpleNamespace(tools_by_name=named))}
    )
    graph._one_tool_at_a_time(agent)
    yield named
    graph.open_plan()


def _call(t):
    if t.func is None:
        return asyncio.run(t.ainvoke({}))
    return t.invoke({})


def test_until_the_plan_is_answered_only_loading_and_describing_run(tools):
    refused = []
    graph.hold_for_plan(refused.append)
    replies = {name: _call(t) for name, t in tools.items()}
    for name in (
        "handoff_to_data_expert_agent",
        "load_modality",
        "get_modality_details",
        "ask_user",
    ):
        assert replies[name] == f"{name} ran"
    for name in (
        "execute_custom_code",
        "assess_data_quality",
        "handoff_to_transcriptomics_expert",
    ):
        assert replies[name] == graph.PLAN_REFUSAL
    assert "answer the plan first" in graph.PLAN_REFUSAL
    assert sorted(refused) == [
        "assess_data_quality",
        "execute_custom_code",
        "handoff_to_transcriptomics_expert",
    ]

    graph.open_plan()
    assert {name: _call(t) for name, t in tools.items()} == {
        name: f"{name} ran" for name in tools
    }
    assert len(refused) == 3


def test_without_a_plan_nothing_is_refused(tools):
    assert {name: _call(t) for name, t in tools.items()} == {
        name: f"{name} ran" for name in tools
    }


def _code_steps(done):
    record = json.loads(Path(done["result_path"]).read_text())
    return [
        s["parameters"].get("description")
        for s in record["steps"]
        if s["tool"] == "execute_custom_code"
    ]


@pytest.mark.slow
def test_a_real_run_refuses_code_and_analysis_until_the_plan_is_answered(
    tmp_path, data
):
    proc = McpProcess(loading("plan_gate_process.json", tmp_path, data), tmp_path)
    try:
        _initialize(proc)
        run = proc.tool("lobster_run", run_args(data, tmp_path / "out"))
        waiting = proc.wait(run["run_id"])
        assert waiting["status"] == "needs_decision", waiting
        assert waiting["plan"] == {
            "requested": True,
            "asked": True,
            "refused": ["execute_custom_code", "handoff_to_transcriptomics_expert"],
        }
        decision = waiting["decision"]
        reply = proc.tool(
            "lobster_answer",
            {
                "run_id": run["run_id"],
                "decision_id": decision["decision_id"],
                "request_version": decision["request_version"],
                "answer": "Go ahead",
                "source": "model",
            },
        )
        assert reply["outcome"] == "answered"
        done = proc.wait(run["run_id"])
        assert done["status"] == "completed", done
        assert done["loaded_files"][0]["path"] == str(data.resolve())
        # The code refused before the answer never ran; the same code after it did.
        assert _code_steps(done) == ["after the plan"]
        assert done["plan"] == waiting["plan"]
        record = json.loads(Path(done["result_path"]).read_text())
        assert record["plan"] == waiting["plan"]
    finally:
        proc.close()


@pytest.mark.slow
def test_plan_false_runs_code_before_any_question(tmp_path, data):
    proc = McpProcess(loading("plan_free_process.json", tmp_path, data), tmp_path)
    try:
        _initialize(proc)
        args = {**run_args(data, tmp_path / "out"), "plan": False}
        run = proc.tool("lobster_run", args)
        done = proc.wait(run["run_id"])
        assert done["status"] == "completed", done
        assert _code_steps(done) == ["before any question"]
        assert done["plan"] == {"requested": False, "asked": False}
    finally:
        proc.close()
