"""Plan first: a first run asks to approve its plan before any expert works.

With ``plan`` (the default) the first turn's prompt tells the supervisor to
write a short plan and ask for approval with ``ask_user`` before it hands a
step to an expert. The status says whether that happened (``plan.asked``):
the first question of the turn came before any expert started. Nothing is
inferred beyond that, and a later turn never plans.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_contract, mcp_record
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_answer import answer_args, interrupt
from tests.unit.mcp.test_continue import go_on
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Gate,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

PLAN = interrupt(
    "text_input",
    {"question": "Plan: research_agent finds ... Go ahead?"},
    "Plan: research_agent finds ... Go ahead?",
)
EXPERT = {
    "type": "agent_change",
    "agent": "transcriptomics_expert",
    "status": "working",
}


def waiting(server, run_id):
    for _ in range(400):
        status = call(server, "lobster_status", {"run_id": run_id})
        if status["status"] != "running":
            return status
        import time

        time.sleep(0.025)
    raise AssertionError("the run never asked")


def test_plan_is_a_boolean_that_defaults_to_true():
    run = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"]
    plan = run["inputSchema"]["properties"]["plan"]
    assert plan["type"] == "boolean" and plan["default"] is True
    assert "plan" not in run["inputSchema"]["required"]
    for words in ("show the plan to the user", "source"):
        assert words in run["description"], words


def test_the_first_prompt_asks_for_a_plan_before_any_expert():
    prompt = mcp_record.compose_question_prompt("q", None, plan=True)
    assert mcp_record.PLAN_FIRST in prompt
    assert prompt.index(mcp_record.PLAN_FIRST) < prompt.index('{"question"')
    for words in ("experts", "steps", "data", "open questions", "ask_user"):
        assert words in mcp_record.PLAN_FIRST, words
    assert "time" in mcp_record.PLAN_FIRST and "estimate" in mcp_record.PLAN_FIRST
    assert mcp_record.PLAN_FIRST not in mcp_record.compose_question_prompt(
        "q", None, plan=False
    )
    assert mcp_record.PLAN_FIRST not in mcp_record.compose_continue_prompt("go on")


def test_an_approved_plan_is_recorded_as_asked(engine, tmp_path, data):
    opened, script = engine
    script["events"] = [PLAN]
    script["resume_events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    asked = waiting(server, run["run_id"])
    assert asked["status"] == "needs_decision"
    assert asked["plan"] == {"requested": True, "asked": True}
    assert mcp_record.PLAN_FIRST in opened[0]["client"].prompts[0]

    call(
        server, "lobster_answer", answer_args(run["run_id"], asked, "Go ahead", "model")
    )
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "completed"
    assert done["plan"] == {"requested": True, "asked": True}
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["plan"] == {"requested": True, "asked": True}
    assert record["request"]["plan"] is True
    assert [d["source"] for d in record["decisions"]] == ["model"]


def test_a_question_after_an_expert_started_is_not_the_plan(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [EXPERT, PLAN]
    script["resume_events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    asked = waiting(server, run["run_id"])
    assert asked["plan"] == {"requested": True, "asked": False}
    call(server, "lobster_answer", answer_args(run["run_id"], asked, "yes"))
    assert wait_terminal(server, run["run_id"])["plan"]["asked"] is False


def test_a_run_that_never_asks_says_so(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    done = wait_terminal(server, run["run_id"])
    assert done["plan"] == {"requested": True, "asked": False}


def test_plan_false_skips_the_plan(engine, tmp_path, data):
    opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    args = {**run_args(data, tmp_path / "out"), "plan": False}
    run = call(server, "lobster_run", args)
    done = wait_terminal(server, run["run_id"])
    assert done["plan"] == {"requested": False, "asked": False}
    assert mcp_record.PLAN_FIRST not in opened[0]["client"].prompts[0]
    record = json.loads(Path(done["result_path"]).read_text())
    assert record["request"]["plan"] is False


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_a_plan_that_is_not_a_boolean_is_refused(tmp_path, data, value):
    gate = Gate()
    args = {**run_args(data, tmp_path / "out"), "plan": value}
    refused = call(McpServer(ThreadRunner(gate)), "lobster_run", args)
    assert refused["error_text"].startswith("bad_argument: plan")
    assert gate.calls == []


def test_a_later_turn_never_plans(engine, tmp_path, data):
    opened, script = engine
    script["events"] = [COMPLETE]
    script["resume_events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    wait_terminal(server, run["run_id"])
    turn = go_on(server, run_id=run["run_id"])
    done = wait_terminal(server, turn["run_id"])
    assert "plan" not in done
    assert mcp_record.PLAN_FIRST not in opened[0]["client"].prompts[1]
