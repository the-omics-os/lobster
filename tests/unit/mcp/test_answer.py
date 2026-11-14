"""A host answers a question Lobster asks during an MCP run (``lobster_answer``).

The scripted client stands in for the engine session; the run API, the thread
runner and the server are the production code. The two ``slow`` tests drive the
real engine on a replay script (no network, no model).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import pytest

from lobster.cli_internal import mcp_decision, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.core.runs import runs_for
from tests.unit.mcp.replay_input import loading
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    REPLAY,
    Clock,
    Hold,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

SLIDER_PROMPT = "Which mitochondrial threshold? (range: 0.0-1.0, default: 0.05)"
ALREADY_ANSWERED = (
    "already_answered: this question was already answered; read lobster_status "
    "for the state of the run."
)


def interrupt(component, payload, fallback, ask_id="call_1", interrupt_id="lg_1"):
    return {
        "type": "interrupt",
        "interrupt_id": interrupt_id,
        "data": {
            "component": component,
            "data": payload,
            "fallback_prompt": fallback,
            "ask_id": ask_id,
        },
    }


SLIDER = interrupt(
    "threshold_slider",
    {
        "label": "Which mitochondrial threshold?",
        "min": 0.0,
        "max": 1.0,
        "default": 0.05,
        "unit": "",
    },
    SLIDER_PROMPT,
)
SELECT = interrupt(
    "select",
    {"question": "Which normalization?", "options": ["DESeq2", "CPM", "TMM"]},
    "Which normalization?\n  1. DESeq2\n  2. CPM\n  3. TMM\nEnter number: ",
)


ABSENT = object()


def initialize(server: McpServer, name="claude-code") -> None:
    info = {} if name is ABSENT else {"name": name, "version": "1"}
    server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": info,
                },
            }
        )
    )


@pytest.fixture
def server(engine):
    srv = McpServer(ThreadRunner())
    initialize(srv)
    yield srv
    srv.runner.stop(5)


def start(server, tmp_path, data, name="out"):
    return call(server, "lobster_run", run_args(data, tmp_path / name))["run_id"]


def wait_waiting(server, run_id, timeout=30, version=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = call(server, "lobster_status", {"run_id": run_id})
        if status["status"] != "running" or (
            version is not None and status.get("decision")
        ):
            if status["status"] == "needs_decision" and (
                version is None or status["decision"]["request_version"] == version
            ):
                return status
            if status["status"] != "running":
                break
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} is not waiting: {status}")


def answer_args(run_id, waiting, answer, source="human", **extra):
    decision = waiting["decision"]
    return {
        "run_id": run_id,
        "decision_id": decision["decision_id"],
        "request_version": decision["request_version"],
        "answer": answer,
        "source": source,
        **extra,
    }


# 1. the status of a waiting run -------------------------------------------


def test_needs_decision_status_carries_the_question_and_no_poll_hint(
    engine, server, tmp_path, data
):
    _opened, script = engine
    script["events"] = [SLIDER]
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    decision = waiting["decision"]
    assert waiting == {
        "status": "needs_decision",
        "run_id": run_id,
        "elapsed_s": waiting["elapsed_s"],
        "phase": "waiting",
        # the first question came before any expert: the plan's approval
        "plan": {"requested": True, "asked": True},
        "decision": {
            "decision_id": decision["decision_id"],
            "request_version": 1,
            "question": "Which mitochondrial threshold?",
            "kind": "threshold_slider",
            "min": 0.0,
            "max": 1.0,
            "default": 0.05,
            "fallback_prompt": SLIDER_PROMPT,
            "answer_shapes": decision["answer_shapes"],
        },
    }
    assert decision["decision_id"].startswith("dec_")
    assert isinstance(waiting["elapsed_s"], int)
    assert decision["answer_shapes"] and all(
        isinstance(shape, str) for shape in decision["answer_shapes"]
    )
    text = json.dumps(waiting)
    assert "lg_1" not in text and "interrupt_id" not in text
    assert "poll_after_s" not in waiting


@pytest.mark.parametrize(
    "event, kind, extra",
    [
        (SELECT, "select", {"options": ["DESeq2", "CPM", "TMM"]}),
        (
            interrupt(
                "confirm", {"question": "Proceed?", "default": False}, "Proceed?"
            ),
            "confirm",
            {"default": False},
        ),
        (
            interrupt("text_input", {"question": "Name?", "placeholder": ""}, "Name?"),
            "text_input",
            {},
        ),
        (
            interrupt(
                "cell_type_selector",
                {"clusters": [{"id": 0, "size": 5, "markers": ["CD3E"]}]},
                "Label the clusters",
            ),
            "cell_type_selector",
            {},
        ),
    ],
)
def test_needs_decision_status_per_component_kind(
    engine, server, tmp_path, data, event, kind, extra
):
    _opened, script = engine
    script["events"] = [event]
    run_id = start(server, tmp_path, data)
    decision = wait_waiting(server, run_id)["decision"]
    assert decision["kind"] == kind
    if kind == "select":  # a numbered "Enter number" prompt would invite an index
        assert "fallback_prompt" not in decision
        assert "Enter number" not in json.dumps(decision)
        assert not any("index" in shape for shape in decision["answer_shapes"])
    else:
        assert decision["fallback_prompt"] == event["data"]["fallback_prompt"]
    optional = {"options", "min", "max", "default"} & set(decision)
    assert optional == set(extra)
    for key, value in extra.items():
        assert decision[key] == value


def test_a_waiting_run_has_no_timer(engine, server, tmp_path, data, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mcp_stdio, "_now", clock)
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    clock.now += 10 * 86400
    # Real time too: the worker wakes once a second to re-read its cancel
    # event; two and a half seconds see it wake at least twice, and a timer on
    # the real clock (or on the wake-ups) shorter than that would fire.
    time.sleep(2.5)
    later = call(server, "lobster_status", {"run_id": run_id})
    assert later["status"] == "needs_decision"
    assert later["decision"] == waiting["decision"]
    assert later["elapsed_s"] == waiting["elapsed_s"] + 10 * 86400
    reply = call(server, "lobster_answer", answer_args(run_id, later, 0.1, "model"))
    assert reply["outcome"] == "answered"
    assert wait_terminal(server, run_id)["status"] == "completed"


# 2. answer, then the run completes ----------------------------------------


def test_answer_completes_the_run_and_records_who_said_so(
    engine, server, tmp_path, data
):
    opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    decision_id = waiting["decision"]["decision_id"]

    reply = call(
        server,
        "lobster_answer",
        answer_args(run_id, waiting, 0.1, "model", comment="per lab SOP"),
    )
    assert reply == {
        "run_id": run_id,
        "outcome": "answered",
        "source": "model",
        "answer": {"value": 0.1},
    }
    done = wait_terminal(server, run_id)
    assert done["status"] == "completed" and done["run_id"] == run_id
    assert done["summary"] == COMPLETE["response"]

    client = opened[0]["client"]
    assert client.resumes == [
        {
            "lg_1": {
                "__lobster_decision__": {
                    "decision_id": decision_id,
                    "request_version": 1,
                    "answer": {"value": 0.1},
                    "source": "model",
                    "comment": "per lab SOP",
                    "channel": "mcp:claude-code",
                }
            }
        }
    ]
    # The handle stays the origin run; the engine wrote the answer's run id.
    manager = runs_for(client)
    on_disk = json.loads(Path(done["result_path"]).read_text())
    assert done["result"]["run_id"] == on_disk["run_id"] == manager.latest_run_id
    assert on_disk["run_id"] != run_id
    assert on_disk["response"] == COMPLETE["response"]
    assert "response" not in done["result"]
    assert manager.runs[run_id].resumed_by == on_disk["run_id"]


@pytest.mark.parametrize("source", ["human", "model"])
def test_source_reaches_the_record_as_told(engine, server, tmp_path, data, source):
    opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    call(server, "lobster_answer", answer_args(run_id, waiting, 0.1, source))
    wait_terminal(server, run_id)
    envelope = opened[0]["client"].resumes[0]["lg_1"]["__lobster_decision__"]
    assert envelope["source"] == source
    assert "comment" in envelope and envelope["comment"] is None


@pytest.mark.parametrize(
    "client_name, channel",
    [
        ("claude-code", "mcp:claude-code"),
        ("Claude Code", "mcp:Claude-Code"),
        ("evil\n# Answer: 'x'", "mcp:evil-Answer-x"),
        ("x" * 200, "mcp:" + "x" * 40),
        ("", "mcp:unknown"),
        (None, "mcp:unknown"),
        (7, "mcp:unknown"),
        (ABSENT, "mcp:unknown"),
    ],
)
def test_channel_is_the_sanitised_host_name(
    engine, tmp_path, data, client_name, channel
):
    opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    srv = McpServer(ThreadRunner())
    initialize(srv, client_name)
    run_id = start(srv, tmp_path, data)
    waiting = wait_waiting(srv, run_id)
    call(srv, "lobster_answer", answer_args(run_id, waiting, 0.1, "human"))
    wait_terminal(srv, run_id)
    envelope = opened[0]["client"].resumes[0]["lg_1"]["__lobster_decision__"]
    assert envelope["channel"] == channel


def test_channel_is_unknown_when_the_host_never_initialized(engine, tmp_path, data):
    opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    srv = McpServer(ThreadRunner())
    run_id = start(srv, tmp_path, data)
    waiting = wait_waiting(srv, run_id)
    call(srv, "lobster_answer", answer_args(run_id, waiting, 0.1, "human"))
    wait_terminal(srv, run_id)
    envelope = opened[0]["client"].resumes[0]["lg_1"]["__lobster_decision__"]
    assert envelope["channel"] == "mcp:unknown"


# 3. refusals: each one changes nothing ------------------------------------


def test_each_refusal_leaves_the_run_waiting_and_answerable(
    engine, server, tmp_path, data
):
    opened, script = engine
    script.update(events=[SELECT], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    ok = answer_args(run_id, waiting, "CPM")
    source_text = "bad_answer: source is required and must be 'human' or 'model'."
    options = json.dumps(["DESeq2", "CPM", "TMM"])
    refusals = [
        ({**ok, "run_id": "nope"}, "Unknown run_id 'nope'"),
        (
            {**ok, "decision_id": "dec_nope"},
            f"unknown_decision: no open decision dec_nope on run {run_id}; "
            "read lobster_status again.",
        ),
        (
            {**ok, "request_version": 2},
            "rejected_stale: the question is now at request_version 1; read "
            "lobster_status and answer that version.",
        ),
        ({**ok, "source": "robot"}, source_text),
        ({k: v for k, v in ok.items() if k != "source"}, source_text),
        (
            {k: v for k, v in ok.items() if k != "answer"},
            "bad_answer: answer is required (a string, number, boolean or object).",
        ),
        (
            {**ok, "answer": ["CPM"]},
            "bad_answer: answer is required (a string, number, boolean or object).",
        ),
        (
            {k: v for k, v in ok.items() if k != "decision_id"},
            "bad_answer: decision_id is required and must be a string.",
        ),
        (
            {**ok, "request_version": "1"},
            "bad_answer: request_version is required and must be an integer.",
        ),
        ({**ok, "comment": 3}, "bad_answer: comment must be a string."),
        (
            {**ok, "answer": "XYZ"},
            f"bad_answer: 'XYZ' is not one of the options {options}; answer "
            "with one option text.",
        ),
    ]
    for arguments, text in refusals:
        refused = call(server, "lobster_answer", arguments)
        assert refused == {"error_text": text}, text
        still = call(server, "lobster_status", {"run_id": run_id})
        assert still["status"] == "needs_decision"
        assert still["decision"] == waiting["decision"]
        assert opened[0]["client"].resumes == []
    reply = call(server, "lobster_answer", ok)
    assert reply["outcome"] == "answered"
    assert wait_terminal(server, run_id)["status"] == "completed"


def test_answer_to_a_run_that_is_not_waiting(engine, server, tmp_path, data):
    _opened, script = engine
    hold = Hold()
    script.update(events=[hold, SELECT], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    assert hold.entered.wait(10)
    ask = {
        "run_id": run_id,
        "decision_id": "dec_x",
        "request_version": 1,
        "answer": "CPM",
        "source": "human",
    }
    refused = call(server, "lobster_answer", ask)
    assert refused == {
        "error_text": f"not_waiting: run {run_id} is running; there is no "
        "question to answer."
    }
    hold.release.set()
    waiting = wait_waiting(server, run_id)
    call(server, "lobster_answer", answer_args(run_id, waiting, "CPM"))
    assert wait_terminal(server, run_id)["status"] == "completed"
    refused = call(server, "lobster_answer", ask)
    assert refused == {
        "error_text": f"not_waiting: run {run_id} is completed; there is no "
        "question to answer."
    }


def test_second_answer_to_the_same_question_is_refused(engine, server, tmp_path, data):
    opened, script = engine
    hold = Hold()
    script.update(events=[SELECT], resume_events=[hold, COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    first = call(server, "lobster_answer", answer_args(run_id, waiting, "CPM"))
    assert first["outcome"] == "answered"
    assert hold.entered.wait(10)
    again = call(server, "lobster_answer", answer_args(run_id, waiting, "TMM"))
    assert again == {"error_text": ALREADY_ANSWERED}
    assert len(opened[0]["client"].resumes) == 1
    hold.release.set()
    assert wait_terminal(server, run_id)["status"] == "completed"
    # The wording stays true once the run has finished.
    finished = call(server, "lobster_answer", answer_args(run_id, waiting, "TMM"))
    assert finished == {"error_text": ALREADY_ANSWERED}
    assert "continues" not in ALREADY_ANSWERED


def test_a_question_that_comes_back_has_the_next_version(
    engine, server, tmp_path, data
):
    opened, script = engine
    again = interrupt(
        "threshold_slider", SLIDER["data"]["data"], SLIDER_PROMPT, interrupt_id="lg_2"
    )
    script.update(events=[SLIDER], resume_events=[[again], [COMPLETE]])
    run_id = start(server, tmp_path, data)
    first = wait_waiting(server, run_id, version=1)
    call(server, "lobster_answer", answer_args(run_id, first, 0.1))
    second = wait_waiting(server, run_id, version=2)
    assert second["decision"]["decision_id"] == first["decision"]["decision_id"]
    stale = call(server, "lobster_answer", answer_args(run_id, first, 0.2))
    assert stale == {
        "error_text": "rejected_stale: the question is now at request_version 2; "
        "read lobster_status and answer that version."
    }
    reply = call(server, "lobster_answer", answer_args(run_id, second, 0.2))
    assert reply["outcome"] == "answered"
    assert wait_terminal(server, run_id)["status"] == "completed"
    # The second resume is keyed by the question's newest interrupt.
    assert list(opened[0]["client"].resumes[1]) == ["lg_2"]


# 5. cancel while waiting ---------------------------------------------------


def test_cancel_while_waiting_abandons_the_decision(engine, server, tmp_path, data):
    opened, script = engine
    script["events"] = [SLIDER]
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    outcome = call(server, "lobster_cancel", {"run_id": run_id})
    assert outcome == {"run_id": run_id, "outcome": "cancel_requested"}
    done = wait_terminal(server, run_id)
    assert done["status"] == "cancelled"
    assert done["cancel_reason"] == "decision_abandoned"
    assert "Which mitochondrial threshold?" in done["summary"]
    assert "decision" not in done and "dec_" not in json.dumps(done)

    manager = runs_for(opened[0]["client"])
    assert manager.runs[run_id].events[-1]["reason"] == "decision_abandoned"
    refused = call(server, "lobster_answer", answer_args(run_id, waiting, 0.1))
    assert refused == {
        "error_text": "rejected_cancelled: the run was stopped; start a new run."
    }
    assert opened[0]["client"].resumes == []
    again = call(server, "lobster_cancel", {"run_id": run_id})
    assert again == {"run_id": run_id, "outcome": "already_terminal"}


def test_cancel_after_an_answer_stops_the_resumed_run(engine, server, tmp_path, data):
    from tests.unit.mcp.test_run import WaitForCancel

    _opened, script = engine
    step = WaitForCancel()
    script.update(events=[SLIDER], resume_events=[step])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    call(server, "lobster_answer", answer_args(run_id, waiting, 0.1))
    assert step.entered.wait(10)
    call(server, "lobster_cancel", {"run_id": run_id})
    done = wait_terminal(server, run_id)
    assert done["status"] == "cancelled" and done["cancel_reason"] == "cancel"


# 6. EOF while waiting -------------------------------------------------------


def test_eof_while_waiting_abandons_the_decision_within_the_bound(
    engine, tmp_path, data, monkeypatch
):
    _opened, script = engine
    script["events"] = [SLIDER]

    def no_exit(code):  # run_mcp_stdio leaves with os._exit only if stop fails
        raise AssertionError("the waiting worker did not stop within the bound")

    monkeypatch.setattr(os, "_exit", no_exit)
    runner = ThreadRunner()
    out = io.StringIO()
    run_line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lobster_run", "arguments": run_args(data, tmp_path)},
        }
    )

    marks = {}

    def stdin():
        yield run_line + "\n"
        run_id = json.loads(json.loads(out.getvalue())["result"]["content"][0]["text"])[
            "run_id"
        ]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if runner.status(run_id)["status"] == "needs_decision":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("never reached needs_decision")
        # EOF is the host closing the pipe now: the bound starts here, not at
        # session start-up (which has no bound and depends on machine load).
        marks["eof"] = time.monotonic()

    assert mcp_stdio.run_mcp_stdio(runner=runner, stdin=stdin(), stdout=out) == 0
    assert time.monotonic() - marks["eof"] < mcp_stdio.SHUTDOWN_WAIT_S
    run_id = json.loads(json.loads(out.getvalue())["result"]["content"][0]["text"])[
        "run_id"
    ]
    final = runner.status(run_id)
    assert final["status"] == "cancelled"
    assert final["cancel_reason"] == "decision_abandoned"


# 7. one run at a time -------------------------------------------------------


def test_a_waiting_run_is_busy_and_the_next_run_starts_after_it(
    engine, server, tmp_path, data
):
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data, "a")
    waiting = wait_waiting(server, run_id)
    busy = call(server, "lobster_run", run_args(data, tmp_path / "b"))
    assert busy["error_text"].startswith(f"engine_busy: run {run_id} is waiting")
    assert "lobster_answer" in busy["error_text"]
    assert "lobster_cancel" in busy["error_text"]

    call(server, "lobster_answer", answer_args(run_id, waiting, 0.1))
    assert wait_terminal(server, run_id)["status"] == "completed"
    second = start(server, tmp_path, data, "b")
    wait_waiting(server, second)
    call(server, "lobster_cancel", {"run_id": second})
    assert wait_terminal(server, second)["cancel_reason"] == "decision_abandoned"
    assert start(server, tmp_path, data, "c")


# two questions at once ------------------------------------------------------


def test_two_questions_at_once_cancel_with_decision_unsupported(
    engine, server, tmp_path, data
):
    _opened, script = engine
    second = interrupt(
        "threshold_slider",
        {"label": "Which gene threshold?", "min": 0, "max": 1, "default": 0.5},
        "Which gene threshold?",
        ask_id="call_2",
        interrupt_id="lg_2",
    )
    script["events"] = [SLIDER, second]
    run_id = start(server, tmp_path, data)
    done = wait_terminal(server, run_id)
    assert done["status"] == "cancelled"
    assert done["cancel_reason"] == "decision_unsupported"


# the answer shapes -------------------------------------------------------------


def view(kind, **data):
    return {"kind": kind, **data}


@pytest.mark.parametrize(
    "decision, answer, recorded",
    [
        (view("select", options=["A", "B", "C"]), "B", {"selected": "B", "index": 1}),
        (
            view("select", options=["1", "2", "3"]),
            {"selected": "2"},
            {"selected": "2", "index": 1},
        ),
        (
            view("select", options=["a", "a"]),
            {"selected": "a", "index": 1},
            {"selected": "a", "index": 0},
        ),
        (view("select", options=["a", "a"]), "a", {"selected": "a", "index": 0}),
        (view("select", options=[3, 5]), {"selected": 3}, {"selected": 3, "index": 0}),
        (
            view("select", options=["A", "B", "C"]),
            {"selected": "B"},
            {"selected": "B", "index": 1},
        ),
        (
            view("select", options=["A", "B", "C"]),
            {"selected": "B", "index": 1},
            {"selected": "B", "index": 1},
        ),
        (view("confirm"), True, {"confirmed": True}),
        (view("confirm"), {"confirmed": False}, {"confirmed": False}),
        (view("confirm"), "yes", {"confirmed": True}),
        (view("confirm"), "No", {"confirmed": False}),
        (view("text_input"), "hello", {"answer": "hello"}),
        (view("text_input"), {"answer": "hello"}, {"answer": "hello"}),
        (view("threshold_slider", min=0.0, max=1.0), 0.1, {"value": 0.1}),
        (view("threshold_slider", min=0.0, max=1.0), 1, {"value": 1.0}),
        (view("threshold_slider", min=0.0, max=1.0), {"value": 0.5}, {"value": 0.5}),
        (view("threshold_slider"), 7, {"value": 7.0}),
        (
            view("cell_type_selector"),
            {"assignments": {"0": "T cells"}},
            {"assignments": {"0": "T cells"}},
        ),
        (
            view("cell_type_selector"),
            {"0": "T cells"},
            {"assignments": {"0": "T cells"}},
        ),
        (view("something_new"), {"any": [1, 2]}, {"any": [1, 2]}),
        (view(None), "free text", "free text"),
    ],
)
def test_answers_are_recorded_in_the_form_the_tui_components_return(
    decision, answer, recorded
):
    assert mcp_decision.check_answer(decision, answer) == recorded


@pytest.mark.parametrize(
    "decision, answer",
    [
        (view("select", options=["A", "B", "C"]), "b"),
        (view("select", options=["A", "B", "C"]), "D"),
        (view("select", options=["A", "B", "C"]), ""),
        (view("select", options=["A", "B", "C"]), 0),
        (view("select", options=["A", "B", "C"]), 2),
        (view("select", options=["A", "B", "C"]), 3),
        (view("select", options=["A", "B", "C"]), -1),
        (view("select", options=["A", "B", "C"]), True),
        (view("select", options=["1", "2", "3"]), 2),
        (view("select", options=[3, 5]), 3),
        (view("select", options=[1]), {"selected": True}),
        (view("select", options=["y"]), {"selected": "y", "index": True}),
        (view("select", options=["A", "B"]), {"selected": "A", "index": 1}),
        (view("select", options=["A", "B", "C"]), 1.5),
        (view("select", options=["A", "B", "C"]), {"selected": "B", "index": 0}),
        (view("select", options=["A", "B", "C"]), {"index": 1}),
        (view("confirm"), "maybe"),
        (view("confirm"), 1),
        (view("confirm"), {"confirmed": "yes"}),
        (view("text_input"), ""),
        (view("text_input"), "   "),
        (view("text_input"), 5),
        (view("text_input"), {"answer": ""}),
        (view("threshold_slider", min=0.0, max=1.0), 1.5),
        (view("threshold_slider", min=0.0, max=1.0), -0.1),
        (view("threshold_slider", min=0.0, max=1.0), float("nan")),
        (view("threshold_slider", min=0.0, max=1.0), True),
        (view("threshold_slider", min=0.0, max=1.0), "0.1"),
        (view("threshold_slider"), float("inf")),
        (view("cell_type_selector"), "T cells"),
        (view("cell_type_selector"), {"assignments": {}}),
        (view("cell_type_selector"), {"assignments": {"0": 5}}),
    ],
)
def test_answers_that_do_not_fit_the_question_are_refused(decision, answer):
    with pytest.raises(mcp_decision.BadAnswer):
        mcp_decision.check_answer(decision, answer)


def test_select_refusal_lists_the_options_and_threshold_refusal_the_range():
    with pytest.raises(mcp_decision.BadAnswer, match=r"\[\"A\", \"B\"\]"):
        mcp_decision.check_answer(view("select", options=["A", "B"]), "C")
    with pytest.raises(mcp_decision.BadAnswer, match=r"between 0\.0 and 1\.0"):
        mcp_decision.check_answer(view("threshold_slider", min=0.0, max=1.0), 2)


# real engine, replay script (no network) ----------------------------------------

_ENTRY = (
    "import sys\n"
    "from lobster.cli_internal.mcp_stdio import run_mcp_stdio\n"
    "sys.exit(run_mcp_stdio())\n"
)


class McpProcess:
    """The server over a pipe, driven like a host."""

    def __init__(self, script: Union[str, Path], tmp_path: Path):
        env = dict(os.environ)
        env.update(
            {
                "LOBSTER_TEST_REPLAY_FIXTURE": str(REPLAY / script),
                "LOBSTER_LLM_PROVIDER": "ollama-cloud",
                "OLLAMA_API_KEY": "replay-dummy",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        self.proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", _ENTRY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=cwd,
        )
        self.n = 0

    def rpc(self, method, params):
        self.n += 1
        message = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())["result"]

    def tool(self, name, arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        text = result["content"][0]["text"]
        return {"error_text": text} if result.get("isError") else json.loads(text)

    def wait(self, run_id, timeout=240):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.tool("lobster_status", {"run_id": run_id})
            if status["status"] != "running":
                return status
            time.sleep(0.5)
        raise AssertionError("still running")

    def close(self):
        try:
            self.proc.stdin.close()
            assert self.proc.wait(timeout=60) == 0
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()


def _initialize(proc: McpProcess) -> None:
    proc.rpc(
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "claude-code", "version": "2"},
        },
    )


@pytest.mark.slow
def test_real_engine_answers_a_question_and_notes_the_host_in_the_notebook(
    tmp_path, data
):
    proc = McpProcess(loading("decision_loaded.json", tmp_path, data), tmp_path)
    try:
        _initialize(proc)
        run = proc.tool("lobster_run", run_args(data, tmp_path / "out"))
        waiting = proc.wait(run["run_id"])
        assert waiting["status"] == "needs_decision", waiting
        decision = waiting["decision"]
        assert decision["kind"] == "threshold_slider"
        assert (decision["min"], decision["max"], decision["default"]) == (0, 1, 0.05)
        assert "mitochondrial" in decision["question"]

        stale = proc.tool(
            "lobster_answer",
            {
                "run_id": run["run_id"],
                "decision_id": decision["decision_id"],
                "request_version": 2,
                "answer": 0.1,
                "source": "model",
            },
        )
        assert stale["error_text"].startswith("rejected_stale:")
        reply = proc.tool(
            "lobster_answer",
            {
                "run_id": run["run_id"],
                "decision_id": decision["decision_id"],
                "request_version": 1,
                "answer": 0.1,
                "source": "model",
            },
        )
        assert reply["outcome"] == "answered"
        done = proc.wait(run["run_id"])
        assert done["status"] == "completed", done
        assert "threshold of 0.1" in done["summary"]
        assert Path(done["result_path"]).exists()
        (given,) = done["decisions"]
        assert (given["source"], given["answer"]) == ("model", {"value": 0.1})
        assert given["channel"] == "mcp:claude-code"
        assert "mitochondrial" in given["question"]
        assert done["decisions_note"] == mcp_decision.RECORDED_NOTE
        assert done["model_source"] == "replay"
        assert (
            json.loads(Path(done["result_path"]).read_text())["request"]["model_source"]
            == "replay"
        )
        assert json.loads(Path(done["result_path"]).read_text())["decisions"] == [given]
        assert "ask_user" not in [s["tool"] for s in done.get("steps", [])]
        notebook = json.loads(Path(done["notebook_path"]).read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        assert (
            "# Decision recorded during the analysis (source: 'model', as "
            "reported by the MCP host 'claude-code')\n"
            "# Lobster cannot verify who answered through an MCP host.\n"
            "# Question: 'Which mitochondrial fraction threshold should QC use?'\n"
            "# Answer: {'value': 0.1}\n"
            "# Comment: None"
        ) in source
    finally:
        proc.close()


@pytest.mark.slow
def test_real_engine_cancel_while_waiting_is_decision_abandoned(tmp_path, data):
    proc = McpProcess("decision.json", tmp_path)
    try:
        _initialize(proc)
        run = proc.tool("lobster_run", run_args(data, tmp_path / "out"))
        waiting = proc.wait(run["run_id"])
        assert waiting["status"] == "needs_decision", waiting
        assert proc.tool("lobster_cancel", {"run_id": run["run_id"]})["outcome"] == (
            "cancel_requested"
        )
        done = proc.wait(run["run_id"])
        assert done["status"] == "cancelled"
        assert done["cancel_reason"] == "decision_abandoned"
    finally:
        proc.close()
