"""Bounds and edge cases of the question a run asks and of the answers it takes.

The live ``needs_decision`` status and the answer path must stay bounded and
must never turn an odd value into a JSON-RPC error or invalid JSON on stdout.
"""

from __future__ import annotations

import io
import json
import threading

import pytest

from lobster.cli_internal import mcp_decision, mcp_stdio
from lobster.cli_internal.mcp_stdio import (
    McpServer,
    RunProgress,
    ThreadRunner,
    ToolError,
)
from tests.unit.mcp.test_answer import (  # noqa: F401  (fixtures)
    SELECT,
    SLIDER,
    answer_args,
    initialize,
    interrupt,
    server,
    start,
    wait_waiting,
)
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Hold,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def summary(kind, payload, question="Q?", fallback=None, version=1):
    return {
        "decision_id": "dec_x",
        "request_version": version,
        "question": question,
        "components": [
            {
                "n": 1,
                "component": kind,
                "data": payload,
                "fallback_prompt": question if fallback is None else fallback,
            }
        ],
    }


def raw_tool(server, name, arguments):
    """The whole JSON-RPC reply, for the cases that must not be protocol errors."""
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return server.handle_line(line)


def refused_text(reply):
    assert "error" not in reply, reply
    assert reply["result"]["isError"] is True, reply
    return reply["result"]["content"][0]["text"]


# the live status is bounded -------------------------------------------------

BIG_OPTIONS = [f"option {i} " + "x" * 50 for i in range(3000)]
BIG_QUESTION = "Which normalization should be used? " * 3000  # 108 KB


def test_view_cuts_question_fallback_and_options():
    shown = mcp_decision.public(
        mcp_decision.view(
            summary(
                "select",
                {"question": BIG_QUESTION, "options": BIG_OPTIONS},
                BIG_QUESTION,
                BIG_QUESTION + "".join(BIG_OPTIONS),
            )
        )
    )
    assert len(json.dumps(shown)) < mcp_stdio.STATUS_MAX_CHARS // 2
    assert shown["decision_id"] == "dec_x" and shown["request_version"] == 1
    assert len(shown["question"]) == mcp_decision.SHOWN_MAX_CHARS
    assert shown["question"].endswith(mcp_decision.CUT)
    assert shown["options"] == BIG_OPTIONS[: mcp_decision.MAX_OPTIONS]
    assert shown["options_truncated"] is True
    assert shown["options_total"] == 3000
    assert "_options" not in json.dumps(shown)


def test_view_cuts_the_fallback_prompt_of_the_kinds_that_keep_it():
    shown = mcp_decision.view(summary("text_input", {}, "Q?", "p" * 50_000))
    assert len(shown["fallback_prompt"]) == mcp_decision.SHOWN_MAX_CHARS
    assert shown["fallback_prompt"].endswith(mcp_decision.CUT)


def test_view_leaves_a_small_question_alone():
    shown = mcp_decision.view(
        summary("select", {"question": "Q?", "options": ["a", "b"]})
    )
    assert shown["options"] == ["a", "b"]
    assert "options_truncated" not in shown and "options_total" not in shown


def test_view_cuts_one_very_long_option_and_a_non_text_option():
    shown = mcp_decision.public(
        mcp_decision.view(
            summary("select", {"options": ["a" * 10_000, {"k": "v" * 10_000}, 3]})
        )
    )
    assert len(json.dumps(shown)) < 5000
    assert shown["options"][0].endswith(mcp_decision.CUT)
    assert shown["options"][1].endswith(mcp_decision.CUT)
    assert shown["options"][2] == 3


def test_an_option_that_was_not_shown_is_still_an_answer():
    decision = mcp_decision.view(
        summary("select", {"options": BIG_OPTIONS, "question": "Q?"})
    )
    assert BIG_OPTIONS[2999] not in decision["options"]
    assert mcp_decision.check_answer(decision, BIG_OPTIONS[2999]) == {
        "selected": BIG_OPTIONS[2999],
        "index": 2999,
    }
    assert mcp_decision.check_answer(decision, {"selected": BIG_OPTIONS[77]}) == {
        "selected": BIG_OPTIONS[77],
        "index": 77,
    }
    with pytest.raises(mcp_decision.BadAnswer) as refused:
        mcp_decision.check_answer(decision, "nope")
    assert len(str(refused.value)) < 1500
    assert "3000" in str(refused.value)


def test_live_status_of_a_huge_question_stays_under_the_status_bound(
    engine, server, tmp_path, data
):
    _opened, script = engine
    script["events"] = [
        interrupt(
            "select",
            {"question": BIG_QUESTION, "options": BIG_OPTIONS},
            BIG_QUESTION + "".join(BIG_OPTIONS),
        )
    ]
    script["resume_events"] = [COMPLETE]
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    text = json.dumps(waiting)
    assert len(text) < mcp_stdio.STATUS_MAX_CHARS
    decision = waiting["decision"]
    assert decision["decision_id"].startswith("dec_")
    assert decision["request_version"] == 1
    assert decision["options_truncated"] is True and "_options" not in text
    # An option past the 50 shown is accepted; one that does not exist is not.
    refused = call(server, "lobster_answer", answer_args(run_id, waiting, "nope"))
    assert refused["error_text"].startswith("bad_answer: 'nope' is not one of")
    reply = call(
        server, "lobster_answer", answer_args(run_id, waiting, BIG_OPTIONS[2999])
    )
    assert reply["answer"] == {"selected": BIG_OPTIONS[2999], "index": 2999}
    assert wait_terminal(server, run_id)["status"] == "completed"


# an answer that arrives after a cancel was asked for ------------------------


class _Manager:
    def __init__(self):
        self.decisions = {}
        self.answers = []

    def answer(self, *args, **kwargs):
        self.answers.append((args, kwargs))
        return object()


def test_submit_refuses_once_the_cancel_event_is_set():
    progress, manager, cancel = RunProgress(), _Manager(), threading.Event()
    progress.begin(lambda: {}, manager)
    # The state the worker leaves while it waits, before it has seen the cancel.
    progress._waiting = {
        "decision_id": "dec_1",
        "request_version": 1,
        "kind": "confirm",
    }
    cancel.set()
    with pytest.raises(ToolError, match="^rejected_cancelled: the run was stopped"):
        progress.submit(
            "run_1", "needs_decision", "dec_1", 1, True, "human", None, "mcp:x", cancel
        )
    assert manager.answers == []
    assert progress._answered is None and progress.is_waiting()


# odd numbers and non-finite values are refusals, never protocol errors ------


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_huge_integer_threshold_is_a_bad_answer(value):
    decision = {"kind": "threshold_slider", "min": 0.0, "max": 1.0}
    with pytest.raises(mcp_decision.BadAnswer):
        mcp_decision.check_answer(decision, value)
    with pytest.raises(mcp_decision.BadAnswer):  # no range given: still a refusal
        mcp_decision.check_answer({"kind": "threshold_slider"}, value)


@pytest.mark.parametrize("literal", ["1" + "0" * 400, "NaN", "Infinity", "-Infinity"])
def test_odd_numbers_reach_the_host_as_a_refusal_not_a_protocol_error(
    engine, server, tmp_path, data, literal
):
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    decision = waiting["decision"]
    line = (
        '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":'
        '"lobster_answer","arguments":{"run_id":"%s","decision_id":"%s",'
        '"request_version":1,"source":"model","answer":%s}}}'
        % (run_id, decision["decision_id"], literal)
    )
    reply = server.handle_line(line)
    assert refused_text(reply).startswith("bad_answer:")
    assert (
        call(server, "lobster_status", {"run_id": run_id})["status"] == "needs_decision"
    )
    ok = call(server, "lobster_answer", answer_args(run_id, waiting, 0.5, "model"))
    assert ok["outcome"] == "answered"
    wait_terminal(server, run_id)


def test_non_finite_numbers_inside_an_object_are_refused_for_every_kind(
    engine, server, tmp_path, data
):
    _opened, script = engine
    script.update(
        events=[interrupt("mystery_widget", {"question": "Q?"}, "Q?")],
        resume_events=[COMPLETE],
    )
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    assert waiting["decision"]["kind"] == "mystery_widget"
    decision = waiting["decision"]
    line = (
        '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":'
        '"lobster_answer","arguments":{"run_id":"%s","decision_id":"%s",'
        '"request_version":1,"source":"model","answer":{"x":[NaN]}}}}'
        % (run_id, decision["decision_id"])
    )
    assert refused_text(server.handle_line(line)).startswith("bad_answer:")
    assert (
        call(server, "lobster_status", {"run_id": run_id})["status"] == "needs_decision"
    )
    call(server, "lobster_cancel", {"run_id": run_id})
    wait_terminal(server, run_id)


class _NaNRunner(mcp_stdio.FakeRunner):
    def status(self, run_id):
        return {"status": "running", "run_id": run_id, "elapsed_s": float("nan")}


def test_a_tool_result_is_never_invalid_json():
    server = McpServer(_NaNRunner())
    reply = raw_tool(server, "lobster_status", {"run_id": "r"})
    text = json.dumps(reply)
    assert "NaN" not in text and "Infinity" not in text
    assert reply["result"]["isError"] is True


# select edge cases (the matching ones are in test_answer.py) ----------------


def test_select_refusal_is_one_line_and_bounded():
    options = [f"opt{i}" for i in range(500)]
    with pytest.raises(mcp_decision.BadAnswer) as refused:
        mcp_decision.check_answer({"kind": "select", "options": options}, "x" * 9000)
    text = str(refused.value)
    assert len(text) < 1200 and "\n" not in text
    assert text.endswith("answer with one option text.")


# answer and comment are bounded ---------------------------------------------


@pytest.mark.parametrize("field", ["answer", "comment"])
def test_text_over_the_limit_is_refused_and_the_limit_is_named(
    engine, server, tmp_path, data, field
):
    _opened, script = engine
    script.update(
        events=[interrupt("text_input", {"question": "Name?"}, "Name?")],
        resume_events=[COMPLETE],
    )
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    limit = mcp_stdio.ANSWER_MAX_CHARS
    assert limit == 4000

    def args(size):
        long = "t" * size
        if field == "answer":
            return answer_args(run_id, waiting, long)
        return answer_args(run_id, waiting, "ok", comment=long)

    too_long = call(server, "lobster_answer", args(limit + 1))
    assert too_long["error_text"] == (
        f"bad_answer: {field} is longer than {limit} characters."
    )
    assert opened_resumes(engine) == []
    assert call(server, "lobster_answer", args(limit))["outcome"] == "answered"
    wait_terminal(server, run_id)


def opened_resumes(engine):
    return engine[0][0]["client"].resumes


def test_reply_echoes_at_most_200_characters_of_the_answer(
    engine, server, tmp_path, data
):
    opened, script = engine
    script.update(
        events=[interrupt("text_input", {"question": "Name?"}, "Name?")],
        resume_events=[COMPLETE],
    )
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    long_answer = "w" * 3000
    reply = call(server, "lobster_answer", answer_args(run_id, waiting, long_answer))
    assert reply["outcome"] == "answered"
    assert len(json.dumps(reply)) < 500
    wait_terminal(server, run_id)
    # The engine still gets the whole answer.
    recorded = opened[0]["client"].resumes[0]["lg_1"]["__lobster_decision__"]["answer"]
    assert recorded == {"answer": long_answer}


def test_a_short_answer_is_echoed_as_recorded():
    assert mcp_stdio._echo({"selected": "CPM", "index": 1}) == {
        "selected": "CPM",
        "index": 1,
    }
    assert len(mcp_stdio._echo({"answer": "w" * 3000})) <= 200 + len(mcp_decision.CUT)


# the recorded decision is not an analysis step ------------------------------


def test_the_recorded_decision_is_not_counted_as_a_step(engine, server, tmp_path, data):
    _opened, script = engine
    hold = Hold("quality_assessment", "ask_user")
    script["events"] = [hold, COMPLETE]
    run_id = start(server, tmp_path, data)
    assert hold.entered.wait(10)
    live = call(server, "lobster_status", {"run_id": run_id})
    # The first turn's own load of the input is the first step.
    assert live["steps_done"] == 2 and live["step"] == "quality_assessment"
    hold.release.set()
    wait_terminal(server, run_id)


def test_only_a_recorded_decision_means_no_step(engine, server, tmp_path, data):
    _opened, script = engine
    hold = Hold("ask_user")
    script["events"] = [hold, COMPLETE]
    run_id = start(server, tmp_path, data)
    assert hold.entered.wait(10)
    live = call(server, "lobster_status", {"run_id": run_id})
    assert live["steps_done"] == 1 and live["step"] == "load_dataset"
    hold.release.set()
    wait_terminal(server, run_id)


# a line that json cannot parse ----------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"n":' + "9" * 5000 + "}}",
        "[" * 100_000,
    ],
)
def test_an_unparseable_line_is_a_parse_error_and_the_server_keeps_serving(line):
    out = io.StringIO()
    ping = '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
    stdin = io.StringIO(line + "\n" + ping + "\n")
    assert (
        mcp_stdio.run_mcp_stdio(runner=mcp_stdio.FakeRunner(), stdin=stdin, stdout=out)
        == 0
    )
    first, second = [json.loads(row) for row in out.getvalue().splitlines()]
    assert first["error"]["code"] == mcp_stdio.PARSE_ERROR and first["id"] is None
    assert [tool["name"] for tool in second["result"]["tools"]][:2] == [
        "lobster_overview",
        "lobster_run",
    ]
