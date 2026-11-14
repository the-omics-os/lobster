"""``lobster_wait``: one call that returns when the host must act, instead of polling.

It returns on a question (``needs_decision``), at the end, or after
``timeout_s`` with a short reply; a new step alone does not wake it.

The scripted client stands in for the engine session; the thread runner, the
server and the stdio loop are the production code.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from lobster.cli_internal import mcp_contract, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_answer import SLIDER
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Hold,
    WaitForCancel,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def timed_wait(server, run_id, timeout_s):
    started = time.monotonic()
    status = call(server, "lobster_wait", {"run_id": run_id, "timeout_s": timeout_s})
    return status, time.monotonic() - started


def test_the_wait_tool_is_listed_after_status():
    tools = {t["name"]: t for t in mcp_contract.list_tools()}
    names = [t["name"] for t in mcp_contract.list_tools()]
    assert names.index("lobster_wait") == names.index("lobster_status") + 1
    wait = tools["lobster_wait"]
    timeout = wait["inputSchema"]["properties"]["timeout_s"]
    assert (timeout["minimum"], timeout["maximum"]) == (0, mcp_contract.WAIT_MAX_S)
    assert wait["inputSchema"]["required"] == ["run_id"]
    assert wait["annotations"]["readOnlyHint"] is True
    assert mcp_contract.WAIT_MAX_S == 300
    assert mcp_contract.WAIT_DEFAULT_S < 60  # Codex's default tool time limit
    assert "Pass timeout_s 300 unless your host ends tool calls sooner" in (
        wait["description"]
    )
    for name in ("lobster_run", "lobster_status"):
        assert "lobster_wait" in tools[name]["description"]
    assert "lobster_wait" in mcp_contract.INSTRUCTIONS
    assert len(mcp_contract.INSTRUCTIONS) <= 512


def test_a_finished_run_is_returned_at_once(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    done = wait_terminal(server, run_id)
    status, took = timed_wait(server, run_id, 30)
    assert status == done and took < 1


def test_an_unchanged_run_comes_back_after_the_timeout(engine, tmp_path, data):
    _opened, script = engine
    hold = Hold("quality_assessment")
    script["events"] = [hold, COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    before = call(server, "lobster_status", {"run_id": run_id})
    status, took = timed_wait(server, run_id, 1)
    assert status == {
        "run_id": run_id,
        "status": "running",
        "step": before["step"],
        "steps_done": before["steps_done"],
        "elapsed_s": status["elapsed_s"],
        "timed_out": True,
    }
    assert 0.9 <= took < 3
    hold.release.set()
    wait_terminal(server, run_id)
    server.runner.stop(5)


def test_a_new_step_does_not_end_the_wait(engine, tmp_path, data):
    _opened, script = engine
    first, second = Hold("quality_assessment"), Hold("filter_and_normalize")
    script["events"] = [first, second, COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert first.entered.wait(10)
    before = call(server, "lobster_status", {"run_id": run_id})
    threading.Timer(0.3, first.release.set).start()
    status, took = timed_wait(server, run_id, 2)
    assert status["timed_out"] is True and 1.9 <= took < 4
    assert status["step"] == "filter_and_normalize"  # the last step, in the reply
    assert status["steps_done"] == before["steps_done"] + 1
    second.release.set()
    wait_terminal(server, run_id)
    server.runner.stop(5)


def test_a_question_ends_the_wait_and_an_open_one_returns_at_once(
    engine, tmp_path, data
):
    _opened, script = engine
    hold = Hold("quality_assessment")
    script["events"] = [hold, SLIDER]
    script["resume_events"] = [COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    threading.Timer(0.5, hold.release.set).start()
    status, took = timed_wait(server, run_id, 30)
    assert status["status"] == "needs_decision" and took < 5
    again, took = timed_wait(server, run_id, 30)
    assert again["status"] == "needs_decision" and took < 1
    server.runner.stop(5)


@pytest.mark.parametrize(
    "arguments, said",
    [
        ({"run_id": "nope"}, "Unknown run_id 'nope'"),
        ({"run_id": "r", "timeout_s": 301}, "timeout_s"),
        ({"run_id": "r", "timeout_s": -1}, "timeout_s"),
        ({"run_id": "r", "timeout_s": "5"}, "timeout_s"),
        ({"run_id": "r", "timeout_s": True}, "timeout_s"),
    ],
)
def test_bad_arguments_are_refused(arguments, said):
    refused = call(McpServer(ThreadRunner()), "lobster_wait", arguments)
    assert said in refused["error_text"]


class Wire:
    """``run_mcp_stdio`` on a thread, fed and read line by line."""

    def __init__(self):
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        self.replies: "queue.Queue[dict]" = queue.Queue()
        self.exit_code = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        self.send(0, "initialize", {"protocolVersion": "2025-11-25"})
        assert self.reply(0)["result"]["serverInfo"]["name"] == "lobster"

    def _serve(self):
        wire = self

        class Out:
            def write(self, text):
                for line in text.splitlines():
                    wire.replies.put(json.loads(line))

            def flush(self):
                pass

        self.exit_code = mcp_stdio.run_mcp_stdio(
            runner=ThreadRunner(), stdin=iter(self.lines.get, None), stdout=Out()
        )

    def send(self, request_id, method, params):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        self.lines.put(json.dumps(message))

    def tool(self, request_id, name, **arguments):
        self.send(request_id, "tools/call", {"name": name, "arguments": arguments})

    def reply(self, request_id, timeout=10):
        deadline = time.monotonic() + timeout
        held = []
        try:
            while True:
                left = deadline - time.monotonic()
                got = self.replies.get(timeout=max(left, 0.01))
                if got.get("id") == request_id:
                    return got
                held.append(got)
        finally:
            for other in held:
                self.replies.put(other)

    def result(self, request_id, timeout=10):
        return json.loads(
            self.reply(request_id, timeout)["result"]["content"][0]["text"]
        )

    def no_reply(self, request_id, within):
        with pytest.raises(queue.Empty):
            self.reply(request_id, timeout=within)

    def close(self, timeout=10):
        self.lines.put(None)
        self.thread.join(timeout)
        return self.exit_code


@pytest.fixture
def held_run(engine, tmp_path, data):
    _opened, script = engine
    hold = Hold("quality_assessment")
    script["events"] = [hold, COMPLETE]
    wire = Wire()
    wire.tool(1, "lobster_run", **run_args(data, tmp_path / "out"))
    run_id = wire.result(1)["run_id"]
    assert hold.entered.wait(10)
    yield wire, run_id, hold
    hold.release.set()
    wire.close()


def test_other_calls_are_answered_while_a_wait_is_open(held_run):
    wire, run_id, hold = held_run
    wire.tool(2, "lobster_wait", run_id=run_id, timeout_s=60)
    wire.tool(3, "lobster_status", run_id=run_id)
    assert wire.result(3, timeout=2)["status"] == "running"
    wire.no_reply(2, within=0.5)
    hold.release.set()
    assert wire.result(2, timeout=10)["status"] == "completed"


def test_a_cancel_during_a_wait_ends_both(engine, tmp_path, data):
    _opened, script = engine
    step = WaitForCancel()
    script.update(events=[step], loads_input=False)
    wire = Wire()
    wire.tool(1, "lobster_run", **run_args(data, tmp_path / "out"))
    run_id = wire.result(1)["run_id"]
    assert step.entered.wait(10)
    wire.tool(2, "lobster_wait", run_id=run_id, timeout_s=60)
    wire.no_reply(2, within=0.3)
    wire.tool(3, "lobster_cancel", run_id=run_id)
    assert wire.result(3, timeout=2)["outcome"] == "cancel_requested"
    assert wire.result(2, timeout=10)["status"] == "cancelled"
    assert wire.close() == 0


def test_a_cancelled_wait_gets_no_reply_and_the_server_goes_on(held_run):
    wire, run_id, hold = held_run
    wire.tool("w", "lobster_wait", run_id=run_id, timeout_s=60)
    wire.no_reply("w", within=0.3)
    wire.send(None, "notifications/cancelled", {"requestId": "w"})
    wire.no_reply("w", within=1.0)
    wire.tool(3, "lobster_status", run_id=run_id)
    assert wire.result(3, timeout=2)["status"] == "running"


def test_a_second_wait_on_a_run_replaces_the_first(held_run):
    wire, run_id, hold = held_run
    wire.tool(2, "lobster_wait", run_id=run_id, timeout_s=60)
    wire.no_reply(2, within=0.3)
    wire.tool(3, "lobster_wait", run_id=run_id, timeout_s=60)
    first = wire.result(2, timeout=2)  # the first, at once
    assert first["status"] == "running"
    assert first["superseded"] is True and "timed_out" not in first
    wire.no_reply(3, within=0.5)
    hold.release.set()
    assert wire.result(3, timeout=10)["status"] == "completed"


def test_end_of_input_during_a_wait_stops_the_server(held_run):
    wire, run_id, hold = held_run
    wire.tool(2, "lobster_wait", run_id=run_id, timeout_s=60)
    wire.no_reply(2, within=0.3)
    started = time.monotonic()
    hold.release.set()  # a run step ends at its own pace, as in production
    assert wire.close(timeout=15) == 0
    assert time.monotonic() - started < 15


def test_an_answer_during_a_wait_lets_the_next_wait_see_the_end(engine, tmp_path, data):
    _opened, script = engine
    hold = Hold("quality_assessment")
    script["events"] = [hold, SLIDER]
    script["resume_events"] = [COMPLETE]
    wire = Wire()
    wire.tool(1, "lobster_run", **run_args(data, tmp_path / "out"))
    run_id = wire.result(1)["run_id"]
    assert hold.entered.wait(10)
    wire.tool(2, "lobster_wait", run_id=run_id, timeout_s=60)
    hold.release.set()
    asked = wire.result(2, timeout=10)
    assert asked["status"] == "needs_decision"
    wire.tool(3, "lobster_wait", run_id=run_id, timeout_s=60)
    assert wire.result(3, timeout=2)["status"] == "needs_decision"  # at once
    decision = asked["decision"]
    wire.tool(
        4,
        "lobster_answer",
        run_id=run_id,
        decision_id=decision["decision_id"],
        request_version=decision["request_version"],
        answer=0.1,
        source="model",
    )
    assert wire.result(4, timeout=5)["outcome"] == "answered"
    wire.tool(5, "lobster_wait", run_id=run_id, timeout_s=60)
    assert wire.result(5, timeout=10)["status"] == "completed"
    assert wire.close() == 0


def test_the_end_ends_a_long_wait(engine, tmp_path, data):
    _opened, script = engine
    hold = Hold("quality_assessment")
    script["events"] = [hold, COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    threading.Timer(0.3, hold.release.set).start()
    status, took = timed_wait(server, run_id, 300)
    assert status["status"] == "completed" and took < 5
    assert "timed_out" not in status and "result_path" in status  # the full status


def test_an_abandoned_wait_is_harmless(held_run):
    """A host that stops waiting without saying so: the next wait on the run
    replaces it, other calls are answered meanwhile, and nothing is held."""
    wire, run_id, hold = held_run
    wire.tool(7, "lobster_wait", run_id=run_id, timeout_s=300)
    wire.tool(8, "lobster_status", run_id=run_id)
    assert wire.result(8, timeout=2)["status"] == "running"
    wire.tool(9, "lobster_wait", run_id=run_id, timeout_s=0)
    assert wire.result(7, timeout=2)["status"] == "running"  # replaced at once
    assert wire.result(9, timeout=2)["timed_out"] is True
    hold.release.set()
    wire.tool(10, "lobster_wait", run_id=run_id, timeout_s=300)
    assert wire.result(10, timeout=10)["status"] == "completed"
