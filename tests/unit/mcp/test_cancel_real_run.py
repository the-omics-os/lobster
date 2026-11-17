"""A cancel stops a real run inside an expert's work, and the server exits.

``lobster mcp --stdio`` runs in its own process on the real engine and graph,
with the replay model the protocol fixtures use: the supervisor hands the work
to the data expert, whose answers come slowly. The cancel lands while the
expert works; the run must end cancelled before the expert's next, very slow
answer is ever asked for, and the server process must then exit when its
input closes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.unit.mcp.test_run import ROOT

pytestmark = pytest.mark.no_auto_config

NEVER_REACHED_S = 120  # the expert's second answer: asked for only without the fix

SCRIPT = {
    "schema": "lobster.replay/1",
    "responses": [
        {
            "tool_calls": [
                {
                    "name": "handoff_to_data_expert_agent",
                    "args": {"task_description": "Your task is to list adapters."},
                }
            ]
        },
        {
            "content": "Looking.",
            "delay_s": 6,
            "tool_calls": [{"name": "get_adapter_info"}],
        },
        {
            "content": "Again.",
            "delay_s": NEVER_REACHED_S,
            "tool_calls": [{"name": "get_adapter_info"}],
        },
        {"content": "Adapters listed."},
        {"content": "Done."},
    ],
}


class Server:
    def __init__(self, tmp: Path, responses=SCRIPT["responses"]):
        script = tmp / "replay.json"
        script.write_text(json.dumps({**SCRIPT, "responses": responses}))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp / "home"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                str(p)
                for p in (
                    ROOT,
                    ROOT / "packages" / "lobster-research",
                    ROOT / "packages" / "lobster-transcriptomics",
                )
            ),
            "HTTPS_PROXY": "http://127.0.0.1:9",  # no update check
            "LOBSTER_TEST_REPLAY_FIXTURE": str(script),
            "LOBSTER_LLM_PROVIDER": "ollama-cloud",
            "OLLAMA_API_KEY": "replay-dummy",
        }
        self.env = env
        self.err = open(tmp / "server.err", "w")
        self.proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-m", "lobster", "mcp", "--stdio"],
            cwd=tmp,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.err,
            text=True,
        )
        self.n = 0
        self.request(
            "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}
        )

    def request(self, method, params):
        self.n += 1
        line = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(line) + "\n")
        self.proc.stdin.flush()
        for reply in self.proc.stdout:
            reply = json.loads(reply)
            if reply.get("id") == self.n:
                return reply["result"]
        raise AssertionError(f"no reply; stderr: {self.stderr()}")

    def tool(self, name, **args):
        result = self.request("tools/call", {"name": name, "arguments": args})
        text = result["content"][0]["text"]
        return text if result.get("isError") else json.loads(text)

    def stderr(self):
        return Path(self.err.name).read_text()[-3000:]


def test_a_cancel_inside_an_expert_ends_the_real_run_and_the_server_exits(tmp_path):
    server = Server(tmp_path)
    try:
        run = server.tool(
            "lobster_run",
            question="List the adapters.",
            output_dir=str(tmp_path / "out"),
            experts=["data_expert_agent"],
            plan=False,
        )
        assert isinstance(run, dict), run
        deadline = time.monotonic() + 120
        while server.tool("lobster_status", run_id=run["run_id"])["phase"] != "running":
            assert time.monotonic() < deadline, server.stderr()
            time.sleep(0.2)
        time.sleep(2)  # the expert is at work: its first answer takes 6 s
        asked = time.monotonic()
        assert server.tool("lobster_cancel", run_id=run["run_id"])
        status = server.tool("lobster_status", run_id=run["run_id"])
        assert status["status"] in ("running", "cancelled"), status
        if status["status"] == "running":
            assert status["phase"] == "cancelling"
        done = server.tool("lobster_wait", run_id=run["run_id"], timeout_s=60)
        took = time.monotonic() - asked
        assert done["status"] == "cancelled", (done, server.stderr())
        assert took < 30, took  # not after the expert's slow second answer
    finally:
        server.proc.stdin.close()
    assert server.proc.wait(timeout=30) == 0, server.stderr()
    server.err.close()


HANDOFF = SCRIPT["responses"][0]


def wait_running(server, run_id):
    deadline = time.monotonic() + 120
    while server.tool("lobster_status", run_id=run_id)["phase"] != "running":
        assert time.monotonic() < deadline, server.stderr()
        time.sleep(0.2)


def ends(server, run_id):
    for _ in range(5):
        done = server.tool("lobster_wait", run_id=run_id, timeout_s=60)
        if done["status"] in ("completed", "failed", "cancelled"):
            return done
    raise AssertionError(server.stderr())


# The second turn is cancelled while its hand-off runs (the expert's first
# answer is slow), or while the supervisor's own answer is slow; the third
# turn then needs a conversation in which every tool call has its result.
SECOND_TURN = {
    "inside the expert": [
        HANDOFF,
        {
            "content": "Looking.",
            "delay_s": 6,
            "tool_calls": [{"name": "get_adapter_info"}],
        },
    ],
    "in the supervisor's answer": [{"content": "Thinking.", "delay_s": 6}],
    "at a question": [
        {"tool_calls": [{"name": "ask_user", "args": {"question": "Which one?"}}]},
        {"content": "Left unanswered."},  # if the abandoned question resumes
    ],
}


@pytest.mark.parametrize("where", sorted(SECOND_TURN))
def test_a_session_continues_after_a_turn_cancelled_mid_work(tmp_path, where):
    responses = [
        HANDOFF,
        {"content": "Ready."},
        {"content": "Turn one done."},
        *SECOND_TURN[where],
        {"content": "Turn three done."},
    ]
    server = Server(tmp_path, responses)
    try:
        run = server.tool(
            "lobster_run",
            question="List the adapters.",
            output_dir=str(tmp_path / "out"),
            experts=["data_expert_agent"],
            plan=False,
        )
        assert ends(server, run["run_id"])["status"] == "completed", server.stderr()
        second = server.tool(
            "lobster_continue", run_id=run["run_id"], instruction="Go on."
        )
        if where == "at a question":
            asked = server.tool("lobster_wait", run_id=second["run_id"], timeout_s=60)
            assert asked["status"] == "needs_decision", (asked, server.stderr())
        else:
            wait_running(server, second["run_id"])
            time.sleep(2)
        server.tool("lobster_cancel", run_id=second["run_id"])
        assert ends(server, second["run_id"])["status"] == "cancelled"
        third = server.tool(
            "lobster_continue", output_dir=run["output_dir"], instruction="Sum up."
        )
        assert isinstance(third, dict), third
        done = ends(server, third["run_id"])
        assert done["status"] == "completed", (done, server.stderr())
    finally:
        server.proc.stdin.close()
    assert server.proc.wait(timeout=30) == 0, server.stderr()
    server.err.close()


def test_the_notebook_after_a_cancelled_turn_still_verifies(tmp_path):
    # Turn 1 loads and filters real data; turn 2 is cancelled inside the
    # expert; turn 3 completes. Its notebook holds turn 1's steps only, and
    # re-running it reproduces the file the session saved.
    from tests.unit.mcp.replay_input import make_input
    from tests.unit.mcp.transcriptomics import quality_service

    quality_service()
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    make_input(data)
    recipe = Path(__file__).parent / "data" / "continue_first_process.json"
    qc = json.loads(recipe.read_text().replace("__INPUT__", str(data)))
    cancelled = [
        {
            "tool_calls": [
                {
                    "name": "handoff_to_transcriptomics_expert",
                    "args": {"task_description": "Your task is to assess again."},
                }
            ]
        },
        {
            "content": "",
            "delay_s": 6,
            "tool_calls": [
                {
                    "name": "assess_data_quality",
                    "args": {"modality_name": "input_quality_assessed"},
                }
            ],
        },
    ]
    responses = [*qc["responses"][:8], *cancelled, {"content": "Turn three done."}]
    server = Server(tmp_path, responses)
    try:
        run = server.tool(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            question="What is in this file?",
            plan=False,  # the script hands off to the analysis expert at once
        )
        assert ends(server, run["run_id"])["status"] == "completed", server.stderr()
        second = server.tool(
            "lobster_continue", run_id=run["run_id"], instruction="Assess again."
        )
        wait_running(server, second["run_id"])
        time.sleep(2)
        server.tool("lobster_cancel", run_id=second["run_id"])
        assert ends(server, second["run_id"])["status"] == "cancelled"
        third = server.tool(
            "lobster_continue", output_dir=str(out), instruction="Sum up."
        )
        done = ends(server, third["run_id"])
        assert done["status"] == "completed", (done, server.stderr())
    finally:
        server.proc.stdin.close()
    assert server.proc.wait(timeout=30) == 0, server.stderr()
    server.err.close()

    notebook = Path(done["notebook_path"])
    cells = json.loads(notebook.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert text.count(": assess_data_quality") == 1  # turn 1's, not turn 2's
    replay = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "replay", "--verify", str(notebook)]
        + ["--input", str(data), "-w", str(out / ".lobster_workspace"), "--json"],
        cwd=tmp_path,
        env=server.env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert replay.returncode == 0, replay.stderr[-2000:]
    assert json.loads(replay.stdout)["verdict"] == "MATCH"


def test_a_file_a_cancelled_turn_saved_is_compared_by_the_next_notebook(tmp_path):
    # Turn 2 filters the raw data, which saves a file, and is cancelled during
    # its next answer; its completed step stays in the session. The notebook
    # taken after it must reproduce that file too: a MATCH that left it out
    # would hide what the cancelled turn wrote.
    from tests.unit.mcp.replay_input import make_input
    from tests.unit.mcp.transcriptomics import quality_service

    quality_service()
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    make_input(data)
    recipe = Path(__file__).parent / "data" / "continue_first_process.json"
    qc = json.loads(recipe.read_text().replace("__INPUT__", str(data)))
    cancelled = [
        {
            "tool_calls": [
                {
                    "name": "handoff_to_transcriptomics_expert",
                    "args": {
                        "task_description": "Your task is to filter the raw data."
                    },
                }
            ]
        },
        {
            "content": "",
            "tool_calls": [
                {"name": "filter_and_normalize", "args": {"modality_name": "input"}}
            ],
        },
        {"content": "Slow.", "delay_s": 15},
    ]
    responses = [*qc["responses"][:8], *cancelled, {"content": "Turn three done."}]
    server = Server(tmp_path, responses)
    try:
        run = server.tool(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            question="What is in this file?",
            plan=False,  # the script hands off to the analysis expert at once
        )
        assert ends(server, run["run_id"])["status"] == "completed", server.stderr()
        second = server.tool(
            "lobster_continue", run_id=run["run_id"], instruction="Filter the raw data."
        )
        deadline = time.monotonic() + 120
        while (
            server.tool("lobster_status", run_id=second["run_id"]).get("steps_done", 0)
            < 2  # the save and the filter
        ):
            assert time.monotonic() < deadline, server.stderr()
            time.sleep(0.5)
        server.tool("lobster_cancel", run_id=second["run_id"])
        stopped = ends(server, second["run_id"])
        assert stopped["status"] == "cancelled"
        assert "filter_and_normalize" in stopped["steps_completed"]
        third = server.tool(
            "lobster_continue", output_dir=str(out), instruction="Sum up."
        )
        done = ends(server, third["run_id"])
        assert done["status"] == "completed", (done, server.stderr())
    finally:
        server.proc.stdin.close()
    assert server.proc.wait(timeout=30) == 0, server.stderr()
    server.err.close()

    assert (
        out / ".lobster_workspace" / "data" / "input_filtered_normalized.h5ad"
    ).is_file()
    replay = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "replay", "--verify", done["notebook_path"]]
        + ["--input", str(data), "-w", str(out / ".lobster_workspace"), "--json"],
        cwd=tmp_path,
        env=server.env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(replay.stdout)
    (slot,) = report["report"]["slots"]
    assert slot["not_compared"] == [], report
    assert not any("not produced" in note for note in report["notes"]), report
    assert report["verdict"] == "MATCH", json.dumps(report, indent=1)[:4000]
