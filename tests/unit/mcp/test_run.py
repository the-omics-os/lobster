"""``lobster mcp --stdio`` runs: immediate return, one active run, terminal states.

Production binding:

- ``init_client_or_raise_startup_diagnostic`` in
  ``lobster/cli_internal/commands/heavy/session_infra.py:770`` opens the
  session with ``workspace=output_dir`` and ``interactive=True``.
- ``runs_for(client).start(prompt, cancel_event=...)`` and ``run.stream()``
  (``lobster/core/runs.py:149-155, 70``) drive the run; ``cancel`` at
  ``runs.py:205`` stops a run that needs a decision.
- ``DataManager.export_notebook`` (``lobster/core/runtime/data_manager.py:5337``)
  writes the notebook from the recorded steps.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pytest

from lobster.cli_internal import mcp_contract, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, RunRequest, ThreadRunner
from lobster.cli_internal.startup_diagnostics import (
    StartupDiagnostic,
    StartupDiagnosticError,
)
from lobster.core.runs import CANCEL_REASONS, new_run_id
from tests.unit.mcp.replay_input import loading

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "tests" / "fixtures" / "protocol-v1" / "replay"


def call(server: McpServer, name: str, arguments: dict) -> dict:
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    result = server.handle_line(line)["result"]
    if result.get("isError"):
        return {"error_text": result["content"][0]["text"]}
    assert set(result) == {"content"}
    return json.loads(result["content"][0]["text"])


def wait_terminal(server: McpServer, run_id: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = call(server, "lobster_status", {"run_id": run_id})
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} not terminal after {timeout}s")


@pytest.fixture
def data(tmp_path):
    source = tmp_path / "pbmc.h5ad"
    source.write_bytes(b"")
    return source


def run_args(source: Path, out: Path) -> dict:
    return {"input_path": str(source), "output_dir": str(out), "analysis": "scrna_qc"}


class Gate:
    """Injected work that blocks until released, then writes files."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = []

    def __call__(self, request: RunRequest, cancel_event: threading.Event) -> dict:
        self.calls.append(request)
        self.entered.set()
        self.release.wait(10)
        out = Path(request.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text("{}")
        (out / "nb.ipynb").write_text("{}")
        return {
            "status": "completed",
            "summary": "done",
            "result_path": str(out / "result.json"),
            "notebook_path": str(out / "nb.ipynb"),
        }


def test_run_returns_at_once_and_serves_status_while_blocked(tmp_path, data):
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    # The gate is closed: an answer here proves lobster_run did not wait.
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert run["status"] == "running" and run["run_id"].startswith("run_")
    assert gate.entered.wait(10) and not gate.release.is_set()

    for _ in range(2):  # the read loop keeps serving while the run is blocked
        live = call(server, "lobster_status", {"run_id": run["run_id"]})
        # The injected work never began an engine run: still starting, no step.
        assert live == {
            "status": "running",
            "run_id": run["run_id"],
            "phase": "starting",
            "elapsed_s": live["elapsed_s"],
            "poll_after_s": mcp_contract.POLL_AFTER_S,
        }
        assert isinstance(live["elapsed_s"], int) and live["elapsed_s"] >= 0
    gate.release.set()
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "completed" and done["run_id"] == run["run_id"]
    for key in ("result_path", "notebook_path"):
        path = Path(done[key])
        assert path.is_absolute() and path.parent == tmp_path / "out" and path.exists()


@pytest.mark.parametrize(
    "make_args, needle",
    [
        (lambda d, o: {**run_args(d, o), "input_path": "pbmc.h5ad"}, "absolute"),
        (lambda d, o: {**run_args(d, o), "output_dir": "out"}, "absolute"),
        (lambda d, o: {**run_args(d, o), "input_path": str(d) + ".missing"}, "exist"),
        (lambda d, o: {**run_args(d, o), "output_dir": str(d)}, "not a directory"),
        (lambda d, o: {**run_args(d, o), "input_path": str(o / "a\nb.h5ad")}, "exist"),
    ],
)
def test_bad_paths_do_not_start_a_run(tmp_path, data, make_args, needle):
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    refused = call(server, "lobster_run", make_args(data, tmp_path / "out"))
    assert needle in refused["error_text"] and "\n" not in refused["error_text"]
    assert gate.calls == []
    gate.release.set()
    assert "run_id" in call(server, "lobster_run", run_args(data, tmp_path / "out"))


def test_failed_run_has_no_traceback_and_no_notebook(tmp_path, data):
    def boom(request, cancel_event):
        raise RuntimeError("engine fell over\nsecond line")

    server = McpServer(ThreadRunner(boom))
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    failed = wait_terminal(server, run["run_id"])
    assert failed["status"] == "failed" and "notebook_path" not in failed
    assert failed["error"]["code"] == "exception"
    assert failed["error"]["retryable"] is False
    assert failed["summary"] == "exception: engine fell over second line"
    assert "Traceback" not in json.dumps(failed)


def test_one_active_run_then_the_next(tmp_path, data):
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    first = call(server, "lobster_run", run_args(data, tmp_path / "a"))
    assert gate.entered.wait(10)
    busy = call(server, "lobster_run", run_args(data, tmp_path / "b"))
    assert busy["error_text"].startswith("engine_busy:")
    assert first["run_id"] in busy["error_text"]
    assert len(gate.calls) == 1

    gate.release.set()
    assert wait_terminal(server, first["run_id"])["status"] == "completed"
    gate.release.clear()
    gate.entered.clear()
    second = call(server, "lobster_run", run_args(data, tmp_path / "b"))
    assert second["run_id"] != first["run_id"]
    assert call(server, "lobster_status", {"run_id": second["run_id"]})["status"] == (
        "running"
    )
    assert call(server, "lobster_status", {"run_id": first["run_id"]})["status"] == (
        "completed"
    )
    gate.release.set()
    wait_terminal(server, second["run_id"])


def test_eof_cancels_the_active_run(tmp_path, data):
    seen = threading.Event()

    def until_cancelled(request, cancel_event):
        seen.set()
        cancel_event.wait(10)
        return {"status": "cancelled", "summary": "stopped", "cancel_reason": "x"}

    runner = ThreadRunner(until_cancelled)
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lobster_run", "arguments": run_args(data, tmp_path)},
        }
    )
    out = io.StringIO()
    started = time.monotonic()
    assert (
        mcp_stdio.run_mcp_stdio(
            runner=runner, stdin=io.StringIO(line + "\n"), stdout=out
        )
        == 0
    )
    # Bounded: the work honours the cancel, so EOF does not wait it out.
    assert seen.is_set() and time.monotonic() - started < 30
    run_id = json.loads(json.loads(out.getvalue())["result"]["content"][0]["text"])[
        "run_id"
    ]
    assert runner.status(run_id)["status"] == "cancelled"


# --- production binding through the run API, with a scripted client ---------


class FakeProvenance:
    def __init__(self, activities):
        self.activities = activities


class FakeDataManager:
    def __init__(self, workspace: Path, export_fails: bool, activities=()):
        self.workspace = workspace
        self.export_fails = export_fails
        if activities is not None:
            self.provenance = FakeProvenance(list(activities))
        self.modalities = {"pbmc": ad.AnnData(np.zeros((3, 2)))}

    def provenance_ok(self):
        return hasattr(self, "provenance")

    def log_tool_usage(self, tool_name, parameters, description=None, ir=None):
        if not self.provenance_ok():  # like the real one without a tracker
            return None
        activity = {"type": tool_name, "parameters": parameters, "ir": ir}
        self.provenance.activities.append(activity)
        return activity

    def list_modalities(self):
        return ["pbmc"]

    def remove_modality(self, name):
        self.removed = [*getattr(self, "removed", []), name]

    def list_modalities_with_lineage(self):
        return [
            {
                "name": "pbmc",
                "base_name": "pbmc",
                "version": 1,
                "processing_step": "raw",
                "parent_modality": None,
                "created_at": "2026-11-13T10:00:00",
                "shape": {"n_obs": 3, "n_vars": 2},
            }
        ]

    def export_notebook(self, name, description="", filter_strategy="successful"):
        if self.export_fails:
            raise ValueError("No activities recorded - nothing to export")
        path = self.workspace / "notebooks" / f"{name}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path


class ScriptedClient:
    session_id = "session_test"

    def __init__(
        self,
        workspace: Path,
        events,
        export_fails=False,
        activities=(),
        resume_events=(),
        loads_input=True,
    ):
        self.data_manager = FakeDataManager(workspace, export_fails, activities)
        self.loads_input = loads_input
        self.events = events
        self.resume_events = resume_events
        self.prompts = []
        self.resumes = []
        self.loaded = []  # session files restored with load_session

    def load_session(self, session_path):
        self.loaded.append(session_path)
        return {"success": True, "messages_loaded": 2, "provenance_restored": 0}

    def query(self, prompt, stream=True, cancel_event=None):
        # A first turn reads the file its recipe names, as the real model does.
        named = re.search(r"^Input file: (.+)$", prompt, re.M)
        if self.loads_input and named and not self.prompts:
            self.data_manager.log_tool_usage(
                "load_dataset", {"name": "input", "source": named[1], "adapter": "x"}
            )
        self.prompts.append(prompt)
        yield from self._play(self.events, cancel_event)

    def resume_from_interrupt(self, response, stream=True, cancel_event=None):
        self.resumes.append(response)
        events = self.resume_events
        if events and isinstance(events[0], list):  # one script per resume
            events = events.pop(0)
        yield from self._play(events, cancel_event)

    def _play(self, events, cancel_event):
        for event in events:
            if isinstance(event, Exception):
                raise event
            if isinstance(event, Hold):
                if self.data_manager.provenance_ok():
                    self.data_manager.provenance.activities.extend(
                        {"type": name} for name in event.types
                    )
                event.entered.set()
                event.release.wait(10)
                continue
            if isinstance(event, Write):
                event.apply()
                continue
            if isinstance(event, Act):
                self.data_manager.log_tool_usage(event.tool, event.parameters)
                continue
            if isinstance(event, WaitForCancel):
                # Like the real client: the stream ends once the cancel is seen.
                event.entered.set()
                cancel_event.wait(10)
                return
            yield event


class Write:
    """Scripted step: the session writes a file in place, as a tool does."""

    def __init__(self, path: Path, content: bytes):
        self.path, self.content = path, content

    def apply(self):
        with open(self.path, "wb") as handle:  # same inode, like an h5ad rewrite
            handle.write(self.content)


class Act:
    """Scripted step: the session records one activity, as a tool does."""

    def __init__(self, tool: str, **parameters):
        self.tool, self.parameters = tool, parameters


class WaitForCancel:
    """Scripted step that blocks until the run's cancel_event is set."""

    def __init__(self):
        self.entered = threading.Event()


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """Patch the session factory; returns (opened sessions, set_events)."""
    from lobster.cli_internal.commands.heavy import session_infra

    opened = []
    script = {
        "events": [],
        "export_fails": False,
        "raise": None,
        "activities": (),
        "resume_events": [],
        "loads_input": True,  # the first turn loads its input file
        "files": {},  # path under the workspace -> bytes, there when it opens
    }

    def fake_init(workspace=None, interactive=True, session_id=None, **kwargs):
        # The real start-up creates the workspace before it can fail.
        Path(workspace).mkdir(parents=True, exist_ok=True)
        if script["raise"]:
            raise script["raise"]
        for relative, content in script["files"].items():
            target = Path(workspace) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        client = ScriptedClient(
            workspace,
            script["events"],
            export_fails=script["export_fails"],
            activities=script["activities"],
            resume_events=script["resume_events"],
            loads_input=script["loads_input"],
        )
        opened.append(
            {
                "workspace": workspace,
                "interactive": interactive,
                "client": client,
                "session_id": session_id,
            }
        )
        return client

    monkeypatch.setattr(
        session_infra, "init_client_or_raise_startup_diagnostic", fake_init
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return opened, script


HANDLE = new_run_id()


def run_engine(tmp_path, data):
    request = RunRequest(HANDLE, str(data), str(tmp_path / "out"), "scrna_qc")
    return mcp_stdio.run_engine(request, threading.Event())


COMPLETE = {
    "type": "complete",
    "response": "QC kept 2638 of 2700 cells.",
    "duration": 1.5,
    "token_usage": {"input": 10, "output": 5},
    "last_agent": "supervisor",
    "session_id": "session_test",
}


@pytest.mark.parametrize("export_fails", [False, True])
def test_engine_completed(engine, tmp_path, data, export_fails):
    opened, script = engine
    script.update(events=[COMPLETE], export_fails=export_fails)
    status = run_engine(tmp_path, data)

    session = opened[0]
    assert session["workspace"] == tmp_path / "out" / ".lobster_workspace"
    assert session["interactive"] is True  # the host can answer a question
    assert str(data) in session["client"].prompts[0]
    assert list((tmp_path / "cwd").iterdir()) == []  # nothing in the process cwd

    assert status["status"] == "completed"
    assert status["summary"] == COMPLETE["response"]
    result = status["result"]
    assert result["run_id"] == HANDLE  # the MCP handle is the engine's run id
    assert result["schema_version"] == "lobster.result/1"
    assert result["has_data"] is True
    assert [m["name"] for m in status["modality_details"]] == ["pbmc"]
    on_disk = json.loads(Path(status["result_path"]).read_text())
    # The summary travels once: the status' result drops it, result.json keeps it
    # (and the record of the request and the run). The modality names stay in
    # both; the objects are modality_details.
    assert "response" not in result and result["modalities"] == ["pbmc"]
    assert on_disk.pop("response") == COMPLETE["response"]
    assert on_disk.pop("modality_details") == status["modality_details"]
    for key in ("request", "steps", "metrics", "warnings"):
        del on_disk[key]
    assert on_disk == json.loads(json.dumps(result))
    if export_fails:
        assert status["notebook_path"] is None
    else:
        notebook = Path(status["notebook_path"])
        assert notebook.is_absolute() and notebook.exists()
        assert notebook.name == "lobster_scrna_qc.ipynb"


def test_engine_error_event_is_the_run_api_failure(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [
        {
            "type": "error",
            "error": "provider down",
            "error_type": "provider_error",
            "is_rate_limit": True,
        }
    ]
    status = run_engine(tmp_path, data)
    assert status["status"] == "failed"
    assert status["error"] == {
        "code": "provider_error",
        "message": "provider down",
        "retryable": True,
    }
    assert "notebook_path" not in status


def test_engine_stream_exception_is_the_run_api_failure(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [RuntimeError("model call crashed")]
    status = run_engine(tmp_path, data)
    assert status["status"] == "failed"
    # runs.py records {code: exception, message: str(exc)} before re-raising.
    assert status["error"] == {
        "code": "exception",
        "message": "model call crashed",
        "retryable": False,
    }


def test_engine_two_questions_at_once_are_stopped_as_unsupported(
    engine, tmp_path, data
):
    """One answer per run: a batch of two questions cannot be answered here."""
    _opened, script = engine

    def slider(interrupt_id, ask_id, label):
        return {
            "type": "interrupt",
            "interrupt_id": interrupt_id,
            "data": {
                "component": "threshold_slider",
                "data": {"label": label, "min": 0.0, "max": 1.0, "default": 0.05},
                "fallback_prompt": label,
                "ask_id": ask_id,
            },
        }

    script["events"] = [
        slider("lg_1", "call_1", "Which mitochondrial threshold?"),
        slider("lg_2", "call_2", "Which gene threshold?"),
    ]
    status = run_engine(tmp_path, data)
    assert status["status"] == "cancelled"
    assert status["cancel_reason"] == "decision_unsupported"
    assert "Which mitochondrial threshold?" in status["summary"]
    assert "one question at a time" in status["summary"]
    from lobster.core.runs import runs_for

    manager = runs_for(_opened[0]["client"])
    run = manager.runs[HANDLE]
    assert run.events[-1]["reason"] == "decision_unsupported"
    assert [d.state for d in manager.decisions.values()] == ["cancelled"] * 2
    text = json.dumps(status)
    assert "dec_" not in text and "lg_1" not in text and "notebook_path" not in text


def test_engine_startup_diagnostic_fails_the_run(engine, tmp_path, data):
    _opened, script = engine
    script["raise"] = StartupDiagnosticError(
        StartupDiagnostic(code="no_provider", title="No LLM provider is configured")
    )
    status = run_engine(tmp_path, data)
    assert status["status"] == "failed"
    assert status["error"] == {
        "code": "no_provider",
        "message": "No LLM provider is configured",
        "retryable": False,
    }


def test_engine_cancel_before_start_opens_no_session(engine, tmp_path, data):
    opened, _script = engine
    cancel = threading.Event()
    cancel.set()
    request = RunRequest(new_run_id(), str(data), str(tmp_path / "out"), "scrna_qc")
    status = mcp_stdio.run_engine(request, cancel)
    assert status["status"] == "cancelled"
    assert status["summary"] == "Stopped before the analysis started."
    assert status["cancel_reason"] == "cancel"
    assert opened == []


def test_engine_cancel_during_session_start_sends_no_prompt(
    engine, tmp_path, data, monkeypatch
):
    """A cancel that lands while the session starts: no prompt is sent."""
    from lobster.cli_internal.commands.heavy import session_infra

    opened, _script = engine
    cancel = threading.Event()
    fake_init = session_infra.init_client_or_raise_startup_diagnostic

    def init_then_cancel(**kwargs):
        client = fake_init(**kwargs)
        cancel.set()
        return client

    monkeypatch.setattr(
        session_infra, "init_client_or_raise_startup_diagnostic", init_then_cancel
    )
    request = RunRequest(new_run_id(), str(data), str(tmp_path / "out"), "scrna_qc")
    status = mcp_stdio.run_engine(request, cancel)
    assert status["status"] == "cancelled"
    assert status["summary"] == (
        "Stopped while the session was starting; the analysis did not start."
    )
    assert status["cancel_reason"] == "cancel"
    assert opened[0]["client"].prompts == []


# --- the real engine, end to end, on a replay script (no network) ------------


_ENTRY = (
    "import sys\n"
    "from lobster.cli_internal.mcp_stdio import run_mcp_stdio\n"
    "sys.exit(run_mcp_stdio())\n"
)


def _mcp_session(script: Union[str, Path], tmp_path: Path, data: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "LOBSTER_TEST_REPLAY_FIXTURE": str(REPLAY / script),
            # Startup requires a configured provider; the replay model answers.
            "LOBSTER_LLM_PROVIDER": "ollama-cloud",
            "OLLAMA_API_KEY": "replay-dummy",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    proc = subprocess.Popen(  # nosec B603
        # The server entry without the CLI root callback (its update notice
        # goes to stdout until the seam fixes it); test_contract covers the CLI.
        [sys.executable, "-c", _ENTRY],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=cwd,
    )
    try:

        def rpc(n, method, params):
            proc.stdin.write(
                json.dumps(
                    {"jsonrpc": "2.0", "id": n, "method": method, "params": params}
                )
                + "\n"
            )
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            return json.loads(reply["result"]["content"][0]["text"])

        run = rpc(
            1,
            "tools/call",
            {"name": "lobster_run", "arguments": run_args(data, tmp_path / "out")},
        )
        deadline = time.monotonic() + 240
        n = 1
        while time.monotonic() < deadline:
            n += 1
            status = rpc(
                n,
                "tools/call",
                {"name": "lobster_status", "arguments": {"run_id": run["run_id"]}},
            )
            if status["status"] != "running":
                break
            time.sleep(0.5)
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert not (cwd / ".lobster_workspace").exists()
    return status


@pytest.mark.slow
def test_real_engine_completes_on_replay(tmp_path, data):
    script = loading("query_loaded.json", tmp_path, data)
    status = _mcp_session(script, tmp_path, data)
    assert status["status"] == "completed", status
    assert status["result"]["schema_version"] == "lobster.result/1"
    assert status["summary"]
    assert Path(status["result_path"]).parent == tmp_path / "out"


@pytest.mark.slow
def test_real_engine_failure_on_replay(tmp_path, data):
    status = _mcp_session("failure.json", tmp_path, data)
    assert status["status"] == "failed", status
    assert set(status["error"]) == {"code", "message", "retryable"}


# Result bounds: both sides of each bound.
def _completed(summary: str, pad=()) -> dict:
    return {
        "status": "completed",
        "run_id": "r1",
        "summary": summary,
        "result_path": "/out/result.json",
        "notebook_path": None,
        "result": {"response": summary[:10], "pad": list(pad)},
    }


def _padded(size: int) -> dict:
    """A completed status whose JSON is exactly *size* characters, made of many
    short texts (a text longer than a field is cut before the size is judged)."""
    status = _completed("done")
    pad = status["result"]["pad"]
    while size - len(json.dumps(status, default=str)) - 4 > 900:
        pad.append("p" * 400)
    pad.append("")
    pad[-1] = "p" * (size - len(json.dumps(status, default=str)))
    return status


@pytest.mark.parametrize("extra, cut", [(0, False), (1, True)])
def test_summary_bound(extra, cut):
    summary = "a" * (mcp_stdio.SUMMARY_MAX_CHARS + extra)
    bounded = mcp_stdio._bounded(_completed(summary))["summary"]
    assert len(bounded) == mcp_stdio.SUMMARY_MAX_CHARS
    assert bounded.endswith(mcp_stdio.SUMMARY_CUT) is cut


def test_summary_cut_never_splits_an_astral_character():
    keep = mcp_stdio.SUMMARY_MAX_CHARS - len(mcp_stdio.SUMMARY_CUT)
    summary = "a" * (keep - 1) + "\U0001f600" * 50
    bounded = mcp_stdio._bounded(_completed(summary))["summary"]
    assert bounded == "a" * (keep - 1) + "\U0001f600" + mcp_stdio.SUMMARY_CUT
    bounded.encode("utf-8")  # a lone surrogate would raise here


@pytest.mark.parametrize(
    "size, omitted",
    [(mcp_stdio.STATUS_MAX_CHARS - 1, False), (mcp_stdio.STATUS_MAX_CHARS, True)],
)
def test_status_bound(size, omitted):
    status = _padded(size)
    assert len(json.dumps(status, default=str)) == size
    bounded = mcp_stdio._bounded(status)
    assert ("result" in bounded) is not omitted
    assert bounded.get("result_omitted", False) is omitted
    assert len(json.dumps(bounded, default=str)) < mcp_stdio.STATUS_MAX_CHARS


def test_thread_runner_bounds_the_final_status(tmp_path, data):
    def big(request, cancel_event):
        text = "x" * 50_000
        return {**_completed(text), "result": {"pad": ["x" * 500] * 100}}

    server = McpServer(ThreadRunner(big))
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    done = wait_terminal(server, run_id)
    assert done["run_id"] == run_id and done["result_omitted"] is True
    assert "result" not in done
    assert done["summary"].endswith(mcp_stdio.SUMMARY_CUT)


# lobster_cancel.
def test_cancel_stops_a_running_run(tmp_path, data):
    entered, finish = threading.Event(), threading.Event()

    def slow_step(request, cancel_event):
        entered.set()
        cancel_event.wait(10)
        finish.wait(10)  # the step in progress returns only when released
        return {
            "status": "cancelled",
            "summary": "stopped",
            "cancel_reason": "cancel",
        }

    server = McpServer(ThreadRunner(slow_step))
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "a"))["run_id"]
    assert entered.wait(10)
    first = call(server, "lobster_cancel", {"run_id": run_id})
    assert first == {"run_id": run_id, "outcome": "cancel_requested"}
    # Cooperative: until the step returns, the run is still active.
    assert call(server, "lobster_status", {"run_id": run_id})["status"] == "running"
    busy = call(server, "lobster_run", run_args(data, tmp_path / "b"))
    assert busy["error_text"].startswith("engine_busy:")
    again = call(server, "lobster_cancel", {"run_id": run_id})
    assert again["outcome"] == "cancel_requested"  # idempotent while running

    finish.set()
    assert wait_terminal(server, run_id)["status"] == "cancelled"
    after = call(server, "lobster_cancel", {"run_id": run_id})
    assert after == {"run_id": run_id, "outcome": "already_terminal"}
    assert "run_id" in call(server, "lobster_run", run_args(data, tmp_path / "b"))


def test_cancel_unknown_run_is_a_tool_error():
    server = McpServer(ThreadRunner(lambda request, cancel_event: {}))
    refused = call(server, "lobster_cancel", {"run_id": "nope"})
    assert refused["error_text"] == "Unknown run_id 'nope'"
    assert "error_text" in call(server, "lobster_cancel", {})


def test_cancel_through_the_thread_runner_and_the_run_api(engine, tmp_path, data):
    """Production path: ThreadRunner(run_engine) -> runs_for(client).start; only
    the session is scripted."""
    opened, script = engine
    step = WaitForCancel()
    script.update(events=[step], loads_input=False)  # no step before the cancel
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert step.entered.wait(10)
    outcome = call(server, "lobster_cancel", {"run_id": run_id})["outcome"]
    assert outcome == "cancel_requested"
    done = wait_terminal(server, run_id)
    assert done["status"] == "cancelled" and done["cancel_reason"] == "cancel"
    assert done["summary"] == (
        "Stopped before the analysis finished. No analysis step had completed."
    )
    assert len(opened) == 1 and opened[0]["client"].prompts
    assert call(server, "lobster_cancel", {"run_id": run_id})["outcome"] == (
        "already_terminal"
    )


def test_fake_runner_cancel_uses_an_engine_reason(tmp_path, data):
    runner = mcp_stdio.FakeRunner()
    run_id = new_run_id()
    runner.start(RunRequest(run_id, str(data), str(tmp_path / "out"), "scrna_qc"))
    assert runner.cancel(run_id) == "cancel_requested"
    assert runner.status(run_id)["cancel_reason"] in CANCEL_REASONS


def test_query_that_raises_at_start_fails_the_handle(
    engine, tmp_path, data, monkeypatch
):
    """run API: start() ends the run failed/exception and re-raises; the host
    sees a failed run, never a handle left running, and may start again."""

    def raising(self, prompt, stream=True, cancel_event=None):
        raise RuntimeError("provider refused")

    monkeypatch.setattr(ScriptedClient, "query", raising)
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "a"))["run_id"]
    done = wait_terminal(server, run_id)
    assert done["status"] == "failed"
    assert done["error"]["code"] == "exception"
    assert "provider refused" in done["error"]["message"]
    again = call(server, "lobster_run", run_args(data, tmp_path / "b"))
    assert again["status"] == "running"


# --- progress: elapsed time, phase and step in the status -------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_elapsed_counts_from_the_accepted_run_and_stops_at_the_end(
    tmp_path, data, monkeypatch
):
    clock = Clock()
    monkeypatch.setattr(mcp_stdio, "_now", clock)
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert gate.entered.wait(10)
    clock.now += 56.9  # start-up before the first step counts
    assert call(server, "lobster_status", {"run_id": run_id})["elapsed_s"] == 56
    clock.now += 28
    gate.release.set()
    done = wait_terminal(server, run_id)
    assert done["elapsed_s"] == 84
    clock.now += 100
    assert call(server, "lobster_status", {"run_id": run_id})["elapsed_s"] == 84


def test_failed_and_cancelled_statuses_carry_elapsed_s(tmp_path, data):
    def boom(request, cancel_event):
        raise RuntimeError("boom")

    def stopped(request, cancel_event):
        return {"status": "cancelled", "summary": "stopped", "cancel_reason": "cancel"}

    for work, state in ((boom, "failed"), (stopped, "cancelled")):
        server = McpServer(ThreadRunner(work))
        run_id = call(server, "lobster_run", run_args(data, tmp_path / state))["run_id"]
        done = wait_terminal(server, run_id)
        assert done["status"] == state
        assert isinstance(done["elapsed_s"], int) and done["elapsed_s"] >= 0
        assert "phase" not in done and "poll_after_s" not in done


class Hold:
    """Scripted step: records provenance activities, then blocks until released."""

    def __init__(self, *types):
        self.types = types
        self.entered = threading.Event()
        self.release = threading.Event()


def test_running_status_names_the_latest_recorded_step(
    engine, monkeypatch, tmp_path, data
):
    clock = Clock()
    monkeypatch.setattr(mcp_stdio, "_now", clock)
    _opened, script = engine
    hold = Hold("quality_assessment", "filter_and_normalize")
    script["events"] = [hold, COMPLETE]
    script["activities"] = [{"type": "left_over_from_an_earlier_session"}]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    clock.now += 12
    live = call(server, "lobster_status", {"run_id": run_id})
    assert live == {
        "status": "running",
        "run_id": run_id,
        "phase": "running",
        "elapsed_s": 12,
        "step": "filter_and_normalize",
        "steps_done": 3,  # the load of the input and two steps; not the earlier session's
        "poll_after_s": mcp_contract.POLL_AFTER_S,
    }
    hold.release.set()
    done = wait_terminal(server, run_id)
    assert done["status"] == "completed" and done["elapsed_s"] == 12
    assert not {"phase", "step", "steps_done", "poll_after_s"} & set(done)
    # The summary is sent once; the whole result stays in result.json.
    assert "response" not in done["result"]
    assert json.dumps(done).count(COMPLETE["response"]) == 1
    on_disk = json.loads(Path(done["result_path"]).read_text())
    assert on_disk["response"] == COMPLETE["response"]


def test_running_status_has_no_step_before_one_is_recorded(engine, tmp_path, data):
    _opened, script = engine
    script["loads_input"] = False  # nothing is recorded before the hold
    hold = Hold()
    script["events"] = [hold, COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    live = call(server, "lobster_status", {"run_id": run_id})
    assert live["phase"] == "running" and live["steps_done"] == 0
    assert "step" not in live
    hold.release.set()
    wait_terminal(server, run_id)


def test_running_status_without_a_provenance_source_omits_step_fields(
    engine, tmp_path, data
):
    _opened, script = engine
    script["activities"] = None  # a session with no provenance tracker
    hold = Hold()
    script["events"] = [hold, COMPLETE]
    server = McpServer(ThreadRunner())
    run_id = call(server, "lobster_run", run_args(data, tmp_path / "out"))["run_id"]
    assert hold.entered.wait(10)
    live = call(server, "lobster_status", {"run_id": run_id})
    assert live["phase"] == "running"
    assert "step" not in live and "steps_done" not in live
    hold.release.set()
    wait_terminal(server, run_id)


def test_fake_runner_statuses_carry_the_new_fields(tmp_path, data):
    runner = mcp_stdio.FakeRunner()
    run_id = new_run_id()
    runner.start(RunRequest(run_id, str(data), str(tmp_path / "out"), "scrna_qc"))
    live = runner.status(run_id)
    assert live["status"] == "running" and live["phase"] == "running"
    assert live["poll_after_s"] == mcp_contract.POLL_AFTER_S
    assert isinstance(live["elapsed_s"], int) and live["step"]
    done = runner.status(run_id)
    assert done["status"] == "completed" and isinstance(done["elapsed_s"], int)
    assert "phase" not in done


def _described(count: int) -> dict:
    """A completed status with more modalities than the bound can hold."""
    return {
        **_completed("done", pad="p" * 100),
        "metrics": {"cells_before": 3},
        "warnings": [{"code": "no_qc_plots", "message": "none"}],
        "files": [{"path": "/out/result.json", "category": "result"}],
        "steps": [{"seq": i, "tool": "t" * 100} for i in range(5)],
        "modality_details": [
            {"name": f"m{i}", "stage": "raw", "n_obs": i, "file": "f" * 300}
            for i in range(count)
        ],
    }


def test_status_bound_drops_result_then_steps_then_modality_detail():
    small = len(json.dumps(_described(0), default=str))
    bound = mcp_stdio.STATUS_MAX_CHARS
    assert small < bound
    per_modality = len(json.dumps(_described(1), default=str)) - small

    fits = mcp_stdio._bounded(_described((bound - small) // per_modality - 1))
    assert {"result", "steps"} <= set(fits) and "result_omitted" not in fits

    # Too long only because of the modalities: the result goes, the steps go,
    # then the detail; metrics, warnings, files and paths always stay.
    long = mcp_stdio._bounded(_described(bound // per_modality + 5))
    assert "result" not in long and long["result_omitted"] is True
    assert "steps" not in long and long["steps_omitted"] is True
    assert long["modality_details"][0] == {"name": "m0"}
    assert long["modality_details_names_only"] is True
    assert long["metrics"] == {"cells_before": 3}
    assert long["warnings"] and long["files"] and long["result_path"]
    assert len(json.dumps(long, default=str)) < bound
