"""``lobster mcp --stdio``: Lobster as an MCP server for desktop and agent hosts.

Wire format is MCP's stdio transport: one JSON-RPC 2.0 message per line on
stdin and stdout. This is not protocol v1 (``lobster agent --stdio`` speaks
that); the two commands share no framing.

* stdout carries only JSON-RPC messages; logs and stray prints go to stderr.
* ``initialize``, ``ping`` and ``tools/list`` answer without importing the
  engine: hosts give up on any request after 60 seconds.
* ``lobster_run`` returns at once with a ``run_id``; ``lobster_status`` is a
  second call. One run at a time; a run lives in this process and dies with
  it. Production runs go through the run API (``lobster.core.runs``) with
  ``output_dir`` as the session workspace.
* A completed run leaves its session idle in this process (one only); a
  later ``lobster_continue`` adds a turn to it, or restores it from
  ``output_dir`` after a restart.
* A question from the engine puts the run in ``needs_decision``; the worker
  waits (no timer) until ``lobster_answer``, ``lobster_cancel`` or EOF.
* A tool result is one text block holding a JSON object and nothing else, the
  shape hosts unwrap into a plain value. Its first key, ``headline``, is one
  line for the host to show its user (``mcp_headline``). Tool failures set
  ``isError`` with a one-line message; protocol mistakes are JSON-RPC errors.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from lobster.cli_internal import mcp_contract as contract
from lobster.cli_internal import (
    mcp_decision,
    mcp_headline,
    mcp_inspect,
    mcp_lock,
    mcp_overview,
    mcp_record,
    mcp_turns,
    mcp_watch,
    response_style,
)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# The supervisor gets its ask_user tool and the client turns an interrupt into
# a decision the host answers with ``lobster_answer``. Without the tool every
# question would stop the run again, so the two ship together.
INTERACTIVE = True
# How long EOF waits for the active run to stop at its next graph event.
SHUTDOWN_WAIT_S = 5.0
# How often lobster_wait re-reads a run's status.
WAIT_STEP_S = 0.25
# Tools answered on a thread of their own, so a long call holds no other reply.
_SLOW_TOOLS = frozenset({"lobster_wait", "lobster_docs"})
# Result bounds: Codex truncates a tool result at
# 10k tokens, Claude Code moves text over 50,000 characters into a file.
SUMMARY_MAX_CHARS = 8000
# The summary's length when the rest of the status leaves it no more room.
SUMMARY_MIN_CHARS = 2000
STATUS_MAX_CHARS = 40_000
SUMMARY_CUT = "... (cut; full text in result_path)"
# Any other text in a status (names, messages, parameter values) and an error
# message: one reply must not carry a provider's whole response body.
FIELD_MAX_CHARS = 1000
ERROR_CODE_MAX_CHARS = 100
# Paths are the host's way to the files: a cut one would point nowhere.
_WHOLE_FIELDS = frozenset({"summary", "path", "result_path", "notebook_path"})
# Answers from the host: free text and comments are bounded (they reach the
# provenance, the notebook and the model); the reply echoes only a short part.
ANSWER_MAX_CHARS = 4000
ECHO_MAX_CHARS = 200
MAX_FILES = 50


_now = time.monotonic


class ToolError(Exception):
    """A tool call the server refuses; the message reaches the model as is."""


class RunProgress:
    """What a run reports while it works, and where it waits for an answer.

    The worker thread calls ``begin`` once when the engine run starts (a probe
    returning step fields, or raising when the session exposes none) and
    ``wait_for_answer`` at each question; the server thread calls ``snapshot``
    and ``submit``. One lock guards all of it. Until ``begin`` the phase is
    ``starting``.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._probe: Optional[Callable[[], Dict[str, Any]]] = None
        self._manager: Any = None
        self._waiting: Optional[Dict[str, Any]] = None  # the decision shown
        self._answered: Any = None  # the Run an accepted answer started
        self._given: List[Dict[str, Any]] = []  # accepted answers, as recorded
        self._given_total = 0
        # A first turn's plan: did the host ask for one, did the run ask it.
        self._plan: Optional[Dict[str, bool]] = None

    def plan_requested(self, requested: bool) -> None:
        with self._cond:
            self._plan = {"requested": requested, "asked": False}

    def plan_asked(self) -> None:
        with self._cond:
            if self._plan is not None:
                self._plan["asked"] = True

    def plan_refused(self, tool: str) -> None:
        """Name *tool* in ``plan.refused``: it was called before the plan's
        answer and did not run."""
        with self._cond:
            if self._plan is not None:
                refused = self._plan.setdefault("refused", [])
                if tool not in refused:
                    refused.append(tool)

    def plan(self) -> Optional[Dict[str, Any]]:
        with self._cond:
            if self._plan is None:
                return None
            plan = dict(self._plan)
            if "refused" in plan:
                plan["refused"] = list(plan["refused"])
            return plan

    def begin(self, probe: Callable[[], Dict[str, Any]], manager: Any = None) -> None:
        with self._cond:
            self._probe = probe
            self._manager = manager

    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            probe, waiting = self._probe, self._waiting
        if waiting is not None:
            return {"phase": "waiting", "decision": mcp_decision.public(waiting)}
        if probe is None:
            return {"phase": "starting"}
        fields: Dict[str, Any] = {}
        try:
            fields = probe()
        except Exception:  # unknown stays absent, never invented
            pass
        from lobster.utils import download_activity  # heavy package: not at start

        download = download_activity.snapshot()
        return {
            "phase": "running",
            **fields,
            **({"download": download} if download else {}),
        }

    def answers_given(self) -> Tuple[List[Dict[str, Any]], int]:
        """``(entries, total)`` of the answers accepted so far."""
        with self._cond:
            return list(self._given), self._given_total

    def is_waiting(self) -> bool:
        with self._cond:
            return self._waiting is not None

    def wake(self) -> None:
        """Let a waiting worker re-read its cancel event."""
        with self._cond:
            self._cond.notify_all()

    def wait_for_answer(
        self, run: Any, decision: Dict[str, Any], cancel_event: threading.Event
    ) -> Any:
        """Block until an answer starts the next run, which is returned, or
        until the cancel event is set: then the decision is abandoned (the run
        ends ``cancelled``, reason ``decision_abandoned``) and None returned.

        No deadline. The one-second wake-up only re-reads the cancel event.
        """
        with self._cond:
            self._waiting = decision
            while self._answered is None and not cancel_event.is_set():
                self._cond.wait(1.0)
            self._waiting = None
            following, self._answered = self._answered, None
            if following is None:
                # Under the lock: a lobster_answer that lost the race now sees
                # a cancelled decision, never a window where it can still win.
                self._manager.cancel(run.run_id, reason="decision_abandoned")
            return following

    def submit(
        self,
        run_id: str,
        state: str,
        decision_id: str,
        request_version: int,
        answer: Any,
        source: str,
        comment: Optional[str],
        channel: str,
        cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """Hand an answer to the waiting worker; a refusal is a ToolError and
        changes nothing."""
        from lobster.core.runs import DecisionError

        with self._cond:
            manager, waiting = self._manager, self._waiting
            known = manager.decisions.get(decision_id) if manager else None
            closed = known.state if known is not None else None
            if waiting is None or decision_id != waiting["decision_id"]:
                if closed == "answered":
                    raise ToolError(_ALREADY_ANSWERED)
                if closed == "cancelled":
                    raise ToolError(_CANCELLED)
                if waiting is None:
                    raise ToolError(
                        f"not_waiting: run {run_id} is {state}; there is no "
                        "question to answer."
                    )
                raise ToolError(
                    f"unknown_decision: no open decision {decision_id} on run "
                    f"{run_id}; read lobster_status again."
                )
            if cancel_event.is_set():
                # A cancel was asked for and the worker has not woken yet: the
                # answer would start work nobody wants.
                raise ToolError(_CANCELLED)
            if request_version != waiting["request_version"]:
                raise ToolError(_stale(waiting["request_version"]))
            try:
                recorded = mcp_decision.check_answer(waiting, answer)
            except mcp_decision.BadAnswer as exc:
                raise ToolError(f"bad_answer: {exc}") from None
            try:
                following = manager.answer(
                    decision_id,
                    recorded,
                    source=source,
                    request_version=request_version,
                    comment=comment,
                    channel=channel,
                    cancel_event=cancel_event,
                )
            except DecisionError as exc:  # lost a race with the run API
                raise ToolError(_refusal(exc, known)) from None
            self._given_total += 1
            if len(self._given) < mcp_decision.MAX_RECORDED:
                self._given.append(
                    mcp_decision.recorded_entry(
                        waiting, recorded, source, channel, comment
                    )
                )
            self._waiting = None
            self._answered = following
            self._cond.notify_all()
        return {
            "run_id": run_id,
            "outcome": "answered",
            "source": source,
            "answer": _echo(recorded),
        }


_ALREADY_ANSWERED = (
    "already_answered: this question was already answered; read lobster_status "
    "for the state of the run."
)
_CANCELLED = "rejected_cancelled: the run was stopped; start a new run."


def _echo(recorded: Any) -> Any:
    """The recorded answer as the reply shows it: whole when short."""
    return mcp_decision.bounded(recorded, ECHO_MAX_CHARS)


def _stale(version: int) -> str:
    return (
        f"rejected_stale: the question is now at request_version {version}; "
        "read lobster_status and answer that version."
    )


def _refusal(exc: Exception, decision: Any) -> str:
    code = getattr(exc, "code", "")
    if code == "already_answered":
        return _ALREADY_ANSWERED
    if code == "rejected_cancelled":
        return _CANCELLED
    if code == "rejected_stale" and decision is not None:
        return _stale(decision.request_version)
    return f"{code}: {_one_line(exc)}"


@dataclass
class EngineSession:
    """A completed run's session, kept for the next turn."""

    client: Any
    manager: Any
    output_dir: Path
    stem: str  # the notebook name base
    input_path: Optional[str]
    turns: int  # completed turns
    latest_run_id: str  # the run of the last completed turn
    claim: Optional[mcp_lock.FolderClaim] = None  # released when the session closes
    # Restored sessions only: modalities earlier turns made that the restore did
    # not bring back. The first turn after the restore says so, once.
    not_restored: Tuple[str, ...] = ()
    experts: Optional[Tuple[str, ...]] = None  # the run's, kept for every turn


class SessionSlot:
    """The one idle session this server keeps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Optional[EngineSession] = None

    def put(self, session: EngineSession) -> None:
        with self._lock:
            old, self._session = self._session, session
        if old is not None and old is not session:
            _close(old)

    def peek(self) -> Optional[EngineSession]:
        with self._lock:
            return self._session

    def take(self) -> Optional[EngineSession]:
        with self._lock:
            session, self._session = self._session, None
            return session

    def clear(self) -> None:
        """Close the idle session, if there is one: its folder is free again."""
        session = self.take()
        if session is not None:
            _close(session)


def _close(session: EngineSession) -> None:
    if session.claim is not None:
        session.claim.release()


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    input_path: Optional[str]
    output_dir: str
    question: Optional[str] = None  # the host's, already normalised; first runs only
    progress: Optional[RunProgress] = None
    channel: str = "mcp:unknown"  # who carries the run: ``mcp:<host name>``
    instruction: Optional[str] = None  # the host's, already normalised
    stem: str = mcp_record.RUN_STEM  # the notebook name base
    turn: int = 1  # 2 or later: a ``lobster_continue``
    continues: Optional[str] = None  # the run whose session this turn goes on with
    restore: bool = False  # the session is read from output_dir, not kept here
    session: Optional[EngineSession] = None  # the idle session, taken by the runner
    slot: Optional[SessionSlot] = None  # where a finished turn leaves its session
    # The claim on output_dir, made when the request was accepted. A session
    # that keeps the folder takes it over (``kept``); otherwise the run releases it.
    claim: Optional[mcp_lock.FolderClaim] = None
    experts: Optional[Tuple[str, ...]] = None  # None: the project's
    project_ws: Optional[Path] = None  # where provider, model and experts are set
    plan: bool = False  # a first turn loads its input, then asks to approve its plan


class Runner(Protocol):
    def start(self, request: RunRequest) -> None: ...

    def status(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    def cancel(self, run_id: str) -> Optional[str]:
        """Ask one run to stop: ``cancel_requested``, ``already_terminal``, or
        None for an unknown run."""
        ...

    def answer(
        self,
        run_id: str,
        decision_id: str,
        request_version: int,
        answer: Any,
        source: str,
        comment: Optional[str],
        channel: str,
    ) -> Optional[Dict[str, Any]]:
        """Answer the question a run waits on; None for an unknown run, a
        ToolError for a refusal."""
        ...

    def stop(self, timeout: float) -> bool:
        """Cancel live work; True when nothing is left running."""
        ...

    def idle(self) -> Optional[EngineSession]:
        """The session a completed run left for the next turn, if any."""
        ...


class FakeRunner:
    """Test double: reports ``running`` once, then writes two files and completes."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}

    def start(self, request: RunRequest) -> None:
        if request.claim is not None:  # a fake run keeps no session
            request.claim.release()
        self._runs[request.run_id] = {
            "output_dir": request.output_dir,
            "polls": 0,
            "started": _now(),
        }

    def status(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self._runs.get(run_id)
        if run is None:
            return None
        if "done" in run:
            return run["done"]
        run["polls"] += 1
        elapsed_s = int(_now() - run["started"])
        if run["polls"] < 2:
            return {
                "status": "running",
                "run_id": run_id,
                "elapsed_s": elapsed_s,
                "phase": "running",
                "step": "fake_step",
                "steps_done": 1,
                "poll_after_s": contract.POLL_AFTER_S,
            }
        out = Path(run["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        result_path = out / "result.json"
        notebook_path = out / "analysis.ipynb"
        result_path.write_text(json.dumps({"run_id": run_id, "fake": True}))
        notebook_path.write_text(
            json.dumps(
                {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
            )
        )
        run["done"] = {
            "status": "completed",
            "run_id": run_id,
            "elapsed_s": elapsed_s,
            "summary": "Fake run completed; no analysis was performed.",
            "notebook_path": str(notebook_path),
            "result_path": str(result_path),
        }
        return run["done"]

    def cancel(self, run_id: str) -> Optional[str]:
        run = self._runs.get(run_id)
        if run is None:
            return None
        if "done" in run:
            return "already_terminal"
        run["done"] = {
            "status": "cancelled",
            "run_id": run_id,
            "elapsed_s": int(_now() - run["started"]),
            "summary": "Stopped before the analysis finished.",
            "cancel_reason": "cancel",
        }
        return "cancel_requested"

    def answer(
        self, run_id, decision_id, request_version, answer, source, comment, channel
    ) -> Optional[Dict[str, Any]]:
        status = self.status(run_id)
        if status is None:
            return None
        raise ToolError(
            f"not_waiting: run {run_id} is {status['status']}; there is no "
            "question to answer."
        )

    def stop(self, timeout: float) -> bool:
        return True

    def idle(self) -> Optional[EngineSession]:
        return None


Work = Callable[[RunRequest, threading.Event], Dict[str, Any]]


class ThreadRunner:
    """Runs each request's work on its own thread; ``status`` is a snapshot.

    ``work`` returns the run's final status object. It is ``run_engine`` in
    production and an injected callable in tests.
    """

    def __init__(self, work: Optional[Work] = None):
        self._work = work or run_engine
        self._lock = threading.Lock()
        self._status: Dict[str, Dict[str, Any]] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._started: Dict[str, float] = {}
        self._progress: Dict[str, RunProgress] = {}
        # continues and turn of a later turn; where an earlier session folder went
        self._turn: Dict[str, Dict[str, Any]] = {}
        self.slot = SessionSlot()

    def idle(self) -> Optional[EngineSession]:
        return self.slot.peek()

    def start(self, request: RunRequest) -> None:
        cancel_event = threading.Event()
        progress = RunProgress()
        session = None
        if request.turn == 1:
            self.slot.clear()  # a new run closes the idle session
        elif request.restore:
            self.slot.clear()
        else:
            session = self.slot.take()  # back in the slot when the turn ends
        if request.turn == 1:
            progress.plan_requested(request.plan)
        request = replace(request, progress=progress, session=session, slot=self.slot)
        thread = threading.Thread(
            target=self._drive,
            args=(request, cancel_event),
            daemon=True,
            name=f"lobster-mcp-{request.run_id}",
        )
        with self._lock:
            self._status[request.run_id] = {
                "status": "running",
                "run_id": request.run_id,
            }
            self._cancel[request.run_id] = cancel_event
            self._threads[request.run_id] = thread
            self._started[request.run_id] = _now()
            self._progress[request.run_id] = progress
            if request.turn > 1:
                self._turn[request.run_id] = {"turn": request.turn}
                if request.continues:
                    self._turn[request.run_id]["continues"] = request.continues
            moved = _moved_to(request)
            if moved:
                self._turn.setdefault(request.run_id, {}).update(moved)
        thread.start()

    def _drive(self, request: RunRequest, cancel_event: threading.Event) -> None:
        try:
            final = self._work(request, cancel_event)
        except Exception as exc:  # never a traceback in a tool result
            raised = {"code": type(exc).__name__, "message": str(exc)}
            key = mcp_record.rejected_key(raised)
            # A provider's refusal of the key can echo part of it: name the class only.
            shown = type(exc).__name__ if key else repr(exc)
            print(f"lobster mcp: run {request.run_id} failed: {shown}", file=sys.stderr)
            final = _failed(
                {
                    "code": raised["code"] if key else "exception",
                    "message": _one_line(exc),
                    "retryable": False,
                }
            )
        finally:
            claim = request.claim
            if claim is not None and not claim.kept:  # no session keeps the folder
                claim.release()
        final["run_id"] = request.run_id
        plan = request.progress.plan() if request.progress is not None else None
        if plan is not None:
            final["plan"] = plan
        with self._lock:
            final.update(self._turn.get(request.run_id, {}))
            final["elapsed_s"] = int(_now() - self._started[request.run_id])
        final = _bounded(final)
        with self._lock:
            self._status[request.run_id] = final

    def status(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._status.get(run_id)
            if status is None:
                return None
            status = dict(status)
            started = self._started[run_id]
            progress = self._progress[run_id]
            later = self._turn.get(run_id, {})
            stopping = self._cancel[run_id].is_set()
        if status.get("status") in contract.TERMINAL_STATES:
            return status
        live = {
            **status,
            **later,
            "elapsed_s": int(_now() - started),
            **progress.snapshot(),
        }
        if stopping and live.get("phase") in ("starting", "running"):
            live["phase"] = "cancelling"
        plan = progress.plan()
        if plan is not None:
            live["plan"] = plan
        if "decision" in live:  # nothing changes until it is answered
            return {**live, "status": "needs_decision"}
        return {**live, "poll_after_s": contract.POLL_AFTER_S}

    def cancel(self, run_id: str) -> Optional[str]:
        with self._lock:
            status = self._status.get(run_id)
            if status is None:
                return None
            if status.get("status") in contract.TERMINAL_STATES:
                return "already_terminal"
            # Cooperative: the work sees the event after its current step.
            self._cancel[run_id].set()
            progress = self._progress[run_id]
            thread = self._threads[run_id]
        waiting = progress.is_waiting()
        progress.wake()
        if waiting:
            # A waiting worker ends at once: let the status show it.
            thread.join(2.0)
        return "cancel_requested"

    def answer(
        self, run_id, decision_id, request_version, answer, source, comment, channel
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if run_id not in self._status:
                return None
            cancel_event = self._cancel[run_id]
            progress = self._progress[run_id]
        state = self.status(run_id)["status"]
        return progress.submit(
            run_id,
            state,
            decision_id,
            request_version,
            answer,
            source,
            comment,
            channel,
            cancel_event,
        )

    def stop(self, timeout: float) -> bool:
        with self._lock:
            events = list(self._cancel.values())
            threads = list(self._threads.values())
            progresses = list(self._progress.values())
        for event in events:
            event.set()
        for progress in progresses:
            progress.wake()
        for thread in threads:
            thread.join(timeout)
        self.slot.clear()
        return not any(thread.is_alive() for thread in threads)


def _startup_error(
    diagnostic: Any, project_ws: Optional[Path], workspace: Optional[Path] = None
) -> Dict[str, Any]:
    message = diagnostic.title
    fixes = {
        "missing_credentials": mcp_overview.key_fix,
        "provider_not_configured": mcp_overview.provider_fix,
    }
    detail = " ".join(diagnostic.detail_lines)
    if detail and diagnostic.code not in fixes:  # those get the server's own fix
        message = f"{message}: {_one_line(detail)}"
    if diagnostic.code not in fixes and workspace is not None:
        # The engine reports some providers' missing key as a plain failure.
        message += _missing_key_fix(workspace, project_ws)
    if diagnostic.code in fixes:
        message = f"{message}. {fixes[diagnostic.code]()}"
        root = project_ws.parent if project_ws is not None else None
        if (
            diagnostic.code == "missing_credentials"
            and root is not None
            and Path.cwd().resolve() != root.resolve()
        ):
            message += (
                f" This server was started outside the project (in "
                f"{Path.cwd()}); start it in {root} to use the project's "
                "settings."
            )
    return {"code": diagnostic.code, "message": message, "retryable": False}


def _missing_key_fix(workspace: Path, project_ws: Optional[Path]) -> str:
    """The overview's two fixes, when the key of the provider the session
    resolves is found nowhere; else nothing."""
    try:
        provider = mcp_overview.run_settings(workspace)[0]
        key = mcp_overview.key_presence(provider, project_ws or workspace)
    except Exception:
        return ""
    if not key or key.get("present") is not False:
        return ""
    names = key["names"]
    found = "was" if len(names) == 1 else "were"
    return (
        f". No {' and '.join(names)} {found} found for {provider}. "
        f"{mcp_overview.key_fix(names)}"
    )


def _discard_unstarted(request: RunRequest) -> None:
    """A first turn that never started leaves no session folder; its claim made
    the folder in the same call (a run without a claim leaves it be)."""
    if request.claim is not None:
        request.claim.discard()


def run_engine(request: RunRequest, cancel_event: threading.Event) -> Dict[str, Any]:
    """Production work: one Lobster session per run, driven through the run API.

    The session's workspace is ``<output_dir>/.lobster_workspace``; nothing is
    written to the process cwd. A completed first run leaves its session in
    ``request.slot`` for ``lobster_continue``; otherwise the client is dropped
    when the run ends.
    """
    if cancel_event.is_set():
        if request.turn == 1:  # nothing began: leave the folder as it was
            _discard_unstarted(request)
        return _cancelled("cancel", "Stopped before the analysis started.")
    if request.turn > 1:
        return _continue_turn(request, cancel_event)

    from lobster.cli_internal.startup_diagnostics import StartupDiagnosticError
    from lobster.core.runs import runs_for

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # the claim made it; a fake claim not
    # The session lives in a hidden folder; output_dir gets the deliverables.
    workspace = output_dir / mcp_record.WORKSPACE_DIR
    try:
        # The reply is read after the run: "run" style, whatever the user's
        # LOBSTER_RESPONSE_STYLE says (it is only a default).
        team = _prepare_session(workspace, request.project_ws, request.experts)
        with response_style.forced("run"):
            client = _build_client(workspace, request.project_ws, request.experts)
    except StartupDiagnosticError as exc:
        # Read before the folder goes: the session's provider file is in it.
        error = _startup_error(exc.diagnostic, request.project_ws, workspace)
        # A run that never began must not hold its folder.
        _discard_unstarted(request)
        return _failed(error)
    if cancel_event.is_set():
        _discard_unstarted(request)
        return _cancelled(
            "cancel",
            "Stopped while the session was starting; the analysis did not start.",
        )

    prompt = mcp_record.compose_question_prompt(
        request.question or "", request.input_path, plan=request.plan
    )
    manager = runs_for(client)
    manager.report_files = True  # the status lists the files this run wrote
    data_manager = getattr(client, "data_manager", None)
    if not _record_question(data_manager, request):
        # The model would act on words the notebook does not show.
        _discard_unstarted(request)
        return _not_recorded("question")
    status = _run_turn(client, manager, prompt, request, cancel_event, output_dir)
    if team is not None:
        status["experts_loaded"] = team
    status.update(_run_settings(workspace))
    if status["status"] == "completed" and request.slot is not None:
        session = EngineSession(
            client,
            manager,
            output_dir,
            request.stem,
            request.input_path,
            1,
            request.run_id,
            request.claim,
            experts=request.experts,
        )
        if request.claim is not None:
            request.claim.kept = True
        request.slot.put(session)
    return status


def _not_recorded(what: str = "instruction") -> Dict[str, Any]:
    return _failed(
        {
            "code": f"{what}_not_recorded",
            "message": f"The {what} could not be recorded, so the run did not start.",
            "retryable": False,
        }
    )


def _continue_turn(
    request: RunRequest, cancel_event: threading.Event
) -> Dict[str, Any]:
    """Another turn in a kept session, or in one restored from ``output_dir``.

    The session goes back to the slot however the turn ends, so a cancelled or
    failed turn leaves it at its last completed turn. The session folder is
    never removed here: it holds earlier work.
    """
    output_dir = Path(request.output_dir)
    session = request.session
    try:
        if session is None:
            session = _restore_session(request, output_dir)
            if isinstance(session, dict):  # a failed status
                return session
        if cancel_event.is_set():
            return _cancelled("cancel", "Stopped before the turn started.")
        client, manager = session.client, session.manager
        manager.report_files = True
        if not _record_instruction(getattr(client, "data_manager", None), request):
            return _not_recorded()
        prompt = mcp_record.compose_continue_prompt(
            request.instruction, session.not_restored
        )
        try:
            status = _run_turn(
                client,
                manager,
                prompt,
                request,
                cancel_event,
                output_dir,
                session.not_restored,
            )
        except BaseException:
            _record_outcome(client, "failed")
            raise
        team = _team_of(output_dir / mcp_record.WORKSPACE_DIR)
        if team is not None:
            status["experts_loaded"] = team
        status.update(_run_settings(output_dir / mcp_record.WORKSPACE_DIR))
        session.not_restored = ()  # the model was told once, whatever became of the turn
        if status["status"] == "completed":
            session.turns, session.latest_run_id = request.turn, request.run_id
        elif status["status"] in mcp_record.UNFINISHED_OUTCOMES:
            _record_outcome(client, status["status"], status.get("steps_completed", ()))
        return status
    finally:
        if isinstance(session, EngineSession) and request.slot is not None:
            if mcp_turns.blocked_reason(output_dir / mcp_record.WORKSPACE_DIR):
                _close(session)  # it holds data that must not reach a result
            else:
                request.slot.put(session)


def _restore_session(request: RunRequest, output_dir: Path):
    """The session of an earlier server process, opened through the engine's own
    ``--session-id`` path (``resolve_session_continuation`` and
    ``client.load_session``); a failed status when it cannot be."""
    from lobster.cli_internal.commands.heavy.session_infra import (
        resolve_session_continuation,
    )
    from lobster.cli_internal.startup_diagnostics import StartupDiagnosticError
    from lobster.core.runs import runs_for

    workspace = output_dir / mcp_record.WORKSPACE_DIR
    session_file, session_id, found = resolve_session_continuation(workspace, "latest")
    if not found:
        return _failed(
            {
                "code": "no_session_found",
                "message": f"No saved session in {workspace}.",
                "retryable": False,
            }
        )
    try:
        _prepare_session(workspace, request.project_ws, request.experts)
        with response_style.forced("run"):
            client = _build_client(
                workspace, request.project_ws, request.experts, session_id=session_id
            )
    except StartupDiagnosticError as exc:
        return _failed(_startup_error(exc.diagnostic, request.project_ws, workspace))
    if session_file is not None:
        loaded = client.load_session(session_file)
        if not loaded.get("success"):
            return _failed(
                {
                    "code": "session_not_restored",
                    "message": _one_line(loaded.get("error") or "load_session failed"),
                    "retryable": False,
                }
            )
    if request.claim is not None:
        request.claim.kept = True
    return EngineSession(
        client,
        runs_for(client),
        output_dir,
        request.stem,
        request.input_path,
        request.turn - 1,
        request.continues or "",
        request.claim,
        _not_restored(client, output_dir),
        request.experts,
    )


def _not_restored(client: Any, output_dir: Path) -> Tuple[str, ...]:
    """The modalities earlier turns made that the restored session lacks."""
    try:
        gone = mcp_record.not_restored(
            mcp_record.last_known_modalities(output_dir),
            client.data_manager.list_modalities(),
        )
    except Exception as exc:  # the notice is extra: the turn still runs
        print(
            f"lobster mcp: restored modalities not compared: {exc!r}", file=sys.stderr
        )
        return ()
    return tuple(gone)


class _Stop:
    """What ``_stop_on_cancel`` saw: whether it stopped the turn, and the
    conversation before the turn (the client's list and the graph thread's
    messages; None when it could not be read)."""

    def __init__(self, messages_before: Optional[int], thread_before: Any) -> None:
        self.messages_before = messages_before
        self.thread_before = thread_before
        self.tripped = False


def _thread_config(client: Any) -> Dict[str, Any]:
    return {"configurable": {"thread_id": client.session_id}}


def _thread_messages(client: Any) -> Optional[List[Any]]:
    try:
        state = client.graph.get_state(_thread_config(client))
        return list(state.values.get("messages") or [])
    except Exception:
        return None


@contextmanager
def _plan_first(request: RunRequest) -> Iterator[None]:
    """In a first turn with plan, refuse every tool but loading and
    describing the input until the plan is answered (or the turn ends)."""
    if not (request.plan and request.turn == 1 and request.progress is not None):
        yield
        return
    from lobster.agents import graph

    graph.hold_for_plan(request.progress.plan_refused)
    try:
        yield
    finally:
        graph.open_plan()


@contextmanager
def _stop_on_cancel(client: Any, cancel_event: threading.Event) -> Iterator[_Stop]:
    """For the turn, a callback on *client* that raises at the next tool or
    model call once *cancel_event* is set. The client checks the event only
    between the supervisor's stream events, and an expert's whole work is one
    of those: without this a cancel waited for the expert to finish."""
    callbacks = getattr(client, "callbacks", None)
    messages = getattr(client, "messages", None)
    stop = _Stop(
        len(messages) if isinstance(messages, list) else None,
        _thread_messages(client),
    )
    if not isinstance(callbacks, list):
        yield stop
        return
    from langchain_core.callbacks import BaseCallbackHandler

    class StopOnCancel(BaseCallbackHandler):
        raise_error = True  # the exception ends the graph run
        run_inline = True

        def _check(self, *_args: Any, **_kwargs: Any) -> None:
            if cancel_event.is_set():
                stop.tripped = True
                raise RunCancelled("cancelled through lobster_cancel")

        on_tool_start = _check
        on_chat_model_start = _check
        on_llm_start = _check

    from lobster.utils import download_activity  # heavy package: not at start

    handler = StopOnCancel()
    callbacks.append(handler)
    download_activity.set_stop(cancel_event)  # a file download ends at once
    try:
        yield stop
    finally:
        download_activity.set_stop(None)
        callbacks.remove(handler)


class RunCancelled(Exception):
    """Raised inside the engine's graph to end a cancelled turn."""


def _forget_cancelled_turn(client: Any, stop: _Stop) -> None:
    """Put the conversation back as it was before a cancelled turn: the graph
    thread can hold a tool call without its result, which the next turn
    refuses. The client's own rollback cuts the thread at the length of its
    message list, which holds no tool calls or results, so after a first turn
    it cuts into that turn. The work the turn did stays in the session."""
    if stop.thread_before is None:
        return
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    try:
        if stop.messages_before is not None:
            client.messages = client.messages[: stop.messages_before]
        # The forgotten question may have carried the session context.
        if hasattr(client, "_session_context_hash"):
            client._session_context_hash = None
        client.graph.update_state(
            _thread_config(client),
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *stop.thread_before]},
        )
        save = getattr(client, "_save_session_json", None)
        if save is not None:
            save()
    except Exception as exc:  # the next turn then says what is wrong
        print(
            f"lobster mcp: conversation not put back after a cancel: {exc!r}",
            file=sys.stderr,
        )


def _run_turn(
    client: Any,
    manager: Any,
    prompt: str,
    request: RunRequest,
    cancel_event: threading.Event,
    output_dir: Path,
    not_restored: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Start the turn on the run API, follow it through its questions, and
    report how it ended. A turn that stopped or failed says which steps had
    completed: their results stay in the session."""
    data_manager = getattr(client, "data_manager", None)
    turn = replace(
        mcp_record.Turn.begin(data_manager, request.turn),
        not_restored=not_restored,
    )
    _note_run_status(client, "running")
    try:
        with _stop_on_cancel(client, cancel_event) as stop, _plan_first(request):
            status = _follow_turn(
                client, manager, prompt, request, cancel_event, output_dir, turn
            )
    except BaseException as exc:
        _record_turn_end(client, output_dir, request.turn, "failed")
        _note_run_status(
            client,
            "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            type(exc).__name__,
        )
        raise
    # The session's catalog run ends as its latest turn did.
    _note_run_status(client, status["status"], _end_reason(status))
    if status["status"] == "cancelled":
        _forget_cancelled_turn(client, stop)
    if status["status"] != "completed":  # a completed turn noted itself in _completed
        _record_turn_end(client, output_dir, request.turn, status["status"])
    if status["status"] in mcp_record.UNFINISHED_OUTCOMES:
        steps, total = mcp_record.completed_steps(data_manager, turn)
        if steps:
            status["steps_completed"] = steps
            if total > len(steps):
                status["steps_completed_total"] = total
        if status["status"] == "cancelled":
            status["summary"] = mcp_record.stopped_summary(
                status["summary"], total, first_turn=request.turn == 1
            )
    return status


_RUN_STATUS = {
    "running": None,  # no end status while the turn runs
    "completed": "completed",
    "cancelled": "aborted",
    "failed": "errored",
}


def _note_run_status(client: Any, outcome: str, reason: Optional[str] = None) -> None:
    from lobster.core.catalog.factory import note_run_status

    if outcome in _RUN_STATUS:
        session_id = getattr(client, "session_id", None)
        note_run_status(client, _RUN_STATUS[outcome], session_id, reason)


def _end_reason(status: Dict[str, Any]) -> Optional[str]:
    """Why a turn did not complete: its error code, or why it was cancelled."""
    error = status.get("error")
    if isinstance(error, dict) and error.get("code"):
        return str(error["code"])
    return status.get("cancel_reason") or None


def _follow_turn(
    client: Any,
    manager: Any,
    prompt: str,
    request: RunRequest,
    cancel_event: threading.Event,
    output_dir: Path,
    turn: mcp_record.Turn,
) -> Dict[str, Any]:
    from lobster.agents.graph import PLAN_LOADER, open_plan

    data_manager = getattr(client, "data_manager", None)
    run = manager.start(prompt, cancel_event=cancel_event, run_id=request.run_id)
    if request.progress is not None:
        request.progress.begin(_step_probe(client, turn), manager)
    last = _drain(run)
    if (
        request.plan
        and request.turn == 1
        and request.progress is not None
        and run.state == "needs_decision"
        # The plan comes first: no expert but the one that loads and describes
        # the input has started (its first message streams as agent_change)
        # before the turn's first question.
        and not any(
            e.get("type") == "agent_change" and e.get("agent") != PLAN_LOADER
            for e in run.events
        )
    ):
        request.progress.plan_asked()
    while run.state == "needs_decision":
        pending = [
            d
            for d in manager.decisions.values()
            if d.origin_run_id == run.run_id and d.state == "pending"
        ]
        titles = [_one_line(d.summary()["question"] or "a question") for d in pending]
        if len(pending) != 1 or request.progress is None:
            # One answer per call: a batch of questions cannot be answered here.
            manager.cancel(run.run_id, reason="decision_unsupported")
            return _cancelled(
                run.events[-1].get("reason"),
                "Stopped: the analysis asked for more than one decision at "
                f"once ({'; '.join(titles)}). This server answers one question "
                "at a time.",
                decision={"title": titles[0] if titles else ""},
            )
        following = request.progress.wait_for_answer(
            run, mcp_decision.view(last["decision"]), cancel_event
        )
        if following is None:
            return _cancelled(
                run.events[-1].get("reason"),
                f"Stopped while waiting for an answer to: {titles[0]}",
            )
        if request.plan and request.turn == 1:
            open_plan()  # the first answer is the plan's
        run = following
        last = _drain(run)
    if run.state == "completed" and run.result is not None:
        refusal = mcp_record.input_refusal(
            data_manager,
            turn,
            request.input_path,
            output_dir / mcp_record.WORKSPACE_DIR,
        )
        if refusal is not None:
            failed = _failed(refusal)
            said = str(run.result.get("response") or "").strip()
            if said:
                # What the model said is commentary on a run that analysed nothing.
                failed["notes"] = mcp_record.notes_tail(said)
                failed["notes_chars"] = len(said)
            return failed
        return _completed(client, run.result, request, output_dir, turn)
    if run.state == "cancelled":
        return _cancelled(last.get("reason"), "Stopped before the analysis finished.")
    return _failed(
        last.get("error")
        or {
            "code": last.get("reason") or "failed",
            "message": "The run ended without a result.",
            "retryable": False,
        }
    )


def _record_instruction(data_manager: Any, request: RunRequest) -> bool:
    try:
        return mcp_record.record_instruction(
            data_manager, request.instruction, request.channel
        )
    except Exception as exc:  # no data manager, or a tracker that refuses
        print(f"lobster mcp: instruction not recorded: {exc!r}", file=sys.stderr)
        return False


def _record_question(data_manager: Any, request: RunRequest) -> bool:
    try:
        return mcp_record.record_question(
            data_manager, request.question, request.channel
        )
    except Exception as exc:  # no data manager, or a tracker that refuses
        print(f"lobster mcp: question not recorded: {exc!r}", file=sys.stderr)
        return False


def _record_turn_end(client: Any, output_dir: Path, number: int, outcome: str) -> None:
    """Note how the turn ended in the session folder; see ``mcp_turns``."""
    mcp_record.record_turn_end(
        getattr(client, "data_manager", None),
        output_dir / mcp_record.WORKSPACE_DIR,
        number,
        outcome,
    )


def _record_outcome(client: Any, outcome: str, steps: Sequence[str] = ()) -> None:
    """Note in the session that the last instruction's turn did not finish, so
    the next turn can say so. It runs before the session goes back to the slot."""
    try:
        mcp_record.record_outcome(client.data_manager, outcome, steps)
    except Exception as exc:  # the turn's own ending is what the host needs
        print(f"lobster mcp: outcome not recorded: {exc!r}", file=sys.stderr)


def _completed(
    client: Any,
    result: Dict[str, Any],
    request: RunRequest,
    output_dir: Path,
    turn: mcp_record.Turn,
) -> Dict[str, Any]:
    """The status of a finished run; ``result.json`` holds the whole record."""
    data_manager = getattr(client, "data_manager", None)
    reply = mcp_record.reply_of(client, result.get("response"))
    summary = reply.text
    result = {**result, "response": summary}
    published = mcp_record.publish(
        output_dir,
        mcp_record.deliverables(
            result.get("files_written"),
            _export_notebook(client, request.stem, request.turn),
            request.turn,
            output_dir,
            skip=[n for n, _s in mcp_record.file_loads(data_manager, turn) or []],
        ),
    )
    views = mcp_record.publish_plot_views(data_manager, turn, output_dir)
    published = [*published, *views.published]
    notebook = next((p for p in published if p["category"] == "notebook"), None)
    open_question = _open_question(summary)  # judged on the whole text, before any cut
    sections = mcp_record.collect(
        client,
        turn,
        result,
        exported=notebook is not None,
        open_question=open_question,
        final_answer=reply.final,
        data_files={
            # Files of earlier turns stay where the host was told they are; in a
            # first run a file already in the folder is the user's, not ours.
            **(
                {p.stem: str(p) for p in sorted(output_dir.glob("*.h5ad"))}
                if request.turn > 1
                else {}
            ),
            **{
                Path(p["source"]).stem: p["path"]
                for p in published
                if p["category"] == "data"
            },
        },
        plots=views,
    )
    unfinished = _unfinished(client)
    model_source = _model_source()
    given, given_total = (
        request.progress.answers_given() if request.progress else ([], 0)
    )
    record = mcp_record.result_record(
        result,
        sections,
        mcp_record.request_record(
            request.stem,
            request.input_path,
            request.instruction,
            request.channel,
            request.turn,
            unfinished,
            model_source,
            turn.not_restored,
            question=request.question,
            experts=request.experts,
            plan=request.plan if request.turn == 1 else None,
        ),
        request.continues,
        notes=reply.notes,
        notes_chars=reply.notes_chars,
    )
    record["engine"] = {
        **record.get("engine", {}),
        **_run_settings(output_dir / mcp_record.WORKSPACE_DIR),
    }
    loaded, loaded_total = mcp_record.loaded_files(data_manager, turn)
    if loaded:
        record["loaded_files"] = loaded
        if loaded_total > len(loaded):
            record["loaded_files_total"] = loaded_total
    record.update(_moved_to(request))
    plan = request.progress.plan() if request.progress is not None else None
    if plan is not None:
        record["plan"] = plan
    answers: Dict[str, Any] = {}
    if given:
        answers["decisions"] = given
        if given_total > len(given):
            answers["decisions_total"] = given_total
    record.update(answers)
    if "files_written" in record:
        record["files_written"] = mcp_record.written_files(
            record["files_written"], output_dir, published
        )
    result_path = output_dir / mcp_record.result_name(request.turn)
    result_path.write_text(json.dumps(record, indent=2, default=str))
    # Only once its files are out: a turn that could not publish them failed.
    # From here on, a folder whose result file goes missing later still shows
    # that a turn of its session finished.
    _record_turn_end(client, output_dir, request.turn, "completed")
    status: Dict[str, Any] = {
        "status": "completed",
        "summary": summary,
        "result_path": str(result_path),
        "notebook_path": notebook["path"] if notebook else None,
    }
    if open_question:
        status["open_question"] = True
    if reply.notes:  # the text itself is in result.json
        status["notes_chars"] = reply.notes_chars
    status.update(
        _files_status(
            [
                {
                    "path": str(result_path),
                    "category": "result",
                    "size_bytes": result_path.stat().st_size,
                },
                *published,
            ]
        )
    )
    status.update(mcp_record.status_fields(sections))
    for key in ("loaded_files", "loaded_files_total"):
        if key in record:
            status[key] = record[key]
    if unfinished[0]:
        status["unfinished_instructions"] = unfinished[0]
    if answers:
        status.update(answers, decisions_note=mcp_decision.RECORDED_NOTE)
    if model_source:
        status["model_source"] = model_source
    # The summary and the file list travel once; result.json keeps them.
    left_out = {"response", "files_written"}
    status["result"] = {k: v for k, v in result.items() if k not in left_out}
    return status


def _run_settings(workspace: Path) -> Dict[str, str]:
    """``provider`` and ``model`` as the engine resolves them for the session
    in *workspace*: every turn of a session resolves from the provider file
    its first run was given. Empty for a replayed model, and a value that
    cannot be read is left out."""
    if _model_source():
        return {}
    try:
        provider, _source, model, _model_from = mcp_overview.run_settings(workspace)
    except Exception:
        return {}
    return {k: v for k, v in (("provider", provider), ("model", model)) if v}


def _model_source() -> Optional[str]:
    """``replay`` when the engine's model factory hands out a scripted replay
    model (it does so whenever the variable is set), else None."""
    from lobster.config.providers.replay_provider import REPLAY_ENV

    return "replay" if os.environ.get(REPLAY_ENV) else None


def _unfinished(client: Any):
    try:
        return mcp_record.unfinished_instructions(client.data_manager)
    except Exception as exc:  # the list is extra: never fail a completed turn
        print(f"lobster mcp: unfinished instructions unread: {exc!r}", file=sys.stderr)
        return [], 0


_QUESTION_TAIL = " \t\r\n*_`~\"'\u201c\u201d\u2018\u2019)]>"


def _open_question(summary: str) -> bool:
    """True when the last paragraph of *summary* ends with a question mark
    (markdown emphasis, quotes and whitespace after it do not count)."""
    paragraphs = [p for p in re.split(r"\n\s*\n", summary) if p.strip()]
    return bool(paragraphs) and paragraphs[-1].rstrip(_QUESTION_TAIL).endswith("?")


def _files_status(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``files`` (path, category, size_bytes, and format and title for a plot
    view; at most ``MAX_FILES``) of a completed status: the deliverables in
    ``output_dir``."""
    shown = [
        {
            **{k: f.get(k) for k in ("path", "category", "size_bytes")},
            **{k: f[k] for k in ("format", "title") if k in f},
        }
        for f in files[:MAX_FILES]
    ]
    status: Dict[str, Any] = {"files": shown}
    if len(files) > MAX_FILES:
        status.update(files_truncated=True, files_total=len(files))
    return status


def _drain(run: Any) -> Dict[str, Any]:
    """Stream *run* to its end or its pause; the last ``run_state`` event."""
    last: Dict[str, Any] = {}
    try:
        for event in run.stream():
            if event.get("type") == "run_state":
                last = event
    except Exception:
        # The run API recorded the failure before it re-raised; report that.
        last = run.events[-1] if run.events else {}
        if last.get("type") != "run_state":
            raise
    return last


def _step_probe(client: Any, turn: mcp_record.Turn) -> Callable[[], Dict[str, Any]]:
    """Step fields from the session's provenance activities recorded since *turn*
    began.

    The tracker appends one activity per recorded analysis step; no engine API
    is involved. A reused workspace reloads earlier activities, hence the
    baseline.
    """
    baseline = turn.baseline

    def probe() -> Dict[str, Any]:
        if baseline is None:  # no tracker: the probe reports nothing
            raise LookupError("session has no provenance tracker")
        # The recorded decision (ask_user) is the host's answer, not a step.
        activities = [
            a
            for a in list(client.data_manager.provenance.activities)[baseline:]
            if a["type"] != "ask_user"
        ]
        fields: Dict[str, Any] = {"steps_done": len(activities)}
        if activities:
            fields["step"] = _one_line(activities[-1]["type"], limit=80)
        return fields

    return probe


def _bounded(status: Dict[str, Any]) -> Dict[str, Any]:
    """Cut ``summary`` to ``SUMMARY_MAX_CHARS``. While the status text is too
    long: when the rest fits with ``SUMMARY_MIN_CHARS`` of summary, cut the
    summary to the room the rest leaves; otherwise drop ``result``, then
    ``steps``, then the
    objects of ``modality_details`` (the names stay), then the
    ``steps_completed`` of ``unfinished_instructions`` (the instructions stay),
    then ``decisions`` (``decisions_total`` stays), and cut the summary to the
    room left only after that (never below ``SUMMARY_MIN_CHARS`` characters of
    JSON text). ``metrics``, ``warnings``,
    ``files``, ``run_id`` and the paths always stay. Lengths are of the JSON
    text, where a character outside ASCII takes up to twelve.

    All of it stays whole in ``result.json``. Slicing a str cuts between code
    points, so an astral character is never split into a lone surrogate.
    """
    status = _cut_fields(status)
    summary = status.get("summary")
    if isinstance(summary, str) and len(summary) > SUMMARY_MAX_CHARS:
        keep = SUMMARY_MAX_CHARS - len(SUMMARY_CUT)
        status = {**status, "summary": summary[:keep] + SUMMARY_CUT}
    rest = len(json.dumps({**status, "summary": ""}, default=str))
    if rest < STATUS_MAX_CHARS - 1 - SUMMARY_MIN_CHARS:
        status = _fit_summary(status)  # the fields stay whole
    for key in ("result", "steps"):
        if key in status and len(json.dumps(status, default=str)) >= STATUS_MAX_CHARS:
            status = {k: v for k, v in status.items() if k != key}
            status[f"{key}_omitted"] = True
    if (
        isinstance(status.get("modality_details"), list)
        and len(json.dumps(status, default=str)) >= STATUS_MAX_CHARS
    ):
        status = {
            **status,
            "modality_details": [
                {"name": m.get("name")} for m in status["modality_details"]
            ],
            "modality_details_names_only": True,
        }
    if (
        isinstance(status.get("unfinished_instructions"), list)
        and len(json.dumps(status, default=str)) >= STATUS_MAX_CHARS
    ):
        status = {
            **status,
            "unfinished_instructions": [
                {k: v for k, v in e.items() if k != "steps_completed"}
                for e in status["unfinished_instructions"]
            ],
            "unfinished_steps_omitted": True,
        }
    if (
        isinstance(status.get("decisions"), list)
        and len(json.dumps(status, default=str)) >= STATUS_MAX_CHARS
    ):
        total = status.get("decisions_total", len(status["decisions"]))
        status = {k: v for k, v in status.items() if k != "decisions"}
        status.update(decisions_omitted=True, decisions_total=total)
    return _fit_summary(status)


def _fit_summary(status: Dict[str, Any]) -> Dict[str, Any]:
    """Cut ``summary`` to the room the rest of a too-long status leaves, not
    below ``SUMMARY_MIN_CHARS`` characters of JSON text."""
    summary = status.get("summary")
    if (
        not isinstance(summary, str)
        or len(json.dumps(status, default=str)) < STATUS_MAX_CHARS
    ):
        return status
    rest = len(json.dumps({**status, "summary": ""}, default=str))
    room = max(STATUS_MAX_CHARS - 1 - rest, SUMMARY_MIN_CHARS)
    return {**status, "summary": _cut_summary(summary, room)}


def _cut_summary(summary: str, room: int) -> str:
    """*summary* whose JSON text (without its quotes) is at most *room*
    characters, cut with ``SUMMARY_CUT`` when it is longer."""

    def fits(text: str) -> bool:
        return len(json.dumps(text)) - 2 <= room

    if fits(summary):
        return summary
    if summary.endswith(SUMMARY_CUT):
        summary = summary[: -len(SUMMARY_CUT)]
    low, high = 0, len(summary)  # the longest prefix that fits
    while low < high:
        mid = (low + high + 1) // 2
        if fits(summary[:mid] + SUMMARY_CUT):
            low = mid
        else:
            high = mid - 1
    return summary[:low] + SUMMARY_CUT


def _cut_fields(value: Any) -> Any:
    """*value* with every text (and every key) of at most ``FIELD_MAX_CHARS``
    characters, except the values in ``_WHOLE_FIELDS``."""
    if isinstance(value, str):
        return mcp_decision.cut(value, FIELD_MAX_CHARS)
    if isinstance(value, list):
        return [_cut_fields(v) for v in value]
    if isinstance(value, dict):
        return {
            (mcp_decision.cut(k, FIELD_MAX_CHARS) if isinstance(k, str) else k): (
                v if k in _WHOLE_FIELDS else _cut_fields(v)
            )
            for k, v in value.items()
        }
    return value


def _export_notebook(client: Any, stem: str, turn: int = 1) -> Optional[str]:
    """Notebook from the recorded steps (in the session folder), or None; a
    failed export never fails a run. Each turn's has its own name."""
    try:
        path = client.data_manager.export_notebook(
            mcp_record.notebook_stem(stem, turn), filter_strategy="successful"
        )
        return str(Path(path).resolve())
    except Exception as exc:
        print(f"lobster mcp: no notebook: {_one_line(exc)}", file=sys.stderr)
        return None


def _provider_name() -> Optional[str]:
    """The provider the engine is configured to use, or None when that cannot
    be read."""
    try:
        from lobster.config.llm_factory import LLMFactory

        return LLMFactory.get_current_provider()
    except Exception:
        return None


def _failed(error: Dict[str, Any]) -> Dict[str, Any]:
    """A failed status. The message is one line of at most ``FIELD_MAX_CHARS``
    characters; a longer one is written whole to stderr, never to the reply."""
    if mcp_record.rejected_key(error):
        # The provider's own text can echo part of the key, so it is not kept.
        error = {
            "code": "provider_key_rejected",
            "message": (
                f"The {_provider_name() or 'configured'} provider rejected the "
                "API key it was given. Check that the key is valid for it, "
                "then start a new run."
            ),
            "retryable": False,
        }
    message = " ".join(str(error.get("message")).split())
    if len(message) > FIELD_MAX_CHARS:
        print(
            f"lobster mcp: error message cut in the reply: {message}", file=sys.stderr
        )
    error = {
        **error,
        "code": _one_line(error.get("code"), ERROR_CODE_MAX_CHARS),
        "message": _one_line(message, FIELD_MAX_CHARS),
    }
    return {
        "status": "failed",
        "summary": _one_line(f"{error['code']}: {error['message']}"),
        "error": error,
    }


def _cancelled(reason: Optional[str], summary: str, **extra: Any) -> Dict[str, Any]:
    # cancel_reason: the engine's closed list (runs.CANCEL_REASONS: cancel,
    # decision_abandoned, decision_unsupported), passed verbatim.
    return {"status": "cancelled", "summary": summary, "cancel_reason": reason, **extra}


def _moved_to(held: Any) -> Dict[str, str]:
    """``previous_workspace_moved_to`` for a run whose claim moved an earlier
    session folder aside (*held*: the claim, or a request carrying one)."""
    claim = getattr(held, "claim", held)
    moved = getattr(claim, "moved_to", None)
    return {"previous_workspace_moved_to": str(moved)} if moved else {}


def _one_line(value: Any, limit: int = 300) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class _Wait:
    """One open lobster_wait. ``ended`` stops it early; ``cancelled`` (the host
    withdrew the request, or its input ended) means no reply is sent."""

    run_id: str
    key: str
    ended: threading.Event
    cancelled: bool = False


class _NoReply(Exception):
    pass


class McpServer:
    """JSON-RPC dispatch for one host connection.

    With *send*, slow tools reply from a thread of their own through it; without,
    every reply is ``handle_line``'s return value.
    """

    def __init__(
        self,
        runner: Runner,
        send: Optional[Callable[[Dict[str, Any]], None]] = None,
        start: Optional[Path] = None,
        follow_project: bool = False,
    ):
        self.runner = runner
        self._send = send
        # The process serving a host enters a project that appears later.
        self._follow_project = follow_project
        # Where the project is looked for: the folder the server was started in.
        self._start_folder = Path(start) if start is not None else Path.cwd()
        self._wait_lock = threading.Lock()
        self._waits: Dict[str, _Wait] = {}  # run_id -> the open wait on it
        self._watch = mcp_watch.WaitWatch()  # stacks to stderr for a late wait
        self._waits_by_request: Dict[str, _Wait] = {}
        self._active: Optional[str] = None
        self._out: Dict[str, Path] = {}  # run_id -> resolved output_dir
        self._channel = "mcp:unknown"  # who carries answers, from initialize
        self._methods: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "initialize": self._initialize,
            "ping": lambda params: {},
            "tools/list": lambda params: {"tools": contract.list_tools()},
            "tools/call": self._call_tool,
        }
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "lobster_overview": lambda args: mcp_overview.overview(self._start_folder),
            "lobster_run": self._lobster_run,
            "lobster_status": self._lobster_status,
            "lobster_wait": self._lobster_wait,
            "lobster_answer": self._lobster_answer,
            "lobster_cancel": self._lobster_cancel,
            "lobster_continue": self._lobster_continue,
            "lobster_inspect": self._lobster_inspect,
            "lobster_docs": self._lobster_docs,
        }

    def handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        """The reply to one inbound line, or None when none is due."""
        try:
            message = json.loads(line)
        except (ValueError, RecursionError):
            # ValueError covers a JSONDecodeError and an integer of more than
            # 4300 digits; RecursionError a deeply nested array.
            return _error(None, PARSE_ERROR, "Parse error")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Invalid Request")
        method = message.get("method")
        if "id" not in message or not isinstance(method, str):
            # Notifications need no answer; this server sends no requests, so
            # an inbound response has nothing to match.
            if method == "notifications/cancelled":
                self._cancel_wait(message.get("params"))
            return None
        request_id = message["id"]
        handler = self._methods.get(method)
        if handler is None:
            return _error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS, "params must be an object")
        if (
            self._send is not None
            and method == "tools/call"
            and params.get("name") in _SLOW_TOOLS
        ):
            threading.Thread(
                target=self._reply_later,
                args=(request_id, method, params),
                daemon=True,
                name="lobster-mcp-reply",
            ).start()
            return None
        return self._reply(request_id, method, params)

    def _reply_later(self, request_id: Any, method: str, params: Dict[str, Any]):
        if params.get("name") != "lobster_wait":
            return self._send_reply(request_id, method, params)
        late_after_s = _late_after_s(params)
        if late_after_s is None:
            return self._send_reply(request_id, method, params)
        with self._watch.watching(late_after_s):
            self._send_reply(request_id, method, params)

    def _send_reply(self, request_id: Any, method: str, params: Dict[str, Any]):
        reply = self._reply(request_id, method, params)
        if reply is None:
            return
        try:
            self._send(reply)
        except (OSError, ValueError) as exc:  # the host's pipe is gone
            print(f"lobster mcp: reply not sent: {exc!r}", file=sys.stderr)

    def _reply(
        self, request_id: Any, method: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            if method == "tools/call":
                result = self._call_tool(params, request_id)
            else:
                result = self._methods[method](params)
        except _NoReply:
            return None
        except _InvalidParams as exc:
            return _error(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # report, never a traceback on stdout
            print(f"lobster mcp: {method} failed: {exc!r}", file=sys.stderr)
            return _error(request_id, INTERNAL_ERROR, type(exc).__name__)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from lobster.version import __version__

        info = params.get("clientInfo")
        name = info.get("name") if isinstance(info, dict) else None
        self._channel = "mcp:" + _host_token(name)
        return {
            "protocolVersion": contract.negotiate_version(
                params.get("protocolVersion")
            ),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": contract.SERVER_NAME, "version": __version__},
            "instructions": mcp_overview.start_text(self._enter_project()),
        }

    def _enter_project(self) -> Path:
        """The start folder, after making the root of a project that now
        contains it the working folder (see enter_start_folder). Never while a
        run is active, and never out of a project root."""
        if self._follow_project and self._active_status() is None:
            from lobster.core.runtime.workspace import find_project_root

            here = Path.cwd()
            root = find_project_root(self._start_folder)
            if root is not None and find_project_root(here) != here:
                os.chdir(root)
        return self._start_folder

    def _call_tool(
        self, params: Dict[str, Any], request_id: Any = None
    ) -> Dict[str, Any]:
        name = params.get("name")
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise _InvalidParams(f"Unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _InvalidParams("arguments must be an object")
        if name not in _SLOW_TOOLS:  # those reply from threads of their own
            self._enter_project()
        try:
            if name == "lobster_wait":
                payload = self._lobster_wait(arguments, request_id)
            else:
                payload = tool(arguments)
        except ToolError as exc:
            text = _one_line(exc, limit=1000)
            return {"content": [{"type": "text", "text": text}], "isError": True}
        if isinstance(payload, dict):  # first, so a host's preview shows it
            payload = {"headline": mcp_headline.headline(name, payload), **payload}
        try:
            text = json.dumps(payload, default=str, allow_nan=False)
        except ValueError:  # NaN or Infinity is not JSON
            text = "internal: the result held a number that is not valid JSON."
            return {"content": [{"type": "text", "text": text}], "isError": True}
        return {"content": [{"type": "text", "text": text}]}

    def _lobster_run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = mcp_record.normalise_question(args.get("question"))
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        output_dir = args.get("output_dir")
        if output_dir is not None:
            output_dir = _string_arg(args, "output_dir")
        input_path = args.get("input_path")
        if input_path is not None:
            input_path = _string_arg(args, "input_path")
        plan = args.get("plan", True)
        if not isinstance(plan, bool):
            raise ToolError("bad_argument: plan must be true or false.")
        if output_dir is None:
            output_dir = str(self._new_run_folder())
        _check_paths(input_path, output_dir)
        linked = Path(output_dir).parent
        if linked.is_symlink() and mcp_record.holds_project(linked.parent):
            raise ToolError(
                f"path_not_allowed: {linked} is a link; a run goes in a real "
                "runs folder of the project."
            )
        resolved = _resolve(output_dir, "output_dir")
        project = _project_of(resolved)
        if project is not None:
            _check_in_runs(output_dir, resolved, project)
        experts = _check_experts(args.get("experts"), project)
        if experts is None and project is not None:
            from lobster.core.projects import project_settings

            _refuse_not_installed(project_settings(project).experts, project)
        _refuse_without_loader(input_path, experts, project)
        self._refuse_if_busy()
        from lobster.core.runs import new_run_id

        run_id = new_run_id()
        existed = os.path.lexists(resolved)
        # The folder is claimed here, before the reply, not by the worker later:
        # a second process that asks next must already see it taken.
        claim = _claim(
            lambda: mcp_lock.claim_new(
                resolved, lambda: mcp_record.unfinished_leftover(resolved)
            ),
            output_dir,
        )
        if project is not None and not existed:
            _record_run_folder(project, resolved)
        self._start(
            RunRequest(
                run_id,
                input_path,
                output_dir,
                question,
                channel=self._channel,
                claim=claim,
                experts=experts,
                project_ws=project.workspace if project is not None else None,
                plan=plan,
            )
        )
        self._active = run_id
        self._out[run_id] = resolved
        started: Dict[str, Any] = {
            "status": "running",
            "run_id": run_id,
            "output_dir": output_dir,
            "question_recorded": True,
            **_moved_to(claim),
        }
        if experts is not None:
            started["experts"] = list(experts)
        if project is not None:
            note = mcp_overview.settings_note(project)
            if note:
                started["project_settings_not_applied"] = note
        return started

    def _new_run_folder(self) -> Path:
        """A folder that does not exist yet in the runs folder of the project
        that contains the server's start folder."""
        from lobster.core.projects import runs_dir

        project, broken = mcp_overview.find_project_or_reason(self._start_folder)
        if project is None:
            raise ToolError(
                "bad_argument: output_dir is needed. "
                + (broken or mcp_overview.no_project_text(self._start_folder))
            )
        if runs_dir(project).is_symlink():
            raise ToolError(
                f"path_not_allowed: {runs_dir(project)} is a link; a run goes in "
                "a real runs folder of the project. Pass output_dir."
            )
        stamp = mcp_lock._stamp()
        folder, n = runs_dir(project) / stamp, 2
        while os.path.lexists(folder):
            folder, n = runs_dir(project) / f"{stamp}-{n}", n + 1
        return folder

    def _start(self, request: RunRequest) -> None:
        try:
            self.runner.start(request)
        except BaseException:
            if request.claim is not None:
                request.claim.release()
            raise

    def _active_status(self) -> Optional[Dict[str, Any]]:
        """The status of the run that has not ended, if there is one."""
        if self._active is None:
            return None
        active = self.runner.status(self._active)
        if not active or active.get("status") in contract.TERMINAL_STATES:
            return None
        return active

    def _refuse_if_busy(self) -> None:
        active = self._active_status()
        if active is None:
            return
        if active["status"] == "needs_decision":
            now = (
                "is waiting for an answer; answer it with lobster_answer "
                "or stop it with lobster_cancel"
            )
        else:
            now = (
                "is still running; wait with lobster_wait until it is "
                "completed, failed or cancelled"
            )
        raise ToolError(
            f"engine_busy: run {self._active} {now}, then start the next run."
        )

    def _lobster_continue(self, args: Dict[str, Any]) -> Dict[str, Any]:
        run_id, folder = args.get("run_id"), args.get("output_dir")
        if (run_id is None) == (folder is None):
            raise ToolError(
                "exactly_one_of: pass run_id or output_dir, "
                + ("not both." if run_id is not None else "you passed neither.")
            )
        key = "run_id" if run_id is not None else "output_dir"
        target = _string_arg(args, key)
        instruction = args.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolError(
                "instruction_required: say in one sentence what to do next "
                "(500 characters at most)."
            )
        try:
            instruction = mcp_record.normalise_instruction(instruction)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        if folder is not None and not Path(target).is_absolute():
            raise ToolError("output_dir must be an absolute path")
        self._refuse_if_busy()
        idle = self.runner.idle()
        restore = False
        if run_id is not None:
            output_dir = self._continuable_dir(target, idle)
        else:
            output_dir = _resolve(target, "output_dir")
            if idle is None or idle.output_dir.resolve() != output_dir:
                restore = True
        claim = None
        if restore:
            claim = _claim(
                lambda: mcp_lock.claim_existing(output_dir),
                str(output_dir),
                missing_ok=True,
                hint="close it or wait for it to finish, then call lobster_continue again",
            )
            try:
                disk = mcp_record.read_disk_session(output_dir)
            except LookupError as exc:
                if claim is not None:
                    claim.release()
                raise ToolError(
                    f"no_session_found: {output_dir} cannot be continued: {exc}. "
                    "Start a new run with lobster_run in a new output_dir."
                ) from None
            stem, input_path, experts = disk.stem, disk.input_path, disk.experts
            turn, continues = disk.turn, disk.continues
        else:
            stem, input_path, experts = idle.stem, idle.input_path, idle.experts
            turn, continues = idle.turns + 1, idle.latest_run_id
        from lobster.core.runs import new_run_id

        new_id = new_run_id()
        self._start(
            RunRequest(
                new_id,
                input_path,
                str(output_dir),
                channel=self._channel,
                instruction=instruction,
                stem=stem,
                turn=turn,
                continues=continues,
                restore=restore,
                claim=claim,
                experts=experts,
                project_ws=_project_ws(output_dir),
            )
        )
        self._active = new_id
        self._out[new_id] = output_dir
        started: Dict[str, Any] = {"status": "running", "run_id": new_id}
        if continues:
            started["continues"] = continues
        started.update(turn=turn, output_dir=str(output_dir), instruction_recorded=True)
        return started

    def _lobster_docs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from lobster.cli_internal import mcp_docs

        query, slug = args.get("query"), args.get("slug")
        given = [
            name
            for name, value in (("query", query), ("slug", slug))
            if value is not None
        ]
        if len(given) != 1 or not isinstance(args[given[0]], str):
            raise ToolError("exactly_one_of: pass query (to search) or slug (to read).")
        text = args[given[0]].strip()
        if not text:
            raise ToolError(f"exactly_one_of: {given[0]} must not be empty.")
        wrong = "page" if query is not None else "limit"
        if wrong in args:
            raise ToolError(
                f"bad_argument: {wrong} goes with "
                f"{'slug' if wrong == 'page' else 'query'}."
            )

        def bounded(name: str, default: int, top: Optional[int]) -> int:
            value = args.get(name, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or (top is not None and value > top)
            ):
                most = f" to {top}" if top is not None else " or more"
                raise ToolError(f"bad_argument: {name} must be a whole number 1{most}.")
            return value

        try:
            if query is not None:
                if len(text) > contract.DOCS_QUERY_MAX:
                    raise ToolError(
                        f"bad_argument: query has at most {contract.DOCS_QUERY_MAX} "
                        "characters."
                    )
                return mcp_docs.search(text, bounded("limit", 5, mcp_docs.MAX_RESULTS))
            return mcp_docs.read(text, bounded("page", 1, None))
        except mcp_docs.DocsError as exc:
            raise ToolError(f"docs_unavailable: {_one_line(exc)}") from None
        except ValueError as exc:  # a page past the end
            raise ToolError(f"bad_argument: {_one_line(exc)}") from None

    def _lobster_inspect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        raw = _string_arg(args, "path")
        column = args.get("column")
        if column is not None and (not isinstance(column, str) or not column):
            raise ToolError("column must be a non-empty string such as obs:<name>")
        items = args.get("max_items", mcp_inspect.DEFAULT_ITEMS)
        if (
            not isinstance(items, int)
            or isinstance(items, bool)
            or not 1 <= items <= mcp_inspect.MAX_ITEMS
        ):
            raise ToolError(
                f"max_items must be an integer from 1 to {mcp_inspect.MAX_ITEMS}"
            )
        if not Path(raw).is_absolute():
            raise ToolError("path must be an absolute path")
        path = _resolve(raw, "path")  # a symlink is judged by where it leads
        # The top of an output_dir (not its subfolders: it may be a whole home
        # folder) and that folder's own session folder.
        if not any(
            path.parent == root or path.is_relative_to(root / mcp_record.WORKSPACE_DIR)
            for root in self._out.values()
        ):
            raise ToolError(
                "path_not_allowed: lobster_inspect reads files directly inside an "
                "output_dir that this server ran or continued, and in that "
                "folder's session folder."
            )
        busy = self._out.get(self._active) if self._active_status() else None
        if busy is not None and path.is_relative_to(busy / mcp_record.WORKSPACE_DIR):
            raise ToolError(
                f"engine_busy: run {self._active} is still writing in that "
                "session folder; inspect the files in the top level of output_dir "
                "or wait until the run ends."
            )
        try:
            return mcp_inspect.inspect(path, column, items)
        except mcp_inspect.InspectRefused as exc:
            raise ToolError(str(exc)) from None

    def _continuable_dir(self, run_id: str, idle: Optional[EngineSession]) -> Path:
        """The folder of *run_id*'s session, or the reason it cannot go on."""
        folder = self._out.get(run_id)
        status = self.runner.status(run_id) if folder is not None else None
        if status is None:
            raise ToolError(
                f"unknown_run: this server has no run {run_id!r}. After a restart "
                "pass the output_dir of the earlier run instead."
            )
        blocked = mcp_turns.blocked_reason(folder / mcp_record.WORKSPACE_DIR)
        if blocked:
            raise ToolError(
                f"not_continuable: {blocked}. Start a new run with lobster_run "
                "in a new output_dir."
            )
        if status.get("status") != "completed":
            raise ToolError(
                f"not_continuable: run {run_id} is {status.get('status')}; only "
                "a completed run can be continued."
            )
        if idle is None or idle.output_dir.resolve() != folder:
            raise ToolError(
                f"session_closed: the session of run {run_id} is no longer kept "
                "(a newer run closed it). Pass its output_dir to restore it "
                "from disk."
            )
        if idle.latest_run_id != run_id:
            raise ToolError(
                f"not_latest: run {run_id} is not the latest turn of its "
                f"session; continue run {idle.latest_run_id}."
            )
        return folder

    def _lobster_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        run_id = _string_arg(args, "run_id")
        status = self.runner.status(run_id)
        if status is None:
            raise ToolError(f"Unknown run_id {run_id!r}")
        return status

    def _lobster_wait(self, args: Dict[str, Any], request_id: Any = None):
        """The run's status once it settles, or after ``timeout_s`` if given."""
        run_id = _string_arg(args, "run_id")
        timeout = args.get("timeout_s")
        if timeout is not None and not _good_timeout(timeout):
            raise ToolError(
                f"bad_argument: timeout_s must be a number from 0 to "
                f"{contract.WAIT_MAX_S}."
            )
        status = self.runner.status(run_id)
        if status is None:
            raise ToolError(f"Unknown run_id {run_id!r}")
        if _settled(status):
            return status
        wait = self._open_wait(run_id, request_id)
        try:
            deadline = None if timeout is None else time.monotonic() + timeout
            # shortcut: the status is re-read every WAIT_STEP_S rather than the
            # worker waking the wait, so the end shows up to WAIT_STEP_S late.
            while not wait.ended.wait(
                WAIT_STEP_S
                if deadline is None
                else max(0.0, min(WAIT_STEP_S, deadline - time.monotonic()))
            ):
                status = self.runner.status(run_id)
                if _settled(status):
                    return status
                if deadline is not None and time.monotonic() >= deadline:
                    return _still(run_id, status)
            if wait.cancelled:
                raise _NoReply()
            status = self.runner.status(run_id)  # a later wait took its place
            return status if _settled(status) else _still(run_id, status, True)
        finally:
            self._close_wait(wait)

    def _open_wait(self, run_id: str, request_id: Any) -> _Wait:
        wait = _Wait(run_id, _request_key(request_id), threading.Event())
        with self._wait_lock:
            earlier = self._waits.get(run_id)
            if earlier is not None:  # one wait per run: the earlier one returns
                earlier.ended.set()
            self._waits[run_id] = wait
            self._waits_by_request[wait.key] = wait
        return wait

    def _close_wait(self, wait: _Wait) -> None:
        with self._wait_lock:
            if self._waits.get(wait.run_id) is wait:
                del self._waits[wait.run_id]
            if self._waits_by_request.get(wait.key) is wait:
                del self._waits_by_request[wait.key]

    def _cancel_wait(self, params: Any) -> None:
        """notifications/cancelled: the host withdrew a request. Only a wait
        can still be open; it ends without a reply."""
        if not isinstance(params, dict) or "requestId" not in params:
            return
        with self._wait_lock:
            wait = self._waits_by_request.get(_request_key(params["requestId"]))
            if wait is not None:
                wait.cancelled = True
                wait.ended.set()

    def close(self) -> None:
        """The host is gone: every open wait ends without a reply."""
        with self._wait_lock:
            for wait in self._waits.values():
                wait.cancelled = True
                wait.ended.set()

    def _lobster_answer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        run_id = _string_arg(args, "run_id")
        decision_id = args.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            raise ToolError("bad_answer: decision_id is required and must be a string.")
        version = args.get("request_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ToolError(
                "bad_answer: request_version is required and must be an integer."
            )
        answer = args.get("answer")
        if not isinstance(answer, (str, int, float, bool, dict)):
            raise ToolError(
                "bad_answer: answer is required (a string, number, boolean or object)."
            )
        _check_answer_size(answer)
        source = args.get("source")
        if source not in contract.ANSWER_SOURCES:  # a str, never filled in here
            raise ToolError(
                "bad_answer: source is required and must be 'human' or 'model'."
            )
        comment = args.get("comment")
        if comment is not None and not isinstance(comment, str):
            raise ToolError("bad_answer: comment must be a string.")
        if comment is not None and len(comment) > ANSWER_MAX_CHARS:
            raise ToolError(_too_long("comment"))
        outcome = self.runner.answer(
            run_id, decision_id, version, answer, source, comment, self._channel
        )
        if outcome is None:
            raise ToolError(f"Unknown run_id {run_id!r}")
        return outcome

    def _lobster_cancel(self, args: Dict[str, Any]) -> Dict[str, Any]:
        run_id = _string_arg(args, "run_id")
        outcome = self.runner.cancel(run_id)
        if outcome is None:
            raise ToolError(f"Unknown run_id {run_id!r}")
        return {"run_id": run_id, "outcome": outcome}


class _InvalidParams(Exception):
    pass


def _settled(status: Dict[str, Any]) -> bool:
    """A status the host must act on: a wait returns it at once."""
    state = status.get("status")
    return state in contract.TERMINAL_STATES or state == "needs_decision"


def _still(
    run_id: str, status: Dict[str, Any], superseded: bool = False
) -> Dict[str, Any]:
    """The short reply of a wait that ended with nothing for the host to do."""
    short = {"run_id": run_id}
    short.update(
        (k, status[k])
        for k in ("status", "step", "steps_done", "download", "elapsed_s")
        if k in status
    )
    short["superseded" if superseded else "timed_out"] = True
    return short


def _request_key(request_id: Any) -> str:
    # JSON-RPC ids are strings or numbers: 1 and "1" are different requests.
    return json.dumps(request_id)


def _too_long(field: str, limit: int = ANSWER_MAX_CHARS) -> str:
    return f"bad_answer: {field} is longer than {limit} characters."


def _check_answer_size(answer: Any) -> None:
    """Refuse an answer that is too long or not plain JSON (NaN, Infinity)."""
    if isinstance(answer, str):
        if len(answer) > ANSWER_MAX_CHARS:
            raise ToolError(_too_long("answer"))
        return
    try:
        size = len(json.dumps(answer, allow_nan=False))
    except (ValueError, OverflowError):
        raise ToolError(
            "bad_answer: answer must be plain JSON without NaN or Infinity."
        ) from None
    if size > 5 * ANSWER_MAX_CHARS:  # an object may carry one entry per cluster
        raise ToolError(_too_long("answer", 5 * ANSWER_MAX_CHARS))


def _host_token(name: Any) -> str:
    """The host's name as one short printable token (it ends up in a notebook)."""
    if not isinstance(name, str):
        return "unknown"
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")[:40].strip("-")
    return token or "unknown"


def _build_client(
    workspace: Path,
    project_ws: Optional[Path],
    experts: Optional[Tuple[str, ...]],
    **kwargs: Any,
) -> Any:
    """The engine client of one run, its session in *workspace*, which
    _prepare_session has given the run's provider file and expert list. A run
    in a project uses the project's keys, wherever the server was started."""
    from lobster.cli_internal.commands.heavy.session_infra import (
        init_client_or_raise_startup_diagnostic,
    )

    return init_client_or_raise_startup_diagnostic(
        workspace=workspace,
        interactive=INTERACTIVE,
        credentials_from=project_ws,
        **kwargs,
    )


def _prepare_session(
    workspace: Path, project_ws: Optional[Path], experts: Optional[Tuple[str, ...]]
) -> Optional[List[str]]:
    """Give the run's session folder the project's provider file and the
    run's expert list; returns every expert that loads when a list applies."""
    if project_ws is not None:
        _copy_project_provider(project_ws, workspace)
    return _write_expert_list(workspace, project_ws, experts)


def _refusal(code: str, title: str) -> Exception:
    from lobster.cli_internal.startup_diagnostics import (
        StartupDiagnostic,
        StartupDiagnosticError,
    )

    return StartupDiagnosticError(StartupDiagnostic(code=code, title=title))


def _write_expert_list(
    workspace: Path, project_ws: Optional[Path], experts: Optional[Tuple[str, ...]]
) -> Optional[List[str]]:
    """The project's config.toml in the run's session folder, its expert list
    replaced by *experts* when given; the engine builds the team from it.
    Returns every expert that loads, or None when no list applies (every
    installed expert) or the folder already has one (a continued run keeps
    its first run's). A file that does not read back as written, or a listed
    expert the license tier removes, stops the run before it starts."""
    from lobster.config.workspace_agent_config import (
        CONFIG_FILE_NAME,
        WorkspaceAgentConfig,
    )

    target = workspace / CONFIG_FILE_NAME
    if target.exists():
        return None
    config = WorkspaceAgentConfig()
    source = project_ws / CONFIG_FILE_NAME if project_ws is not None else None
    if source is not None and source.exists():
        try:
            config = mcp_overview.load_agent_config(source)
        except Exception as exc:  # tomllib, pydantic or OSError
            raise _refusal(
                "project_settings_unreadable",
                f"{source} cannot be read ({_one_line(exc, limit=200)}); the run "
                "did not start. Ask the user to repair the file.",
            ) from None
    names = list(experts) if experts is not None else list(config.enabled_agents)
    if not names and config.preset:
        from lobster.config.agent_presets import expand_preset

        names = list(expand_preset(config.preset) or [])
    if not names and config == WorkspaceAgentConfig():
        return None
    config = config.model_copy(update={"enabled_agents": names, "preset": None})
    loaded = None
    if names:
        loaded, removed = _expert_team(names)
        if removed:
            raise _refusal(
                "expert_not_in_tier",
                f"The license tier of this install does not include "
                f"{', '.join(removed)}; the run did not start. Leave "
                f"{'it' if len(removed) == 1 else 'them'} out of the list.",
            )
    try:
        config.save(workspace)
        written = mcp_overview.load_agent_config(target)
    except Exception:
        written = None
    # only what the server set: an ignored table may not survive the round trip
    if written is None or (written.enabled_agents, written.preset) != (names, None):
        raise _refusal(
            "expert_list_not_applied",
            f"The expert list could not be written to {target} as meant; the "
            "run did not start rather than run with every expert.",
        )
    return loaded


def _team_of(workspace: Path) -> Optional[List[str]]:
    """The experts loaded for the list or preset in *workspace*'s config.toml,
    or None without one; the engine's registry is loaded by then."""
    try:
        names = mcp_overview.load_agent_config(workspace / "config.toml")
    except Exception:  # no file, or one the first turn did not write
        return None
    experts = names.effective_agents()
    if not experts:
        return None
    return _expert_team(experts)[0]


def _expert_team(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(the experts that load for *names*: they and their child experts, as
    the engine's graph adds them and its license tier filter keeps them; the
    names that filter removes). Loads the engine's expert registry."""
    from lobster.config.subscription_tiers import is_agent_available
    from lobster.core.component_registry import component_registry

    try:
        from lobster.core.license_manager import get_current_tier

        tier = get_current_tier()
    except ImportError:  # as agents.graph does
        tier = "free"
    agents = component_registry.list_agents()
    loaded = {n for n in names if n in agents}
    for name in list(loaded):
        loaded.update(c for c in agents[name].child_agents or () if c in agents)
    removed = [n for n in names if not is_agent_available(n, tier)]
    return sorted(n for n in loaded if is_agent_available(n, tier)), removed


def _copy_project_provider(project_ws: Path, workspace: Path) -> None:
    """The project's provider file in the run's session folder, which the
    engine reads before the global file and LOBSTER_LLM_PROVIDER. It holds
    names, never a key. One already there (a continued run) is kept."""
    from lobster.config.workspace_config import CONFIG_FILE_NAME as NAME
    from lobster.config.workspace_config import WorkspaceProviderConfig

    source = project_ws / NAME
    if WorkspaceProviderConfig.exists(workspace) or not source.exists():
        return
    try:  # not WorkspaceProviderConfig.load: it turns an error into "no provider"
        config = mcp_overview.load_provider_config(source)
    except Exception as exc:  # OSError, JSON or pydantic
        raise _refusal(
            "project_settings_not_copied",
            f"{source} cannot be read ({_one_line(exc, limit=200)}); the run "
            "did not start rather than use another provider. Ask the user to "
            "repair the file.",
        ) from None
    if not config.global_provider:
        return
    try:
        config.save(workspace)
    except Exception as exc:  # OSError, or the file's own serialisation
        said = getattr(exc, "strerror", None) or _one_line(exc, limit=200)
        raise _refusal(
            "project_settings_not_copied",
            "The project's provider settings could not be copied into "
            f"{workspace}: {said}",
        ) from None


def _check_experts(raw: Any, project: Any = None) -> Optional[Tuple[str, ...]]:
    """*raw* as the run's expert names, or None; a list that is not one, a
    name this server cannot load or one *project* lacks is a tool error that
    names the fix."""
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not 1 <= len(raw) <= contract.EXPERTS_MAX
        or not all(
            isinstance(n, str) and mcp_record.EXPERT_NAME.fullmatch(n) for n in raw
        )
        or len(set(raw)) != len(raw)
    ):
        raise ToolError(
            f"bad_argument: experts must be a list of 1 to {contract.EXPERTS_MAX} "
            "different expert names (letters, digits and _); lobster_overview "
            "lists them."
        )
    _refuse_not_installed(raw, project)
    loaded = mcp_overview.loaded_experts()
    late = [n for n in raw if loaded is not None and n not in loaded]
    if late:
        raise ToolError(
            f"restart_needed: {mcp_overview.restart_text(late)} Then call "
            "lobster_run again."
        )
    _check_project_experts(tuple(raw), project)
    return tuple(raw)


def _refuse_not_installed(names: Sequence[str], project: Any) -> None:
    """A tool error naming each of *names* that is not installed, with the
    line that installs it; the engine would leave it out without a word."""
    missing = [n for n in names if n not in mcp_overview.installed_experts()]
    if not missing:
        return
    where = f"in {project.root} " if project is not None else ""
    fixes = []
    for name in missing:
        package = mcp_overview.expert_package(name)
        fixes.append(
            f"{name}: {where}run {mcp_overview.install_line(package)}"
            if package
            else f"{name}: no package is known to provide it; "
            "check the name with lobster_overview or "
            f"{mcp_overview.lobster_command('agents', 'list')}"
        )
    raise ToolError(
        "agent_not_available: these experts are not installed. "
        + "; ".join(fixes)
        + ". Then call lobster_run again."
    )


LOADER = "data_expert_agent"


def _refuse_without_loader(
    input_path: Optional[str], experts: Optional[Tuple[str, ...]], project: Any
) -> None:
    """A tool error when a run given a file has a team without the expert that
    loads files: the run's list, else the project's (with none, every expert
    loads). Without it the run would end in input_not_loaded after spending."""
    if input_path is None:
        return
    names: Sequence[str] = ()
    if experts is not None:
        names = experts
    elif project is not None:
        from lobster.core.projects import project_settings

        names = project_settings(project).experts
    if not names or LOADER in names:
        return
    said = (
        f"data_expert_needed: the file at {input_path} would not be loaded: "
        f"the team ({', '.join(names)}) lacks {LOADER}, the expert that loads "
        "data files."
    )
    if LOADER not in mcp_overview.installed_experts():
        package = mcp_overview.expert_package(LOADER)
        if package is None:
            raise ToolError(
                f"{said} {LOADER} is not installed and no package is known to "
                "provide it; check with "
                f"{mcp_overview.lobster_command('agents', 'list')}."
            )
        raise ToolError(
            f"{said} {LOADER} is not installed: run this line in a shell: "
            f"{mcp_overview.install_line(package)}, restart the Lobster MCP "
            "server (reconnect it in your host), tell the user what you ran, "
            "then call lobster_run again."
        )
    if experts is not None:
        raise ToolError(f"{said} Add {LOADER} to experts, then call lobster_run again.")
    again = mcp_overview.lobster_command(
        "init", str(project.root), "--agents", ",".join([*names, LOADER])
    )
    raise ToolError(
        f"{said} The project {project.name} lists its experts without it. To "
        f"add it, run this line in a shell: {again} (the project keeps its id "
        "and its runs), tell the user what you ran, then call lobster_run again."
    )


def _project_of(output_dir: Path) -> Any:
    """The Lobster project that contains *output_dir*, or None."""
    from lobster.core.projects import enclosing_project

    return enclosing_project(output_dir)


def _check_in_runs(output_dir: str, resolved: Path, project: Any) -> None:
    """In a project a run folder is one level into its runs folder: what
    deleting the project removes, and never inside its hidden workspace."""
    from lobster.core.projects import runs_dir

    runs = runs_dir(project)
    if runs.is_symlink() or resolved.parent != runs.resolve():
        raise ToolError(
            f"path_not_allowed: {output_dir} is inside the Lobster project "
            f"{project.name}; a run there goes one level into its runs folder: "
            f"{runs / '<name>'}."
        )


def _project_ws(output_dir: Path) -> Optional[Path]:
    project = _project_of(output_dir)
    return project.workspace if project is not None else None


def _check_project_experts(experts: Optional[Tuple[str, ...]], project: Any) -> None:
    """A run in a project may name only the experts the project has (all
    installed ones when it lists none)."""
    if experts is None or project is None:
        return
    from lobster.core.projects import project_settings

    have = project_settings(project).experts
    lacking = [n for n in experts if have and n not in have]
    if lacking:
        again = mcp_overview.lobster_command(
            "init", str(project.root), "--agents", ",".join(list(have) + lacking)
        )
        raise ToolError(
            f"agent_not_available: the project {project.name} does not list "
            f"{', '.join(lacking)}. To add them, run this line in a shell: "
            f"{again} (the project keeps its id and its runs), tell the user "
            "what you ran, then call lobster_run again."
        )


def _record_run_folder(project: Any, folder: Path) -> None:
    """Note *folder*, which this run made, in the project so that deleting the
    project removes it; a failure is said on stderr and the run goes on."""
    from lobster.core.projects import record_created_path

    try:
        record_created_path(project, folder)
    except Exception as exc:  # the run's files are not at risk
        print(
            f"lobster mcp: {folder} was not recorded in the project: {exc}",
            file=sys.stderr,
        )


def _check_paths(input_path: Optional[str], output_dir: str) -> None:
    target = Path(output_dir)
    if not target.is_absolute() or (
        input_path is not None and not Path(input_path).is_absolute()
    ):
        raise ToolError("input_path and output_dir must be absolute paths")
    _resolve(output_dir, "output_dir")
    if target.exists() and not target.is_dir():
        raise ToolError(f"output_dir is not a directory: {output_dir}")
    if input_path is not None:
        _check_input(input_path)
    if mcp_record.output_dir_in_use(target):
        raise ToolError(_in_use(output_dir, target, mcp_lock.holder(target)))


def _check_input(input_path: str) -> None:
    source = Path(input_path)
    _resolve(input_path, "input_path")
    if not source.exists():
        raise ToolError(f"input_path does not exist: {input_path}")
    if not os.access(source, os.R_OK):
        raise ToolError(f"input_path is not readable: {input_path}")
    if not (source.is_file() or source.is_dir()):  # a pipe or a device never ends
        raise ToolError(f"input_path is not a regular file or a folder: {input_path}")


def _in_use(output_dir: str, target: Path, pid: Optional[int], hint: str = "") -> str:
    """The refusal of a folder that holds a run (``hint``: what to do instead)."""
    if pid is not None:
        who = f"another Lobster process (pid {pid}) is using {output_dir}"
        return f"output_dir_in_use: {who}; {hint or 'choose a new folder'}."
    if hint:
        return f"output_dir_in_use: {output_dir} is in use; {hint}."
    if mcp_record.holds_project(target):
        return mcp_record.project_refusal(output_dir)
    if (target / "result.json").exists():
        return (
            f"output_dir_in_use: {output_dir} already holds a Lobster run; choose a "
            f"new folder. {mcp_record.CONTINUE_HINT}"
        )
    if mcp_record.finished_turn_recorded(target):
        return (
            f"output_dir_in_use: {output_dir} holds the session of a Lobster run "
            "that finished, but its result file is not there; choose a new folder."
        )
    workspace = target / mcp_record.WORKSPACE_DIR
    if workspace.is_symlink() or (
        workspace.is_dir() and not mcp_turns.is_marked(workspace)
    ):
        return (
            f"output_dir_in_use: {output_dir} holds a Lobster workspace "
            f"({mcp_record.WORKSPACE_DIR}) that this server did not create, such as "
            "one of lobster chat or lobster query; nothing in it was changed; "
            "choose a new folder."
        )
    return (
        f"output_dir_in_use: {output_dir} holds a Lobster session with no result "
        "to continue from that cannot be cleared for a new run; choose a new "
        "folder."
    )


def _claim(take: Callable[[], Any], output_dir: str, *, missing_ok=False, hint=""):
    """The claim *take* makes, or a tool error saying who holds the folder (with
    ``missing_ok`` a folder without a session folder gives None)."""
    try:
        return take()
    except mcp_lock.InUse as exc:
        raise ToolError(_in_use(output_dir, Path(output_dir), exc.pid, hint)) from None
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ToolError(
            f"output_dir_not_writable: {output_dir} cannot be made."
        ) from None
    except OSError as exc:
        raise ToolError(
            f"output_dir_not_writable: {output_dir} cannot be used ({_one_line(exc, 200)})."
        ) from None


def _resolve(raw: str, field: str) -> Path:
    """*raw* as a resolved path. A name the operating system cannot resolve (an
    embedded NUL, a symlink loop) is a tool error, not a protocol error."""
    try:
        return Path(raw).resolve()
    except (OSError, ValueError, RuntimeError):
        raise ToolError(
            f"path_not_allowed: {field} is not a path this server can use"
        ) from None


def _string_arg(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"{key} is required and must be a non-empty string")
    return value


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def enter_start_folder(workspace: Optional[Path]) -> Path:
    """Make the root of the project that contains *workspace* (default: the
    current folder) the working folder, else *workspace* itself; returns the
    resolved start folder. Called once, as the server starts.

    A run of this build takes its provider file from the working folder (a run
    in a project reads that project's .env files itself), so this makes
    ``--workspace`` and a start in a subfolder the same as a start in the
    project folder. Values read later relative to the
    first working folder are made absolute first."""
    from lobster.core.runtime.workspace import find_project_root

    if sys.argv and sys.argv[0] and not os.path.isabs(sys.argv[0]):
        sys.argv[0] = os.path.abspath(sys.argv[0])  # the overview's lobster path
    # shortcut: the global settings folder (XDG_CONFIG_HOME) is fixed when
    # lobster.config is imported, before this runs; a relative value, invalid
    # by the XDG spec, is not made absolute and goes wrong after the move.
    value = os.environ.get("LOBSTER_SSL_CERT_PATH")
    if value and not os.path.isabs(value):
        os.environ["LOBSTER_SSL_CERT_PATH"] = os.path.abspath(value)
    start = (workspace if workspace is not None else Path.cwd()).resolve()
    os.chdir(find_project_root(start) or start)
    return start


def _good_timeout(timeout: Any) -> bool:
    return (
        not isinstance(timeout, bool)
        and isinstance(timeout, (int, float))
        and 0 <= timeout <= contract.WAIT_MAX_S
    )


def _late_after_s(params: Dict[str, Any]) -> Optional[float]:
    """How long a lobster_wait with *params* may take before it is late; None
    for a wait without a timer, which is never late. A bad timer is refused at
    once, so it is late after the grace alone."""
    arguments = params.get("arguments")
    timeout = arguments.get("timeout_s") if isinstance(arguments, dict) else None
    if timeout is None:
        return None
    return (timeout if _good_timeout(timeout) else 0) + mcp_watch.GRACE_S


def run_mcp_stdio(
    *,
    runner: Optional[Runner] = None,
    stdin: Optional[IO[str]] = None,
    stdout: Optional[IO[str]] = None,
    start: Optional[Path] = None,
    follow_project: bool = False,
) -> int:
    """Serve one host over stdio until EOF; returns the exit code. *start* is
    the folder the project is found from (default: the current folder);
    *follow_project* lets the server enter a project made later."""
    if stdout is None:
        from lobster.cli_internal.agent_stdio import _claim_stdout

        stdout = _claim_stdout()
    out = stdout
    mcp_watch.stacks_on_signal()
    from lobster.core.catalog.factory import abort_runs_on_sigterm

    # Runs begin on worker threads, where no handler can be set: set it here.
    abort_runs_on_sigterm()
    if stdin is None:
        # MCP stdio is UTF-8; Codex passes no LANG, so a Windows locale
        # default would misread it. Untested on Windows.
        sys.stdin.reconfigure(encoding="utf-8")
        stdin = sys.stdin
    write_lock = threading.Lock()  # replies come from the reader and reply threads

    def send(reply: Dict[str, Any]) -> None:
        text = json.dumps(reply, separators=(",", ":")) + "\n"
        with write_lock:
            out.write(text)
            out.flush()

    server = McpServer(
        runner or ThreadRunner(), send=send, start=start, follow_project=follow_project
    )
    for line in stdin:
        if not line.strip():
            continue
        reply = server.handle_line(line)
        if reply is not None:
            send(reply)
    # EOF: the host is gone. End the open waits, cancel the active run and
    # wait briefly; a run dies with this process.
    server.close()
    if not server.runner.stop(SHUTDOWN_WAIT_S):
        # shortcut: a worker inside a model or tool call cannot be interrupted
        # and engine threads would hold interpreter exit; leave without them.
        out.flush()
        os._exit(0)
    return 0
