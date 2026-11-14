"""Canonical run API: one Lobster execution contract for every front end.

A *session* is the stable logical identity (the ``AgentClient`` and its
``session_id``). A *run* is one attempt inside it: ``start`` creates a run for
a prompt, and ``answer`` resumes a paused run as a **new** run. Every event a
run produces carries ``run_id`` and a per-run ``seq`` and is kept in a replay
buffer, so a consumer that reconnects calls ``events(run_id, after_seq)``.

Event vocabulary is the client stream (``AgentClient._stream_query``) plus:

* ``run_state``          ``{state, ...}``; states ``running``,
  ``needs_decision`` (pause), ``completed``, ``failed``, ``cancelled``.
  A finished run ends with its terminal ``run_state`` (``completed`` carries
  the ``lobster.result/1`` as ``result``); a paused run ends with
  ``run_state needs_decision {decision}`` followed by its
  ``decision_required`` events, and gets ``completed {resumed_by}`` in its
  buffer once answered. A cancelled run carries ``reason``: ``cancel``,
  ``decision_abandoned`` or ``decision_unsupported``.
* ``decision_required``  ``{decision_id, request_version, data, origin_run_id}``
  replaces the client's ``interrupt`` event. Lobster mints ``decision_id``
  once per question; LangGraph's ``interrupt_id`` stays inside the engine.

The decision contract is engine-owned and frozen (Phase 0 A2): a question
that surfaces again keeps its ``decision_id`` and gets the next
``request_version``; an answer carries ``source`` (``human`` or ``model``)
and an optional ``comment``; a refused answer leaves the run unchanged.
"""

from __future__ import annotations

import re
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
# shortcut: in-memory buffer capped per run; becomes sqlite-backed with the
# SqliteSaver work (Phase 0 B4).
_MAX_EVENTS_PER_RUN = 10_000


class DecisionError(RuntimeError):
    """An answer the run API refuses; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


_RUN_ID_RE = re.compile(r"run_[0-9a-f]{32}")
ANSWER_SOURCES = frozenset({"human", "model"})
# Closed list for a cancelled run (Phase 0 A2, C5); ``failed`` keeps its own.
CANCEL_REASONS = frozenset({"cancel", "decision_abandoned", "decision_unsupported"})
# Key of the resume value ``ask_user`` unwraps (answer plus who gave it).
DECISION_ENVELOPE = "__lobster_decision__"


@dataclass
class Decision:
    decision_id: str
    interrupt_id: Optional[str]
    origin_run_id: str
    data: Any
    request_version: int = 1
    state: str = "pending"  # pending -> answered | cancelled
    resumed_by: Optional[str] = None
    ask_id: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """The ``decision`` object of ``run_state needs_decision``."""
        data = self.data if isinstance(self.data, dict) else {}
        payload = data.get("data") if isinstance(data.get("data"), dict) else {}
        question = (
            payload.get("question")
            or payload.get("label")
            or data.get("fallback_prompt")
            or ""
        )
        return {
            "decision_id": self.decision_id,
            "request_version": self.request_version,
            "question": question,
            # 1.3 emits exactly one component per decision.
            "components": [
                {
                    "n": 1,
                    "component": data.get("component"),
                    "data": payload,
                    "fallback_prompt": data.get("fallback_prompt", question),
                }
            ],
        }


@dataclass
class Run:
    run_id: str
    session_id: Optional[str]
    state: str = "running"
    resumed_from: Optional[str] = None
    resumed_by: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    cancel_event: Optional[threading.Event] = None
    # When the work began; a run that resumes another keeps that run's start, so
    # a result lists the files of the whole piece of work.
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # The token tracker and its counters when the work began, so the result can
    # say what the run cost; a run that resumes another keeps that run's mark.
    usage_mark: Optional[Tuple[Any, Dict[str, float]]] = None
    _seq: int = 0
    _source: Optional[Iterator[Dict[str, Any]]] = None
    _manager: Optional["RunManager"] = None

    def stream(self) -> Iterator[Dict[str, Any]]:
        """Drive the run and yield its events as they happen."""
        if self._source is None:
            raise RuntimeError(f"run {self.run_id} has already been streamed")
        source, self._source = self._source, None
        error: Optional[Dict[str, Any]] = None
        decisions: List[Dict[str, Any]] = []
        try:
            yield self._record({"type": "run_state", "state": "running"})
            for event in source:
                etype = event.get("type")
                if etype == "interrupt":
                    # Drain the batch: the client yields every pending
                    # interrupt before it returns.
                    decisions.append(self._manager._register_decision(self, event))
                    continue
                if etype == "complete":
                    self.result = self._manager._result_from(self, event)
                elif etype == "error":
                    error = {
                        "code": event.get("error_type") or "error",
                        "message": str(event.get("error", "")),
                        "retryable": bool(event.get("is_rate_limit")),
                    }
                yield self._record(dict(event))

            if decisions:
                # Record everything before yielding: a consumer may stop
                # reading at the first decision.
                self.state = "needs_decision"
                first = self._manager.decisions[decisions[0]["decision_id"]]
                pending = [
                    self._record(
                        {
                            "type": "run_state",
                            "state": "needs_decision",
                            "decision": first.summary(),
                        }
                    )
                ]
                pending.extend(self._record(d) for d in decisions)
                yield from pending
                return

            if self.result is not None:
                yield self._finish("completed", result=self.result)
            elif self.cancel_event is not None and self.cancel_event.is_set():
                yield self._finish("cancelled", reason="cancel")
            elif error is not None:
                yield self._finish("failed", error=error)
            else:
                yield self._finish("failed", reason="stream_ended")
        except GeneratorExit:
            # The consumer stopped reading (a cancel in the Go launcher).
            if self.state not in TERMINAL_STATES and self.state != "needs_decision":
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self._finish("cancelled", reason="cancel")
                else:
                    self._finish("failed", reason="consumer_closed")
            raise
        except BaseException as exc:
            if self.state not in TERMINAL_STATES:
                error = {"code": "exception", "message": str(exc), "retryable": False}
                self._record({"type": "error", **error})
                self._finish("failed", reason="exception", error=error)
            raise

    def _finish(self, state: str, **extra: Any) -> Dict[str, Any]:
        self.state = state
        return self._record({"type": "run_state", "state": state, **extra})

    def _record(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event["run_id"] = self.run_id
        self._seq += 1
        event["seq"] = self._seq
        # run_state events are always kept, so events[-1] of an ended run is
        # its terminal state even past the cap.
        if len(self.events) < _MAX_EVENTS_PER_RUN or event["type"] == "run_state":
            self.events.append(event)
        return event


class RunManager:
    """Runs and decisions of one Lobster session (one ``AgentClient``)."""

    def __init__(self, client: Any, *, report_files: bool = False):
        self.client = client
        # Off by default: a completed result lists the files written since the
        # run began only for a host that asked (the MCP server). The protocol v1
        # result of ``agent --stdio`` and ``query --json`` stays as it was.
        self.report_files = report_files
        self.runs: Dict[str, Run] = {}
        self.decisions: Dict[str, Decision] = {}
        self._by_ask_id: Dict[str, Decision] = {}
        self.latest_run_id: Optional[str] = None
        # Re-entrant: answer() holds it across _new_run() and the origin's finish.
        self._lock = threading.RLock()

    @property
    def session_id(self) -> Optional[str]:
        return getattr(self.client, "session_id", None)

    def start(
        self,
        prompt: str,
        *,
        cancel_event: Optional[threading.Event] = None,
        run_id: Optional[str] = None,
    ) -> Run:
        """Create a run for *prompt*; iterate ``run.stream()`` to execute it.

        *run_id* lets the caller pick the id (``new_run_id()`` shape, unused);
        otherwise one is minted.
        """
        cancel_event = cancel_event or threading.Event()
        run = self._new_run(cancel_event, run_id=run_id)
        try:
            run._source = self.client.query(
                prompt, stream=True, cancel_event=cancel_event
            )
        except BaseException as exc:
            # The run is registered already: end it rather than leave it running.
            error = {"code": "exception", "message": str(exc), "retryable": False}
            self.end(run.run_id, error=error)
            raise
        return run

    def answer(
        self,
        decision_id: str,
        answer: Any,
        *,
        source: str,
        request_version: Optional[int] = None,
        comment: Optional[str] = None,
        channel: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Run:
        """Resume the paused run with *answer*; returns the new run at once.

        *source* says who answered (``human`` or ``model``); it is recorded
        as told. *channel* names the surface that carried the answer (for
        example ``mcp:claude-code``); the envelope has no ``channel`` key when
        none is given. A refused answer raises :class:`DecisionError` and
        leaves the run unchanged.
        """
        if not isinstance(decision_id, str):
            raise DecisionError("bad_answer", "decision_id must be a string")
        if not isinstance(source, str) or source not in ANSWER_SOURCES:
            raise DecisionError(
                "bad_answer", f"source must be one of {sorted(ANSWER_SOURCES)}"
            )
        if comment is not None and not isinstance(comment, str):
            raise DecisionError("bad_answer", "comment must be a string")
        if channel is not None and not isinstance(channel, str):
            raise DecisionError("bad_answer", "channel must be a string")
        # One transition under the lock, so a concurrent cancel() or a
        # re-surfacing of the same question sees either before or after.
        with self._lock:
            decision = self.decisions.get(decision_id)
            if decision is None:
                raise DecisionError("unknown_decision", f"no decision {decision_id}")
            if decision.state == "answered":
                raise DecisionError(
                    "already_answered",
                    f"decision {decision_id} was already answered",
                    resumed_by=decision.resumed_by,
                )
            origin = self.runs[decision.origin_run_id]
            if decision.state == "cancelled" or origin.state == "cancelled":
                raise DecisionError(
                    "rejected_cancelled", f"run {origin.run_id} was cancelled"
                )
            if (
                request_version is not None
                and request_version != decision.request_version
            ):
                raise DecisionError(
                    "rejected_stale",
                    f"decision {decision_id} is at version {decision.request_version}",
                )
            decision.state = "answered"
            run = self._new_run(
                cancel_event or threading.Event(), resumed_from=origin.run_id
            )
            decision.resumed_by = run.run_id
            origin.resumed_by = run.run_id
            interrupt_id = decision.interrupt_id
            record = {
                "decision_id": decision.decision_id,
                "request_version": decision.request_version,
                "answer": answer,
                "source": source,
                "comment": comment,
            }
            if channel is not None:
                record["channel"] = channel
            envelope = {DECISION_ENVELOPE: record}
            resume = {interrupt_id: envelope} if interrupt_id else envelope
            # A generator: nothing runs until the caller iterates run.stream().
            run._source = self.client.resume_from_interrupt(
                resume, stream=True, cancel_event=run.cancel_event
            )
            if origin.state == "needs_decision" and not any(
                d.state == "pending" and d.origin_run_id == origin.run_id
                for d in self.decisions.values()
            ):
                origin._finish("completed", resumed_by=run.run_id)
        return run

    def cancel(self, run_id: str, reason: str = "") -> str:
        """Request cooperative cancellation; returns the outcome code.

        *reason* is one of ``CANCEL_REASONS`` (default ``cancel``).
        """
        reason = reason or "cancel"
        if reason not in CANCEL_REASONS:
            raise ValueError(f"cancel reason must be one of {sorted(CANCEL_REASONS)}")
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                return "unknown_run"
            if run.state in TERMINAL_STATES:
                return "already_terminal"
            if run.state == "needs_decision":
                for decision in self.decisions.values():
                    if decision.origin_run_id == run_id and decision.state == "pending":
                        decision.state = "cancelled"
                run._finish("cancelled", reason=reason)
                return "cancelled"
            if run.cancel_event is not None:
                run.cancel_event.set()
            return "cancel_requested"

    def end(
        self, run_id: str, *, error: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """End a run its consumer dropped: ``cancelled`` (``cancel``), or
        ``failed`` (``exception``) with *error*. Returns the terminal event,
        or None when the run had already ended."""
        with self._lock:
            run = self.runs.get(run_id)
            if run is None or run.state in TERMINAL_STATES:
                return None
            for decision in self.decisions.values():
                if decision.origin_run_id == run_id and decision.state == "pending":
                    decision.state = "cancelled"
            if error is None:
                return run._finish("cancelled", reason="cancel")
            run._record({"type": "error", **error})
            return run._finish("failed", reason="exception", error=error)

    def head(self, run_id: str) -> Optional[Run]:
        """The live end of *run_id*'s answer chain (follows ``resumed_by``)."""
        run = self.runs.get(run_id)
        while run is not None and run.resumed_by in self.runs:
            run = self.runs[run.resumed_by]
        return run

    def events(self, run_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
        """Recorded events of *run_id* with ``seq`` greater than *after_seq*."""
        run = self.runs.get(run_id)
        if run is None:
            return []
        return [event for event in run.events if event["seq"] > after_seq]

    def get(self) -> Dict[str, Any]:
        """Snapshot of the session: latest run, its state, last result."""
        latest = self.runs.get(self.latest_run_id) if self.latest_run_id else None
        last_result = None
        for run in reversed(list(self.runs.values())):
            if run.result is not None:
                last_result = run.result
                break
        return {
            "session_id": self.session_id,
            "latest_run_id": self.latest_run_id,
            "state": latest.state if latest else None,
            "pending_decisions": [
                d.decision_id for d in self.decisions.values() if d.state == "pending"
            ],
            "result": last_result,
        }

    def _new_run(
        self,
        cancel_event: threading.Event,
        resumed_from: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Run:
        if run_id is not None and not is_run_id(run_id):
            raise ValueError(f"malformed run_id: {run_id!r}")
        run = Run(
            run_id=run_id or new_run_id(),
            session_id=self.session_id,
            resumed_from=resumed_from,
            cancel_event=cancel_event,
            _manager=self,
        )
        with self._lock:
            if run.run_id in self.runs:
                raise ValueError(f"duplicate run_id: {run.run_id}")
            if resumed_from in self.runs:
                run.started_at = self.runs[resumed_from].started_at
                run.usage_mark = self.runs[resumed_from].usage_mark
            else:
                run.usage_mark = mark_usage(self.client)
            self.runs[run.run_id] = run
            self.latest_run_id = run.run_id
        return run

    def _register_decision(self, run: Run, event: Dict[str, Any]) -> Dict[str, Any]:
        data = event.get("data", {})
        ask_id = None
        if isinstance(data, dict) and "ask_id" in data:
            data = dict(data)
            ask_id = data.pop("ask_id")
        with self._lock:
            decision = self._by_ask_id.get(ask_id) if ask_id else None
            if decision is not None:
                # The same question surfaced again: keep its id, bump the version.
                decision.request_version += 1
                decision.interrupt_id = event.get("interrupt_id")
                decision.origin_run_id = run.run_id
                decision.data = data
                decision.state = "pending"
                decision.resumed_by = None
            else:
                decision = Decision(
                    decision_id=f"dec_{uuid.uuid4().hex}",
                    interrupt_id=event.get("interrupt_id"),
                    origin_run_id=run.run_id,
                    data=data,
                    ask_id=ask_id,
                )
                self.decisions[decision.decision_id] = decision
                if ask_id:
                    self._by_ask_id[ask_id] = decision
            return {
                "type": "decision_required",
                "decision_id": decision.decision_id,
                "origin_run_id": run.run_id,
                "request_version": decision.request_version,
                "data": decision.data,
            }

    def _result_from(self, run: Run, event: Dict[str, Any]) -> Dict[str, Any]:
        """``lobster.result/1`` for a completed run."""
        started_at = run.started_at if self.report_files else None
        return build_result(
            self.client,
            run.run_id,
            self.session_id,
            event,
            started_at=started_at,
            usage_mark=run.usage_mark,
        )


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def is_run_id(value: Any) -> bool:
    return isinstance(value, str) and _RUN_ID_RE.fullmatch(value) is not None


def build_result(
    client: Any,
    run_id: str,
    session_id: Optional[str],
    event: Dict[str, Any],
    started_at: Optional[datetime] = None,
    usage_mark: Optional[Tuple[Any, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """``lobster.result/1`` from a completion ``event`` (or ``client.query`` result).

    With *started_at* (tz-aware) the result also lists the files written since
    then: ``files_written``, or null plus ``files_written_error`` when the
    workspace could not be read. With *usage_mark* (from ``mark_usage``)
    ``token_usage`` also carries the ``run_*`` cost of the work since the mark.
    """
    from lobster.version import __version__

    data_manager = getattr(client, "data_manager", None)
    has_data = None
    modalities = None
    if data_manager is not None:
        try:
            modalities = list(data_manager.list_modalities())
            has_data = bool(modalities)
        except Exception:
            modalities = None
    result = {
        "schema_version": "lobster.result/1",
        "engine": {"name": "lobster-ai", "version": __version__},
        "session_id": event.get("session_id", session_id),
        "run_id": run_id,
        "response": event.get("response", ""),
        "duration": event.get("duration"),
        "token_usage": with_run_cost(event.get("token_usage"), client, usage_mark),
        "last_agent": event.get("last_agent"),
        "has_data": has_data,
        "modalities": modalities,
    }
    if started_at is not None:
        result.update(_files_written(data_manager, started_at))
    return result


RUN_COST_KEYS = (
    "run_cost_usd",
    "run_total_tokens",
    "run_cost_complete",
    "run_unpriced_invocations",
    "run_missing_usage_invocations",
)


def mark_usage(client: Any) -> Optional[Tuple[Any, Dict[str, float]]]:
    """The client's token tracker and its counters now; None without a tracker.
    Accounting never stops a run."""
    try:
        tracker = client.token_tracker
        return tracker, tracker.mark()
    except Exception:
        return None


def with_run_cost(
    usage: Any, client: Any, mark: Optional[Tuple[Any, Dict[str, float]]]
) -> Any:
    """*usage* (a result's ``token_usage``) with the ``run_*`` cost of the work
    since *mark*. Without a mark, when the client's tracker is no longer the one
    marked, or when it cannot cost the run, *usage* is returned as it is: the
    ``run_*`` keys are then absent, never zero."""
    if mark is None or not isinstance(usage, dict):
        return usage
    tracker, counters = mark
    try:
        if getattr(client, "token_tracker", None) is not tracker:
            return usage
        cost = tracker.cost_since(counters)
        if not isinstance(cost, dict):
            return usage
        return {**usage, **{key: cost[key] for key in RUN_COST_KEYS}}
    except Exception:
        return usage


def _files_written(data_manager: Any, since: datetime) -> Dict[str, Any]:
    """The result fields for the files on disk written since *since*.

    Nothing when the session has no file source. A storage fault is reported,
    never raised: the run did its work.
    """
    from lobster.core.figures.errors import FigureError, safe_reason

    files_since = getattr(data_manager, "files_written_since", None)
    if not callable(files_since):
        return {}
    try:
        files = files_since(since)
    except (FigureError, OSError) as exc:
        # safe_reason is data-free (no path, no message).
        reason = safe_reason(exc) if isinstance(exc, FigureError) else "os_error"
        return {"files_written": None, "files_written_error": reason}
    return {"files_written": files} if isinstance(files, list) else {}


_managers: "weakref.WeakKeyDictionary[Any, RunManager]" = weakref.WeakKeyDictionary()
_managers_lock = threading.Lock()


def runs_for(client: Any) -> RunManager:
    """The one ``RunManager`` of *client* (created on first use)."""
    with _managers_lock:
        manager = _managers.get(client)
        if manager is None:
            manager = RunManager(client)
            _managers[client] = manager
        return manager
