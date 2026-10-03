"""Structural routing telemetry.

This recorder writes the ordered tool sequence with agent attribution and classifies
each code-execution call by whether a specialist handoff occurred earlier in the session:

  * **pre-handoff fallback** — code ran before any specialist was reached.
  * **post-handoff escalation** — a specialist was reached and code ran afterward.

The recorded facts are event order and tool identity. This module does not assess whether
a specialist's response was sufficient or infer intent from prose.

Design rules
------------
* **Observational and fail-open**, following the AQUADIF monitor: it counts, never
  blocks, never alters routing, and any internal error is swallowed. Telemetry must not
  be able to break a session.
* **Facts only.** Names, ordering, counts. Never data values, never user prose.
* **Dual-signal attribution with a hard-fail self-test.** A tracer that is silently blind
  reports a clean system, which would invert the conclusion it exists to inform. See
  ``attribution_health``.
* **Bounded.** A pathological session cannot grow the record without limit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_EXEC_TOOL = "execute_custom_code"
HANDOFF_PREFIX = "handoff_to_"

#: Cap on recorded events per session. A runaway loop must not produce an unbounded file.
MAX_EVENTS = 2000

#: Shared schema version. Claim checks write into the same event stream, so
#: both readers use one record shape.
SCHEMA_VERSION = 1

#: Event kinds. Kept explicit so downstream readers and claim-check recording use
#: constants rather than strings scattered through call sites.
EVENT_TOOL_CALL = "tool_call"
EVENT_SESSION_SUMMARY = "routing_summary"


@dataclass
class ToolEvent:
    """One tool invocation, as observed."""

    index: int
    tool_name: str
    #: Agent credited by the callback's own attribution (metadata/run_name chain).
    agent_from_metadata: str | None = None
    #: Agent implied by the most recent ``handoff_to_*`` in the sequence.
    agent_from_handoff_chain: str | None = None
    #: True when the two signals disagree — see ``attribution_health``.
    attribution_conflict: bool = False
    #: For code-exec calls only: did a handoff complete earlier in this session?
    is_post_handoff: bool | None = None
    #: For code-exec calls only: the agent handed off to most recently.
    preceding_handoff_agent: str | None = None
    #: For code-exec calls only: was a modality/file written since that handoff?
    artifact_written_since_handoff: bool | None = None
    #: LangChain's run identity. `index` gives a total order within this
    #: stream, but a counter cannot express NESTING — it cannot say that a tool call
    #: happened *inside* a particular agent's run. `run_id`/`parent_run_id` are the parent
    #: edges the callback layer already computes (`callbacks.py` maintains `run_to_agent`
    #: and `current_run_id`), and they are also the join key between this stream,
    #: `provenance.jsonl` and `session.json`. Optional: absent when the callback does not
    #: supply them, so nothing here depends on their presence.
    run_id: str | None = None
    parent_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": EVENT_TOOL_CALL,
            "index": self.index,
            "tool_name": self.tool_name,
            "agent": self.agent_from_metadata or self.agent_from_handoff_chain,
            "agent_from_metadata": self.agent_from_metadata,
            "agent_from_handoff_chain": self.agent_from_handoff_chain,
            "attribution_conflict": self.attribution_conflict,
            "is_post_handoff": self.is_post_handoff,
            "preceding_handoff_agent": self.preceding_handoff_agent,
            "artifact_written_since_handoff": self.artifact_written_since_handoff,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
        }


class RoutingTelemetryRecorder:
    """Records the routing shape of one session.

    Attached to ``TokenTrackingCallback`` the way ``AquadifMonitor`` is: one optional
    reference, one injection point, fail-open. Only the token tracker feeds it, so
    display handlers cannot double-count.
    """

    def __init__(
        self,
        session_dir: str | Path | None = None,
        data_manager: Any = None,
        max_events: int = MAX_EVENTS,
    ) -> None:
        self.events: list[ToolEvent] = []
        self.max_events = max_events
        self._data_manager = data_manager
        self._session_dir = Path(session_dir) if session_dir else None
        self._truncated = False

        # Handoff-chain state
        self._current_handoff_agent: str | None = None
        self._seen_handoff = False

        # Artifact tracking: modality names known at the last handoff boundary.
        self._modalities_at_handoff: set[str] | None = None

        # Attribution health
        self._metadata_signal_seen = 0
        self._handoff_signal_seen = 0
        self._conflicts = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tool_invocation(
        self,
        tool_name: str,
        current_agent: str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        """Record one tool call. Never raises.

        ``run_id``/``parent_run_id`` are optional: they carry LangChain's run
        identity so this stream can be joined to ``provenance.jsonl`` and so nesting is
        expressible, which a bare ``index`` counter cannot do.
        """
        try:
            self._record(tool_name, current_agent, run_id, parent_run_id)
        except (
            Exception
        ):  # nosec B110 # Advisory telemetry must never change tool execution results. noqa: BLE001 - telemetry must never break a session
            pass

    def _record(
        self,
        tool_name: str,
        current_agent: str | None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        if len(self.events) >= self.max_events:
            self._truncated = True
            return

        tool_name = tool_name or "unknown_tool"
        event = ToolEvent(
            index=len(self.events),
            tool_name=tool_name,
            agent_from_metadata=current_agent or None,
            agent_from_handoff_chain=self._current_handoff_agent,
            run_id=str(run_id) if run_id else None,
            parent_run_id=str(parent_run_id) if parent_run_id else None,
        )

        if current_agent:
            self._metadata_signal_seen += 1
        if self._current_handoff_agent:
            self._handoff_signal_seen += 1

        # Conflict only counts where both signals exist and name different agents.
        # "supervisor" is not a conflict: between handoffs the supervisor is legitimately
        # the actor, so metadata saying supervisor while the chain remembers the last
        # delegate is expected, not a disagreement.
        if (
            current_agent
            and self._current_handoff_agent
            and current_agent != self._current_handoff_agent
            and current_agent != "supervisor"
        ):
            event.attribution_conflict = True
            self._conflicts += 1

        if tool_name.startswith(HANDOFF_PREFIX):
            target = tool_name[len(HANDOFF_PREFIX) :]
            self._seen_handoff = True
            self._current_handoff_agent = target
            # Snapshot modality names so a later code call can be asked whether the
            # specialist wrote anything in between.
            self._modalities_at_handoff = self._modality_names()
        elif tool_name == "transfer_back_to_supervisor":
            self._current_handoff_agent = None
        elif tool_name == CODE_EXEC_TOOL:
            event.is_post_handoff = self._seen_handoff
            event.preceding_handoff_agent = self._current_handoff_agent
            event.artifact_written_since_handoff = (
                self._artifact_written_since_handoff()
            )

        self.events.append(event)

    def _modality_names(self) -> set[str] | None:
        """Modality names, or None when unreadable.

        None is distinct from an empty set: "could not measure" must not read as
        "nothing was written".
        """
        if self._data_manager is None:
            return None
        try:
            records = self._data_manager.list_modality_records() or []
            return {str(r.get("name")) for r in records if r.get("name")}
        except Exception:  # noqa: BLE001
            return None

    def _artifact_written_since_handoff(self) -> bool | None:
        """Did a modality appear since the last handoff? None when unmeasurable."""
        if self._modalities_at_handoff is None:
            return None
        current = self._modality_names()
        if current is None:
            return None
        return bool(current - self._modalities_at_handoff)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @property
    def attribution_health(self) -> dict[str, Any]:
        """Whether the tracer can be believed.

        A recorder that observed no tool calls, or whose two signals disagree
        constantly, is not measuring the system — it is producing a clean-looking
        artifact. That failure silently inverts any conclusion drawn from it, so it is
        surfaced rather than buried.
        """
        total = len(self.events)
        conflict_rate = (self._conflicts / total) if total else 0.0
        blind = total == 0
        return {
            "events_recorded": total,
            "metadata_signal_seen": self._metadata_signal_seen,
            "handoff_signal_seen": self._handoff_signal_seen,
            "attribution_conflicts": self._conflicts,
            "conflict_rate": round(conflict_rate, 4),
            "truncated": self._truncated,
            "blind": blind,
            "trustworthy": (not blind) and conflict_rate <= 0.25,
        }

    def three_way_split(self) -> dict[str, Any]:
        """The decomposition the escalation debate turns on.

        ``mis_route`` is intentionally absent: judging it needs an expected owner per
        task, which is benchmark ground truth the engine does not have at runtime. The
        destination agent is recorded instead, so an offline analysis with ground truth
        can compute it without re-instrumenting anything.
        """
        code_events = [e for e in self.events if e.tool_name == CODE_EXEC_TOOL]
        post = [e for e in code_events if e.is_post_handoff]
        pre = [e for e in code_events if e.is_post_handoff is False]

        handoffs = [
            e.tool_name[len(HANDOFF_PREFIX) :]
            for e in self.events
            if e.tool_name.startswith(HANDOFF_PREFIX)
        ]

        # Of the post-handoff code calls, how many ran despite an artifact existing?
        # That is the sharpest signal available without intent inference: the specialist
        # demonstrably produced something and the supervisor recomputed regardless.
        post_with_artifact = [
            e for e in post if e.artifact_written_since_handoff is True
        ]

        return {
            "handoffs": handoffs,
            "distinct_agents_used": sorted(set(handoffs)),
            "code_calls_total": len(code_events),
            "code_calls_pre_handoff": len(pre),
            "code_calls_post_handoff": len(post),
            "post_handoff_with_artifact_present": len(post_with_artifact),
            "tool_calls_total": len(self.events),
        }

    def routing_diversity(self) -> dict[str, Any]:
        """Tool-use spread, for the "not 100% one tool" success criterion.

        Nothing currently reports this, so the claim has never been checkable.
        """
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.tool_name] = counts.get(event.tool_name, 0) + 1
        total = sum(counts.values())
        top_share = (max(counts.values()) / total) if total else 0.0
        return {
            "distinct_tools": len(counts),
            "top_tool_share": round(top_share, 4),
            "tool_counts": dict(sorted(counts.items())),
        }

    def session_summary(self) -> dict[str, Any]:
        return {
            "kind": EVENT_SESSION_SUMMARY,
            "v": SCHEMA_VERSION,
            "split": self.three_way_split(),
            "diversity": self.routing_diversity(),
            "attribution_health": self.attribution_health,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> Path | None:
        """Write events plus a summary to ``routing_events.jsonl``.

        Beside ``provenance.jsonl`` in the session directory, same append-and-fsync
        durability, and the same shared schema claim checks write into. Returns the path, or
        None when persistence is disabled or fails — never raises.
        """
        if self._session_dir is None:
            return None
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_dir / "routing_events.jsonl"
            with open(path, "a", encoding="utf-8") as handle:
                handle.writelines(
                    json.dumps({"v": SCHEMA_VERSION, **event.to_dict()}) + "\n"
                    for event in self.events
                )
                handle.write(json.dumps(self.session_summary()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return path
        except Exception:  # noqa: BLE001 - fail open
            return None


def summarize_sessions(paths: list[Path]) -> dict[str, Any]:
    """Aggregate the three-way split across many session files.

    This is the offline reader that turns per-session records into the number the
    escalation debate needs. Sessions whose attribution was untrustworthy are counted
    and **excluded**, not silently pooled: including a blind session would bias the
    result toward "no escalation" for the wrong reason.
    """
    totals = {
        "sessions_read": 0,
        "sessions_excluded_untrustworthy": 0,
        "code_calls_total": 0,
        "code_calls_pre_handoff": 0,
        "code_calls_post_handoff": 0,
        "post_handoff_with_artifact_present": 0,
        "sessions_with_any_code": 0,
    }

    for path in paths:
        try:
            summary = None
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") == EVENT_SESSION_SUMMARY:
                    summary = record
            if summary is None:
                continue

            if not summary.get("attribution_health", {}).get("trustworthy", False):
                totals["sessions_excluded_untrustworthy"] += 1
                continue

            totals["sessions_read"] += 1
            split = summary.get("split", {})
            for key in (
                "code_calls_total",
                "code_calls_pre_handoff",
                "code_calls_post_handoff",
                "post_handoff_with_artifact_present",
            ):
                totals[key] += int(split.get(key) or 0)
            if split.get("code_calls_total"):
                totals["sessions_with_any_code"] += 1
        except (
            Exception
        ):  # nosec B112 # A malformed telemetry file must not abort independent file aggregation. noqa: BLE001 - one bad file must not stop the sweep
            continue

    code_total = totals["code_calls_total"]
    if code_total:
        totals["post_handoff_share"] = round(
            totals["code_calls_post_handoff"] / code_total, 4
        )
        totals["pre_handoff_share"] = round(
            totals["code_calls_pre_handoff"] / code_total, 4
        )
    else:
        # No code use at all is a legitimate reading, but it is NOT evidence about the
        # escalation split — there is nothing to apportion. Say so rather than
        # reporting 0%.
        totals["post_handoff_share"] = None
        totals["pre_handoff_share"] = None
        totals["note"] = (
            "no execute_custom_code calls recorded; the split is undefined rather "
            "than zero"
        )

    return totals
