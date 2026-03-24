"""
AquadifMonitor: Runtime introspection service for AQUADIF tool category usage.

This module has ZERO imports from lobster.* to prevent circular imports when
callbacks.py imports it. It uses pure stdlib only.

Usage:
    from lobster.core.aquadif_monitor import AquadifMonitor, CodeExecEntry

    monitor = AquadifMonitor(tool_metadata_map={})
    # tool_metadata_map is populated later by graph.py after tool creation

    # Called from TokenTrackingCallback.on_tool_start (single injection point)
    monitor.record_tool_invocation(tool_name="analyze_cells", current_agent="transcriptomics_expert")

    # Called from DataManagerV2.log_tool_usage (provenance observation point)
    monitor.record_provenance_call(tool_name="analyze_cells", has_real_ir=True)

    # Get summary for Omics-OS Cloud SSE enrichment
    summary = monitor.get_session_summary()

Design:
    - Instantiated once per session in AgentClient.__init__
    - Passed as optional arg (aquadif_monitor=None disables all monitoring)
    - All public methods are fail-open: exceptions swallowed, never re-raised
    - threading.Lock used for compound mutations (deque append + dict update)
    - collections.deque(maxlen=100) for bounded CODE_EXEC log
    - Single dict increments are GIL-safe in CPython (no lock needed)

CRITICAL: Do NOT add any imports from lobster.* to this module.
          This prevents import cycles since callbacks.py imports AquadifMonitor.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class CodeExecEntry:
    """Single CODE_EXEC invocation record in the bounded log."""

    tool_name: str
    timestamp: str  # ISO 8601 format string
    agent: str  # Agent name from TokenTrackingCallback.current_agent


class AquadifMonitor:
    """
    Runtime introspection service for AQUADIF tool category usage and provenance compliance.

    Injected as an optional service into the existing callback chain. All methods
    are fail-open: exceptions are caught internally, never re-raised.

    Pass aquadif_monitor=None to callers to disable all monitoring with zero overhead.

    Thread safety:
        - Simple dict counter increments (single int ops) are GIL-safe in CPython
        - Compound mutations (deque append + dict update) are lock-protected
        - Lock-held sections are kept short; methods never call each other while holding lock
    """

    def __init__(self, tool_metadata_map: Dict[str, Dict[str, Any]]):
        """
        Initialize the monitor.

        Args:
            tool_metadata_map: Mapping of tool name to AQUADIF metadata dict.
                Format: {tool_name: {"categories": [...], "provenance": bool}}
                Built at graph construction time from tool .metadata attributes.
                May be empty ({}) if constructed before tools are known; graph.py
                populates the map after tool creation via _tool_metadata_map assignment.
        """
        self._tool_metadata_map = tool_metadata_map
        self._lock = threading.Lock()

        # Category distribution: {category_name: invocation_count}
        # Example: {"ANALYZE": 3, "IMPORT": 1}
        self._category_counts: Dict[str, int] = {}

        # Provenance tracking: {tool_name: "real_ir" | "hollow_ir" | "missing"}
        # "missing" = provenance required but log_tool_usage never called
        # "hollow_ir" = ir=None bridge pattern (tracked but no AnalysisStep notebook)
        # "real_ir" = full AnalysisStep provenance (fully compliant)
        self._provenance_status: Dict[str, str] = {}

        # Tool invocation counts: {tool_name: count}
        # Used to compute total_invocations in get_session_summary()
        self._tool_invocation_counts: Dict[str, int] = {}

        # CODE_EXEC invocation log (bounded to maxlen=100 — auto-evicts oldest)
        self._code_exec_log: deque = deque(maxlen=100)

    def record_tool_invocation(
        self, tool_name: str, current_agent: str = "unknown"
    ) -> None:
        """
        Record a tool invocation. Called from TokenTrackingCallback.on_tool_start.

        SINGLE INJECTION POINT — only TokenTrackingCallback.on_tool_start calls this.
        Other display handlers (TerminalCallbackHandler, StreamingCallbackHandler)
        must NOT call this method to avoid double-counting in cloud sessions.

        Increments:
            - category_counts for each category in tool's metadata
            - tool_invocation_counts for this tool name
            - code_exec_log if tool has CODE_EXEC category
            - sets provenance_status[tool_name] = "missing" if provenance required
              and tool not yet seen

        Args:
            tool_name: Name of the tool being invoked (from serialized["name"])
            current_agent: Active agent name (from TokenTrackingCallback.current_agent)
        """
        try:
            metadata = self._tool_metadata_map.get(tool_name, {})
            categories = metadata.get("categories", [])
            requires_provenance = metadata.get("provenance", False)

            # Increment category counts (single dict ops are GIL-safe in CPython)
            for cat in categories:
                self._category_counts[cat] = self._category_counts.get(cat, 0) + 1

            # Compound mutation: update invocation count + provenance status + deque append
            with self._lock:
                self._tool_invocation_counts[tool_name] = (
                    self._tool_invocation_counts.get(tool_name, 0) + 1
                )

                # Pre-set provenance status to "missing" if required and not yet seen
                # Only set on first invocation — subsequent calls must not overwrite
                # a later record_provenance_call update (real_ir or hollow_ir)
                if requires_provenance and tool_name not in self._provenance_status:
                    self._provenance_status[tool_name] = "missing"

                # Log CODE_EXEC invocations to bounded deque
                if "CODE_EXEC" in categories:
                    self._code_exec_log.append(
                        CodeExecEntry(
                            tool_name=tool_name,
                            timestamp=datetime.now().isoformat(),
                            agent=current_agent,
                        )
                    )
        except Exception:
            pass  # Fail-open: monitor exception must never crash tool invocation

    def record_provenance_call(self, tool_name: str, has_real_ir: bool) -> None:
        """
        Record a provenance observation. Called from DataManagerV2.log_tool_usage.

        Observes the actual provenance call at the authoritative recording site.
        Does NOT parse output strings (brittle) — uses direct observation (correct).

        Status transitions:
            missing  → real_ir   (when has_real_ir=True)
            missing  → hollow_ir (when has_real_ir=False, ir=None bridge)
            hollow_ir → real_ir  (when has_real_ir=True, upgrade)
            real_ir  → real_ir   (no-op; real_ir cannot be downgraded)

        Args:
            tool_name: Name of the tool that called log_tool_usage
            has_real_ir: True if ir is not None (AnalysisStep present), False for ir=None bridge
        """
        try:
            with self._lock:
                if has_real_ir:
                    # Real AnalysisStep — set to real_ir (can upgrade from any status)
                    self._provenance_status[tool_name] = "real_ir"
                else:
                    # ir=None bridge pattern — only set to hollow_ir if not already real_ir
                    # real_ir status must not be downgraded
                    if self._provenance_status.get(tool_name) != "real_ir":
                        self._provenance_status[tool_name] = "hollow_ir"
        except Exception:
            pass  # Fail-open

    def get_category_distribution(self) -> Dict[str, int]:
        """
        Return current category invocation counts.

        Returns:
            Dict copy of {category_name: invocation_count}.
            Returns empty dict if no tools have been invoked.
            Modifying the returned dict does not affect internal state.
        """
        # Simple dict copy — GIL-safe for read in CPython
        return dict(self._category_counts)

    def get_provenance_status(self) -> Dict[str, List[str]]:
        """
        Return provenance compliance status grouped by status.

        Statuses:
            real_ir: Tools with real AnalysisStep IR (fully tracked)
            hollow_ir: Tools using ir=None bridge (tracked, no notebook output)
            missing: Provenance-required tools that never called log_tool_usage

        Returns:
            Dict with keys "real_ir", "hollow_ir", "missing", each mapping to
            a list of tool names in that status. Lock-protected for consistency.
        """
        with self._lock:
            result: Dict[str, List[str]] = {
                "real_ir": [],
                "hollow_ir": [],
                "missing": [],
            }
            for tool_name, status in self._provenance_status.items():
                if status in result:
                    result[status].append(tool_name)
            return result

    def get_code_exec_log(self) -> List[CodeExecEntry]:
        """
        Return a snapshot of the bounded CODE_EXEC invocation log.

        Returns:
            List copy of CodeExecEntry objects (newest up to maxlen=100).
            Modifying the returned list does not affect internal state.
        """
        with self._lock:
            return list(self._code_exec_log)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Return a structured summary of monitor state for Omics-OS Cloud SSE enrichment.

        Acquires lock for a consistent snapshot. O(1) computation from pre-built counters.
        Mirrors the pattern of TokenTrackingCallback.get_usage_summary().

        Returns:
            Dict with keys:
                category_distribution: {category_name: count}
                provenance_status: {"real_ir": [...], "hollow_ir": [...], "missing": [...]}
                code_exec_count: int
                code_exec_log: list of serialized dicts (tool_name, timestamp, agent)
                total_invocations: int (sum of all tool invocation counts)
        """
        with self._lock:
            # Build provenance status grouping under lock
            provenance_status: Dict[str, List[str]] = {
                "real_ir": [],
                "hollow_ir": [],
                "missing": [],
            }
            for tool_name, status in self._provenance_status.items():
                if status in provenance_status:
                    provenance_status[status].append(tool_name)

            return {
                "category_distribution": dict(self._category_counts),
                "provenance_status": provenance_status,
                "code_exec_count": len(self._code_exec_log),
                "code_exec_log": [
                    {
                        "tool_name": entry.tool_name,
                        "timestamp": entry.timestamp,
                        "agent": entry.agent,
                    }
                    for entry in self._code_exec_log
                ],
                "total_invocations": sum(self._tool_invocation_counts.values()),
            }
