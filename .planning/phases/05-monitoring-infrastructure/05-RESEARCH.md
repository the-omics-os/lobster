# Phase 5: Monitoring Infrastructure - Research

**Researched:** 2026-03-01
**Domain:** Python thread-safe stateful service, LangChain callback integration, runtime introspection
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MON-01 | `AquadifMonitor` class in `lobster/core/aquadif_monitor.py` with thread-safe state, bounded data structures, and fail-open error handling | stdlib `threading.Lock` + `collections.deque` — no new dependencies required |
| MON-02 | `graph.py` builds `tool_name → {categories, provenance}` lookup dict at graph construction time and passes it to monitor | All tools already have `.metadata` and `.tags` post-decorator; dict can be built by iterating all tool lists after agent creation |
| MON-03 | `TokenTrackingCallback.on_tool_start` calls `monitor.record_tool_invocation()` as the single injection point | `TokenTrackingCallback.on_tool_start` already exists at line 1002 in `lobster/utils/callbacks.py`; monitor is passed as optional constructor arg |
| MON-04 | `DataManagerV2.log_tool_usage` calls `monitor.record_provenance_call(tool_name, has_real_ir)` | `log_tool_usage` is a single method at line 1728 in `data_manager_v2.py`; only one code site to modify |
| MON-05 | CODE_EXEC invocations logged to bounded `deque` with tool name, timestamp, agent attribution | `deque(maxlen=100)` in `AquadifMonitor`; CODE_EXEC detected via `.metadata["categories"]` lookup in `record_tool_invocation` |
| MON-06 | `get_session_summary()` returns structured dict for Omics-OS Cloud SSE enrichment | Mirrors `TokenTrackingCallback.get_usage_summary()` pattern; consumed by `StreamingCallbackHandler` in `lobster-cloud/api/callbacks/streaming_callback.py` |
</phase_requirements>

---

## Summary

Phase 5 builds `AquadifMonitor`, a shared stateful service that plugs into the existing callback chain to provide runtime introspection of tool category usage and provenance compliance. The architecture is already locked from a brutalist peer review (2026-03-01): monitor is NOT a new callback handler, it is called from exactly one place (`TokenTrackingCallback.on_tool_start`) to avoid double-counting, and provenance is detected by observing `DataManagerV2.log_tool_usage` calls (not by parsing output strings).

The implementation is purely stdlib Python — no new dependencies. `threading.Lock` handles thread safety, `collections.deque` provides bounded CODE_EXEC log storage, and counter dicts are GIL-safe for single increments. The monitor is opt-in (defaults to `None`) so zero behavioral change to existing tool execution paths. All existing 221+ tools already have `.metadata` and `.tags` assigned, so the lookup dict is trivially buildable from the graph construction loop.

The key integration challenge is plumbing: `AquadifMonitor` must be constructed in `client.py`, passed through `create_bioinformatics_graph()`, stored on `TokenTrackingCallback`, and also passed to `DataManagerV2`. The graph construction function signature already accepts many optional kwargs, so adding `aquadif_monitor: Optional[AquadifMonitor] = None` follows the established pattern. The `StreamingCallbackHandler` in lobster-cloud reads a session summary at query end — `get_session_summary()` will be a new sibling of `TokenTrackingCallback.get_usage_summary()`.

**Primary recommendation:** Implement `AquadifMonitor` as a pure-stdlib service class with no external dependencies; wire it through the existing client → graph → callback chain; add a single hook into `DataManagerV2.log_tool_usage`.

---

## Standard Stack

### Core (all stdlib — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `threading` | stdlib | `threading.Lock` for compound state mutations | Correct tool for shared mutable state in a multi-threaded LangGraph execution |
| `collections.deque` | stdlib | Bounded circular buffer for CODE_EXEC log | `maxlen=100` gives automatic eviction; no memory growth |
| `dataclasses` | stdlib | `CodeExecEntry` dataclass for log entries | Consistent with existing codebase patterns (`AgentInfo`, `GraphMetadata`, etc.) |
| `datetime` | stdlib | Timestamps in CODE_EXEC entries | Already used in `TokenInvocation` and `AgentEvent` |

### Existing Codebase (already present — use, don't reinvent)

| Component | File | How This Phase Uses It |
|-----------|------|----------------------|
| `TokenTrackingCallback` | `lobster/utils/callbacks.py:755` | Single injection point — add `aquadif_monitor` param to `__init__`, call in `on_tool_start` |
| `DataManagerV2.log_tool_usage` | `lobster/core/data_manager_v2.py:1728` | Add `monitor.record_provenance_call(tool_name, has_real_ir=(ir is not None))` inside the `if self.provenance:` block |
| `create_bioinformatics_graph()` | `lobster/agents/graph.py:283` | Pass `aquadif_monitor` kwarg through to `TokenTrackingCallback` and `DataManagerV2` |
| `AquadifCategory` enum | `lobster/config/aquadif.py` | Use for CODE_EXEC detection: `"CODE_EXEC" in tool_metadata.get("categories", [])` |
| `get_usage_summary()` | `lobster/utils/callbacks.py:1226` | Pattern to mirror for `get_session_summary()` |
| `StreamingCallbackHandler` | `lobster-cloud/api/callbacks/streaming_callback.py:43` | Consumer of `get_session_summary()` at query end |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `threading.Lock` | `asyncio.Lock` | LangGraph callbacks fire from worker threads (confirmed in `StreamingCallbackHandler` docstring: "callbacks are triggered from worker threads"). `threading.Lock` is correct. `asyncio.Lock` would deadlock from thread context. |
| Single `threading.Lock` on whole monitor | Lock-per-counter | Simpler code, lower contention risk since counters are simple int ops (GIL-safe). Lock only needed for compound mutations like appending to deque + incrementing counter atomically. |
| `deque(maxlen=100)` | `list` | `list` is unbounded — memory leak in long sessions. `deque` auto-evicts. |
| Observing `DataManagerV2` | Parsing output strings | Parsing output is brittle (format changes break detection). `DataManagerV2.log_tool_usage` is the authoritative provenance record — observing it is both simpler and correct. |

**Installation:** No new packages required. Pure stdlib.

---

## Architecture Patterns

### Recommended File Structure

```
lobster/
├── core/
│   └── aquadif_monitor.py   # NEW: AquadifMonitor service class
├── utils/
│   └── callbacks.py          # MODIFY: add aquadif_monitor to TokenTrackingCallback
├── agents/
│   └── graph.py              # MODIFY: build tool_metadata_map, pass monitor through
└── core/
    └── data_manager_v2.py    # MODIFY: add hook in log_tool_usage

tests/
└── unit/
    └── core/
        └── test_aquadif_monitor.py  # NEW: unit tests for AquadifMonitor
```

### Pattern 1: AquadifMonitor Class Design

**What:** Thread-safe stateful service with fail-open error handling and bounded data structures.
**When to use:** Instantiated once per session in `client.py`, passed as optional arg throughout.

```python
# lobster/core/aquadif_monitor.py
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CodeExecEntry:
    """Single CODE_EXEC invocation record."""
    tool_name: str
    timestamp: str  # ISO format
    agent: str  # from TokenTrackingCallback.current_agent


class AquadifMonitor:
    """
    Runtime introspection service for AQUADIF tool category usage.

    Injected as an optional service into the existing callback chain.
    All methods are fail-open: exceptions are caught and logged, never re-raised.
    Pass aquadif_monitor=None to disable all monitoring with zero overhead.
    """

    def __init__(self, tool_metadata_map: Dict[str, Dict[str, Any]]):
        """
        Args:
            tool_metadata_map: {tool_name: {"categories": [...], "provenance": bool}}
                Built at graph construction time from tool .metadata attributes.
        """
        self._tool_metadata_map = tool_metadata_map
        self._lock = threading.Lock()

        # Category distribution: {category_name: count}
        self._category_counts: Dict[str, int] = {}

        # Provenance tracking: {tool_name: "real_ir" | "hollow_ir" | "missing"}
        # "missing" = provenance required but log_tool_usage never called
        self._provenance_status: Dict[str, str] = {}

        # CODE_EXEC log (bounded)
        self._code_exec_log: deque = deque(maxlen=100)

        # Tool invocation counts (for missing provenance detection)
        self._tool_invocation_counts: Dict[str, int] = {}

    def record_tool_invocation(self, tool_name: str, current_agent: str = "unknown") -> None:
        """
        Called from TokenTrackingCallback.on_tool_start.

        SINGLE INJECTION POINT — only TokenTrackingCallback calls this.
        Display handlers (Terminal, Textual, Streaming) must NOT call this.
        """
        try:
            metadata = self._tool_metadata_map.get(tool_name, {})
            categories = metadata.get("categories", [])
            requires_provenance = metadata.get("provenance", False)

            # GIL-safe: simple dict increments are atomic in CPython
            for cat in categories:
                self._category_counts[cat] = self._category_counts.get(cat, 0) + 1

            # Track invocations for provenance "missing" detection
            with self._lock:
                self._tool_invocation_counts[tool_name] = (
                    self._tool_invocation_counts.get(tool_name, 0) + 1
                )
                # Pre-set as "missing" if provenance required; will be updated by record_provenance_call
                if requires_provenance and tool_name not in self._provenance_status:
                    self._provenance_status[tool_name] = "missing"

                # Log CODE_EXEC invocations
                if "CODE_EXEC" in categories:
                    self._code_exec_log.append(CodeExecEntry(
                        tool_name=tool_name,
                        timestamp=datetime.now().isoformat(),
                        agent=current_agent,
                    ))
        except Exception:
            pass  # Fail-open: monitor exception never crashes tool invocation

    def record_provenance_call(self, tool_name: str, has_real_ir: bool) -> None:
        """
        Called from DataManagerV2.log_tool_usage.

        Observes actual provenance call, does NOT parse output strings.
        ir=None tools are tracked as "hollow_ir", not violations.
        """
        try:
            with self._lock:
                if has_real_ir:
                    self._provenance_status[tool_name] = "real_ir"
                else:
                    # ir=None bridge pattern — hollow but not a violation
                    # Only set to hollow_ir if not already real_ir
                    if self._provenance_status.get(tool_name) != "real_ir":
                        self._provenance_status[tool_name] = "hollow_ir"
        except Exception:
            pass  # Fail-open

    def get_category_distribution(self) -> Dict[str, int]:
        """Returns {category_name: invocation_count} dict."""
        # Simple dict copy — GIL-safe for read
        return dict(self._category_counts)

    def get_provenance_status(self) -> Dict[str, List[str]]:
        """
        Returns {status: [tool_names]} grouping.

        Statuses:
            real_ir: Tools with real AnalysisStep IR (fully tracked)
            hollow_ir: Tools with ir=None bridge (tracked but no notebook)
            missing: Provenance-required tools that never called log_tool_usage
        """
        with self._lock:
            result: Dict[str, List[str]] = {"real_ir": [], "hollow_ir": [], "missing": []}
            for tool_name, status in self._provenance_status.items():
                if status in result:
                    result[status].append(tool_name)
            return result

    def get_code_exec_log(self) -> list:
        """Returns snapshot of CODE_EXEC log entries (bounded deque copy)."""
        with self._lock:
            return list(self._code_exec_log)

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Structured summary for Omics-OS Cloud SSE enrichment.

        Acquires lock for consistent snapshot. O(1) to compute from counters.
        """
        with self._lock:
            provenance_status = {"real_ir": [], "hollow_ir": [], "missing": []}
            for tool_name, status in self._provenance_status.items():
                if status in provenance_status:
                    provenance_status[status].append(tool_name)

            return {
                "category_distribution": dict(self._category_counts),
                "provenance_status": provenance_status,
                "code_exec_count": len(self._code_exec_log),
                "code_exec_log": [
                    {
                        "tool_name": e.tool_name,
                        "timestamp": e.timestamp,
                        "agent": e.agent,
                    }
                    for e in self._code_exec_log
                ],
                "total_invocations": sum(self._tool_invocation_counts.values()),
            }
```

### Pattern 2: Building tool_metadata_map in graph.py

**What:** Collect all tool `.metadata` attributes at graph construction time into a flat dict.
**When to use:** After the single-pass agent creation loop — all tools exist and have `.metadata` assigned.

```python
# In create_bioinformatics_graph(), after the agent creation loop:
# Build tool_name → metadata lookup dict for AquadifMonitor (Path B)
# Do NOT rely on LangChain passing .tags/.metadata through on_tool_start kwargs
# (undocumented, fragile, breaks on LangChain upgrades — see brutalist review 2026-03-01)

tool_metadata_map: Dict[str, Dict[str, Any]] = {}

# Collect from all supervisor agent tools (handoff tools have DELEGATE metadata)
for agent_tool in agent_tools:
    if hasattr(agent_tool, "metadata"):
        tool_metadata_map[agent_tool.name] = agent_tool.metadata

# Collect from shared workspace tools
for t in [list_available_modalities, get_content_from_workspace,
          delete_from_workspace, execute_custom_code]:
    if hasattr(t, "metadata"):
        tool_metadata_map[t.name] = t.metadata

# Collect from all created agents' tools (walk PregelNode tool lists)
for agent_name, agent in created_agents.items():
    agent_tools_list = _extract_agent_tools(agent)  # see Pattern 3
    for t in agent_tools_list:
        if hasattr(t, "metadata"):
            tool_metadata_map[t.name] = t.metadata

# Initialize monitor (opt-in: None if not requested)
if aquadif_monitor is not None:
    aquadif_monitor._tool_metadata_map = tool_metadata_map
```

### Pattern 3: Extracting Tools from Compiled Agents (PregelNode traversal)

**What:** Walk compiled LangGraph agent to extract tool objects.
**When to use:** Building `tool_metadata_map` in `graph.py`.

```python
# Reuse existing pattern from contract_mixins.py (Phase 2 discovery)
# contract_mixins.py already solves PregelNode traversal:
#
# agent = factory_function(**kwargs)
# agent_node = agent.nodes.get("tools")
# if agent_node and hasattr(agent_node, "runnable"):
#     tools = agent_node.runnable.tools
#
# For graph.py, walk created_agents dict after creation:

def _extract_agent_tools(compiled_agent) -> list:
    """Extract tool list from a compiled PregelNode agent."""
    try:
        tools_node = compiled_agent.nodes.get("tools")
        if tools_node and hasattr(tools_node, "runnable"):
            return tools_node.runnable.tools or []
    except Exception:
        pass
    return []
```

### Pattern 4: Injecting Monitor into TokenTrackingCallback

**What:** Add optional `aquadif_monitor` to `TokenTrackingCallback.__init__`, call in `on_tool_start`.
**When to use:** The single wiring point — all other callbacks must NOT call the monitor.

```python
# lobster/utils/callbacks.py — TokenTrackingCallback modifications

class TokenTrackingCallback(BaseCallbackHandler):
    def __init__(
        self,
        session_id: str,
        pricing_config: Optional[Dict[str, Dict[str, float]]] = None,
        aquadif_monitor=None,  # Optional[AquadifMonitor] — avoid import cycle with TYPE_CHECKING
    ):
        # ... existing init ...
        self.aquadif_monitor = aquadif_monitor

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        # ... existing tool_name extraction and handoff detection ...
        tool_name = serialized.get("name", "unknown_tool") if serialized else "unknown_tool"
        self.current_tool = tool_name

        # Existing handoff detection ...

        # AQUADIF monitoring (fail-open, single injection point)
        if self.aquadif_monitor is not None:
            try:
                self.aquadif_monitor.record_tool_invocation(
                    tool_name=tool_name,
                    current_agent=self.current_agent or "unknown",
                )
            except Exception:
                pass  # Fail-open: never crash tool invocation
```

### Pattern 5: DataManagerV2 Provenance Hook

**What:** Single line addition inside `log_tool_usage`.
**When to use:** Called whenever a tool reports provenance — the authoritative observation point.

```python
# lobster/core/data_manager_v2.py — log_tool_usage modification

def log_tool_usage(
    self,
    tool_name: str,
    parameters: Dict[str, Any],
    description: str = None,
    ir: Optional["AnalysisStep"] = None,
) -> Optional[Dict[str, Any]]:
    # ... existing provenance tracking ...

    # AQUADIF monitoring hook (fail-open)
    if hasattr(self, "_aquadif_monitor") and self._aquadif_monitor is not None:
        try:
            self._aquadif_monitor.record_provenance_call(
                tool_name=tool_name,
                has_real_ir=(ir is not None),
            )
        except Exception:
            pass  # Fail-open

    # ... existing return logic ...
```

### Pattern 6: Wiring in client.py

**What:** Construct `AquadifMonitor` in `AgentClient.__init__`, pass through graph and DataManagerV2.

```python
# lobster/core/client.py — AgentClient.__init__ modifications

from lobster.core.aquadif_monitor import AquadifMonitor  # lazy import if needed

# After graph construction (tool_metadata_map is built inside create_bioinformatics_graph):
# Monitor is initialized with empty map; graph.py populates it after tool creation
self.aquadif_monitor = AquadifMonitor(tool_metadata_map={})

# Pass to TokenTrackingCallback (modify constructor call):
self.token_tracker = TokenTrackingCallback(
    session_id=self.session_id,
    pricing_config=MODEL_PRICING,
    aquadif_monitor=self.aquadif_monitor,
)

# Pass to graph (which populates tool_metadata_map and passes to DataManagerV2):
self.graph, self.graph_metadata = create_bioinformatics_graph(
    data_manager=self.data_manager,
    ...
    aquadif_monitor=self.aquadif_monitor,  # new kwarg
)

# DataManagerV2 gets monitor reference set by graph.py after tool map is built:
# graph.py: data_manager._aquadif_monitor = aquadif_monitor
```

### Anti-Patterns to Avoid

- **Multiple handlers calling monitor:** `TerminalCallbackHandler`, `TextualCallbackHandler`, and `StreamingCallbackHandler` must NOT call `monitor.record_tool_invocation()`. Cloud sessions have both `TokenTrackingCallback` and `StreamingCallbackHandler` active — double-counting would corrupt all metrics.
- **Relying on LangChain kwargs for tool metadata:** `on_tool_start` kwargs do NOT reliably contain `.tags` or `.metadata` from the tool object. Build `tool_metadata_map` at graph construction time (Path B).
- **Parsing output strings for provenance:** Tool output format is not a contract. Use `DataManagerV2.log_tool_usage` observation (authoritative, format-independent).
- **Unbounded data structures:** Never use plain `list` for the CODE_EXEC log — long sessions accumulate thousands of entries. Always use `deque(maxlen=100)`.
- **Module-level monitor instantiation:** `AquadifMonitor` is session-scoped, not module-scoped. Instantiate in `AgentClient.__init__`, not at import time.
- **Treating ir=None as a violation:** 20+ tools use ir=None as the bridge pattern from v1.0. Marking them as violations on day 1 creates dashboard noise. Track separately as `hollow_ir`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe counter increment | Custom atomic counter class | Plain dict with `threading.Lock` on compound ops | Single int increments are GIL-safe in CPython; only compound mutations need locking |
| Bounded log buffer | Custom ring buffer | `collections.deque(maxlen=N)` | stdlib, zero-overhead, thread-safe appends under GIL |
| Tool metadata discovery | Walk LangGraph internals at callback time | Build `tool_metadata_map` at graph construction time | LangGraph tool metadata is guaranteed available at construction; callback-time access is undocumented |
| Provenance detection | Parse tool output strings | Observe `DataManagerV2.log_tool_usage` | DM is the authoritative provenance record; output parsing is brittle |
| Monitor disable switch | Flag checks throughout | `aquadif_monitor=None` guard in callers | None check is one line; no performance overhead when disabled |

**Key insight:** The entire monitor is stdlib Python threading primitives. No new dependencies, no new concepts — just careful wiring of existing components.

---

## Common Pitfalls

### Pitfall 1: Import Cycle Between aquadif_monitor.py and callbacks.py

**What goes wrong:** `callbacks.py` imports `AquadifMonitor`; `aquadif_monitor.py` might import from `lobster.*` creating circular import.
**Why it happens:** `core/` and `utils/` both sit in the same package namespace.
**How to avoid:** Use `TYPE_CHECKING` for the type annotation in `callbacks.py`. Use `hasattr` checks or late binding rather than structural imports. `aquadif_monitor.py` should be a pure-stdlib module with zero lobster imports.
**Warning signs:** `ImportError: cannot import name 'AquadifMonitor'` at startup.

### Pitfall 2: Double-Counting in Cloud Sessions

**What goes wrong:** Both `TokenTrackingCallback` and `StreamingCallbackHandler` receive all callbacks. If `StreamingCallbackHandler.on_tool_start` is modified to call the monitor, every tool invocation is counted twice.
**Why it happens:** Cloud `AgentClient` adds both callbacks to `self.callbacks` list. Both receive every event.
**How to avoid:** Only `TokenTrackingCallback.on_tool_start` calls `monitor.record_tool_invocation()`. Add a comment in `StreamingCallbackHandler.on_tool_start` explicitly noting it must NOT call the monitor.
**Warning signs:** Category counts double the expected values in cloud sessions.

### Pitfall 3: tool_metadata_map Missing Tools

**What goes wrong:** Not all tools are collected — especially shared workspace tools (`list_available_modalities`, `execute_custom_code`) and tools from child agents.
**Why it happens:** The map-building loop only iterates `agent_tools` (supervisor-accessible handoff tools), missing tools inside child agents.
**How to avoid:** Use `_extract_agent_tools()` (PregelNode traversal) on every entry in `created_agents`, not just `agent_tools`. Also explicitly include the 4 shared workspace tools.
**Warning signs:** `record_tool_invocation` calls with `tool_metadata_map.get(tool_name, {})` returning `{}` for known tools.

### Pitfall 4: DataManagerV2 Monitor Reference Not Set

**What goes wrong:** `DataManagerV2` is constructed in `client.py` BEFORE `graph.py` runs, so the monitor doesn't exist yet at DM construction time.
**Why it happens:** The construction order is: `DataManagerV2.__init__` → `create_bioinformatics_graph()`.
**How to avoid:** Use a setattr pattern (`data_manager._aquadif_monitor = aquadif_monitor`) inside `create_bioinformatics_graph()` after monitor is initialized, rather than passing it to `DataManagerV2.__init__`. This avoids changing `DataManagerV2`'s constructor signature.
**Warning signs:** `record_provenance_call` is never called even when tools run.

### Pitfall 5: threading.Lock Used in Non-Thread-Safe Way

**What goes wrong:** Acquiring lock in one method while another path also holds the lock causes deadlock.
**Why it happens:** `get_session_summary()` acquires lock and iterates `_provenance_status`; if `record_provenance_call` is called from the same thread while `get_session_summary` holds the lock, deadlock occurs (only if using non-reentrant lock).
**How to avoid:** Keep lock-held sections short. `threading.Lock` (not `RLock`) is correct — none of these methods call each other. Compound state mutations need the lock; simple counter dict increments do not.
**Warning signs:** Application hangs during session queries.

### Pitfall 6: Monitor Construction Order

**What goes wrong:** `AquadifMonitor(tool_metadata_map={})` is constructed before tools exist, but needs to be populated after graph construction.
**Why it happens:** `tool_metadata_map` is only fully known after the single-pass agent creation loop in `graph.py`.
**How to avoid:** Construct monitor with `{}`, then call `graph.py` which calls `aquadif_monitor._tool_metadata_map = tool_metadata_map` after building the map. The monitor is already attached to `TokenTrackingCallback` and `DataManagerV2` by reference — so the populated map is immediately visible to all holders.
**Warning signs:** `category_counts` stays empty even after tool invocations.

---

## Code Examples

Verified patterns from codebase inspection:

### Existing on_tool_start in TokenTrackingCallback (modification target)

```python
# Source: lobster/utils/callbacks.py:1002-1028 (verified)
def on_tool_start(
    self, serialized: Dict[str, Any], input_str: str, **kwargs
) -> None:
    if serialized is None:
        serialized = {}
    tool_name = serialized.get("name", "unknown_tool")
    self.current_tool = tool_name

    # Detect agent handoffs (same pattern as TerminalCallbackHandler)
    if tool_name.startswith("handoff_to_"):
        target = tool_name.replace("handoff_to_", "")
        if self._is_valid_agent(target):
            self.current_agent = target
    elif tool_name == "transfer_back_to_supervisor":
        self.current_agent = "supervisor"
    # ← INSERT: self.aquadif_monitor.record_tool_invocation(tool_name, self.current_agent or "unknown")
```

### Existing log_tool_usage in DataManagerV2 (modification target)

```python
# Source: lobster/core/data_manager_v2.py:1728-1773 (verified)
def log_tool_usage(
    self,
    tool_name: str,
    parameters: Dict[str, Any],
    description: str = None,
    ir: Optional["AnalysisStep"] = None,
) -> Optional[Dict[str, Any]]:
    if self.provenance:
        activity_id = self.provenance.create_activity(
            activity_type=tool_name,
            agent="data_manager",
            parameters=parameters,
            description=description or f"{tool_name} operation",
            ir=ir,
        )
        # ← INSERT: if hasattr(self, "_aquadif_monitor") and self._aquadif_monitor:
        #               self._aquadif_monitor.record_provenance_call(tool_name, has_real_ir=(ir is not None))
        ...
    return None
```

### AQUADIF metadata already on all tools (confirmed Phase 4)

```python
# Source: lobster/agents/graph.py:277-278 (verified - delegation tools)
invoke_agent_lazy.metadata = {"categories": ["DELEGATE"], "provenance": False}
invoke_agent_lazy.tags = ["DELEGATE"]

# Example from packages (all 221 tools have this pattern):
analyze_modality.metadata = {"categories": ["ANALYZE"], "provenance": True}
analyze_modality.tags = ["ANALYZE"]
```

### Existing PregelNode traversal in contract_mixins.py (reuse for tool extraction)

```python
# Source: lobster/testing/contract_mixins.py (verified — used in Phase 2-4)
# When extracting tools from a compiled agent:
agent_node = compiled_agent.nodes.get("tools")
if agent_node and hasattr(agent_node, "runnable"):
    tools = agent_node.runnable.tools  # List of tool objects with .metadata
```

### get_usage_summary() pattern to mirror for get_session_summary()

```python
# Source: lobster/utils/callbacks.py:1226-1262 (verified)
def get_usage_summary(self) -> Dict[str, Any]:
    return {
        "session_id": self.session_id,
        "total_input_tokens": self.total_input_tokens,
        "by_agent": {...},
        "invocations": [...],
    }
# ← AquadifMonitor.get_session_summary() follows same O(1) pattern
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Parse output strings for provenance | Observe DataManagerV2.log_tool_usage | Brutalist review 2026-03-01 | Correct observation vs fragile guess |
| New callback handler per concern | Single injection via existing always-on callback | Brutalist review 2026-03-01 | No double-counting in cloud sessions |
| Rely on LangChain kwargs for .tags | Build lookup dict at graph construction time | Brutalist review 2026-03-01 | Stable across LangChain upgrades |
| ir=None as violation | ir=None as "hollow_ir" status | Brutalist review 2026-03-01 | No dashboard noise on day 1 |

**No deprecated approaches to worry about for this phase** — the architecture was designed fresh.

---

## Open Questions

1. **`create_bioinformatics_graph()` signature extension**
   - What we know: Function already accepts 12 kwargs; adding `aquadif_monitor: Optional[AquadifMonitor] = None` is clean
   - What's unclear: Whether `client.py` constructs the monitor before or passes it to graph for construction
   - Recommendation: Construct in `client.py` with empty map, pass to graph, have graph populate map and set on DM

2. **get_session_summary() call site in lobster-cloud**
   - What we know: `StreamingCallbackHandler` in `lobster-cloud/api/callbacks/streaming_callback.py` handles SSE streaming; `get_usage_summary()` is currently called from `token_usage_panel.py`
   - What's unclear: Exactly where in `chat_stream.py` the session summary should be appended as a final SSE event
   - Recommendation: Phase 5 PLAN should scope this to lobster-core only; cloud integration can be a separate plan (MON-07 candidate) or included in Plan 2

3. **Whether to expose AquadifMonitor on AgentClient publicly**
   - What we know: `client.py` will have `self.aquadif_monitor`; cloud adapter uses `AgentClient`
   - What's unclear: Whether lobster-cloud accesses it via `client.aquadif_monitor.get_session_summary()` or via a separate mechanism
   - Recommendation: Expose as public attribute on `AgentClient` — clean, testable, follows `self.token_tracker` pattern

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` — skipping this section.

---

## Sources

### Primary (HIGH confidence)

- `lobster/utils/callbacks.py` — `TokenTrackingCallback` class, `on_tool_start` at line 1002, `get_usage_summary` at line 1226 (direct code inspection)
- `lobster/core/data_manager_v2.py` — `log_tool_usage` at line 1728 (direct code inspection)
- `lobster/agents/graph.py` — `create_bioinformatics_graph()`, `_create_lazy_delegation_tool()` with DELEGATE metadata at line 277 (direct code inspection)
- `lobster/config/aquadif.py` — `AquadifCategory` enum, `PROVENANCE_REQUIRED` frozenset (direct code inspection)
- `lobster/testing/contract_mixins.py` — PregelNode traversal pattern (direct code inspection)
- `lobster/core/client.py` — `AgentClient.__init__`, callback construction at line 131 (direct code inspection)
- `lobster-cloud/api/callbacks/streaming_callback.py` — `StreamingCallbackHandler` — SSE consumer (direct code inspection)
- `.planning/ROADMAP.md` — Phase 5 architecture (locked design from brutalist review)

### Secondary (MEDIUM confidence)

- `StreamingCallbackHandler` docstring: "callbacks are triggered from worker threads when query() runs in a ThreadPoolExecutor" — confirms `threading.Lock` (not `asyncio.Lock`) is correct

### Tertiary (LOW confidence)

- None — all findings backed by direct source inspection

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — stdlib only, codebase confirms no new dependencies needed
- Architecture: HIGH — all integration points directly verified in source code; design locked by brutalist review
- Pitfalls: HIGH — identified from direct analysis of construction order, callback registration, and existing dual-callback cloud architecture
- Code examples: HIGH — all snippets extracted from verified source locations with line numbers

**Research date:** 2026-03-01
**Valid until:** 2026-04-01 (stable stdlib patterns; LangGraph API is fast-moving but we're using Path B which avoids LangGraph internals)
