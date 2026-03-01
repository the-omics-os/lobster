---
phase: 05-monitoring-infrastructure
plan: "02"
subsystem: monitoring
tags: [aquadif, monitoring, callbacks, wiring, runtime]
dependency_graph:
  requires: ["05-01"]
  provides: ["AquadifMonitor runtime wiring", "tool_metadata_map construction"]
  affects: ["lobster/core/client.py", "lobster/agents/graph.py", "lobster/utils/callbacks.py", "lobster/core/data_manager_v2.py"]
tech_stack:
  added: []
  patterns: ["setattr injection", "fail-open try/except", "single injection point", "PregelNode tool traversal"]
key_files:
  created: []
  modified:
    - lobster/core/client.py
    - lobster/agents/graph.py
    - lobster/utils/callbacks.py
    - lobster/core/data_manager_v2.py
decisions:
  - "Single injection point in TokenTrackingCallback.on_tool_start — Terminal/Streaming/Simple handlers never call monitor to prevent double-counting in cloud sessions"
  - "AquadifMonitor constructed with empty tool_metadata_map in client.py; graph.py populates it after all tool objects exist"
  - "setattr pattern (data_manager._aquadif_monitor = ...) avoids changing DataManagerV2.__init__ signature"
  - "token_tracker.aquadif_monitor = self.aquadif_monitor attribute assignment avoids changing TokenTrackingCallback constructor signature"
  - "hasattr guard in data_manager_v2.log_tool_usage provides zero-overhead opt-out for instances without monitor"
  - "record_provenance_call hook placed inside if self.provenance block — only fires when provenance tracking is active"
  - "PregelNode tool traversal via agent.nodes.get('tools').runnable.tools in graph.py — fail-open for agents with non-standard node structure"
metrics:
  duration: "~8 minutes"
  completed: "2026-03-01"
  tasks_completed: 2
  files_modified: 4
  tests_run: 223
  tests_passing: 223
---

# Phase 5 Plan 02: AquadifMonitor Callback Chain Wiring Summary

**One-liner:** AquadifMonitor wired into runtime via single injection point in TokenTrackingCallback, with tool_metadata_map built by graph.py PregelNode traversal and provenance call observation in DataManagerV2.

## What Was Built

Plan 02 connects the AquadifMonitor service class (created in Plan 01) to the live runtime system through 4 surgical file modifications:

**client.py** — Construction and injection entry point:
- Constructs `AquadifMonitor(tool_metadata_map={})` immediately after `TokenTrackingCallback` setup
- Sets `self.token_tracker.aquadif_monitor = self.aquadif_monitor` (attribute assignment, no constructor change)
- Passes `aquadif_monitor=self.aquadif_monitor` to the primary `create_bioinformatics_graph()` call

**callbacks.py** — Single injection point for tool invocation tracking:
- Added `self.aquadif_monitor = None` to `TokenTrackingCallback.__init__` (default opt-out)
- Added fail-open `record_tool_invocation()` call in `on_tool_start` (after handoff detection)
- No changes to `TerminalCallbackHandler`, `StreamingTerminalCallback`, or `SimpleTerminalCallback` — prevents double-counting in cloud sessions where multiple handlers run concurrently

**graph.py** — Tool metadata map construction and DataManagerV2 wiring:
- Added `aquadif_monitor=None` parameter to `create_bioinformatics_graph()` signature
- After graph compilation: traverses `agent_tools` (DELEGATE tools), shared workspace tools (`list_available_modalities`, `get_content_from_workspace`, `delete_from_workspace`, `execute_custom_code`), and all child agents via `agent.nodes.get("tools").runnable.tools` PregelNode traversal
- Populates `aquadif_monitor._tool_metadata_map` with the built map
- Sets `data_manager._aquadif_monitor = aquadif_monitor` via setattr (no DataManagerV2 constructor change)

**data_manager_v2.py** — Provenance observation hook:
- Added fail-open `record_provenance_call(tool_name, has_real_ir=(ir is not None))` call inside the `if self.provenance:` block of `log_tool_usage`
- Uses `hasattr(self, "_aquadif_monitor")` guard so DataManagerV2 instances without the monitor attribute have zero overhead

## Verification Results

All tests pass with zero modifications to existing test files:

| Test Suite | Count | Result |
|------------|-------|--------|
| AquadifMonitor unit tests | 55 | PASS |
| DataManagerV2 tests | ~120 | PASS |
| Client tests | ~48 | PASS |
| AQUADIF contract tests | 26 | PASS |
| **Total** | **223** | **PASS** |

Verified:
- Import chain: `AquadifMonitor` → `callbacks.py` → no circular import (zero lobster.* imports in aquadif_monitor.py)
- Single injection point: only 3 lines in `TokenTrackingCallback` reference `aquadif_monitor` (init + condition + call)
- Monitor opt-in: existing tests that don't set aquadif_monitor work because `self.aquadif_monitor = None` default means zero overhead
- Pre-existing `test_extraction_with_nested_structure` failure confirmed pre-existing (not caused by this plan)

## Deviations from Plan

None - plan executed exactly as written.

The plan specified attribute-set pattern for both the callback and DataManagerV2 — these were implemented as designed. The graph.py PregelNode traversal code matched the plan's `agent.nodes.get("tools")` pattern exactly.

## Key Design Decisions

**Single injection point** (critical for correctness): `TokenTrackingCallback` is the only callback that fires once per tool invocation across both local and cloud deployments. Terminal/Streaming handlers may fire multiple times or not at all depending on the display mode. Using only `TokenTrackingCallback` ensures exactly-once counting per tool call.

**Empty map construction + lazy population**: The monitor is constructed with `tool_metadata_map={}` in `client.py` before graph creation. After `create_bioinformatics_graph()` completes, graph.py populates `aquadif_monitor._tool_metadata_map` with all discovered tools. This works because the monitor reference is shared — setting `_tool_metadata_map` on the shared object is immediately visible from `callbacks.py`.

**No constructor signature changes**: Both `TokenTrackingCallback.__init__` and `DataManagerV2.__init__` are left untouched. The attribute-set pattern preserves backward compatibility for all existing callers.

## Self-Check: PASSED

Files verified to exist:
- `lobster/core/aquadif_monitor.py` — FOUND (created in Plan 01)
- `lobster/core/client.py` — FOUND, modified
- `lobster/agents/graph.py` — FOUND, modified
- `lobster/utils/callbacks.py` — FOUND, modified
- `lobster/core/data_manager_v2.py` — FOUND, modified

Commits verified:
- `777af3d` — feat(05-02): wire AquadifMonitor into callback chain

## Self-Check: PASSED

All files found and commit verified. 223/223 tests passing, 0 regressions.
