---
phase: 05-monitoring-infrastructure
plan: 01
subsystem: monitoring
tags: [aquadif, monitoring, threading, stdlib, tdd, introspection]

# Dependency graph
requires:
  - phase: 04-agent-rollout
    provides: All 221 tools tagged with .metadata and .tags (AQUADIF-compliant)
provides:
  - AquadifMonitor service class with 6 public methods for runtime introspection
  - CodeExecEntry dataclass for CODE_EXEC bounded log entries
  - lobster/core/aquadif_monitor.py — pure stdlib, zero lobster imports
  - 55 unit tests covering all methods, thread safety, fail-open, edge cases
affects:
  - 05-02-PLAN.md (callback chain wiring — imports AquadifMonitor from this plan)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "AquadifMonitor as pure-stdlib service with zero lobster imports (prevents import cycles)"
    - "threading.Lock for compound mutations; GIL-safe single dict increments without lock"
    - "collections.deque(maxlen=100) for bounded CODE_EXEC log (auto-eviction)"
    - "Fail-open public methods (try/except swallows, never re-raises)"
    - "real_ir wins over hollow_ir (non-downgrade rule for provenance status)"

key-files:
  created:
    - lobster/core/aquadif_monitor.py
    - tests/unit/core/test_aquadif_monitor.py
  modified: []

key-decisions:
  - "AquadifMonitor uses zero lobster.* imports to prevent circular import when callbacks.py imports it"
  - "threading.Lock only on compound mutations (deque append + dict update); single dict increments are GIL-safe in CPython"
  - "deque(maxlen=100) for CODE_EXEC log — auto-evicts, no memory growth in long sessions"
  - "real_ir status cannot be downgraded to hollow_ir (non-downgrade rule)"
  - "record_tool_invocation only pre-sets provenance_status to 'missing' on first invocation — subsequent calls do not overwrite real_ir or hollow_ir"

patterns-established:
  - "Pattern 1: Pure-stdlib service class with ZERO lobster imports to prevent circular imports"
  - "Pattern 2: Fail-open public methods — try/except swallows all exceptions silently"
  - "Pattern 3: Bounded deque for any log that could grow unboundedly in a session"
  - "Pattern 4: Separate lock for compound mutations vs GIL-safe single increments"

requirements-completed: [MON-01, MON-05, MON-06]

# Metrics
duration: 3min
completed: 2026-03-01
---

# Phase 05 Plan 01: AquadifMonitor Service Class Summary

**Thread-safe stdlib AquadifMonitor service with 55 unit tests — tracks category distribution, provenance status (real_ir/hollow_ir/missing), and bounded CODE_EXEC log for runtime AQUADIF introspection**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-01T09:28:23Z
- **Completed:** 2026-03-01T09:31:17Z
- **Tasks:** 1 TDD feature (RED + GREEN; no refactor needed)
- **Files modified:** 2

## Accomplishments

- `AquadifMonitor` service class with 6 public methods: `record_tool_invocation`, `record_provenance_call`, `get_category_distribution`, `get_provenance_status`, `get_code_exec_log`, `get_session_summary`
- `CodeExecEntry` dataclass with `tool_name`, `timestamp` (ISO), `agent` fields
- 55 unit tests: category counting, provenance state machine, CODE_EXEC log, thread safety (100-thread concurrency), fail-open, edge cases, and no-lobster-imports enforcement
- Pure stdlib implementation (threading, collections.deque, dataclasses, datetime) — zero new dependencies

## Task Commits

TDD process produced 2 commits:

1. **RED — Failing tests** - `68a88b1` (test)
2. **GREEN — Implementation** - `fcae6e9` (feat)

**Plan metadata:** (this SUMMARY + STATE.md update commit)

_Note: No refactor phase needed — implementation was clean first pass._

## Files Created/Modified

- `lobster/core/aquadif_monitor.py` - AquadifMonitor service class + CodeExecEntry dataclass (pure stdlib, 269 lines)
- `tests/unit/core/test_aquadif_monitor.py` - 55 unit tests across 10 test classes (554 lines)

## Decisions Made

- **Zero lobster imports:** `aquadif_monitor.py` has no `from lobster.*` or `import lobster` to prevent circular imports when `callbacks.py` (Plan 02 target) imports it. Enforced by a test.
- **Lock scope:** `threading.Lock` only wraps compound mutations (deque append + dict update). Single `dict.get(k, 0) + 1` increments are GIL-safe in CPython and run without lock.
- **real_ir non-downgrade rule:** Once a tool achieves `real_ir` status, calling `record_provenance_call(has_real_ir=False)` is a no-op. This handles race conditions where `ir=None` tools might be logged after a real IR is already recorded.
- **Pre-set to "missing" only on first invocation:** `record_tool_invocation` only sets provenance_status to `"missing"` if the tool is not already in the dict. This ensures `real_ir`/`hollow_ir` set by `record_provenance_call` isn't overwritten on the tool's second invocation.
- **deque maxlen=100:** Bounds CODE_EXEC log memory. Long sessions with frequent custom code execution won't accumulate thousands of entries. Oldest entry is auto-evicted.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `AquadifMonitor` class is ready for wiring into the callback chain (Plan 02)
- Import path: `from lobster.core.aquadif_monitor import AquadifMonitor, CodeExecEntry`
- Plan 02 targets: `lobster/utils/callbacks.py` (TokenTrackingCallback injection), `lobster/agents/graph.py` (tool_metadata_map construction), `lobster/core/data_manager_v2.py` (provenance hook), `lobster/core/client.py` (orchestration)
- No blockers.

---
*Phase: 05-monitoring-infrastructure*
*Completed: 2026-03-01*
