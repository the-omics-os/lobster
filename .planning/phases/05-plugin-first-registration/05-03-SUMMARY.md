---
phase: 05-plugin-first-registration
plan: 03
subsystem: plugin-architecture
tags: [entry-points, plugin-registration, download-services, queue-preparers, fallback-gating]

# Dependency graph
requires:
  - phase: 05-02
    provides: pyproject.toml entry-point declarations for queue_preparers and download_services
  - phase: 05-01
    provides: TDD RED scaffold for TestFallbackGating and TestEntryPointDiscovery
provides:
  - _ALLOW_HARDCODED_FALLBACK = False in queue_preparation_service.py (PLUG-06)
  - _ALLOW_HARDCODED_FALLBACK = False in download_orchestrator.py (PLUG-06)
  - Hardcoded Phase 2 fallback gated behind module-level constant in both routers
  - TestFallbackGating GREEN in both test files (4 new tests)
  - TestDefaultRegistration + TestDownloadOrchestrator updated for 5 databases (PLUG-05)
affects:
  - phase 06 (any phase building on plugin-first architecture)
  - external documentation on router initialization behavior

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Module-level boolean constant gate pattern for opt-in emergency fallback
    - discovered_names set tracking during entry-point Phase 1 loop
    - Warning log when no entry points found and fallback disabled

key-files:
  created: []
  modified:
    - lobster/services/data_access/queue_preparation_service.py
    - lobster/tools/download_orchestrator.py
    - tests/unit/services/data_access/test_queue_preparation_service.py
    - tests/unit/services/data_access/test_download_services.py

key-decisions:
  - "[Phase 05]: _ALLOW_HARDCODED_FALLBACK gate uses early return (not complex flag-check logic) — simple semantics per RESEARCH.md anti-pattern note"
  - "[Phase 05]: discovered_names set tracked in Phase 1 loop enables warning log only when EP discovery yields zero results"
  - "[Phase 05]: Phase 2 hardcoded block remains structurally intact — never deleted, only unreachable when flag is False (emergency recovery preserved)"

patterns-established:
  - "Fallback gating: module-level _ALLOW_HARDCODED_FALLBACK = False + early return before Phase 2 block"
  - "EP tracking: discovered_names: set = set() initialized before loop, add(name) on success"
  - "5-DB assertions: tests now assert geo, sra, pride, massive, metabolights via entry-point path"

requirements-completed: [PLUG-01, PLUG-02, PLUG-05, PLUG-06]

# Metrics
duration: 15min
completed: 2026-03-04
---

# Phase 05 Plan 03: Fallback Gating SUMMARY

**_ALLOW_HARDCODED_FALLBACK = False in both routers gates hardcoded Phase 2 block, completing plugin-first-registration with all TestFallbackGating tests GREEN and 5-DB assertions validating the entry-point path**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-04T07:42:00Z
- **Completed:** 2026-03-04T07:56:29Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `_ALLOW_HARDCODED_FALLBACK = False` module-level constant to `queue_preparation_service.py` and `download_orchestrator.py`
- Modified `_register_default_preparers` and `_register_default_services` to track `discovered_names` set during Phase 1 entry-point loop, then gate Phase 2 with early return when flag is False
- Warning log emitted when no entry points discovered and fallback is disabled — prevents silent no-op
- Updated `TestDefaultRegistration` to assert all 5 databases (added `metabolights`)
- Updated `TestDownloadOrchestrator.test_auto_registration` to assert all 5 databases (added `sra` + `metabolights`)
- All 20 plan-specific tests GREEN: TestFallbackGating(4) + TestEntryPointDiscovery(4) + TestDefaultRegistration(2) + TestDownloadOrchestrator(6) + TestPluginRegistrationContract(4)

## Task Commits

Each task was committed atomically:

1. **Task 1: Gate hardcoded fallback in both routers** - `37aca5d` (feat)
2. **Task 2: Update registration tests for 5 databases** - `3ec9808` (feat)

**Plan metadata:** (docs commit — see final commit below)

## Files Created/Modified

- `lobster/services/data_access/queue_preparation_service.py` — Added `_ALLOW_HARDCODED_FALLBACK = False` constant; modified `_register_default_preparers` with `discovered_names` tracking and early-return gate before Phase 2; updated docstring
- `lobster/tools/download_orchestrator.py` — Added `_ALLOW_HARDCODED_FALLBACK = False` constant; modified `_register_default_services` with `discovered_names` tracking and early-return gate before Phase 2; updated docstring
- `tests/unit/services/data_access/test_queue_preparation_service.py` — `TestDefaultRegistration`: added `metabolights` assertion, count `>= 5`, updated docstring
- `tests/unit/services/data_access/test_download_services.py` — `TestDownloadOrchestrator.test_auto_registration`: added `sra` + `metabolights` assertions, updated comment to "all 5 databases"

## Decisions Made

- Used simple early-return gate (`if not _ALLOW_HARDCODED_FALLBACK: ... return`) rather than complex conditional logic — per RESEARCH.md recommendation to avoid "yields nothing fallback" anti-pattern
- `discovered_names` set is local to `_register_default_preparers`/`_register_default_services` — only used for the warning log, no other side effects
- Phase 2 hardcoded block intentionally preserved intact for emergency recovery (set `_ALLOW_HARDCODED_FALLBACK = True` in dev only)

## Deviations from Plan

None — plan executed exactly as written. The TDD-tagged Task 1 had the GREEN target tests already written in Plan 01; implementation directly turned them GREEN.

## Issues Encountered

None. The 8 pre-existing failures in `test_content_access_service.py::TestRealAPI*` are live-API tests that fail without network access — unrelated to this plan's changes (scope boundary rule applied, logged to context).

## Next Phase Readiness

Phase 05 (Plugin-First Registration) is now COMPLETE:
- PLUG-01: TestEntryPointDiscovery GREEN for QueuePreparationService
- PLUG-02: TestEntryPointDiscovery GREEN for DownloadOrchestrator
- PLUG-03: TestPluginRegistrationContract GREEN (entry-point contract)
- PLUG-04: TestPluginRegistrationContract GREEN (load validation)
- PLUG-05: TestDefaultRegistration + TestDownloadOrchestrator assert 5 DBs
- PLUG-06: TestFallbackGating GREEN in both files — hardcoded fallback gated

No blockers for subsequent phases. The plugin-first architecture is fully validated and the hardcoded fallback exists only as an emergency escape hatch.

## Self-Check: PASSED

- FOUND: .planning/phases/05-plugin-first-registration/05-03-SUMMARY.md
- FOUND: lobster/services/data_access/queue_preparation_service.py
- FOUND: lobster/tools/download_orchestrator.py
- FOUND: commit 37aca5d (Task 1: fallback gating)
- FOUND: commit 3ec9808 (Task 2: 5-DB test assertions)

---
*Phase: 05-plugin-first-registration*
*Completed: 2026-03-04*
