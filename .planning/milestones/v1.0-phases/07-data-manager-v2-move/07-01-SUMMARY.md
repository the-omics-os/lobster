---
phase: 07-data-manager-v2-move
plan: 01
subsystem: core
tags: [refactoring, data-manager, shim, backward-compat, mock-patch]

# Dependency graph
requires:
  - phase: 06-core-subpackage-creation-moves
    provides: runtime/ subpackage structure and proven move-and-shim pattern
provides:
  - DataManagerV2 at canonical path lobster.core.runtime.data_manager
  - Backward-compat shim at lobster.core.data_manager_v2 with DeprecationWarning
  - All ~114 mock.patch strings updated to target canonical module
affects: [07-02, 08-cli-decomposition]

# Tech tracking
tech-stack:
  added: []
  patterns: [move-and-shim for high-blast-radius files, mechanical mock.patch migration]

key-files:
  created:
    - lobster/core/data_manager_v2.py (shim)
  modified:
    - lobster/core/runtime/data_manager.py (moved from core/data_manager_v2.py)
    - lobster/core/runtime/__init__.py
    - tests/unit/core/test_core_subpackage_shims.py
    - tests/unit/core/test_data_manager_v2.py
    - tests/unit/agents/test_agent_registry.py

key-decisions:
  - "Internal imports in moved file updated to canonical paths immediately (provenance, queues, runtime)"
  - "Import statements in test files left on shim path -- shim handles them transparently"
  - "2 pre-existing KeyError failures on validation key accepted as out-of-scope"

patterns-established:
  - "High-blast-radius move: git mv + shim + mechanical mock.patch update"

requirements-completed: [DMGR-01, DMGR-02]

# Metrics
duration: 5min
completed: 2026-03-04
---

# Phase 7 Plan 01: data_manager_v2 Move Summary

**Moved 3,999 LOC DataManagerV2 to core/runtime/data_manager.py with backward-compat shim and 114 mock.patch updates**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-04T18:10:20Z
- **Completed:** 2026-03-04T18:15:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- DataManagerV2 (3,999 LOC, ~200 importers) moved to canonical path with git history preserved
- Backward-compat shim emits DeprecationWarning and re-exports all names
- 114 mock.patch strings updated from old path to canonical path across 2 test files
- Shim test suite extended to 14 pairs (28 tests: re-export + isinstance identity)
- 96 tests pass, 5 skipped, 2 pre-existing failures unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Move data_manager_v2.py to core/runtime/data_manager.py and create shim** - `4b4be51` (feat)
2. **Task 2: Update mock.patch strings in test files to target canonical module path** - `d03cc43` (fix)

## Files Created/Modified
- `lobster/core/runtime/data_manager.py` - DataManagerV2 at canonical location (moved from core/data_manager_v2.py)
- `lobster/core/data_manager_v2.py` - Backward-compat shim with DeprecationWarning
- `lobster/core/runtime/__init__.py` - Updated docstring to include data manager
- `tests/unit/core/test_core_subpackage_shims.py` - Extended to 14 shim pairs
- `tests/unit/core/test_data_manager_v2.py` - 112 mock.patch strings updated to canonical path
- `tests/unit/agents/test_agent_registry.py` - 2 mock.patch strings updated to canonical path

## Decisions Made
- Internal imports in the moved file updated to canonical paths immediately (avoids spurious DeprecationWarnings from shims)
- Import statements in test files left targeting the shim path (shim handles them transparently)
- 2 pre-existing KeyError failures on `validation` key accepted as out-of-scope (pre-date this move)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- 2 pre-existing test failures (TestExportDocumentation::test_store_and_retrieve_metadata, TestGEOMetadataStorage::test_store_geo_metadata_with_optional_fields) with `KeyError: 'validation'` -- these are pre-existing from earlier phases and not caused by this move

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Canonical import path established for DataManagerV2
- Plan 07-02 can proceed: update scaffold templates, import-linter, CI deprecated-import check
- All ~200 source importers work transparently via shim (no source updates needed)

## Self-Check: PASSED

All artifacts verified:
- [x] lobster/core/runtime/data_manager.py exists
- [x] lobster/core/data_manager_v2.py (shim) exists
- [x] 07-01-SUMMARY.md exists
- [x] Commit 4b4be51 exists
- [x] Commit d03cc43 exists

---
*Phase: 07-data-manager-v2-move*
*Completed: 2026-03-04*
