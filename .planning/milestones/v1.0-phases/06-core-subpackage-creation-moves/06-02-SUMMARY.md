---
phase: 06-core-subpackage-creation-moves
plan: 02
subsystem: core
tags: [refactoring, subpackages, shims, backward-compat, deprecation, provenance, notebooks]

# Dependency graph
requires:
  - phase: 06-01
    provides: 3 subpackages (governance, queues, runtime), shim pattern, test scaffold
provides:
  - 2 new subpackages (notebooks/, provenance/) completing all 5 core subpackages
  - 7 files moved to canonical locations with backward-compatible shims
  - __getattr__ lazy shim pattern for module/package name collisions
  - Import-linter layers contract for subpackage dependency ordering
affects: [07-data-manager-move, core-imports]

# Tech tracking
tech-stack:
  added: []
  patterns: [__getattr__ lazy shim for module/package name collision, file rename on move (notebook_executor -> executor)]

key-files:
  created:
    - lobster/core/notebooks/__init__.py
    - lobster/core/provenance/__init__.py
    - lobster/core/notebooks/executor.py
    - lobster/core/notebooks/exporter.py
    - lobster/core/notebooks/validator.py
    - lobster/core/provenance/analysis_ir.py
    - lobster/core/provenance/provenance.py
    - lobster/core/provenance/lineage.py
    - lobster/core/provenance/ir_coverage.py
  modified:
    - lobster/core/notebook_executor.py (now shim)
    - lobster/core/notebook_exporter.py (now shim)
    - lobster/core/notebook_validator.py (now shim)
    - lobster/core/analysis_ir.py (now shim)
    - lobster/core/lineage.py (now shim)
    - lobster/core/ir_coverage.py (now shim)
    - .importlinter
    - tests/unit/core/test_core_subpackage_shims.py

key-decisions:
  - "__getattr__ lazy shim in provenance/__init__.py -- fires only on old-style attribute access, not on new-style submodule imports"
  - "Notebook files renamed on move (notebook_executor.py -> executor.py) for cleaner subpackage naming"
  - "exporter.py internal imports updated to canonical paths immediately (not deferred)"
  - "Test hasattr() moved inside catch_warnings block to capture __getattr__ lazy shim warnings"

patterns-established:
  - "__getattr__ lazy shim: for module-to-package promotions where the old module name collides with the new package name"
  - "Import-linter layers contract: notebooks > provenance > governance > queues > runtime"

requirements-completed: [CORE-01, CORE-02, CORE-03, CORE-04, CORE-05]

# Metrics
duration: 4min
completed: 2026-03-04
---

# Phase 06 Plan 02: Notebooks & Provenance Subpackage Moves Summary

**7 files moved to notebooks/ and provenance/ subpackages with __getattr__ lazy shim resolving provenance module/package name collision, all 31 shim tests passing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-04T17:32:24Z
- **Completed:** 2026-03-04T17:36:38Z
- **Tasks:** 2
- **Files modified:** 17

## Accomplishments
- Moved 4 provenance files to core/provenance/ with __getattr__ lazy shim resolving the module/package name collision
- Moved 3 notebook files to core/notebooks/ with file rename (notebook_executor -> executor, etc.)
- Updated exporter.py internal imports to canonical paths (analysis_ir, provenance)
- Created 6 standard shims + 1 __getattr__ lazy shim at old paths
- Updated import-linter with canonical download_queue path + new subpackage layers contract
- All 31 tests pass: 13 re-export, 13 isinstance identity, 5 subpackage existence

## Task Commits

Each task was committed atomically:

1. **Task 1: Move provenance/ (4 files) with __getattr__ lazy shim** - `1cf1f54` (feat)
2. **Task 2: Move notebooks/ (3 files with rename), update import-linter, run full verification** - `47b7898` (feat)

## Files Created/Modified
- `lobster/core/provenance/__init__.py` - Provenance subpackage with __getattr__ lazy shim
- `lobster/core/provenance/provenance.py` - Moved W3C-PROV tracking (canonical)
- `lobster/core/provenance/analysis_ir.py` - Moved analysis IR (125 importers, canonical)
- `lobster/core/provenance/lineage.py` - Moved lineage tracking (canonical)
- `lobster/core/provenance/ir_coverage.py` - Moved IR coverage reporter (canonical)
- `lobster/core/notebooks/__init__.py` - Notebooks subpackage init
- `lobster/core/notebooks/executor.py` - Moved notebook executor (renamed, canonical)
- `lobster/core/notebooks/exporter.py` - Moved notebook exporter (renamed, canonical, imports updated)
- `lobster/core/notebooks/validator.py` - Moved notebook validator (renamed, canonical)
- `lobster/core/analysis_ir.py` - Shim with DeprecationWarning
- `lobster/core/lineage.py` - Shim with DeprecationWarning
- `lobster/core/ir_coverage.py` - Shim with DeprecationWarning
- `lobster/core/notebook_executor.py` - Shim with DeprecationWarning
- `lobster/core/notebook_exporter.py` - Shim with DeprecationWarning
- `lobster/core/notebook_validator.py` - Shim with DeprecationWarning
- `.importlinter` - Updated canonical path + layers contract
- `tests/unit/core/test_core_subpackage_shims.py` - Removed xfail markers, fixed __getattr__ test

## Decisions Made
- Used __getattr__ lazy shim in provenance/__init__.py to resolve the module/package name collision (Python can't have both core/provenance.py and core/provenance/)
- Renamed notebook files on move (notebook_executor.py -> executor.py) for cleaner subpackage naming
- Updated exporter.py internal imports to canonical paths immediately rather than deferring
- Moved hasattr() inside catch_warnings block in test to properly capture __getattr__ lazy shim warnings

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test to capture __getattr__ lazy shim warnings**
- **Found during:** Task 2 (verification)
- **Issue:** `test_shim_reexports_and_warns[provenance]` failed because `hasattr()` call was outside the `catch_warnings` block. The provenance __getattr__ shim fires on attribute access, not on module import.
- **Fix:** Moved `hasattr(mod, expected_name)` inside the `catch_warnings` context manager
- **Files modified:** `tests/unit/core/test_core_subpackage_shims.py`
- **Verification:** All 31 tests pass
- **Committed in:** `47b7898` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for test that didn't account for lazy shim behavior. No scope creep.

## Issues Encountered
None beyond the test fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 core/ subpackages exist and are populated: governance/, queues/, runtime/, notebooks/, provenance/
- All 13 files in new canonical locations with backward-compatible shims
- Import-linter layers contract enforces subpackage dependency ordering
- Phase 06 is COMPLETE -- ready for Phase 07 (data_manager_v2 move)

---
*Phase: 06-core-subpackage-creation-moves*
*Completed: 2026-03-04*
