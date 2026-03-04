---
phase: 09-repo-hygiene-packaging-cleanup
plan: 02
subsystem: infra
tags: [cleanup, shims, empty-dirs, geo, imports]

requires:
  - phase: 04-geo-decomposition
    provides: Canonical GEO module paths (geo/downloader.py, geo/parser.py)
provides:
  - 12 empty placeholder directories removed
  - Deprecated geo_parser.py and geo_downloader.py shim files removed
  - Test imports updated to canonical paths
affects: []

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - tests/unit/tools/test_geo_downloader.py
    - tests/integration/test_gse248556_bug_fixes.py

key-decisions:
  - "Removed lobster/data/ parent dir after its child cache/ was removed (became empty)"
  - "tests/resilience/ already absent -- skipped (12 of 13 planned dirs existed)"

patterns-established: []

requirements-completed: [HYGN-03, HYGN-04]

duration: 2min
completed: 2026-03-04
---

# Phase 09 Plan 02: Empty Dirs and Deprecated Shims Cleanup Summary

**Removed 12 empty placeholder directories and 2 deprecated GEO shim files, updating test imports to canonical paths**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-04T20:06:14Z
- **Completed:** 2026-03-04T20:08:18Z
- **Tasks:** 2
- **Files modified:** 5 (2 test files updated, 2 shim files deleted, 1 parent dir removed)

## Accomplishments
- Removed 12 empty placeholder directories left from GEO decomposition and project scaffolding
- Removed 1 additional parent directory (lobster/data/) that became empty after child removal
- Updated 2 test files to import GEODownloadManager from canonical path
- Removed geo_parser.py (zero importers) and geo_downloader.py (importers updated) shim files
- All 20 geo_downloader unit tests pass with canonical imports

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Remove empty dirs, update imports, remove shims** - `5ed9e40` (chore)

Task 1 (empty directory removal) had no git-tracked changes since empty directories are not tracked by git. Combined into single commit with Task 2.

## Files Created/Modified
- `tests/unit/tools/test_geo_downloader.py` - Updated import to canonical path
- `tests/integration/test_gse248556_bug_fixes.py` - Updated import to canonical path
- `lobster/tools/geo_parser.py` - Removed (deprecated shim, zero importers)
- `lobster/tools/geo_downloader.py` - Removed (deprecated shim, importers updated)

## Decisions Made
- Combined Task 1 and Task 2 into single commit since empty directories are not git-tracked
- Also removed lobster/data/ parent directory that became empty after lobster/data/cache/ removal
- tests/resilience/ was already absent (12 of 13 planned directories existed)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Removed newly-empty parent directory lobster/data/**
- **Found during:** Task 1 (empty directory removal)
- **Issue:** After removing lobster/data/cache/, the parent lobster/data/ became empty
- **Fix:** Also removed lobster/data/ with rmdir
- **Files modified:** (directory only, not git-tracked)
- **Verification:** find confirms zero empty directories remain

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Minimal -- cleaned up parent that became empty. No scope creep.

## Issues Encountered
- tests/resilience/ directory did not exist (already absent). Skipped without issue.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All empty placeholder directories cleaned up
- All deprecated GEO shim files removed
- Repository ready for further hygiene tasks in remaining Phase 09 plans

---
*Phase: 09-repo-hygiene-packaging-cleanup*
*Completed: 2026-03-04*
