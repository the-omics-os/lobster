---
phase: 03-geo-strategy-engine-hardening
plan: 01
subsystem: data-access
tags: [geo, strategy-engine, null-handling, tdd, pipeline-selection]

# Dependency graph
requires:
  - phase: 01-metadata-key-consistency
    provides: consistent metadata store keys
provides:
  - _is_null_value() shared helper for null detection across GEO pipeline
  - MATRIX_FILETYPES, RAW_MATRIX_FILETYPES, H5_FILETYPES frozenset constants
  - Hardened PipelineContext.has_file() and get_file_info() with null rejection
  - _sanitize_null_values producing "" instead of truthy "NA"
  - Null-safe _derive_analysis and pipeline step methods
affects: [geo-service, download-orchestrator, strategy-rules]

# Tech tracking
tech-stack:
  added: []
  patterns: [_is_null_value centralized null checker, frozenset constants for file type validation]

key-files:
  created:
    - tests/unit/services/data_access/test_geo_strategy.py
  modified:
    - lobster/services/data_access/geo/strategy.py
    - packages/lobster-research/lobster/agents/data_expert/assistant.py
    - lobster/services/data_access/geo_queue_preparer.py
    - lobster/services/data_access/geo_service.py

key-decisions:
  - "_is_null_value uses strip().lower() comparison against frozenset for O(1) lookup"
  - "bool False and numeric 0 are NOT null -- they are valid domain values"
  - "get_file_info returns ('', '') when name is null-like, preventing cascade"
  - "raw_data_available uses bool() AND not _is_null_value() for defense-in-depth"

patterns-established:
  - "_is_null_value() as single source of truth for null detection in GEO pipeline"
  - "FILETYPES frozenset constants replacing inline lists in rule evaluation"

requirements-completed: [GSTR-01, GSTR-02]

# Metrics
duration: 12min
completed: 2026-03-04
---

# Phase 03 Plan 01: Null Value Handling Summary

**Shared _is_null_value() helper and FILETYPES constants hardening all null paths through GEO strategy engine, sanitizer, queue preparer, and pipeline steps**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-04T05:02:45Z
- **Completed:** 2026-03-04T05:14:43Z
- **Tasks:** 1 feature (TDD: RED + GREEN, no refactor needed)
- **Files modified:** 5

## Accomplishments
- Fixed root cause: _sanitize_null_values now produces "" instead of truthy "NA" at all 3 assignment sites
- Added _is_null_value() centralized helper used by 4 modules (strategy.py, geo_queue_preparer.py, geo_service.py, and indirectly via PipelineContext)
- Added MATRIX_FILETYPES, RAW_MATRIX_FILETYPES, H5_FILETYPES frozenset constants replacing inline lists
- Fixed 8 distinct null-handling bugs across the GEO pipeline
- 37 new tests covering all null behaviors, zero regressions (535 existing tests pass)

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests** - `bf5ac97` (test)
2. **GREEN: Implementation** - `824081f` (fix)

_Note: No refactor commit needed -- code was clean after GREEN._

## Files Created/Modified
- `tests/unit/services/data_access/test_geo_strategy.py` - 37 tests across 7 test classes covering all null-handling behaviors
- `lobster/services/data_access/geo/strategy.py` - Added _is_null_value(), FILETYPES constants, fixed PipelineContext and all rules
- `packages/lobster-research/lobster/agents/data_expert/assistant.py` - Fixed 3 sites: sanitized[key] = "NA" -> sanitized[key] = ""
- `lobster/services/data_access/geo_queue_preparer.py` - Imported _is_null_value, fixed _derive_analysis bool() patterns
- `lobster/services/data_access/geo_service.py` - Imported _is_null_value, fixed _try_processed_matrix_first and _try_raw_matrix_first null guards

## Decisions Made
- _is_null_value uses strip().lower() comparison against frozenset for O(1) lookup and whitespace tolerance
- bool False and numeric 0 are NOT null -- they are valid domain values (important for raw_data_available=False)
- get_file_info returns ("", "") when name is null-like, which cascades correctly through rule evaluation
- raw_data_available in data_availability uses bool() AND not _is_null_value() for defense-in-depth

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Null handling is now consistent across the entire GEO strategy engine
- Ready for plan 02 (strategy rule improvements, ARCHIVE_FIRST removal if planned)
- _is_null_value() available for any future GEO pipeline modules

## Self-Check: PASSED

All files verified present. All commits verified in git log.

---
*Phase: 03-geo-strategy-engine-hardening*
*Completed: 2026-03-04*
