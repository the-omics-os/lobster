---
phase: 04-geo-service-decomposition
plan: 01
subsystem: data-access
tags: [geo, refactoring, deduplication, soft-download, helpers]

# Dependency graph
requires:
  - phase: 03-geo-strategy-engine
    provides: Strategy engine and pipeline type constants used by GEO service
provides:
  - helpers.py with shared GEO utilities (RetryOutcome, RetryResult, ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file, _is_data_valid, _retry_with_backoff)
  - soft_download.py with deduplicated SOFT pre-download logic (build_soft_url, pre_download_soft_file)
  - Re-exports from geo_service.py for full backward compatibility
affects: [04-02-PLAN, 04-03-PLAN, geo-provider-dedup]

# Tech tracking
tech-stack:
  added: []
  patterns: [module-extraction-with-re-export, standalone-function-from-method]

key-files:
  created:
    - lobster/services/data_access/geo/helpers.py
    - lobster/services/data_access/geo/soft_download.py
    - tests/unit/services/data_access/test_soft_download.py
  modified:
    - lobster/services/data_access/geo_service.py

key-decisions:
  - "helpers.py uses lazy imports for pandas/anndata to avoid heavy deps at module level"
  - "_retry_with_backoff accepts console param instead of self for standalone usage"
  - "build_soft_url auto-detects GSE vs GSM from prefix, no id_type param needed"

patterns-established:
  - "Method-to-function extraction: convert self methods to standalone functions with explicit params"
  - "Re-export backward compatibility: import from new module then re-export at original location"

requirements-completed: [GDEC-03]

# Metrics
duration: 13min
completed: 2026-03-04
---

# Phase 4 Plan 1: Shared Helpers and SOFT Download Deduplication Summary

**Extracted shared GEO utilities to helpers.py and deduplicated SOFT pre-download logic to soft_download.py with GSE+GSM URL support**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-04T06:03:34Z
- **Completed:** 2026-03-04T06:17:32Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `helpers.py` with 7 shared symbols extracted from geo_service.py (RetryOutcome, RetryResult, ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file, _is_data_valid, _retry_with_backoff)
- Created `soft_download.py` with `build_soft_url` (handles both GSE series and GSM sample URLs) and `pre_download_soft_file` (cached path, HTTPS download, SSL error handling)
- Replaced inline definitions in geo_service.py with imports + thin wrapper methods, reducing file by 321 lines
- Full backward compatibility: all 132 existing GEO tests pass unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for SOFT download helpers** - `a313192` (test)
2. **Task 1 GREEN: Create helpers.py and soft_download.py** - `b7769bb` (feat)
3. **Task 2: Re-exports in geo_service.py** - `d3047c2` (refactor)

_TDD task had RED and GREEN commits._

## Files Created/Modified
- `lobster/services/data_access/geo/helpers.py` - Shared utilities: RetryOutcome, RetryResult, ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file, _is_data_valid, _retry_with_backoff
- `lobster/services/data_access/geo/soft_download.py` - SOFT pre-download: build_soft_url, pre_download_soft_file with GSE+GSM support
- `tests/unit/services/data_access/test_soft_download.py` - 27 tests covering all behaviors
- `lobster/services/data_access/geo_service.py` - Replaced inline definitions with imports from helpers, thin wrapper methods for _is_data_valid and _retry_with_backoff

## Decisions Made
- `helpers.py` uses lazy imports for pandas/anndata to avoid pulling heavy dependencies at module level (only stdlib + lobster.utils.logger at import time)
- `_retry_with_backoff` standalone function accepts an explicit `console` parameter instead of `self` reference, enabling use from both GEOService and future standalone callers
- `build_soft_url` auto-detects GSE vs GSM from the accession prefix rather than requiring a separate `id_type` parameter -- simpler API since the prefix is always present

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- helpers.py and soft_download.py are ready for Plans 02 and 03 to import
- Plan 02 can extract metadata_fetch.py and download_execution.py using these shared modules
- Plan 03 can replace the 7 copy-pasted SOFT download blocks with calls to pre_download_soft_file
- geo_provider.py SOFT dedup (7th instance) is tracked for Plan 03

## Self-Check: PASSED

- All 3 created files exist on disk
- All 3 task commits (a313192, b7769bb, d3047c2) verified in git history
- 27 new tests pass, 132 existing GEO tests pass

---
*Phase: 04-geo-service-decomposition*
*Completed: 2026-03-04*
