---
phase: 01-geo-safety-contract-hotfixes
plan: 02
subsystem: core
tags: [geo, retry, typed-results, gds, queue-preparer, entrez]

# Dependency graph
requires:
  - "01-01: MetadataEntry TypedDict and centralized metadata writes"
provides:
  - "RetryOutcome enum and RetryResult dataclass for typed retry results"
  - "GDS-to-GSE canonicalization in GEOQueuePreparer.prepare_queue_entry"
  - "original_accession preservation in queue entry metadata"
affects: [01-03, geo_service, geo_queue_preparer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Typed retry results via RetryResult dataclass with .succeeded and .needs_fallback properties"
    - "GDS canonicalization at queue preparation time with original_accession traceability"

key-files:
  created:
    - tests/unit/services/data_access/test_geo_retry_types.py
    - tests/unit/services/data_access/test_geo_queue_preparer.py
  modified:
    - lobster/services/data_access/geo_service.py
    - lobster/services/data_access/geo_queue_preparer.py

key-decisions:
  - "RetryOutcome enum uses lowercase string values (success, exhausted, soft_file_missing) matching Python naming"
  - "Call site 2 and 3 only check result.succeeded (no needs_fallback) since they are download paths not metadata paths"
  - "GDS resolution via lightweight Entrez eSummary in queue preparer (not full GEOService._fetch_gds_metadata_and_convert)"
  - "Graceful fallback: GDS resolution failure passes original accession through unchanged"

patterns-established:
  - "_retry_with_backoff always returns RetryResult -- callers use .succeeded and .needs_fallback, never string comparisons"
  - "GDS accessions canonicalized to GSE at queue preparation time with original_accession in metadata"

requirements-completed: [GSAF-05, GSAF-01]

# Metrics
duration: 10min
completed: 2026-03-04
---

# Phase 1 Plan 02: Typed Retry Results and GDS Canonicalization Summary

**RetryResult dataclass replaces string sentinel "SOFT_FILE_MISSING" across all _retry_with_backoff call sites; GDS accessions canonicalized to GSE at queue preparation with original_accession traceability**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-04T02:27:04Z
- **Completed:** 2026-03-04T02:37:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- RetryOutcome enum (SUCCESS, EXHAUSTED, SOFT_FILE_MISSING) and RetryResult dataclass added to geo_service.py
- All 3 _retry_with_backoff call sites converted from string comparison to typed property checks
- Zero remaining "SOFT_FILE_MISSING" string literal comparisons in geo_service.py (verified by AST test)
- GDS accessions canonicalized to GSE in GEOQueuePreparer.prepare_queue_entry with original_accession preserved
- 27 new tests (20 for typed retry, 7 for GDS canonicalization)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace string sentinels with RetryResult type (RED)** - `5897487` (test)
2. **Task 1: Replace string sentinels with RetryResult type (GREEN)** - `e5135f2` (feat)
3. **Task 2: Canonicalize GDS accessions to GSE in queue preparation (RED)** - `9e771a6` (test)
4. **Task 2: Canonicalize GDS accessions to GSE in queue preparation (GREEN)** - `085ebe3` (feat)

_TDD tasks have paired RED/GREEN commits._

## Files Created/Modified
- `tests/unit/services/data_access/test_geo_retry_types.py` - 20 tests for RetryOutcome enum, RetryResult properties, _retry_with_backoff return types, and string sentinel elimination
- `tests/unit/services/data_access/test_geo_queue_preparer.py` - 7 tests for GDS canonicalization, original_accession preservation, pass-through for non-GDS accessions
- `lobster/services/data_access/geo_service.py` - Added RetryOutcome enum, RetryResult dataclass; updated _retry_with_backoff return type and all 3 call sites
- `lobster/services/data_access/geo_queue_preparer.py` - Added prepare_queue_entry override with GDS canonicalization and _resolve_gds_to_gse via Entrez eSummary

## Decisions Made
- Used RetryOutcome enum with lowercase string values for consistency with Python naming conventions
- Call sites 2 and 3 (download paths) only check `result.succeeded` since they don't have fallback paths like call site 1
- GDS resolution implemented as lightweight Entrez eSummary lookup in queue preparer rather than delegating to full GEOService._fetch_gds_metadata_and_convert, which fetches full metadata unnecessarily
- On GDS resolution failure, the original accession passes through unchanged (no error thrown) for robustness

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None - implementation proceeded as designed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RetryResult type is stable for any future _retry_with_backoff consumers
- GDS canonicalization ready for downstream queue processing
- All GEO safety fixes from plans 01, 02, and 03 are now complete

## Self-Check: PASSED

All 4 created/modified files exist. All 4 task commits verified in git log.

---
*Phase: 01-geo-safety-contract-hotfixes*
*Completed: 2026-03-04*
