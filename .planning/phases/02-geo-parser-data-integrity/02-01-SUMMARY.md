---
phase: 02-geo-parser-data-integrity
plan: 01
subsystem: data-access
tags: [geo, parser, dataclass, partial-parse, data-integrity]

# Dependency graph
requires:
  - phase: 01-geo-safety-contract-hotfixes
    provides: GEO retry types, metadata key consistency
provides:
  - ParseResult dataclass in geo/parser.py with integrity metadata
  - Partial parse signaling through all 5 geo_service.py call sites
affects: [02-02, geo-service-decomposition]

# Tech tracking
tech-stack:
  added: []
  patterns: [ParseResult wrapper pattern for signaling data integrity]

key-files:
  created:
    - tests/unit/services/data_access/test_geo_parser_partial.py
    - tests/unit/services/data_access/test_geo_service_partial_handling.py
  modified:
    - lobster/services/data_access/geo/parser.py
    - lobster/services/data_access/geo_service.py

key-decisions:
  - "ParseResult dataclass placed before GEOParser class in parser.py (module-level, importable)"
  - "isinstance guard on all call sites for backward compatibility with any code path returning bare DataFrame"
  - "Custom _WarningCapture handler in tests because lobster logger sets propagate=False"

patterns-established:
  - "ParseResult wrapper: all parse methods return ParseResult instead of bare Optional[DataFrame]"
  - "Call site pattern: parse_result = parser.method(); matrix = parse_result.data if isinstance(...) else parse_result; check is_partial"

requirements-completed: [GPAR-01, GPAR-02]

# Metrics
duration: 26min
completed: 2026-03-04
---

# Phase 02 Plan 01: Partial Parse Signaling Summary

**ParseResult dataclass wrapping chunk parser output with is_partial/rows_read/truncation_reason, propagated through all 5 geo_service.py call sites with logged warnings**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-04T03:08:36Z
- **Completed:** 2026-03-04T03:35:27Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ParseResult dataclass with is_partial, rows_read, truncation_reason fields and is_complete/is_empty properties
- parse_large_file_in_chunks now signals memory-limited truncation via ParseResult instead of silently returning truncated data
- parse_expression_file and parse_supplementary_file updated to return ParseResult (all return paths)
- All 5 geo_service.py call sites extract .data and log warnings when partial results detected
- 22 tests covering dataclass properties, chunk parser partial/complete/empty states, and call site warning propagation

## Task Commits

Each task was committed atomically:

1. **Task 1: ParseResult dataclass and chunk parser partial signaling** - `42c5d8d` (test)
2. **Task 2: Propagate partial parse handling to all geo_service.py call sites** - `0c82ddf` (feat)

## Files Created/Modified
- `lobster/services/data_access/geo/parser.py` - Added ParseResult dataclass, updated parse_large_file_in_chunks/parse_expression_file/parse_supplementary_file return types
- `lobster/services/data_access/geo_service.py` - Updated all 5 call sites to handle ParseResult, added ParseResult import
- `tests/unit/services/data_access/test_geo_parser_partial.py` - 15 tests for ParseResult and parser partial signaling
- `tests/unit/services/data_access/test_geo_service_partial_handling.py` - 7 tests for call site partial result handling

## Decisions Made
- ParseResult dataclass placed at module level in parser.py (before GEOParser class) for clean importability
- isinstance guard on call sites ensures backward compatibility if any code path returns bare DataFrame/None
- Custom _WarningCapture handler used in geo_service tests because lobster logger sets propagate=False, preventing pytest caplog capture
- When memory break triggers before any chunks collected (0 rows), the result is still marked is_partial=True with truncation_reason

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed empty-chunks partial signaling**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** When memory limit triggered break on first chunk (0 chunks collected), code fell through to "no data" path returning is_partial=False instead of True
- **Fix:** Updated "no data" return path to propagate truncated/truncation_reason flags
- **Files modified:** lobster/services/data_access/geo/parser.py
- **Verification:** test_memory_limit_returns_partial passes
- **Committed in:** 42c5d8d (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was necessary for correct partial signaling when memory limit hits on first chunk. No scope creep.

## Issues Encountered
- Lobster logger uses propagate=False with its own StreamHandler, preventing pytest caplog from capturing warnings. Solved with custom _WarningCapture handler attached directly to the geo_service logger.
- Pre-existing test failures in drug_discovery (ModuleNotFoundError) and real API test (test_search_literature_real) confirmed unrelated to changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ParseResult is ready for plan 02-02 (further GEO parser improvements)
- All existing tests pass, no regressions introduced
- Pattern established for future parse methods to use ParseResult wrapper

## Self-Check: PASSED

All files exist. All commits verified. ParseResult importable and functional.

---
*Phase: 02-geo-parser-data-integrity*
*Completed: 2026-03-04*
