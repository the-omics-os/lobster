---
phase: 04-geo-service-decomposition
plan: 03
subsystem: data-access
tags: [geo, soft-download, deduplication, testing, facade, decomposition]

# Dependency graph
requires:
  - phase: 04-02
    provides: 5 domain modules (MetadataFetcher, DownloadExecutor, ArchiveProcessor, MatrixParser, SampleConcatenator) and GEOService facade
  - phase: 04-01
    provides: soft_download.py shared helper and helpers.py shared symbols
provides:
  - SOFT pre-download deduplication across all source files (8 blocks replaced)
  - 40 narrow unit tests for 5 domain modules
  - 31 facade compatibility and structural verification tests
  - Complete Phase 4 test coverage (71 new tests total for this plan)
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pre_download_soft_file() as single source of truth for SOFT downloads"
    - "Domain module tests use mock_service fixture pattern for isolation"
    - "Structural tests verify decomposition invariants (module count, facade LOC, no SOFT blocks)"

key-files:
  created:
    - tests/unit/services/data_access/test_geo_metadata_fetch.py
    - tests/unit/services/data_access/test_geo_download_execution.py
    - tests/unit/services/data_access/test_geo_archive_processing.py
    - tests/unit/services/data_access/test_geo_matrix_parsing.py
    - tests/unit/services/data_access/test_geo_concatenation.py
    - tests/unit/services/data_access/test_geo_facade_compat.py
    - tests/unit/services/data_access/test_geo_decomposition.py
  modified:
    - lobster/services/data_access/geo/metadata_fetch.py
    - lobster/services/data_access/geo/download_execution.py
    - lobster/services/data_access/geo/matrix_parsing.py
    - lobster/tools/providers/geo_provider.py
    - lobster/services/data_access/geo_fallback_service.py

key-decisions:
  - "pre_download_soft_file() handles both GSE and GSM via prefix detection -- no id_type param needed"
  - "Domain module tests mock the service parameter for full isolation (no DataManagerV2 needed)"
  - "Structural tests grep source files for PRE-DOWNLOAD SOFT as regression guard"

patterns-established:
  - "mock_service fixture: MagicMock with cache_dir, data_manager, console, geo_downloader etc. for domain module testing"
  - "Facade compat tests: import checks, __getattr__ forwarding, mock.patch.object patterns"

requirements-completed: [GDEC-01, GDEC-02, GDEC-03, GDEC-04]

# Metrics
duration: 12min
completed: 2026-03-04
---

# Phase 4 Plan 03: SOFT Dedup + Domain Tests + Facade Verification Summary

**Replaced 8 SOFT pre-download blocks with shared helper, added 71 tests covering all 5 domain modules plus facade compatibility and structural verification**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-04T06:51:11Z
- **Completed:** 2026-03-04T07:03:58Z
- **Tasks:** 3
- **Files modified:** 12 (5 source + 7 test)

## Accomplishments
- Zero SOFT pre-download blocks remain in any source file except soft_download.py (8 blocks replaced across 5 files, net -295 lines)
- 40 narrow unit tests for all 5 domain modules testing in complete isolation via mocked service
- 17 facade compatibility tests proving all import patterns, mock patterns, and API surface preserved
- 14 structural tests verifying module existence, class methods, facade size (246 LOC), no circular imports
- All 265 GEO tests pass (194 existing + 71 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace all SOFT pre-download blocks** - `989716d` (refactor)
2. **Task 2: Write narrow unit tests for 5 domain modules** - `608017a` (test)
3. **Task 3: Write facade compatibility and structural tests** - `2424164` (test)

## Files Created/Modified

**Source files modified (SOFT deduplication):**
- `lobster/services/data_access/geo/metadata_fetch.py` - Replaced SOFT block in _fetch_gse_metadata with pre_download_soft_file()
- `lobster/services/data_access/geo/download_execution.py` - Replaced 4 SOFT blocks in _try_supplementary_first, _try_supplementary_fallback, _try_emergency_fallback, _try_geoparse_download
- `lobster/services/data_access/geo/matrix_parsing.py` - Replaced SOFT block in _download_single_sample (handles GSM URLs)
- `lobster/tools/providers/geo_provider.py` - Replaced SOFT block in extract_download_urls
- `lobster/services/data_access/geo_fallback_service.py` - Replaced 2 SOFT blocks in try_series_matrix, download_single_cell_sample

**Test files created:**
- `tests/unit/services/data_access/test_geo_metadata_fetch.py` - 14 tests: init, routing, field extraction, sample type detection
- `tests/unit/services/data_access/test_geo_download_execution.py` - 6 tests: init, validation, strategy setting, pipeline selection
- `tests/unit/services/data_access/test_geo_archive_processing.py` - 6 tests: init, Kallisto/Salmon/mixed detection
- `tests/unit/services/data_access/test_geo_matrix_parsing.py` - 8 tests: init, matrix validation, data type, scoring
- `tests/unit/services/data_access/test_geo_concatenation.py` - 6 tests: init, inner/outer join decision, error handling
- `tests/unit/services/data_access/test_geo_facade_compat.py` - 17 tests: imports, facade identity, delegation, __getattr__, mock patching
- `tests/unit/services/data_access/test_geo_decomposition.py` - 14 tests: module existence, classes, facade LOC, circular imports, SOFT dedup

## Decisions Made

- **pre_download_soft_file() handles GSE and GSM transparently** -- the shared helper uses geo_id prefix detection (GSE -> series, GSM -> samples) so callers don't need to specify an id_type parameter
- **Domain module tests use mock_service fixture** -- each test creates a MagicMock with all expected attributes (cache_dir, data_manager, console, etc.) for complete isolation from real infrastructure
- **Structural tests grep for PRE-DOWNLOAD SOFT as regression guard** -- if anyone copy-pastes a SOFT block in the future, the test will catch it

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- ParseResult dataclass constructor: Test initially used wrong keyword args (file_category, file_type) that don't exist on ParseResult. Fixed by replacing the test with a _score_expression_file comparison test instead.
- PosixPath.mkdir is read-only: Test fixtures initially tried to mock mkdir on a real Path object. Fixed by using pytest's tmp_path fixture for GEOService initialization.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 4 GEO Service Decomposition is COMPLETE
- All 4 GDEC requirements satisfied (GDEC-01 through GDEC-04)
- 265 total GEO tests pass (194 existing + 71 new from this plan)
- geo_service.py reduced from ~3,200 LOC to 246-line facade
- SOFT download logic exists in exactly ONE location (soft_download.py)
- Ready for Phase 5

---
*Phase: 04-geo-service-decomposition*
*Completed: 2026-03-04*

## Self-Check: PASSED
- All 8 created files verified present
- All 3 task commits verified (989716d, 608017a, 2424164)
