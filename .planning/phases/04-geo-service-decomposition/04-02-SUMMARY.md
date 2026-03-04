---
phase: 04-geo-service-decomposition
plan: 02
subsystem: data-access
tags: [geo, refactoring, decomposition, facade, domain-modules]

# Dependency graph
requires:
  - phase: 04-geo-service-decomposition
    provides: helpers.py and soft_download.py shared modules from Plan 01
provides:
  - 5 domain modules (MetadataFetcher, DownloadExecutor, ArchiveProcessor, MatrixParser, SampleConcatenator)
  - Thin GEOService facade with __getattr__ forwarding
  - geo/__init__.py package marker
affects: [04-03-PLAN, geo-provider-dedup]

# Tech tracking
tech-stack:
  added: []
  patterns: [facade-with-composition, __getattr__-forwarding, lazy-module-initialization]

key-files:
  created:
    - lobster/services/data_access/geo/metadata_fetch.py
    - lobster/services/data_access/geo/download_execution.py
    - lobster/services/data_access/geo/archive_processing.py
    - lobster/services/data_access/geo/matrix_parsing.py
    - lobster/services/data_access/geo/concatenation.py
    - lobster/services/data_access/geo/__init__.py
  modified:
    - lobster/services/data_access/geo_service.py
    - tests/unit/services/data_access/test_geo_service_partial_handling.py
    - tests/unit/services/data_access/test_geo_temp_cleanup.py

key-decisions:
  - "self.service pattern: domain modules receive parent GEOService instance, access shared state via self.service.*"
  - "__getattr__ with lazy init: handles test mocks that patch __init__ to lambda by lazily creating modules on first access"
  - "__dict__.get() in __getattr__: prevents infinite recursion when accessing module instances before __init__ completes"
  - "Test patch paths updated: tarfile and BulkRNASeqService patched at new archive_processing module location"

patterns-established:
  - "Facade composition: __init__ creates domain module instances, public methods delegate explicitly, private methods forwarded via __getattr__"
  - "Domain module pattern: class receives parent service in __init__, all instance access via self.service.* prefix"
  - "Lazy imports preserved: DataExpertAssistant and BulkRNASeqService kept as method-level imports in extracted modules"

requirements-completed: [GDEC-01, GDEC-02]

# Metrics
duration: 26min
completed: 2026-03-04
---

# Phase 4 Plan 2: GEO Service Decomposition Summary

**Decomposed 5,633-line GEOService monolith into 5 focused domain modules (1,386+1,213+826+1,040+670 LOC) with 246-line facade preserving all 167 tests**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-04T06:20:23Z
- **Completed:** 2026-03-04T06:46:00Z
- **Tasks:** 2
- **Files modified:** 9 (5 created + 4 modified)

## Accomplishments
- Extracted all 53 methods from GEOService into 5 focused domain modules with single responsibility each
- Converted geo_service.py from 5,633 lines to 246-line thin facade with composition pattern
- All 167 existing GEO tests pass unchanged (except 2 tests updated for new module paths)
- All public symbols (GEOService, GEODataSource, GEOResult, RetryOutcome, RetryResult, ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file) remain importable from geo_service.py
- __getattr__ forwarding ensures backward compatibility for all private method access patterns

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 5 domain module classes and extract all methods** - `a58b3ed` (feat)
2. **Task 2: Convert geo_service.py to facade and update geo/__init__.py** - `4b2ca4e` (refactor)

## Files Created/Modified
- `lobster/services/data_access/geo/metadata_fetch.py` - MetadataFetcher: 12 methods for metadata fetching, extraction, platform validation, sample type detection (1,386 lines)
- `lobster/services/data_access/geo/download_execution.py` - DownloadExecutor: 10 methods for download coordination, pipeline steps, strategic download (1,213 lines)
- `lobster/services/data_access/geo/archive_processing.py` - ArchiveProcessor: 8 methods for TAR extraction, nested archives, 10X handling, quantification files (826 lines)
- `lobster/services/data_access/geo/matrix_parsing.py` - MatrixParser: 18 methods for matrix validation, file classification, sample downloads, transpose logic (1,040 lines)
- `lobster/services/data_access/geo/concatenation.py` - SampleConcatenator: 5 methods for sample storage, concatenation, clinical metadata injection (670 lines)
- `lobster/services/data_access/geo/__init__.py` - Minimal package marker (no re-exports)
- `lobster/services/data_access/geo_service.py` - Thin facade: __init__ + 3 public delegations + __getattr__ + 2 helper wrappers (246 lines)
- `tests/unit/services/data_access/test_geo_service_partial_handling.py` - Updated log capture to include domain module loggers
- `tests/unit/services/data_access/test_geo_temp_cleanup.py` - Updated patch paths for tarfile and BulkRNASeqService

## Decisions Made
- **self.service pattern**: Domain modules receive parent GEOService in __init__, access all shared state (data_manager, cache_dir, console, geo_downloader, geo_parser, pipeline_engine, tenx_loader) via self.service.* -- minimal change from original self.* pattern
- **__getattr__ with lazy init**: Tests that patch `__init__` to `lambda self: None` bypass module creation; __getattr__ lazily initializes modules on first private method access to maintain backward compatibility
- **__dict__.get() for recursion safety**: Accessing `self._metadata_fetcher` inside `__getattr__` would trigger infinite recursion; using `self.__dict__.get("_metadata_fetcher")` avoids this
- **Test patch path updates**: Two test files needed patch target changes since tarfile and BulkRNASeqService are now imported in archive_processing.py instead of geo_service.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed __getattr__ infinite recursion**
- **Found during:** Task 2 (facade conversion)
- **Issue:** `__getattr__` accessing `self._metadata_fetcher` triggers `__getattr__` recursively when modules not yet initialized
- **Fix:** Used `self.__dict__.get()` to safely access module instances without triggering __getattr__
- **Files modified:** lobster/services/data_access/geo_service.py
- **Committed in:** 4b2ca4e

**2. [Rule 1 - Bug] Added lazy module initialization for test compatibility**
- **Found during:** Task 2 (test verification)
- **Issue:** Tests that patch `__init__` to noop don't create domain modules, causing AttributeError on private method access
- **Fix:** Added lazy initialization in `__getattr__` that creates modules if none exist
- **Files modified:** lobster/services/data_access/geo_service.py
- **Committed in:** 4b2ca4e

**3. [Rule 3 - Blocking] Updated test patch paths for relocated imports**
- **Found during:** Task 2 (test verification)
- **Issue:** test_geo_service_partial_handling.py captured logger from geo_service module only; test_geo_temp_cleanup.py patched tarfile and BulkRNASeqService at old geo_service location
- **Fix:** Updated log capture to include all domain module loggers; updated patch targets to archive_processing module
- **Files modified:** tests/unit/services/data_access/test_geo_service_partial_handling.py, tests/unit/services/data_access/test_geo_temp_cleanup.py
- **Committed in:** 4b2ca4e

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for backward compatibility with existing test patterns. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 domain modules ready for Plan 03 SOFT dedup work
- Plan 03 can replace the 7 copy-pasted SOFT download blocks in domain modules with calls to pre_download_soft_file
- Plan 03 can add narrow unit tests for individual domain modules
- geo_provider.py SOFT dedup (7th instance) tracked for Plan 03
- Facade compatibility tests can verify __getattr__ forwarding paths

## Self-Check: PASSED

- All 7 created/modified files exist on disk
- Both task commits (a58b3ed, 4b2ca4e) verified in git history
- 167 GEO tests pass, all public symbols importable
- Facade is 246 lines (target < 400)

---
*Phase: 04-geo-service-decomposition*
*Completed: 2026-03-04*
