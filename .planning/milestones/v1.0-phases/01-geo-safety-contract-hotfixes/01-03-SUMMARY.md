---
phase: 01-geo-safety-contract-hotfixes
plan: 03
subsystem: security, concurrency
tags: [tar-security, path-traversal, symlink, cve-2007-4559, compare-and-swap, download-queue, race-condition]

# Dependency graph
requires:
  - phase: none
    provides: none
provides:
  - Reject-all tar extraction with symlink/hardlink escape detection
  - Shared _is_safe_member removing geo/downloader.py duplicate
  - CAS status transitions preventing duplicate workers on download queue
  - Orchestrator CAS claim with silent no-op for concurrent workers
affects: [download-orchestrator, geo-downloader, archive-utils]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reject-all archive policy: validate ALL members before extracting ANY"
    - "Compare-and-swap status transitions: expected_current_status parameter returns None on mismatch"
    - "Shared security implementation: single _is_safe_member for all tar extraction paths"

key-files:
  created: []
  modified:
    - lobster/core/archive_utils.py
    - lobster/services/data_access/geo/downloader.py
    - lobster/core/download_queue.py
    - lobster/tools/download_orchestrator.py
    - tests/unit/core/test_archive_utils.py
    - tests/unit/core/test_download_queue.py

key-decisions:
  - "Reject-all policy over skip-unsafe: entire archive rejected when any member is unsafe, preventing partial malicious extraction"
  - "CAS returns Optional[DownloadQueueEntry] rather than bool: caller can use the entry directly on success, check None on failure"
  - "Orchestrator returns empty tuple on CAS failure: silent no-op with log, not an error"
  - "Replaced geo/downloader.py inline extraction with ArchiveExtractor.extract_safely: single extraction call instead of manual member iteration"

patterns-established:
  - "Reject-all archive validation: check all members, raise RuntimeError with offending paths if any unsafe"
  - "CAS update_status: optional expected_current_status param, backward compatible, returns None on mismatch"

requirements-completed: [GSAF-04, GSAF-06]

# Metrics
duration: 7min
completed: 2026-03-04
---

# Phase 01 Plan 03: Security & Concurrency Hardening Summary

**Reject-all tar extraction with CVE-2007-4559 symlink detection and CAS-based download queue claiming to prevent duplicate workers**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-04T02:15:26Z
- **Completed:** 2026-03-04T02:22:24Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Hardened tar extraction to reject entire archives when any path-traversal, symlink escape, or hardlink escape is detected (no partial extraction)
- Eliminated duplicate `is_safe_member` from `geo/downloader.py` by delegating to shared `ArchiveExtractor.extract_safely`
- Added compare-and-swap (CAS) status transitions to `DownloadQueue.update_status` with backward-compatible optional parameter
- Orchestrator now uses CAS to atomically claim download entries, preventing race conditions between concurrent workers
- 87 total tests pass (41 archive_utils + 46 download_queue)

## Task Commits

Each task was committed atomically (TDD: test + feat):

1. **Task 1: Harden tar extraction** - `64924f6` (test) + `1b23a35` (feat)
2. **Task 2: CAS status transitions** - `485341e` (test) + `3cd364d` (feat)

_TDD tasks have separate test (RED) and implementation (GREEN) commits._

## Files Created/Modified
- `lobster/core/archive_utils.py` - Enhanced `_is_safe_member` with symlink/hardlink checks; reject-all `extract_safely`
- `lobster/services/data_access/geo/downloader.py` - Removed inline `is_safe_member`, delegates to shared `ArchiveExtractor`
- `lobster/core/download_queue.py` - Added `expected_current_status` CAS parameter to `update_status`
- `lobster/tools/download_orchestrator.py` - CAS claim on `execute_download` with silent no-op on failure
- `tests/unit/core/test_archive_utils.py` - 14 new security tests (path traversal, symlink, reject-all, shared impl)
- `tests/unit/core/test_download_queue.py` - 4 new CAS tests (success, failure, backward compat, concurrent)

## Decisions Made
- Reject-all policy over skip-unsafe: prevents partial malicious extraction which is more dangerous than full rejection
- CAS returns `Optional[DownloadQueueEntry]` instead of `bool`: richer return type lets callers use the entry directly
- Orchestrator returns empty tuple `("", {})` on CAS failure: clean no-op that doesn't interfere with caller logic
- Replaced geo/downloader manual member-by-member extraction with single `ArchiveExtractor.extract_safely` call: simpler and ensures shared security checks

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Archive extraction is now secure against path traversal and symlink escape attacks
- Download queue race condition is fixed with CAS transitions
- All changes are backward compatible -- no API breaks for existing callers

## Self-Check: PASSED

All 6 modified/created files verified on disk. All 4 task commits verified in git log.

---
*Phase: 01-geo-safety-contract-hotfixes*
*Completed: 2026-03-04*
