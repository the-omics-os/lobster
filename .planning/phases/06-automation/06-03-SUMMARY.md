---
phase: 06-automation
plan: 03
subsystem: vector-search
tags: [chromadb, s3, auto-download, ontology, cache, rich-progress]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: ChromaDBBackend, BaseVectorBackend ABC, vector store infrastructure
  - phase: 06-02
    provides: Build script for ontology tarballs, ONTOLOGY_COLLECTIONS naming convention
provides:
  - Auto-download of pre-built ontology tarballs from S3 on first use
  - Cache management at ~/.lobster/ontology_cache/ for downloaded tarballs
  - Rich progress bar for download feedback
  - Graceful degradation on network/extraction failure
  - 14 unit tests covering all auto-download paths
affects: [06-04, cloud-deployment, end-user-cold-start]

# Tech tracking
tech-stack:
  added: [requests (streaming download), rich.progress (download bar), tarfile (extraction)]
  patterns: [atomic-download-with-tmp-rename, source-client-data-copy, direct-client-check-to-avoid-recursion]

key-files:
  created:
    - tests/unit/core/vector/test_auto_download.py
  modified:
    - lobster/core/vector/backends/chromadb_backend.py

key-decisions:
  - "Direct client.get_collection() check in _ensure_ontology_data to avoid recursion through _get_or_create_collection"
  - "Atomic download via .tmp file rename prevents partial file corruption on network failure"
  - "Separate PersistentClient for source tarball data avoids SQLite locking issues"
  - "Network/tarball errors degrade gracefully to empty collection (warning, not crash)"
  - "tarfile.extractall(filter='data') for Python 3.12+ safe extraction"
  - "Module-level _download_with_progress function with lazy requests/rich imports"

patterns-established:
  - "Atomic download pattern: write to .tmp, rename to final dest on success, cleanup on failure"
  - "Source-client data copy: open separate PersistentClient on extracted dir, read all, add to target"
  - "Mock-client test pattern: replace backend._client with MagicMock for chromadb-free testing"

requirements-completed: [DATA-05, DATA-06]

# Metrics
duration: 6min
completed: 2026-02-19
---

# Phase 06 Plan 03: Ontology Auto-Download Summary

**S3 auto-download of pre-built ontology tarballs in ChromaDB backend with Rich progress bar, ~/.lobster/ontology_cache/ caching, and graceful network failure degradation**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-19T08:53:32Z
- **Completed:** 2026-02-19T08:59:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Extended ChromaDBBackend with `_ensure_ontology_data()` method that auto-downloads pre-built ontology tarballs from S3 on first use
- Added `ONTOLOGY_TARBALLS` mapping for 3 ontologies (mondo, uberon, cell_ontology) pointing to `https://lobster-ontology-data.s3.amazonaws.com/v1/`
- Implemented `_download_with_progress()` with Rich progress bar (BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn)
- Atomic download pattern prevents partial file corruption (download to .tmp, rename on success)
- Source-client data copy pattern avoids SQLite locking between ChromaDB PersistentClients
- Created 14 unit tests covering: URL constants, cache hit/miss, network errors, corrupt tarballs, atomic cleanup
- All 178 vector tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Auto-download extension for ChromaDB backend** - `03c9336` (feat)
2. **Task 2: Unit tests for auto-download and cache management** - `2d83b88` (test)

## Files Created/Modified
- `lobster/core/vector/backends/chromadb_backend.py` - Added ONTOLOGY_TARBALLS, ONTOLOGY_CACHE_DIR, _download_with_progress(), _ensure_ontology_data() method, modified _get_or_create_collection()
- `tests/unit/core/vector/test_auto_download.py` - 14 unit tests across 3 test classes (TestOntologyTarballs, TestEnsureOntologyData, TestDownloadWithProgress)

## Decisions Made
- Used direct `client.get_collection()` in `_ensure_ontology_data` instead of `self.collection_exists()`/`self.count()` to avoid infinite recursion through `_get_or_create_collection` which calls `_ensure_ontology_data` again
- Atomic download with `.tmp` suffix and `rename()` on success to prevent partial file corruption (per research pitfall #4)
- Used separate PersistentClient for reading source tarball data, then copied documents into backend's own persist path (avoids SQLite locking per research pitfall #2)
- Used `tarfile.extractall(filter="data")` for Python 3.12+ safe extraction (per research pitfall #3)
- Network errors, corrupt tarballs, and ChromaDB loading errors all degrade gracefully (return False, log warning) rather than crashing
- Tests mock chromadb via `backend._client` replacement rather than `sys.modules` patching, since ChromaDBBackend can be instantiated without chromadb (only `_get_client()` needs it)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed infinite recursion in _ensure_ontology_data**
- **Found during:** Task 2 (unit test revealed the issue)
- **Issue:** `_ensure_ontology_data` called `self.collection_exists()` and `self.count()` which both route through `_get_or_create_collection()` which calls `_ensure_ontology_data()` again -- infinite recursion
- **Fix:** Changed to use direct `client.get_collection()` call with count, bypassing `_get_or_create_collection`
- **Files modified:** lobster/core/vector/backends/chromadb_backend.py
- **Verification:** All 14 tests pass, no recursion
- **Committed in:** 2d83b88 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for correctness. No scope creep.

## Issues Encountered
- chromadb is not installed in the dev venv (it's an optional `[vector-search]` extra). Initial test approach using `pytest.importorskip("chromadb")` caused all tests to be skipped. Resolved by using mock-client pattern that replaces `backend._client` directly, allowing all 14 tests to run without requiring chromadb installed.

## User Setup Required
None - no external service configuration required. The S3 bucket `lobster-ontology-data` needs to be created and populated with tarballs (via the build script from 06-02) before end users can benefit from auto-download.

## Next Phase Readiness
- Auto-download infrastructure complete and tested
- ChromaDB backend transparently fetches ontology data from S3 on first use
- Cloud handoff spec (06-04) already completed, defines how vector.omics-os.com replaces local ChromaDB
- Phase 06 is now feature-complete: all 4 plans executed

## Self-Check: PASSED

- [x] `lobster/core/vector/backends/chromadb_backend.py` exists
- [x] `tests/unit/core/vector/test_auto_download.py` exists
- [x] Commit `03c9336` exists in git log
- [x] Commit `2d83b88` exists in git log

---
*Phase: 06-automation*
*Completed: 2026-02-19*
