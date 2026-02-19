---
phase: 05-scalability
plan: 01
subsystem: infra
tags: [faiss, pgvector, vector-search, backend, factory-pattern]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: BaseVectorBackend ABC, ChromaDBBackend, VectorSearchConfig factory, SearchBackend enum
provides:
  - FAISSBackend implementing BaseVectorBackend with IndexIDMap, L2 normalization, cosine distance conversion
  - PgVectorBackend stub with NotImplementedError on all methods (v2.0 placeholder)
  - Extended VectorSearchConfig.create_backend() factory with faiss and pgvector branches
  - 24 unit tests for FAISS backend (mocked), pgvector stub, and ABC contract
affects: [05-scalability, service-layer, deployment]

# Tech tracking
tech-stack:
  added: [faiss-cpu (optional)]
  patterns: [IndexIDMap wrapping IndexFlatL2, L2-normalize-then-search, squared-L2-to-cosine conversion, stub backend with NotImplementedError]

key-files:
  created:
    - lobster/core/vector/backends/faiss_backend.py
    - lobster/core/vector/backends/pgvector_backend.py
    - tests/unit/core/vector/test_backends.py
  modified:
    - lobster/core/vector/config.py
    - tests/unit/core/vector/test_config.py

key-decisions:
  - "IndexIDMap wrapping IndexFlatL2 for explicit integer ID assignment and single-vector deletion"
  - "L2 normalization before add and search so squared L2 / 2.0 = cosine distance"
  - "pgvector stub raises NotImplementedError with guidance to use chromadb or faiss"
  - "Lazy faiss import inside _ensure_faiss() method (same pattern as ChromaDB._get_client())"

patterns-established:
  - "Stub backend pattern: NotImplementedError with version and alternative guidance"
  - "String-to-int ID mapping for FAISS (which only supports int64 IDs)"
  - "sys.modules patching for mocking optional C-extension dependencies (faiss)"

requirements-completed: [INFRA-05, INFRA-06, INFRA-07, TEST-01]

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 05 Plan 01: FAISS Backend & pgvector Stub Summary

**FAISS in-memory backend with IndexIDMap, L2 normalization, and cosine distance conversion; pgvector stub for v2.0; factory wiring for zero-code backend switching**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T08:06:59Z
- **Completed:** 2026-02-19T08:10:59Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- FAISSBackend with IndexIDMap(IndexFlatL2), L2 normalization, string-to-int ID mapping, upsert semantics, and squared-L2-to-cosine distance conversion
- PgVectorBackend stub raising NotImplementedError with helpful v2.0 guidance on all 4 abstract methods
- VectorSearchConfig.create_backend() extended to route faiss and pgvector backends via lazy imports
- 24 new unit tests covering FAISS (mocked), pgvector stub, and ABC contract; all 146 existing vector tests pass unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement FAISS backend, pgvector stub, and factory wiring** - `468065e` (feat)
2. **Task 2: Unit tests for all backends and updated config factory tests** - `c574c91` (test)

## Files Created/Modified
- `lobster/core/vector/backends/faiss_backend.py` - FAISS backend with IndexIDMap, L2 normalization, cosine distance conversion
- `lobster/core/vector/backends/pgvector_backend.py` - pgvector stub raising NotImplementedError with v2.0 guidance
- `lobster/core/vector/config.py` - Extended create_backend() with faiss and pgvector branches
- `tests/unit/core/vector/test_backends.py` - 24 tests: FAISS (15), pgvector (6), ABC contract (3)
- `tests/unit/core/vector/test_config.py` - Replaced unsupported test with faiss/pgvector factory tests

## Decisions Made
- Used IndexIDMap wrapping IndexFlatL2 (not raw IndexFlatL2) so explicit integer IDs survive deletion without shifting
- L2-normalize all vectors before insertion and search so squared L2 distances can be converted to cosine distances via division by 2.0
- pgvector stub raises NotImplementedError with message suggesting chromadb and faiss as current alternatives
- Lazy faiss import inside _ensure_faiss() method, matching the ChromaDB._get_client() pattern
- sys.modules patching for mocking faiss in tests (same pattern as test_embedders.py and test_rerankers.py)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. FAISS backend uses faiss-cpu which is an optional dependency (`pip install faiss-cpu`).

## Next Phase Readiness
- FAISS backend ready for use via `LOBSTER_VECTOR_BACKEND=faiss` environment variable
- pgvector placeholder in place for v2.0 cloud deployment implementation
- Service layer completely untouched -- backend switching is transparent
- All 146 vector tests pass, confirming zero regressions

## Self-Check: PASSED

All 6 files verified present. Both commits (468065e, c574c91) confirmed in git log.

---
*Phase: 05-scalability*
*Completed: 2026-02-19*
