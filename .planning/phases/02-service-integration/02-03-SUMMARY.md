---
phase: 02-service-integration
plan: 03
subsystem: disease-ontology
tags: [strangler-fig, disease-matching, vector-search, backend-branching, ontology, pydantic]

# Dependency graph
requires:
  - phase: 02-service-integration-02
    provides: VectorSearchService.match_ontology() with alias resolution, typed OntologyMatch returns, ONTOLOGY_COLLECTIONS constant
provides:
  - DiseaseOntologyService backend branching (json keyword vs embeddings semantic)
  - _convert_ontology_match() mapping OntologyMatch -> DiseaseMatch (SCHM-03)
  - Explicit logger.warning on ImportError fallback in DiseaseStandardizationService
  - Deleted duplicate lobster/config/disease_ontology.json
  - 15 new tests covering backend switching, converter, fallback, config cleanup
affects: [annotation-agent, metadata-assistant, phase-03-cell-type-annotation]

# Tech tracking
tech-stack:
  added: []
  patterns: [strangler-fig-backend-branching, lazy-import-for-vector-service, builtins-import-mock-for-testing]

key-files:
  created: []
  modified:
    - packages/lobster-metadata/lobster/services/metadata/disease_ontology_service.py
    - packages/lobster-metadata/lobster/services/metadata/disease_standardization_service.py
    - packages/lobster-metadata/tests/services/metadata/test_disease_ontology_service.py
  deleted:
    - lobster/config/disease_ontology.json

key-decisions:
  - "Lazy import of VectorSearchService inside __init__ body (not module-level) to avoid pulling deps when backend=json"
  - "builtins.__import__ patching for fallback tests since VectorSearchService is imported dynamically"
  - "Always build keyword index even with embeddings backend (needed for fallback and legacy APIs)"

patterns-established:
  - "Backend branching via config.backend field check in __init__ with graceful ImportError fallback"
  - "_convert_ontology_match() as bridge between vector search OntologyMatch and disease-specific DiseaseMatch"

requirements-completed: [MIGR-01, MIGR-02, MIGR-03, MIGR-04, MIGR-05, MIGR-06, TEST-06]

# Metrics
duration: 5min
completed: 2026-02-18
---

# Phase 2 Plan 03: DiseaseOntologyService Migration Summary

**Strangler Fig migration of DiseaseOntologyService with backend branching (json/embeddings), OntologyMatch-to-DiseaseMatch converter, silent fallback fix, and duplicate config cleanup**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-18T22:00:08Z
- **Completed:** 2026-02-18T22:05:20Z
- **Tasks:** 2
- **Files modified:** 3 (+ 1 deleted)

## Accomplishments
- Added backend branching to DiseaseOntologyService: when backend="json" uses existing keyword matching unchanged; when backend="embeddings" delegates to VectorSearchService.match_ontology("mondo")
- Implemented _convert_ontology_match() for seamless OntologyMatch -> DiseaseMatch field mapping (SCHM-03 compatibility)
- Fixed silent ImportError fallback in DiseaseStandardizationService with explicit logger.warning message
- Deleted duplicate lobster/config/disease_ontology.json (canonical copy stays in lobster-metadata package)
- Added 15 new tests across 5 test classes: TestBackendSwitching (5), TestConvertOntologyMatch (5), TestFallbackBehavior (2), TestDuplicateConfigRemoved (2), TestSilentFallbackFixed (1)

## Task Commits

Each task was committed atomically:

1. **Task 1: Backend branching, silent fallback fix, duplicate config deletion** - `79438a6` (feat)
2. **Task 2: Tests for backend switching, converter, fallback** - No separate commit (all files in gitignored lobster-metadata package)

**Plan metadata:** (pending) (docs: complete plan)

_Note: lobster-metadata is a private/gitignored package. Code changes in Tasks 1-2 are applied to local files but only the deleted config file from Task 1 is git-trackable._

## Files Created/Modified
- `packages/lobster-metadata/lobster/services/metadata/disease_ontology_service.py` - Added TYPE_CHECKING import for OntologyMatch, vector backend initialization in __init__, vector delegation in match_disease(), _convert_ontology_match() converter method
- `packages/lobster-metadata/lobster/services/metadata/disease_standardization_service.py` - Added logger.warning on ImportError fallback (was silent)
- `packages/lobster-metadata/tests/services/metadata/test_disease_ontology_service.py` - Added 15 new tests in 5 classes (TestBackendSwitching, TestConvertOntologyMatch, TestFallbackBehavior, TestDuplicateConfigRemoved, TestSilentFallbackFixed)
- `lobster/config/disease_ontology.json` - DELETED (duplicate removed; canonical in lobster-metadata)

## Decisions Made
- VectorSearchService imported lazily inside __init__ body (not module-level) to avoid pulling heavy vector deps when backend="json"
- Used builtins.__import__ patching for fallback tests since VectorSearchService is dynamically imported and can't be patched as a module attribute
- Keyword index always built regardless of backend (needed for fallback path and legacy API methods)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mock approach for lazy-imported VectorSearchService**
- **Found during:** Task 2 (writing tests)
- **Issue:** Plan suggested patching `lobster.services.metadata.disease_ontology_service.VectorSearchService` but VectorSearchService is lazy-imported inside __init__ body, so it's not a module attribute and can't be patched that way
- **Fix:** For backend switching tests: patched `lobster.core.vector.service.VectorSearchService` (source module). For fallback tests: patched `builtins.__import__` to raise ImportError for the specific import path.
- **Files modified:** packages/lobster-metadata/tests/services/metadata/test_disease_ontology_service.py
- **Verification:** All 36 tests pass
- **Committed in:** N/A (gitignored file)

---

**Total deviations:** 1 auto-fixed (1 bug in test mock approach)
**Impact on plan:** Necessary fix for test correctness. No scope creep.

## Issues Encountered
- lobster-metadata package is gitignored (.gitignore line 8: `packages/lobster-metadata/`). This means all code changes (disease_ontology_service.py, disease_standardization_service.py, test file) are applied locally but not tracked by git. Only the deleted `lobster/config/disease_ontology.json` from Task 1 was committable. The lobster-metadata package needed to be installed via `pip install -e` before tests could run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- DiseaseOntologyService is now Phase 2-ready with backend branching
- Switching to embeddings backend requires: (1) vector-search deps installed, (2) setting backend="embeddings" in disease_ontology.json config
- All 36 disease ontology tests pass (21 Phase 1 + 15 Phase 2)
- All 72 vector search tests pass (Phase 1 + Phase 2)
- Migration-stable API contract preserved: match_disease() signature unchanged regardless of backend

## Self-Check: PASSED

- disease_ontology_service.py exists with backend branching code
- disease_standardization_service.py exists with logger.warning fix
- test_disease_ontology_service.py exists with 36 tests (21 existing + 15 new)
- lobster/config/disease_ontology.json verified deleted
- packages/lobster-metadata/lobster/config/disease_ontology.json verified present (canonical)
- Commit 79438a6 verified in git log
- All 36 disease ontology tests pass
- All 72 vector search tests pass (70 passed, 2 skipped)

---
*Phase: 02-service-integration*
*Completed: 2026-02-18*
