---
phase: 10-fix-provenance-getattr-shim
plan: 01
subsystem: core
tags: [python, getattr, shim, provenance, backward-compat, pep562]

# Dependency graph
requires:
  - phase: 06-core-subpackage-restructure
    provides: provenance subpackage with single-module __getattr__ shim
provides:
  - Multi-module __getattr__ shim searching all 4 provenance submodules
  - AnalysisStep, ParameterSpec, LineageMetadata, IRCoverageAnalyzer resolvable via old import path
affects: [11-shim-migration, de-analysis-expert]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-module __getattr__ search with _SUBMODULES tuple]

key-files:
  created: []
  modified:
    - lobster/core/provenance/__init__.py
    - tests/unit/core/test_core_subpackage_shims.py

key-decisions:
  - "Search order: provenance.py first (most common), then analysis_ir, lineage, ir_coverage"
  - "Do NOT update call sites -- shim handles them transparently, migration is Phase 11 scope"

patterns-established:
  - "Multi-module __getattr__ shim: _SUBMODULES tuple + loop over importlib.import_module for subpackage backward compat"

requirements-completed: [CORE-04]

# Metrics
duration: 2min
completed: 2026-03-04
---

# Phase 10 Plan 01: Fix Provenance __getattr__ Shim Summary

**Multi-module __getattr__ shim extending provenance package to search all 4 submodules (provenance, analysis_ir, lineage, ir_coverage), fixing ImportError at 6 DE analysis call sites**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-04T20:47:15Z
- **Completed:** 2026-03-04T20:49:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Extended provenance `__getattr__` from single-module to multi-module search across all 4 submodules
- `from lobster.core.provenance import AnalysisStep` now resolves without ImportError
- 11 new parametrized test cases (5 resolution + 5 identity + 1 AttributeError) added to shim test suite
- Existing ProvenanceTracker backward-compat behavior fully preserved

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing multi-module shim tests** - `bc61783` (test)
2. **Task 1 (GREEN): Extend __getattr__ to search all submodules** - `7728313` (feat)
3. **Task 2: Smoke test and broader suite verification** - no file changes (verification only)

_TDD task with RED/GREEN commits._

## Files Created/Modified
- `lobster/core/provenance/__init__.py` - Multi-module __getattr__ shim with _SUBMODULES tuple
- `tests/unit/core/test_core_subpackage_shims.py` - TestProvenanceMultiModuleShim class + AttributeError test

## Decisions Made
- Search order: provenance.py first (most common usage), then analysis_ir, lineage, ir_coverage
- Do NOT update the 6 de_analysis_expert.py call sites -- shim handles them transparently, migration belongs to Phase 11

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing test failure in `test_data_manager_v2.py::TestExportDocumentation::test_store_and_retrieve_metadata` (validation key KeyError) -- documented in Phase 07 as known issue, out of scope

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Provenance shim complete, all 4 submodules searchable
- DE analysis expert import paths now work transparently
- Phase 11 (shim migration to canonical paths) can proceed when scheduled

---
*Phase: 10-fix-provenance-getattr-shim*
*Completed: 2026-03-04*
