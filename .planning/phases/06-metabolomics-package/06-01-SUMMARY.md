---
phase: 06-metabolomics-package
plan: 01
subsystem: metabolomics
tags: [metabolomics, LC-MS, GC-MS, NMR, PLS-DA, OPLS-DA, VIP, PQN, anndata, sklearn]

# Dependency graph
requires:
  - phase: 04-ms-proteomics-core
    provides: ProteomicsQualityService/PreprocessingService pattern for service architecture
provides:
  - packages/lobster-metabolomics/ package scaffold with pyproject.toml and entry points
  - MetabPlatformConfig with 3 platform configs (lc_ms, gc_ms, nmr)
  - MetabolomicsQualityService with assess_quality method
  - MetabolomicsPreprocessingService with 4 methods (filter, impute, normalize, batch correct)
  - MetabolomicsAnalysisService with 5 methods (univariate, PCA, PLS-DA, OPLS-DA, fold change)
  - MetabolomicsAnnotationService with 2 methods (annotate_by_mz, classify_lipids)
  - BUG-14 fix for sparse matrix zero-checking in core metabolomics schema
affects: [06-02-PLAN, 06-03-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: [MetabPlatformConfig for platform-specific defaults, NIPALS-based OPLS-DA, PLS-DA via sklearn PLSRegression with VIP scores, PQN normalization, bundled metabolite reference DB for m/z annotation]

key-files:
  created:
    - packages/lobster-metabolomics/pyproject.toml
    - packages/lobster-metabolomics/lobster/agents/metabolomics/config.py
    - packages/lobster-metabolomics/lobster/services/quality/metabolomics_quality_service.py
    - packages/lobster-metabolomics/lobster/services/quality/metabolomics_preprocessing_service.py
    - packages/lobster-metabolomics/lobster/services/analysis/metabolomics_analysis_service.py
    - packages/lobster-metabolomics/lobster/services/annotation/metabolomics_annotation_service.py
  modified:
    - lobster/core/schemas/metabolomics.py

key-decisions:
  - "D36: Bundled ~80 common metabolites in reference DB for v1 m/z annotation (amino acids, organic acids, sugars, nucleotides, fatty acids, lipids)"
  - "D37: Custom OPLS-DA via NIPALS (~100 lines numpy) instead of pyopls dependency (unmaintained since 2020)"

patterns-established:
  - "MetabPlatformConfig pattern: dataclass with platform-specific defaults, detect function with scoring, get_platform_config helper"
  - "Metabolomics service pattern: dense-first processing (toarray check), NaN-aware operations, 3-tuple return with full IR"

requirements-completed: [MET-01, MET-02, MET-03, MET-04, MET-05, MET-17]

# Metrics
duration: 11min
completed: 2026-02-23
---

# Phase 6 Plan 01: Metabolomics Package Services Summary

**4 stateless services (quality, preprocessing, analysis, annotation) with MetabPlatformConfig for LC-MS/GC-MS/NMR, bundled metabolite reference DB, PLS-DA/OPLS-DA multivariate analysis, and BUG-14 sparse matrix fix**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-23T05:53:19Z
- **Completed:** 2026-02-23T06:04:36Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Created complete lobster-metabolomics package scaffold (PEP 420, entry points, 3 platform configs)
- Implemented 4 stateless services with 13 public methods total, all returning 3-tuples with AnalysisStep IR
- Custom OPLS-DA implementation via NIPALS algorithm (~100 lines) with cross-validation Q2 and permutation testing
- Fixed BUG-14: sparse matrix zero-checking now uses nnz-based calculation in metabolomics schema validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Package scaffold + PlatformConfig + BUG-14 fix** - `6eb232c` (feat)
2. **Task 2: Four stateless services** - `a0f8adc` (feat)

## Files Created/Modified
- `packages/lobster-metabolomics/pyproject.toml` - Package metadata, dependencies, entry points for metabolomics_expert
- `packages/lobster-metabolomics/LICENSE` - AGPL-3.0-or-later license
- `packages/lobster-metabolomics/README.md` - Package description with platform support table
- `packages/lobster-metabolomics/lobster/agents/metabolomics/config.py` - MetabPlatformConfig with lc_ms/gc_ms/nmr configs, detect_platform_type(), get_platform_config()
- `packages/lobster-metabolomics/lobster/agents/metabolomics/py.typed` - PEP 561 marker
- `packages/lobster-metabolomics/lobster/services/quality/metabolomics_quality_service.py` - QC assessment: RSD, TIC CV, QC sample evaluation, missing value analysis
- `packages/lobster-metabolomics/lobster/services/quality/metabolomics_preprocessing_service.py` - Feature filtering, imputation (5 methods), normalization (5 methods), batch correction (3 methods)
- `packages/lobster-metabolomics/lobster/services/analysis/metabolomics_analysis_service.py` - Univariate stats with FDR, PCA, PLS-DA with VIP scores, OPLS-DA (NIPALS), fold changes
- `packages/lobster-metabolomics/lobster/services/annotation/metabolomics_annotation_service.py` - m/z annotation with ~80 metabolite reference DB, lipid class classification
- `lobster/core/schemas/metabolomics.py` - BUG-14 fix: sparse matrix zero-checking uses nnz-based calculation

## Decisions Made
- D36: Bundled ~80 common metabolites in reference DB for v1 (amino acids, organic acids, sugars, nucleotides, fatty acids, representative lipids). Full HMDB database lookup deferred to v2.
- D37: Custom OPLS-DA via NIPALS algorithm (~100 lines of numpy) instead of depending on pyopls (unmaintained since March 2020). Includes 7-fold CV for Q2 and permutation testing.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 4 services ready for Plan 02 (tool wrapping in shared_tools.py)
- Entry points in pyproject.toml point to metabolomics_expert module (created in Plan 02)
- PlatformConfig available for tool-level platform-aware defaults
- BUG-14 fix ensures correct schema validation for sparse metabolomics data

## Self-Check: PASSED

All 10 files verified present. Both task commits (6eb232c, a0f8adc) verified in git log.

---
*Phase: 06-metabolomics-package*
*Completed: 2026-02-23*
