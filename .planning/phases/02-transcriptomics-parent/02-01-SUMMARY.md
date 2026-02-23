---
phase: 02-transcriptomics-parent
plan: 01
subsystem: transcriptomics
tags: [scanpy, harmony, dpt, paga, trajectory, batch-integration, lisi, silhouette, pydeseq2, bulk-rnaseq, normalization, qc]

# Dependency graph
requires: []
provides:
  - "EnhancedSingleCellService.integrate_batches (Harmony/ComBat + LISI/silhouette)"
  - "EnhancedSingleCellService.compute_trajectory (DPT + PAGA with auto root selection)"
  - "BulkPreprocessingService with 4 methods (assess, filter, normalize, detect_batch)"
affects: [02-02-PLAN, 02-03-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy harmonypy import with HARMONY_AVAILABLE flag (matching scrublet pattern)"
    - "Manual LISI implementation via sklearn NearestNeighbors (avoids scib dependency)"
    - "pyDESeq2 size factors from dds.obs (not obsm) per current API"
    - "BulkPreprocessingService is stateless (no constructor args)"

key-files:
  created:
    - "packages/lobster-transcriptomics/lobster/services/analysis/bulk_preprocessing_service.py"
    - "packages/lobster-transcriptomics/tests/services/analysis/test_bulk_preprocessing_service.py"
  modified:
    - "packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py"
    - "packages/lobster-transcriptomics/tests/services/analysis/test_enhanced_singlecell_service.py"

key-decisions:
  - "Manual LISI instead of scib dependency (simpler, lighter, ~20 lines with sklearn KNN)"
  - "pyDESeq2 size factors read from dds.obs not dds.obsm (matches current pydeseq2 API)"
  - "BulkPreprocessingService in transcriptomics package (not core) to keep package self-contained"
  - "Auto root cell selection uses DC1 minimum (standard DPT convention)"

patterns-established:
  - "HARMONY_AVAILABLE flag at module top with try/except import (same as SCRUBLET_AVAILABLE)"
  - "Batch integration quality metrics: batch_silhouette + median_lisi in stats dict"
  - "Trajectory always computes PAGA even when only DPT requested (connectivity info useful)"
  - "Bulk sample QC uses sklearn PCA (sample-level) not scanpy PCA (cell-level)"

requirements-completed: [SCT-02, SCT-03, BLK-03, BLK-04, BLK-05, BLK-06]

# Metrics
duration: 7min
completed: 2026-02-22
---

# Phase 2 Plan 01: Services Summary

**SC batch integration (Harmony/ComBat + LISI/silhouette), DPT trajectory inference, and 4-method BulkPreprocessingService (QC, filter, normalize, batch detect)**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-23T01:21:38Z
- **Completed:** 2026-02-23T01:29:02Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- EnhancedSingleCellService gains integrate_batches (Harmony/ComBat) with LISI + silhouette quality metrics and compute_trajectory (DPT + PAGA) with auto/group/explicit root selection
- New BulkPreprocessingService with 4 methods: assess_sample_quality, filter_genes, normalize_counts (DESeq2/VST/CPM), detect_batch_effects
- All 6 new methods return proper (AnnData, Dict, AnalysisStep) 3-tuples with code_template IR
- 26 new tests (25 pass, 1 skip for harmonypy not installed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add integrate_batches and compute_trajectory to EnhancedSingleCellService** - `4df6f6f` (feat)
2. **Task 2: Create BulkPreprocessingService with 4 methods** - `321982f` (feat)

## Files Created/Modified
- `packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py` - Added integrate_batches, compute_trajectory, _compute_lisi, HARMONY_AVAILABLE flag, and IR methods
- `packages/lobster-transcriptomics/lobster/services/analysis/bulk_preprocessing_service.py` - NEW: BulkPreprocessingService with assess_sample_quality, filter_genes, normalize_counts, detect_batch_effects
- `packages/lobster-transcriptomics/tests/services/analysis/test_enhanced_singlecell_service.py` - Added 12 tests for integration + trajectory + LISI
- `packages/lobster-transcriptomics/tests/services/analysis/test_bulk_preprocessing_service.py` - NEW: 14 tests for all 4 bulk methods

## Decisions Made
- Manual LISI computation (sklearn NearestNeighbors + inverse Simpson) to avoid scib dependency (~20 lines vs ~50MB package)
- pyDESeq2 size factors from `dds.obs["size_factors"]` not `dds.obsm` (current pyDESeq2 API stores them in obs)
- BulkPreprocessingService placed in transcriptomics package (not core) per research recommendation
- Auto root cell selection for DPT uses DC1 minimum (standard convention in scanpy ecosystem)
- LISI k parameter auto-reduces when smallest batch has fewer cells than k+1

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pyDESeq2 size_factors location**
- **Found during:** Task 2 (normalize_counts DESeq2 method)
- **Issue:** Plan referenced `dds.obsm["size_factors"]` but current pyDESeq2 stores size factors in `dds.obs["size_factors"]` and pre-computed normalized counts in `dds.layers["normed_counts"]`
- **Fix:** Changed to `dds.obs["size_factors"].values` and use `dds.layers["normed_counts"]` when available
- **Files modified:** bulk_preprocessing_service.py
- **Verification:** test_normalize_counts_deseq2 passes
- **Committed in:** 321982f (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor API location fix. No scope creep.

## Issues Encountered
- Pre-existing test failures in `test_enhanced_singlecell_service.py` for `annotate_cell_types` tests (16 tests call without required `cluster_key` parameter). Not caused by this plan's changes. Out of scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 6 service methods are ready for Plan 02 (SC tools) and Plan 03 (bulk tools) to wrap as @tool functions
- integrate_batches + compute_trajectory provide 3-tuple returns for Plan 02 tool wrappers
- BulkPreprocessingService 4 methods provide 3-tuple returns for Plan 03 tool wrappers
- HARMONY_AVAILABLE flag ready for Plan 02's integrate_batches tool to check before calling

## Self-Check: PASSED

All 4 modified files verified on disk. Both task commits (4df6f6f, 321982f) verified in git log.

---
*Phase: 02-transcriptomics-parent*
*Completed: 2026-02-22*
