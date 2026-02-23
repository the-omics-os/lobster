---
phase: 04-ms-proteomics-core
plan: 01
subsystem: proteomics-services
tags: [anndata, ptm, phosphoproteomics, peptide-to-protein, normalization, maxquant]

# Dependency graph
requires:
  - phase: 03-transcriptomics-children
    provides: Established service 3-tuple pattern and IR creation conventions
provides:
  - import_ptm_site_data service method for MaxQuant PTM site parsing
  - summarize_peptide_to_protein service method for peptide-to-protein rollup
  - normalize_ptm_to_protein service method for PTM-to-protein normalization
affects: [04-02-parent-agent-tools, 04-03-prompt-rewrite]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - PTM site ID construction (gene_residuePosition_multiplicity format)
    - MaxQuant intensity column auto-detection (LFQ vs raw)
    - Dual-input service method (ptm_adata + protein_adata)
    - Regression-based normalization via scipy.stats.linregress

key-files:
  created: []
  modified:
    - packages/lobster-proteomics/lobster/services/quality/proteomics_preprocessing_service.py

key-decisions:
  - "D30: Start with median and sum for peptide-to-protein rollup (median polish deferred as future enhancement)"
  - "D31: PTM site IDs use gene_residuePosition_m{multiplicity} format with deduplication suffix for collisions"
  - "D32: Unmatched PTM sites kept with raw values (not dropped) during PTM-to-protein normalization"

patterns-established:
  - "PTM site ID format: {gene}_{amino_acid}{position} with _m{multiplicity} suffix and _dup{N} for collisions"
  - "Dual-AnnData service method: normalize_ptm_to_protein accepts two independent AnnData objects"
  - "MaxQuant column identification via case-insensitive partial matching helper"

requirements-completed: [MSP-02, MSP-04, MSP-05]

# Metrics
duration: 4min
completed: 2026-02-23
---

# Phase 4 Plan 1: MS Proteomics Core Services Summary

**3 new ProteomicsPreprocessingService methods for PTM site import, peptide-to-protein summarization, and PTM-to-protein normalization with full AnalysisStep IR provenance**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T03:23:50Z
- **Completed:** 2026-02-23T03:27:42Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added `import_ptm_site_data` method that parses MaxQuant PTM site files (Phospho/Acetyl/GlyGly), filters by localization probability, auto-detects LFQ vs raw intensity columns, and constructs gene_residuePosition site IDs
- Added `summarize_peptide_to_protein` method that rolls up peptide-level AnnData to protein-level using median or sum aggregation while preserving obs (sample) metadata exactly
- Added `normalize_ptm_to_protein` method that normalizes PTM site abundances against total protein levels using log subtraction (ratio) or regression residuals, with graceful handling of unmatched sites

## Task Commits

Each task was committed atomically:

1. **Task 1: Add import_ptm_site_data method** - `bab6394` (feat)
2. **Task 2: Add summarize_peptide_to_protein and normalize_ptm_to_protein methods** - `3a8d39e` (feat)

## Files Created/Modified
- `packages/lobster-proteomics/lobster/services/quality/proteomics_preprocessing_service.py` - Added 3 public service methods, 3 IR helper methods, 3 private helpers for PTM column identification, intensity detection, and site ID construction (+756 lines)

## Decisions Made
- Started with median and sum for peptide-to-protein rollup; median polish (Tukey) deferred as future enhancement since simple methods are robust for well-filtered data
- PTM site IDs use `{gene}_{amino_acid}{position}` format with `_m{multiplicity}` suffix for multiply modified sites and `_dup{N}` suffix for collisions
- Unmatched PTM sites are kept with raw values during normalization (not dropped), enabling users to still analyze them

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 3 service methods are ready for Plan 02 to wrap as agent tools
- `import_ptm_site_data` provides the service layer for the `import_ptm_sites` tool (MSP-02)
- `summarize_peptide_to_protein` provides the service layer for the `summarize_peptide_to_protein` tool (MSP-04)
- `normalize_ptm_to_protein` provides the service layer for the `normalize_ptm_to_protein` tool (MSP-05)

## Self-Check: PASSED

- FOUND: proteomics_preprocessing_service.py
- FOUND: bab6394 (Task 1 commit)
- FOUND: 3a8d39e (Task 2 commit)
- FOUND: 04-01-SUMMARY.md

---
*Phase: 04-ms-proteomics-core*
*Completed: 2026-02-23*
