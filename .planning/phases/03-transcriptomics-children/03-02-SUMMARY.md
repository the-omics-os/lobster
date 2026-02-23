---
phase: 03-transcriptomics-children
plan: 02
subsystem: agents
tags: [de-analysis, pydeseq2, differential-expression, transcriptomics, tool-refactoring]

# Dependency graph
requires:
  - phase: 02-transcriptomics-parent
    provides: transcriptomics expert with bulk RNA-seq tools and SC+bulk unified prompt
provides:
  - Merged DE tools (3->2): run_differential_expression + run_de_with_formula
  - Merged formula tool: suggest_de_formula (metadata analysis + construction + validation)
  - New result tools: filter_de_results + export_de_results
  - Renamed tools: prepare_de_design, run_pathway_enrichment
  - Deprecated: construct_de_formula_interactive
affects: [03-transcriptomics-children, docs-site-agent-pages]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Auto-detect pseudobulk vs bulk in unified DE tool"
    - "Column name normalization for DE result filtering across methods"
    - "All-in-one formula tool (analyze + construct + validate)"

key-files:
  created: []
  modified:
    - packages/lobster-transcriptomics/lobster/agents/transcriptomics/de_analysis_expert.py
    - packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py

key-decisions:
  - "Unified run_differential_expression auto-detects pseudobulk via adata.uns flags"
  - "Column name normalization handles pyDESeq2 (padj, log2FoldChange) and generic (FDR, logFC) variations"
  - "Export tool creates 'exports' directory in workspace path"

patterns-established:
  - "DE tool merger pattern: auto-detect data type, route to appropriate service method"
  - "Column name normalization: try multiple known column names in priority order"

requirements-completed: [DEA-01, DEA-02, DEA-04, DEA-05, DEA-06, DEA-07, DEA-08]

# Metrics
duration: 8min
completed: 2026-02-23
---

# Phase 03 Plan 02: DE Analysis Expert Tools Summary

**Merged 3 DE tools into 2 with auto-detection, merged formula tools, added filter/export tools, renamed 2 tools for consistency**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-23T02:30:15Z
- **Completed:** 2026-02-23T02:38:29Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Merged 3 overlapping DE tools (run_pseudobulk_differential_expression + run_differential_expression_analysis) into unified run_differential_expression with auto-detection of pseudobulk vs bulk data
- Merged 2 formula tools (suggest_formula_for_design + construct_de_formula_interactive) into suggest_de_formula that does metadata analysis + formula construction + validation in one call
- Added filter_de_results (padj/lfc/baseMean filtering with column name normalization) and export_de_results (CSV/Excel with publication-ready columns)
- Renamed prepare_differential_expression_design to prepare_de_design and run_pathway_enrichment_analysis to run_pathway_enrichment
- Deprecated construct_de_formula_interactive with clear deprecation message
- Updated all cross-references in prompts.py and response strings

## Task Commits

Each task was committed atomically:

1. **Task 1: Merge DE tools (3->2) + merge formula tools (2->1) + deprecate interactive + rename 2 tools** - `b8b786a` (feat)
2. **Task 2: Add filter_de_results + export_de_results tools** - `db19c49` (feat)

## Files Created/Modified
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/de_analysis_expert.py` - Refactored DE tools: 12 tools (was 11), cleaner composition, 2 new result tools
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py` - Updated Available Tools section with new tool names

## Decisions Made
- Unified run_differential_expression auto-detects pseudobulk via adata.uns['pseudobulk_design'], adata.uns['is_pseudobulk'], or adata.uns['formula_design'] flags
- Column name normalization in filter_de_results tries multiple known patterns in priority order (padj/pvalue_adj/FDR, log2FoldChange/logFC/mean_log2FC, baseMean/base_mean/AveExpr)
- Export tool standardizes columns to publication format (gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added AnalysisStep IR to run_differential_expression pseudobulk path**
- **Found during:** Task 1
- **Issue:** The merged pseudobulk path in run_differential_expression needed explicit IR creation since it directly calls bulk_rnaseq_service.run_pydeseq2_from_pseudobulk (which returns tuple without IR)
- **Fix:** Added inline AnalysisStep creation with pyDESeq2 code_template
- **Files modified:** de_analysis_expert.py
- **Verification:** IR is logged via log_tool_usage with ir=ir
- **Committed in:** b8b786a

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Necessary for provenance tracking. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DE analysis expert fully refactored with clean 12-tool composition
- Ready for Plan 03 (prompt refinement) or downstream testing
- All tools have AnalysisStep IR for provenance tracking

## Self-Check: PASSED

All files and commits verified:
- de_analysis_expert.py: FOUND
- prompts.py: FOUND
- 03-02-SUMMARY.md: FOUND
- Commit b8b786a: FOUND
- Commit db19c49: FOUND

---
*Phase: 03-transcriptomics-children*
*Completed: 2026-02-23*
