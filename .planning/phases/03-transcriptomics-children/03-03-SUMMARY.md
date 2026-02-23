---
phase: 03-transcriptomics-children
plan: 03
subsystem: agents
tags: [de-analysis, gsea, pathway-enrichment, bulk-rnaseq, publication-export, transcriptomics]

# Dependency graph
requires:
  - phase: 03-transcriptomics-children/02
    provides: Merged DE tools, filter/export tools, renamed tools
provides:
  - 3 new bulk DE pipeline tools (run_bulk_de_direct, run_gsea_analysis, extract_and_export_de_results)
  - Rewritten DE analysis expert prompt with all 15 active tools
  - Complete DE pipeline: DE -> filter -> GSEA -> publication export with LFC shrinkage
affects: [docs-site-agent-pages]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GSEA via PathwayEnrichmentService with ranked gene DataFrame from DE results"
    - "Column name normalization reused from filter_de_results for GSEA ranking"
    - "Graceful LFC shrinkage: apply when available, warn when not"

key-files:
  created: []
  modified:
    - packages/lobster-transcriptomics/lobster/agents/transcriptomics/de_analysis_expert.py
    - packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py

key-decisions:
  - "run_gsea_analysis uses PathwayEnrichmentService.gene_set_enrichment_analysis with ranked gene DataFrame construction"
  - "LFC shrinkage in extract_and_export_de_results is graceful: checks if already applied, warns if DeseqDataSet unavailable"
  - "Prompt organizes tools by workflow stage with explicit tool selection guide"

patterns-established:
  - "Ranked gene extraction pattern: discover fold change/p-value columns, build ['gene', 'score'] DataFrame"
  - "Graceful feature degradation: attempt shrinkage, warn if unavailable, proceed with unshrunk results"

requirements-completed: [DEA-09, DEA-10, DEA-11, DOC-04]

# Metrics
duration: 4min
completed: 2026-02-23
---

# Phase 03 Plan 03: Bulk DE Pipeline Tools + Prompt Rewrite Summary

**3 bulk DE pipeline tools (direct bulk DE, GSEA via PathwayEnrichmentService, publication export with LFC shrinkage) + complete prompt rewrite with 15 active tools and workflow guides**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-23T02:42:04Z
- **Completed:** 2026-02-23T02:46:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added run_bulk_de_direct for one-shot 2-group bulk DE without pseudobulk aggregation
- Added run_gsea_analysis wrapping PathwayEnrichmentService with ranked gene DataFrame extraction from DE results (supports log2fc and signed_pvalue ranking)
- Added extract_and_export_de_results with graceful LFC shrinkage detection and publication-ready table export
- Rewrote DE analysis expert prompt with all 15 tools organized by workflow stage, separate SC/bulk workflows, and tool selection guide

## Task Commits

Each task was committed atomically:

1. **Task 1: Add run_bulk_de_direct + run_gsea_analysis + extract_and_export_de_results tools** - `a7e197b` (feat)
2. **Task 2: Rewrite DE analysis expert prompt for all Phase 3 changes** - `77e7d4a` (feat)

## Files Created/Modified
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/de_analysis_expert.py` - Added 3 new tools (15 total in base_tools), numpy import for GSEA signed_pvalue metric
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py` - Complete rewrite of create_de_analysis_expert_prompt() with 15 tools, SC/bulk workflows, tool selection guide

## Decisions Made
- run_gsea_analysis constructs ranked gene DataFrame with column name normalization (same pattern as filter_de_results) and supports two ranking metrics: log2fc (default) and signed_pvalue
- extract_and_export_de_results handles LFC shrinkage gracefully: checks if already applied during DE, warns if DeseqDataSet unavailable, proceeds with unshrunk results
- Prompt organizes all 15 tools by workflow stage rather than flat list, with explicit tool selection guide for disambiguating overlapping tools

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DE analysis expert fully complete with 15 tools covering complete DE pipeline
- All Phase 3 (transcriptomics children) plans complete (3/3)
- Ready for next phase or documentation updates

## Self-Check: PASSED

All files and commits verified:
- de_analysis_expert.py: FOUND
- prompts.py: FOUND
- Commit a7e197b: FOUND
- Commit 77e7d4a: FOUND

---
*Phase: 03-transcriptomics-children*
*Completed: 2026-02-23*
