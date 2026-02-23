---
phase: 02-transcriptomics-parent
plan: 03
subsystem: transcriptomics
tags: [bulk-rnaseq, salmon, kallisto, featurecounts, mygene, gene-id-conversion, deseq2, vst, cpm, normalization, batch-effects, tool-routing, prompt-engineering]

# Dependency graph
requires:
  - "02-01: BulkPreprocessingService (assess_sample_quality, filter_genes, normalize_counts, detect_batch_effects)"
  - "02-01: BulkRNASeqService (load_from_quantification_files for Salmon/kallisto import)"
  - "02-02: Renamed tools (filter_and_normalize, select_variable_features, cluster_cells, find_marker_genes)"
provides:
  - "8 bulk RNA-seq tools on transcriptomics_expert (import, metadata, QC, filter, normalize, batch, gene IDs, DE validation)"
  - "Unified SC + bulk prompt with decision tree routing all 22+ tools"
  - "prepare_bulk_for_de validation checkpoint before DE handoff"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy mygene import with fallback message (matching harmonypy/scrublet pattern)"
    - "Auto-detect source format for import_bulk_counts (dir=Salmon/kallisto, file=featureCounts/CSV)"
    - "Auto-detect sample ID column by matching metadata values to obs_names"
    - "Validation-only tool pattern (prepare_bulk_for_de) that checks without modifying data"
    - "SC/bulk tool boundary enforcement in prompt Important_Rules"

key-files:
  created: []
  modified:
    - "packages/lobster-transcriptomics/lobster/agents/transcriptomics/transcriptomics_expert.py"
    - "packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py"

key-decisions:
  - "No new decisions required - followed plan exactly as written"

patterns-established:
  - "Bulk tool section placed between SC analysis tools and tool collection in transcriptomics_expert.py"
  - "import_bulk_counts cascading format detection: dir-based -> featureCounts -> CSV/TSV"
  - "prepare_bulk_for_de as validation gate before DE handoff (no data modification)"
  - "Prompt decision tree explicitly separates SC and bulk tool paths"

requirements-completed: [BLK-01, BLK-02, BLK-03, BLK-04, BLK-05, BLK-06, BLK-07, BLK-08, DOC-02]

# Metrics
duration: 6min
completed: 2026-02-22
---

# Phase 2 Plan 03: Bulk Tools + Prompt Summary

**8 bulk RNA-seq tools (Salmon/kallisto/featureCounts import through DE-readiness validation) and unified SC+bulk prompt with decision tree routing 22+ tools**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-23T01:38:20Z
- **Completed:** 2026-02-23T01:44:40Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 8 bulk RNA-seq tools added: import_bulk_counts (4-format auto-detect), merge_sample_metadata (auto sample ID matching), assess_bulk_sample_quality, filter_bulk_genes, normalize_bulk_counts (DESeq2/VST/CPM), detect_batch_effects, convert_gene_identifiers (mygene lazy import), prepare_bulk_for_de (validation-only)
- Prompt fully rewritten with unified SC + bulk routing: 22+ tools listed by correct (new) names, decision tree separates SC and bulk paths, standard workflows for both data types documented
- All 15 tools in transcriptomics_expert.py have ir=ir in log_tool_usage for full reproducibility
- Total tool count: ~22 direct tools (7 shared + 7 SC/clustering + 8 bulk) + delegation tools

## Task Commits

Each task was committed atomically:

1. **Task 1: Add 8 bulk RNA-seq tools** - `d0b82ea` (feat)
2. **Task 2: Update prompt with SC + bulk routing** - `ff07f5b` (feat)

## Files Created/Modified
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/transcriptomics_expert.py` - Added 8 bulk tools (import_bulk_counts, merge_sample_metadata, assess_bulk_sample_quality, filter_bulk_genes, normalize_bulk_counts, detect_batch_effects, convert_gene_identifiers, prepare_bulk_for_de), new imports (BulkRNASeqService, BulkPreprocessingService, pandas, numpy, AnalysisStep), updated tool collection with bulk_tools list
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/prompts.py` - Complete rewrite of create_transcriptomics_expert_prompt() with unified SC+bulk tool listing, decision tree, standard workflows for both data types, integration metrics guidance, SC/bulk tool boundary rules

## Decisions Made
None - followed plan exactly as written.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 02 (transcriptomics parent) is now complete: all 3 plans executed
- Unified transcriptomics expert has 22+ direct tools covering SC and bulk RNA-seq end-to-end
- Prompt routes correctly between SC and bulk workflows with clear decision tree
- Ready for Phase 03 and beyond

## Self-Check: PASSED

All 2 modified files verified on disk. Both task commits (d0b82ea, ff07f5b) verified in git log.

---
*Phase: 02-transcriptomics-parent*
*Completed: 2026-02-22*
