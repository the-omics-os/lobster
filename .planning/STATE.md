# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Every agent has exactly the right tools — no overlap, no gaps, no wrong abstraction level — so the LLM picks the correct tool every time and produces reliable, reproducible science.
**Current focus:** Phase 2 - Transcriptomics Parent

## Current Position

Phase: 2 of 7 (Transcriptomics Parent) -- COMPLETE
Plan: 3 of 3 completed
Status: Phase Complete
Last activity: 2026-02-22 — Completed 02-03-PLAN.md (8 bulk tools, unified SC+bulk prompt)

Progress: [███░░░░░░░] 29%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 6 min
- Total execution time: 0.57 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-genomics-domain | 3 | 18 min | 6 min |
| 02-transcriptomics-parent | 3 | 17 min | 5.7 min |

**Recent Trend:**
- Last 5 plans: 01-03 (6 min), 02-01 (7 min), 02-02 (4 min), 02-03 (6 min)
- Trend: Stable (~5-6 min/plan)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- D1: New variant_analysis_expert child agent for clinical workflows
- D2: No separate affinity proteomics agent (PlatformConfig handles dual mode)
- D3: New lobster-metabolomics package (10 tools, existing infrastructure)
- D4: Single transcriptomics parent for SC + bulk (auto-detection + 8 bulk-specific tools)
- D5: Merge redundant DE tools (3→2) to reduce LLM confusion
- D6: Deprecate interactive terminal tools (cloud incompatible)
- D7: Merge list_modalities + get_modality_info (pilot in genomics)
- D8: Create import_proteomics_data tool (parsers exist but unreachable)
- D9: Add PTM analysis tools (phosphoproteomics >30% of published MS)
- D10: Tool taxonomy decorator (@tool_meta) deferred to P3, apply incrementally

- D11: Used sgkit.genomic_relationship (VanRaden) for kinship instead of pc_relate (no PCA prerequisite)
- D12: Position-based clumping without full LD matrix computation (sufficient for standard GWAS post-processing)
- D13: Composite priority scoring: consequence severity (0-0.4) + population rarity (0-0.3) + pathogenicity (0-0.3)
- D14: query_population_frequencies and query_clinical_databases reuse annotate_variants output when available

- D15: Tool list stays at 12 (same count, different composition: +3 new, -2 relocated, -2 merged into 1)
- D16: clump_results suggests variant_analysis_expert handoff when significant clumps found

- D17: Lazy prompt import inside factory function for child agents (allows AGENT_CONFIG entry point discovery before prompt exists)
- D18: lookup_variant uses EnsemblService directly with colocated_variants parsing for comprehensive single-variant reports

- D19: Manual LISI instead of scib dependency (simpler, lighter, ~20 lines with sklearn KNN)
- D20: pyDESeq2 size factors from dds.obs not dds.obsm (matches current pyDESeq2 API)
- D21: BulkPreprocessingService in transcriptomics package (not core) to keep self-contained
- D22: Auto root cell selection for DPT uses DC1 minimum (standard scanpy convention)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 02-03-PLAN.md (Phase 02 complete)
Resume file: None
