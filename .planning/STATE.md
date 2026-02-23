# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-22)

**Core value:** Every agent has exactly the right tools — no overlap, no gaps, no wrong abstraction level — so the LLM picks the correct tool every time and produces reliable, reproducible science.
**Current focus:** Phase 6 in progress (Metabolomics Package)

## Current Position

Phase: 6 of 7 (Metabolomics Package)
Plan: 2 of 3 completed
Status: In Progress
Last activity: 2026-02-23 — Completed 06-02-PLAN.md (10 tools, agent factory, entry point)

Progress: [████████████] 90%

## Performance Metrics

**Velocity:**
- Total plans completed: 17
- Average duration: 5.5 min
- Total execution time: 1.57 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-genomics-domain | 3 | 18 min | 6 min |
| 02-transcriptomics-parent | 3 | 17 min | 5.7 min |
| 03-transcriptomics-children | 3 | 16 min | 5.3 min |
| 04-ms-proteomics-core | 3 | 10 min | 3.3 min |
| 05-proteomics-children-affinity | 4/4 | 21 min | 5.3 min |

**Recent Trend:**
- Last 5 plans: 05-02 (4 min), 05-03 (8 min), 05-04 (5 min), 06-01 (11 min), 06-02 (6 min)
- Trend: Stabilizing around 6-8 min for tool/agent work

*Updated after each plan completion*
| Phase 06 P01 | 11min | 2 tasks | 7 files |
| Phase 06 P02 | 6min | 2 tasks | 5 files |

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

- D23: score_gene_set stores scored modality via store_modality (not direct dict assignment)
- D24: component_registry with fallback direct import for vector_search (not registered as entry point)

- D25: Unified run_differential_expression auto-detects pseudobulk via adata.uns flags
- D26: Column name normalization in filter_de_results handles pyDESeq2 and generic DE output variations

- D27: run_gsea_analysis uses PathwayEnrichmentService with ranked gene DataFrame construction from DE results
- D28: Graceful LFC shrinkage in extract_and_export_de_results (checks if applied, warns if unavailable)
- D29: DE prompt organizes 15 tools by workflow stage with explicit tool selection guide

- D30: Start with median and sum for peptide-to-protein rollup (median polish deferred as future enhancement)
- D31: PTM site IDs use gene_residuePosition_m{multiplicity} format with deduplication suffix for collisions
- D32: Unmatched PTM sites kept with raw values (not dropped) during PTM-to-protein normalization

- D33: Pathway service wraps core PathwayEnrichmentService rather than reimplementing Enrichr calls
- D34: Built-in SIGNOR-style kinase-substrate mapping with ~20 well-known kinases as KSEA default
- D35: STRING service gracefully degrades to basic edge counts when networkx unavailable

- D36: Bundled ~80 common metabolites in reference DB for v1 m/z annotation (amino acids, organic acids, sugars, nucleotides, fatty acids, lipids)
- D37: Custom OPLS-DA via NIPALS (~100 lines numpy) instead of pyopls dependency (unmaintained since 2020)

- D38: Minimal prompts.py created in Plan 02 (Rule 3 blocking fix) so factory function works; Plan 03 will expand

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-23
Stopped at: Completed 06-02-PLAN.md (10 tools, agent factory, entry point)
Resume file: None
