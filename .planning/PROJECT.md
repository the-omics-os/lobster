# Core Tools Refactor — Lobster AI

## What This Is

A systematic redesign of domain-specific core tools for every omics agent in Lobster AI. Each agent currently has tools that grew organically — with redundancy, gaps, and wrong abstraction levels. This project replaces that with a principled, minimal core tool set per domain: the bioinformatics equivalent of "Read, Write, Edit, Search, Execute" for a code editor. The refactor also creates a new `lobster-metabolomics` package and a new `variant_analysis_expert` child agent for genomics.

## Core Value

Every agent has exactly the right tools — no overlap, no gaps, no wrong abstraction level — so the LLM picks the correct tool every time and produces reliable, reproducible science.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Redesign genomics_expert parent tools (12 tools, +3 new: ld_prune, compute_kinship, clump_results)
- [ ] Create variant_analysis_expert child agent (9 tools for clinical/post-GWAS)
- [ ] Redesign transcriptomics_expert parent tools (14 SC + 8 bulk-specific tools)
- [ ] Enhance annotation_expert child tools (11 tools, +1 new: score_gene_set)
- [ ] Consolidate de_analysis_expert child tools (11 tools, merge 3→2 DE runners, +3 bulk additions)
- [ ] Redesign proteomics_expert parent tools (13 MS tools, +5 new including import and PTM)
- [ ] Enhance proteomics_de_analysis_expert child tools (7 tools, +4 new including pathway enrichment)
- [ ] Enhance biomarker_discovery_expert child tools (7 tools, +3 new including panel selection)
- [ ] Add 4 affinity-specific tools to proteomics_expert (import, LOD, bridge normalization, cross-platform)
- [ ] Create lobster-metabolomics package with metabolomics_expert agent (10 tools, structured for future children)
- [ ] Fix all 17 bugs found during research (2 CRASH, 1 DATA CORRUPTION, 3 RULE VIOLATIONS, 3 DEAD CODE, 3 UX/LLM CONFUSION, 5 LOGIC ISSUES)
- [ ] Deprecate interactive terminal tools (D6: manually_annotate_clusters_interactive, construct_de_formula_interactive)
- [ ] Update all agent prompts to reflect new tool inventories
- [ ] Implement D10 tool taxonomy decorator (@tool_meta) foundation

### Out of Scope

- Raw data preprocessing (XCMS, STAR, MaxQuant) — Lobster receives processed feature tables
- Cell-cell communication tools — advanced, not core
- Fine-mapping/PRS tools — post-GWAS P3 future
- NMR-specific processing — defer to post-MVP
- Targeted metabolomics standard curves — defer to post-MVP
- Separate affinity proteomics agent — D2 confirmed: stays within proteomics_expert
- Separate population_genetics_expert — rejected in D1: too coupled with GWAS pipeline

## Context

**Codebase state**: Lobster AI v1.0.x with 8 agent packages extracted (Kraken refactor 98% complete). 14 agents across 8 packages. ~82 domain-specific tools currently implemented.

**Research basis**: 6 parallel AI domain specialists analyzed 425 bio-skills, mapped canonical workflows per domain, proposed ideal tool sets from scratch, then self-compared against current implementation. Research documents in `kevin_notes/refactor_core_tools/`.

**Key numbers**:
- Current: ~82 domain tools across all agents
- Proposed: ~111 domain tools (+29 net new)
- New agents: 1 (variant_analysis_expert) + 1 new package (lobster-metabolomics)
- Bugs to fix: 17 (2 crash, 1 data corruption, 14 others)
- Decisions validated: 10 (D1-D10)

**Existing infrastructure for metabolomics**: MetabolomicsAdapter (complete), MetabolomicsSchema (complete), MetaboLightsProvider (complete), MetaboLightsDownloadService (complete), OmicsTypeRegistry entry (registered). Services and agent need to be built.

## Constraints

- **Tool count**: 8-15 tools per agent — LLM performance degrades beyond ~15
- **3-tuple pattern**: Every service returns `(AnnData, Dict, AnalysisStep)` — non-negotiable
- **IR mandatory**: Every tool must pass `ir=` to `log_tool_usage()` — provenance tracking
- **PEP 420**: No `lobster/__init__.py` — namespace package requirement
- **AGENT_CONFIG first**: Define at module top before heavy imports — <50ms entry point discovery
- **Backward compatibility**: Existing `lobster-custom-*` packages must continue working
- **No pyproject.toml edits**: Dependency changes go through humans
- **ComponentRegistry**: Agent/service discovery via entry points only — no hardcoded registries

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| D1: New variant_analysis_expert child agent | Clinical workflows need different knowledgebase lookups than population-scale GWAS | — Pending |
| D2: No separate affinity proteomics agent | Downstream analysis identical for MS and affinity; PlatformConfig handles dual mode | — Pending |
| D3: New lobster-metabolomics package | 10 tools, linear workflow, significant existing infrastructure | — Pending |
| D4: Single transcriptomics parent for SC + bulk | Auto-detection works; shared DE child; 8 bulk-specific tools added | — Pending |
| D5: Merge redundant DE tools (3→2) | Three DE runners confuses LLM; one simple + one formula-based is sufficient | — Pending |
| D6: Deprecate interactive terminal tools | Rich UI doesn't work in cloud/API/LLM-agent context | — Pending |
| D7: Merge list_modalities + get_modality_info | Reduces LLM tool selection confusion; pilot in genomics | — Pending |
| D8: Create import_proteomics_data tool | Parsers exist (MaxQuant, DIA-NN, Spectronaut) but unreachable by LLM | — Pending |
| D9: Add PTM analysis tools (3 across agents) | Phosphoproteomics >30% of published MS proteomics; currently zero capability | — Pending |
| D10: Tool taxonomy decorator (@tool_meta) | Deferred to P3; apply incrementally to new/modified tools | — Pending |
| Metabolomics: structure for future children | Plan package layout to support annotation_expert extraction later | — Pending |
| Include all 17 bugs in project scope | Fix alongside tool refactoring in same phases | — Pending |

| Docs with domain phases | Brutalist review: stale docs if deferred to Phase 7; moved DOC-01..06 into Phases 1-6 | — Pending |
| Parallel phase execution | Brutalist review: domains are independent; Wave A (1‖2‖4‖6) → Wave B (3‖5) → Wave C (7) | — Pending |
| integrate_batches returns quality metrics | Brutalist review: batch integration is iterative; tool must return LISI/silhouette for LLM re-invocation | — Pending |

---
*Last updated: 2026-02-22 after brutalist review adjustments*
