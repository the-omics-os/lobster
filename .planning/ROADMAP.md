# Roadmap: Core Tools Refactor

## Overview

A systematic redesign of domain-specific core tools across 14 agents in Lobster AI. Every agent gets exactly the right tools — no overlap, no gaps, no wrong abstraction level. The refactor adds 29 new tools (+35% increase), creates a new variant_analysis_expert child agent, ships a complete lobster-metabolomics package, and fixes 17 bugs found during research. Each phase tackles one domain's complete tool set, from parent to children, ensuring the LLM picks the correct tool every time and produces reliable, reproducible science.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Genomics Domain** - Complete genomics parent improvements + new variant_analysis_expert child agent + prompt update
- [ ] **Phase 2: Transcriptomics Parent** - Unified SC + bulk transcriptomics parent with 22 tools + prompt update
- [x] **Phase 3: Transcriptomics Children** - Annotation expert + DE analysis expert enhancements + prompt updates (completed 2026-02-23)
- [ ] **Phase 4: MS Proteomics Core** - Parent improvements, import tools, PTM basics, batch correction + prompt update
- [ ] **Phase 5: Proteomics Children & Affinity** - DE child, biomarker child, affinity-specific tools + prompt update
- [ ] **Phase 6: Metabolomics Package** - New lobster-metabolomics package with 10 core tools + prompt creation
- [ ] **Phase 7: Cross-Cutting Infrastructure** - Skills docs, @tool_meta foundation, remaining cross-cutting bug fixes

## Dependency Graph & Execution Waves

Domains are independent. Phases only block where parent→child relationships exist.

```
Wave A (independent parents + new package):
  Phase 1: Genomics         ─┐
  Phase 2: Transcriptomics  ─┼─ can run in parallel
  Phase 4: MS Proteomics    ─┤
  Phase 6: Metabolomics     ─┘

Wave B (children that depend on their parents):
  Phase 3: Tx Children      ── depends on Phase 2
  Phase 5: Prot Children    ── depends on Phase 4

Wave C (cross-cutting):
  Phase 7: Infrastructure   ── depends on all above
```

**Recommended execution order** (solo-dev / single context):
1 → 2 → 3 → 4 → 5 → 6 → 7

**Parallel execution** (multi-worktree / multiple agents):
Wave A: 1 ‖ 2 ‖ 4 ‖ 6 → Wave B: 3 ‖ 5 → Wave C: 7

## Phase Details

### Phase 1: Genomics Domain
**Goal**: GWAS pipeline completion + clinical genomics capability via variant_analysis_expert
**Depends on**: Nothing (Wave A)
**Requirements**: GEN-01, GEN-02, GEN-03, GEN-04, GEN-05, GEN-06, GEN-07, GEN-08, GEN-09, GEN-10, GEN-11, GEN-12, GEN-13, GEN-14, GEN-15, GEN-16, DOC-01
**Success Criteria** (what must be TRUE):
  1. User can run complete GWAS pipeline: load VCF → LD prune → PCA → kinship → GWAS → clump results
  2. User can hand off significant GWAS hits to variant_analysis_expert for clinical interpretation
  3. User can query single variants by rsID/coordinates and get comprehensive annotations (VEP, gnomAD, ClinVar)
  4. User can prioritize variant lists by consequence severity, population frequency, and pathogenicity
  5. All genomics tools produce provenance records (BUG-06 fixed); genomics_expert prompt updated for new tools + handoff
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Service layer: GWASService methods (LD prune, kinship, clump) + VariantAnnotationService methods (normalize, frequencies, clinical, prioritize) + IR fix + summarize_modality factory
- [ ] 01-02-PLAN.md — Parent agent refactoring: wire new tools, remove relocated tools, merge helpers, add child_agents config, update prompt
- [ ] 01-03-PLAN.md — Child agent creation: variant_analysis_expert with 8 tools, child prompt, entry point registration

### Phase 2: Transcriptomics Parent
**Goal**: Unified transcriptomics_expert handles both SC and bulk RNA-seq with appropriate tool routing
**Depends on**: Nothing (Wave A)
**Requirements**: SCT-01, SCT-02, SCT-03, SCT-04, SCT-05, SCT-06, SCT-07, SCT-08, BLK-01, BLK-02, BLK-03, BLK-04, BLK-05, BLK-06, BLK-07, BLK-08, DOC-02
**Success Criteria** (what must be TRUE):
  1. User can import bulk RNA-seq counts from Salmon/kallisto/featureCounts and merge with metadata
  2. User can run SC QC with doublet detection (Scrublet) and multi-sample batch integration (Harmony/scVI)
  3. `integrate_batches` returns integration quality metrics (LISI, silhouette) so LLM can re-invoke with different params
  4. User can assess bulk sample quality (PCA outliers, batch quantification) before DE handoff
  5. BUG-01 fixed; transcriptomics_expert prompt updated for bulk routing + new SC tools
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Service layer: EnhancedSingleCellService (integrate_batches, compute_trajectory) + BulkPreprocessingService (assess, filter, normalize, detect batch)
- [ ] 02-02-PLAN.md — SC agent updates: 3 new SC tools (detect_doublets, integrate_batches, compute_trajectory) + 4 tool renames + BUG-01 fix
- [ ] 02-03-PLAN.md — Bulk tools + prompt: 8 bulk RNA-seq tools + prompt update with SC/bulk routing decision tree

### Phase 3: Transcriptomics Children
**Goal**: Annotation expert and DE analysis expert enhanced for production workflows
**Depends on**: Phase 2 (parent must be stable)
**Requirements**: ANN-01, ANN-02, ANN-03, ANN-04, DEA-01, DEA-02, DEA-03, DEA-04, DEA-05, DEA-06, DEA-07, DEA-08, DEA-09, DEA-10, DEA-11, DOC-04
**Success Criteria** (what must be TRUE):
  1. User can annotate cell types with auto, manual, or semantic methods and score gene sets for validation
  2. User can run simple DE (one tool) or formula-based DE (separate tool) without confusion
  3. User can run complete bulk DE pipeline: prepare design → DE → filter → GSEA → export publication tables
  4. Interactive terminal tools (manually_annotate_clusters_interactive, construct_de_formula_interactive) are deprecated (cloud-compatible)
  5. BUG-04 and BUG-05 fixed; de_analysis_expert prompt updated for merged DE tools + bulk additions
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Annotation expert: score_gene_set tool, rename annotate_cell_types_auto, deprecate interactive, BUG-04 + BUG-05 fixes, annotation prompt update
- [ ] 03-02-PLAN.md — DE expert refactoring: merge 3 DE tools into 2, merge formula tools, deprecate interactive, add filter + export tools, rename 2 tools
- [ ] 03-03-PLAN.md — DE bulk additions + prompt: run_bulk_de_direct, run_gsea_analysis, extract_and_export_de_results, DE prompt rewrite

### Phase 4: MS Proteomics Core
**Goal**: Proteomics parent can import MS data, handle PTMs, and batch-correct
**Depends on**: Nothing (Wave A)
**Requirements**: MSP-01, MSP-02, MSP-03, MSP-04, MSP-05, MSP-06, MSP-07, MSP-08, MSP-09, MSP-10, DOC-03
**Success Criteria** (what must be TRUE):
  1. User can import MS data from MaxQuant/DIA-NN/Spectronaut via LLM-accessible tool (BUG-07 fixed)
  2. User can import PTM site-level data (phospho/acetyl/ubiquitin) and normalize to protein abundance
  3. User can batch-correct MS data (ComBat/median centering) and summarize peptides to proteins for TMT
  4. BUG-03, BUG-08, BUG-09, BUG-10 fixed
  5. Proteomics_expert prompt updated for import tools + PTM workflow + affinity tool additions
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Proteomics Children & Affinity
**Goal**: Proteomics DE and biomarker children enhanced; affinity proteomics fully supported
**Depends on**: Phase 4 (parent must be stable)
**Requirements**: PDE-01, PDE-02, PDE-03, PDE-04, PDE-05, BIO-01, BIO-02, BIO-03, AFP-01, AFP-02, AFP-03, AFP-04, AFP-05, AFP-06, BUG-13, DOC-05
**Success Criteria** (what must be TRUE):
  1. User can run complete proteomics DE with pathway enrichment (GO/Reactome/KEGG) and PPI networks (STRING)
  2. User can run differential PTM analysis with kinase enrichment (KSEA) for phosphoproteomics
  3. User can select biomarker panels (LASSO/stability/Boruta), evaluate with nested CV, and extract hub proteins
  4. User can import affinity data (Olink/SomaScan/Luminex), assess LOD quality, and normalize with bridge samples
  5. BUG-02, BUG-13 fixed; biomarker_discovery_expert prompt updated for panel selection tools
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Metabolomics Package
**Goal**: Complete lobster-metabolomics package ships with 10 core tools
**Depends on**: Nothing (Wave A)
**Requirements**: MET-01, MET-02, MET-03, MET-04, MET-05, MET-06, MET-07, MET-08, MET-09, MET-10, MET-11, MET-12, MET-13, MET-14, MET-15, MET-16, MET-17, DOC-06
**Success Criteria** (what must be TRUE):
  1. User can run complete metabolomics pipeline: import LC-MS/GC-MS/NMR → QC → filter → impute → normalize → batch correct
  2. User can run univariate statistics and multivariate analysis (PCA/PLS-DA/OPLS-DA)
  3. User can annotate metabolites via m/z matching (HMDB/KEGG) with MSI confidence levels
  4. User can analyze lipid classes and run pathway enrichment
  5. BUG-14 fixed; metabolomics_expert prompt created; package structured for future child agents
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Cross-Cutting Infrastructure
**Goal**: Skills docs updated, @tool_meta foundation in place, remaining cross-cutting bugs fixed
**Depends on**: All previous phases (Wave C)
**Requirements**: DOC-07, INF-01, BUG-15, BUG-16, BUG-17
**Success Criteria** (what must be TRUE):
  1. Skills (lobster-dev) reference documentation updated for new architecture patterns
  2. @tool_meta decorator foundation implemented and applied to all new tools from Phases 1-6
  3. BUG-15, BUG-16, BUG-17 fixed (PCA LD pruning defaults, bulk terminology, modality listing efficiency)
  4. Tool taxonomy system ready for incremental rollout to existing tools (v2 future work)
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

## Progress

**Execution Order:**
Solo: 1 → 2 → 3 → 4 → 5 → 6 → 7
Parallel: (1 ‖ 2 ‖ 4 ‖ 6) → (3 ‖ 5) → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Genomics Domain | 0/TBD | Not started | - |
| 2. Transcriptomics Parent | 0/TBD | Not started | - |
| 3. Transcriptomics Children | 0/TBD | Complete    | 2026-02-23 |
| 4. MS Proteomics Core | 0/TBD | Not started | - |
| 5. Proteomics Children & Affinity | 0/TBD | Not started | - |
| 6. Metabolomics Package | 0/TBD | Not started | - |
| 7. Cross-Cutting Infrastructure | 0/TBD | Not started | - |
