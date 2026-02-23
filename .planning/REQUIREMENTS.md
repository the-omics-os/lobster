# Requirements: Core Tools Refactor

**Defined:** 2026-02-22
**Core Value:** Every agent has exactly the right tools — no overlap, no gaps, no wrong abstraction level

## v1 Requirements

### Genomics Parent (genomics_expert)

- [x] **GEN-01**: Add `ld_prune` tool — standalone LD pruning as prerequisite for PCA, GWAS, admixture
- [x] **GEN-02**: Add `compute_kinship` tool — pairwise kinship matrix and related-pair flagging
- [x] **GEN-03**: Add `clump_results` tool — LD-clump GWAS results into independent loci
- [x] **GEN-04**: Merge `list_modalities` + `get_modality_info` into `summarize_modality`
- [x] **GEN-05**: Move `predict_variant_consequences` from parent to variant_analysis_expert child
- [x] **GEN-06**: Move `get_ensembl_sequence` from parent to variant_analysis_expert child
- [x] **GEN-07**: Add IR (provenance) to `load_vcf` and `load_plink` (BUG-06 fix)

### Genomics Child (variant_analysis_expert — NEW)

- [x] **GEN-08**: Create variant_analysis_expert child agent with modular folder structure
- [x] **GEN-09**: Implement `normalize_variants` tool — left-align indels, split multiallelic
- [x] **GEN-10**: Implement `predict_consequences` tool — VEP batch annotation with SIFT/PolyPhen/CADD
- [x] **GEN-11**: Implement `query_population_frequencies` tool — gnomAD allele frequency lookup
- [x] **GEN-12**: Implement `query_clinical_databases` tool — ClinVar pathogenicity and disease associations
- [x] **GEN-13**: Implement `prioritize_variants` tool — rank by consequence severity + frequency + pathogenicity
- [x] **GEN-14**: Implement `lookup_variant` tool — single-variant comprehensive lookup by rsID/coordinates
- [x] **GEN-15**: Implement `retrieve_sequence` tool — Ensembl sequence fetch (relocated from parent)
- [x] **GEN-16**: Implement `summarize_modality` tool — shared with parent

### SC Transcriptomics Parent (transcriptomics_expert)

- [x] **SCT-01**: Add `detect_doublets` tool — Scrublet doublet detection on raw counts
- [x] **SCT-02**: Add `integrate_batches` tool — Harmony/scVI batch integration for multi-sample. Must return integration quality metrics (LISI, silhouette) so LLM can re-invoke with different parameters for iterative refinement.
- [x] **SCT-03**: Add `compute_trajectory` tool — DPT/PAGA pseudotime trajectory inference
- [x] **SCT-04**: Rename `filter_and_normalize_modality` → `filter_and_normalize`
- [x] **SCT-05**: Rename `select_highly_variable_genes` → `select_variable_features`
- [x] **SCT-06**: Rename `cluster_modality` → `cluster_cells`
- [x] **SCT-07**: Rename `find_marker_genes_for_clusters` → `find_marker_genes`
- [x] **SCT-08**: Fix BUG-01: `subcluster_cells` TypeError on `len(None)`

### Bulk Transcriptomics (on transcriptomics_expert parent)

- [x] **BLK-01**: Add `import_bulk_counts` tool — Salmon/kallisto/featureCounts import
- [x] **BLK-02**: Add `merge_sample_metadata` tool — join external metadata with count matrix
- [x] **BLK-03**: Add `assess_bulk_sample_quality` tool — PCA outlier detection, sample correlation, batch quantification
- [x] **BLK-04**: Add `filter_bulk_genes` tool — bulk-appropriate gene filtering (min counts in min samples)
- [x] **BLK-05**: Add `normalize_bulk_counts` tool — DESeq2 size factors, VST, CPM
- [x] **BLK-06**: Add `detect_batch_effects` tool — variance decomposition, correction recommendation
- [x] **BLK-07**: Add `convert_gene_identifiers` tool — Ensembl/Symbol/Entrez mapping
- [x] **BLK-08**: Add `prepare_bulk_for_de` tool — validation checkpoint before DE handoff

### Annotation Expert Child (transcriptomics)

- [x] **ANN-01**: Add `score_gene_set` tool — gene set scoring via sc.tl.score_genes
- [x] **ANN-02**: Rename `annotate_cell_types` → `annotate_cell_types_auto`
- [x] **ANN-03**: Fix BUG-04: Replace try/except ImportError with component_registry for VectorSearchService
- [x] **ANN-04**: Fix BUG-05: Use data_manager.store_modality() instead of direct dict assignment

### DE Analysis Expert Child (transcriptomics)

- [x] **DEA-01**: Merge 3 DE tools into 2: `run_differential_expression` (simple) + `run_de_with_formula` (advanced)
- [x] **DEA-02**: Merge `construct_de_formula_interactive` + `suggest_formula_for_design` → `suggest_de_formula`
- [x] **DEA-03**: Deprecate `manually_annotate_clusters_interactive` (BUG-11)
- [x] **DEA-04**: Deprecate `construct_de_formula_interactive` (BUG-12)
- [x] **DEA-05**: Add `filter_de_results` tool — standalone result filtering
- [x] **DEA-06**: Add `export_de_results` tool — publication-ready CSV/Excel export
- [x] **DEA-07**: Rename `prepare_differential_expression_design` → `prepare_de_design`
- [x] **DEA-08**: Rename `run_pathway_enrichment_analysis` → `run_pathway_enrichment`
- [x] **DEA-09**: Add `run_bulk_de_direct` tool — one-shot DE for simple bulk comparisons
- [x] **DEA-10**: Add `run_gsea_analysis` tool — ranked gene set enrichment
- [x] **DEA-11**: Add `extract_and_export_de_results` tool — publication-ready tables with LFC shrinkage

### MS Proteomics Parent (proteomics_expert)

- [x] **MSP-01**: Add `import_proteomics_data` tool — wrap MaxQuantParser/DIANNParser/SpectronautParser (BUG-07 fix)
- [x] **MSP-02**: Add `import_ptm_sites` tool — phospho/acetyl/ubiquitin site-level import
- [x] **MSP-03**: Add `correct_batch_effects` tool — ComBat/median centering for MS batch correction
- [x] **MSP-04**: Add `summarize_peptide_to_protein` tool — peptide/PSM to protein rollup for TMT
- [x] **MSP-05**: Add `normalize_ptm_to_protein` tool — separate PTM regulation from protein abundance
- [x] **MSP-06**: Merge `add_peptide_mapping` into `import_proteomics_data`
- [x] **MSP-07**: Fix BUG-03: validate_antibody_specificity inflated correlations (use pairwise-complete)
- [x] **MSP-08**: Fix BUG-10: detect_platform_type silent default to mass_spec (return "unknown")
- [x] **MSP-09**: Fix BUG-08: filter_proteomics_data dead affinity branch
- [x] **MSP-10**: Fix BUG-09: remove unused cross_reactivity_threshold config

### Proteomics DE Child (proteomics_de_analysis_expert)

- [ ] **PDE-01**: Add `run_pathway_enrichment` tool — GO/Reactome/KEGG on DE results
- [ ] **PDE-02**: Add `run_differential_ptm_analysis` tool — site-level DE with protein adjustment
- [ ] **PDE-03**: Add `run_kinase_enrichment` tool — KSEA for phosphoproteomics
- [ ] **PDE-04**: Add `run_string_network_analysis` tool — STRING PPI network queries
- [ ] **PDE-05**: Fix BUG-02: UnboundLocalError on min_group in DE

### Biomarker Discovery Child (biomarker_discovery_expert)

- [ ] **BIO-01**: Add `select_biomarker_panel` tool — multi-method feature selection (LASSO, stability, Boruta)
- [ ] **BIO-02**: Add `evaluate_biomarker_panel` tool — nested CV model evaluation with AUC
- [ ] **BIO-03**: Add `extract_hub_proteins` tool — post-WGCNA hub protein extraction

### Affinity Proteomics (on proteomics_expert parent)

- [ ] **AFP-01**: Add `import_affinity_data` tool — Olink NPX/SomaScan ADAT/Luminex MFI parsing
- [ ] **AFP-02**: Add `assess_lod_quality` tool — LOD-based quality assessment per platform
- [ ] **AFP-03**: Add `normalize_bridge_samples` tool — inter-plate normalization via bridge samples
- [ ] **AFP-04**: Add `assess_cross_platform_concordance` tool — cross-platform protein comparison
- [ ] **AFP-05**: Enhance `assess_proteomics_quality` for affinity-specific LOD metrics
- [ ] **AFP-06**: Enhance `check_proteomics_status` for affinity-specific metadata display

### Metabolomics (NEW lobster-metabolomics package)

- [ ] **MET-01**: Create `packages/lobster-metabolomics/` package structure (structured for future child agents)
- [ ] **MET-02**: Create `MetabolomicsQualityService` — RSD, TIC, QC sample evaluation
- [ ] **MET-03**: Create `MetabolomicsPreprocessingService` — filter, impute, normalize (PQN/TIC/IS), batch correct
- [ ] **MET-04**: Create `MetabolomicsAnalysisService` — univariate stats, PLS-DA, fold change, pathway enrichment
- [ ] **MET-05**: Create `MetabolomicsAnnotationService` — m/z matching to HMDB/KEGG, MSI levels
- [ ] **MET-06**: Implement `assess_metabolomics_quality` tool
- [ ] **MET-07**: Implement `filter_metabolomics_features` tool
- [ ] **MET-08**: Implement `handle_missing_values` tool
- [ ] **MET-09**: Implement `normalize_metabolomics` tool
- [ ] **MET-10**: Implement `correct_batch_effects` tool
- [ ] **MET-11**: Implement `run_metabolomics_statistics` tool
- [ ] **MET-12**: Implement `run_multivariate_analysis` tool (PCA/PLS-DA/OPLS-DA)
- [ ] **MET-13**: Implement `annotate_metabolites` tool
- [ ] **MET-14**: Implement `analyze_lipid_classes` tool
- [ ] **MET-15**: Implement `run_pathway_enrichment` tool
- [ ] **MET-16**: Register metabolomics_expert via entry points in pyproject.toml
- [ ] **MET-17**: Fix BUG-14: metabolomics schema sparse matrix zero-checking

### Bug Fixes (Cross-Cutting)

- [ ] **BUG-13**: Add post-correction validation to `correct_plate_effects`
- [ ] **BUG-15**: Fix PCA defaults to no LD pruning in genomics (default recommend standalone tool)
- [ ] **BUG-16**: Fix bulk data getting SC terminology in shared tools
- [ ] **BUG-17**: Fix `list_modalities` loading all AnnData to check type

### Prompts & Documentation

- [x] **DOC-01**: Update genomics_expert prompt for new tool inventory + variant_analysis handoff
- [x] **DOC-02**: Update transcriptomics_expert prompt for bulk-specific tool routing
- [ ] **DOC-03**: Update proteomics_expert prompt for import tools + PTM + affinity tools
- [x] **DOC-04**: Update de_analysis_expert prompt for merged DE tools + bulk additions
- [ ] **DOC-05**: Update biomarker_discovery_expert prompt for panel selection tools
- [ ] **DOC-06**: Create metabolomics_expert prompt
- [ ] **DOC-07**: Update skills/lobster-dev references for new architecture

### Infrastructure

- [ ] **INF-01**: Implement @tool_meta decorator foundation (D10) — apply to new tools only

## v2 Requirements

### Future Genomics
- **GEN-V2-01**: Fine-mapping tools (SuSiE wrapper)
- **GEN-V2-02**: PRS calculation tools
- **GEN-V2-03**: Selection statistics (Fst, Tajima's D) via scikit-allel

### Future Metabolomics
- **MET-V2-01**: Targeted metabolomics child agent (standard curves, absolute quantification)
- **MET-V2-02**: Metabolomics annotation child agent (SIRIUS, MetFrag, in-silico fragmentation)
- **MET-V2-03**: GC-MS specific tools (library matching, derivatization artifact removal)
- **MET-V2-04**: Multi-platform integration workflow

### Future Proteomics
- **MSP-V2-01**: DIA-specific workflow tools (library-free mode, spectral library management)
- **MSP-V2-02**: TMT-specific tools beyond peptide-to-protein rollup

### Future Cross-Domain
- **INF-V2-01**: Full @tool_meta rollout to all existing tools
- **INF-V2-02**: Automated tool redundancy detection via taxonomy

## Out of Scope

| Feature | Reason |
|---------|--------|
| Raw data preprocessing (XCMS, STAR, MaxQuant) | Lobster receives processed feature tables, not raw instrument data |
| Cell-cell communication (CellChat, LIANA) | Advanced analysis beyond core tool set; future expansion |
| Separate affinity proteomics agent | D2: downstream analysis identical; PlatformConfig handles dual mode |
| Separate population_genetics_expert | D1: PCA/kinship/LD pruning too coupled with GWAS pipeline |
| NMR-specific processing | Defer to post-MVP metabolomics expansion |
| Spatial transcriptomics | Different omics domain; separate project |
| scATAC-seq tools | Different omics domain; separate project |
| Perturb-seq / CRISPR screen tools | Specialized; separate project |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GEN-01 | Phase 1 | Complete |
| GEN-02 | Phase 1 | Complete |
| GEN-03 | Phase 1 | Complete |
| GEN-04 | Phase 1 | Complete |
| GEN-05 | Phase 1 | Complete |
| GEN-06 | Phase 1 | Complete |
| GEN-07 | Phase 1 | Complete |
| GEN-08 | Phase 1 | Complete |
| GEN-09 | Phase 1 | Complete |
| GEN-10 | Phase 1 | Complete |
| GEN-11 | Phase 1 | Complete |
| GEN-12 | Phase 1 | Complete |
| GEN-13 | Phase 1 | Complete |
| GEN-14 | Phase 1 | Complete |
| GEN-15 | Phase 1 | Complete |
| GEN-16 | Phase 1 | Complete |
| SCT-01 | Phase 2 | Complete |
| SCT-02 | Phase 2 | Complete |
| SCT-03 | Phase 2 | Complete |
| SCT-04 | Phase 2 | Complete |
| SCT-05 | Phase 2 | Complete |
| SCT-06 | Phase 2 | Complete |
| SCT-07 | Phase 2 | Complete |
| SCT-08 | Phase 2 | Complete |
| BLK-01 | Phase 2 | Complete |
| BLK-02 | Phase 2 | Complete |
| BLK-03 | Phase 2 | Complete |
| BLK-04 | Phase 2 | Complete |
| BLK-05 | Phase 2 | Complete |
| BLK-06 | Phase 2 | Complete |
| BLK-07 | Phase 2 | Complete |
| BLK-08 | Phase 2 | Complete |
| ANN-01 | Phase 3 | Complete |
| ANN-02 | Phase 3 | Complete |
| ANN-03 | Phase 3 | Complete |
| ANN-04 | Phase 3 | Complete |
| DEA-01 | Phase 3 | Complete |
| DEA-02 | Phase 3 | Complete |
| DEA-03 | Phase 3 | Complete |
| DEA-04 | Phase 3 | Complete |
| DEA-05 | Phase 3 | Complete |
| DEA-06 | Phase 3 | Complete |
| DEA-07 | Phase 3 | Complete |
| DEA-08 | Phase 3 | Complete |
| DEA-09 | Phase 3 | Complete |
| DEA-10 | Phase 3 | Complete |
| DEA-11 | Phase 3 | Complete |
| MSP-01 | Phase 4 | Complete |
| MSP-02 | Phase 4 | Complete |
| MSP-03 | Phase 4 | Complete |
| MSP-04 | Phase 4 | Complete |
| MSP-05 | Phase 4 | Complete |
| MSP-06 | Phase 4 | Complete |
| MSP-07 | Phase 4 | Complete |
| MSP-08 | Phase 4 | Complete |
| MSP-09 | Phase 4 | Complete |
| MSP-10 | Phase 4 | Complete |
| PDE-01 | Phase 5 | Pending |
| PDE-02 | Phase 5 | Pending |
| PDE-03 | Phase 5 | Pending |
| PDE-04 | Phase 5 | Pending |
| PDE-05 | Phase 5 | Pending |
| BIO-01 | Phase 5 | Pending |
| BIO-02 | Phase 5 | Pending |
| BIO-03 | Phase 5 | Pending |
| AFP-01 | Phase 5 | Pending |
| AFP-02 | Phase 5 | Pending |
| AFP-03 | Phase 5 | Pending |
| AFP-04 | Phase 5 | Pending |
| AFP-05 | Phase 5 | Pending |
| AFP-06 | Phase 5 | Pending |
| MET-01 | Phase 6 | Pending |
| MET-02 | Phase 6 | Pending |
| MET-03 | Phase 6 | Pending |
| MET-04 | Phase 6 | Pending |
| MET-05 | Phase 6 | Pending |
| MET-06 | Phase 6 | Pending |
| MET-07 | Phase 6 | Pending |
| MET-08 | Phase 6 | Pending |
| MET-09 | Phase 6 | Pending |
| MET-10 | Phase 6 | Pending |
| MET-11 | Phase 6 | Pending |
| MET-12 | Phase 6 | Pending |
| MET-13 | Phase 6 | Pending |
| MET-14 | Phase 6 | Pending |
| MET-15 | Phase 6 | Pending |
| MET-16 | Phase 6 | Pending |
| MET-17 | Phase 6 | Pending |
| DOC-01 | Phase 1 | Complete |
| DOC-02 | Phase 2 | Complete |
| DOC-03 | Phase 4 | Pending |
| DOC-04 | Phase 3 | Complete |
| DOC-05 | Phase 5 | Pending |
| DOC-06 | Phase 6 | Pending |
| DOC-07 | Phase 7 | Pending |
| INF-01 | Phase 7 | Pending |
| BUG-13 | Phase 5 | Pending |
| BUG-15 | Phase 7 | Pending |
| BUG-16 | Phase 7 | Pending |
| BUG-17 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 76 total
- Mapped to phases: 76
- Unmapped: 0

---
*Requirements defined: 2026-02-22*
*Last updated: 2026-02-22 after brutalist review adjustments (docs moved to domain phases, parallel execution enabled)*
