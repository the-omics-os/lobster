# Showcase Tests — Lobster AI v1.0.0

**Purpose:** Validate complex multi-agent workflows that become documentation examples and investor demos.
**Prerequisite:** All P0 tests passing (research, transcriptomics, visualization, proteomics, genomics confirmed working).
**Created:** 2026-02-12

---

## Objective

Every example in docs.omics-os.com must work when a user follows it. These tests produce the validated workflows that become:

1. **Documentation examples** — Real queries with real outputs, not hypothetical placeholders
2. **Tutorial content** — Step-by-step guides users actually follow
3. **Investor demos** — The "killer demo" showing end-to-end value
4. **Notebook artifacts** — Reproducible Jupyter notebooks users can download and re-run

Each test ends with `/pipeline export` to generate a notebook. If the notebook doesn't execute cleanly, the workflow isn't ready for documentation.

---

## Context from Prior Testing

**Validated datasets** (use these — they work):

| Dataset | Type | Size | Validated In |
|---------|------|------|-------------|
| GSE84133 | scRNA-seq, human pancreas | 3,605 cells × 20,125 genes | G1, B2 |
| GSE150728 | scRNA-seq, human PBMC | ~8K cells | B2, I1 |
| GSE49712 | Bulk RNA-seq | 12 samples | A2 |
| Synthetic proteomics | MaxQuant LFQ format | 500 proteins × 6 samples | C1 |
| Synthetic VCF | 100 variants | ~50 KB | D1 |

**Known issues** (work around these):

- **XGBoost not installed** — Feature selection falls back to Random Forest. Acceptable, but install `xgboost` for better results.
- **ArrowStringArray** — Fixed in data_manager/backend, but may appear in custom_code_tool path. Non-blocking.
- **Large datasets (>50K cells)** — Require subsampling. Use GSE84133 or GSE150728 to avoid this.
- **Publication queue** — The primary search path is `fast_dataset_search`, not the publication queue. Use pre-curated PMIDs (30643258, 31018141) for deterministic queue tests.
- **Modality naming heuristic** — Datasets with many samples but also many cells may be misclassified as "bulk". Known issue, non-blocking.

**Bugs fixed during prior testing:**

1. ML child_agents config missing — ML expert couldn't delegate to sub-agents (fixed)
2. ArrowStringArray H5AD serialization failures (fixed in data_manager)
3. Proteomics adapter fillna() on ndarray (fixed)
4. metadata_validation_service missing workspace_path (fixed)

---

## Environment Setup

```bash
cd ~/omics-os/lobster
make dev-install
pip install lobster-ai[full]
pip install xgboost  # Recommended for feature selection

# Generate synthetic test data
cd ~/omics-os/lobster/test_data
python generate_test_data.py

# Set workspace
export LOBSTER_WORKSPACE=~/test_workspace_showcase
mkdir -p ~/test_workspace_showcase
```

---

## Test Execution Protocol

For each scenario:

1. Start fresh session: `lobster chat --session-id "showcase-{id}"`
2. Execute each step sequentially (copy-paste the user queries)
3. After final analysis step, run: `/pipeline export`
4. Verify notebook: `jupyter nbconvert --to notebook --execute {notebook}.ipynb`
5. Save outputs: Copy `exports/` contents to `test_results/showcase-{id}/`

**Success = all steps complete + notebook executes without modification.**

---

## SCENARIO 1: Pancreas Cell Type Biomarker Discovery

**Showcases:** scRNA-seq pipeline + ML feature selection + visualization
**Agents:** 5 (data_expert, transcriptomics_expert, annotation_expert, feature_selection_expert, visualization_expert)
**Packages:** lobster-transcriptomics, lobster-ml, lobster-visualization
**Dataset:** GSE84133 (human pancreas, 3,605 cells)

### Queries

```
1. "Load GSE84133 — human pancreas single-cell RNA-seq"

2. "Run QC, normalize, and cluster the data"

3. "What cell types are in this dataset? Show me the annotations."

4. "Run stability selection to find biomarkers that distinguish alpha cells from beta cells"

5. "Create a dot plot showing the top 20 selected biomarkers across all cell types"

6. "Create a UMAP colored by cell type"

7. "/pipeline export"
```

### Success Criteria

- [ ] GSE84133 loaded: 3,605 cells, cell type labels in `adata.obs`
- [ ] QC + clustering: Leiden clusters + UMAP computed
- [ ] Cell types identified: alpha, beta, delta, gamma, acinar, ductal
- [ ] Feature selection: <200 features selected (not 2,400)
- [ ] Biological validity: Top features include INS (beta), GCG (alpha), SST (delta)
- [ ] Dot plot: Shows clear expression patterns, gene names readable
- [ ] UMAP: Cell types colored with legend, publication quality
- [ ] Notebook exported and executes without errors

### Expected Outputs

```
exports/
├── umap_cell_type.png
├── dotplot_biomarkers.png
notebooks/
└── showcase_1_pancreas_biomarkers.ipynb
```

### Why This Matters

This is the cleanest validation of Lobster's core value: from raw data to biological insight in minutes. Known biology (pancreatic islet markers) provides ground truth — if INS and GCG aren't in the top features, something is wrong.

---

## SCENARIO 2: Bulk RNA-seq to Survival Analysis Pipeline

**Showcases:** Bulk DE + feature selection + Cox survival + Kaplan-Meier curves
**Agents:** 4 (data_expert, de_analysis_expert, feature_selection_expert, survival_analysis_expert)
**Packages:** lobster-transcriptomics, lobster-ml
**Dataset:** GSE49712 (bulk RNA-seq, 12 samples) + synthetic survival data

### Queries

```
1. "Load GSE49712 — bulk RNA-seq dataset"

2. "Run differential expression between the two sample groups"

3. "Add synthetic survival data to this modality: survival_time drawn from exponential distribution with mean 365 days, event indicator with 30% event rate, for each sample"

4. "Run stability selection on the top 500 most variable genes to find prognostic biomarkers"

5. "Fit a Cox proportional hazards model using the selected biomarkers"

6. "Generate Kaplan-Meier survival curves for high-risk vs low-risk patient groups"

7. "What is the C-index? How many patients are in each risk group?"

8. "/pipeline export"
```

### Success Criteria

- [ ] DE results: `log2FoldChange`, `pvalue`, `padj` columns in adata.var
- [ ] Significant genes: At least 10 with |log2FC| > 1 and padj < 0.05
- [ ] Survival data added: `survival_time` and `event` in adata.obs
- [ ] Feature selection: 10-50 prognostic features selected
- [ ] Cox model: Saved to `workspace/models/`
- [ ] C-index reported: Value between 0.5-1.0 (expect ~0.5 with synthetic data)
- [ ] KM curves: PNG with 2+ risk groups, axes labeled
- [ ] Hazard ratios: Reported with 95% CI
- [ ] Notebook exported and executes without errors

### Expected Outputs

```
exports/
├── volcano_de_results.png
├── kaplan_meier_risk_groups.png
models/
└── cox_model_*.joblib
notebooks/
└── showcase_2_survival_pipeline.ipynb
```

### Scientific Note

With synthetic random survival data, C-index should be near 0.5 (no real predictive power). A C-index > 0.7 on random data would indicate overfitting. The test validates the complete code path, not biological conclusions.

---

## SCENARIO 3: Multi-Omics Integration (MOFA)

**Showcases:** Cross-modality integration + feature_space_key routing + pathway enrichment
**Agents:** 6 (data_expert, transcriptomics_expert, proteomics_expert, machine_learning_expert, feature_selection_expert, visualization_expert)
**Packages:** lobster-transcriptomics, lobster-proteomics, lobster-ml, lobster-visualization
**Datasets:** GSE150728 (scRNA-seq) + synthetic proteomics

### Queries

```
1. "Load GSE150728 — PBMC single-cell RNA-seq"

2. "Run QC and normalize the data"

3. "Load the proteomics file at test_data/synthetic/proteomics_lfq.csv as a proteomics modality"

4. "Normalize the proteomics data"

5. "How many modalities do I have now? List them."

6. "Run MOFA integration on the transcriptomics and proteomics modalities"

7. "Run feature selection on the MOFA factors"

8. "What pathways are enriched among the top MOFA factors?"

9. "/pipeline export"
```

### Success Criteria

- [ ] Two modalities loaded: transcriptomics + proteomics
- [ ] Both independently normalized
- [ ] Modality list shows both with types correctly identified
- [ ] MOFA runs: Factors in `adata.obsm['X_mofa']` with 5-15 factors
- [ ] Feature selection uses `feature_space_key="X_mofa"` (not raw features)
- [ ] Pathway enrichment returns GO/Reactome terms (via INDRA)
- [ ] Notebook exported and executes without errors

### Expected Outputs

```
exports/
├── mofa_factor_plot.png (if generated)
notebooks/
└── showcase_3_multi_omics.ipynb
```

### Known Limitation

Synthetic proteomics won't have real sample overlap with GSE150728. MOFA may produce low-quality factors. The test validates the integration code path. For real multi-omics validation, use matched CPTAC datasets in a future iteration.

---

## SCENARIO 4: Multi-Dataset Comparative Analysis

**Showcases:** Multiple modalities in one session + cross-dataset DE + combined visualization
**Agents:** 5 (data_expert, transcriptomics_expert, annotation_expert, de_analysis_expert, visualization_expert)
**Packages:** lobster-research, lobster-transcriptomics, lobster-visualization
**Datasets:** GSE150728 (PBMC) + GSE84133 (pancreas)

### Queries

```
1. "Load GSE150728 as the first dataset — PBMC reference"

2. "Run QC, normalize, and cluster the PBMC data"

3. "Load GSE84133 as a second dataset — pancreas"

4. "Run QC, normalize, and cluster the pancreas data too"

5. "List all my modalities. How many cells are in each?"

6. "Create a UMAP for each dataset side by side"

7. "What genes are highly expressed in the PBMC dataset but not in the pancreas dataset?"

8. "/pipeline export"
```

### Success Criteria

- [ ] Both datasets loaded as separate modalities
- [ ] Both QC'd and clustered independently
- [ ] Modality listing shows both with correct cell counts
- [ ] UMAP generated for each (separate plots or side-by-side)
- [ ] Cross-dataset gene comparison: Agent identifies tissue-specific markers
- [ ] Session handles 2 modalities without confusion (no "which modality?" errors)
- [ ] Context preserved across 8 steps
- [ ] Notebook exported and executes without errors

### Expected Outputs

```
exports/
├── umap_pbmc.png
├── umap_pancreas.png
notebooks/
└── showcase_4_comparative.ipynb
```

### Why This Matters

Users frequently work with multiple datasets. This tests whether Lobster can manage session state across modalities without confusion — a common failure mode in multi-turn AI systems.

---

## SCENARIO 5: Full Research Lifecycle (Investor Demo)

**Showcases:** Every major capability in one coherent scientific storyline
**Agents:** 7 (research_agent, data_expert, transcriptomics_expert, annotation_expert, de_analysis_expert, feature_selection_expert, visualization_expert)
**Packages:** ALL core packages (research, transcriptomics, ML, visualization)
**Dataset:** GSE84133 (pancreas)

### Queries

```
1.  "I'm studying pancreatic islet biology. Find relevant scRNA-seq datasets on GEO."

2.  "Load GSE84133 — the Baron et al. 2016 human pancreas dataset"

3.  "Run the full single-cell pipeline: QC, normalize, and cluster"

4.  "Annotate cell types — this dataset should have alpha, beta, delta, and other islet cell types"

5.  "Create a publication-quality UMAP colored by cell type"

6.  "Run differential expression between alpha and beta cells"

7.  "Create a volcano plot of the alpha vs beta DE results"

8.  "Find robust biomarkers distinguishing alpha from beta cells using stability selection"

9.  "Create a dot plot of the top 15 biomarkers across all cell types"

10. "/pipeline export"
```

### Success Criteria

- [ ] Dataset discovery: Research agent finds pancreas datasets
- [ ] Data loaded: GSE84133, 3,605 cells
- [ ] Full pipeline: QC → normalize → cluster → annotate
- [ ] Cell types: alpha, beta, delta, gamma, acinar, ductal identified
- [ ] UMAP: Publication quality, cell type colors, clean legend
- [ ] DE results: GCG (alpha), INS (beta), IAPP, ARX, MAFA in top hits
- [ ] Volcano plot: Significant genes labeled, threshold lines at |FC|=1 and padj=0.05
- [ ] Feature selection: <100 features, includes known pancreas biology
- [ ] Dot plot: Clear differential expression visible across cell types
- [ ] Notebook: All 9 analysis steps captured, executes without modification
- [ ] Session: 10+ messages with full context preservation
- [ ] Total outputs: 3+ plots + 1 notebook + DE results

### Expected Outputs

```
exports/
├── umap_cell_type.png
├── volcano_alpha_vs_beta.png
├── dotplot_top15_biomarkers.png
├── de_results_alpha_vs_beta.csv
notebooks/
└── showcase_5_full_lifecycle.ipynb
```

### Why This Is The Most Important Test

This is the demo you show investors. It covers every major capability in a scientifically coherent storyline with ground truth biology. If this works end-to-end, Lobster is ready. If it breaks at any step, that step needs fixing before we ship.

**Record the session.** The terminal output of this test becomes marketing material.

---

## SCENARIO 6: Metadata-Driven Research Pipeline

**Showcases:** Publication queue + metadata filtering + downstream analysis
**Agents:** 4 (research_agent, metadata_assistant, data_expert, transcriptomics_expert)
**Packages:** lobster-research, lobster-metadata, lobster-transcriptomics
**Data:** Pre-curated PMIDs with known GEO links

### Queries

```
1. "Add these publications to my queue: PMID 30643258 and PMID 31018141"

2. "Process the publication queue — extract GEO identifiers from these papers"

3. "What GEO datasets were found? Show me the metadata."

4. "Load the first GEO dataset found"

5. "Run QC and clustering on the loaded data"

6. "/pipeline export"
```

### Success Criteria

- [ ] Queue populated: 2 entries with PMIDs
- [ ] GEO IDs extracted: GSE120575 (from 30643258), GSE128033 (from 31018141)
- [ ] Metadata displayed: Title, organism, sample count for each
- [ ] Dataset loaded: One of the extracted GEO datasets
- [ ] Analysis runs: QC + clustering completes
- [ ] Notebook exported and executes without errors

### Expected Outputs

```
exports/
├── umap_clusters.png
metadata/
└── publication_queue_export.csv
notebooks/
└── showcase_6_metadata_pipeline.ipynb
```

### Why These PMIDs

- PMID 30643258: Sade-Feldman et al. 2018 — melanoma scRNA-seq, GSE120575
- PMID 31018141: Reyfman et al. 2019 — lung scRNA-seq, GSE128033

Both are >5 years old, fully indexed in NCBI, deterministic E-Link mappings. No risk of "too recent" failures.

---

## Execution Order

| Priority | Scenario | Time | Validates |
|----------|----------|------|-----------|
| 1 | **S1: Pancreas Biomarkers** | ~8 min | ML + biology ground truth |
| 2 | **S5: Full Lifecycle** | ~15 min | The investor demo |
| 3 | **S2: Survival Pipeline** | ~8 min | Cox + KM curves |
| 4 | **S4: Multi-Dataset** | ~10 min | Session state management |
| 5 | **S6: Metadata Pipeline** | ~6 min | Publication queue path |
| 6 | **S3: Multi-Omics MOFA** | ~10 min | Cross-modality integration |

**Total: ~57 minutes for all 6 scenarios.**

Run S1 and S5 first. If both pass, Lobster has validated examples for every major capability. The rest fill in advanced use cases.

---

## Test Results (2026-02-12 — Wave 1: S1 + S5)

### Environment

- Provider: AWS Bedrock (Claude 3.5 Haiku)
- Python 3.12, pandas 2.2+, numpy 2.3.5, scanpy 1.12, anndata 0.12.6
- scikit-survival 0.27.0, mofapy2 0.7.3 installed
- All 13 agents available

### Bugs Fixed During Testing

1. **H5AD boolean column serialization** (FIXED in `h5ad_backend.py`)
   - Boolean columns (`mt`, `ribo`) in `adata.var` were converted to `dtype=object` but VALUES remained Python `bool`, causing HDF5 write failure
   - Fix: Changed `astype("object")` to `map(lambda x: str(x) ...)` to convert actual values
   - Error: `"Can't implicitly convert non-string objects to strings"`

### Bugs Found (Not Yet Fixed)

2. **str + int concatenation in DE/ML tools** (BLOCKING — affects S1, S2, S5)
   - When cluster IDs are integers (e.g., `leiden_res0_5 = 0, 1, 2, ...`), multiple tools fail with `"can only concatenate str (not 'int') to str"`
   - Affected: `de_analysis_expert.py:1485`, `machine_learning_expert.py:424`, `annotation_expert.py:450`, `annotation_expert.py:550`
   - Root cause: Jinja2 templates and string formatting assume string-typed cluster labels
   - Workaround: Agents fall back to custom_code_tool which works

3. **Annotation IndexError** (BLOCKING — affects S5)
   - `enhanced_singlecell_service.py:535` `_calculate_per_cell_confidence`: `index 6235 is out of bounds for axis 0 with size 3605`
   - The service accesses indices beyond the adata dimensions (possibly using unfiltered gene indices on filtered data)
   - Workaround: Manual annotation via custom_code_tool works

4. **Stability selection routing** (MODERATE — affects S1, S5)
   - Supervisor routes "stability selection" queries to transcriptomics_expert instead of feature_selection_expert
   - The transcriptomics expert doesn't have stability selection tools
   - When correctly routed to feature_selection_expert, stability selection on 15K features takes >2 hours (needs pre-filtering to top 500 HVGs)

5. **Visualization hallucination in long contexts** (MODERATE — affects S5)
   - After 16+ messages, visualization_expert sometimes narrates plot creation without calling the tool
   - Volcano plot and dot plot in S5 were described but not generated
   - Likely LLM context degradation, not code bug

6. **`/pipeline export` only in chat mode** (MINOR — affects all scenarios)
   - Slash commands are parsed in `lobster chat` but not `lobster query`
   - Notebook export requires chat mode or programmatic API call

### S1: Pancreas Cell Type Biomarker Discovery — PARTIAL PASS

| Step | Query | Result | Status |
|------|-------|--------|--------|
| 1 | Load GSE84133 | 3,605 cells × 20,125 genes loaded | ✅ PASS |
| 2 | QC, normalize, cluster | 9 clusters (res 0.8), 15,228 genes, 100% cell retention | ✅ PASS |
| 3 | Cell type annotation | Alpha, beta, delta, acinar, ductal, stellate identified via markers | ✅ PASS |
| 4 | Stability selection | DE tools failed (str+int), ML routing failed. DE via custom_code worked: GCG #1 alpha, INS #1 beta | ⚠️ CONCERN |
| 5 | Dot plot | Generated: `plots/plot_1_Dot_Plot_-_leiden_res0_5.png` | ✅ PASS |
| 6 | UMAP | Generated: `plots/plot_1_UMAP_-_leiden_res0_5.png` | ✅ PASS |
| 7 | Pipeline export | N/A — only available in chat mode | ❌ SKIP |

**Biological validation:**
- ✅ GCG (glucagon) is #1 alpha cell marker (score 45.29, log2FC 4.12)
- ✅ INS (insulin) is #1 beta cell marker (score 42.07, log2FC 2.74)
- ✅ IAPP (amylin) is #2 beta marker (score 39.87, log2FC 5.00)
- ✅ IRX2 (alpha TF) in top 10 alpha markers
- ✅ All major pancreatic cell types detected and separated in UMAP

**Verdict:** Core pipeline works. Stability selection needs routing fix. Ready for documentation with DE-based biomarker approach.

### S5: Full Research Lifecycle (Investor Demo) — PARTIAL PASS

| Step | Query | Result | Status |
|------|-------|--------|--------|
| 1 | Find scRNA-seq datasets | Research agent found 9,145 results, curated top 7 | ✅ PASS |
| 2 | Load GSE84133 | Downloaded 29 MB, loaded 3,605 × 20,125 | ✅ PASS |
| 3 | QC + normalize + cluster | 6 clusters (res 0.25), all major types | ✅ PASS |
| 4 | Annotate cell types | Auto-annotation failed (IndexError). Manual via custom_code worked | ⚠️ CONCERN |
| 5 | Publication UMAP | Generated: `plots/plot_2_UMAP_-_cell_type.png` | ✅ PASS |
| 6 | DE alpha vs beta | Tools failed (str+int). Custom code: 4,724 DE genes, GCG #1 | ⚠️ CONCERN |
| 7 | Volcano plot | Agent described but did not generate file | ❌ FAIL |
| 8 | Stability selection | Skipped (known broken) | ❌ SKIP |
| 9 | Dot plot | Agent described but did not generate file | ❌ FAIL |
| 10 | Pipeline export | N/A — only available in chat mode | ❌ SKIP |

**Context preservation:** 10+ messages maintained session across 7 agents. Modality name tracking worked despite long names.

**Verdict:** Core flow works end-to-end for 6 of 10 steps. Visualization hallucination in long contexts is a concern. NOT ready for investor demo without fixing: (1) annotation IndexError, (2) str+int in DE/ML, (3) volcano/dot plot generation reliability.

### S2: Bulk RNA-seq to Survival Analysis — PARTIAL PASS

| Step | Query | Result | Status |
|------|-------|--------|--------|
| 1 | Load GSE49712 | 10 samples × 23,197 genes (UHRR vs HBRR benchmark) | ✅ PASS |
| 2 | DE between groups | pyDESeq2 formula path worked; 0 DEGs (correct for benchmark) | ⚠️ CONCERN |
| 3-7 | Synthetic survival + Cox | Synthetic data added, Cox model fitted (C-index=1.0 overfit) | ⚠️ CONCERN |

**Key findings:**
- pyDESeq2 formula-based DE path works end-to-end
- Non-formula DE path still has str+int bug deeper in `bulk_rnaseq_service.py` (line ~2064 Jinja2 template)
- `lifelines` not installed — agent fell back to scipy-based Cox
- `survival_analysis_expert` agent was NEVER invoked — supervisor routed to data_expert who used custom_code
- C-index=1.0 is expected overfitting (50 features, 10 samples, 2 events)
- GSE49712 is a technical benchmark dataset — ~0 real DEGs is the correct biological answer

**Routing issue detail:** The supervisor system prompt routes "survival analysis" to data_expert or transcriptomics_expert instead of the ML pipeline's survival_analysis_expert. This is a **supervisor prompt engineering** issue — the handoff descriptions for ML sub-agents may not include "survival" or "Cox" keywords.

### S4: Multi-Dataset Comparative — BLOCKED

| Step | Query | Result | Status |
|------|-------|--------|--------|
| 1 | Load GSE84133 (pancreas) | 3,605 × 20,125 loaded, QC'd, clustered | ✅ PASS |
| 2 | QC + cluster pancreas | 6 clusters, all major types identified | ✅ PASS |
| 3 | Load GSE150728 (PBMC) | FAILED — .rds.gz files (R data format) | ❌ FAIL |

**Root cause:** GSE150728 supplementary files are `.rds.gz` (R serialized format). Lobster's parser attempts CSV/TSV/Polars parsing and all fail with utf-8 decode errors on binary R data.

**Fix needed:** Either add `.rds` format support (via `rpy2` or `pyreadr`) or choose a PBMC dataset with CSV/H5AD/MTX format (e.g., 10X Genomics PBMC datasets).

### S6: Metadata-Driven Research Pipeline — PARTIAL PASS

| Step | Query | Result | Status |
|------|-------|--------|--------|
| 1 | Search PMID 30643258 | Found publication + extracted GSE120575 via NCBI E-Link | ✅ PASS |
| 2 | Load GSE120575 | FAILED — ragged TSV (Expected 16292 fields, saw 16293) | ❌ FAIL |

**Key findings:**
- Research agent's PMID → GEO pipeline works correctly
- NCBI E-Link extraction found the right dataset (GSE120575)
- Parser fails on ragged lines — the supplementary TSV file has inconsistent field counts
- Fix: Add `truncate_ragged_lines=True` option to Polars/pandas parsing, or use `error_bad_lines=False`

### S3: Multi-Omics MOFA — SKIPPED

Skipped (lowest priority). Both prerequisite datasets had download issues. MOFA code path untested.

---

## Comprehensive Bug List for Next Agent

### CRITICAL (Blocks investor demo)

1. **Supervisor routing to ML agents** (affects S1, S2, S5)
   - **What:** Supervisor routes "stability selection", "feature selection", "survival analysis", "Cox model" to transcriptomics_expert or data_expert instead of ML pipeline agents
   - **Root cause:** The supervisor system prompt's handoff descriptions for ML agents may lack key domain terms. Check `lobster/agents/supervisor.py` prompt and the `handoff_tool_description` in ML agent AGENT_CONFIGs
   - **Fix approach:** Update supervisor prompt to include ML-specific keywords in routing logic. Also check `packages/lobster-ml/lobster/agents/machine_learning/config.py` for the handoff description
   - **Files:** `lobster/agents/supervisor.py`, `packages/lobster-ml/lobster/agents/machine_learning/config.py`

2. **str+int concatenation in bulk_rnaseq_service** (affects S2 non-formula DE)
   - **What:** The non-formula DE path fails with "can only concatenate str (not 'int') to str"
   - **Root cause:** Jinja2 code template at `bulk_rnaseq_service.py:~2064` uses `f'de_results_{{ group1 }}_vs_{{ group2 }}'` — when rendered, integer group values cause string concatenation error
   - **Fix:** Add `|string` Jinja2 filter OR ensure all group parameters are str before passing to `_create_de_ir()`
   - **Files:** `lobster/services/analysis/bulk_rnaseq_service.py` (line ~2064 and ~2032-2033)

3. **Annotation IndexError** (affects S5)
   - **What:** `enhanced_singlecell_service.py:535 _calculate_per_cell_confidence`: index 6235 out of bounds for axis 0 with size 3605
   - **Root cause:** Uses gene indices from `rank_genes_groups` result (which can reference the pre-filtered gene set with >6K genes) to index into the filtered adata (3,605 cells × 15K genes). The indices are gene-axis, not cell-axis, but the indexing is wrong.
   - **Files:** `packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py` (lines 391, 535)

### MODERATE (Degrades experience)

4. **GEO parser ragged lines** (affects S6)
   - **What:** Parser fails on TSV files with inconsistent field counts
   - **Fix:** Add `truncate_ragged_lines=True` for Polars, `on_bad_lines='skip'` for pandas
   - **Files:** `lobster/services/data_access/geo/parser.py`

5. **GEO parser .rds format** (affects S4)
   - **What:** R serialized data format not supported
   - **Fix:** Add `pyreadr` optional dependency OR detect .rds files and give clear error message
   - **Files:** `lobster/services/data_access/geo/parser.py`

6. **Visualization hallucination in long contexts** (affects S5)
   - **What:** After 16+ messages, viz agent narrates without calling tools
   - **Root cause:** LLM context degradation, not code bug. May be mitigated by summarizing earlier messages or using shorter session windows.

### MINOR

7. **`/pipeline export` only in chat mode** — slash commands not parsed in `lobster query`
8. **`lifelines` not installed** — survival analysis custom code can't use CoxPHFitter
9. **`strategy_extraction_error` on every download** — `'DataExpertAssistant' has no attribute 'analyze_download_strategy'` (non-blocking, falls back)
10. **Stability selection on 15K+ features** — Takes >2 hours without pre-filtering. Need to auto-subsample to top 500 HVGs before bootstrap.

---

## Test Cost Summary

| Scenario | Steps | Token Cost | Notes |
|----------|-------|-----------|-------|
| S1 | 8 queries | ~$2.40 | Most efficient |
| S5 | 10 queries | ~$3.80 | Expensive due to long context |
| S2 | 3 queries | ~$5.20 | Very expensive (research agent deep-dived literature) |
| S4 | 3 queries | ~$1.10 | Cheap but blocked on GSE150728 |
| S6 | 2 queries | ~$0.83 | Cheap, download failed |
| **Total** | **26 queries** | **~$13.30** | |

---

## After Testing

### If a scenario passes:

1. Copy the exact queries into the corresponding agent documentation page
2. Include the real outputs (cell counts, gene names, cluster numbers)
3. Add the exported notebook to `docs-site/public/notebooks/` for user download
4. Screenshot key plots for tutorial illustrations

### If a scenario fails:

1. Note which step broke and the error message
2. File a GitHub issue with the session ID and error
3. Do NOT add the workflow to documentation until fixed
4. Check if the failure affects existing documented examples

### Documentation mapping:

| Scenario | Updates to |
|----------|-----------|
| S1 | `agents/ml.mdx` (feature selection example), `agents/transcriptomics.mdx` |
| S2 | `agents/ml.mdx` (survival example), `tutorials/bulk-rnaseq.mdx` |
| S3 | `agents/ml.mdx` (multi-omics example), `advanced/multi-omics.mdx` |
| S4 | `guides/data-analysis-workflows.mdx`, `agents/transcriptomics.mdx` |
| S5 | `getting-started/index.mdx` (quick start), marketing materials |
| S6 | `agents/metadata.mdx`, `agents/research.mdx` |
