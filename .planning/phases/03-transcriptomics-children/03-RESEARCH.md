# Phase 3: Transcriptomics Children - Research

**Researched:** 2026-02-22
**Domain:** Annotation expert enhancement (gene set scoring, renames, bug fixes) + DE analysis expert overhaul (tool merges, deprecations, bulk DE pipeline, GSEA, publication export)
**Confidence:** HIGH

## Summary

Phase 3 enhances the two child agents of the transcriptomics family: `annotation_expert` (4 requirements: ANN-01 through ANN-04) and `de_analysis_expert` (11 requirements: DEA-01 through DEA-11), plus one prompt update (DOC-04). The annotation expert changes are surgical — one new tool, one rename, two bug fixes. The DE analysis expert changes are substantial — merging 3 DE tools into 2, merging 2 formula tools into 1, deprecating 2 interactive terminal tools, adding 5 new tools for bulk DE pipeline completion (filter results, export, bulk direct DE, GSEA, publication tables), renaming 2 tools, and updating the prompt.

The codebase is well-positioned for this work. Both child agents already exist in `packages/lobster-transcriptomics/lobster/agents/transcriptomics/` with full modular structure (config.py, prompts.py, state.py, separate agent files). The `PathwayEnrichmentService` in core already has a complete `gene_set_enrichment_analysis()` method using `gseapy.prerank`. The `BulkRNASeqService` already has `run_differential_expression_analysis()` (simple 2-group) and `run_pydeseq2_analysis()` (formula-based). pyDESeq2 is already a dependency of `lobster-transcriptomics` (>=0.5.2). gseapy is already a dependency of the core package (>=1.1.0). `scanpy.tl.score_genes` is available through the existing scanpy dependency. The `component_registry` pattern for VectorSearchService is well-documented. The `store_modality()` API on DataManagerV2 is the correct pattern (replacing direct dict assignment).

**Primary recommendation:** Implement in 3 plans: Plan 01 for annotation expert (ANN-01 through ANN-04 — new score_gene_set tool, rename, 2 bug fixes), Plan 02 for DE analysis expert tool refactoring (DEA-01 through DEA-08 — tool merges, deprecations, renames, 2 new tools), Plan 03 for DE bulk additions + prompt (DEA-09 through DEA-11, DOC-04 — 3 new bulk DE tools + prompt update).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ANN-01 | Add `score_gene_set` tool — gene set scoring via sc.tl.score_genes | `scanpy.tl.score_genes` is available (scanpy already a dep). No existing wrapper in codebase. Need new tool in `annotation_expert.py` that calls sc.tl.score_genes on a modality with a user-provided gene list. Returns per-cell scores stored in adata.obs. Should create AnalysisStep IR for provenance. |
| ANN-02 | Rename `annotate_cell_types` to `annotate_cell_types_auto` | Tool defined at line 142 of `annotation_expert.py`. Rename function name, update `tool_name=` in log_tool_usage, update the base_tools list, update annotation_expert prompt in prompts.py. Straightforward rename. |
| ANN-03 | Fix BUG-04: Replace try/except ImportError with component_registry for VectorSearchService | Lines 61-66 of `annotation_expert.py` use `try: from lobster.services.vector.service import VectorSearchService; HAS_VECTOR_SEARCH = True; except ImportError: HAS_VECTOR_SEARCH = False`. Per Hard Rule #7, must replace with `component_registry.get_service('vector_search')` pattern. Note: VectorSearchService lives at `lobster/core/vector/service.py` (shim) pointing to `lobster/services/vector/service.py`. The `_get_vector_service()` lazy function at line 1275-1281 also imports directly. Both must be changed. |
| ANN-04 | Fix BUG-05: Use data_manager.store_modality() instead of direct dict assignment | Line 1442 of `annotation_expert.py`: `data_manager.modalities[annotated_modality_name] = adata_annotated` — this is the ONLY instance of direct dict assignment in the annotation expert. Must replace with `data_manager.store_modality(name=..., adata=..., parent_name=..., step_summary=...)`. |
| DEA-01 | Merge 3 DE tools into 2: `run_differential_expression` (simple) + `run_de_with_formula` (advanced) | Currently has `run_pseudobulk_differential_expression` (line 489), `run_differential_expression_with_formula` (line 1125), and `run_differential_expression_analysis` (line 1327). Merge `run_pseudobulk_differential_expression` + `run_differential_expression_analysis` into a single `run_differential_expression` that handles both simple 2-group and pseudobulk with auto-detection. Keep `run_differential_expression_with_formula` as `run_de_with_formula` (renamed). D5 decision: reduces LLM confusion from 3 similar DE tools to 2 with clear roles. |
| DEA-02 | Merge `construct_de_formula_interactive` + `suggest_formula_for_design` into `suggest_de_formula` | `suggest_formula_for_design` (line 737) analyzes metadata and suggests formulas. `construct_de_formula_interactive` (line 950) builds formula from components with validation. Merge into single `suggest_de_formula` that does both: analyzes metadata, suggests formulas, AND constructs/validates the chosen formula. The "interactive" was never truly interactive (it builds formula from LLM-provided params). |
| DEA-03 | Deprecate `manually_annotate_clusters_interactive` (BUG-11) | Tool at line 273 of `annotation_expert.py`. D6: Interactive terminal tools are cloud-incompatible. Remove from base_tools list, add deprecation warning if called. `manually_annotate_clusters` (non-interactive, line 388) stays as the recommended replacement. |
| DEA-04 | Deprecate `construct_de_formula_interactive` (BUG-12) | Tool at line 950 of `de_analysis_expert.py`. D6: Interactive terminal tools are cloud-incompatible. Remove from base_tools list. Replaced by new merged `suggest_de_formula` (DEA-02). |
| DEA-05 | Add `filter_de_results` tool — standalone result filtering | NEW tool. Currently DE results include all genes. Need tool that takes a DE modality + thresholds (padj, lfc, baseMean) and returns filtered results as a new modality subset. Extract from DE results stored in adata.uns, apply thresholds, store filtered subset. |
| DEA-06 | Add `export_de_results` tool — publication-ready CSV/Excel export | NEW tool. Takes a DE modality, extracts results from adata.uns, and exports as formatted CSV or Excel with sorted gene lists, formatted p-values, and standard DE column naming (gene, log2FoldChange, padj, baseMean). Save to workspace. |
| DEA-07 | Rename `prepare_differential_expression_design` to `prepare_de_design` | Tool defined at line 354 of `de_analysis_expert.py`. Simple rename: function name, tool_name in log_tool_usage, base_tools list, prompt references. |
| DEA-08 | Rename `run_pathway_enrichment_analysis` to `run_pathway_enrichment` | Tool defined at line 1919 of `de_analysis_expert.py`. Simple rename: function name, tool_name in log_tool_usage, base_tools list, prompt references. |
| DEA-09 | Add `run_bulk_de_direct` tool — one-shot DE for simple bulk comparisons | NEW tool. Wraps `BulkRNASeqService.run_differential_expression_analysis()` for simple 2-group bulk DE without requiring pseudobulk aggregation. Takes modality_name, group_key, group1, group2. For when bulk data is already imported and user wants quick DE without formula complexity. |
| DEA-10 | Add `run_gsea_analysis` tool — ranked gene set enrichment | NEW tool wrapping `PathwayEnrichmentService.gene_set_enrichment_analysis()`. Extracts ranked gene list from DE results (sorted by log2FC * -log10(pvalue)), runs GSEA via gseapy.prerank. Service method already exists in core `lobster/services/analysis/pathway_enrichment_service.py` (line 237). gseapy already a dependency (>=1.1.0). |
| DEA-11 | Add `extract_and_export_de_results` tool — publication-ready tables with LFC shrinkage | NEW tool. Extracts DE results from a modality, applies optional LFC shrinkage (pyDESeq2's `ds.lfc_shrink()`), formats as publication-ready table (gene, baseMean, log2FC, lfcSE, stat, pvalue, padj), and exports to CSV/Excel. Combines extraction + shrinkage + export in one operation for complete bulk DE pipeline. |
| DOC-04 | Update de_analysis_expert prompt for merged DE tools + bulk additions | Update `create_de_analysis_expert_prompt()` in prompts.py to reflect: merged tool names (run_differential_expression, run_de_with_formula, suggest_de_formula), new tools (filter_de_results, export_de_results, run_bulk_de_direct, run_gsea_analysis, extract_and_export_de_results), removed tools (construct_de_formula_interactive, old 3-way DE split), bulk DE workflow guidance. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scanpy | >=1.11.4 | sc.tl.score_genes for gene set scoring (ANN-01) | Already in pyproject.toml; standard for scRNA-seq |
| pydeseq2 | >=0.5.2 | LFC shrinkage via ds.lfc_shrink() (DEA-11) | Already in lobster-transcriptomics pyproject.toml |
| gseapy | >=1.1.0 | GSEA via gp.prerank for ranked enrichment (DEA-10) | Already in core pyproject.toml |
| pandas | >=1.5.0 | DataFrame manipulation for DE export/filtering | Already in pyproject.toml |
| anndata | >=0.9.0 | Data storage format | Already in pyproject.toml |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| openpyxl | >=3.1.0 | Excel export for publication tables (DEA-06, DEA-11) | Already available via pandas; used for .xlsx export |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gseapy for GSEA | Custom GSEA from scratch | gseapy already installed, well-tested, uses Enrichr API. No reason to hand-roll. |
| sc.tl.score_genes for gene set scoring | Custom correlation-based scoring | sc.tl.score_genes is scanpy's built-in method, uses mean expression minus reference gene set. Standard, well-tested. |
| pd.to_excel for publication tables | csv-only export | Excel is standard for publication supplementary tables. openpyxl comes free with pandas install. |

**Installation changes:** None. All required libraries are already dependencies.

## Architecture Patterns

### Recommended Project Structure

All changes are within the existing `packages/lobster-transcriptomics/` package:

```
packages/lobster-transcriptomics/
├── lobster/
│   └── agents/
│       └── transcriptomics/
│           ├── annotation_expert.py     # UPDATE: +1 new tool (score_gene_set), rename 1 tool, 2 bug fixes, deprecate 1 tool
│           ├── de_analysis_expert.py    # UPDATE: merge tools, deprecate 1, +5 new tools, rename 2
│           ├── prompts.py              # UPDATE: annotation_expert prompt (minor), de_analysis_expert prompt (major rewrite)
│           ├── config.py               # No changes
│           ├── state.py                # No changes
│           ├── shared_tools.py         # No changes
│           └── transcriptomics_expert.py # No changes
└── tests/
    └── (no new test files needed — update existing test_annotation_expert.py and test_de_analysis_expert.py)
```

### Pattern 1: New Tool Wrapping Existing Scanpy Function (ANN-01: score_gene_set)

**What:** Wrap `sc.tl.score_genes` in a tool closure following the standard tool pattern.
**When to use:** When scanpy/numpy/pandas already has the functionality and we just need to expose it as an agent tool with proper modality management and IR.
**Example:**

```python
@tool
def score_gene_set(
    modality_name: str,
    gene_list: list,
    score_name: str = "gene_set_score",
    ctrl_size: int = 50,
    use_raw: bool = False,
) -> str:
    """Score cells for expression of a gene set using scanpy.tl.score_genes."""
    import scanpy as sc

    if modality_name not in data_manager.list_modalities():
        return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

    adata = data_manager.get_modality(modality_name)

    # Filter gene_list to genes present in adata
    valid_genes = [g for g in gene_list if g in adata.var_names]
    if not valid_genes:
        return f"None of the provided genes found in modality. Available gene sample: {list(adata.var_names[:5])}"

    sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name, ctrl_size=ctrl_size, use_raw=use_raw)

    # Store result
    scored_name = f"{modality_name}_scored"
    data_manager.store_modality(
        name=scored_name, adata=adata, parent_name=modality_name,
        step_summary=f"Scored {len(valid_genes)} genes as '{score_name}'"
    )

    # Build IR
    ir = AnalysisStep(
        operation="scanpy.tl.score_genes",
        tool_name="score_gene_set",
        description=f"Gene set scoring: {len(valid_genes)} genes, stored as '{score_name}'",
        library="scanpy",
        code_template="sc.tl.score_genes(adata, gene_list={{ gene_list }}, score_name='{{ score_name }}')",
        imports=["import scanpy as sc"],
        parameters={...},
    )
    data_manager.log_tool_usage("score_gene_set", {...}, stats, ir=ir)
    return f"Scored {len(valid_genes)} genes..."
```

### Pattern 2: Tool Merging (DEA-01: 3 DE tools -> 2)

**What:** Merge overlapping tools by unifying their parameters and adding auto-detection logic.
**When to use:** When multiple tools do the same operation with slightly different entry conditions.
**Strategy:**

1. Create the new unified tool (`run_differential_expression`) that accepts a superset of parameters
2. Auto-detect whether input is pseudobulk or simple bulk based on adata.uns keys
3. Route internally to the appropriate service method
4. Remove old tools from the base_tools list
5. Keep old function code temporarily (commented or deleted) to prevent regressions

```python
@tool
def run_differential_expression(
    modality_name: str,
    groupby: str,
    group1: str,
    group2: str,
    method: str = "deseq2",  # Unified default
    alpha: float = 0.05,
    shrink_lfc: bool = True,
    save_result: bool = True,
) -> str:
    """Run differential expression analysis (simple 2-group comparison).

    Works for both pseudobulk (from single-cell) and bulk RNA-seq data.
    For complex multi-factor designs, use run_de_with_formula instead.
    """
    # ... validates, calls bulk_rnaseq_service, returns formatted result
```

### Pattern 3: Tool Deprecation (DEA-03/DEA-04)

**What:** Remove deprecated interactive tools from the tool list, keep function definition with deprecation warning.
**When to use:** When a tool is cloud-incompatible (D6) and has a non-interactive replacement.
**Strategy:**

1. Remove from `base_tools` list (agent no longer sees it)
2. Keep function body with a deprecation warning log at top
3. Function still callable if someone constructs it manually, but agent won't offer it

```python
# In base_tools list:
base_tools = [
    # ... remove manually_annotate_clusters_interactive
    # ... remove construct_de_formula_interactive
]

# Function stays defined but with deprecation:
@tool
def manually_annotate_clusters_interactive(...) -> str:
    """DEPRECATED: Use manually_annotate_clusters instead. This tool requires terminal access and is incompatible with cloud deployment."""
    logger.warning("manually_annotate_clusters_interactive is deprecated. Use manually_annotate_clusters instead.")
    return "This tool is deprecated. Use manually_annotate_clusters for direct cluster annotation."
```

### Pattern 4: BUG-04 Fix — component_registry for Optional Services (ANN-03)

**What:** Replace `try/except ImportError` with `component_registry.get_service()`.
**When to use:** Any optional service that may or may not be installed (Hard Rule #7).
**Example:**

```python
# BEFORE (BUG-04):
try:
    from lobster.services.vector.service import VectorSearchService
    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False

# AFTER:
from lobster.core.component_registry import component_registry

# Lazy check inside factory (NOT at module level per Hard Rule #10)
def annotation_expert(...):
    has_vector_search = component_registry.get_service("vector_search") is not None

    # ... later, when building tools:
    if has_vector_search:
        base_tools.append(annotate_cell_types_semantic)

    # Inside the semantic tool:
    def _get_vector_service():
        nonlocal _vector_service
        if _vector_service is None:
            svc_cls = component_registry.get_service("vector_search", required=True)
            _vector_service = svc_cls()
        return _vector_service
```

**CRITICAL:** The `component_registry` call must be inside the factory function, NOT at module level (Hard Rule #10). The HAS_VECTOR_SEARCH module-level check must be moved inside the factory.

### Pattern 5: New Service-Wrapping Tool for GSEA (DEA-10)

**What:** Wrap the existing `PathwayEnrichmentService.gene_set_enrichment_analysis()` method.
**When to use:** When the service already exists and just needs a tool wrapper.
**Key detail:** The GSEA service expects a `pd.DataFrame` with columns `['gene', 'score']`. The tool must extract DE results from adata.uns, create the ranked gene list (gene name + log2FC * -log10(pvalue) or just log2FC), and pass to the service.

```python
@tool
def run_gsea_analysis(
    modality_name: str,
    de_results_key: str = None,
    ranking_metric: str = "log2fc",  # or "signed_pvalue"
    databases: list = None,
    min_size: int = 15,
    max_size: int = 500,
) -> str:
    """Run Gene Set Enrichment Analysis on DE results."""
    # Extract DE results from adata.uns
    # Build ranked gene DataFrame
    # Call PathwayEnrichmentService.gene_set_enrichment_analysis()
    # Store results, return formatted response
```

### Anti-Patterns to Avoid

- **Anti-pattern: Module-level component_registry calls.** ANN-03 fix MUST NOT call `component_registry.get_service()` at module level. This triggers loading ALL agents at import time (Hard Rule #10). Move inside factory function.
- **Anti-pattern: Direct dict assignment for modalities.** ANN-04 fix replaces `data_manager.modalities[name] = adata` with `data_manager.store_modality(...)`. Never use direct dict assignment.
- **Anti-pattern: Keeping deprecated tools in base_tools.** DEA-03/DEA-04 must REMOVE from base_tools, not just add warnings.
- **Anti-pattern: Creating new service files.** All required service methods already exist. Do NOT create new service files — only add tool wrappers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gene set scoring | Custom mean-expression scorer | `scanpy.tl.score_genes` | Handles control gene set normalization, well-tested, standard |
| GSEA analysis | Custom prerank implementation | `gseapy.prerank` via `PathwayEnrichmentService.gene_set_enrichment_analysis()` | Already implemented and tested in core. gseapy handles permutations, FDR correction, NES computation |
| LFC shrinkage | Custom shrinkage estimator | `pydeseq2.DeseqDataSet.lfc_shrink()` | Already available. Implements apeGLM-like shrinkage. pyDESeq2 already a dependency. |
| Publication table formatting | Custom Excel generator | `pandas.DataFrame.to_excel()` with openpyxl | Standard pandas feature, handles formatting, multiple sheets |
| Component registry for optional services | `try/except ImportError` | `component_registry.get_service()` | Project convention (Hard Rule #7), consistent across codebase |

**Key insight:** This phase is primarily about tool-level refactoring (merges, renames, wrappers), NOT about creating new service logic. All scientific computation already exists in the service layer.

## Common Pitfalls

### Pitfall 1: DE Tool Merge Breaking Existing Workflows

**What goes wrong:** Merging 3 DE tools into 2 could break the prompt's tool references, test expectations, or cross-tool references (e.g., `iterate_de_analysis` calls `run_differential_expression_with_formula` directly at line 1575).
**Why it happens:** Internal tool-to-tool calls use function references, not string names. If you rename `run_differential_expression_with_formula` to `run_de_with_formula`, the `iterate_de_analysis` function's direct Python call still works (it's a closure), but the logged tool_name and prompt references must also change.
**How to avoid:** Search for ALL references to old tool names: in prompts.py, in the tool registry list, in log_tool_usage calls, in "Next step" response strings, and in internal function calls from other tools (iterate_de_analysis, compare_de_iterations).
**Warning signs:** Tests pass but prompts reference non-existent tools; LLM tries to call old tool names.

### Pitfall 2: ANN-03 Module-Level Component Registry Call

**What goes wrong:** Moving `try/except ImportError` to `component_registry.get_service()` at module level triggers loading ALL agents at import time, causing slow startup.
**Why it happens:** Hard Rule #10 exists specifically because component_registry calls at module level are a known performance trap.
**How to avoid:** Put the component_registry check INSIDE the `annotation_expert()` factory function. The `HAS_VECTOR_SEARCH` module-level variable must go away entirely. Use a local variable inside the factory.
**Warning signs:** Agent startup becomes noticeably slower; import cycles appear.

### Pitfall 3: GSEA Ranked Gene List Format

**What goes wrong:** PathwayEnrichmentService.gene_set_enrichment_analysis() expects a DataFrame with exactly columns ['gene', 'score']. If the DE results use different column names (e.g., 'log2FoldChange', 'padj'), the GSEA tool must transform them.
**Why it happens:** DE results in adata.uns have varying column names depending on which DE method was used (BulkRNASeqService uses different result keys for different methods).
**How to avoid:** In the `run_gsea_analysis` tool, handle multiple DE result formats: check for 'log2FoldChange' (pyDESeq2), 'logFC' (generic), 'mean_log2FC' (simple DE). Build a standardized ['gene', 'score'] DataFrame regardless of input format.
**Warning signs:** GSEA tool fails with "ranked_genes must have 'gene' and 'score' columns".

### Pitfall 4: Deprecation vs Deletion of Interactive Tools

**What goes wrong:** Completely deleting `manually_annotate_clusters_interactive` and `construct_de_formula_interactive` breaks backward compatibility if any saved sessions or notebooks reference them.
**Why it happens:** Tools are identified by name in provenance logs and exported notebooks.
**How to avoid:** Keep the function defined (with deprecation warning), just remove from `base_tools` list so the LLM never sees it. If called directly (e.g., from a script), it returns a helpful deprecation message pointing to the replacement.
**Warning signs:** Old notebooks or scripts fail with "tool not found" errors.

### Pitfall 5: DEA-11 (extract_and_export) vs DEA-06 (export_de_results) Overlap

**What goes wrong:** DEA-06 and DEA-11 could become confusing duplicates if not clearly differentiated.
**Why it happens:** Both export DE results. The distinction is: DEA-06 is a simple export of existing results; DEA-11 additionally applies LFC shrinkage before export.
**How to avoid:** Clear tool descriptions. `export_de_results` = "Export existing DE results as CSV/Excel". `extract_and_export_de_results` = "Extract results with LFC shrinkage and export publication-ready tables". Consider whether DEA-06 and DEA-11 should be a single tool with an optional `shrink_lfc` parameter.
**Warning signs:** LLM picks wrong export tool; users get unshrunk results when they wanted shrunk.

## Code Examples

### ANN-01: score_gene_set Tool (Verified from scanpy docs)

```python
import scanpy as sc

# scanpy.tl.score_genes signature:
sc.tl.score_genes(
    adata,
    gene_list,          # List of gene names
    ctrl_size=50,       # Number of control genes per scored gene
    gene_pool=None,     # All genes if None
    n_bins=25,          # Number of expression bins
    score_name='score', # Column name in adata.obs
    random_state=0,
    copy=False,
    use_raw=None,       # Use adata.raw if available
)
# Result: adata.obs[score_name] contains per-cell scores
```

### ANN-03: component_registry Pattern (from codebase)

```python
# In factory function (NOT at module level):
from lobster.core.component_registry import component_registry

def annotation_expert(data_manager, ...):
    # Check availability inside factory
    component_registry.load_components()  # Idempotent
    vector_search_cls = component_registry.get_service("vector_search")
    has_vector_search = vector_search_cls is not None

    _vector_service = None
    def _get_vector_service():
        nonlocal _vector_service
        if _vector_service is None:
            cls = component_registry.get_service("vector_search", required=True)
            _vector_service = cls()
        return _vector_service

    # ... build tools ...
    if has_vector_search:
        base_tools.append(annotate_cell_types_semantic)
```

### DEA-10: GSEA Using Existing Service

```python
from lobster.services.analysis.pathway_enrichment_service import PathwayEnrichmentService

pathway_service = PathwayEnrichmentService()

# Build ranked gene list from DE results
de_results = adata.uns["de_results_condition_treatment_vs_control"]["results_df"]
ranked_genes = pd.DataFrame({
    "gene": de_results.index,
    "score": de_results["log2FoldChange"],  # or signed p-value
})
ranked_genes = ranked_genes.dropna().sort_values("score", ascending=False)

# Call existing service method
adata_out, stats, ir = pathway_service.gene_set_enrichment_analysis(
    adata=adata,
    ranked_genes=ranked_genes,
    databases=["GO_Biological_Process_2023", "KEGG_2021_Human"],
    organism="human",
)
```

### DEA-05: DE Result Filtering Pattern

```python
# DE results are stored in adata.uns with keys like:
# "de_results_condition_treatment_vs_control" -> {"results_df": DataFrame, "analysis_stats": dict}

# Standard filtering:
results_df = de_results["results_df"]
filtered = results_df[
    (results_df["padj"] < padj_threshold) &
    (results_df["log2FoldChange"].abs() > lfc_threshold) &
    (results_df["baseMean"] > min_base_mean)
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3 separate DE tools | 2 merged DE tools (simple + formula) | Phase 3 (D5) | Reduces LLM confusion; clearer tool selection |
| Interactive terminal tools | Non-interactive API tools | Phase 3 (D6) | Cloud compatibility; no terminal dependency |
| try/except ImportError | component_registry.get_service() | Ongoing (Hard Rule #7) | Consistent plugin discovery pattern |
| Direct modality dict assignment | data_manager.store_modality() | Ongoing best practice | Lineage tracking, dirty marking, auto-save |

**Deprecated/outdated:**
- `manually_annotate_clusters_interactive`: Requires Rich terminal; cloud-incompatible (D6). Use `manually_annotate_clusters` instead.
- `construct_de_formula_interactive`: Misleading name (never truly interactive). Replaced by `suggest_de_formula`.
- `run_pseudobulk_differential_expression`: Merged into `run_differential_expression`.
- `run_differential_expression_analysis`: Merged into `run_differential_expression`.

## Open Questions

1. **DEA-06 vs DEA-11 overlap**
   - What we know: Both export DE results. DEA-11 adds LFC shrinkage.
   - What's unclear: Whether they should be two separate tools or one tool with an optional `shrink_lfc` parameter.
   - Recommendation: Implement as two tools per requirements spec, but note the overlap in the prompt so the LLM picks correctly. If shrinkage is cheap, DEA-11 could become the default export tool and DEA-06 becomes redundant.

2. **component_registry service name for VectorSearchService**
   - What we know: VectorSearchService is at `lobster/core/vector/service.py` (shim to `lobster/services/vector/service.py`).
   - What's unclear: Whether it's registered as `"vector_search"` in the entry points or under a different name.
   - Recommendation: Check entry points in pyproject.toml. If not registered, the import shim approach may need to stay but wrapped in a lazy pattern rather than module-level try/except. Alternatively, register it as an entry point.

3. **DEA-09 (run_bulk_de_direct) vs merged DEA-01 (run_differential_expression)**
   - What we know: DEA-01 merges 3 DE tools into `run_differential_expression` (simple). DEA-09 adds `run_bulk_de_direct` for one-shot bulk DE.
   - What's unclear: These seem to overlap — `run_differential_expression` (simple 2-group) IS essentially `run_bulk_de_direct`.
   - Recommendation: Make `run_differential_expression` the single tool for simple DE (both pseudobulk and bulk). `run_bulk_de_direct` may be unnecessary if the merged tool handles both. Implement DEA-09 as an alias or additional entry point only if the merged tool's interface is confusing for bulk-only users.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `annotation_expert.py` (1606 lines), `de_analysis_expert.py` (2046 lines), `prompts.py` (457 lines)
- Codebase inspection: `pathway_enrichment_service.py` (GSEA method at line 237, gp.prerank usage)
- Codebase inspection: `bulk_rnaseq_service.py` (DE methods, lfc_shrink at line 1551)
- Codebase inspection: `component_registry.py` (get_service pattern at line 229)
- Codebase inspection: `data_manager_v2.py` (store_modality at line 889)
- scanpy documentation: sc.tl.score_genes API (verified via training data, standard scanpy method)

### Secondary (MEDIUM confidence)
- pyDESeq2 lfc_shrink: Based on codebase usage pattern (line 1551 of bulk_rnaseq_service.py). API verified from existing working code.
- gseapy.prerank: Based on existing usage in pathway_enrichment_service.py (line 304). Already tested and working.

### Tertiary (LOW confidence)
- None. All findings are from direct codebase inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and used in codebase
- Architecture: HIGH - All patterns follow existing codebase conventions with direct code references
- Pitfalls: HIGH - Identified from actual codebase issues (direct dict assignment, module-level imports) and tool interaction patterns

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable domain, internal codebase changes only)
