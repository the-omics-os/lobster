# AQUADIF Tool Taxonomy Contract

**AQUADIF** is the 10-category taxonomy for Lobster AI tools. Every tool declares what it does (category) and whether it must produce provenance — making the system introspectable, enforceable, and teachable to coding agents.

When designing tools for a new agent, internalize these categories first. Tools should be designed through the AQUADIF lens, not retrofitted with categories afterward.

## Quick Reference

| Category | Definition | Provenance |
|----------|-----------|------------|
| IMPORT | Load external data formats into the workspace | Required |
| QUALITY | Assess data integrity, calculate QC metrics, detect technical artifacts | Required |
| FILTER | Subset data by removing samples, features, or observations | Required |
| PREPROCESS | Transform data representation (normalize, batch correct, scale, impute) | Required |
| ANALYZE | Extract patterns, perform statistical tests, compute embeddings | Required |
| ANNOTATE | Add biological meaning (cell types, gene names, pathway labels) | Required |
| DELEGATE | Hand off work to a specialist child agent | Not required |
| SYNTHESIZE | Combine or interpret results across multiple analyses | Required |
| UTILITY | Workspace management, status checks, listing, exporting | Not required |
| CODE_EXEC | Custom code execution (escape hatch for unsupported operations) | Conditional |

## Category Definitions

### IMPORT

**Definition:** Load external data formats into the workspace.

**Assign this when:** The tool reads files from disk or downloads data from external sources and converts them into the internal AnnData representation. Covers parsers for domain-specific formats.

**Provenance:** Required

**Examples from transcriptomics:**
- `import_bulk_counts` — Reads CSV/TSV count matrices and creates an AnnData object with gene expression data
- `load_10x_data` — Parses 10X Genomics CellRanger output (matrix.mtx, barcodes.tsv, features.tsv) into single-cell AnnData
- `import_h5ad` — Loads saved AnnData objects from disk for session resumption

**Conceptual examples from other domains** (not exhaustive — adapt to your domain):
- Importing genomic interval files (BED, narrowPeak, broadPeak) as region-by-sample matrices
- Importing signal track files (bigWig, bedGraph) as coverage matrices over defined genomic windows
- Importing fragment files from assays that produce per-read coordinate data (e.g., ATAC-seq fragments)

These illustrate that IMPORT is not limited to count matrices — any domain-specific file format that needs parsing into the internal AnnData representation is an IMPORT tool.

**Boundary with `data_expert`:** Lobster's `data_expert` agent (in core) already handles generic file loading via `load_modality(adapter="...")` and database downloads via `execute_download_from_queue`. Your domain IMPORT tools should only handle **formats requiring domain expertise to parse** (vendor-specific outputs, domain-specific file structures with scientific defaults). Do not duplicate generic CSV/H5AD loading — that is `data_expert`'s job. See `creating-agents.md` → "Data Loading Boundary" for the full decision rule.

### QUALITY

**Definition:** Assess data integrity, calculate QC metrics, detect technical artifacts.

**Assign this when:** The tool calculates metrics that indicate data fitness for analysis or identifies technical problems requiring attention. Produces statistics about the data without transforming it.

**Provenance:** Required

**Examples from transcriptomics:**
- `assess_data_quality` — Calculates per-cell QC metrics (n_genes_by_counts, total_counts, pct_counts_mt) to identify low-quality cells
- `detect_batch_effects` — Tests for unwanted technical variation between sequencing batches using kBET or silhouette scores
- `detect_doublets` — Identifies multiplets in single-cell data using Scrublet or DoubletDetection algorithms

### FILTER

**Definition:** Subset data by removing samples, features, or observations.

**Assign this when:** The tool removes rows or columns from the data based on criteria, reducing dataset size while preserving the representation of remaining elements.

**Provenance:** Required

**Examples from transcriptomics:**
- `filter_cells` — Removes low-quality cells based on QC metric thresholds (min_genes, min_counts, max_pct_mt)
- `filter_genes` — Removes features not expressed in sufficient cells or with low variance
- `filter_by_metadata` — Subsets samples matching specific clinical or experimental conditions

### PREPROCESS

**Definition:** Transform data representation (normalize, batch correct, scale, impute).

**Assign this when:** The tool changes the values in the data matrix to improve downstream analysis. Unlike FILTER (which removes), PREPROCESS transforms and retains.

**Provenance:** Required

**Examples from transcriptomics:**
- `normalize_counts` — Applies library size normalization (CPM, TPM, DESeq2 size factors)
- `integrate_batches` — Removes batch effects using Harmony, scVI, or Combat while preserving biological signal
- `scale_data` — Z-scores features for dimensionality reduction input
- `impute_missing` — Fills missing values using k-NN or matrix factorization

### ANALYZE

**Definition:** Extract patterns, perform statistical tests, compute embeddings.

**Assign this when:** The tool discovers structure in the data or quantifies relationships. Produces new derived matrices, embeddings, or statistical test results.

**Provenance:** Required

**Examples from transcriptomics:**
- `run_pca` — Computes principal components to capture variance in high-dimensional expression space
- `cluster_cells` — Groups similar cells using Leiden or Louvain community detection on k-NN graph
- `compute_trajectory` — Infers pseudotime ordering using diffusion pseudotime (DPT) or PAGA
- `run_differential_expression` — Tests for expression differences between cell groups using pyDESeq2 or t-tests
- `compute_gene_set_enrichment` — Evaluates pathway overrepresentation using GSEA or hypergeometric tests

### ANNOTATE

**Definition:** Add biological meaning (cell types, gene names, pathway labels).

**Assign this when:** The tool assigns interpretable labels to data elements using biological knowledge, ontologies, or reference databases. Annotates rather than computes.

**Provenance:** Required

**Examples from transcriptomics:**
- `annotate_cell_types_auto` — Assigns cell type labels using reference-based methods (CellTypist, scANVI marker mapping)
- `annotate_cell_types_manual` — Applies user-provided cell type labels to clusters based on marker gene expression
- `score_gene_signatures` — Calculates per-cell scores for gene sets representing biological processes (cell cycle, stress response)

### DELEGATE

**Definition:** Hand off work to a specialist child agent.

**Assign this when:** The tool creates a delegation handoff to another agent that has deeper expertise in a subdomain. Used by parent agents to manage complexity.

**Provenance:** Not required

**Examples from transcriptomics:**
- `handoff_to_annotation_expert` — Parent transcriptomics agent delegates cell type annotation to the annotation_expert child agent
- `handoff_to_de_analysis_expert` — Parent transcriptomics agent delegates differential expression and pathway analysis to the de_analysis_expert child agent

### SYNTHESIZE

**Definition:** Combine or interpret results across multiple analyses.

**Assign this when:** The tool integrates outputs from different analysis steps or modalities to produce higher-level insights. Goes beyond single-analysis results to cross-cutting interpretation.

**Provenance:** Required

**Note:** Currently unimplemented in Lobster AI — this is an intentional gap representing future work. No examples exist yet. If your domain workflow naturally requires combining results from multiple analyses into a unified interpretation, this is the correct category.

### UTILITY

**Definition:** Workspace management, status checks, listing, exporting.

**Assign this when:** The tool provides operational support for the analysis session but does not directly analyze or transform scientific data. Includes I/O, session state, and metadata queries.

**Provenance:** Not required

**Examples from transcriptomics:**
- `list_modalities` — Shows all loaded datasets in the current workspace with summary statistics
- `get_modality_info` — Returns metadata and shape information for a specific dataset
- `export_results` — Saves analysis results to files (CSV, plots, Jupyter notebook)
- `get_session_status` — Reports current workspace state and available operations

### CODE_EXEC

**Definition:** Custom code execution (escape hatch for unsupported operations).

**Assign this when:** The tool allows users to run arbitrary Python code when no existing tool covers their use case. This is a safety valve for missing functionality, not a primary workflow mechanism.

**Provenance:** Conditional (required if code modifies scientific data; not required if code only inspects or visualizes)

**Examples from transcriptomics:**
- `execute_custom_analysis` — Runs user-provided Python code with access to workspace AnnData objects for one-off custom calculations

## Multi-Category Decision Flowchart

Tools may have 1-3 categories. Follow this process to assign them correctly.

### Step 1: What is the PRIMARY action?

Pick ONE category that best describes the tool's main purpose. This becomes the first element in the `categories` list and determines the provenance requirement.

Ask: "If I had to describe this tool in one word from the AQUADIF vocabulary, what would it be?"

### Step 2: Are there SECONDARY aspects?

Does the tool ALSO do something significant that falls into another category?

Pick 0-2 additional categories. Only add secondary categories if they represent substantial functionality, not minor side effects.

Examples of when to add secondary categories:
- A normalization tool (PREPROCESS primary) that also filters out zero-variance genes (FILTER secondary)
- A quality assessment tool (QUALITY primary) that produces diagnostic plots (UTILITY secondary)

Examples of when NOT to add secondary categories:
- A clustering tool (ANALYZE primary) that logs provenance — logging is required infrastructure, not a UTILITY category
- An import tool (IMPORT primary) that validates file format — validation is part of importing, not separate QUALITY

### Step 3: Validate constraints

- Total categories <= 3
- First category in list = PRIMARY (determines provenance)
- No duplicate categories

### Step 4: Set metadata

```python
tool_function.metadata = {
    "categories": ["PRIMARY", "SECONDARY"],  # Order matters
    "provenance": True  # Match primary category requirement
}
tool_function.tags = ["PRIMARY", "SECONDARY"]  # Same as categories
```

### Boundary Cases

These are the most common ambiguities. When in doubt, use these rules.

**FILTER vs PREPROCESS:**
- FILTER: Removes data elements (rows/columns). Output has fewer elements than input.
- PREPROCESS: Transforms values within elements. Output has same elements as input, but values change.
- Example: Removing low-count cells is FILTER. Normalizing count values is PREPROCESS.

**QUALITY vs ANALYZE:**
- QUALITY: Assesses fitness for purpose. Answers "Is this data good enough to analyze?"
- ANALYZE: Extracts scientific patterns. Answers "What biological structure exists in the data?"
- Example: Calculating per-cell QC metrics is QUALITY. Computing cell-cell distances is ANALYZE.

**ANALYZE vs ANNOTATE:**
- ANALYZE: Computes patterns, clusters, or statistical relationships from the data.
- ANNOTATE: Assigns biological meaning using external knowledge (ontologies, references, markers).
- Example: Clustering cells into groups is ANALYZE. Labeling those clusters as "T cells" is ANNOTATE.

## Metadata Assignment Pattern

After creating a tool with the `@tool` decorator, assign AQUADIF metadata and tags. This pattern works for both shared tools and agent-specific tools.

```python
from lobster.config.aquadif import AquadifCategory

def create_shared_tools(data_manager, quality_service, analysis_service):
    """Create domain tools with AQUADIF metadata."""

    @tool
    def assess_quality(modality_name: str) -> str:
        """Assess data quality for a modality.

        Args:
            modality_name: Name of the dataset to assess

        Returns:
            Summary of QC metrics and data fitness
        """
        adata = data_manager.get_modality(modality_name)
        result, stats, ir = quality_service.assess(adata)
        data_manager.log_tool_usage("assess_quality", {"modality_name": modality_name}, stats, ir=ir)
        return f"QC complete: {stats}"

    # AQUADIF metadata assignment
    # Must happen AFTER @tool decorator (schema extraction complete)
    assess_quality.metadata = {
        "categories": ["QUALITY"],  # 1-3 categories, first = primary
        "provenance": True           # True if primary category requires provenance
    }
    assess_quality.tags = ["QUALITY"]  # Same as categories for callback propagation

    @tool
    def normalize_and_filter(modality_name: str, min_genes: int = 200) -> str:
        """Normalize counts and filter low-quality cells.

        Args:
            modality_name: Name of the dataset to process
            min_genes: Minimum genes per cell (cells below threshold removed)

        Returns:
            Summary of normalization and filtering results
        """
        adata = data_manager.get_modality(modality_name)

        # Normalize (PREPROCESS)
        result, norm_stats, norm_ir = analysis_service.normalize(adata)

        # Filter (FILTER)
        result, filter_stats, filter_ir = quality_service.filter_cells(result, min_genes=min_genes)

        # Combine IRs for multi-step operations
        combined_ir = norm_ir + filter_ir
        data_manager.log_tool_usage("normalize_and_filter",
                                   {"modality_name": modality_name, "min_genes": min_genes},
                                   {**norm_stats, **filter_stats},
                                   ir=combined_ir)
        return f"Normalized and filtered: {norm_stats['cells_before']} → {filter_stats['cells_after']} cells"

    # Multi-category example: primary is PREPROCESS (normalization is the main transformation)
    normalize_and_filter.metadata = {
        "categories": ["PREPROCESS", "FILTER"],  # Primary first
        "provenance": True                        # PREPROCESS requires provenance
    }
    normalize_and_filter.tags = ["PREPROCESS", "FILTER"]

    return [assess_quality, normalize_and_filter]
```

### Key teaching points from this pattern:

1. **Import AquadifCategory** at the top of your agent file for enum access (though string literals work for simple cases)
2. **Metadata assignment happens AFTER @tool** — the decorator must finish first
3. **categories is a list** — order matters, first element determines provenance requirement
4. **provenance is a boolean** — match it to the primary category's requirement:
   - IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE → `True`
   - DELEGATE, UTILITY → `False`
   - CODE_EXEC → `True` if code modifies data, `False` if code only inspects
5. **tags mirrors categories** — `.tags` must be set because LangChain callbacks receive `.tags` but not `.metadata`. Both fields must always contain the same category list
6. **Multi-step tools** — if a tool calls multiple services, combine the `AnalysisStep` objects (use `+` operator)

### For provenance-required tools:

The tool MUST call `log_tool_usage()` with the `ir` parameter (AnalysisStep object from service).

```python
# CORRECT: Provenance logged (provenance: True)
result, stats, ir = service.analyze(adata)
data_manager.log_tool_usage("tool_name", params, stats, ir=ir)

# INCORRECT: Provenance missing (contract tests will fail)
result, stats, ir = service.analyze(adata)
data_manager.log_tool_usage("tool_name", params, stats)  # ir parameter missing!
```

### For non-provenance tools (UTILITY, DELEGATE):

If `metadata["provenance"]` is `False`, do NOT call `log_tool_usage(ir=ir)`. The metadata declaration must match runtime behavior — logging provenance while declaring `provenance: False` is a contract violation. Use a simple return without provenance tracking:

```python
# CORRECT: UTILITY tool — no provenance logging
@tool
def list_modalities() -> str:
    """List available datasets."""
    modalities = data_manager.list_modalities()
    return f"Available: {modalities}"

list_modalities.metadata = {"categories": ["UTILITY"], "provenance": False}
list_modalities.tags = ["UTILITY"]
```

## Contract Tests

Your agent MUST pass these tests. They are implemented in Phase 2 (`lobster/testing/contract_mixins.py`).

**What the tests validate:**

1. **Metadata presence** — Every tool has `.metadata` dict with both `categories` and `provenance` keys
2. **Category validity** — All categories come from the AQUADIF 10-category set (no typos, no custom categories)
3. **Category cap** — Maximum 3 categories per tool
4. **Provenance compliance** — `provenance` boolean matches the primary category's requirement
5. **Provenance call validation** — Tools with `provenance: True` contain a `log_tool_usage()` call with `ir=` parameter (AST-based inspection)
6. **Minimum viable parent** — Parent agents without child agents must have at least one IMPORT tool and one QUALITY tool
7. **No closure scoping drift** — For multi-category tools, ensure metadata is assigned to the correct closure variable
8. **Metadata uniqueness** — Each tool instance has its own metadata dict (not shared references)

**How to run contract tests during development:**

```bash
# For a specific agent package
cd packages/lobster-transcriptomics
pytest -v -k "contract"

# For all agents
pytest -v -k "contract" packages/
```

If a test fails, read the assertion message — it will tell you which tool and which rule failed.

**Copy-paste test template** for your agent's test file:

```python
VALID_CATEGORIES = {
    "IMPORT", "QUALITY", "FILTER", "PREPROCESS", "ANALYZE",
    "ANNOTATE", "DELEGATE", "SYNTHESIZE", "UTILITY", "CODE_EXEC",
}
PROVENANCE_REQUIRED = {
    "IMPORT", "QUALITY", "FILTER", "PREPROCESS",
    "ANALYZE", "ANNOTATE", "SYNTHESIZE",
}

class TestAquadifCompliance:
    """Validate AQUADIF contract compliance for all tools."""

    def test_all_tools_have_metadata(self, tools):
        for t in tools:
            assert hasattr(t, "metadata"), f"{t.name}: missing .metadata"
            assert "categories" in t.metadata, f"{t.name}: missing categories"
            assert "provenance" in t.metadata, f"{t.name}: missing provenance"

    def test_categories_are_valid(self, tools):
        for t in tools:
            for cat in t.metadata["categories"]:
                assert cat in VALID_CATEGORIES, f"{t.name}: invalid category '{cat}'"

    def test_max_three_categories(self, tools):
        for t in tools:
            assert len(t.metadata["categories"]) <= 3, f"{t.name}: >3 categories"

    def test_provenance_matches_primary(self, tools):
        for t in tools:
            primary = t.metadata["categories"][0]
            expected = primary in PROVENANCE_REQUIRED
            assert t.metadata["provenance"] == expected, \
                f"{t.name}: provenance={t.metadata['provenance']} but {primary} requires {expected}"

    def test_tags_match_categories(self, tools):
        for t in tools:
            assert hasattr(t, "tags"), f"{t.name}: missing .tags"
            assert t.tags == t.metadata["categories"], \
                f"{t.name}: .tags {t.tags} != .metadata['categories'] {t.metadata['categories']}"

    def test_tool_count_in_range(self, tools):
        assert 8 <= len(tools) <= 20, f"Tool count {len(tools)} outside 8-20 range"

    def test_minimum_viable_parent(self, tools):
        cats = {cat for t in tools for cat in t.metadata["categories"]}
        assert "IMPORT" in cats, "Missing IMPORT tool"
        assert "QUALITY" in cats, "Missing QUALITY tool"
        assert "ANALYZE" in cats or "DELEGATE" in cats, "Missing ANALYZE or DELEGATE"
```

## Example: Transcriptomics Tool Categorization

This is a conceptual example showing how tools in the transcriptomics domain map to AQUADIF categories. These are NOT complete implementations — they teach the categorization principle.

### IMPORT tools
- `import_bulk_counts` — Load count matrix from CSV/TSV → AnnData
- `load_10x_data` — Load 10X Genomics CellRanger output → AnnData
- `import_h5ad` — Load saved AnnData from disk

### QUALITY tools
- `assess_data_quality` — Calculate per-cell QC metrics (n_genes, total_counts, pct_mt)
- `detect_batch_effects` — Test for unwanted technical variation between batches
- `detect_doublets` — Identify multiplets in single-cell data

### FILTER tools
- `filter_cells` — Remove low-quality cells based on QC thresholds
- `filter_genes` — Remove lowly expressed or low-variance genes
- `filter_by_metadata` — Subset samples by clinical or experimental criteria

### PREPROCESS tools
- `normalize_counts` — Apply library size normalization (CPM, DESeq2, scran)
- `integrate_batches` — Remove batch effects using Harmony or scVI
- `scale_data` — Z-score features for dimensionality reduction
- `impute_missing` — Fill missing values using k-NN imputation

### ANALYZE tools
- `run_pca` — Compute principal components
- `compute_neighbors` — Build k-nearest neighbor graph
- `cluster_cells` — Group cells using Leiden or Louvain
- `compute_trajectory` — Infer pseudotime using diffusion pseudotime (DPT)
- `run_differential_expression` — Test for expression differences between groups
- `compute_gene_set_enrichment` — Evaluate pathway overrepresentation

### ANNOTATE tools (via child agent)
Transcriptomics expert delegates to `annotation_expert` child agent:
- `annotate_cell_types_auto` — Reference-based cell type assignment
- `annotate_cell_types_manual` — User-provided cell type labels
- `score_gene_signatures` — Calculate gene set scores per cell

### DELEGATE tools
- `handoff_to_annotation_expert` — Delegate cell type annotation to specialist
- `handoff_to_de_analysis_expert` — Delegate differential expression to specialist

### UTILITY tools
- `list_modalities` — Show loaded datasets in workspace
- `get_modality_info` — Get metadata for a specific dataset
- `export_results` — Save analysis outputs to files
- `get_session_status` — Report current workspace state

### SYNTHESIZE tools
No implementations yet in transcriptomics domain — future work.

### CODE_EXEC tools
- `execute_custom_analysis` — Run arbitrary Python code with workspace access (escape hatch)

## Design Principles

When building a new agent following this contract:

1. **Internalize AQUADIF first** — Read this document before writing tools, design tools through the AQUADIF lens
2. **Category minimalism** — Prefer single category when possible; only add secondary categories for substantial additional functionality
3. **Provenance discipline** — Always pass `ir` parameter to `log_tool_usage()` for provenance-required tools
4. **Conceptual not prescriptive** — Use the transcriptomics examples to understand categorization principles, not as templates to copy
5. **Honest gaps** — If your domain doesn't need SYNTHESIZE tools, that's fine; don't force implementations
6. **Boundary cases** — When ambiguous, consult the decision flowchart boundary cases section

---

**Version:** Phase 1 (2026-02-27)
**Next:** Phase 2 will implement contract tests that validate compliance with this specification.
