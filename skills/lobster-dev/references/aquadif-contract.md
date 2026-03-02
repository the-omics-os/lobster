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

### Category Decision Quick Reference

When categorizing a tool, find what it does in this table:

| If your tool... | Category |
|---|---|
| Reads files from disk or downloads data into AnnData | IMPORT |
| Calculates QC metrics, checks data fitness, detects artifacts | QUALITY |
| Removes rows/columns, subsets observations or features | FILTER |
| Normalizes, batch corrects, scales, imputes, reshapes values | PREPROCESS |
| Clusters, embeds, runs statistics, computes trajectories | ANALYZE |
| Assigns labels using ontologies, references, or ID mappings | ANNOTATE |
| Hands off to a child agent | DELEGATE |
| Combines results from multiple analyses into interpretation | SYNTHESIZE |
| Lists datasets, shows status, exports files (read-only ops) | UTILITY |
| Executes arbitrary user code | CODE_EXEC |

## Metadata Assignment Pattern

After creating a tool with the `@tool` decorator, assign AQUADIF metadata and tags. This pattern works for both shared tools and agent-specific tools.

```python
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

1. **Use string literals** for category names (`"ANALYZE"`, `"IMPORT"`, etc.) — do NOT import `AquadifCategory` enum in tool files
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

### Anti-Patterns (what NOT to do)

```python
# WRONG: Metadata before @tool (decorator overwrites it)
my_tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
@tool
def my_tool(...): ...

# WRONG: Only setting .metadata (callbacks won't see categories)
my_tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
# Missing: my_tool.tags = ["ANALYZE"]

# WRONG: .tags differs from .metadata["categories"]
my_tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
my_tool.tags = ["ANALYZE", "EXTRA"]  # Mismatch!

# WRONG: Provenance flag contradicts primary category
my_tool.metadata = {"categories": ["IMPORT"], "provenance": False}  # IMPORT requires True

# WRONG: Importing enum in tool file (unnecessary coupling)
from lobster.config.aquadif import AquadifCategory
my_tool.metadata = {"categories": [AquadifCategory.ANALYZE], "provenance": True}
# Use string literals instead: "ANALYZE"
```

## Contract Tests

Your agent MUST pass these tests. They are implemented in Phase 2 (`lobster/testing/contract_mixins.py`).

**What the tests validate:**

1. **Metadata presence** — Every tool has `.metadata` dict with both `categories` and `provenance` keys
2. **Category validity** — All categories come from the AQUADIF 10-category set (no typos, no custom categories)
3. **Category cap** — Maximum 3 categories per tool
4. **Provenance compliance** — `provenance` boolean matches the primary category's requirement
5. **Provenance call validation** — Tools with `provenance: True` contain a `log_tool_usage()` call with `ir=` parameter (AST-based inspection)
6. **Minimum viable parent** — Parent agents must have at least one IMPORT + one QUALITY + one ANALYZE or DELEGATE tool. If your agent's DELEGATE tools are injected at runtime (via `delegation_tools` parameter from `graph.py`), set `is_parent_agent = False` in the contract test to skip this check — the test cannot see runtime-injected tools.
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

## Patterns from Research Package (Reference Implementation)

This section shows real categorization decisions from `lobster-research` (`research_agent` + `data_expert`). Use as reference when categorizing tools in new domain packages.

### research_agent tools (11 tools)

| Tool | Category | Provenance | Rationale |
|------|----------|------------|-----------|
| `search_literature` | UTILITY | False | Read-only database query; no data transformation |
| `find_related_entries` | UTILITY | False | Cross-reference lookup; returns links, no transformation |
| `fast_dataset_search` | UTILITY | False | Lightweight search wrapper; read-only |
| `get_dataset_metadata` | QUALITY | True | Validates dataset fitness; produces quality assessment |
| `validate_dataset_metadata` | QUALITY | True | Schema validation + download readiness check |
| `prepare_dataset_download` | UTILITY | False | Queue management (administrative); does not load data |
| `extract_methods` | UTILITY | False | Text extraction from publication; no data transformation |
| `fast_abstract_search` | UTILITY | False | Abstract retrieval; read-only literature access |
| `read_full_publication` | UTILITY | False | Full-text extraction; read-only content access |
| `process_publication_entry` | PREPROCESS | True | Transforms raw publication record → structured metadata |
| `process_publication_queue` | PREPROCESS | True | Batch transformation of publication records |

**Note on factory-created tools:** `write_to_workspace` and `get_content_from_workspace` are created via shared factories. Tag them at the injection site in the factory function:
```python
write_to_workspace = create_write_to_workspace_tool(data_manager)
write_to_workspace.metadata = {"categories": ["UTILITY"], "provenance": False}
write_to_workspace.tags = ["UTILITY"]
```

### data_expert tools (10 tools in base_tools)

| Tool | Category | Provenance | Rationale |
|------|----------|------------|-----------|
| `execute_download_from_queue` | IMPORT | True | Downloads + stores external dataset into workspace |
| `load_modality` | IMPORT | True | Loads local file via adapter into workspace |
| `validate_modality_compatibility` | QUALITY | True | Assesses multi-modal data fitness for integration |
| `concatenate_samples` | PREPROCESS | True | Transforms separate samples → unified AnnData |
| `create_mudata_from_modalities` | PREPROCESS | True | Combines modalities → MuData representation |
| `list_available_modalities` | UTILITY | False | Lists workspace contents; read-only |
| `get_modality_details` | UTILITY | False | Shape/column inspection; read-only |
| `remove_modality` | UTILITY | False | Workspace cleanup; no scientific transformation |
| `get_adapter_info` | UTILITY | False | Displays registry info; read-only |
| `get_queue_status` | UTILITY | False | Download queue status; read-only |
| `execute_custom_code` | CODE_EXEC | False | Arbitrary code execution escape hatch |

**Key boundary decisions:**
- `get_modality_details` and `remove_modality` both have `ir=ir` in their `log_tool_usage` calls, but they are UTILITY (read-only + workspace cleanup). UTILITY can still call `log_tool_usage` — but do NOT declare `provenance: True` unless the tool's primary purpose is data transformation.
- `prepare_dataset_download` queues a download — it's administrative (UTILITY), not data loading (IMPORT). The IMPORT only happens when `execute_download_from_queue` runs.
- `process_publication_entry` is PREPROCESS because it transforms raw publication queue entries into structured, enriched metadata records (substantial data transformation).

### Parent agent configuration note

Both `research_agent` and `data_expert` are parents of `metadata_assistant` at runtime, but neither has DELEGATE tools in their base tools list. DELEGATE tools are injected via `delegation_tools` parameter from `graph.py`. This means `is_parent_agent = False` in their contract tests — the MVP parent check cannot see runtime-injected delegation tools.

---

## Runtime Monitoring

The `AquadifMonitor` service (`lobster/core/aquadif_monitor.py`) tracks AQUADIF compliance at runtime. It is wired into the callback chain and requires zero configuration from plugin authors — your tools are automatically monitored once metadata is assigned.

**What it tracks:**
- **Category distribution** — per-session count of tool invocations by category (`get_category_distribution()`)
- **Provenance status** — per-tool status: `real_ir` (genuine AnalysisStep), `hollow_ir` (ir=None bridge), `missing` (no provenance call observed) (`get_provenance_status()`)
- **CODE_EXEC log** — bounded circular buffer (100 entries) of custom code executions with agent attribution (`get_code_exec_log()`)
- **Session summary** — structured dict consumable by Omics-OS Cloud SSE enrichment (`get_session_summary()`)

**How it's wired:**
1. `client.py` constructs `AquadifMonitor(tool_metadata_map={})`
2. `graph.py` builds `tool_name -> {categories, provenance}` lookup dict from all agent tools and populates the monitor's map
3. `TokenTrackingCallback.on_tool_start` calls `monitor.record_tool_invocation()` (single injection point — no other handlers call monitor)
4. `DataManagerV2.log_tool_usage` calls `monitor.record_provenance_call(tool_name, has_real_ir)` — provenance detection by observation

**Design properties:** Pure stdlib (threading, collections.deque), fail-open (monitor errors never crash tool invocations), thread-safe, bounded data structures.

**For plugin authors:** You do NOT need to interact with AquadifMonitor directly. Assign `.metadata` and `.tags` to your tools, and the monitor will track them automatically.

## Migrating Existing Agents to AQUADIF

Use this checklist when adding AQUADIF metadata to an agent that already has tools but lacks `.metadata` and `.tags`.

### Migration Checklist

1. **Inventory tools** — `grep -rn "@tool" packages/lobster-<domain>/lobster/agents/<domain>/`. Skip delegation tools (auto-tagged by graph builder).
2. **Categorize each tool** — Primary category first ("What is the ONE thing this tool does?"). Secondary only if substantial. Apply the 80% rule: if 80%+ of logic is one category, use only that.
3. **Determine provenance** — Check primary category against the Quick Reference table above.
4. **Add metadata inline** — After each `@tool` closure, add `.metadata` and `.tags` (use string literals, not enum imports).
5. **Create contract test** — One `AgentContractTestMixin` class per agent. Set `is_parent_agent = False` for child agents.
6. **Run contract tests** — `pytest -m contract tests/agents/test_aquadif_<domain>.py -v`. All 14 mixin methods must pass.
7. **Run existing tests** — `pytest packages/lobster-<domain>/tests/ -v`. Confirm zero regressions — metadata assignment does not change tool behavior.

**Common failures:** Missing `.tags`, `.tags` ≠ `.metadata["categories"]`, provenance flag contradicts primary category, missing `log_tool_usage(ir=ir)` in provenance-required tool.

### Worked Example: Transcriptomics (22 tools)

The transcriptomics package was the first migrated. All 22 tools across 2 files:

**transcriptomics_expert.py (15 tools):**

| Tool | Category | Prov | Notes |
|---|---|---|---|
| `cluster_cells` | ANALYZE | Yes | |
| `subcluster_cells` | ANALYZE | Yes | |
| `evaluate_clustering_quality` | QUALITY | Yes | |
| `find_marker_genes` | ANALYZE | Yes | |
| `detect_doublets` | QUALITY | Yes | |
| `integrate_batches` | PREPROCESS | Yes | |
| `compute_trajectory` | ANALYZE | Yes | |
| `import_bulk_counts` | IMPORT | Yes | |
| `merge_sample_metadata` | ANNOTATE | Yes | Enriches obs with labels, not file loading |
| `assess_bulk_sample_quality` | QUALITY | Yes | |
| `filter_bulk_genes` | FILTER | Yes | |
| `normalize_bulk_counts` | PREPROCESS | Yes | |
| `detect_batch_effects` | QUALITY | Yes | |
| `convert_gene_identifiers` | ANNOTATE | Yes | ID mapping = biological meaning, not transform |
| `prepare_bulk_for_de` | PREPROCESS | Yes | Reshaping = data transformation |

**shared_tools.py (7 tools):**

| Tool | Category | Prov | Notes |
|---|---|---|---|
| `check_data_status` | UTILITY | No | |
| `assess_data_quality` | QUALITY | Yes | |
| `filter_and_normalize` | PREPROCESS, FILTER | Yes | Rare multi-category: normalization 80%, filtering 20% |
| `create_analysis_summary` | UTILITY | No | |
| `select_variable_features` | FILTER | Yes | Subsets feature space, not pattern extraction |
| `run_pca` | ANALYZE | Yes | |
| `compute_neighbors_and_embed` | ANALYZE | Yes | |

**Distribution:** ANALYZE 32%, QUALITY 23%, PREPROCESS 18%, FILTER 14%, ANNOTATE 9%, IMPORT 5%, UTILITY 9%. 21/22 single-category. This lean ratio is expected.

---

**Version:** Phase 4 (2026-03-01) — Research package reference + migration guide merged
**Contract tests:** `packages/lobster-research/tests/agents/test_aquadif_research.py`
