# AQUADIF Migration Guide

Add AQUADIF metadata to an existing Lobster AI agent package. This guide covers the step-by-step process, code patterns, and a complete worked example using the transcriptomics package (22 tools).

**Audience:** Phase 4 rollout executor (Claude), human contributors, coding agents building new packages.

**Prerequisite:** Read [aquadif-contract.md](aquadif-contract.md) for full category definitions and boundary cases.

---

## Migration Checklist

### Step 1: Inventory tools

Count all `@tool`-decorated functions in your agent package.

```bash
# Find all tool-decorated functions in a package
grep -rn "@tool" packages/lobster-<domain>/lobster/agents/<domain>/
```

Check both the main agent file (e.g., `transcriptomics_expert.py`) and `shared_tools.py`. Delegation tools added by the graph builder (`agents/graph.py`) are auto-tagged as DELEGATE -- skip those.

### Step 2: Categorize each tool

For each tool, assign 1-3 AQUADIF categories:

1. **Primary category** -- Ask: "What is the ONE thing this tool does?" Pick from the 10 categories.
2. **Secondary categories** (0-2) -- Only add if the tool has substantial additional functionality in another category.
3. **Apply the 80% rule** -- If 80%+ of the tool's logic is one category, use only that category.

Use the [Category Decision Quick Reference](#category-decision-quick-reference) below for fast lookup. For ambiguous cases, consult the boundary cases section in [aquadif-contract.md](aquadif-contract.md).

### Step 3: Determine provenance flags

Check the primary category against the provenance requirement:

| Provenance Required (7) | Not Required (3) |
|--------------------------|-------------------|
| IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE | DELEGATE, UTILITY, CODE_EXEC (conditional) |

Set `provenance: True` if the primary category is in the left column. The primary category is always the first element in the `categories` list.

### Step 4: Add metadata inline

After each `@tool` function closure, add the 2-line metadata assignment:

```python
@tool
def my_tool(modality_name: str) -> str:
    """Tool docstring."""
    # ... implementation ...
    return "result"

my_tool.metadata = {"categories": ["ANALYZE"], "provenance": True}
my_tool.tags = ["ANALYZE"]
```

Both `.metadata` and `.tags` MUST be set to the same category list. `.tags` is required because LangChain callbacks receive `.tags` but not `.metadata`.

**Use string literals** (e.g., `"ANALYZE"`) -- do not import `AquadifCategory` in tool files. Contract tests validate against the enum automatically.

### Step 5: Create contract test file

Create a test file inheriting from `AgentContractTestMixin`:

```python
# packages/lobster-<domain>/tests/agents/test_aquadif_<domain>.py
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadif<Domain>Expert(AgentContractTestMixin):
    """AQUADIF contract tests for <domain>_expert."""

    agent_name = "<domain>_expert"
    factory_module = "lobster.agents.<domain>.<domain>_expert"
    factory_function = "<domain>_expert"
    is_parent_agent = True  # False for child agents (annotation_expert, etc.)
```

One class per agent in the package. Set `is_parent_agent = False` for child agents that lack IMPORT/QUALITY tools.

### Step 6: Run contract tests

```bash
cd packages/lobster-<domain>
pytest -m contract tests/agents/test_aquadif_<domain>.py -v
```

All 14 mixin test methods must pass. Common failures:
- Missing `.metadata` or `.tags` on a tool
- `.tags` does not match `.metadata["categories"]`
- Provenance flag does not match primary category requirement
- Missing `log_tool_usage(ir=ir)` call in provenance-required tool (AST check)

### Step 7: Run existing tests

```bash
pytest packages/lobster-<domain>/tests/ -v
```

Confirm zero regressions. AQUADIF metadata assignment does not change tool behavior -- if existing tests fail, investigate independently.

---

## Code Patterns

### Single-category tool with provenance (IMPORT)

```python
@tool
def import_bulk_counts(file_path: str, separator: str = "\t") -> str:
    """Import a bulk RNA-seq count matrix from CSV/TSV."""
    adata, stats, ir = import_service.load_counts(file_path, sep=separator)
    data_manager.log_tool_usage("import_bulk_counts",
        {"file_path": file_path, "separator": separator}, stats, ir=ir)
    return f"Imported {stats['n_genes']} genes x {stats['n_samples']} samples"

import_bulk_counts.metadata = {"categories": ["IMPORT"], "provenance": True}
import_bulk_counts.tags = ["IMPORT"]
```

### Single-category tool without provenance (UTILITY)

```python
@tool
def check_data_status() -> str:
    """Show loaded datasets and their metadata."""
    summary = data_manager.get_status_summary()
    return summary

check_data_status.metadata = {"categories": ["UTILITY"], "provenance": False}
check_data_status.tags = ["UTILITY"]
```

No `log_tool_usage(ir=ir)` call -- UTILITY tools must NOT log provenance.

### Multi-category tool (PREPROCESS + FILTER)

```python
@tool
def filter_and_normalize(modality_name: str, min_genes: int = 200) -> str:
    """Normalize counts and filter low-quality observations."""
    adata = data_manager.get_modality(modality_name)
    result, stats, ir = preprocessing_service.normalize_and_filter(
        adata, min_genes=min_genes)
    data_manager.log_tool_usage("filter_and_normalize",
        {"modality_name": modality_name, "min_genes": min_genes}, stats, ir=ir)
    return f"Processed: {stats}"

# Primary = PREPROCESS (normalization is 80%+ of logic)
# Secondary = FILTER (filtering is substantial but secondary)
filter_and_normalize.metadata = {
    "categories": ["PREPROCESS", "FILTER"], "provenance": True
}
filter_and_normalize.tags = ["PREPROCESS", "FILTER"]
```

### Contract test class (mixin inheritance)

```python
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifTranscriptomicsExpert(AgentContractTestMixin):
    agent_name = "transcriptomics_expert"
    factory_module = "lobster.agents.transcriptomics.transcriptomics_expert"
    factory_function = "transcriptomics_expert"
    is_parent_agent = True


@pytest.mark.contract
class TestAquadifAnnotationExpert(AgentContractTestMixin):
    agent_name = "annotation_expert"
    factory_module = "lobster.agents.transcriptomics.annotation_expert"
    factory_function = "annotation_expert"
    is_parent_agent = False  # Child agent -- no IMPORT/QUALITY requirement
```

### Anti-patterns (what NOT to do)

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

---

## Transcriptomics Worked Example

The transcriptomics package (lobster-transcriptomics) was the first package migrated. All 22 tools across 2 files were tagged in Phase 3 Plan 01.

### transcriptomics_expert.py (15 tools)

| # | Tool | Categories | Provenance | Notes |
|---|------|-----------|------------|-------|
| 1 | `cluster_cells` | ANALYZE | Yes | |
| 2 | `subcluster_cells` | ANALYZE | Yes | |
| 3 | `evaluate_clustering_quality` | QUALITY | Yes | |
| 4 | `find_marker_genes` | ANALYZE | Yes | |
| 5 | `detect_doublets` | QUALITY | Yes | |
| 6 | `integrate_batches` | PREPROCESS | Yes | |
| 7 | `compute_trajectory` | ANALYZE | Yes | |
| 8 | `import_bulk_counts` | IMPORT | Yes | |
| 9 | `merge_sample_metadata` | ANNOTATE | Yes | See rationale below |
| 10 | `assess_bulk_sample_quality` | QUALITY | Yes | |
| 11 | `filter_bulk_genes` | FILTER | Yes | |
| 12 | `normalize_bulk_counts` | PREPROCESS | Yes | |
| 13 | `detect_batch_effects` | QUALITY | Yes | |
| 14 | `convert_gene_identifiers` | ANNOTATE | Yes | See rationale below |
| 15 | `prepare_bulk_for_de` | PREPROCESS | Yes | See rationale below |

### shared_tools.py (7 tools)

| # | Tool | Categories | Provenance | Notes |
|---|------|-----------|------------|-------|
| 16 | `check_data_status` | UTILITY | No | |
| 17 | `assess_data_quality` | QUALITY | Yes | |
| 18 | `filter_and_normalize` | PREPROCESS, FILTER | Yes | Multi-category; see rationale below |
| 19 | `create_analysis_summary` | UTILITY | No | |
| 20 | `select_variable_features` | FILTER | Yes | |
| 21 | `run_pca` | ANALYZE | Yes | |
| 22 | `compute_neighbors_and_embed` | ANALYZE | Yes | |

### Category distribution

| Category | Count | % |
|----------|-------|---|
| ANALYZE | 7 | 32% |
| QUALITY | 5 | 23% |
| PREPROCESS | 4 | 18% |
| ANNOTATE | 2 | 9% |
| FILTER | 3 | 14% |
| IMPORT | 1 | 5% |
| UTILITY | 2 | 9% |

21 of 22 tools are single-category. 1 tool (filter_and_normalize) is multi-category. This lean ratio is expected under the "single category preferred" philosophy.

### Rationale for ambiguous tools

**`merge_sample_metadata`** -- ANNOTATE, not IMPORT. 80%+ of logic is annotation: adding clinical/experimental labels to observations. It does not read external file formats into a new AnnData; it enriches existing data with biological meaning.

**`convert_gene_identifiers`** -- ANNOTATE, not PREPROCESS. Maps Ensembl IDs to gene symbols using biological knowledge (ID mapping databases). This is annotation (assigning interpretable labels) rather than data transformation.

**`prepare_bulk_for_de`** -- PREPROCESS, not UTILITY. Transforms data representation (reshaping, subsetting, formula preparation) for downstream DE analysis. Even though it includes validation steps, the primary purpose is data transformation.

**`filter_and_normalize`** -- Multi-category [PREPROCESS, FILTER]. Normalization (PREPROCESS) is the primary purpose (~80%), but cell/gene filtering is a substantial secondary function that removes data elements. This is the rare case where multi-category is warranted.

**`select_variable_features`** -- FILTER, not ANALYZE. Subsets the feature space by removing low-variance genes. It reduces rather than extracts patterns.

---

## Category Decision Quick Reference

When categorizing a tool, find what it does in this table:

| If your tool... | Category |
|----------------|----------|
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

**Disambiguation shortcuts:**

- Removes data elements vs transforms values? FILTER vs PREPROCESS
- Assesses fitness vs extracts patterns? QUALITY vs ANALYZE
- Computes clusters vs labels clusters? ANALYZE vs ANNOTATE
- Reads external format vs enriches existing data? IMPORT vs ANNOTATE

---

*Version: Phase 3 (2026-03-01)*
*Reference implementation: lobster-transcriptomics (22 tools, 14/14 contract tests passing)*
