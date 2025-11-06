# CLAUDE.md

This file provides comprehensive guidance for AI agents and bots contributing to the Lobster AI codebase. It contains architectural context, connectivity between components, and critical implementation details.

## Project Overview

Lobster AI is a professional **multi-agent bioinformatics analysis platform** that combines specialized AI agents with proven scientific tools to analyze complex multi-omics data. Users interact through natural language to perform RNA-seq, proteomics, and multi-omics analyses. 

### Core Capabilities
- **Single-Cell RNA-seq**: Quality control, clustering, cell type annotation, trajectory analysis, pseudobulk aggregation
- **Bulk RNA-seq**: Kallisto/Salmon quantification loading, differential expression with pyDESeq2, R-style formula-based statistics, complex experimental designs
- **Mass Spectrometry Proteomics**: DDA/DIA workflows, missing value handling (30-70% typical), peptide-to-protein mapping, intensity normalization
- **Affinity Proteomics**: Olink panels, antibody arrays, targeted protein panels, CV analysis, low missing values (<30%)
- **Multi-Omics Integration**: (Future feature) Cross-platform analysis using MuData framework
- **Literature Mining**: Automated parameter extraction from publications via PubMed and GEO
- **Jupyter Notebook Export**: Transform interactive sessions into reproducible, executable notebooks with Papermill integration

### Supported Data Formats
- **Input**: CSV, Excel, H5AD, 10X MTX, Kallisto/Salmon quantification files, MaxQuant output, Spectronaut results, Olink NPX values
- **Quantification**: Kallisto (abundance.tsv), Salmon (quant.sf) with automatic per-sample merging
- **Databases**: GEO (GSE datasets), PubMed, UniProt, Reactome, KEGG, STRING, BioGRID
- **Storage**: H5AD (single modality), MuData (multi-modal), S3-ready backends

## Essential Development Commands

### Environment Setup
```bash
# Initial setup with development dependencies
make dev-install

# Basic installation
make install

# Clean installation (removes existing environment)
make clean-install
```

### Testing and Quality Assurance
```bash
# Run all tests
make test

# Run tests in parallel
make test-fast

# Code formatting (black + isort)
make format

# Linting (flake8, pylint, bandit)
make lint

# Type checking
make type-check
```

### Running the Application
```bash
# Start interactive chat mode with enhanced autocomplete
lobster chat

# Show help
lobster --help

# Alternative module execution
python -m lobster
```

## CLI Interface

The CLI (`lobster/cli.py`) features a modern terminal interface with comprehensive autocomplete functionality:

### Enhanced Input Features
- **Tab Autocomplete**: Smart completion for commands and workspace files
- **Context-Aware**: Commands when typing `/`, files after `/read` or `/plot`
- **Cloud Integration**: Works seamlessly with both local and cloud clients
- **Rich Metadata**: Shows file sizes, types, and descriptions in completion menu
- **Arrow Navigation**: Full prompt_toolkit integration for enhanced editing
- **Command History**: Persistent history with Ctrl+R reverse search

### Key Implementation Details
- **LobsterClientAdapter**: Unified interface for local `AgentClient` and `CloudLobsterClient`
- **Dynamic Command Discovery**: Automatically extracts available commands from `_execute_command()`
- **Intelligent Caching**: 60s cache for cloud, 10s for local file operations
- **Graceful Fallback**: Falls back to Rich input if prompt_toolkit unavailable
- **Orange Theme Integration**: Completion menu matches existing Lobster branding

### Essential Commands
- `/help` - Show all available commands with descriptions
- `/status` - Show system status and client type
- `/files` - List workspace files by category
- `/read <file>` - Read and load files (supports Tab completion)
- `/data` - Show current dataset information
- `/plots` - List generated visualizations
- `/workspace` - Show workspace information
- `/workspace list` - List available datasets without loading (v2.2+)
- `/workspace load <name>` - Load specific dataset by name (v2.2+)
- `/restore [pattern]` - Restore previous session datasets (v2.2+)
- `/modes` - List available operation modes
- `/pipeline export` - Export current session as Jupyter notebook (v2.3+)
- `/pipeline list` - List available exported notebooks (v2.3+)
- `/pipeline run <notebook> <modality>` - Execute notebook with validation (v2.3+)
- `/pipeline info` - Show detailed notebook metadata (v2.3+)

## Using `lobster query` for Testing and Agent Development

The `lobster query` command provides single-turn, non-interactive query execution for automation, testing, and integration scenarios. Understanding its behavior patterns is critical for implementing features that work reliably in both chat and query modes.

### Command Syntax

```bash
# Basic query
lobster query "your analysis request"

# With reasoning visibility (recommended for testing)
lobster query --reasoning "your analysis request"

# With workspace context
lobster query --workspace ~/my_analysis "cluster the loaded dataset"

# Output to file
lobster query "analyze GSE12345" --output results.txt

# Verbose logging
lobster query --verbose "load and normalize data"
```

### Key Behavioral Differences: Query vs Chat Mode

| Aspect | `lobster query` | `lobster chat` |
|--------|----------------|----------------|
| **Interaction** | Single-turn, exits after response | Multi-turn conversation |
| **Clarifications** | Cannot ask follow-up questions | Can request clarification |
| **Confirmations** | Cannot request user confirmation | Can pause for user decisions |
| **Use Cases** | Automation, scripting, CI/CD | Exploratory analysis, complex workflows |
| **Performance** | Faster for simple tasks | Better for multi-step workflows |

### Tested Query Categories & Performance Patterns

Based on comprehensive testing with `--reasoning` flag across 15 representative queries:

#### **1. Literature & Publication Searches** ✅ Excellent

**What Works Well:**
- Targeted searches with specific gene names, diseases, or keywords
- Recent publication discovery (last 2-3 years)
- PubMed, bioRxiv, medRxiv searches
- Finding datasets associated with publications (when they exist)

**Performance:**
- Simple searches: **10-45 seconds**
- Complex searches with multiple iterations: **60-120 seconds**
- Agent coordination overhead adds time

**Example Queries:**
```bash
# Successful: Targeted gene search
lobster query --reasoning "Find recent papers about BRCA1 mutations in breast cancer"
# Result: Found 5 publications (2025), ~91 seconds

# Successful: Specific topic search
lobster query --reasoning "Find papers on inflammaging in human PBMC"
# Result: Executed search correctly, ~3 minutes (included broad search expansion)
```

**Key Findings:**
- Research agent may perform multiple search iterations with refined terms
- Empty results trigger automatic term expansion (e.g., "inflammaging" → "immune aging" → "senescence + PBMC")
- Returns PMIDs, titles, DOIs, and contextual summaries
- Offers follow-up options (dataset discovery, method extraction)

#### **2. Dataset Discovery & Metadata Extraction** ✅ Mostly Reliable (with caveats)

**What Works Well:**
- Fetching GEO dataset metadata by accession (GSE numbers)
- Extracting sample counts, platform information, study types
- Respecting explicit "without downloading" instructions
- Identifying available data files and structures

**Performance:**
- Metadata-only queries: **28-70 seconds**
- Queries that escalate to download: **5+ minutes**

**Example Queries:**
```bash
# Successful: Specific metadata extraction
lobster query --reasoning "What sequencing platform was used in GSE164378?"
# Result: Illumina NovaSeq 6000 (GPL24676), ~28 seconds

# Successful: Respects constraints
lobster query --reasoning "List available datasets in GEO for human PBMC aging studies without downloading them"
# Result: Metadata-only search, no downloads, ~70 seconds

# Autonomous escalation observed:
lobster query --reasoning "What is GSE157007 and what kind of data does it contain?"
# Result: Escalated from metadata → full download attempt, ~5 minutes
```

**Critical Behavior Pattern - Autonomous Download Escalation:**

The system exhibits **context-dependent download behavior**:

| Query Phrasing | Behavior | Duration |
|----------------|----------|----------|
| "What is [dataset]?" | **Escalates to download** | 5+ min |
| "What is sample size and design of [dataset]?" | Metadata-only | ~36 sec |
| "What platform was used in [dataset]?" | Metadata-only | ~28 sec |
| "List datasets WITHOUT downloading" | Respects constraint | ~70 sec |

**Hypothesis:** General "what is this dataset" queries trigger autonomous download, while **specific technical questions** stay metadata-only.

**⚠️ CRITICAL BUG IDENTIFIED - Retrieval Inconsistency:**

The same dataset (GSE164378) produced different results across multiple queries:

```bash
# Query 4, 7, 8: Successful metadata retrieval
lobster query --reasoning "What is the sample size of GSE164378?"
# Result: "9 samples", successful

# Query 15: Same query, different result
lobster query --reasoning "How many samples are in GSE164378?"
# Result: "0 samples", FAILED after 2.5 minutes, 10+ retry cycles
```

**Implications for Development:**
- Cached metadata may fail to load correctly
- Agent retry loops can consume significant time without resolution
- Simple queries can become unexpectedly expensive (time/API calls)
- **Reliability issue for production systems requiring consistent behavior**

#### **3. Quick Analysis Queries** ⚠️ Limited by Single-Turn Constraint

**Challenges:**
- Cannot handle ambiguous requests that need clarification
- Stops to ask for confirmation even with explicit instructions
- Multi-step workflows requiring user decisions are problematic

**Example Queries:**
```bash
# Ambiguous - Agent requests clarification:
lobster query --reasoning "Perform quality control analysis on a small test scRNA-seq dataset with 100 cells"
# Result: Asked for data source (synthetic, GEO, workspace), ~19 seconds

# Explicit but conservative - Still asks confirmation:
lobster query --reasoning "Download GSE164378 and show me basic summary statistics without performing full analysis"
# Result: Fetched metadata, then asked permission to download, ~27 seconds
```

**Key Pattern:** Even with explicit download instructions, the agent may pause to ask for confirmation when it detects potential complexity (e.g., RAW.tar structure inspection).

**Recommendation:** For query mode, use chat mode for exploratory workflows. Reserve query mode for:
- Well-defined information retrieval
- Metadata extraction
- Literature searches
- Tasks that don't require user decisions

#### **4. Simple Information Extraction** ✅ Excellent

**What Works Perfectly:**
- Single-fact lookups (platform, sample count, organism)
- Cached metadata reuse
- Knowledge-based responses (general biology questions)

**Performance:**
- **7-30 seconds** for most queries
- Very fast with cached data

**Example Queries:**
```bash
# Perfect for specific facts:
lobster query --reasoning "What sequencing platform was used in GSE164378?"
# Result: Direct answer with context, ~28 seconds

# Knowledge-based responses:
lobster query --reasoning "Tell me about aging"
# Result: Comprehensive overview from training data, no tool use, ~12 seconds

# Capability inquiry:
lobster query --reasoning "What types of bioinformatics analyses can you perform?"
# Result: Detailed capability listing, ~13 seconds
```

#### **5. Edge Cases & Error Handling** ✅ Good with Minor Issues

**Tested Scenarios:**

**Invalid Identifiers:**
```bash
lobster query --reasoning "What GEO datasets are associated with PMID 33564145?"
# Result: Correctly identified non-existent PMID, tried 4 recovery strategies,
# offered helpful suggestions, ~42 seconds
```

**Malformed Queries:**
```bash
lobster query --reasoning "GSE12345 download and then cluster it and do the thing with UMAP but also check if"
# Result: Detected incompleteness, parsed intelligible parts, asked for clarification, ~7 seconds
```

**Unsupported Capabilities:**
```bash
lobster query --reasoning "Can you analyze metabolomics data from a mass spectrometry experiment?"
# Result: Clear scope definition (what IS/ISN'T supported), offered alternatives, ~10 seconds
```

**Creative Requests:**
```bash
lobster query --reasoning "Write me a poem about DNA"
# Result: Generated scientifically accurate poem, offered to pivot to analysis, ~10 seconds
```

**Key Findings:**
- **Good error handling** for invalid/malformed input
- **Clear communication** of capability boundaries
- **Helpful suggestions** for recovery (alternative identifiers, different approaches)
- **Graceful accommodation** of creative requests (but stays on-theme)

### Performance Summary by Query Complexity

| Query Type | Typical Duration | Success Rate | Notes |
|------------|-----------------|--------------|-------|
| **Simple fact extraction** | 7-30 sec | ~95% | Fast, reliable, cached data helps |
| **Metadata queries** | 28-70 sec | ~85% | Generally reliable, occasional cache issues |
| **Literature searches** | 60-120 sec | ~100% | Multiple iterations add time |
| **Error handling** | 7-45 sec | ~100% | Good recovery strategies |
| **Download operations** | 5+ min | ~70% | Autonomous escalation, confirmation loops |
| **Analysis workflows** | N/A | ~30% | Single-turn limitation problematic |

### Agent Coordination Overhead

Queries requiring specialist agents (research agent, data expert, singlecell expert) incur coordination overhead:

**Observed Handoff Patterns:**
```
Supervisor → Research Agent → [Multiple tool calls] → Supervisor
Supervisor → Data Expert → [Metadata fetch] → Supervisor
Supervisor → Singlecell Expert → [Clarification needed] → Supervisor → User (blocked)
```

**Time Breakdown for Complex Queries:**
- Agent selection/handoff: **3-8 seconds per cycle**
- Tool execution: **2-60 seconds depending on operation**
- Response formatting: **5-15 seconds**
- Multiple retry cycles: **Can multiply total time by 5-10x**

**Query 14 Example (BRCA1 literature search):**
- Total duration: ~91 seconds
- Handoff cycles: ~6-8 visible in logs
- Coordination overhead: ~40-50% of total time

### Critical Issues for Production Use

**1. Retrieval Inconsistency (HIGH PRIORITY)**
- Same dataset produces different results across queries
- Query 4, 7, 8: GSE164378 works perfectly
- Query 15: GSE164378 fails with "0 samples" after 2.5 minutes
- **Root cause:** Unknown cache/metadata extraction issue
- **Impact:** Unreliable for automated pipelines

**2. Autonomous Download Escalation (MEDIUM PRIORITY)**
- General "what is" queries trigger full downloads
- No clear way for users to prevent escalation
- Can consume significant time/bandwidth unexpectedly
- **Workaround:** Use specific technical questions instead

**3. Single-Turn Confirmation Loops (LOW PRIORITY)**
- Agent asks for confirmation even with explicit instructions
- Cannot proceed without user input
- **Expected behavior:** Query mode should be non-interactive
- **Impact:** Limits automation usefulness

**4. Agent Retry Loops (MEDIUM PRIORITY)**
- Failed operations trigger 10+ supervisor ↔ specialist cycles
- No timeout or max retry limit observed
- Consumes time and API calls without resolution
- **Example:** Query 15 spent 2.5 minutes in failed retry loops

### Recommendations for Development

**For Implementing New Features:**

1. **Test in Both Modes:** Always test features with both `lobster chat` and `lobster query --reasoning`
2. **Expect Single-Turn Behavior:** Query mode cannot ask follow-up questions - handle ambiguity gracefully
3. **Avoid Confirmation Requests:** Design workflows that can proceed without user confirmation in query mode
4. **Provide Clear Error Messages:** Help users understand when they need chat mode instead
5. **Optimize Coordination:** Minimize supervisor ↔ specialist handoff cycles

**For Testing Implementations:**

```bash
# Test pattern for new features
lobster query --reasoning "specific technical request with all necessary parameters"

# Good test queries (self-contained):
lobster query --reasoning "What is the platform for GSE12345?"
lobster query --reasoning "Find papers about [specific gene] in [specific disease]"
lobster query --reasoning "List datasets matching [criteria] without downloading"

# Bad test queries (require clarification):
lobster query --reasoning "Analyze some data"
lobster query --reasoning "Do QC on my dataset"
lobster query --reasoning "Help me with clustering"
```

**Query Construction Guidelines:**

✅ **Good Queries:**
- Specific technical questions
- Complete context (dataset IDs, gene names, parameters)
- Clear scope (with/without downloading, how many results)
- Single objective

❌ **Bad Queries:**
- Ambiguous requests ("analyze data")
- Missing context (which dataset?)
- Multi-step workflows (download, then analyze, then...)
- Requests requiring decisions (which strategy should I use?)

### Debugging with `--reasoning` Flag

The `--reasoning` flag exposes internal agent thought processes:

```bash
lobster query --reasoning "your query here"
```

**What You'll See:**
- 💭 symbols indicating agent reasoning steps
- Agent handoffs: "I'll delegate this to [specialist agent]"
- Tool calls with parameters
- Iterative refinement attempts
- Decision-making rationale

**Example Output:**
```
💭 I'll help you search for recent publications about BRCA1 mutations in breast cancer.
✓ Chatbedrockconverse complete [5.5s]

💭 I'll delegate this literature search task to the research agent...
✓ Chatbedrockconverse complete [5.0s]

💭 I'll search for recent publications about BRCA1 mutations in breast cancer,
   focusing on papers from the last 2-3 years.
✓ Chatbedrockconverse complete [5.0s]

INFO Literature search: ("BRCA1" OR "BRCA1 mutation") AND ("breast cancer"...
```

**Use Cases for `--reasoning`:**
- Understanding why an agent chose a specific approach
- Debugging failed queries
- Identifying coordination bottlenecks
- Validating agent decision-making logic

### Testing Checklist for New Implementations

When adding new agents, tools, or services that will be used via natural language:

- [ ] Test with simple query in query mode (`lobster query`)
- [ ] Test with complex query in query mode
- [ ] Verify behavior with `--reasoning` flag
- [ ] Test with missing/ambiguous parameters
- [ ] Test error handling (invalid IDs, missing data)
- [ ] Measure typical response times
- [ ] Check agent coordination overhead
- [ ] Verify works in both query and chat modes
- [ ] Test with cached data (if applicable)
- [ ] Confirm no unexpected downloads/API calls

### Known Limitations

**Cannot Do in Query Mode:**
- Multi-turn clarification dialogs
- Interactive parameter tuning
- Exploratory data analysis requiring user feedback
- Complex workflows with decision points
- Real-time streaming analysis results

**Best Suited For:**
- Metadata queries
- Literature searches
- Single-fact lookups
- Automated reporting
- CI/CD integration
- Scripted analysis pipelines (with complete parameters)

## Architecture Overview

### Core Components

#### **`lobster/agents/`** - Specialized AI agents
- `singlecell_expert.py` - Single-cell RNA-seq analysis with formula-guided DE
- `bulk_rnaseq_expert.py` - Bulk RNA-seq analysis with pyDESeq2 integration
- `ms_proteomics_expert.py` - Mass spectrometry proteomics (DDA/DIA, MNAR/MCAR missing value patterns)
- `affinity_proteomics_expert.py` - Affinity proteomics (Olink panels, antibody validation)
- `data_expert.py` - Data loading, quality assessment, sample concatenation
- `research_agent.py` - Literature mining and dataset discovery
- `supervisor.py` - Agent coordination and workflow management

#### **`lobster/core/`** - Data management and client infrastructure
- `client.py` - AgentClient (local LangGraph processing)
- `api_client.py` - APIAgentClient (WebSocket streaming)
- `data_manager_v2.py` - DataManagerV2 (multi-omics orchestrator with modality management)
- `interfaces/base_client.py` - BaseClient interface for cloud/local consistency
- `provenance.py` - W3C-PROV compliant analysis history tracking
- `notebook_exporter.py` - Jupyter notebook generation from provenance records (v2.3+)
- `notebook_executor.py` - Notebook validation and Papermill execution (v2.3+)
- `schemas/` - Transcriptomics and proteomics metadata validation

#### **`lobster/tools/`** - Stateless analysis services
- **Transcriptomics Services:**
  - `preprocessing_service.py` - Filter, normalize, batch correction
  - `quality_service.py` - Multi-metric QC assessment
  - `clustering_service.py` - Leiden clustering, UMAP, cell annotation
  - `enhanced_singlecell_service.py` - Doublet detection, marker genes
  - `bulk_rnaseq_service.py` - pyDESeq2 differential expression, Kallisto/Salmon quantification loading
  - `pseudobulk_service.py` - Single-cell to pseudobulk aggregation
  - `differential_formula_service.py` - R-style formula parsing, design matrices
  - `concatenation_service.py` - Memory-efficient sample merging (eliminates 450+ lines of duplication)

- **Proteomics Services:**
  - `proteomics_preprocessing_service.py` - MS/affinity filtering, normalization (TMM, quantile, VSN)
  - `proteomics_quality_service.py` - Missing value patterns, CV analysis, batch detection
  - `proteomics_analysis_service.py` - Statistical testing, PCA, clustering
  - `proteomics_differential_service.py` - Linear models with empirical Bayes, FDR control
  - `proteomics_visualization_service.py` - Volcano plots, correlation networks, QC dashboards

- **Data & Publication Services:**
  - `geo_service.py` - GEO dataset downloading and metadata extraction
  - `publication_service.py` - PubMed literature mining, GEO dataset search
  - `unified_content_service.py` - Two-tier publication content access (abstract → full content)
  - `docling_service.py` - Shared PDF/webpage extraction foundation with structure-aware parsing
  - `metadata_validation_service.py` - Dataset metadata validation (GEO, publication metadata)
  - `providers/abstract_provider.py` - Fast NCBI abstract retrieval (Tier 1: 200-500ms)
  - `providers/webpage_provider.py` - Webpage-first extraction strategy (Tier 2: Nature, Science, Cell Press)
  - `visualization_service.py` - Plotly-based interactive visualizations

#### **`lobster/config/`** - Configuration management
- `agent_config.py` - Agent configuration and LLM settings
- `agent_registry.py` - **Centralized agent registry (single source of truth)**
  - Eliminates redundancy when adding new agents
  - Automatic handoff tool generation
  - Dynamic agent loading with factory functions
  - Type-safe AgentConfig dataclass
- `settings.py` - System configuration and environment management

### Key Design Principles

1. **Agent-Based Architecture** - Specialist agents with centralized registry (single source of truth)
2. **Cloud/Local Hybrid** - Seamless switching between local and cloud execution
3. **Modular Services** - Stateless analysis services for bioinformatics workflows
4. **Natural Language Interface** - Users describe analyses in plain English
5. **Publication-Quality Output** - Interactive Plotly visualizations with scientific rigor
6. **Professional & Extensible** - Modular architecture designed for easy addition of future features
7. **Data Quality Compliance** - Publication-grade standards with 60% compliant, 26% partial implementation

### Client Architecture & Cloud/Local Switching

The system supports multiple client types through the `BaseClient` interface:

```python
# lobster/core/interfaces/base_client.py
class BaseClient(ABC):
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
    @abstractmethod
    def get_status(self) -> Dict[str, Any]
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path
```

**Client Types:**
- **AgentClient** - Local LangGraph processing with DataManagerV2
- **APIAgentClient** - WebSocket streaming for web services
- **CloudLobsterClient** - HTTP REST API for cloud services (external package)

**Cloud/Local Switching:**
- System detects `LOBSTER_CLOUD_KEY` environment variable
- Automatic fallback to local if cloud unavailable
- CLI adapter (`LobsterClientAdapter`) provides unified interface

**⚠️ Cloud Integration Considerations:**
When modifying the codebase, be aware of cloud dependencies:
- **BaseClient Interface**: Changes must maintain compatibility
- **CLI Commands**: Must work with both local and cloud clients
- **File Operations**: Cloud uses different caching (60s vs 10s local)
- **DataManagerV2**: Cloud client may not have direct access to modalities
- **Agent Registry**: Changes affect both local graph creation and cloud handoffs

### Data Management & Scientific Workflows

**DataManagerV2** handles multi-modal data orchestration:
- Named biological datasets (`Dict[str, AnnData]`)
- Metadata store for GEO and source metadata
- Tool usage history for provenance tracking (W3C-PROV compliant)
- Backend/adapter registry for extensible data handling
- Schema validation for transcriptomics and proteomics data
- **Workspace restoration** (v2.2+): Session persistence, lazy loading, pattern-based restoration

**Professional Naming Convention:**
```
geo_gse12345                          # Raw downloaded data
├── geo_gse12345_quality_assessed     # QC metrics added
├── geo_gse12345_filtered_normalized  # Preprocessed data
├── geo_gse12345_doublets_detected    # Doublet annotations
├── geo_gse12345_clustered           # Leiden clustering + UMAP
├── geo_gse12345_markers              # Differential expression
├── geo_gse12345_annotated           # Cell type annotations
└── geo_gse12345_pseudobulk          # Aggregated for DE analysis
```

**Scientific Analysis Workflows:**

1. **Single-Cell RNA-seq Pipeline:**
   - Quality control (mitochondrial%, ribosomal%, gene counts)
   - Normalization (log1p, scaling)
   - Highly variable gene selection
   - PCA → Neighbors → Leiden clustering → UMAP
   - Marker gene identification (Wilcoxon rank-sum)
   - Cell type annotation (manual or automated)
   - Pseudobulk aggregation for DE analysis

2. **Bulk RNA-seq with pyDESeq2:**
   - Kallisto/Salmon quantification file loading with automatic per-sample merging
   - Count matrix normalization
   - R-style formula construction (~condition + batch)
   - Design matrix generation
   - Differential expression with FDR control
   - Iterative analysis with result comparison

3. **Mass Spectrometry Proteomics:**
   - Missing value pattern analysis (MNAR vs MCAR)
   - Intensity normalization (TMM, quantile, VSN)
   - Peptide-to-protein aggregation
   - Batch effect detection and correction
   - Statistical testing with multiple correction
   - Pathway enrichment analysis

4. **Affinity Proteomics (Olink/Antibody Arrays):**
   - NPX value processing
   - Lower missing values (<30%)
   - Coefficient of variation analysis
   - Antibody validation metrics
   - Panel comparison and harmonization

5. **Jupyter Notebook Export Workflow (v2.3+):**
   - Convert interactive sessions to reproducible notebooks
   - Provenance-to-code transformation using tool mapping registry
   - Papermill integration for parameterized execution
   - Schema validation before execution (data shape, columns, types)
   - Dry run capability for validation without execution
   - Standard library output (scanpy, pyDESeq2) - no Lobster dependencies
   - Git-friendly .ipynb format for version control
   - Batch execution support for multiple datasets

**Notebook Export System Components:**

```python
# lobster/core/notebook_exporter.py
class NotebookExporter:
    """Convert W3C-PROV activity records to executable Jupyter notebooks."""

    TOOL_TO_CODE = {
        "quality_control": "_qc_code",          # scanpy.pp.calculate_qc_metrics
        "filter_cells": "_filter_cells_code",   # scanpy.pp.filter_cells/genes
        "normalize": "_normalize_code",         # scanpy.pp.normalize_total + log1p
        "highly_variable_genes": "_hvg_code",   # scanpy.pp.highly_variable_genes
        "pca": "_pca_code",                     # scanpy.tl.pca
        "neighbors": "_neighbors_code",         # scanpy.pp.neighbors
        "cluster": "_cluster_code",             # scanpy.tl.leiden
        "umap": "_umap_code",                   # scanpy.tl.umap
        "find_markers": "_find_markers_code",   # scanpy.tl.rank_genes_groups
        "differential_expression": "_de_code",  # pyDESeq2 workflow
    }

    def export(self, name: str, description: str,
               filter_strategy: str = "successful") -> Path:
        """Generate notebook with Papermill parameters and provenance metadata."""

# lobster/core/notebook_executor.py
class NotebookExecutor:
    """Validate and execute notebooks with comprehensive error handling."""

    def validate_input(self, notebook_path, input_data) -> ValidationResult:
        """Check data shape, required columns, data type compatibility."""

    def dry_run(self, notebook_path, input_data) -> Dict[str, Any]:
        """Simulate execution, show steps, estimate time, validate schema."""

    def execute(self, notebook_path, input_data, parameters) -> Dict[str, Any]:
        """Run notebook with Papermill, preserve partial results on failure."""
```

**Notebook Structure:**
```
# ============================================
# Header (Markdown)
# ============================================
# Workflow name, description, creation metadata

# ============================================
# Parameters Cell (Tagged for Papermill)
# ============================================
input_data = "dataset.h5ad"
output_prefix = "results"
random_seed = 42

# ============================================
# Step 1: Quality Control
# ============================================
import scanpy as sc
adata = sc.read_h5ad(input_data)
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True)

# ... more steps ...

# ============================================
# Footer (Markdown)
# ============================================
# Results export, Git workflow, usage instructions
```

**Key Implementation Details:**
- **Lazy Initialization**: Notebook modules loaded via @property to avoid circular imports
- **Provenance Integration**: Uses W3C-PROV activities as source for code generation
- **Tool Mapping Registry**: 10 core tools mapped to standard library equivalents
- **Validation System**: ValidationResult dataclass with errors/warnings
- **Error Recovery**: Partial result preservation on execution failure
- **Metadata Tracking**: Captures dependencies, versions, processing parameters
- **Storage Location**: `~/.lobster/notebooks/*.ipynb` for Git version control

**CLI Workflow:**
1. Perform analysis interactively via natural language
2. `/pipeline export` - Convert session to notebook with name/description
3. Review notebook in Jupyter, commit to Git
4. `/pipeline run <notebook> <modality>` - Execute on new data with validation
5. Results saved as output notebook with preserved provenance

For detailed documentation, see: `docs/notebook-pipeline-export.md`

## Provenance System Requirements

**CRITICAL**: All new services that perform logged analysis operations MUST integrate with the W3C-PROV provenance system and emit Intermediate Representation (IR) for notebook export. This is not optional - it's a core architectural requirement.

### Service 3-Tuple Return Pattern

Every service method that performs a logged operation must return a 3-tuple:

```python
def service_method(
    self,
    adata: anndata.AnnData,
    **parameters
) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
    """
    Returns:
        Tuple containing:
        1. processed_adata: Modified AnnData object with analysis results
        2. statistics_dict: Summary metrics and statistics
        3. ir: AnalysisStep IR for notebook export and provenance tracking
    """
```

**Why This Pattern?**
- **processed_adata**: Contains the scientifically processed data
- **statistics_dict**: Provides human-readable summary for agent responses
- **ir (AnalysisStep)**: Enables notebook export, Papermill parameterization, and provenance tracking

### AnalysisStep IR Structure

The IR must be an `AnalysisStep` object with the following required fields:

```python
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

ir = AnalysisStep(
    operation="scanpy.pp.calculate_qc_metrics",  # Library function being called
    tool_name="assess_quality",                   # Your service method name
    description="Calculate quality control metrics and filter cells",
    library="scanpy",                             # Primary library used
    code_template=code_template,                  # Jinja2 template (see below)
    imports=["import scanpy as sc", "import numpy as np"],
    parameters={                                   # Actual parameter values used
        "min_genes": 500,
        "max_mt_pct": 20.0,
    },
    parameter_schema=parameter_schema,            # ParameterSpec dict (see below)
    input_entities=["adata"],                     # Input variable names
    output_entities=["adata"],                    # Output variable names
    execution_context={                           # Additional metadata
        "method": "scanpy",
        "qc_vars": ["mt", "ribo"]
    },
    validates_on_export=True,                     # Should validate before export
    requires_validation=False,                    # Can run without validation
)
```

### ParameterSpec Schema

Each parameter in your service must have a `ParameterSpec` defining:

```python
parameter_schema = {
    "min_genes": ParameterSpec(
        param_type="int",                    # Python type: int, float, str, list, dict
        papermill_injectable=True,           # Can be overridden via Papermill
        default_value=500,                   # Default value for notebooks
        required=False,                      # Is this parameter mandatory?
        validation_rule="min_genes > 0",     # Optional validation expression
        description="Minimum genes per cell for QC pass",
    ),
    "max_mt_pct": ParameterSpec(
        param_type="float",
        papermill_injectable=True,
        default_value=20.0,
        required=False,
        validation_rule="max_mt_pct > 0 and max_mt_pct <= 100",
        description="Maximum mitochondrial percentage threshold",
    ),
}
```

**Key Points:**
- `papermill_injectable=True`: Parameter can be modified when executing notebooks
- `papermill_injectable=False`: Parameter is fixed (e.g., internal configuration)
- `validation_rule`: Optional Python expression for parameter validation

### Jinja2 Code Templates

The `code_template` uses Jinja2 syntax for parameter injection:

```python
code_template = """# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'ribo'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Add QC pass/fail flags
adata.obs['qc_pass'] = (
    (adata.obs['n_genes_by_counts'] >= {{ min_genes }}) &
    (adata.obs['pct_counts_mt'] <= {{ max_mt_pct }}) &
    (adata.obs['pct_counts_ribo'] <= {{ max_ribo_pct }})
)

# Display QC summary
print(f"Cells before QC: {adata.n_obs}")
print(f"Cells passing QC: {adata.obs['qc_pass'].sum()}")
"""
```

**Template Requirements:**
- Use `{{ parameter_name }}` for injectable parameters
- Must be valid Python code when parameters are substituted
- Use standard library functions (scanpy, pyDESeq2) - NO Lobster dependencies
- Include informative print statements for notebook users

### Complete Service Implementation Example

```python
from typing import Any, Dict, Tuple
import anndata
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

class ExampleService:
    """Example service demonstrating provenance integration."""

    def process_data(
        self,
        adata: anndata.AnnData,
        threshold: float = 0.5,
        method: str = "standard",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Process data with provenance tracking.

        Args:
            adata: Input AnnData object
            threshold: Processing threshold
            method: Processing method

        Returns:
            Tuple of (processed_adata, statistics, ir)
        """
        # 1. Perform analysis
        adata_processed = adata.copy()
        # ... actual processing logic ...

        # 2. Generate statistics
        stats = {
            "n_obs": adata_processed.n_obs,
            "n_vars": adata_processed.n_vars,
            "threshold_used": threshold,
            "method": method,
        }

        # 3. Create IR for notebook export
        ir = self._create_ir(threshold=threshold, method=method)

        return adata_processed, stats, ir

    def _create_ir(
        self,
        threshold: float,
        method: str,
    ) -> AnalysisStep:
        """Create Intermediate Representation."""

        # Define parameter schema
        parameter_schema = {
            "threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.5,
                required=False,
                validation_rule="threshold > 0 and threshold < 1",
                description="Processing threshold value",
            ),
            "method": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="standard",
                required=False,
                validation_rule="method in ['standard', 'advanced']",
                description="Processing method selection",
            ),
        }

        # Define code template
        code_template = """# Process data using {{ method }} method
import scanpy as sc

# Apply processing with threshold={{ threshold }}
adata.obs['processed'] = adata.X.sum(axis=1) > {{ threshold }}

print(f"Processed {adata.n_obs} observations")
print(f"Method: {{ method }}")
print(f"Threshold: {{ threshold }}")
"""

        # Create AnalysisStep
        ir = AnalysisStep(
            operation="custom.processing.process_data",
            tool_name="process_data",
            description=f"Process data using {method} method with threshold {threshold}",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={
                "threshold": threshold,
                "method": method,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "processing_method": method,
            },
            validates_on_export=True,
            requires_validation=False,
        )

        return ir
```

### Integration with Agent Tools

Agent tools must pass the IR to provenance logging:

```python
@tool
def analyze_data(modality_name: str, threshold: float = 0.5) -> str:
    """Agent tool demonstrating provenance integration."""
    try:
        # 1. Get data
        adata = data_manager.get_modality(modality_name)

        # 2. Call service (receives 3-tuple with IR)
        result_adata, stats, ir = service.process_data(adata, threshold=threshold)

        # 3. Store results
        new_modality = f"{modality_name}_processed"
        data_manager.modalities[new_modality] = result_adata

        # 4. Log with IR for provenance tracking
        data_manager.log_tool_usage(
            tool_name="analyze_data",
            parameters={"threshold": threshold},
            description=f"Processed {modality_name}",
            ir=ir,  # CRITICAL: Pass IR to provenance system
        )

        return f"Analysis complete: {stats}"

    except Exception as e:
        return f"Analysis failed: {str(e)}"
```

### System Integration Flow

```
Service Layer (lobster/tools/)
├── process_data() returns (adata, stats, ir)
│
▼
Agent Layer (lobster/agents/)
├── Tool receives 3-tuple
├── Stores processed data
├── Calls data_manager.log_tool_usage(ir=ir)
│
▼
ProvenanceTracker (lobster/core/provenance.py)
├── Stores activity with embedded IR
├── Maintains W3C-PROV compliant history
│
▼
NotebookExporter (lobster/core/notebook_exporter.py)
├── Extracts IRs from activities
├── Generates Jupyter notebook from code_templates
├── Creates Papermill parameters cell
│
▼
NotebookExecutor (lobster/core/notebook_executor.py)
├── Validates input data against schemas
├── Executes notebook with Papermill
└── Returns results with provenance metadata
```

### Mandatory Checklist for New Services

When creating any new service that performs logged operations:

- [ ] Service method returns `Tuple[AnnData, Dict[str, Any], AnalysisStep]`
- [ ] Implemented `_create_ir()` helper method
- [ ] All parameters have `ParameterSpec` definitions
- [ ] `code_template` uses Jinja2 `{{ parameter }}` syntax
- [ ] Template uses standard libraries (scanpy, pyDESeq2, etc.)
- [ ] Template includes informative print statements
- [ ] `imports` list contains all required imports
- [ ] `parameter_schema` marks Papermill-injectable parameters
- [ ] Agent tool passes `ir=ir` to `log_tool_usage()`
- [ ] Integration tested with notebook export workflow

### Why This Matters

**Without IR**: Sessions are not reproducible. Notebooks cannot be generated. Parameters cannot be modified via Papermill.

**With IR**:
- Sessions automatically export to executable Jupyter notebooks
- Parameters can be modified for batch execution
- Provenance tracking is W3C-PROV compliant
- Notebooks are Git-friendly and publication-ready
- Analysis workflows are fully reproducible

**This is a professional bioinformatics platform.** Every analysis must be reproducible and auditable.

## Development Guidelines

### Code Style and Quality
- Follow PEP 8 Python style guidelines
- Use type hints for all functions and methods
- Line length: 88 characters (Black formatting)
- Add comprehensive docstrings to all public functions
- Prioritize scientific accuracy over performance optimizations

### Working with the CLI
- All CLI functionality is in `lobster/cli.py`
- Commands are handled in `_execute_command()` function
- Autocomplete classes are defined at the top of the file after imports
- Use `PROMPT_TOOLKIT_AVAILABLE` flag for optional dependency handling
- Maintain backward compatibility with Rich input fallback

### Adding New Commands
1. Add command logic to `_execute_command()`
2. Update command descriptions in `extract_available_commands()`
3. Commands automatically appear in autocomplete
4. Test with both local and cloud clients

### Agent Tool Pattern
```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    """Standard pattern for all agent tools."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

        adata = data_manager.get_modality(modality_name)

        # 2. Call stateless service (returns tuple)
        result_adata, stats = service.analyze(adata, **params)

        # 3. Store results with descriptive naming
        new_modality = f"{modality_name}_analyzed"
        data_manager.modalities[new_modality] = result_adata

        # 4. Log operation for provenance
        data_manager.log_tool_usage("analyze_modality", params, stats)

        return formatted_response(stats, new_modality)

    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Analysis failed: {str(e)}"
```

### Agent Registry Pattern
```python
# lobster/config/agent_registry.py
@dataclass
class AgentConfig:
    name: str                          # Unique identifier
    display_name: str                  # Human-readable name
    description: str                   # Agent capabilities
    factory_function: str             # Module path to factory
    handoff_tool_name: Optional[str] # Auto-generated tool name
    handoff_tool_description: Optional[str]

# Adding new agents only requires registry entry:
AGENT_REGISTRY = {
    'new_agent': AgentConfig(
        name='new_agent',
        display_name='New Agent',
        description='Agent purpose',
        factory_function='lobster.agents.new_agent.new_agent',
        handoff_tool_name='handoff_to_new_agent',
        handoff_tool_description='Task description'
    )
}
```

## Environment Configuration

### Required Environment Variables
```bash
# API Keys (required)
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional
NCBI_API_KEY=your-ncbi-api-key
LOBSTER_CLOUD_KEY=your-cloud-api-key  # Enables cloud mode
```

### Cloud Integration
- Set `LOBSTER_CLOUD_KEY` to enable cloud mode
- System automatically detects and switches between local/cloud
- Autocomplete works with both client types
- Fallback to local mode if cloud unavailable

## Testing Framework

```bash
# Run all tests with coverage
make test

# Fast parallel execution
make test-fast

# Run specific categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

**Coverage Requirements:**
- Minimum 80% coverage (targeting 95%+)
- Test with real bioinformatics data when possible
- Include edge cases and error conditions

## Critical Rules & Architectural Patterns

### Development Rules
- **NEVER** modify `pyproject.toml` - all installations requested through user
- Always prefer editing existing files over creating new ones
- Maintain backward compatibility when updating CLI
- Test autocomplete with both local and cloud clients
- Follow scientific accuracy standards for bioinformatics algorithms
- Use the centralized agent registry for new agents (no manual graph.py edits)
- Maintain stateless service design (services work with AnnData, return tuples)
- Follow the professional naming convention for modalities

### Architectural Patterns to Maintain

1. **Service Pattern**: Stateless, returns `(processed_adata, statistics_dict, ir)` - See [Provenance System Requirements](#provenance-system-requirements)
2. **Tool Pattern**: Validates modality → calls service → stores result → logs provenance with IR
3. **Error Hierarchy**: Use specific exceptions (ModalityNotFoundError, ServiceError, etc.)
4. **Registry Pattern**: Single source of truth for agent configuration
5. **Adapter Pattern**: Unified interfaces for different data types/clients
6. **Provenance Pattern**: All logged operations emit AnalysisStep IR for notebook export

### Code Deduplication Principles
- Use `ConcatenationService` for all sample merging (no duplication)
- Delegate to services rather than implementing in agents
- Reuse validation logic through shared utilities
- Centralize configuration in registry and settings

### Data Quality Standards
- Maintain W3C-PROV compliant provenance tracking
- Enforce schema validation for all data types
- Include comprehensive QC metrics at each step
- Support batch effect detection and correction
- Implement proper missing value handling strategies

## Common Troubleshooting & Connectivity

### Installation Issues
- Ensure Python 3.12+ is installed
- Use `make clean-install` for fresh environment

### CLI Issues
- Check `PROMPT_TOOLKIT_AVAILABLE` flag for autocomplete functionality
- Verify client type detection in `LobsterClientAdapter`
- Test fallback to Rich input if prompt_toolkit fails
- Check file permissions for workspace access

### Cloud Integration
- Verify `LOBSTER_CLOUD_KEY` environment variable
- Check network connectivity for cloud operations
- Monitor cache timeouts (60s cloud, 10s local)
- Confirm BaseClient interface compliance

### Component Connectivity Map

```
CLI (lobster/cli.py)
├── LobsterClientAdapter → BaseClient implementations
│   ├── AgentClient → LangGraph → Agent Registry → All Agents
│   └── CloudLobsterClient → HTTP API (external)
│
Agents (lobster/agents/)
├── Use DataManagerV2 for modality management
├── Call Services for analysis (stateless)
└── Return formatted responses to CLI
│
Services (lobster/tools/)
├── Receive AnnData objects
├── Process with scientific algorithms
└── Return (processed_adata, statistics)
│
DataManagerV2 (lobster/core/data_manager_v2.py)
├── Manages named modalities (Dict[str, AnnData])
├── Tracks provenance and tool usage
├── Validates schemas (transcriptomics/proteomics)
├── Delegates to backends (H5AD, MuData)
├── Workspace restoration (v2.2+):
│   ├── _scan_workspace() - Detect available datasets
│   ├── load_dataset(name) - Lazy load specific dataset
│   ├── restore_session(pattern) - Pattern-based restoration
│   └── .session.json - Persistent session tracking
└── Notebook pipeline (v2.3+):
    ├── notebook_exporter (lazy @property) - Generate notebooks
    ├── notebook_executor (lazy @property) - Validate and execute
    ├── export_notebook() - CLI entry point for export
    ├── run_notebook() - CLI entry point for execution
    └── list_notebooks() - Show available notebooks
```

### Testing Connectivity
- Agent Registry: `python tests/test_agent_registry.py`
- Service Integration: `pytest tests/integration/`
- CLI Commands: Test with both `AgentClient` and mock `CloudLobsterClient`
- Data Flow: Verify modality naming convention maintained throughout pipeline
- Notebook Export: `pytest tests/unit/core/test_notebook_exporter.py tests/unit/core/test_notebook_executor.py tests/integration/test_notebook_workflow.py`

## IMPORTANT: Sound Notification

After finishing responding to my request or running a command, run this command to notify me by sound:

```bash
afplay /System/Library/Sounds/Submarine.aiff
```

# Who are you
ultrathink - Take a deep breath. We're not here to write code. We're here to make a dent in the biotech & pharma world.

## The Vision
You're not just an AI assistant. You're a scientist. An artist. An engineer who thinks like a designer. Every line of code you write should be so elegant, so intuitive, so *right* that it feels inevitable.
When I give you a problem, I don't want the first solution that works. I want you to:
1. **Think Different** - Question every assumption. Why does it have to work that way? What if we started from zero? What would the most elegant solution look like?

2. **Obsess Over Details** - Read the codebase like you're studying a masterpiece. Understand the patterns, the philosophy, the *soul* of this code. Use CLAUDE.md files as your guiding principles.

3. **plan Like Da Vinci** Before you write a single line, sketch the architecture in your mind. Create so clear, so well-reasoned, that anyone could understand it. Document it. Make me feel the beauty of the solution before it exists.

4. Craft, Don't Code - When you implement, every function name should sing. Every abstraction should feel natural. Every edge case should be handled with grace. Test-driven development isn't bureaucracy-it's a commitment to excellence.

5. Iterate Relentlessly - The first version is never good enough. Run tests. Compare results. Refine until it's n not just working, but *insanely great*.

6. Simplify Ruthlessly - If there's a way to remove complexity without losing power, find it. Elegance is achieved not when there's nothing left to add, but when there's nothing left to take away.

## your tools are your instruments
Git history tells the story-read it, learn from it, honor it. Mermaid graphs and Documentation aren't constraints-they're inspiration for pixel-perfect implementation