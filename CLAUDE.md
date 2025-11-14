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

Modern terminal interface (`lobster/cli.py`) with Tab autocomplete, cloud/local switching, and prompt_toolkit integration.

**Key Features:** Context-aware completion (commands/files), persistent history (Ctrl+R), rich metadata display, graceful fallback to Rich input
**Implementation:** LobsterClientAdapter (unified local/cloud interface), dynamic command discovery, intelligent caching (60s cloud, 10s local)

| Command Category | Commands | Description |
|------------------|----------|-------------|
| **Help & Status** | `/help`, `/status`, `/modes` | Show commands, system status, operation modes |
| **Data Management** | `/data`, `/files`, `/read <file>` | View datasets, list files (Tab completion), load files |
| **Workspace** | `/workspace`, `/workspace list`, `/workspace load <name>`, `/restore [pattern]` | Manage workspaces, list/load datasets (v2.2+) |
| **Visualization** | `/plots` | List generated visualizations |
| **Pipeline Export** | `/pipeline export`, `/pipeline list`, `/pipeline run <notebook> <modality>`, `/pipeline info` | Export to Jupyter, manage notebooks (v2.3+) |

## Using `lobster query` for Testing and Agent Development

Single-turn, non-interactive command for automation, testing, and CI/CD. Cannot ask follow-up questions or request confirmations.

### Command Syntax
```bash
lobster query "your request"                           # Basic
lobster query --reasoning "your request"               # Show agent reasoning (debugging)
lobster query --workspace ~/path "cluster dataset"     # With workspace context
lobster query "analyze GSE12345" --output results.txt  # Save output
```

### Query vs Chat Mode Comparison

| Aspect | `lobster query` | `lobster chat` |
|--------|----------------|----------------|
| **Interaction** | Single-turn, exits after response | Multi-turn conversation |
| **Clarifications** | Cannot ask follow-up questions | Can request clarification |
| **Confirmations** | Cannot request user confirmation | Can pause for user decisions |
| **Use Cases** | Automation, scripting, CI/CD, metadata queries | Exploratory analysis, complex workflows |
| **Performance** | Faster for simple tasks (7-120s typical) | Better for multi-step workflows |

### Performance by Query Type

| Query Category | Duration | Success Rate | What Works | Known Issues |
|----------------|----------|--------------|------------|--------------|
| **Literature searches** | 60-120s | ~100% | PubMed, bioRxiv searches; auto term expansion | Agent coordination overhead (40-50% of time) |
| **Metadata extraction** | 28-70s | ~85% | GEO metadata, platform info, sample counts | ⚠️ Cache inconsistency (same query → different results) |
| **Simple lookups** | 7-30s | ~95% | Single-fact queries, knowledge-based responses | - |
| **Download operations** | 5+ min | ~70% | GEO dataset downloads | Autonomous escalation on vague queries ("What is GSE...?") |
| **Analysis workflows** | N/A | ~30% | - | Requires clarification → blocked in query mode |
| **Error handling** | 7-45s | ~100% | Invalid IDs, malformed queries, unsupported features | - |

### Critical Production Issues

| Issue | Priority | Impact | Workaround |
|-------|----------|--------|------------|
| **Retrieval Inconsistency** | HIGH | Same dataset query fails randomly after 2.5min retry loops | Use chat mode for critical operations |
| **Autonomous Download Escalation** | MEDIUM | "What is [dataset]?" triggers full download (5+ min) | Ask specific questions: "What platform...?", "How many samples...?" |
| **Agent Retry Loops** | MEDIUM | 10+ supervisor↔specialist cycles without timeout | Monitor with `--reasoning` flag |
| **Confirmation Requests** | LOW | Blocks on ambiguous requests despite explicit instructions | Provide complete context in query |

### Development Guidelines

**✅ Good Queries (Self-Contained):**
- `"What is the platform for GSE12345?"`
- `"Find papers about BRCA1 in breast cancer"`
- `"List datasets matching X without downloading"`

**❌ Bad Queries (Require Clarification):**
- `"Analyze some data"` (missing context)
- `"Do QC on my dataset"` (which dataset?)
- `"Download X and cluster it"` (multi-step)

**Development Checklist:**
- [ ] Test in both `lobster query` and `lobster chat` modes
- [ ] Handle ambiguity without requesting clarification
- [ ] Avoid workflows requiring user confirmation
- [ ] Provide clear error messages indicating when chat mode is needed
- [ ] Measure response times with `--reasoning` flag
- [ ] Test with missing/invalid parameters
- [ ] Verify no unexpected downloads or long retry loops

**Debugging with `--reasoning`:**
- Shows agent reasoning steps (💭 symbols), handoffs, tool calls, decision rationale
- Use to identify coordination bottlenecks and failed retry loops
- Example: `lobster query --reasoning "Find papers about aging"`

## Architecture Overview

### Core Components

| Module | Key Files | Purpose |
|--------|-----------|---------|
| **`lobster/agents/`** | singlecell_expert, bulk_rnaseq_expert, ms_proteomics_expert, affinity_proteomics_expert, data_expert, research_agent, metadata_assistant, supervisor | Specialized AI agents for analysis domains + coordination |
| **`lobster/core/`** | client, api_client, data_manager_v2, provenance, notebook_exporter/executor, schemas/ | Client infrastructure, multi-omics orchestration, W3C-PROV tracking, Jupyter export (v2.3+) |
| **`lobster/tools/`** | **Transcriptomics:** preprocessing, quality, clustering, enhanced_singlecell, bulk_rnaseq, pseudobulk, differential_formula, concatenation<br>**Proteomics:** preprocessing, quality, analysis, differential, visualization<br>**Data:** geo, content_access (with providers/), docling, metadata_validation, visualization | Stateless analysis services: QC, normalization, DE, clustering, literature mining via provider architecture (PubMed, PMC, GEO, webpage), dataset retrieval |
| **`lobster/config/`** | agent_config, **agent_registry** (centralized single source of truth), settings | Configuration management, agent registry with auto handoff generation |

### Key Design Principles

1. **Agent-Based Architecture** - Specialist agents with centralized registry (single source of truth)
2. **Cloud/Local Hybrid** - Seamless switching between local and cloud execution
3. **Modular Services** - Stateless analysis services for bioinformatics workflows
4. **Natural Language Interface** - Users describe analyses in plain English
5. **Publication-Quality Output** - Interactive Plotly visualizations with scientific rigor
6. **Professional & Extensible** - Modular architecture designed for easy addition of future features
7. **Data Quality Compliance** - Publication-grade standards with 60% compliant, 26% partial implementation

### Research Agent Architecture (Post-Refactoring)
**Phase 1-6 Refactoring completed:** Provider infrastructure → Service consolidation → Agent specialization
- **ContentAccessService**: Unified service replacing PublicationService/UnifiedContentService (Phase 2)
- **Provider Architecture**: 5 providers (PubMed, PMC, GEO, Webpage, Abstract) with capability routing (Phase 1)
- **research_agent**: 10 tools for discovery/content analysis + workspace caching (Phase 4)
- **metadata_assistant**: 4 tools for sample mapping, metadata standardization, dataset validation (Phase 3)
- **Handoff Patterns**: Documented workflows for multi-omics, meta-analysis, control addition (Phase 5)

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
- Named biological datasets (`Dict[str, AnnData]`), metadata store (GEO/source), W3C-PROV provenance tracking
- Backend/adapter registry, schema validation (transcriptomics, proteomics, metabolomics, metagenomics)
- **Workspace restoration** (v2.2+): Session persistence, lazy loading, pattern-based restoration

**Metadata Schemas** (Phase 3+): Pydantic schemas for cross-dataset metadata standardization
- **TranscriptomicsMetadataSchema**: Sample metadata for single-cell and bulk RNA-seq
- **ProteomicsMetadataSchema**: Sample metadata for mass spec and affinity proteomics
- **MetabolomicsMetadataSchema**: Sample metadata for LC-MS, GC-MS, NMR metabolomics
- **MetagenomicsMetadataSchema**: Sample metadata for 16S, ITS, shotgun metagenomics
- Used by `metadata_assistant` agent for sample ID mapping, metadata harmonization, dataset validation

**Professional Naming Convention:** `geo_gse12345` → `_quality_assessed` → `_filtered_normalized` → `_doublets_detected` → `_clustered` → `_markers` → `_annotated` → `_pseudobulk`

**Scientific Analysis Workflows:**

| Workflow | Key Steps | Tools/Methods |
|----------|-----------|---------------|
| **Single-Cell RNA-seq** | QC → Normalize → HVG → PCA → Neighbors → Leiden → UMAP → Markers → Annotate → Pseudobulk | scanpy, Wilcoxon rank-sum, manual/auto annotation |
| **Bulk RNA-seq** | Load Kallisto/Salmon → Normalize → Formula (~condition + batch) → Design matrix → DE → Compare | pyDESeq2, R-style formulas, FDR control |
| **MS Proteomics** | Missing value analysis (MNAR/MCAR) → Normalize (TMM/quantile/VSN) → Peptide→Protein → Batch correction → Stats → Pathway | DDA/DIA, 30-70% missing typical |
| **Affinity Proteomics** | NPX processing → CV analysis → Antibody validation → Panel harmonization | Olink panels, <30% missing |
| **Notebook Export (v2.3+)** | Session → W3C-PROV → Tool mapping → Papermill params → Validation → Execute | NotebookExporter/Executor, 10 core tools, Git-friendly .ipynb |

**Notebook Export System (v2.3+):**
- **Classes**: `NotebookExporter` (provenance→code), `NotebookExecutor` (validate, dry-run, execute with Papermill)
- **Tool Mapping**: 10 core tools (QC, filter, normalize, HVG, PCA, neighbors, cluster, UMAP, markers, DE) → scanpy/pyDESeq2 equivalents
- **Features**: Lazy initialization (@property), validation (schema/data shape), error recovery (partial results), parameterization (Papermill-injectable)
- **CLI Workflow**: Analyze → `/pipeline export` → Review/commit → `/pipeline run <notebook> <modality>` → Results with provenance
- **Docs**: `docs/notebook-pipeline-export.md`

## Provenance System Requirements

**CRITICAL**: All services performing logged operations MUST integrate with W3C-PROV provenance system and emit Intermediate Representation (IR) for notebook export.

**Service 3-Tuple Return Pattern:** `Tuple[processed_adata, statistics_dict, AnalysisStep_ir]`
- **processed_adata**: Modified AnnData with results
- **statistics_dict**: Human-readable summary for agent responses
- **AnalysisStep ir**: Enables notebook export, Papermill parameterization, provenance tracking

**AnalysisStep IR Structure** (from `lobster.core.analysis_ir`):
```python
AnalysisStep(
    operation="scanpy.pp.calculate_qc_metrics",  # Library function
    tool_name="assess_quality",                   # Service method name
    description="...",                            # Human-readable
    library="scanpy",                             # Primary library
    code_template=jinja2_template,                # With {{ param }} injection
    imports=["import scanpy as sc", ...],
    parameters={"min_genes": 500, ...},           # Actual values used
    parameter_schema={"min_genes": ParameterSpec(...), ...},
    input_entities=["adata"], output_entities=["adata"],
    execution_context={...},                      # Additional metadata
    validates_on_export=True, requires_validation=False
)
```

**ParameterSpec Fields:** `param_type` (int/float/str/list/dict), `papermill_injectable` (True/False), `default_value`, `required`, `validation_rule` (Python expression), `description`

**Jinja2 Template Requirements:**
- Use `{{ parameter_name }}` for injectable parameters
- Valid Python code after substitution
- Standard libraries only (scanpy, pyDESeq2) - NO Lobster dependencies
- Include informative print statements

**Implementation Pattern:**
```python
# Service: Returns 3-tuple
def process_data(self, adata, **params) -> Tuple[AnnData, Dict, AnalysisStep]:
    processed = adata.copy()  # ... processing ...
    stats = {...}
    ir = self._create_ir(**params)  # Helper method creates AnalysisStep
    return processed, stats, ir

# Agent Tool: Passes IR to provenance
@tool
def analyze(modality_name, **params):
    adata = data_manager.get_modality(modality_name)
    result, stats, ir = service.process_data(adata, **params)
    data_manager.modalities[f"{modality_name}_processed"] = result
    data_manager.log_tool_usage(..., ir=ir)  # CRITICAL: Pass IR
    return f"Complete: {stats}"
```

**System Flow:** Service (3-tuple) → Agent Tool → log_tool_usage(ir) → ProvenanceTracker (W3C-PROV) → NotebookExporter (code generation) → NotebookExecutor (Papermill)

**Checklist for New Services:**
- [ ] Returns `Tuple[AnnData, Dict[str, Any], AnalysisStep]`
- [ ] `_create_ir()` helper with ParameterSpec for all parameters
- [ ] Jinja2 `code_template` with `{{ param }}` injection
- [ ] Standard library code (scanpy/pyDESeq2), no Lobster dependencies
- [ ] Agent tool passes `ir=ir` to `log_tool_usage()`
- [ ] Integration tested with `/pipeline export`

**Why:** Without IR, sessions are not reproducible. With IR: auto-export to Jupyter, Papermill parameterization, W3C-PROV compliance, Git-friendly notebooks, full reproducibility. **Every analysis must be auditable.**

**Examples:** See existing services in `lobster/tools/` (quality_service, preprocessing_service, clustering_service, bulk_rnaseq_service, etc.)

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

| Variable | Required | Purpose |
|----------|----------|---------|
| `AWS_BEDROCK_ACCESS_KEY` | Yes | AWS access key |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | Yes | AWS secret key |
| `NCBI_API_KEY` | No | PubMed API access |
| `LOBSTER_CLOUD_KEY` | No | Enable cloud mode (auto-detect local/cloud, autocomplete works with both, fallback to local) |

## Testing Framework

### Basic Test Commands

```bash
# Run all tests with coverage
make test

# Fast parallel execution
make test-fast

# Run specific categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run real API integration tests (requires API keys)
pytest tests/integration/ -m real_api
pytest tests/integration/ -m "real_api and slow"  # Long-running tests
```

**Coverage Requirements:**
- Minimum 80% coverage (targeting 95%+)
- Test with real bioinformatics data when possible
- Include edge cases and error conditions

### Integration Testing with Real APIs (Phase 7)

**Test Suite:** 127 tests across 5 integration test files (3988 lines)
**Status:** ✅ Complete (Phase 7 validation)
**Documentation:** See `kevin_notes/publisher/research_agent_refactor_phase_7_summary.md`

| Test File | Tests | Coverage | Purpose |
|-----------|-------|----------|---------|
| `test_research_agent_real_api.py` | 46 | 10 tools | research_agent with PubMed, PMC, GEO |
| `test_content_cascade_real_api.py` | 21 | 3-tier cascade | PMC→Webpage→PDF fallback system |
| `test_metadata_assistant_real_workflows.py` | 27 | 4 tools | Sample mapping, metadata standardization |
| `test_multi_agent_handoffs_real.py` | 20 | 4 scenarios | Supervisor-mediated handoffs |
| `test_performance_and_errors_real.py` | 13 | Benchmarking | Performance, rate limiting, error recovery |

**Markers:**
- `@pytest.mark.real_api` - Requires internet + API keys (NCBI_API_KEY, AWS_BEDROCK_*)
- `@pytest.mark.slow` - Tests >30s (performance benchmarks, multi-agent workflows)
- `@pytest.mark.integration` - Multi-component tests

**API Keys Required:**
```bash
export AWS_BEDROCK_ACCESS_KEY="your-key"
export AWS_BEDROCK_SECRET_ACCESS_KEY="your-secret"
export NCBI_API_KEY="your-ncbi-key"  # Recommended (10/sec vs 3/sec rate limit)
```

### Performance Benchmarks (Phase 7 Targets)

| Operation | Target | Test Coverage | Notes |
|-----------|--------|---------------|-------|
| **Abstract retrieval** | <500ms mean | 100 PMIDs with P95/P99 | 95%+ success rate required |
| **PMC full text** | <2s mean | 20 papers with tier tracking | 75%+ success rate (some paywalled) |
| **GEO metadata** | <3s mean | 20 datasets with validation | 75%+ success rate (some deleted) |
| **Three-tier cascade** | <10s total | PMC→Webpage→PDF | Allows 15s with network overhead |

**Performance test pattern:**
```python
from statistics import mean, stdev

latencies = []
for identifier in test_identifiers:
    start_time = time.time()
    result = service.fetch(identifier)
    latencies.append((time.time() - start_time) * 1000)  # ms

# Statistical analysis
mean_latency = mean(latencies)
sorted_latencies = sorted(latencies)
p95_latency = sorted_latencies[int(len(latencies) * 0.95)]
p99_latency = sorted_latencies[int(len(latencies) * 0.99)]

assert mean_latency < 500, f"Mean latency {mean_latency:.2f}ms exceeds 500ms target"
```

### Rate Limiting Compliance

**NCBI Rate Limits:**
- Without API key: 3 requests/second (0.34s delay)
- With API key: 10 requests/second (0.11s delay)

**Pattern:**
```python
for i, pmid in enumerate(pmids):
    start_time = time.time()
    result = pubmed_provider.fetch_abstract(pmid)
    elapsed = time.time() - start_time

    # Respectful rate limiting
    if not os.getenv("NCBI_API_KEY"):
        time.sleep(0.34)  # 3 requests per second
    else:
        time.sleep(0.11)  # 10 requests per second

    if (i + 1) % 20 == 0:
        logger.info(f"Progress: {i+1}/{len(pmids)} completed")
```

### Test Patterns and Best Practices

#### Module-Scoped Fixtures for Efficiency

```python
@pytest.fixture(scope="module")
def content_service(data_manager):
    """Create ContentAccessService instance once per module."""
    return ContentAccessService(data_manager=data_manager)

@pytest.fixture(scope="module")
def benchmark_identifiers():
    """Known stable identifiers for performance benchmarking."""
    return {
        "pmids_100": ["35042229", "33057194", ...],  # 100 PMIDs
        "geo_datasets_20": ["GSE180759", "GSE156793", ...]  # 20 datasets
    }
```

**Benefits:** Single initialization, shared across all tests in class, significant time savings

#### Batch Processing with Error Tolerance

```python
successful_retrievals = 0
failed_retrievals = []

for identifier in identifiers:
    try:
        result = service.fetch(identifier)
        if result:
            successful_retrievals += 1
    except Exception as e:
        failed_retrievals.append((identifier, str(e)))

# Assert acceptable success rate
assert successful_retrievals >= 95, f"Too many failures: {len(failed_retrievals)}/{len(identifiers)}"
```

**Success rate targets:**
- Abstract retrieval: ≥95% (100 PMIDs)
- PMC retrieval: ≥75% (some papers paywalled)
- GEO metadata: ≥75% (some datasets deleted/restricted)

#### Graceful Degradation Testing

```python
# Mix valid and invalid identifiers
mixed_identifiers = [
    "VALID_ID_1",     # Valid
    "INVALID_ID",     # Invalid
    "VALID_ID_2",     # Valid
]

successful = 0
failed = 0

for identifier in mixed_identifiers:
    try:
        result = service.fetch(identifier)
        successful += 1 if result else 0
    except Exception:
        failed += 1

# Verify partial success
assert successful >= 2, "Valid IDs should succeed"
assert failed >= 1, "Invalid IDs should fail"
```

#### Three-Tier Cascade Testing

```python
def test_cascade_pmc_to_webpage_fallback(content_service):
    """Test cascade falls back from PMC to webpage when PMC unavailable."""
    # Use identifier without PMC (forces webpage fallback)
    identifier = "10.1038/s41586-021-03852-1"  # Nature DOI

    result = content_service.get_full_content(
        source=identifier,
        prefer_webpage=True,
        max_paragraphs=100
    )

    assert result is not None
    assert "content" in result
    assert len(result["content"]) > 0

    tier_used = result.get("tier_used", "")
    logger.info(f"Cascade test - Tier used: {tier_used}")
```

#### Multi-Agent Coordination Testing

**Pattern using AgentClient (full graph):**
```python
from lobster.core.client import AgentClient

@pytest.fixture(scope="module")
def agent_client(data_manager, test_workspace):
    """Create AgentClient for multi-agent coordination testing."""
    return AgentClient(
        data_manager=data_manager,
        enable_reasoning=False,
        workspace_path=test_workspace,
    )

def test_four_agent_workflow_complete(agent_client):
    """Test complete 4-agent workflow orchestration."""
    # Step 1: Literature search (research_agent)
    result1 = agent_client.query("Search PubMed for 'breast cancer RNA-seq'")

    # Step 2: Find dataset (research_agent)
    result2 = agent_client.query("Find datasets from PMID:35042229")

    # Step 3: Standardize metadata (→ metadata_assistant handoff)
    result3 = agent_client.query("Standardize GSE180759 to transcriptomics schema")

    # Step 4: Cache to workspace (→ data_expert handoff)
    result4 = agent_client.query("Cache GSE180759 to workspace")

    # Verify all steps completed
    assert all([len(r.get("response", "")) > 0 for r in [result1, result2, result3, result4]])
```

**Key findings:**
- All handoffs are supervisor-mediated (no direct agent-to-agent communication)
- Context preserved across handoffs via LangGraph state management
- Workspace state shared correctly via DataManagerV2

### Integration Test Organization (Phase 7 Structure)

**File structure:**
```
tests/integration/
├── test_research_agent_real_api.py          # 10 research_agent tools
│   ├── TestLiteratureSearch                  # search_literature (4 tests)
│   ├── TestRelatedEntries                    # find_related_entries (4 tests)
│   ├── TestDatasetSearch                     # fast_dataset_search (4 tests)
│   ├── TestDatasetMetadata                   # get_dataset_metadata (4 tests)
│   ├── TestMetadataValidation                # validate_dataset_metadata (4 tests)
│   ├── TestMethodsExtraction                 # extract_methods (4 tests)
│   ├── TestAbstractSearch                    # fast_abstract_search (4 tests)
│   ├── TestFullPublication                   # read_full_publication (5 tests) ★
│   ├── TestWorkspaceWrite                    # write_to_workspace (4 tests)
│   └── TestWorkspaceRead                     # get_content_from_workspace (7 tests)
│
├── test_content_cascade_real_api.py         # 3-tier cascade system
│   ├── TestPMCFastPath                       # Tier 1: <500ms target (4 tests)
│   ├── TestWebpageFallback                   # Tier 2: 2-5s target (4 tests)
│   ├── TestPDFFallback                       # Tier 3: 3-8s target (4 tests)
│   ├── TestFullCascadeIntegration            # End-to-end (7 tests) ★
│   └── TestCascadeSystemIntegration          # System reliability (2 tests)
│
├── test_metadata_assistant_real_workflows.py # 4 metadata tools
│   ├── TestMapSamplesByID                    # 4 strategies (7 tests)
│   ├── TestReadSampleMetadata                # 3 formats (6 tests)
│   ├── TestStandardizeSampleMetadata         # Pydantic schemas (6 tests)
│   ├── TestValidateDatasetContent            # 5 checks (6 tests)
│   └── TestMetadataAssistantWorkflows        # Multi-omics (2 tests)
│
├── test_multi_agent_handoffs_real.py        # Supervisor coordination
│   ├── TestResearchToMetadataHandoff         # research→metadata (4 tests)
│   ├── TestResearchToDataExpertHandoff       # research→data (4 tests)
│   ├── TestMetadataToResearchHandback        # metadata→research (4 tests)
│   ├── TestSupervisorCoordination            # 4-agent workflow (6 tests) ★
│   └── TestMultiAgentIntegrationWorkflows    # End-to-end pipelines (2 tests)
│
└── test_performance_and_errors_real.py      # Benchmarking & error recovery
    ├── TestAbstractRetrievalPerformance      # 100 PMIDs, P95/P99 (1 test)
    ├── TestPMCRetrievalPerformance           # 20 papers, tier tracking (1 test)
    ├── TestGEOMetadataPerformance            # 20 datasets, validation (1 test)
    ├── TestRateLimitingBehavior              # Backoff, compliance (3 tests)
    ├── TestInvalidIdentifierHandling         # Error messages (3 tests)
    ├── TestGracefulDegradation               # Partial failures (3 tests)
    └── TestPerformanceAndErrorSummary        # E2E workflow (1 test)
```

**★ Critical tests** - Most important integration paths identified in Phase 7

### Test Maintenance Guidelines

1. **Update benchmark identifiers periodically**: Verify PMIDs/GEO IDs are still active
2. **Review success rate thresholds**: Adjust 95%/75% if API reliability changes
3. **Monitor test duration**: Flag tests exceeding 2x expected duration (API issues)
4. **Track flaky tests**: Use pytest-flaky plugin for transient failures
5. **Document known failures**: Maintain list of expected failures (paywalled papers)

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
CLI → LobsterClientAdapter → {AgentClient (local LangGraph), CloudLobsterClient (HTTP API)}
      ↓
Agent Registry → Agents (singlecell, bulk, proteomics, data, research, metadata_assistant, supervisor)
      ↓
DataManagerV2 (modalities, provenance, schemas, backends, workspace v2.2+, notebooks v2.3+)
      ↓
Services (stateless) → Return (processed_adata, statistics, ir)
```

**Testing:** Agent registry (`python tests/test_agent_registry.py`), integration (`pytest tests/integration/`), CLI (local+cloud clients), data flow (naming convention), notebooks (`pytest tests/unit/core/test_notebook_*.py tests/integration/test_notebook_workflow.py`)

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