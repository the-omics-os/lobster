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
| **`lobster/agents/`** | singlecell_expert, bulk_rnaseq_expert, ms_proteomics_expert, affinity_proteomics_expert, data_expert, research_agent, supervisor | Specialized AI agents for analysis domains + coordination |
| **`lobster/core/`** | client, api_client, data_manager_v2, provenance, notebook_exporter/executor, schemas/ | Client infrastructure, multi-omics orchestration, W3C-PROV tracking, Jupyter export (v2.3+) |
| **`lobster/tools/`** | **Transcriptomics:** preprocessing, quality, clustering, enhanced_singlecell, bulk_rnaseq, pseudobulk, differential_formula, concatenation<br>**Proteomics:** preprocessing, quality, analysis, differential, visualization<br>**Data:** geo, publication, unified_content, docling, metadata_validation, providers/, visualization | Stateless analysis services: QC, normalization, DE, clustering, literature mining, GEO datasets |
| **`lobster/config/`** | agent_config, **agent_registry** (centralized single source of truth), settings | Configuration management, agent registry with auto handoff generation |

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
- Named biological datasets (`Dict[str, AnnData]`), metadata store (GEO/source), W3C-PROV provenance tracking
- Backend/adapter registry, schema validation (transcriptomics/proteomics)
- **Workspace restoration** (v2.2+): Session persistence, lazy loading, pattern-based restoration

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
CLI → LobsterClientAdapter → {AgentClient (local LangGraph), CloudLobsterClient (HTTP API)}
      ↓
Agent Registry → Agents (singlecell, bulk, proteomics, data, research, supervisor)
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