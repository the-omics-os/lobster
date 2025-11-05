# Agents API Reference

## Overview

The Agents API provides specialized AI agents for different analytical domains in bioinformatics. Each agent is designed as an expert in its specific area, offering a comprehensive set of tools for data analysis, visualization, and interpretation. All agents follow the standard tool pattern and integrate seamlessly with the DataManagerV2 system.

## Agent Registry

All agents are managed through the centralized agent registry in `lobster.config.agent_registry`.

### AgentRegistryConfig

```python
@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system."""
    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None
```

### Available Agents

```python
AGENT_REGISTRY = {
    'data_expert_agent': AgentRegistryConfig(...),
    'singlecell_expert_agent': AgentRegistryConfig(...),
    'bulk_rnaseq_expert_agent': AgentRegistryConfig(...),
    'research_agent': AgentRegistryConfig(...),
    'method_expert_agent': AgentRegistryConfig(...),
    'ms_proteomics_expert_agent': AgentRegistryConfig(...),
    'affinity_proteomics_expert_agent': AgentRegistryConfig(...),
    'machine_learning_expert_agent': AgentRegistryConfig(...)
}
```

## Agent Tool Pattern

All agent tools follow the standard pattern:

```python
@tool
def agent_tool(modality_name: str, **params) -> str:
    """
    Standard pattern for all agent tools.

    Args:
        modality_name: Name of the modality to operate on
        **params: Tool-specific parameters

    Returns:
        str: Formatted response for LLM consumption
    """
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
        data_manager.log_tool_usage("agent_tool", params, stats)

        return formatted_response(stats, new_modality)

    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Analysis failed: {str(e)}"
```

## Data Expert Agent

Handles all data fetching, downloading, and extraction operations.

### Factory Function

```python
def data_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "data_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### fetch_geo_metadata_and_strategy_config

```python
@tool
def fetch_geo_metadata_and_strategy_config(geo_id: str, data_source: str = 'geo') -> str
```

Fetch and validate GEO dataset metadata without downloading the full dataset.

**Parameters:**
- `geo_id` (str): GEO accession number (e.g., GSE12345 or GDS5826)
- `data_source` (str): Data source identifier

**Returns:**
- `str`: Formatted metadata summary with validation results and recommendation

#### download_geo_dataset

```python
@tool
def download_geo_dataset(
    geo_id: str,
    sample_limit: Optional[int] = None,
    concatenation_strategy: str = "guided"
) -> str
```

Download and process GEO dataset with guided concatenation.

**Parameters:**
- `geo_id` (str): GEO accession number
- `sample_limit` (Optional[int]): Maximum number of samples to process
- `concatenation_strategy` (str): Strategy for combining samples

**Returns:**
- `str`: Processing summary with dataset information

#### load_local_file

```python
@tool
def load_local_file(
    file_path: str,
    adapter_type: str = "auto_detect",
    modality_name: str = None
) -> str
```

Load a local file into the data management system.

**Parameters:**
- `file_path` (str): Path to the file to load
- `adapter_type` (str): Type of adapter to use for loading
- `modality_name` (str): Name to assign to the loaded modality

**Returns:**
- `str`: Loading status and modality information

#### restore_workspace_datasets

```python
@tool
def restore_workspace_datasets(pattern: str = "recent") -> str
```

Restore datasets from workspace based on pattern matching for session continuation.

**Parameters:**
- `pattern` (str): Dataset pattern to match. Options:
  - `"recent"`: Load most recently used datasets (default)
  - `"all"`: Load all available datasets
  - `"*"`: Load all datasets (same as "all")
  - `"<dataset_name>"`: Load specific dataset by name
  - `"<partial_name>*"`: Load datasets matching partial name

**Returns:**
- `str`: Summary of loaded datasets with details including shape, size, and availability

**Features:**
- Flexible pattern matching for targeted dataset loading
- Intelligent memory management and duplicate detection
- Comprehensive reporting with modality details
- Integration with provenance tracking system
- Support for session continuation workflows

**Example Usage:**
```python
# Restore recent datasets for continued analysis
restore_workspace_datasets("recent")

# Load specific dataset by name
restore_workspace_datasets("geo_gse123456")

# Load all datasets matching pattern
restore_workspace_datasets("geo_*")
```

## Single-Cell Expert Agent

Specialized in single-cell RNA-seq analysis with complete workflow support.

### Factory Function

```python
def singlecell_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "singlecell_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### check_data_status

```python
@tool
def check_data_status(modality_name: str = "") -> str
```

Check the current status of loaded data.

#### assess_data_quality

```python
@tool
def assess_data_quality(
    modality_name: str,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0
) -> str
```

Perform comprehensive quality assessment of single-cell data.

**Parameters:**
- `modality_name` (str): Name of the modality to assess
- `min_genes` (int): Minimum genes per cell
- `min_cells` (int): Minimum cells per gene
- `max_pct_mito` (float): Maximum mitochondrial percentage

#### filter_and_normalize_modality

```python
@tool
def filter_and_normalize_modality(
    modality_name: str,
    min_genes: int = 200,
    min_cells: int = 3,
    max_genes: int = 5000,
    max_pct_mito: float = 20.0,
    normalization_method: str = "log1p"
) -> str
```

Filter cells and genes, then normalize expression data.

#### detect_doublets_in_modality

```python
@tool
def detect_doublets_in_modality(
    modality_name: str,
    expected_doublet_rate: float = 0.1,
    n_neighbors: int = 15
) -> str
```

Detect potential doublets in single-cell data.

#### cluster_modality

```python
@tool
def cluster_modality(
    modality_name: str,
    resolution: float = 0.7,
    n_pcs: int = 50,
    use_rep: str = "X_pca"
) -> str
```

Perform clustering and UMAP embedding.

#### find_marker_genes_for_clusters

```python
@tool
def find_marker_genes_for_clusters(
    modality_name: str,
    cluster_column: str = "leiden",
    method: str = "wilcoxon",
    n_genes: int = 10
) -> str
```

Find marker genes for each cluster.

#### annotate_cell_types

```python
@tool
def annotate_cell_types(
    modality_name: str,
    cluster_column: str = "leiden",
    annotation_method: str = "interactive"
) -> str
```

Annotate cell types based on cluster markers.

### Visualization Tools

#### create_umap_plot

```python
@tool
def create_umap_plot(
    modality_name: str,
    color_by: str = "leiden",
    point_size: float = 1.0
) -> str
```

Create UMAP visualization colored by specified metadata.

#### create_qc_plots

```python
@tool
def create_qc_plots(modality_name: str) -> str
```

Create comprehensive quality control plots.

#### create_violin_plot

```python
@tool
def create_violin_plot(
    modality_name: str,
    genes: List[str],
    group_by: str = "leiden"
) -> str
```

Create violin plots for gene expression.

## Bulk RNA-seq Expert Agent

Specialized in bulk RNA-seq analysis with pyDESeq2 integration.

### Factory Function

```python
def bulk_rnaseq_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "bulk_rnaseq_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### analyze_differential_expression

```python
@tool
def analyze_differential_expression(
    modality_name: str,
    condition_column: str,
    reference_condition: str = None,
    formula: str = None,
    batch_column: str = None
) -> str
```

Perform differential expression analysis using pyDESeq2.

**Parameters:**
- `modality_name` (str): Name of the modality containing count data
- `condition_column` (str): Column name for the main condition
- `reference_condition` (str): Reference level for comparison
- `formula` (str): Custom formula for complex designs
- `batch_column` (str): Column name for batch effects

#### create_ma_plot

```python
@tool
def create_ma_plot(
    modality_name: str,
    comparison: str = "default",
    alpha: float = 0.05
) -> str
```

Create MA plot for differential expression results.

#### create_volcano_plot

```python
@tool
def create_volcano_plot(
    modality_name: str,
    comparison: str = "default",
    alpha: float = 0.05,
    log2fc_threshold: float = 1.0
) -> str
```

Create volcano plot for differential expression results.

## Mass Spectrometry Proteomics Expert

Specialized in MS proteomics analysis including DDA/DIA workflows.

### Factory Function

```python
def ms_proteomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "ms_proteomics_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### assess_proteomics_quality

```python
@tool
def assess_proteomics_quality(
    modality_name: str,
    missing_value_threshold: float = 0.7
) -> str
```

Perform quality assessment specific to MS proteomics data.

#### handle_missing_values_proteomics

```python
@tool
def handle_missing_values_proteomics(
    modality_name: str,
    method: str = "imputation",
    imputation_method: str = "knn"
) -> str
```

Handle missing values in proteomics data using appropriate strategies.

#### normalize_proteomics_data

```python
@tool
def normalize_proteomics_data(
    modality_name: str,
    method: str = "tmm",
    log_transform: bool = True
) -> str
```

Normalize proteomics intensity data.

## Affinity Proteomics Expert

Specialized in affinity proteomics including Olink panels and antibody arrays.

### Factory Function

```python
def affinity_proteomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "affinity_proteomics_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### analyze_affinity_proteomics

```python
@tool
def analyze_affinity_proteomics(
    modality_name: str,
    panel_type: str = "olink",
    qc_threshold: float = 0.8
) -> str
```

Analyze affinity proteomics data with panel-specific processing.

#### validate_antibody_performance

```python
@tool
def validate_antibody_performance(
    modality_name: str,
    cv_threshold: float = 0.3
) -> str
```

Validate antibody performance using CV analysis.

## Research Agent

Handles literature discovery, dataset identification, and **automatic PMID/DOI → PDF resolution** for computational method extraction with **structure-aware Docling parsing** (v2.3+).

**Phase 1 Enhancement (v2.2+)**: Automatic resolution of PMIDs and DOIs to accessible PDF URLs using tiered waterfall strategy (PMC → bioRxiv/medRxiv → Publisher → Suggestions). Achieves 70-80% automatic resolution success rate with graceful fallback to alternative access strategies for paywalled papers.

**Phase 2 & 3 Enhancement (v2.3+)**: Structure-aware PDF parsing with Docling replaces naive PyPDF2 truncation. Intelligent Methods section detection achieves >90% hit rate (vs ~30% previously), extracts parameter tables and formulas, and includes comprehensive retry logic with automatic fallback. See [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md) for technical details.

### Factory Function

```python
def research_agent(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "research_agent",
    handoff_tools: List = None
)
```

### Core Components

**ResearchAgentAssistant**: Helper class providing PDF resolution methods, batch processing, and formatted reporting. Uses `PublicationResolver` for tiered waterfall resolution strategy.

### Tools

#### search_literature

```python
@tool
def search_literature(
    query: str,
    max_results: int = 10,
    publication_year: int = None
) -> str
```

Search scientific literature using PubMed.

#### find_datasets_for_publication

```python
@tool
def find_datasets_for_publication(
    pmid: str,
    dataset_types: List[str] = None
) -> str
```

Find associated datasets for a publication.

#### search_geo_datasets

```python
@tool
def search_geo_datasets(
    query: str,
    organism: str = "Homo sapiens",
    study_type: str = "Expression profiling by high throughput sequencing"
) -> str
```

Search GEO database for relevant datasets.

#### extract_paper_methods ✨ (v2.3+ Enhanced with Docling)

```python
@tool
def extract_paper_methods(url_or_pmid: str) -> str
```

Extract computational analysis methods from a research paper using **structure-aware Docling PDF parsing** (v2.3+). **Automatically resolves PMIDs and DOIs to PDF URLs** using tiered waterfall strategy:
1. PubMed Central (PMC) - Free full text
2. bioRxiv/medRxiv - Preprint servers
3. Publisher Direct - Open access detection
4. Generate alternative access suggestions if paywalled

**Input formats (v2.3+ Enhanced)**: All identifier types auto-detected and resolved:
- **Bare DOI:** `"10.1101/2024.08.29.610467"` (NEW - auto-detected)
- **DOI with prefix:** `"DOI:10.1038/s41586-021-12345-6"`
- **PMID:** `"PMID:39370688"` or `"39370688"` (both formats supported)
- **Direct URL:** `"https://www.nature.com/articles/..."` (passed through)
- **PMC URL:** `"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC..."` (format auto-detected)

**Returns**: Structured JSON with:
- Software packages, parameters, quality control steps, and analysis workflows
- **v2.3+ enhancements**: Parameter tables (pandas DataFrames), mathematical formulas (LaTeX), auto-detected tools, extraction metadata

**Extraction Quality (v2.3+ Enhanced)**:
- Methods section detection: >90% hit rate (vs ~30% with PyPDF2)
- Complete section extraction (no arbitrary truncation)
- Table and formula preservation
- Smart image filtering (40-60% context size reduction)
- Document caching (2-5s first parse → <100ms cached)
- **Robust DOI resolution:** No more FileNotFoundError crashes for valid DOIs
- **Format auto-detection:** PMC HTML content correctly identified (not misclassified as PDF)

**See also**: [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md) for comprehensive Docling integration details.

#### resolve_paper_access ✨ (Phase 1 New)

```python
@tool
def resolve_paper_access(identifier: str) -> str
```

Diagnostic tool to check paper accessibility before extraction. Returns resolution result with PDF URL (if accessible) or alternative access strategies (if paywalled). Useful for competitive analysis workflows where accessibility needs to be checked first.

**Resolution sources**: PMC, bioRxiv, medRxiv, publisher direct access, institutional repositories.

#### extract_methods_batch ✨ (Phase 1 New)

```python
@tool
def extract_methods_batch(
    identifiers: str,  # Comma-separated
    max_papers: int = 5
) -> str
```

Extract computational methods from multiple papers in batch (2-5 papers recommended). Conservative sequential processing to avoid API rate limits. Returns aggregated report with accessible/paywalled/failed breakdown.

**Use case**: Competitive analysis of methods across multiple publications.

## ~~Method Expert Agent~~ (DEPRECATED v2.2+)

**Deprecated:** Method Expert functionality has been merged into Research Agent with Phase 1 enhancements. The Research Agent now handles all method extraction with automatic PMID/DOI → PDF resolution (70-80% success rate).

### Replacement

For method extraction, use **Research Agent** tools:
- `extract_paper_methods` - Extract methods with auto-resolution (accepts PMIDs, DOIs, URLs)
- `extract_methods_batch` - Batch method extraction (2-5 papers)
- `resolve_paper_access` - Check PDF accessibility before extraction

See [Research Agent](#research-agent) section for complete documentation.

### Migration Guide

**Old workflow (deprecated):**
```python
# ❌ No longer works
handoff_to_method_expert_agent("Extract methods from PMID:12345678")
```

**New workflow (Phase 1):**
```python
# ✅ Use research_agent instead
handoff_to_research_agent("Extract methods from PMID:12345678")
# Auto-resolves PMID → PDF → extracts methods
```

## Machine Learning Expert Agent

Handles ML transformations and model preparation.

### Factory Function

```python
def machine_learning_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "machine_learning_expert_agent",
    handoff_tools: List = None
)
```

### Tools

#### check_ml_readiness

```python
@tool
def check_ml_readiness(modality_name: str) -> str
```

Check if data is ready for ML workflows.

#### prepare_ml_features

```python
@tool
def prepare_ml_features(
    modality_name: str,
    feature_selection_method: str = "variance",
    n_features: int = 2000
) -> str
```

Prepare ML-ready feature matrices.

#### create_train_test_splits

```python
@tool
def create_train_test_splits(
    modality_name: str,
    target_column: str,
    test_size: float = 0.2,
    stratify: bool = True
) -> str
```

Create stratified train/test splits for ML.

## Agent Configuration and Model Management

### Agent Model Configuration

Each agent can be configured with specific LLM parameters:

```python
@dataclass
class AgentModelConfig:
    """Model configuration for a specific agent."""
    name: str
    model_config: ModelConfig
    fallback_model: Optional[str] = None
    enabled: bool = True
    custom_params: Dict = field(default_factory=dict)
    thinking_config: Optional[ThinkingConfig] = None
```

### Example Agent Configuration

```python
# Configure single-cell expert with specific model
singlecell_config = AgentModelConfig(
    name="singlecell_expert_agent",
    model_config=ModelConfig(
        provider=ModelProvider.BEDROCK_ANTHROPIC,
        model_id="us.anthropic.claude-3-sonnet-20240229-v1:0",
        tier=ModelTier.STANDARD,
        temperature=0.7
    ),
    thinking_config=ThinkingConfig(enabled=True, budget_tokens=2000)
)
```

## Error Handling in Agents

All agents implement consistent error handling:

### Exception Types

- `ModalityNotFoundError`: When requested modality doesn't exist
- `ServiceError`: When underlying service operations fail
- `ValidationError`: When input validation fails
- `ProcessingError`: When data processing operations fail

### Error Response Format

```python
def handle_error(error: Exception, tool_name: str) -> str:
    """Standard error handling for agent tools."""
    error_message = f"{tool_name} failed: {str(error)}"
    logger.error(error_message, exc_info=True)

    return json.dumps({
        "status": "error",
        "error_type": type(error).__name__,
        "message": error_message,
        "tool": tool_name,
        "timestamp": datetime.now().isoformat()
    })
```

## Agent Handoff Mechanism

Agents can hand off tasks to other specialized agents:

### Handoff Tools

Each agent automatically gets handoff tools to other agents based on the registry:

```python
# Auto-generated handoff tools
@tool
def handoff_to_singlecell_expert(
    task_description: str,
    modality_name: str = None
) -> str:
    """Assign single-cell RNA-seq analysis tasks to the single-cell expert"""
```

### Usage Example

```python
# In data expert agent
@tool
def complex_analysis_task(geo_id: str) -> str:
    # Download data
    dataset = download_geo_dataset(geo_id)

    # Hand off to appropriate expert
    if is_single_cell_data(dataset):
        return handoff_to_singlecell_expert(
            "Perform complete single-cell analysis workflow",
            modality_name=dataset
        )
    else:
        return handoff_to_bulk_rnaseq_expert(
            "Perform differential expression analysis",
            modality_name=dataset
        )
```

## Integration with DataManagerV2

All agents seamlessly integrate with DataManagerV2:

- **Modality Management**: Agents read from and write to DataManagerV2 modalities
- **Provenance Tracking**: All agent operations are logged for reproducibility
- **Plot Management**: Visualization tools automatically store plots in DataManagerV2
- **Quality Metrics**: Agents can access quality metrics from the data manager

## Supervisor Configuration API (v2.3+)

The supervisor agent now features dynamic configuration and automatic agent discovery.

### SupervisorConfig

```python
from lobster.config.supervisor_config import SupervisorConfig

@dataclass
class SupervisorConfig:
    """Configuration for supervisor agent behavior."""

    # Interaction settings
    ask_clarification_questions: bool = True
    max_clarification_questions: int = 3
    require_download_confirmation: bool = True
    require_metadata_preview: bool = True

    # Response settings
    auto_suggest_next_steps: bool = True
    verbose_delegation: bool = False
    include_expert_output: bool = True
    summarize_expert_output: bool = False

    # Context inclusion
    include_data_context: bool = True
    include_workspace_status: bool = True
    include_system_info: bool = False
    include_memory_stats: bool = False

    # Workflow settings
    workflow_guidance_level: str = "standard"  # minimal, standard, detailed
    show_available_tools: bool = True
    show_agent_capabilities: bool = True

    # Advanced settings
    delegation_strategy: str = "auto"  # auto, conservative, aggressive
    error_handling: str = "informative"  # silent, informative, verbose
```

### Dynamic Prompt Generation

```python
from lobster.agents.supervisor import create_supervisor_prompt

# Create prompt with custom configuration
config = SupervisorConfig(
    ask_clarification_questions=False,
    workflow_guidance_level='minimal'
)

prompt = create_supervisor_prompt(
    data_manager=data_manager,
    config=config,
    active_agents=['data_expert_agent', 'singlecell_expert_agent']
)
```

### Agent Capability Extraction

```python
from lobster.config.agent_capabilities import AgentCapabilityExtractor

# Extract capabilities for an agent
capabilities = AgentCapabilityExtractor.extract_capabilities('singlecell_expert_agent')

# Get all agent capabilities
all_capabilities = AgentCapabilityExtractor.get_all_agent_capabilities()

# Get formatted summary
summary = AgentCapabilityExtractor.get_agent_capability_summary(
    'singlecell_expert_agent',
    max_tools=5
)
```

### Graph Integration

```python
from lobster.agents.graph import create_bioinformatics_graph
from lobster.config.supervisor_config import SupervisorConfig

# Create graph with custom supervisor configuration
supervisor_config = SupervisorConfig(
    ask_clarification_questions=True,
    max_clarification_questions=5,
    workflow_guidance_level='detailed'
)

graph = create_bioinformatics_graph(
    data_manager=data_manager,
    supervisor_config=supervisor_config
)
```

## Usage Examples

### Basic Agent Usage

```python
from lobster.core.client import AgentClient

# Create client with agent system
client = AgentClient()

# Query routed to appropriate agent
result = client.query("Load GSE194247 and perform single-cell analysis")
```

### Direct Agent Tool Usage

```python
from lobster.agents.singlecell_expert import singlecell_expert
from lobster.core.data_manager_v2 import DataManagerV2

# Create agent
data_manager = DataManagerV2()
sc_agent = singlecell_expert(data_manager)

# Get tools
tools = sc_agent.get_tools()

# Use specific tool
result = tools['assess_data_quality'].invoke({
    'modality_name': 'my_dataset',
    'min_genes': 200
})
```

This agents API provides a comprehensive set of specialized tools for bioinformatics analysis, with each agent focusing on its domain expertise while maintaining consistency through the standard tool pattern and integration with the core DataManagerV2 system. The v2.3+ supervisor configuration system enables dynamic agent discovery and customizable interaction modes.