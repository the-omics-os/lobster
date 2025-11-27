# Custom Feature Agent - AI-Powered Feature Generation

## Overview

The **Custom Feature Agent** is a META-AGENT that leverages the Claude Code SDK to automatically generate production-ready Lobster components through natural language descriptions. Instead of manually writing boilerplate code, developers describe their requirements and the agent creates compliant agents, services, tests, and documentation.

This guide covers the automated feature generation system. For manual agent development, see:
- [Creating Agents Guide](09-creating-agents.md) - Manual development patterns
- [Custom Agent Tutorial](26-tutorial-custom-agent.md) - Step-by-step manual tutorial

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────┐
│      Custom Feature Agent           │
│  ┌─────────────────────────────┐   │
│  │  1. Validate Requirements   │   │
│  │  2. Research Best Practices │   │
│  │  3. Check Existing Files    │   │
│  │  4. Create Git Branch       │   │
│  └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       Claude Code SDK               │
│  ┌─────────────────────────────┐   │
│  │  • Reads codebase structure │   │
│  │  • Applies unified template │   │
│  │  • Generates compliant code │   │
│  │  • Creates test files       │   │
│  │  • Writes documentation     │   │
│  └─────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Generated Components         │
│  • lobster/agents/*_expert.py       │
│  • lobster/services/{category}/*.py │
│  • tests/unit/services/{category}/* │
│  • wiki/*.md                        │
└─────────────────────────────────────┘
```

## Feature Types

The agent supports three feature types:

| Type | Description | Generated Files |
|------|-------------|-----------------|
| `agent` | Standalone specialist agent | Agent file, state class, tests, wiki |
| `service` | Stateless analysis service | Service file, tests, wiki |
| `agent_with_service` | Complete agent + service pair | All of the above |

## Services Directory Structure

Services are organized by function in `lobster/services/`:

```
lobster/services/
├── analysis/           # Statistical analysis, clustering, DE, differential
├── data_access/        # External APIs, databases, GEO, content fetching
├── data_management/    # Modality CRUD, concatenation, storage
├── metadata/           # Standardization, validation, ID mapping
├── ml/                 # Machine learning, embeddings, predictions
├── orchestration/      # Workflow coordination, pipeline execution
├── quality/            # QC, preprocessing, filtering
└── visualization/      # Plotting, charts, visualization services
```

The SDK automatically determines the appropriate category based on the feature requirements.

## Usage

### Via Natural Language (Recommended)

```bash
lobster chat
```

Then describe your feature:

```
Create a new metabolomics analysis service that:
- Loads and normalizes metabolomics data from various formats
- Performs pathway enrichment analysis
- Generates volcano plots and heatmaps
- Integrates with existing bulk RNA-seq data
```

### Via Tool Call

```python
create_new_feature(
    feature_type="agent_with_service",
    feature_name="metabolomics",
    requirements="""
    Create a metabolomics analysis agent that:
    - Supports LC-MS and GC-MS data formats
    - Performs feature detection and alignment
    - Calculates metabolite abundances
    - Integrates with pathway databases (KEGG, MetaCyc)
    """
)
```

## Generated Code Patterns

### Service Pattern (3-Tuple Return)

All generated services follow the standard pattern:

```python
class MetabolomicsService:
    """Stateless service for metabolomics analysis."""

    def analyze(
        self,
        adata: AnnData,
        **params
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform metabolomics analysis.

        Returns:
            Tuple of (processed_adata, stats_dict, ir)
        """
        # Processing logic
        processed = adata.copy()
        # ... analysis code ...

        stats = {
            "n_metabolites": processed.n_vars,
            "n_samples": processed.n_obs,
            # ... other statistics ...
        }

        ir = self._create_ir(**params)

        return processed, stats, ir

    def _create_ir(self, **params) -> AnalysisStep:
        """Create W3C-PROV compliant intermediate representation."""
        return AnalysisStep(
            operation="metabolomics.analyze",
            tool_name="analyze",
            description="Metabolomics analysis",
            library="lobster",
            code_template="""
# Metabolomics analysis
service = MetabolomicsService()
result, stats, ir = service.analyze(adata, **{{ params }})
""",
            imports=["from lobster.services.analysis.metabolomics_service import MetabolomicsService"],
            parameters=params,
            parameter_schema={...}
        )
```

### Agent Tool Pattern

Generated agent tools follow the modality validation pattern:

```python
@tool
def analyze_metabolomics(
    modality_name: str,
    normalization: str = "median"
) -> str:
    """Perform metabolomics analysis on the specified modality."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

        adata = data_manager.get_modality(modality_name)

        # 2. Call stateless service
        result, stats, ir = service.analyze(adata, normalization=normalization)

        # 3. Store result with descriptive naming
        new_name = f"{modality_name}_metabolomics_analyzed"
        data_manager.modalities[new_name] = result

        # 4. Log with IR for provenance
        data_manager.log_tool_usage(
            "analyze_metabolomics",
            {"normalization": normalization},
            stats,
            ir=ir  # IR is mandatory
        )

        return format_response(stats, new_name)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"Analysis failed: {str(e)}"
```

### IDownloadService Pattern

For data access services that download from external sources:

```python
from lobster.core.interfaces.download_service import IDownloadService

class MyDatabaseDownloadService(IDownloadService):
    """Download service for MyDatabase."""

    def supports_database(self, database: str) -> bool:
        return database.lower() in ["mydatabase", "mydb"]

    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[Dict] = None
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """Download and process dataset from MyDatabase."""
        # Implementation
        pass

    def validate_strategy_params(
        self,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate download strategy parameters."""
        pass

    def get_supported_strategies(self) -> List[str]:
        """Return list of supported download strategies."""
        return ["default", "parallel", "streaming"]
```

### ModalityManagementService Pattern

For services that manage modalities:

```python
from lobster.services.data_management.modality_management_service import ModalityManagementService

class MyModalityService:
    """Service with modality management capabilities."""

    def __init__(self, data_manager: DataManagerV2):
        self.modality_service = ModalityManagementService(data_manager)

    def load_data(self, path: str, name: str) -> Tuple[AnnData, Dict, AnalysisStep]:
        """Load data using the standardized modality service."""
        return self.modality_service.load_modality(
            modality_name=name,
            file_path=path,
            adapter="csv",
            dataset_type="metabolomics",
            validate=True
        )
```

## Naming Conventions

### Feature Names

- **Format**: `snake_case`, lowercase
- **Pattern**: `^[a-z][a-z0-9_]*$`
- **No trailing underscores**

Valid examples:
```
spatial_transcriptomics
metabolomics_v2
gene2vec
rna_velocity
```

Invalid examples:
```
SpatialTranscriptomics  # PascalCase
spatial-transcriptomics # hyphens
10x_genomics            # starts with number
spatial_                # trailing underscore
```

### Modality Naming

Generated code follows professional modality naming:

```
geo_gse12345                      # Original data
├── geo_gse12345_quality_assessed # After QC
├── geo_gse12345_filtered         # After filtering
├── geo_gse12345_normalized       # After normalization
├── geo_gse12345_clustered        # After clustering
└── geo_gse12345_annotated        # After annotation
```

## Integration Instructions

After generation, the agent provides integration instructions:

### 1. Registry Entry

```python
# Add to lobster/config/agent_registry.py
'metabolomics_expert_agent': AgentRegistryConfig(
    name='metabolomics_expert_agent',
    display_name='Metabolomics Expert',
    description='Handles metabolomics analysis including pathway enrichment',
    factory_function='lobster.agents.metabolomics_expert.metabolomics_expert',
    handoff_tool_name='handoff_to_metabolomics_expert_agent',
    handoff_tool_description='Hand off metabolomics analysis tasks'
),
```

### 2. Agent Configuration (Optional)

```python
# Add to lobster/config/agent_config.py
"metabolomics_expert_agent": "claude-4-sonnet",
```

### 3. Git Workflow

```bash
# On feature branch (auto-created)
make test              # Run tests
make lint              # Check linting
make type-check        # Type checking

git add .
git commit -m "feat: Add metabolomics analysis service"
git push origin feature/metabolomics_241126
```

## Testing the Custom Feature Agent

### Unit Tests

The agent includes comprehensive unit tests at `tests/unit/agents/test_custom_feature_agent.py`:

```bash
# Run all custom feature agent tests
pytest tests/unit/agents/test_custom_feature_agent.py -v

# Run specific test class
pytest tests/unit/agents/test_custom_feature_agent.py::TestFeatureNameValidation -v
```

### Test Coverage

| Test Class | Coverage |
|------------|----------|
| `TestFeatureNameValidation` | Name format, reserved names, edge cases |
| `TestFeatureTypeValidation` | Valid/invalid feature types |
| `TestCheckExistingFiles` | File detection across all service categories |
| `TestCustomFeatureAgentCreation` | Agent factory function |
| `TestPackageDetection` | Import detection, stdlib categorization |
| `TestBranchCreation` | Git operations (mocked) |
| `TestSDKIntegrationMocked` | SDK configuration, prompts |
| `TestErrorHandling` | Validation errors, edge cases |

### Manual Testing

For full integration testing with real SDK calls:

1. Switch to a test branch first
2. Run the agent with a test feature
3. Verify generated files
4. Delete test branch when done

```bash
git checkout -b test/custom-feature-test
# Run agent to create test feature
# Verify generated code
git checkout main
git branch -D test/custom-feature-test
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Feature name invalid" | Name doesn't match pattern | Use `snake_case`, start with letter |
| "Requirements too short" | Less than 20 characters | Provide detailed requirements |
| "Existing files found" | Feature already exists | Choose different name or update existing |
| "SDK timeout" | Complex generation | Retry or simplify requirements |

### Debug Mode

Enable debug logging for detailed SDK output:

```python
create_new_feature(
    feature_type="service",
    feature_name="test_feature",
    requirements="...",
    debug=True  # Enable verbose output
)
```

## Best Practices

### 1. Detailed Requirements

Provide comprehensive requirements for better generation:

```
# Good
Create a spatial transcriptomics service that:
- Handles Visium and Slide-seq data formats
- Performs neighborhood analysis using squidpy
- Calculates Moran's I for spatial autocorrelation
- Generates spatial scatter plots with cluster overlays
- Supports integration with histology images

# Bad
Create a spatial service
```

### 2. Specify Dependencies

Mention required libraries:

```
Requirements should use:
- squidpy for spatial analysis
- scanpy for preprocessing
- matplotlib/plotly for visualization
```

### 3. Review Before Integration

Always review generated code before integrating:
- Verify service follows 3-tuple pattern
- Check IR (AnalysisStep) is properly created
- Confirm tests cover key functionality
- Review wiki documentation accuracy

## Limitations

1. **Not for simple edits**: Use manual editing for small changes
2. **Requires API tokens**: Claude Code SDK needs valid credentials
3. **Branch creation**: Creates git branches (ensure clean working directory)
4. **Review required**: Generated code should be reviewed before merging

## Related Documentation

- [Creating Agents Guide](09-creating-agents.md) - Manual agent development
- [Creating Services Guide](10-creating-services.md) - Service development patterns
- [Testing Guide](12-testing-guide.md) - Comprehensive testing strategies
- [Agent System Architecture](19-agent-system.md) - Technical architecture
- [Download Queue System](35-download-queue-system.md) - Queue-based downloads
