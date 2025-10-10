# Custom Feature Agent

## Overview

The Custom Feature Agent is a powerful tool for extending Lobster with new capabilities. It uses the Claude Code SDK to automatically generate new agents, services, tools, tests, and documentation that follow Lobster's architectural patterns.

When you need analysis capabilities that Lobster doesn't currently support, the Custom Feature Agent can create production-ready code for you, saving hours of development time and ensuring consistency with Lobster's design principles.

## When to Use

Use the Custom Feature Agent when you need to:

- **Add new analysis types**: Spatial transcriptomics, metabolomics, variant calling, etc.
- **Create custom workflows**: Specialized pipelines for your specific research needs
- **Extend existing capabilities**: Add new tools to existing agents
- **Integrate new methods**: Implement new algorithms or statistical approaches
- **Build reusable services**: Create stateless analysis functions for multiple agents

**Examples:**
- "Create an agent for spatial transcriptomics analysis"
- "Build a metabolomics analysis service"
- "Add variant calling capabilities to Lobster"
- "Create a chromatin accessibility analysis pipeline"

## How It Works

The Custom Feature Agent follows this workflow:

```
1. You request a new feature through the supervisor
2. Supervisor hands off to Custom Feature Agent
3. Custom Feature Agent:
   - Validates your feature name and requirements
   - Spawns Claude Code SDK with architectural context
   - Claude Code creates files following Lobster patterns
   - Verifies all files were created successfully
4. Custom Feature Agent reports back with:
   - List of created files
   - Instructions for registry update
   - Next steps for testing and integration
5. You complete manual steps and test your new feature
```

## Available Tools

The Custom Feature Agent provides four main tools:

### `create_new_feature`

**Description**: Main tool for creating new Lobster features

**Parameters**:
- `feature_type` (str): Type of feature to create
  - `"agent"`: Agent with tools and workflows
  - `"service"`: Reusable analysis service
  - `"agent_with_service"`: Complete feature (recommended)
- `feature_name` (str): Name for the feature (lowercase, underscores)
- `requirements` (str): Detailed specifications for the feature

**Example**:
```python
create_new_feature(
    feature_type="agent_with_service",
    feature_name="spatial_transcriptomics",
    requirements="""
    Create an agent for spatial transcriptomics analysis.

    Data types: Visium, Slide-seq, MERFISH

    Tools needed:
    - Load spatial data (H5AD with spatial coordinates)
    - Quality control for spatial data
    - Spatial clustering (graph-based)
    - Neighborhood analysis
    - Spatial gene expression patterns
    - Spatial statistics (Moran's I, Geary's C)

    Service methods:
    - process_spatial_data(adata, method="visium")
    - spatial_clustering(adata, resolution=1.0)
    - neighborhood_enrichment(adata, groups_col="cell_type")
    - spatial_autocorrelation(adata, genes=None)

    Use squidpy for spatial-specific operations.
    """
)
```

### `validate_feature_name_tool`

**Description**: Check if a feature name is valid and available

**Parameters**:
- `feature_name` (str): Name to validate

**Example**:
```python
validate_feature_name_tool("spatial_transcriptomics")
# Returns: ✅ Valid and available, or ❌ Invalid with explanation
```

### `list_existing_patterns`

**Description**: Show existing agents and services as examples

**Example**:
```python
list_existing_patterns()
# Returns: List of agents and services with file paths
```

### `create_feature_summary`

**Description**: Generate summary of features created in current session

**Example**:
```python
create_feature_summary()
# Returns: Summary with files created and next steps
```

## Feature Types

### Agent

Creates an agent with:
- Agent file (`lobster/agents/{name}_expert.py`)
- State class (added to `lobster/agents/state.py`)
- Unit tests (`tests/unit/agents/test_{name}_expert.py`)
- Wiki documentation (`lobster/wiki/{Name}.md`)

**When to use**: When you need a collection of analysis tools and workflows

### Service

Creates a service with:
- Service file (`lobster/tools/{name}_service.py`)
- Unit tests (`tests/unit/tools/test_{name}_service.py`)
- Wiki documentation (`lobster/wiki/{Name}.md`)

**When to use**: When you need reusable, stateless analysis functions

### Agent with Service (Recommended)

Creates both agent and service with:
- All files from both types above
- Integrated workflow between agent and service

**When to use**: For complete features with both interface and implementation

## Usage Workflows

### Basic Workflow: Create a New Agent

1. **Request through supervisor**:
   ```
   User: "I need to analyze spatial transcriptomics data"
   Supervisor: [Hands off to Custom Feature Agent]
   ```

2. **Agent validates and creates**:
   - Validates feature name
   - Spawns Claude Code SDK
   - Creates all necessary files

3. **Complete manual steps**:
   ```bash
   # 1. Add to agent registry (instructions provided)
   # 2. Run tests
   make test

   # 3. Check code quality
   make lint
   make type-check

   # 4. Restart Lobster
   lobster chat
   ```

4. **Test your new agent**:
   ```
   User: "Analyze my spatial data with the new agent"
   ```

### Advanced Workflow: Create Service Only

```
User: "Create a service for processing ATAC-seq data"
Supervisor: [Hands off to Custom Feature Agent]
Custom Feature Agent:
  - Creates service with peak calling methods
  - Creates comprehensive tests
  - Provides integration examples
User: Uses service in existing agents
```

### Validation Workflow

```
User: "Is 'metabolomics_analysis' a valid feature name?"
Supervisor: [Hands off to Custom Feature Agent]
Custom Feature Agent:
  - Validates name syntax
  - Checks for conflicts
  - Returns: ✅ Valid and available
```

## Feature Naming Rules

### Valid Names

✅ **Correct**:
- `spatial_transcriptomics`
- `metabolomics_analysis`
- `variant_calling`
- `chromatin_accessibility`
- `protein_interaction_network`

**Rules**:
- Start with lowercase letter
- Use lowercase letters, numbers, and underscores only
- Descriptive and specific
- No trailing underscores

### Invalid Names

❌ **Incorrect**:
- `SpatialTranscriptomics` (uppercase)
- `spatial-transcriptomics` (hyphens)
- `spatial_transcriptomics_` (trailing underscore)
- `123_analysis` (starts with number)
- `supervisor` (reserved name)
- `data` (reserved name)
- `analysis` (too generic)

**Reserved names**: supervisor, data, base, core, config

## Requirements Guidelines

### Good Requirements Include:

1. **Purpose and Scope**
   - What the feature does
   - What problems it solves

2. **Data Types**
   - Input formats (H5AD, CSV, etc.)
   - Expected data structure

3. **Analysis Methods**
   - Specific algorithms to implement
   - Statistical approaches

4. **Tools/Functions**
   - List of tools needed
   - Function signatures if known

5. **Parameters**
   - Important parameters and defaults
   - Configuration options

6. **Use Cases**
   - Example workflows
   - Expected inputs and outputs

### Example: Good Requirements

```
Create a variant calling agent for analyzing genomic sequencing data.

Data types:
- BAM/SAM alignment files
- VCF variant call files
- Reference genomes (FASTA)

Tools needed:
- load_alignment(bam_path): Load BAM files into AnnData-like structure
- call_variants(alignments, reference): Call variants with quality filtering
- annotate_variants(vcf_data): Add gene and functional annotations
- filter_variants(vcf_data, quality_threshold=30): Quality-based filtering
- visualize_variants(vcf_data): Create Manhattan and QQ plots

Service methods:
- process_alignment(bam_data, min_coverage=10, min_quality=20)
- variant_calling(bam_data, reference_path, caller="bcftools")
- variant_annotation(vcf_data, db="dbSNP")
- variant_quality_control(vcf_data)

Use pysam for BAM/SAM handling and cyvcf2 for VCF processing.
Support both germline and somatic variant calling.
```

### Example: Poor Requirements

```
Create something for genomics.
```
❌ Too vague - no specifics about what, how, or why

## Output Files

The Custom Feature Agent creates the following files:

### For Agent Type:

```
lobster/agents/{name}_expert.py          # Main agent implementation
lobster/agents/state.py                  # State class added
tests/unit/agents/test_{name}_expert.py  # Agent tests
lobster/wiki/{Name}.md                   # User documentation
```

### For Service Type:

```
lobster/tools/{name}_service.py          # Service implementation
tests/unit/tools/test_{name}_service.py  # Service tests
lobster/wiki/{Name}.md                   # User documentation
```

### For Agent with Service:

```
All files from both types above
```

## Manual Steps After Creation

After the Custom Feature Agent creates your files, you need to complete these steps:

### 1. Update Agent Registry (Agents Only)

**File**: `lobster/config/agent_registry.py`

**Action**: Add entry to `AGENT_REGISTRY` dictionary

**Example** (provided by agent):
```python
'spatial_transcriptomics': AgentRegistryConfig(
    name='spatial_transcriptomics',
    display_name='Spatial Transcriptomics Expert',
    description='Analyzes spatial transcriptomics data from Visium and Slide-seq',
    factory_function='lobster.agents.spatial_transcriptomics_expert.spatial_transcriptomics_expert',
    handoff_tool_name='handoff_to_spatial_transcriptomics',
    handoff_tool_description='Hand off spatial transcriptomics analysis tasks'
)
```

### 2. Run Tests

```bash
# Run all tests
make test

# Run only your new tests
pytest tests/unit/agents/test_{name}_expert.py -v
pytest tests/unit/tools/test_{name}_service.py -v
```

### 3. Check Code Quality

```bash
# Linting
make lint

# Type checking
make type-check

# Formatting (auto-fix)
make format
```

### 4. Restart Lobster

```bash
lobster chat
```

### 5. Test Integration

```
User: "Analyze my spatial transcriptomics data"
Supervisor: [Should now recognize and use your new agent]
```

## Troubleshooting

### Error: "Claude Agent SDK not installed"

**Cause**: The claude-agent-sdk package is not installed

**Solution**:
```bash
pip install claude-agent-sdk
```

### Error: "Invalid feature name"

**Cause**: Feature name doesn't follow naming conventions

**Solution**: Use `validate_feature_name_tool` to check your name and see suggestions

### Error: "Feature already exists"

**Cause**: Files for this feature name already exist

**Solutions**:
1. Choose a different, more specific name
2. Manually remove existing files (be careful!)
3. Add a suffix like `_advanced` or `_v2`

### Error: "Requirements too brief"

**Cause**: Insufficient detail in requirements

**Solution**: Expand requirements to include:
- Data types and formats
- Specific tools and methods needed
- Parameters and options
- Use cases and examples

### Warning: "Tests failing after creation"

**Possible Causes**:
- Missing dependencies
- Import errors
- Incorrect patterns used

**Solutions**:
1. Review generated code for obvious issues
2. Check imports and dependencies
3. Run tests with verbose output: `pytest -v`
4. Fix minor issues manually
5. Report pattern issues to Lobster team

### Issue: "Registry update not working"

**Cause**: Incorrect registry syntax or placement

**Solution**:
1. Double-check the provided code snippet
2. Ensure it's inside the `AGENT_REGISTRY` dictionary
3. Verify commas and brackets are correct
4. Restart Lobster after changes

## Best Practices

### 1. Start with Validation

Always validate your feature name before requesting creation:
```
User: "Check if 'metabolomics' is a valid feature name"
```

### 2. Provide Detailed Requirements

Spend time crafting comprehensive requirements. Better requirements = better generated code.

### 3. Use Agent with Service for Complete Features

The `agent_with_service` type creates the most complete implementation with proper separation of concerns.

### 4. Review Generated Code

Always review the generated files before using in production:
- Check that logic matches your requirements
- Verify docstrings are complete
- Ensure tests cover key scenarios
- Confirm naming conventions followed

### 5. Test Thoroughly

Run tests multiple times with different data:
- Unit tests (generated)
- Integration tests (manual)
- Real data tests
- Edge cases

### 6. Iterate if Needed

If the first attempt doesn't meet all needs:
1. Use a different name (e.g., `{name}_v2`)
2. Provide more detailed requirements
3. Manually refine generated code

### 7. Contribute Back

If you create a broadly useful feature:
1. Clean up and document thoroughly
2. Share with Lobster community
3. Consider contributing to core Lobster

## Examples

### Example 1: Spatial Transcriptomics Agent

**Request**:
```
User: "Create a complete agent for analyzing spatial transcriptomics data from Visium experiments"

Requirements:
- Support Visium and Slide-seq data
- Include spatial clustering and neighborhood analysis
- Provide spatial statistics (Moran's I)
- Create spatial visualization tools
- Handle H5AD format with spatial coordinates
```

**Result**:
- Complete agent with 5-7 tools
- Stateless service with 4-5 methods
- Comprehensive test suite
- User documentation

### Example 2: Metabolomics Service

**Request**:
```
User: "Create a service for processing and analyzing metabolomics data"

Requirements:
- Parse common metabolomics formats (mzML, mzXML)
- Peak detection and alignment
- Normalization methods (PQN, VSN)
- Statistical testing for metabolite differences
- Pathway enrichment using KEGG
```

**Result**:
- Stateless service with all methods
- Unit tests for each method
- Integration examples

### Example 3: Variant Calling Agent

**Request**:
```
User: "Build an agent for genomic variant calling from NGS data"

Requirements:
- Process BAM/SAM alignment files
- Call variants using bcftools/GATK
- Annotate variants with dbSNP, ClinVar
- Quality filtering and statistics
- Support both germline and somatic calling
```

**Result**:
- Agent with variant calling pipeline
- Service for core algorithms
- Tests with sample genomic data

## Technical Details

### Architecture

The Custom Feature Agent uses:
- **Claude Code SDK**: For code generation
- **CLAUDE.md**: Architectural patterns and examples
- **Agent Registry**: For automatic agent discovery
- **DataManagerV2**: For data orchestration patterns
- **Stateless Services**: For reusable logic

### Generated Code Quality

Features created by this agent follow:
- ✅ Agent pattern (factory, tools, system prompt)
- ✅ Service pattern (stateless, returns tuples)
- ✅ Tool pattern (validate → service → store → log)
- ✅ Naming conventions (descriptive, consistent)
- ✅ Error handling (specific exceptions)
- ✅ Type hints (comprehensive)
- ✅ Docstrings (Google-style)
- ✅ Testing (unit tests with mocks)
- ✅ Documentation (wiki pages)

### Dependencies

The agent requires:
- `claude-agent-sdk` (for code generation)
- Standard Lobster dependencies

To check SDK availability:
```bash
python -c "import claude_agent_sdk; print('SDK available')"
```

### Limitations

Current limitations:
- No automatic registry registration (manual step required)
- No automatic dependency installation
- No iterative error fixing (simple version)
- No cross-file refactoring
- No integration test generation

Future enhancements:
- Auto-registry registration
- Dependency detection and installation
- Iterative refinement based on test results
- Integration test generation
- Cross-agent refactoring

## See Also

- [Machine Learning Expert](Machine-Learning-Expert.md) - Example of agent with service
- [Single-Cell Expert](Single-Cell-Expert.md) - Example of complex agent
- [Data Expert](Data-Expert.md) - Example of data handling patterns
- [Agent Registry](../config/agent_registry.py) - Where to add new agents
- [CLAUDE.md](../agents/CLAUDE.md) - Full architectural patterns

## Getting Help

If you encounter issues:

1. **Check validation**: Use `validate_feature_name_tool`
2. **Review requirements**: Ensure they're detailed enough
3. **Check examples**: Use `list_existing_patterns`
4. **Read documentation**: Review CLAUDE.md for patterns
5. **Run tests**: See specific error messages
6. **Ask supervisor**: "Help me debug my custom feature"

For bugs or feature requests, contact the Lobster development team.
