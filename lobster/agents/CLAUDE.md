# Lobster Agent & Service Development Guide

This guide is specifically for Claude Code SDK when creating new agents, services, tools, tests, and documentation for the Lobster bioinformatics platform. Follow these patterns **exactly** to ensure compatibility with Lobster's architecture.

## Table of Contents

1. [Lobster Architecture Overview](#lobster-architecture-overview)
2. [Agent Pattern](#agent-pattern)
3. [Service Pattern](#service-pattern)
4. [Tool Pattern](#tool-pattern)
5. [File Templates](#file-templates)
6. [Naming Conventions](#naming-conventions)
7. [Testing Requirements](#testing-requirements)
8. [Documentation Standards](#documentation-standards)
9. [Complete Examples](#complete-examples)

---

## Lobster Architecture Overview

### Multi-Agent System Design

Lobster is a **professional multi-agent bioinformatics analysis platform** with:
- Specialized AI agents for different analysis domains (single-cell, bulk RNA-seq, proteomics, etc.)
- **DataManagerV2**: Central multi-modal data orchestrator
- **Stateless services**: Reusable analysis functions
- **LangGraph-based coordination**: Agent orchestration via centralized registry
- **Professional CLI**: Natural language interface with enhanced autocomplete

### Key Components

```
lobster/
├── agents/              # Specialized AI agents
│   ├── supervisor.py    # Agent coordination
│   ├── singlecell_expert.py
│   ├── bulk_rnaseq_expert.py
│   ├── machine_learning_expert.py
│   └── [your_new_agent].py
├── tools/               # Stateless analysis services
│   ├── preprocessing_service.py
│   ├── quality_service.py
│   └── [your_new_service].py
├── core/                # Data management & client infrastructure
│   ├── data_manager_v2.py
│   └── schemas/
├── config/              # Configuration management
│   ├── agent_registry.py  # Single source of truth for agents
│   └── settings.py
└── utils/               # Utilities
    └── logger.py
```

### DataManagerV2 Core Concepts

**DataManagerV2** is the central data orchestrator that manages:
- **Named biological datasets** (`Dict[str, AnnData]`) called "modalities"
- **Metadata store** for GEO and source metadata
- **Tool usage history** for provenance tracking (W3C-PROV compliant)
- **Backend/adapter registry** for extensible data handling
- **Schema validation** for transcriptomics and proteomics data

**Key Methods:**
```python
data_manager.list_modalities() -> List[str]
data_manager.get_modality(name: str) -> AnnData
data_manager.modalities[name] = adata  # Store new modality
data_manager.save_modality(name: str, path: str) -> None
data_manager.log_tool_usage(tool_name, parameters, description) -> None
```

### Professional Naming Convention

Lobster uses descriptive naming for data transformations:

```
geo_gse12345                          # Raw downloaded data
├── geo_gse12345_quality_assessed     # QC metrics added
├── geo_gse12345_filtered_normalized  # Preprocessed data
├── geo_gse12345_clustered           # Leiden clustering + UMAP
├── geo_gse12345_markers              # Differential expression
├── geo_gse12345_annotated           # Cell type annotations
└── geo_gse12345_pseudobulk          # Aggregated for DE analysis
```

**Pattern**: `{base_name}_{transformation}`

---

## Agent Pattern

### Agent Factory Function Structure

Every agent is created via a factory function that returns a LangGraph agent:

```python
def agent_name_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "agent_name_expert_agent",
    handoff_tools: List = None,
):
    """Create agent_name expert agent using DataManagerV2."""

    # 1. Get settings and create LLM
    settings = get_settings()
    model_params = settings.get_agent_llm_params("agent_name_expert_agent")
    llm = create_llm("agent_name_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # 2. Initialize agent-specific results storage
    agent_results = {"summary": "", "details": {}}

    # 3. Define tools using @tool decorator
    @tool
    def example_tool(modality_name: str, parameter: str = "default") -> str:
        """Tool description for the LLM."""
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Call stateless service
            result_adata, statistics = service.process(adata, parameter=parameter)

            # Store result with descriptive name
            new_modality_name = f"{modality_name}_{transformation}"
            data_manager.modalities[new_modality_name] = result_adata

            # Log operation for provenance
            data_manager.log_tool_usage(
                tool_name="example_tool",
                parameters={"modality_name": modality_name, "parameter": parameter},
                description=f"Applied {transformation} to {modality_name}"
            )

            # Return formatted response
            return f"Successfully processed '{modality_name}'!\n\n{statistics}"

        except Exception as e:
            logger.error(f"Error in example_tool: {e}")
            return f"Error: {str(e)}"

    # 4. Register tools
    base_tools = [example_tool, another_tool, summary_tool]
    tools = base_tools + (handoff_tools or [])

    # 5. Create system prompt
    system_prompt = f"""
You are an expert [domain] specialist for the Lobster bioinformatics platform.

<Role>
[Detailed role description]

**CRITICAL: You ONLY perform tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive tasks from the supervisor
- You execute the requested analysis
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You perform [domain] analysis following best practices:
1. [Task description]
2. [Task description]
3. [Task description]
</Task>

<Available Tools>
- `tool_name`: Description of what it does
- `tool_name`: Description of what it does
</Available Tools>

<Professional Workflows & Tool Usage Order>

## 1. WORKFLOW NAME (Supervisor Request: "...")

### Basic Workflow

# Step 1: [Description]
tool_name(parameters)

# Step 2: [Description]
another_tool(parameters)

# Step 3: Report to supervisor
# WAIT for supervisor instruction before proceeding


## 2. ADVANCED WORKFLOW (Supervisor Request: "...")

[More workflow examples]

</Professional Workflows & Tool Usage Order>

<Critical Operating Principles>
1. **ONLY perform tasks explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Use descriptive modality names** with transformation suffixes
4. **Wait for supervisor instruction** between major steps
5. **Validate modality existence** before processing
6. **Save intermediate results** for reproducibility
7. **NEVER HALLUCINATE OR LIE** - never make up tasks that haven't completed
</Critical Operating Principles>

Today's date: {date.today()}
""".strip()

    # 6. Create and return LangGraph agent
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=AgentNameExpertState,
    )
```

### Agent State Definition

Each agent needs a state class in `lobster/agents/state.py`:

```python
class AgentNameExpertState(AgentState):
    """
    State for the agent_name expert agent.
    """

    next: str

    # Agent-specific context
    task_description: str  # Description of the current task
    analysis_results: Dict[str, Any]  # Analysis results
    file_paths: List[str]  # Paths to input/output files
    methodology_parameters: Dict[str, Any]  # Method parameters
    data_context: str  # Data context
    intermediate_outputs: Dict[str, Any]  # Partial computations
```

### Custom Exception Classes

Define domain-specific exceptions at the top of the agent file:

```python
class AgentNameError(Exception):
    """Base exception for agent_name operations."""
    pass

class ModalityNotFoundError(AgentNameError):
    """Raised when requested modality doesn't exist."""
    pass

class AnalysisError(AgentNameError):
    """Raised when analysis fails."""
    pass
```

---

## Service Pattern

### Stateless Service Structure

Services are **stateless functions** that process data and return results. They **never** access DataManagerV2 directly.

**Location**: `lobster/tools/[domain]_service.py`

```python
"""
[Domain] Service for [description].

This service provides stateless [analysis type] operations for [data types].
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from anndata import AnnData
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ServiceError(Exception):
    """Base exception for service errors."""
    pass


class ValidationError(ServiceError):
    """Raised when input validation fails."""
    pass


class DomainService:
    """
    Stateless service for [domain] analysis.

    All methods are stateless and work with AnnData objects.
    Returns tuples of (processed_adata, statistics_dict).
    """

    @staticmethod
    def process_data(
        adata: AnnData,
        parameter1: str = "default",
        parameter2: int = 100,
        parameter3: bool = True
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """
        Process [data type] following [methodology].

        Args:
            adata: Input AnnData object with [requirements]
            parameter1: Description (default: "default")
            parameter2: Description (default: 100)
            parameter3: Description (default: True)

        Returns:
            Tuple containing:
                - Processed AnnData object
                - Dictionary with statistics:
                    - key1: Description
                    - key2: Description

        Raises:
            ValidationError: If input validation fails
            ServiceError: If processing fails
        """
        try:
            # 1. Validate inputs
            if adata.n_obs == 0:
                raise ValidationError("Input AnnData is empty")

            if parameter2 < 0:
                raise ValidationError(f"parameter2 must be positive, got {parameter2}")

            # 2. Process data (stateless - creates copy)
            adata_processed = adata.copy()

            # Perform analysis
            # ... processing logic ...

            # 3. Generate statistics
            statistics = {
                "n_samples": adata_processed.n_obs,
                "n_features": adata_processed.n_vars,
                "parameter1_used": parameter1,
                "parameter2_used": parameter2,
                "metric1": np.mean(adata_processed.X),
                "metric2": np.std(adata_processed.X),
            }

            # 4. Store metadata in adata.uns
            adata_processed.uns['processing_metadata'] = {
                'method': 'process_data',
                'parameters': {
                    'parameter1': parameter1,
                    'parameter2': parameter2,
                    'parameter3': parameter3,
                },
                'original_shape': adata.shape,
                'processed_shape': adata_processed.shape,
            }

            logger.info(f"Successfully processed data: {statistics}")
            return adata_processed, statistics

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise ServiceError(f"Processing failed: {str(e)}")

    @staticmethod
    def analyze_quality(
        adata: AnnData,
        threshold: float = 0.5
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """
        Analyze data quality and add QC metrics.

        Args:
            adata: Input AnnData object
            threshold: Quality threshold (default: 0.5)

        Returns:
            Tuple containing:
                - AnnData with QC metrics in .obs
                - Dictionary with QC statistics
        """
        # Similar structure to process_data
        pass
```

### Service Design Principles

1. **Stateless**: No instance variables, no state persistence
2. **Pure functions**: Same input → same output
3. **Return tuples**: Always `(AnnData, Dict[str, Any])`
4. **No DataManagerV2**: Services never access data manager
5. **Comprehensive validation**: Validate all inputs
6. **Rich statistics**: Return detailed analysis results
7. **Metadata storage**: Store processing info in `adata.uns`
8. **Error handling**: Specific exceptions with clear messages

---

## Tool Pattern

### Standard Tool Implementation

Tools bridge agents and services. They:
1. Validate modality exists in DataManagerV2
2. Get modality from DataManagerV2
3. Call stateless service
4. Store result in DataManagerV2 with descriptive name
5. Log operation for provenance
6. Return formatted response

```python
@tool
def perform_analysis(
    modality_name: str,
    parameter1: str = "default",
    parameter2: int = 100,
    save_result: bool = True
) -> str:
    """
    Perform [analysis type] on biological data.

    Args:
        modality_name: Name of the modality to process
        parameter1: Description (default: "default")
        parameter2: Description (default: 100)
        save_result: Whether to save results to file (default: True)

    Returns:
        str: Formatted summary of analysis results
    """
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            available = data_manager.list_modalities()
            raise ModalityNotFoundError(
                f"Modality '{modality_name}' not found. Available: {available}"
            )

        # 2. Get modality from data manager
        adata = data_manager.get_modality(modality_name)
        logger.info(f"Processing '{modality_name}': {adata.shape}")

        # 3. Call stateless service
        from lobster.tools.domain_service import DomainService

        service = DomainService()
        processed_adata, statistics = service.process_data(
            adata,
            parameter1=parameter1,
            parameter2=parameter2
        )

        # 4. Store result with descriptive name
        result_modality_name = f"{modality_name}_analyzed"
        data_manager.modalities[result_modality_name] = processed_adata

        # 5. Save to file if requested
        if save_result:
            save_path = f"{result_modality_name}.h5ad"
            data_manager.save_modality(result_modality_name, save_path)

        # 6. Log operation for provenance
        data_manager.log_tool_usage(
            tool_name="perform_analysis",
            parameters={
                "modality_name": modality_name,
                "parameter1": parameter1,
                "parameter2": parameter2,
            },
            description=f"Performed analysis: {statistics}"
        )

        # 7. Format and return response
        response = f"""Successfully analyzed '{modality_name}'!

📊 **Analysis Results:**
- Original shape: {adata.shape[0]} samples × {adata.shape[1]} features
- Processed shape: {processed_adata.shape[0]} samples × {processed_adata.shape[1]} features
- Parameter1: {parameter1}
- Parameter2: {parameter2}

📈 **Statistics:**
- Metric1: {statistics['metric1']:.2f}
- Metric2: {statistics['metric2']:.2f}

💾 **New modality created**: '{result_modality_name}'"""

        if save_result:
            response += f"\n💾 **Saved to**: {save_path}"

        response += "\n\nNext steps: [Suggest next analysis or visualization]"

        # Store in agent results for summary
        agent_results["details"]["analysis"] = response

        return response

    except ModalityNotFoundError as e:
        logger.error(f"Modality not found: {e}")
        return f"Error: {str(e)}"
    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Analysis failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error: {str(e)}"
```

### Tool Design Principles

1. **Validation first**: Always check modality exists
2. **Service delegation**: Call stateless service for logic
3. **Descriptive naming**: Use `{base}_{transformation}` pattern
4. **Provenance logging**: Log every operation
5. **Rich responses**: Format results with markdown
6. **Error handling**: Catch specific exceptions
7. **Next steps**: Suggest follow-up actions
8. **Optional saving**: Let user control file output

---

## File Templates

### Complete Agent Template

```python
"""
[Domain] Expert Agent for [description].

This agent [what it does] using the modular DataManagerV2 system.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import [AgentName]ExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class [AgentName]Error(Exception):
    """Base exception for [domain] operations."""
    pass


class ModalityNotFoundError([AgentName]Error):
    """Raised when requested modality doesn't exist."""
    pass


def [agent_name]_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "[agent_name]_expert_agent",
    handoff_tools: List = None,
):
    """Create [agent_name] expert agent using DataManagerV2."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params("[agent_name]_expert_agent")
    llm = create_llm("[agent_name]_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Agent-specific results storage
    agent_results = {"summary": "", "details": {}}

    # -------------------------
    # TOOLS
    # -------------------------
    @tool
    def example_tool(modality_name: str) -> str:
        """Tool description."""
        # [Implementation following tool pattern]
        pass

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [example_tool]
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
[Agent system prompt following standard structure]
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=[AgentName]ExpertState,
    )
```

### Complete Service Template

```python
"""
[Domain] Service for [description].

This service provides stateless [analysis type] operations.
"""

from typing import Any, Dict, Tuple, Optional
from anndata import AnnData
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ServiceError(Exception):
    """Base exception for service errors."""
    pass


class ValidationError(ServiceError):
    """Raised when input validation fails."""
    pass


class [Domain]Service:
    """Stateless service for [domain] analysis."""

    @staticmethod
    def process_data(
        adata: AnnData,
        parameter: str = "default"
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """
        Process data following [methodology].

        Args:
            adata: Input AnnData object
            parameter: Description

        Returns:
            Tuple of (processed_adata, statistics_dict)
        """
        try:
            # Validation
            if adata.n_obs == 0:
                raise ValidationError("Empty AnnData")

            # Processing
            adata_processed = adata.copy()
            # ... processing logic ...

            # Statistics
            statistics = {
                "metric1": 0,
                "metric2": 0,
            }

            # Metadata
            adata_processed.uns['processing'] = {
                'method': 'process_data',
                'parameters': {'parameter': parameter},
            }

            return adata_processed, statistics

        except ValidationError:
            raise
        except Exception as e:
            raise ServiceError(f"Processing failed: {e}")
```

---

## Naming Conventions

### File Names

| Type | Pattern | Example |
|------|---------|---------|
| Agent | `[domain]_expert.py` | `spatial_transcriptomics_expert.py` |
| Service | `[domain]_service.py` | `spatial_analysis_service.py` |
| State | Add to `state.py` | `SpatialTranscriptomicsExpertState` |
| Test (Agent) | `test_[domain]_expert.py` | `test_spatial_transcriptomics_expert.py` |
| Test (Service) | `test_[domain]_service.py` | `test_spatial_analysis_service.py` |
| Wiki | `[Domain-Name].md` | `Spatial-Transcriptomics.md` |

### Variable Names

| Type | Pattern | Example |
|------|---------|---------|
| Modality | `{base}_{transformation}` | `sample1_filtered_normalized` |
| Tool | `{verb}_{object}` | `analyze_spatial_patterns` |
| Function | `{verb}_{object}` | `process_spatial_data` |
| Class | `{Domain}Service` | `SpatialAnalysisService` |
| Exception | `{Domain}Error` | `SpatialAnalysisError` |

### Agent Names

- **Factory function**: `[domain]_expert()`
- **Agent name parameter**: `"[domain]_expert_agent"`
- **Display name**: "[Domain] Expert"
- **Registry key**: `'[domain]'`

---

## Testing Requirements

### Unit Test Structure

**Location**: `tests/unit/agents/test_[domain]_expert.py`

```python
"""
Unit tests for [domain] expert agent.
"""

import pytest
from unittest.mock import MagicMock, patch
from anndata import AnnData
import numpy as np
import pandas as pd

from lobster.agents.[domain]_expert import [domain]_expert, ModalityNotFoundError
from lobster.agents.state import [Domain]ExpertState
from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 for testing."""
    data_manager = MagicMock(spec=DataManagerV2)

    # Create test AnnData
    test_adata = AnnData(
        X=np.random.rand(100, 50),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(100)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
    )

    data_manager.list_modalities.return_value = ["test_data"]
    data_manager.get_modality.return_value = test_adata
    data_manager.modalities = {}
    data_manager.log_tool_usage = MagicMock()

    return data_manager


def test_agent_creation(mock_data_manager):
    """Test agent can be created successfully."""
    agent = [domain]_expert(mock_data_manager)
    assert agent is not None


def test_tool_execution(mock_data_manager):
    """Test tool executes successfully."""
    agent = [domain]_expert(mock_data_manager)

    # Extract tool for testing
    tools = agent.tools
    tool = next(t for t in tools if t.name == "example_tool")

    result = tool.invoke({"modality_name": "test_data"})

    assert isinstance(result, str)
    assert "Successfully" in result


def test_modality_not_found(mock_data_manager):
    """Test error handling for missing modality."""
    mock_data_manager.list_modalities.return_value = []

    agent = [domain]_expert(mock_data_manager)
    tools = agent.tools
    tool = next(t for t in tools if t.name == "example_tool")

    result = tool.invoke({"modality_name": "nonexistent"})

    assert "Error" in result or "not found" in result


@patch('lobster.tools.[domain]_service.[Domain]Service')
def test_service_integration(mock_service, mock_data_manager):
    """Test integration with service layer."""
    # Mock service response
    mock_adata = MagicMock(spec=AnnData)
    mock_stats = {"metric": 1.0}
    mock_service.return_value.process_data.return_value = (mock_adata, mock_stats)

    agent = [domain]_expert(mock_data_manager)
    tools = agent.tools
    tool = next(t for t in tools if t.name == "example_tool")

    result = tool.invoke({"modality_name": "test_data"})

    mock_service.return_value.process_data.assert_called_once()
    mock_data_manager.log_tool_usage.assert_called_once()
```

### Service Unit Test Structure

**Location**: `tests/unit/tools/test_[domain]_service.py`

```python
"""
Unit tests for [domain] service.
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from lobster.tools.[domain]_service import [Domain]Service, ValidationError, ServiceError


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    return AnnData(
        X=np.random.rand(100, 50),
        obs=pd.DataFrame({'sample': ['A']*50 + ['B']*50}, index=[f"cell_{i}" for i in range(100)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
    )


def test_process_data_success(sample_adata):
    """Test successful data processing."""
    service = [Domain]Service()

    result_adata, statistics = service.process_data(sample_adata, parameter="test")

    assert isinstance(result_adata, AnnData)
    assert isinstance(statistics, dict)
    assert result_adata.shape == sample_adata.shape
    assert 'processing' in result_adata.uns


def test_process_data_validation_error():
    """Test validation error handling."""
    empty_adata = AnnData()
    service = [Domain]Service()

    with pytest.raises(ValidationError):
        service.process_data(empty_adata)


def test_stateless_behavior(sample_adata):
    """Test that service doesn't modify input."""
    original_data = sample_adata.X.copy()
    service = [Domain]Service()

    service.process_data(sample_adata)

    np.testing.assert_array_equal(sample_adata.X, original_data)


def test_statistics_completeness(sample_adata):
    """Test that all expected statistics are returned."""
    service = [Domain]Service()

    _, statistics = service.process_data(sample_adata)

    required_keys = ['metric1', 'metric2']
    for key in required_keys:
        assert key in statistics
```

### Test Coverage Requirements

- **Minimum coverage**: 80% for all new code
- **Agent tests**: Creation, tool execution, error handling, service integration
- **Service tests**: Success cases, validation errors, stateless behavior, statistics
- **Integration tests**: End-to-end workflows (optional)

---

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> Tuple[AnnData, Dict]:
    """
    Brief description of what the function does.

    Longer description with more details about the functionality,
    methodology, or important considerations.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Tuple containing:
            - AnnData: Processed data with [what was added]
            - Dict: Statistics with keys:
                - 'key1': Description
                - 'key2': Description

    Raises:
        ValidationError: When input validation fails
        ServiceError: When processing fails

    Example:
        >>> result, stats = function_name("data", parameter2=20)
        >>> print(stats['key1'])
    """
```

### Wiki Documentation Template

**Location**: `lobster/wiki/[Feature-Name].md`

```markdown
# [Feature Name]

## Overview

[Brief description of what this feature does and why it's useful]

## When to Use

Use the [feature name] agent when you need to:
- [Use case 1]
- [Use case 2]
- [Use case 3]

## Available Tools

### `tool_name`

**Description**: [What the tool does]

**Parameters**:
- `modality_name` (str): The modality to process
- `parameter1` (str, optional): Description (default: "default")
- `parameter2` (int, optional): Description (default: 100)

**Returns**: Formatted summary of results

**Example**:
```python
# Through the supervisor (recommended)
User: "Please [perform task] on my_data"

# Direct tool call (for advanced users)
tool_name("my_data", parameter1="custom", parameter2=50)
```

## Workflows

### Basic Workflow

1. **Load data**: Ask the data expert to load your dataset
2. **Process data**: Request [feature] analysis from supervisor
3. **Visualize results**: Ask visualization expert to create plots

### Advanced Workflow

[More complex workflow with multiple steps]

## Parameters Guide

### Recommended Parameters

| Data Type | Parameter1 | Parameter2 | Use Case |
|-----------|-----------|-----------|----------|
| Small datasets | "default" | 50 | Quick analysis |
| Large datasets | "optimized" | 200 | Production use |

## Output Files

The [feature name] agent creates the following:

- `{modality}_analyzed.h5ad`: Processed data with [what was added]
- Plots in workspace `plots/` directory

## Troubleshooting

### Error: "Modality not found"

**Cause**: The specified modality doesn't exist in DataManagerV2

**Solution**: Check available modalities with `/data` command

### Error: "Processing failed"

**Cause**: [Common cause]

**Solution**: [How to fix]

## Technical Details

### Algorithm

[Brief description of the algorithm or methodology used]

### Dependencies

- Required: [List of required packages]
- Optional: [List of optional packages for enhanced features]

## See Also

- [Related feature 1]
- [Related feature 2]
```

---

## Complete Examples

### Example 1: Simple Analysis Agent

This example shows a minimal agent with one tool for basic data processing:

```python
"""
Example Expert Agent for demonstration.

This agent performs simple data analysis using DataManagerV2.
"""

from datetime import date
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import ExampleExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ExampleError(Exception):
    """Base exception for example operations."""
    pass


class ModalityNotFoundError(ExampleError):
    """Raised when requested modality doesn't exist."""
    pass


def example_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "example_expert_agent",
    handoff_tools: List = None,
):
    """Create example expert agent using DataManagerV2."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params("example_expert_agent")
    llm = create_llm("example_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    example_results = {"summary": "", "details": {}}

    @tool
    def analyze_data(modality_name: str) -> str:
        """
        Analyze data and compute basic statistics.

        Args:
            modality_name: Name of the modality to analyze

        Returns:
            str: Analysis summary
        """
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

            adata = data_manager.get_modality(modality_name)

            # Compute statistics
            import numpy as np
            statistics = {
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "mean": float(np.mean(adata.X)),
                "std": float(np.std(adata.X)),
            }

            # Store in adata.uns
            adata.uns['example_analysis'] = statistics
            result_name = f"{modality_name}_analyzed"
            data_manager.modalities[result_name] = adata

            # Log operation
            data_manager.log_tool_usage(
                tool_name="analyze_data",
                parameters={"modality_name": modality_name},
                description=f"Analyzed {modality_name}"
            )

            response = f"""Successfully analyzed '{modality_name}'!

📊 **Statistics:**
- Samples: {statistics['n_obs']}
- Features: {statistics['n_vars']}
- Mean value: {statistics['mean']:.2f}
- Std deviation: {statistics['std']:.2f}

💾 **New modality**: '{result_name}'"""

            example_results["details"]["analysis"] = response
            return response

        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return f"Error: {str(e)}"

    base_tools = [analyze_data]
    tools = base_tools + (handoff_tools or [])

    system_prompt = f"""
You are an example data analyst for Lobster.

<Role>
You perform basic statistical analysis on biological data.

**CRITICAL: Report results to the supervisor only.**
</Role>

<Communication Flow>
USER → SUPERVISOR → YOU → SUPERVISOR → USER
</Communication Flow>

<Available Tools>
- `analyze_data`: Compute basic statistics on data
</Available Tools>

<Critical Operating Principles>
1. Only perform tasks requested by supervisor
2. Report results to supervisor, not users
3. Use descriptive modality names
</Critical Operating Principles>

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=ExampleExpertState,
    )
```

### Example 2: Agent with Service Integration

This example shows proper service integration:

**Service** (`lobster/tools/example_service.py`):
```python
"""Example service for data processing."""

from typing import Any, Dict, Tuple
from anndata import AnnData
import numpy as np
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ServiceError(Exception):
    """Base exception for service errors."""
    pass


class ExampleService:
    """Stateless service for example processing."""

    @staticmethod
    def process(
        adata: AnnData,
        scale: bool = True
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """
        Process data with optional scaling.

        Args:
            adata: Input AnnData
            scale: Whether to scale data

        Returns:
            Tuple of (processed_adata, statistics)
        """
        adata_result = adata.copy()

        if scale:
            # Z-score normalization
            X = adata_result.X
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            adata_result.X = X

        statistics = {
            "scaled": scale,
            "shape": adata_result.shape,
            "mean": float(np.mean(adata_result.X)),
            "std": float(np.std(adata_result.X)),
        }

        adata_result.uns['processing'] = {
            'method': 'process',
            'scaled': scale,
        }

        return adata_result, statistics
```

**Agent Tool**:
```python
@tool
def process_with_service(modality_name: str, scale: bool = True) -> str:
    """Process data using example service."""
    try:
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

        adata = data_manager.get_modality(modality_name)

        # Call service
        from lobster.tools.example_service import ExampleService
        service = ExampleService()
        result_adata, stats = service.process(adata, scale=scale)

        # Store result
        result_name = f"{modality_name}_processed"
        data_manager.modalities[result_name] = result_adata

        # Log operation
        data_manager.log_tool_usage(
            tool_name="process_with_service",
            parameters={"modality_name": modality_name, "scale": scale},
            description=f"Processed {modality_name} with scaling={scale}"
        )

        return f"Successfully processed! Stats: {stats}"

    except Exception as e:
        return f"Error: {str(e)}"
```

---

## Critical Implementation Checklist

When creating new features for Lobster, ensure:

### Agent Checklist
- [ ] Factory function named `{domain}_expert()`
- [ ] Takes `data_manager`, `callback_handler`, `agent_name`, `handoff_tools`
- [ ] Custom exception classes defined
- [ ] All tools use `@tool` decorator
- [ ] Tools follow validation → service → store → log → response pattern
- [ ] System prompt includes supervisor communication flow
- [ ] System prompt includes workflows with step-by-step examples
- [ ] Returns `create_react_agent()` with correct state schema
- [ ] State class added to `lobster/agents/state.py`

### Service Checklist
- [ ] Stateless class with `@staticmethod` methods
- [ ] All methods return `Tuple[AnnData, Dict[str, Any]]`
- [ ] Comprehensive input validation
- [ ] Rich statistics in return dict
- [ ] Metadata stored in `adata.uns`
- [ ] Custom exception classes
- [ ] No access to DataManagerV2

### Testing Checklist
- [ ] Unit tests in `tests/unit/agents/test_{domain}_expert.py`
- [ ] Unit tests in `tests/unit/tools/test_{domain}_service.py`
- [ ] Mock DataManagerV2 fixture
- [ ] Test agent creation
- [ ] Test tool execution
- [ ] Test error handling
- [ ] Test service integration
- [ ] Test stateless behavior
- [ ] Minimum 80% coverage

### Documentation Checklist
- [ ] Google-style docstrings for all functions
- [ ] Type hints for all parameters
- [ ] Wiki page in `lobster/wiki/{Feature-Name}.md`
- [ ] Overview section
- [ ] When to use section
- [ ] Available tools section with examples
- [ ] Workflows section
- [ ] Troubleshooting section

---

## Summary

When creating new Lobster features:

1. **Follow the patterns exactly** - Agent factory, service stateless, tool bridge
2. **Use DataManagerV2 correctly** - Agents use it, services never do
3. **Name descriptively** - `{base}_{transformation}` for modalities
4. **Test thoroughly** - Unit tests for agents and services
5. **Document completely** - Docstrings and wiki pages
6. **Handle errors gracefully** - Specific exceptions with clear messages
7. **Log operations** - Use `data_manager.log_tool_usage()`
8. **Report to supervisor** - Never directly to users

This ensures consistency, maintainability, and professional-grade code quality across the Lobster platform.
