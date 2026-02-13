# Testing Guide

How to test Lobster AI components effectively.

## Quick Commands

```bash
# Run all tests
make test

# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_my_feature.py -v

# Specific test function
pytest tests/unit/test_my_feature.py::test_specific_function -v

# With coverage
pytest --cov=lobster tests/

# Parallel execution
pytest -n auto tests/
```

## Test Structure

```
tests/
├── unit/                     # Fast, isolated tests
│   ├── agents/
│   ├── services/
│   ├── core/
│   └── tools/
├── integration/              # Tests with external deps
│   ├── test_workflows.py
│   └── test_downloads.py
├── conftest.py               # Shared fixtures
└── fixtures/                 # Test data
```

## Fixtures (`conftest.py`)

```python
import pytest
import anndata as ad
import numpy as np

@pytest.fixture
def sample_adata():
    """Standard AnnData for testing."""
    return ad.AnnData(
        X=np.random.rand(100, 50),
        obs={"cell_type": ["A"] * 50 + ["B"] * 50},
        var={"gene_name": [f"gene_{i}" for i in range(50)]}
    )

@pytest.fixture
def mock_data_manager(sample_adata):
    """Mock DataManagerV2 with v1.0.0 methods."""
    from unittest.mock import MagicMock
    dm = MagicMock()
    dm.get_modality.return_value = sample_adata
    dm.list_modalities.return_value = ["test_modality"]
    dm.store_modality.return_value = None
    dm.log_tool_usage.return_value = None
    return dm

@pytest.fixture
def mock_llm():
    """Mock LLM for agent tests."""
    from unittest.mock import MagicMock
    llm = MagicMock()
    llm.invoke.return_value = "Test response"
    return llm
```

## Testing Services

```python
# tests/unit/services/test_my_service.py

import pytest
from lobster_mydomain.services.mydomain.my_service import MyService


class TestMyService:
    """Tests for MyService."""
    
    def test_returns_3_tuple(self, sample_adata):
        """Service must return (AnnData, dict, AnalysisStep)."""
        service = MyService()
        result = service.analyze(sample_adata, param="test")
        
        assert len(result) == 3
        adata, stats, ir = result
        
        assert isinstance(adata, ad.AnnData) or adata is None
        assert isinstance(stats, dict)
        assert hasattr(ir, "activity_type")
    
    def test_stats_contain_required_fields(self, sample_adata):
        """Stats must include status."""
        service = MyService()
        _, stats, _ = service.analyze(sample_adata, param="test")
        
        assert "status" in stats
    
    def test_provenance_captures_params(self, sample_adata):
        """IR must capture all parameters."""
        service = MyService()
        _, _, ir = service.analyze(sample_adata, param="test", extra=42)
        
        assert ir.params["param"] == "test"
        assert ir.params["extra"] == 42
    
    def test_handles_empty_data(self):
        """Service handles edge cases gracefully."""
        service = MyService()
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 10))
        
        result = service.analyze(empty_adata, param="test")
        assert result[1]["status"] in ("complete", "warning")
```

## Testing Agents

Use the **AgentContractTestMixin** for standard contract validation:

```python
# tests/test_contract.py
from lobster.testing import AgentContractTestMixin


class TestMyExpertAgent(AgentContractTestMixin):
    """Contract tests for my_expert_agent."""

    agent_module = "lobster.agents.mydomain.my_agent"
    factory_name = "my_expert_agent"  # Must match factory function name
```

For custom agent tests:

```python
# tests/unit/agents/test_my_agent.py

import pytest
from unittest.mock import MagicMock, patch


class TestMyAgent:
    """Additional tests for my_expert_agent."""

    def test_agent_config_exists(self):
        """AGENT_CONFIG must be AgentRegistryConfig."""
        from lobster.agents.mydomain.my_agent import AGENT_CONFIG
        from lobster.config.agent_registry import AgentRegistryConfig

        assert isinstance(AGENT_CONFIG, AgentRegistryConfig)
        assert AGENT_CONFIG.name == "my_expert_agent"
        assert AGENT_CONFIG.tier_requirement in ("free", "premium", "enterprise")

    def test_agent_config_loads_fast(self):
        """Config must load in <50ms (entry point discovery)."""
        import time
        start = time.time()

        from lobster.agents.mydomain.my_agent import AGENT_CONFIG

        elapsed = time.time() - start
        assert elapsed < 0.05, f"Config load took {elapsed}s, must be <50ms"

    def test_factory_signature(self):
        """Factory must have standard signature."""
        import inspect
        from lobster.agents.mydomain.my_agent import my_expert_agent

        sig = inspect.signature(my_expert_agent)
        params = list(sig.parameters.keys())

        assert "data_manager" in params
        assert "delegation_tools" in params  # NOT handoff_tools
        assert "workspace_path" in params
        assert "handoff_tools" not in params  # Deprecated
```

## Testing Core Components

```python
# tests/unit/core/test_component_registry.py

import pytest


class TestComponentRegistry:
    """Tests for ComponentRegistry."""
    
    def test_discovers_agents(self):
        """Registry discovers installed agents."""
        from lobster.core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        agents = registry.get_available_agents()
        
        assert len(agents) > 0
        assert "supervisor" in [a["name"] for a in agents]
    
    def test_get_agent_returns_config(self):
        """get_agent returns valid config."""
        from lobster.core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        config = registry.get_agent("supervisor")
        
        assert config is not None
        assert "name" in config
        assert "description" in config
```

## Mocking External Dependencies

```python
# Mocking LLM calls
@patch("langchain_core.language_models.BaseLLM.invoke")
def test_with_mocked_llm(mock_invoke, sample_adata):
    mock_invoke.return_value = "Mocked response"
    # Your test code

# Mocking data loading
@patch.object(DataManagerV2, "get_current_modality_data")
def test_with_mocked_data(mock_get_data, sample_adata):
    mock_get_data.return_value = sample_adata
    # Your test code

# Mocking external APIs
@patch("requests.get")
def test_with_mocked_api(mock_get):
    mock_get.return_value.json.return_value = {"data": "test"}
    # Your test code
```

## Integration Tests

```python
# tests/integration/test_workflow.py

import pytest

@pytest.mark.integration
@pytest.mark.slow
def test_full_analysis_workflow(tmp_path):
    """Test complete analysis from data load to results."""
    from lobster.core.client import AgentClient
    
    client = AgentClient(workspace_dir=str(tmp_path))
    
    # Load data
    result = client.query("Load test_data.h5ad")
    assert "loaded" in result.lower()
    
    # Run analysis
    result = client.query("Run QC and clustering")
    assert "complete" in result.lower()
    
    # Check outputs
    assert (tmp_path / "results").exists()
```

## Test Markers

```python
# In pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
    "requires_network: needs internet",
    "premium: tests premium features",
]

# Usage
@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass
```

## CI/CD Considerations

```bash
# Fast CI (unit only)
pytest tests/unit/ -v --timeout=60

# Full CI (all tests)
pytest tests/ -v --timeout=300 --ignore=tests/integration/slow/

# Coverage report
pytest --cov=lobster --cov-report=xml tests/
```

## Best Practices

1. **Test the 3-tuple** — Every service test should verify the return format
2. **Test AGENT_CONFIG early** — Ensure fast loading for entry point discovery
3. **Mock external deps** — Never make real API calls in unit tests
4. **Use fixtures** — Avoid duplicating test data
5. **Test edge cases** — Empty data, None inputs, invalid params
6. **Mark slow tests** — So CI can skip when appropriate
