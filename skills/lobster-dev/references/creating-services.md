# Creating Services

Services are stateless analysis functions that perform the actual work in Lobster AI.

## Core Pattern: 3-Tuple Return

**CRITICAL**: All services MUST return a 3-tuple:

```python
def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict, AnalysisStep]:
    """
    Returns:
        - AnnData: Processed data (or None if no data output)
        - Dict: Statistics and results summary
        - AnalysisStep: Provenance record (IR - Information Record)
    """
    pass
```

> **AQUADIF Provenance Rule:** Services for tools in these AQUADIF categories MUST return a valid `AnalysisStep` as the third element: IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE. Services for DELEGATE and UTILITY tools may return `None` for the third element. See [references/aquadif-contract.md](aquadif-contract.md) for category definitions.

## Basic Service Structure

```python
# packages/lobster-mydomain/lobster/services/analysis/my_service.py

from typing import Tuple, Dict, Optional
import anndata as ad
from lobster.core.provenance.analysis_ir import AnalysisStep

class MyService:
    """Service for my domain analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def analyze(
        self,
        adata: ad.AnnData,
        param1: str,
        param2: int = 10,
        **kwargs
    ) -> Tuple[ad.AnnData, Dict, AnalysisStep]:
        """Perform analysis.
        
        Args:
            adata: Input AnnData object
            param1: First parameter
            param2: Second parameter (default: 10)
            
        Returns:
            Tuple of (processed_data, stats, provenance_record)
        """
        # Validate input
        if adata is None:
            raise ValueError("Input data required")
        
        # Perform analysis
        result = self._do_analysis(adata, param1, param2)
        
        # Compute statistics
        stats = {
            "n_cells": result.n_obs,
            "n_genes": result.n_vars,
            "param1": param1,
            "param2": param2,
            "status": "complete"
        }
        
        # Create provenance record
        ir = AnalysisStep(
            operation="my_domain.analyze",
            tool_name="analyze",
            description="Performed analysis",
            library="lobster.services.analysis.my_service",
            code_template="# Reproducible code template",
            imports=["import scanpy as sc"],
            parameters={"param1": param1, "param2": param2},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata_result"],
        )
        
        return result, stats, ir
    
    def _do_analysis(self, adata, param1, param2):
        """Internal analysis logic."""
        # Your analysis code here
        import scanpy as sc
        
        processed = adata.copy()
        # ... processing steps
        
        return processed
```

## Service Location

Services go in the appropriate package, organized by **function type** (not domain name):

```
packages/lobster-mydomain/
└── lobster/                              # PEP 420 — NO __init__.py
    └── services/                         # PEP 420 — NO __init__.py
        ├── analysis/
        │   └── mydomain_analysis_service.py
        └── quality/
            └── mydomain_quality_service.py
```

Common function-type directories: `analysis/`, `quality/`, `data_access/`, `visualization/`, `metadata/`.

Or in core for shared services:

```
lobster/
└── services/
    └── analysis/
        └── my_core_service.py
```

**PEP 420 critical:** `lobster/` and `lobster/services/` must NOT have `__init__.py` in plugin packages — they are namespace boundaries shared with the core SDK.

## Wrapping Services in Tools

Tools wrap services for agent use. Define tools inside the agent factory function where `data_manager` is available:

```python
from langchain_core.tools import tool

# Inside your agent factory function:
def my_expert_agent(data_manager, callback_handler=None, ...):

    @tool
    def analyze_my_data(
        modality_name: str,
        param1: str,
        param2: int = 10
    ) -> str:
        """Analyze data using my method.

        Args:
            modality_name: Name of the modality to analyze
            param1: Description shown to LLM
            param2: Another parameter
        """
        # Import service lazily (keeps agent startup fast)
        # PEP 420 namespace: import path is lobster.services.{function_type}.*
        from lobster.services.analysis.my_service import MyService

        # Get data from data manager
        adata = data_manager.get_modality(modality_name)

        # Call service (returns 3-tuple)
        service = MyService()
        result, stats, ir = service.analyze(adata, param1=param1, param2=param2)

        # Store processed data with descriptive name
        data_manager.store_modality(f"{modality_name}_analyzed", result)

        # Log for provenance - CRITICAL: always pass ir!
        data_manager.log_tool_usage(
            "analyze_my_data",
            {"modality_name": modality_name, "param1": param1, "param2": param2},
            stats,
            ir=ir  # Never skip this! Enables notebook export.
        )

        return f"Analysis complete: {stats['n_cells']} cells processed"
```

## Testing Services

```python
# tests/unit/services/test_my_service.py

import pytest
import anndata as ad
import numpy as np

from lobster.services.analysis.my_service import MyService


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    return ad.AnnData(
        X=np.random.rand(100, 50),
        obs={"cell_type": ["A"] * 50 + ["B"] * 50},
        var={"gene_name": [f"gene_{i}" for i in range(50)]}
    )


def test_analyze_returns_tuple(sample_adata):
    """Test that analyze returns proper 3-tuple."""
    service = MyService()
    result = service.analyze(sample_adata, param1="test")
    
    assert len(result) == 3
    assert isinstance(result[0], ad.AnnData)
    assert isinstance(result[1], dict)
    assert hasattr(result[2], "operation")


def test_analyze_stats(sample_adata):
    """Test returned statistics."""
    service = MyService()
    _, stats, _ = service.analyze(sample_adata, param1="test")

    assert "n_cells" in stats
    assert "status" in stats
    assert stats["status"] == "complete"


def test_analyze_provenance(sample_adata):
    """Test provenance record."""
    service = MyService()
    _, _, ir = service.analyze(sample_adata, param1="test", param2=20)

    assert ir.operation == "my_domain.analyze"
    assert ir.parameters["param1"] == "test"
    assert ir.parameters["param2"] == 20


def test_analyze_with_none_raises():
    """Test that None input raises error."""
    service = MyService()
    
    with pytest.raises(ValueError, match="Input data required"):
        service.analyze(None, param1="test")
```

## Common Patterns

### Optional Output Data

When service doesn't modify data:

```python
def extract_stats(self, adata) -> Tuple[None, Dict, AnalysisStep]:
    stats = {"mean_expression": float(adata.X.mean())}
    ir = AnalysisStep(...)
    return None, stats, ir  # No data output
```

### Multiple Analysis Steps

```python
def full_pipeline(self, adata, **params):
    # Step 1
    adata, stats1, ir1 = self.normalize(adata)
    
    # Step 2
    adata, stats2, ir2 = self.cluster(adata)
    
    # Combine stats
    combined_stats = {**stats1, **stats2}
    combined_ir = AnalysisStep(
        operation="my_domain.full_pipeline",
        tool_name="full_pipeline",
        description="Full analysis pipeline",
        library="lobster.services.analysis.my_service",
        code_template="# Pipeline code",
        imports=[],
        parameters=params,
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata_processed"],
    )
    
    return adata, combined_stats, combined_ir
```

### Configurable Services

```python
class ConfigurableService:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.default_method = self.config.get("method", "default")
    
    def analyze(self, adata, method: str = None, **params):
        method = method or self.default_method
        # ...
```

## Best Practices

1. **Always return 3-tuple** — Even if data unchanged
2. **Include status in stats** — `"status": "complete"` or `"status": "error"`
3. **Meaningful provenance** — IR should be human-readable
4. **Validate inputs** — Raise clear errors early
5. **Lazy imports** — Import heavy dependencies inside methods
6. **Copy before modifying** — `adata = adata.copy()` to avoid side effects

## Documentation

After creating your service:
1. Add docstrings with Args/Returns
2. Update `docs-site/content/docs/api-reference/services.mdx`
3. Include usage examples

See docs.omics-os.com/docs/developer/creating-services for full reference.
