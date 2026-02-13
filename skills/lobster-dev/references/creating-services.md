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

## Basic Service Structure

```python
# packages/lobster-mydomain/lobster/services/mydomain/my_service.py

from typing import Tuple, Dict, Optional
import anndata as ad
from lobster.core.provenance import AnalysisStep

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
            activity_type="my_analysis",
            inputs={
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars
            },
            outputs=stats,
            params={"param1": param1, "param2": param2}
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

Services go in the appropriate package:

```
packages/lobster-mydomain/
└── lobster/
    └── services/
        └── mydomain/
            ├── __init__.py
            └── my_service.py
```

Or in core for shared services:

```
lobster/
└── services/
    ├── __init__.py
    └── my_core_service.py
```

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
        from lobster_mydomain.services.mydomain.my_service import MyService
        # Note: Replace 'mydomain' with actual domain (transcriptomics, proteomics, etc.)

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

from lobster_mydomain.services.mydomain.my_service import MyService


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
    assert hasattr(result[2], "activity_type")


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
    
    assert ir.activity_type == "my_analysis"
    assert ir.params["param1"] == "test"
    assert ir.params["param2"] == 20


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
        activity_type="full_pipeline",
        inputs=ir1.inputs,
        outputs=combined_stats,
        params=params,
        sub_steps=[ir1, ir2]
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
