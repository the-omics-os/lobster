# TODO: Gap Fixes for Proteomics Survival & Network Services

**Created**: 2026-01-29
**Target**: Biognosys Pilot ($50K+ deal)
**Services**: `proteomics_survival_service.py`, `proteomics_network_service.py`

---

## Summary of Gaps

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| **HIGH** | `pick_soft_threshold()` method missing | 2-3 hours | Critical for WGCNA correctness |
| **MEDIUM** | KM visualization integration | 1-2 hours | Reporting capability |
| **MEDIUM** | Integer var_names test | 30 min | Bug verification |
| **MEDIUM** | Parallel Cox regression | 1-2 hours | Performance for 9K+ proteins |
| **LOW** | Extract shared BH correction | 1 hour | Code quality |
| **LOW** | Magic numbers to constants | 30 min | Maintainability |

---

## HIGH Priority: pick_soft_threshold() Method

### Problem
WGCNA best practices require automatic soft power selection using scale-free topology criterion. Current implementation only accepts manual `soft_power` parameter.

### Location
`lobster/services/analysis/proteomics_network_service.py`

### Implementation

```python
def pick_soft_threshold(
    self,
    adata: anndata.AnnData,
    powers: Optional[List[int]] = None,
    r_squared_cutoff: float = 0.85,
    mean_connectivity_cutoff: float = 100,
    n_top_variable: int = 5000,
    correlation_method: str = "pearson",
) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
    """
    Determine optimal soft thresholding power using scale-free topology criterion.

    This method evaluates multiple powers and selects the first one that achieves
    scale-free topology (R² > cutoff for power law fit of connectivity distribution).

    Args:
        adata: AnnData object with proteomics data (samples × proteins)
        powers: List of powers to evaluate (default: [1, 2, 3, ..., 20])
        r_squared_cutoff: R² threshold for scale-free topology (default: 0.85)
        mean_connectivity_cutoff: Maximum mean connectivity threshold (default: 100)
        n_top_variable: Number of most variable proteins to use
        correlation_method: Correlation method ('pearson' or 'spearman')

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
            - Results dict with 'selected_power' and power evaluation table
            - Statistics dict
            - IR for notebook export

    Example:
        service = WGCNALiteService()
        results, stats, ir = service.pick_soft_threshold(adata)
        print(f"Optimal power: {results['selected_power']}")

        # Use selected power for module identification
        adata_modules, _, _ = service.identify_modules(
            adata, soft_power=results['selected_power']
        )
    """
```

### Algorithm (WGCNA Reference)

```python
def pick_soft_threshold(self, adata, powers=None, r_squared_cutoff=0.85, ...):
    from sklearn.linear_model import LinearRegression

    if powers is None:
        powers = list(range(1, 21))

    # 1. Get correlation matrix (reuse existing logic from identify_modules)
    X = self._prepare_expression_matrix(adata, n_top_variable)
    corr_matrix = self._compute_correlation(X, correlation_method)

    # 2. Evaluate each power
    power_results = []
    for power in powers:
        # Compute adjacency: (0.5 + 0.5 * cor)^power
        adjacency = np.power(0.5 + 0.5 * corr_matrix, power)
        np.fill_diagonal(adjacency, 0)  # No self-connections

        # Calculate connectivity (sum of adjacency)
        k = adjacency.sum(axis=1)

        # Scale-free topology fitting
        # Fit: log10(P(k)) ~ log10(k)
        k_positive = k[k > 0]
        log_k = np.log10(k_positive)

        # Discretize k into bins for frequency calculation
        hist, bin_edges = np.histogram(k_positive, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter zero-frequency bins
        nonzero = hist > 0
        log_bin = np.log10(bin_centers[nonzero])
        log_freq = np.log10(hist[nonzero])

        # Linear regression for R²
        if len(log_bin) >= 3:
            reg = LinearRegression()
            reg.fit(log_bin.reshape(-1, 1), log_freq)
            r_squared = reg.score(log_bin.reshape(-1, 1), log_freq)
            slope = reg.coef_[0]
        else:
            r_squared = 0.0
            slope = 0.0

        # Calculate mean connectivity
        mean_k = np.mean(k)
        median_k = np.median(k)

        power_results.append({
            'power': power,
            'r_squared': float(r_squared),
            'slope': float(slope),
            'mean_connectivity': float(mean_k),
            'median_connectivity': float(median_k),
            'truncated_r_squared': float(r_squared) * np.sign(slope),  # Negative slope expected
        })

    # 3. Select optimal power
    # WGCNA criterion: R² > cutoff AND mean_k < connectivity_cutoff
    selected_power = None
    for result in power_results:
        if (result['truncated_r_squared'] > r_squared_cutoff and
            result['mean_connectivity'] < mean_connectivity_cutoff):
            selected_power = result['power']
            break

    # Fallback to highest R² if none meet criteria
    if selected_power is None:
        selected_power = max(power_results, key=lambda x: x['truncated_r_squared'])['power']
        logger.warning(f"No power achieved R² > {r_squared_cutoff}. Using power={selected_power}")

    # 4. Prepare results
    results = {
        'selected_power': selected_power,
        'power_table': pd.DataFrame(power_results),
        'r_squared_cutoff': r_squared_cutoff,
        'mean_connectivity_cutoff': mean_connectivity_cutoff,
    }

    stats = {
        'selected_power': selected_power,
        'achieved_r_squared': next(r['r_squared'] for r in power_results if r['power'] == selected_power),
        'n_powers_evaluated': len(powers),
        'analysis_type': 'soft_power_selection',
    }

    ir = self._create_ir_pick_soft_threshold(powers, r_squared_cutoff, mean_connectivity_cutoff)

    return results, stats, ir
```

### IR Template

```python
def _create_ir_pick_soft_threshold(
    self,
    powers: List[int],
    r_squared_cutoff: float,
    mean_connectivity_cutoff: float,
) -> AnalysisStep:
    """Create IR for soft power selection."""
    return AnalysisStep(
        operation="proteomics.network.pick_soft_threshold",
        tool_name="pick_soft_threshold",
        description="Determine optimal soft thresholding power using scale-free topology",
        library="lobster.services.analysis.proteomics_network_service",
        code_template="""# Soft power selection for WGCNA
from lobster.services.analysis.proteomics_network_service import WGCNALiteService

service = WGCNALiteService()
results, stats, _ = service.pick_soft_threshold(
    adata,
    powers={{ powers | tojson }},
    r_squared_cutoff={{ r_squared_cutoff }},
    mean_connectivity_cutoff={{ mean_connectivity_cutoff }}
)

# Display power table
print(results['power_table'].to_string())
print(f"\\nSelected power: {results['selected_power']}")

# Use selected power for module identification
adata_modules, _, _ = service.identify_modules(
    adata, soft_power=results['selected_power']
)""",
        imports=[
            "from lobster.services.analysis.proteomics_network_service import WGCNALiteService"
        ],
        parameters={
            "powers": powers,
            "r_squared_cutoff": r_squared_cutoff,
            "mean_connectivity_cutoff": mean_connectivity_cutoff,
        },
        parameter_schema={
            "powers": ParameterSpec(
                param_type="List[int]",
                papermill_injectable=True,
                default_value=list(range(1, 21)),
                required=False,
                description="Powers to evaluate (default: 1-20)",
            ),
            "r_squared_cutoff": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.85,
                required=False,
                validation_rule="0 < r_squared_cutoff <= 1",
                description="R² threshold for scale-free topology",
            ),
            "mean_connectivity_cutoff": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=100,
                required=False,
                validation_rule="mean_connectivity_cutoff > 0",
                description="Maximum mean connectivity",
            ),
        },
        input_entities=["adata"],
        output_entities=["results"],
    )
```

### Tests to Add

```python
# tests/unit/services/analysis/test_proteomics_network_service.py

def test_pick_soft_threshold_basic(self, service, sample_adata_with_modules):
    """Test basic soft power selection."""
    results, stats, ir = service.pick_soft_threshold(
        sample_adata_with_modules,
        n_top_variable=100,
    )

    assert "selected_power" in results
    assert 1 <= results["selected_power"] <= 20
    assert "power_table" in results
    assert isinstance(results["power_table"], pd.DataFrame)
    assert len(results["power_table"]) == 20
    assert ir is not None

def test_pick_soft_threshold_custom_powers(self, service, sample_adata_with_modules):
    """Test with custom power range."""
    results, stats, ir = service.pick_soft_threshold(
        sample_adata_with_modules,
        powers=[4, 6, 8, 10, 12],
        n_top_variable=100,
    )

    assert results["selected_power"] in [4, 6, 8, 10, 12]
    assert len(results["power_table"]) == 5

def test_pick_soft_threshold_low_rsquared_warning(self, service, small_adata, caplog):
    """Test warning when no power meets R² threshold."""
    # Use small random data unlikely to show scale-free topology
    results, stats, ir = service.pick_soft_threshold(
        small_adata,
        r_squared_cutoff=0.99,  # Very high threshold
        n_top_variable=50,
    )

    assert results["selected_power"] is not None  # Should fallback
    assert "No power achieved" in caplog.text or stats["achieved_r_squared"] >= 0.99

def test_pick_soft_threshold_power_table_columns(self, service, sample_adata_with_modules):
    """Test power table has expected columns."""
    results, _, _ = service.pick_soft_threshold(
        sample_adata_with_modules,
        n_top_variable=100,
    )

    expected_cols = ['power', 'r_squared', 'slope', 'mean_connectivity',
                     'median_connectivity', 'truncated_r_squared']
    for col in expected_cols:
        assert col in results["power_table"].columns
```

---

## MEDIUM Priority: Kaplan-Meier Visualization

### Problem
Survival curves data is stored but no integration with visualization service for plotting.

### Location
Add to: `lobster/services/visualization/proteomics_visualization_service.py`

### Implementation

```python
def create_kaplan_meier_curve(
    self,
    adata: anndata.AnnData,
    survival_curves: Optional[Dict[str, Any]] = None,
    show_confidence_intervals: bool = True,
    show_at_risk_table: bool = True,
    show_censoring_marks: bool = True,
    title: Optional[str] = None,
    color_palette: Optional[List[str]] = None,
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create Kaplan-Meier survival curves with confidence intervals.

    Args:
        adata: AnnData with kaplan_meier results in uns
        survival_curves: Pre-computed survival curves dict (if None, uses adata.uns)
        show_confidence_intervals: Show 95% CI bands
        show_at_risk_table: Add at-risk table annotation
        show_censoring_marks: Show censoring tick marks on curves
        title: Custom plot title
        color_palette: Custom colors for groups

    Returns:
        Tuple[go.Figure, Dict[str, Any]]: KM curve figure and statistics
    """
    try:
        # Get survival curves data
        if survival_curves is None:
            if "kaplan_meier" not in adata.uns:
                raise ProteomicsVisualizationError(
                    "No Kaplan-Meier results in adata.uns. Run kaplan_meier_analysis first."
                )
            km_data = adata.uns["kaplan_meier"]
            survival_curves = km_data["survival_curves"]
            log_rank_p = km_data.get("log_rank_p_value")
            protein = km_data.get("protein", "")
        else:
            log_rank_p = None
            protein = ""

        # Default colors (survival analysis convention)
        if color_palette is None:
            color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]

        fig = go.Figure()

        # Plot each group's survival curve
        for i, (group_name, curve_data) in enumerate(survival_curves.items()):
            color = color_palette[i % len(color_palette)]
            timeline = curve_data["timeline"]
            survival_func = curve_data["survival_function"]

            # Main survival curve (step function)
            fig.add_trace(go.Scatter(
                x=timeline,
                y=survival_func,
                mode="lines",
                line=dict(color=color, width=2, shape="hv"),
                name=f"{group_name} (n={curve_data['n_samples']})",
                legendgroup=group_name,
            ))

            # Confidence intervals
            if show_confidence_intervals and "confidence_lower" in curve_data:
                fig.add_trace(go.Scatter(
                    x=timeline + timeline[::-1],
                    y=curve_data["confidence_upper"] + curve_data["confidence_lower"][::-1],
                    fill="toself",
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    legendgroup=group_name,
                    hoverinfo="skip",
                ))

            # Censoring marks (small vertical ticks)
            if show_censoring_marks:
                # Add tick marks where censoring occurs (placeholder - would need actual censoring times)
                pass

        # Add log-rank p-value annotation
        if log_rank_p is not None:
            p_text = f"p < 0.001" if log_rank_p < 0.001 else f"p = {log_rank_p:.3f}"
            fig.add_annotation(
                x=0.95, y=0.95,
                xref="paper", yref="paper",
                text=f"Log-rank {p_text}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )

        # Update layout
        plot_title = title or f"Kaplan-Meier Survival Curves" + (f" - {protein}" if protein else "")
        fig.update_layout(
            title=dict(text=plot_title, x=0.5),
            xaxis_title="Time (days)",
            yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, None]),
            legend=dict(x=0.7, y=0.95),
            width=self.default_width,
            height=self.default_height,
            template="plotly_white",
        )

        # Statistics
        stats = {
            "plot_type": "kaplan_meier_curve",
            "n_groups": len(survival_curves),
            "groups": list(survival_curves.keys()),
            "log_rank_p_value": log_rank_p,
        }

        return fig, stats

    except Exception as e:
        raise ProteomicsVisualizationError(f"Failed to create KM curve: {str(e)}")
```

### Tests to Add

```python
# tests/unit/services/visualization/test_proteomics_visualization_service.py

def test_create_kaplan_meier_curve_basic(self, service):
    """Test basic KM curve creation."""
    # Create mock survival curves data
    survival_curves = {
        "Low": {
            "timeline": [0, 30, 60, 90, 120],
            "survival_function": [1.0, 0.95, 0.85, 0.75, 0.65],
            "confidence_lower": [1.0, 0.90, 0.78, 0.65, 0.53],
            "confidence_upper": [1.0, 0.98, 0.92, 0.85, 0.77],
            "n_samples": 50,
            "n_events": 18,
        },
        "High": {
            "timeline": [0, 30, 60, 90, 120],
            "survival_function": [1.0, 0.90, 0.70, 0.50, 0.35],
            "confidence_lower": [1.0, 0.84, 0.60, 0.38, 0.23],
            "confidence_upper": [1.0, 0.96, 0.80, 0.62, 0.47],
            "n_samples": 50,
            "n_events": 32,
        },
    }

    # Create mock adata
    adata = anndata.AnnData(X=np.random.randn(100, 10))
    adata.uns["kaplan_meier"] = {
        "survival_curves": survival_curves,
        "log_rank_p_value": 0.023,
        "protein": "EGFR",
    }

    fig, stats = service.create_kaplan_meier_curve(adata)

    assert isinstance(fig, go.Figure)
    assert stats["n_groups"] == 2
    assert stats["log_rank_p_value"] == 0.023
```

---

## MEDIUM Priority: Integer var_names Test

### Problem
Fix for integer var_names exists but needs explicit test verification.

### Location
`tests/unit/services/analysis/test_proteomics_survival_service.py`

### Test to Add

```python
class TestProteomicsSurvivalServiceEdgeCases:
    """Edge case tests for ProteomicsSurvivalService."""

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_integer_var_names_handling(self, service):
        """Test Cox regression with integer var_names (AnnData default)."""
        np.random.seed(42)

        X = np.random.randn(30, 10)
        obs = pd.DataFrame({
            "PFS_days": np.random.exponential(scale=500, size=30),
            "PFS_event": np.random.binomial(1, 0.7, 30),
        })

        # Create AnnData WITHOUT explicit var - defaults to integer indices [0, 1, 2, ...]
        adata = anndata.AnnData(X=X, obs=obs)

        # Verify var_names are integers (RangeIndex)
        assert adata.var_names.tolist() == list(range(10))

        # Should NOT raise KeyError
        adata_cox, stats, ir = service.perform_cox_regression(
            adata,
            duration_col="PFS_days",
            event_col="PFS_event",
            min_samples=20,
        )

        # Verify results stored correctly
        assert "cox_hazard_ratio" in adata_cox.var.columns
        assert len(adata_cox.var["cox_hazard_ratio"]) == 10
        assert stats["n_proteins_tested"] > 0
```

---

## MEDIUM Priority: Parallel Cox Regression

### Problem
`n_jobs` parameter exists but is unused. For 9,283 proteins (Biognosys scale), sequential processing is slow.

### Location
`lobster/services/analysis/proteomics_survival_service.py:351-455`

### Implementation

```python
# Add to imports at top of file
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

# Add helper method
def _fit_single_protein_cox(
    self,
    protein_data: Dict[str, Any],
    penalizer: float,
    covariate_cols: List[str],
) -> Dict[str, Any]:
    """Fit Cox model for a single protein (thread-safe)."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    protein_name = protein_data["protein_name"]
    protein_idx = protein_data["protein_idx"]
    df_cox = protein_data["df_cox"]

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df_cox, duration_col="T", event_col="E", show_progress=False)

        hr = float(np.exp(cph.params_["protein"]))
        # ... rest of result extraction (same as current loop)

        return {
            "protein": protein_name,
            "protein_index": protein_idx,
            "hazard_ratio": hr,
            # ... other fields
            "converged": True,
        }
    except Exception as e:
        return {
            "protein": protein_name,
            "protein_index": protein_idx,
            "hazard_ratio": np.nan,
            # ... other NaN fields
            "converged": False,
        }

# Modify perform_cox_regression method
def perform_cox_regression(self, adata, ..., n_jobs: int = 1):
    # ... existing setup code ...

    # Prepare protein data for parallel processing
    protein_data_iter = self._prepare_protein_data(
        X, protein_names, valid_mask, duration, event,
        covariate_cols, adata_cox, min_samples
    )

    if n_jobs == 1:
        # Sequential (existing behavior)
        cox_results = []
        for protein_data in protein_data_iter:
            result = self._fit_single_protein_cox(protein_data, penalizer, covariate_cols)
            cox_results.append(result)
    else:
        # Parallel processing
        cox_results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self._fit_single_protein_cox,
                    protein_data, penalizer, covariate_cols
                ): protein_data["protein_name"]
                for protein_data in protein_data_iter
            }

            for future in as_completed(futures):
                result = future.result()
                cox_results.append(result)

        # Sort by protein index to maintain order
        cox_results.sort(key=lambda x: x["protein_index"])

    # ... rest of method unchanged ...
```

### Tests to Add

```python
@pytest.mark.skipif(
    not ProteomicsSurvivalService()._lifelines_available,
    reason="lifelines not installed",
)
def test_cox_regression_parallel(self, service, sample_adata_with_survival):
    """Test Cox regression with parallel processing."""
    adata_cox, stats, ir = service.perform_cox_regression(
        sample_adata_with_survival,
        duration_col="PFS_days",
        event_col="PFS_event",
        n_jobs=4,
    )

    assert stats["n_proteins_tested"] > 0
    assert "cox_hazard_ratio" in adata_cox.var.columns

def test_cox_regression_parallel_matches_sequential(self, service, small_adata_with_survival):
    """Test parallel results match sequential."""
    np.random.seed(42)

    # Sequential
    adata_seq, stats_seq, _ = service.perform_cox_regression(
        small_adata_with_survival.copy(),
        duration_col="PFS_days",
        event_col="PFS_event",
        n_jobs=1,
    )

    # Parallel
    adata_par, stats_par, _ = service.perform_cox_regression(
        small_adata_with_survival.copy(),
        duration_col="PFS_days",
        event_col="PFS_event",
        n_jobs=4,
    )

    # Results should match
    np.testing.assert_array_almost_equal(
        adata_seq.var["cox_hazard_ratio"].values,
        adata_par.var["cox_hazard_ratio"].values,
        decimal=5,
    )
```

---

## LOW Priority: Extract Shared BH Correction

### Problem
Identical `_benjamini_hochberg` implementations in two services.

### Location
Create: `lobster/utils/statistics.py`

### Implementation

```python
"""
Statistical utilities for bioinformatics analysis.

Provides common statistical functions used across multiple services,
including multiple testing correction methods.
"""

from typing import List

import numpy as np


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Reference:
        Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing. Journal of
        the Royal Statistical Society B, 57:289–300.

    Args:
        p_values: List of raw p-values to correct

    Returns:
        List of FDR-adjusted p-values (q-values)

    Example:
        >>> p_values = [0.01, 0.03, 0.05, 0.10, 0.50]
        >>> fdr = benjamini_hochberg_correction(p_values)
        >>> fdr
        [0.05, 0.075, 0.0833, 0.125, 0.50]
    """
    n = len(p_values)
    if n == 0:
        return []

    # Get sorted indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate FDR
    fdr = np.zeros(n)
    for i in range(n):
        fdr[sorted_indices[i]] = sorted_p[i] * n / (i + 1)

    # Ensure monotonicity (going from largest to smallest)
    fdr_monotonic = np.zeros(n)
    inverse_sorted = np.argsort(sorted_indices)
    fdr_sorted = fdr[sorted_indices]

    running_min = 1.0
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, fdr_sorted[i])
        fdr_monotonic[i] = running_min

    # Cap at 1.0
    fdr_final = np.minimum(fdr_monotonic[inverse_sorted], 1.0)

    return fdr_final.tolist()


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple testing.

    Args:
        p_values: List of raw p-values

    Returns:
        List of Bonferroni-corrected p-values
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]
```

### Update `lobster/utils/__init__.py`

```python
from lobster.utils.statistics import (
    benjamini_hochberg_correction,
    bonferroni_correction,
)
```

### Refactor Services

```python
# In both survival and network services, replace:
# def _benjamini_hochberg(self, p_values: List[float]) -> List[float]:
#     ... (entire method)

# With:
from lobster.utils.statistics import benjamini_hochberg_correction

# And replace calls:
# fdr_values = self._benjamini_hochberg(p_values)
# With:
fdr_values = benjamini_hochberg_correction(p_values)
```

---

## LOW Priority: Magic Numbers to Constants

### Problem
Several magic numbers scattered in code without explanation.

### Locations
- `proteomics_survival_service.py:360` - variance threshold `1e-6`
- `proteomics_survival_service.py:663` - min group size for KM `5`
- `proteomics_survival_service.py:956` - min group for optimal cutpoint `10`

### Implementation

```python
# Add at top of class
class ProteomicsSurvivalService:
    """..."""

    # Class constants
    MIN_VARIANCE_THRESHOLD = 1e-6  # Proteins with lower variance are skipped
    MIN_KM_GROUP_SIZE = 5  # Minimum samples per KM group
    MIN_OPTIMAL_CUTPOINT_GROUP = 10  # Minimum samples for optimal cutpoint search

    # Replace magic numbers with constants
    # Line 360:
    if np.std(protein_valid) < self.MIN_VARIANCE_THRESHOLD:

    # Line 663:
    if group_mask.sum() < self.MIN_KM_GROUP_SIZE:

    # Line 956:
    if high_mask.sum() < self.MIN_OPTIMAL_CUTPOINT_GROUP:
```

---

## Implementation Order

1. **HIGH** - `pick_soft_threshold()` (2-3 hours)
   - Core WGCNA functionality, required for correct workflow

2. **MEDIUM** - Integer var_names test (30 min)
   - Quick win, verifies existing fix

3. **MEDIUM** - KM visualization (1-2 hours)
   - Adds value for Biognosys reporting

4. **MEDIUM** - Parallel Cox regression (1-2 hours)
   - Performance improvement for production scale

5. **LOW** - Shared BH utility (1 hour)
   - Code quality, not blocking

6. **LOW** - Magic numbers (30 min)
   - Maintainability

---

## Verification Checklist

After implementing each fix:

- [ ] All existing 47 tests still pass
- [ ] New tests added and passing
- [ ] IR template generates valid notebook code
- [ ] `make format` and `make lint` pass
- [ ] No new security vulnerabilities (bandit)

---

## References

- WGCNA Tutorial: https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/Rpackages/WGCNA/Tutorials/
- Scale-free topology: Barabási & Albert (1999) Science
- Lifelines documentation: https://lifelines.readthedocs.io/
- Existing patterns: `lobster/services/analysis/proteomics_differential_service.py`
