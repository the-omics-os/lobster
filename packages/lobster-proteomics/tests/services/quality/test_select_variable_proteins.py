"""Tests for ProteomicsQualityService.select_variable_proteins()."""

import numpy as np
import pytest
import anndata

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService


@pytest.fixture
def quality_service():
    return ProteomicsQualityService()


@pytest.fixture
def sample_proteomics_adata():
    """Create a sample AnnData with known variance structure."""
    np.random.seed(42)
    n_samples = 20
    n_proteins = 50

    # First 10 proteins have 5x higher variance
    high_var = np.random.normal(loc=10, scale=5.0, size=(n_samples, 10))
    low_var = np.random.normal(loc=10, scale=1.0, size=(n_samples, 40))
    X = np.hstack([high_var, low_var])

    # Add ~10% random missing values
    mask = np.random.random(X.shape) < 0.10
    X[mask] = np.nan

    protein_names = [f"Protein_{i}" for i in range(n_proteins)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    adata = anndata.AnnData(
        X=X.astype(np.float32),
        obs={"sample_id": sample_names},
        var={"protein_id": protein_names},
    )
    adata.obs_names = sample_names
    adata.var_names = protein_names
    return adata


class TestSelectVariableProteins:
    """Tests for select_variable_proteins service method."""

    def test_returns_3_tuple(self, quality_service, sample_proteomics_adata):
        result = quality_service.select_variable_proteins(sample_proteomics_adata)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_anndata(self, quality_service, sample_proteomics_adata):
        adata_result, _, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata
        )
        assert isinstance(adata_result, anndata.AnnData)

    def test_highly_variable_in_var(self, quality_service, sample_proteomics_adata):
        adata_result, _, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata
        )
        assert "highly_variable" in adata_result.var.columns

    def test_variability_score_in_var(self, quality_service, sample_proteomics_adata):
        adata_result, _, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata
        )
        assert "variability_score" in adata_result.var.columns

    def test_detection_rate_in_var(self, quality_service, sample_proteomics_adata):
        adata_result, _, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata
        )
        assert "detection_rate" in adata_result.var.columns

    def test_selects_correct_count(self, quality_service, sample_proteomics_adata):
        adata_result, stats, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, n_top_proteins=10
        )
        assert stats["n_selected"] == 10
        assert adata_result.var["highly_variable"].sum() == 10

    def test_cv_method_default(self, quality_service, sample_proteomics_adata):
        _, stats, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, method="cv"
        )
        assert stats["method"] == "cv"

    def test_variance_method(self, quality_service, sample_proteomics_adata):
        _, stats, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, method="variance"
        )
        assert stats["method"] == "variance"

    def test_mad_method(self, quality_service, sample_proteomics_adata):
        _, stats, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, method="mad"
        )
        assert stats["method"] == "mad"

    def test_high_variance_proteins_selected(
        self, quality_service, sample_proteomics_adata
    ):
        """First 10 proteins have 5x higher variance — they should be preferentially selected."""
        adata_result, _, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, n_top_proteins=10, method="variance"
        )
        selected_indices = np.where(adata_result.var["highly_variable"])[0]
        # Most of the top 10 should come from the first 10 high-variance proteins
        high_var_selected = sum(1 for idx in selected_indices if idx < 10)
        assert high_var_selected >= 7  # At least 7/10 from high-var group

    def test_detection_rate_filter(self, quality_service, sample_proteomics_adata):
        """With very strict detection rate, fewer proteins pass."""
        _, stats_strict, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, min_detection_rate=0.99
        )
        _, stats_loose, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, min_detection_rate=0.5
        )
        assert stats_strict["n_passing_detection"] <= stats_loose["n_passing_detection"]

    def test_ir_structure(self, quality_service, sample_proteomics_adata):
        _, _, ir = quality_service.select_variable_proteins(sample_proteomics_adata)
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "proteomics.quality.select_variable_proteins"
        assert ir.tool_name == "select_variable_proteins"
        assert "n_top_proteins" in ir.parameters

    def test_stats_keys(self, quality_service, sample_proteomics_adata):
        _, stats, _ = quality_service.select_variable_proteins(sample_proteomics_adata)
        expected_keys = {"n_total", "n_passing_detection", "n_selected", "method"}
        assert expected_keys.issubset(set(stats.keys()))

    def test_top_proteins_in_stats(self, quality_service, sample_proteomics_adata):
        _, stats, _ = quality_service.select_variable_proteins(sample_proteomics_adata)
        assert "top_proteins" in stats
        assert isinstance(stats["top_proteins"], list)
        assert len(stats["top_proteins"]) > 0

    def test_all_below_detection_raises(self, quality_service):
        """All-NaN data with detection rate > 0 should raise ValueError."""
        X = np.full((10, 5), np.nan, dtype=np.float32)
        adata = anndata.AnnData(X=X)
        adata.var_names = [f"P_{i}" for i in range(5)]
        adata.obs_names = [f"S_{i}" for i in range(10)]

        with pytest.raises(ValueError, match="No proteins pass"):
            quality_service.select_variable_proteins(
                adata, min_detection_rate=0.5
            )

    def test_n_top_proteins_capped(self, quality_service, sample_proteomics_adata):
        """Requesting more proteins than available should cap to available."""
        _, stats, _ = quality_service.select_variable_proteins(
            sample_proteomics_adata, n_top_proteins=1000
        )
        assert stats["n_selected"] <= stats["n_passing_detection"]
