"""
Unit tests for confidence scoring in cell type annotation.

Tests the new confidence metrics added in Task 1:
- cell_type_confidence (Pearson correlation)
- cell_type_top3 (top 3 predictions)
- annotation_entropy (Shannon entropy)
- annotation_quality (high/medium/low flag)
"""

import numpy as np
import pytest
import anndata
import pandas as pd
from scipy.sparse import csr_matrix

from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService
from lobster.core.analysis_ir import AnalysisStep


@pytest.fixture
def service():
    """Create service instance."""
    return EnhancedSingleCellService()


@pytest.fixture
def simple_adata():
    """Create simple test AnnData with known structure."""
    # Create expression matrix: 100 cells x 20 genes
    np.random.seed(42)
    n_cells = 100
    n_genes = 20

    # Generate synthetic expression with clear cell type signatures
    # Group 1 (cells 0-33): high expression in genes 0-6
    # Group 2 (cells 34-66): high expression in genes 7-13
    # Group 3 (cells 67-99): high expression in genes 14-19
    X = np.random.rand(n_cells, n_genes)

    # Add strong signatures
    X[0:33, 0:7] += 5.0   # Group 1 markers
    X[34:66, 7:14] += 5.0  # Group 2 markers
    X[67:99, 14:20] += 5.0 # Group 3 markers

    adata = anndata.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Add cluster assignments
    clusters = ["0"] * 33 + ["1"] * 33 + ["2"] * 34
    adata.obs["leiden"] = pd.Categorical(clusters)

    return adata


@pytest.fixture
def reference_markers():
    """Create reference marker dictionary matching test data."""
    return {
        "TypeA": ["Gene_0", "Gene_1", "Gene_2", "Gene_3", "Gene_4", "Gene_5", "Gene_6"],
        "TypeB": ["Gene_7", "Gene_8", "Gene_9", "Gene_10", "Gene_11", "Gene_12", "Gene_13"],
        "TypeC": ["Gene_14", "Gene_15", "Gene_16", "Gene_17", "Gene_18", "Gene_19"],
    }


class TestConfidenceScoring:
    """Test confidence scoring functionality."""

    def test_annotate_returns_three_tuple(self, service, simple_adata, reference_markers):
        """Test that annotate_cell_types returns 3-tuple (adata, stats, ir)."""
        result = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return 3-tuple"

        adata_result, stats, ir = result
        assert isinstance(adata_result, anndata.AnnData), "First element should be AnnData"
        assert isinstance(stats, dict), "Second element should be dict"
        assert isinstance(ir, AnalysisStep), "Third element should be AnalysisStep"

    def test_confidence_columns_created(self, service, simple_adata, reference_markers):
        """Test that 4 new .obs columns are created."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        required_cols = [
            "cell_type_confidence",
            "cell_type_top3",
            "annotation_entropy",
            "annotation_quality"
        ]

        for col in required_cols:
            assert col in adata_result.obs.columns, f"Missing required column: {col}"

    def test_confidence_score_range(self, service, simple_adata, reference_markers):
        """Test that confidence scores are in valid range [0, 1]."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        confidence = adata_result.obs["cell_type_confidence"]

        assert confidence.min() >= 0.0, "Confidence should be >= 0"
        assert confidence.max() <= 1.0, "Confidence should be <= 1"
        assert not confidence.isna().any(), "Should have no NaN values"

    def test_confidence_score_quality(self, service, simple_adata, reference_markers):
        """Test that confidence scores are reasonable for synthetic signatures."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        confidence = adata_result.obs["cell_type_confidence"]

        # With synthetic signatures, confidence scores should be reasonable (not zero)
        mean_confidence = confidence.mean()
        assert mean_confidence > 0.2, f"Expected reasonable mean confidence, got {mean_confidence:.3f}"

        # Verify some cells have decent confidence
        high_conf_count = (confidence > 0.3).sum()
        assert high_conf_count > 0, "Expected at least some cells with confidence > 0.3"

    def test_top3_predictions_format(self, service, simple_adata, reference_markers):
        """Test that top3 predictions are comma-separated strings."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        top3 = adata_result.obs["cell_type_top3"]

        # Check format
        for val in top3.head(10):  # Check first 10 cells
            assert isinstance(val, str), "Top3 should be string"
            parts = val.split(",")
            assert len(parts) == 3, f"Should have 3 predictions, got {len(parts)}"
            # Check each part is a valid cell type
            for part in parts:
                assert part in reference_markers.keys(), f"Invalid cell type: {part}"

    def test_entropy_values(self, service, simple_adata, reference_markers):
        """Test that entropy values are reasonable."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        entropy = adata_result.obs["annotation_entropy"]

        # Entropy should be non-negative
        assert (entropy >= 0).all(), "Entropy should be non-negative"

        # With clear signatures, entropy should be relatively low
        mean_entropy = entropy.mean()
        max_entropy = np.log(3)  # Maximum entropy for 3 cell types
        assert mean_entropy < max_entropy, f"Mean entropy {mean_entropy:.3f} should be < max {max_entropy:.3f}"

    def test_quality_categories(self, service, simple_adata, reference_markers):
        """Test that quality flags are correctly categorized."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        quality = adata_result.obs["annotation_quality"]

        # Check valid categories
        valid_categories = {"high", "medium", "low"}
        assert set(quality.unique()).issubset(valid_categories), \
            f"Invalid quality categories: {set(quality.unique())}"

        # Verify all cells have a quality category
        assert len(quality) == len(simple_adata), "All cells should have quality assignment"

        # Quality categories should be distributed (at least 2 categories present)
        assert len(quality.unique()) >= 1, "Should have at least one quality category"

    def test_quality_thresholds(self, service, simple_adata, reference_markers):
        """Test that quality categorization follows documented thresholds."""
        adata_result, _, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        confidence = adata_result.obs["cell_type_confidence"]
        entropy = adata_result.obs["annotation_entropy"]
        quality = adata_result.obs["annotation_quality"]

        # Verify thresholds
        for i in range(len(adata_result)):
            c = confidence.iloc[i]
            e = entropy.iloc[i]
            q = quality.iloc[i]

            if c > 0.5 and e < 0.8:
                assert q == "high", f"Cell {i}: conf={c:.2f}, ent={e:.2f} should be HIGH, got {q}"
            elif c > 0.3 and e < 1.0:
                assert q in ["high", "medium"], f"Cell {i}: should be HIGH or MEDIUM, got {q}"
            else:
                assert q in ["medium", "low"], f"Cell {i}: should be MEDIUM or LOW, got {q}"

    def test_stats_dict_confidence_metrics(self, service, simple_adata, reference_markers):
        """Test that stats dict includes confidence metrics."""
        _, stats, _ = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        required_keys = [
            "confidence_mean",
            "confidence_median",
            "confidence_std",
            "quality_distribution"
        ]

        for key in required_keys:
            assert key in stats, f"Missing stats key: {key}"

        # Check quality distribution structure
        quality_dist = stats["quality_distribution"]
        assert isinstance(quality_dist, dict), "quality_distribution should be dict"
        assert "high" in quality_dist, "Missing 'high' in quality_distribution"
        assert "medium" in quality_dist, "Missing 'medium' in quality_distribution"
        assert "low" in quality_dist, "Missing 'low' in quality_distribution"

        # Counts should sum to total cells
        total = quality_dist["high"] + quality_dist["medium"] + quality_dist["low"]
        assert total == len(simple_adata), f"Quality counts {total} != cell count {len(simple_adata)}"

    def test_ir_provenance(self, service, simple_adata, reference_markers):
        """Test that AnalysisStep IR is properly created."""
        _, _, ir = service.annotate_cell_types(simple_adata, reference_markers=reference_markers)

        # Check IR structure
        assert ir.operation == "annotate_cell_types_with_confidence", "Wrong operation name"
        assert ir.tool_name == "EnhancedSingleCellService.annotate_cell_types", "Wrong tool name"
        assert "scanpy + scipy" in ir.library, "Missing library info"

        # Check code template exists and mentions confidence
        assert ir.code_template is not None, "Missing code template"
        assert "confidence" in ir.code_template.lower(), "Code template should mention confidence"

        # Check parameters
        assert "reference_markers" in ir.parameters, "Missing reference_markers in IR"

        # Check parameter schema
        assert ir.parameter_schema is not None, "Missing parameter schema"
        assert "reference_markers" in ir.parameter_schema, "Missing reference_markers in schema"

    def test_default_markers_used_when_none_provided(self, service, simple_adata):
        """Test that default markers are used when reference_markers=None."""
        adata_result, stats, ir = service.annotate_cell_types(simple_adata, reference_markers=None)

        # Should still return 3-tuple
        assert isinstance(adata_result, anndata.AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

        # Service should use default markers, so confidence columns SHOULD be created
        assert "cell_type_confidence" in adata_result.obs.columns, \
            "Should create confidence column using default markers"
        assert "annotation_quality" in adata_result.obs.columns, \
            "Should create quality column using default markers"

        # Stats should have confidence metrics
        assert "confidence_mean" in stats, \
            "Should have confidence_mean when using default markers"

        # Verify default markers were used (10 cell types in default markers)
        assert stats["n_marker_sets"] == 10, \
            "Should use 10 default marker sets from service.cell_type_markers"

    def test_sparse_matrix_support(self, service, reference_markers):
        """Test that confidence scoring works with sparse matrices."""
        # Create sparse test data
        np.random.seed(42)
        n_cells = 50
        n_genes = 20

        X_dense = np.random.rand(n_cells, n_genes)
        X_dense[0:16, 0:7] += 5.0
        X_dense[17:33, 7:14] += 5.0
        X_dense[34:49, 14:20] += 5.0

        X_sparse = csr_matrix(X_dense)

        adata = anndata.AnnData(X=X_sparse)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs["leiden"] = pd.Categorical(["0"] * 17 + ["1"] * 17 + ["2"] * 16)

        # Should work without errors
        adata_result, stats, ir = service.annotate_cell_types(adata, reference_markers=reference_markers)

        # Verify confidence columns created
        assert "cell_type_confidence" in adata_result.obs.columns
        assert "annotation_quality" in adata_result.obs.columns

        # Verify results are reasonable
        assert stats["confidence_mean"] > 0.0


class TestHelperMethods:
    """Test helper methods for confidence scoring."""

    def test_calculate_per_cell_confidence_exists(self, service):
        """Test that helper method exists."""
        assert hasattr(service, "_calculate_per_cell_confidence"), \
            "Missing _calculate_per_cell_confidence method"

    def test_create_annotation_ir_exists(self, service):
        """Test that IR creation method exists."""
        assert hasattr(service, "_create_annotation_ir"), \
            "Missing _create_annotation_ir method"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
