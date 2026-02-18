"""
Integration tests for complete single-cell analysis workflow.

Tests the end-to-end workflow combining:
- Marker gene detection with DEG filtering
- Cell type annotation with confidence scoring
- W3C-PROV provenance tracking throughout
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)


@pytest.fixture
def service():
    """Create service instance."""
    return EnhancedSingleCellService()


@pytest.fixture
def complete_dataset():
    """Create complete test dataset for workflow testing."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    # Create expression matrix with 4 distinct cell populations
    X = np.random.rand(n_cells, n_genes) * 0.3  # Baseline low expression

    # Population 1 - T cells (cells 0-49): CD3D, CD3E, CD4
    X[0:50, 0:5] = np.random.rand(50, 5) * 3.0 + 3.0

    # Population 2 - B cells (cells 50-99): CD79A, CD79B, MS4A1
    X[50:100, 5:10] = np.random.rand(50, 5) * 3.0 + 3.0

    # Population 3 - NK cells (cells 100-149): NKG7, GNLY, NCAM1
    X[100:150, 10:15] = np.random.rand(50, 5) * 3.0 + 3.0

    # Population 4 - Monocytes (cells 150-199): CD14, LYZ, S100A8
    X[150:200, 15:20] = np.random.rand(50, 5) * 3.0 + 3.0

    # Add some weakly expressed genes (should be filtered)
    X[0:50, 20:25] = np.random.rand(50, 5) * 1.0 + 0.5

    adata = anndata.AnnData(X=X)

    # Create realistic gene names
    gene_names = []
    gene_names += ["CD3D", "CD3E", "CD4", "CD8A", "IL7R"]  # T cell markers
    gene_names += ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5"]  # B cell markers
    gene_names += ["NKG7", "GNLY", "NCAM1", "KLRB1", "KLRD1"]  # NK markers
    gene_names += ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A"]  # Monocyte markers
    gene_names += [f"WeakGene_{i}" for i in range(5)]  # Weak markers
    gene_names += [f"Gene_{i}" for i in range(75)]  # Filler genes

    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Add cluster assignments
    clusters = ["0"] * 50 + ["1"] * 50 + ["2"] * 50 + ["3"] * 50
    adata.obs["leiden"] = pd.Categorical(clusters)

    # Preprocess
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


@pytest.fixture
def reference_markers():
    """Create biologically relevant reference markers."""
    return {
        "T_cells": ["CD3D", "CD3E", "CD4", "CD8A", "IL7R"],
        "B_cells": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5"],
        "NK_cells": ["NKG7", "GNLY", "NCAM1", "KLRB1", "KLRD1"],
        "Monocytes": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A"],
    }


class TestCompleteWorkflow:
    """Integration tests for complete analysis workflow."""

    def test_marker_detection_then_annotation(
        self, service, complete_dataset, reference_markers
    ):
        """Test complete workflow: marker detection → cell type annotation."""

        # Step 1: Find marker genes with filtering
        adata_markers, marker_stats, marker_ir = service.find_marker_genes(
            complete_dataset,
            groupby="leiden",
            method="wilcoxon",
            n_genes=25,
            min_fold_change=1.5,
            min_pct=0.25,
            max_out_pct=0.5,
        )

        # Verify marker detection worked
        assert "rank_genes_groups" in adata_markers.uns
        assert marker_stats["total_genes_filtered"] >= 0
        assert isinstance(marker_ir, AnalysisStep)

        # Step 2: Annotate cell types with confidence scoring
        adata_annotated, annotation_stats, annotation_ir = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify annotation worked
        assert "cell_type" in adata_annotated.obs.columns
        assert "cell_type_confidence" in adata_annotated.obs.columns
        assert "annotation_quality" in adata_annotated.obs.columns

        # Verify both IRs were generated
        assert marker_ir.operation == "find_marker_genes_with_filtering"
        assert annotation_ir.operation == "annotate_cell_types_with_confidence"

        # Verify data integrity
        assert len(adata_annotated) == len(complete_dataset)
        assert adata_annotated.n_vars == complete_dataset.n_vars

    def test_workflow_preserves_data(
        self, service, complete_dataset, reference_markers
    ):
        """Test that workflow preserves original data and metadata."""

        # Store original properties
        original_n_obs = complete_dataset.n_obs
        original_n_vars = complete_dataset.n_vars
        original_obs_names = list(complete_dataset.obs_names)
        original_var_names = list(complete_dataset.var_names)

        # Run workflow
        adata_markers, _, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        adata_annotated, _, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify preservation
        assert adata_annotated.n_obs == original_n_obs
        assert adata_annotated.n_vars == original_n_vars
        assert list(adata_annotated.obs_names) == original_obs_names
        assert list(adata_annotated.var_names) == original_var_names
        assert "leiden" in adata_annotated.obs.columns  # Original metadata preserved

    def test_workflow_adds_expected_columns(
        self, service, complete_dataset, reference_markers
    ):
        """Test that workflow adds all expected new columns."""

        # Run workflow
        adata_markers, _, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        adata_annotated, _, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Check marker detection results
        assert "rank_genes_groups" in adata_annotated.uns

        # Check annotation results
        expected_columns = [
            "cell_type",
            "cell_type_confidence",
            "cell_type_top3",
            "annotation_entropy",
            "annotation_quality",
        ]

        for col in expected_columns:
            assert col in adata_annotated.obs.columns, f"Missing column: {col}"

    def test_workflow_with_stringent_filtering(
        self, service, complete_dataset, reference_markers
    ):
        """Test workflow with very stringent marker filtering."""

        # Step 1: Stringent marker detection
        adata_markers, marker_stats, _ = service.find_marker_genes(
            complete_dataset,
            groupby="leiden",
            method="wilcoxon",
            n_genes=25,
            min_fold_change=2.0,  # Stricter
            min_pct=0.4,  # Higher
            max_out_pct=0.3,  # Lower
        )

        # Verify filtering occurred
        assert "filtering_params" in marker_stats
        assert marker_stats["filtering_params"]["min_fold_change"] == 2.0
        assert marker_stats["filtering_params"]["min_pct"] == 0.4
        assert marker_stats["filtering_params"]["max_out_pct"] == 0.3

        # Step 2: Annotation should still work
        adata_annotated, annotation_stats, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify annotation completed
        assert "cell_type" in adata_annotated.obs.columns
        assert annotation_stats["n_clusters"] == 4

    def test_workflow_confidence_correlates_with_quality(
        self, service, complete_dataset, reference_markers
    ):
        """Test that confidence scores correlate with annotation quality flags."""

        # Run complete workflow
        adata_markers, _, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        adata_annotated, _, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Extract confidence and quality
        confidence = adata_annotated.obs["cell_type_confidence"]
        quality = adata_annotated.obs["annotation_quality"]

        # Calculate mean confidence by quality category
        high_conf = confidence[quality == "high"].mean()
        medium_conf = (
            confidence[quality == "medium"].mean() if (quality == "medium").any() else 0
        )
        low_conf = (
            confidence[quality == "low"].mean() if (quality == "low").any() else 0
        )

        # High quality should have higher mean confidence than others
        if (quality == "medium").any():
            assert (
                high_conf > medium_conf
            ), "High quality should have higher confidence than medium"
        if (quality == "low").any():
            assert (
                high_conf > low_conf
            ), "High quality should have higher confidence than low"

    def test_workflow_stats_consistency(
        self, service, complete_dataset, reference_markers
    ):
        """Test that stats from both steps are internally consistent."""

        # Run workflow
        adata_markers, marker_stats, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        adata_annotated, annotation_stats, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify marker stats consistency
        assert marker_stats["groupby"] == "leiden"
        assert len(marker_stats["groups_analyzed"]) == 4

        # Verify annotation stats consistency
        assert annotation_stats["n_clusters"] == 4
        total_quality = sum(annotation_stats["quality_distribution"].values())
        assert total_quality == len(complete_dataset)

    def test_workflow_ir_chain(self, service, complete_dataset, reference_markers):
        """Test that both analysis steps generate proper IR for provenance chain."""

        # Run workflow and collect IRs
        _, _, marker_ir = service.find_marker_genes(complete_dataset, groupby="leiden")

        adata_markers, _, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        _, _, annotation_ir = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify IR chain
        assert marker_ir.operation == "find_marker_genes_with_filtering"
        assert annotation_ir.operation == "annotate_cell_types_with_confidence"

        # Verify both have code templates
        assert marker_ir.code_template is not None
        assert annotation_ir.code_template is not None

        # Verify both have parameter schemas
        assert marker_ir.parameter_schema is not None
        assert annotation_ir.parameter_schema is not None

        # Verify inputs/outputs documented
        assert "adata" in marker_ir.input_entities
        assert "adata" in annotation_ir.input_entities


class TestWorkflowEdgeCases:
    """Test workflow behavior in edge cases."""

    def test_workflow_with_no_reference_markers(self, service, complete_dataset):
        """Test workflow when annotation uses default markers."""

        # Run marker detection
        adata_markers, _, _ = service.find_marker_genes(
            complete_dataset, groupby="leiden"
        )

        # Annotation without reference markers (uses defaults)
        adata_annotated, annotation_stats, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=None  # Uses built-in markers
        )

        # Should still complete
        assert "cell_type" in adata_annotated.obs.columns

        # Should still have confidence scoring (with default markers)
        assert "cell_type_confidence" in adata_annotated.obs.columns
        assert "confidence_mean" in annotation_stats

    def test_workflow_with_lenient_filtering(
        self, service, complete_dataset, reference_markers
    ):
        """Test workflow with very lenient filtering (keeps more genes)."""

        # Lenient marker detection
        adata_markers, marker_stats, _ = service.find_marker_genes(
            complete_dataset,
            groupby="leiden",
            min_fold_change=0.5,  # Very lenient
            min_pct=0.1,
            max_out_pct=0.9,
        )

        # Should complete and have fewer filtered genes
        total_filtered = marker_stats["total_genes_filtered"]

        # Run annotation
        adata_annotated, annotation_stats, _ = service.annotate_cell_types(
            adata_markers, cluster_key="leiden", reference_markers=reference_markers
        )

        # Verify workflow completed
        assert "cell_type" in adata_annotated.obs.columns
        assert annotation_stats["n_clusters"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
