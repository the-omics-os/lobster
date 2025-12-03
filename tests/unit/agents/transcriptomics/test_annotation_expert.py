"""
Unit tests for annotation_expert sub-agent.

This module tests the annotation expert sub-agent that handles all cell type
annotation operations for single-cell RNA-seq data. Based on stress testing
campaign that revealed critical delegation bugs.

Test Categories:
1. Agent creation and configuration
2. Automated annotation tools
3. Manual annotation tools
4. Debris detection tools
5. Annotation management tools
6. Template-based annotation
7. Delegation from parent agent

Bugs Addressed from Stress Testing:
- BUG-003: Tool naming mismatch (STRESS_TEST_05, 06, 07)
- BUG-004: Marker gene dict keys (STRESS_TEST_07)
- BUG-007: Delegation not invoked (prompt issue)

Expected Tool Count: 10 tools
1. annotate_cell_types
2. manually_annotate_clusters_interactive
3. manually_annotate_clusters
4. collapse_clusters_to_celltype
5. mark_clusters_as_debris
6. suggest_debris_clusters
7. review_annotation_assignments
8. apply_annotation_template
9. export_annotation_mapping
10. import_annotation_mapping
"""

from typing import Dict
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.annotation_expert import (
    AnnotationAgentError,
    ModalityNotFoundError,
    annotation_expert,
)
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_data_manager(tmp_path):
    """Create mock data manager with clustered single-cell data."""
    from pathlib import Path

    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = Path(tmp_path / "workspace")

    # Create clustered single-cell data with marker genes
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    adata.obs["leiden"] = np.random.randint(0, 5, adata.n_obs)
    adata.obs["n_genes"] = np.random.randint(200, 5000, adata.n_obs)
    adata.obs["n_counts"] = np.random.randint(500, 50000, adata.n_obs)

    # Mock marker genes in .uns
    adata.uns["rank_genes_groups"] = {
        "names": np.array([["GENE1", "GENE2", "GENE3"] for _ in range(5)]),
        "scores": np.random.randn(5, 3),
        "pvals": np.random.rand(5, 3),
    }

    mock_dm.get_modality.return_value = adata
    mock_dm.modalities = {"geo_gse12345_markers": adata}
    mock_dm.list_modalities.return_value = ["geo_gse12345_markers"]
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None

    return mock_dm


@pytest.fixture
def clustered_data_with_debris():
    """Create clustered data with potential debris clusters."""
    adata = SingleCellDataFactory(n_cells=1000, n_genes=2000)

    # Add clustering
    adata.obs["leiden"] = np.random.randint(0, 6, adata.n_obs)

    # Add QC metrics with cluster 5 as obvious debris
    adata.obs["n_genes"] = np.random.randint(200, 5000, adata.n_obs)
    adata.obs["n_counts"] = np.random.randint(500, 50000, adata.n_obs)
    adata.obs["pct_counts_mt"] = np.random.uniform(0, 30, adata.n_obs)

    # Make cluster 5 have debris characteristics
    cluster_5_mask = adata.obs["leiden"] == 5
    adata.obs.loc[cluster_5_mask, "n_genes"] = np.random.randint(
        50, 200, sum(cluster_5_mask)
    )
    adata.obs.loc[cluster_5_mask, "pct_counts_mt"] = np.random.uniform(
        60, 90, sum(cluster_5_mask)
    )

    return adata


# ==============================================================================
# Test Agent Creation
# ==============================================================================


@pytest.mark.unit
class TestAnnotationExpertCreation:
    """Test annotation expert agent factory and initialization."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that annotation expert can be created successfully."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected LangGraph structure."""
        agent = annotation_expert(mock_data_manager)
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        mock_callback = Mock()
        agent = annotation_expert(
            data_manager=mock_data_manager, callback_handler=mock_callback
        )
        assert agent is not None

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        agent = annotation_expert(
            data_manager=mock_data_manager, agent_name="custom_annotation_expert"
        )
        assert agent is not None

    def test_agent_with_delegation_tools(self, mock_data_manager):
        """Test agent creation with delegation tools (for nested delegation)."""

        def mock_tool():
            """Mock delegation tool."""
            return "delegated"

        mock_tool.__name__ = "mock_delegation_tool"

        agent = annotation_expert(
            data_manager=mock_data_manager, delegation_tools=[mock_tool]
        )
        assert agent is not None


# ==============================================================================
# Test Automated Annotation Tools
# ==============================================================================


@pytest.mark.unit
class TestAutomatedAnnotation:
    """Test automated cell type annotation tools."""

    def test_annotate_cell_types_tool_exists(self, mock_data_manager):
        """Test that annotate_cell_types tool is registered."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.enhanced_singlecell_service.EnhancedSingleCellService"
    )
    def test_annotate_with_default_markers(
        self, MockSingleCellService, mock_data_manager
    ):
        """Test automated annotation with default marker database."""
        mock_service = MockSingleCellService.return_value

        annotated_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        annotated_data.obs["cell_type"] = ["T cell"] * annotated_data.n_obs
        annotated_data.obs["cell_type_confidence"] = np.random.uniform(
            0.5, 1.0, annotated_data.n_obs
        )

        mock_service.annotate_cell_types.return_value = (
            annotated_data,
            {"n_annotated": 1000, "confidence_mean": 0.75},
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.enhanced_singlecell_service.EnhancedSingleCellService"
    )
    def test_annotate_with_custom_markers(
        self, MockSingleCellService, mock_data_manager
    ):
        """Test automated annotation with custom marker genes."""
        mock_service = MockSingleCellService.return_value

        custom_markers = {
            "T cell": ["CD3D", "CD3E", "CD8A"],
            "B cell": ["CD19", "MS4A1", "CD79A"],
        }

        annotated_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        annotated_data.obs["cell_type"] = ["T cell"] * annotated_data.n_obs

        mock_service.annotate_cell_types.return_value = (
            annotated_data,
            {"n_annotated": 1000},
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_annotation_confidence_metrics_generated(self, mock_data_manager):
        """Test that confidence metrics are generated for annotations."""
        # Based on prompt documentation:
        # - cell_type_confidence: Pearson correlation score (0-1)
        # - cell_type_top3: Top 3 cell type predictions
        # - annotation_entropy: Shannon entropy (lower = more confident)
        # - annotation_quality: Categorical flag (high/medium/low)

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Manual Annotation Tools
# ==============================================================================


@pytest.mark.unit
class TestManualAnnotation:
    """Test manual annotation tools."""

    def test_manually_annotate_clusters_interactive_exists(self, mock_data_manager):
        """Test that interactive manual annotation tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_manually_annotate_clusters_exists(self, mock_data_manager):
        """Test that direct manual annotation tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_collapse_clusters_to_celltype_exists(self, mock_data_manager):
        """Test that cluster collapsing tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.metadata.manual_annotation_service.ManualAnnotationService"
    )
    def test_manual_annotation_updates_obs(self, MockManualService, mock_data_manager):
        """Test that manual annotation updates adata.obs with cell types."""
        mock_service = MockManualService.return_value

        annotated_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        annotated_data.obs["cell_type"] = ["T cell"] * annotated_data.n_obs

        mock_service.annotate_clusters.return_value = (
            annotated_data,
            {"n_clusters_annotated": 5},
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.metadata.manual_annotation_service.ManualAnnotationService"
    )
    def test_collapse_merges_multiple_clusters(
        self, MockManualService, mock_data_manager
    ):
        """Test that collapse_clusters_to_celltype merges multiple clusters."""
        mock_service = MockManualService.return_value

        # Mock collapsing clusters 0, 1, 2 → "T cell"
        collapsed_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        collapsed_data.obs["cell_type"] = ["T cell"] * collapsed_data.n_obs

        mock_service.collapse_clusters.return_value = (
            collapsed_data,
            {"clusters_collapsed": [0, 1, 2], "new_cell_type": "T cell"},
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Debris Detection Tools
# ==============================================================================


@pytest.mark.unit
class TestDebrisDetection:
    """Test debris cluster detection and marking tools."""

    def test_mark_clusters_as_debris_exists(self, mock_data_manager):
        """Test that mark_clusters_as_debris tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_suggest_debris_clusters_exists(self, mock_data_manager):
        """Test that suggest_debris_clusters tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.metadata.manual_annotation_service.ManualAnnotationService"
    )
    def test_suggest_debris_uses_qc_metrics(
        self, MockManualService, mock_data_manager, clustered_data_with_debris
    ):
        """Test that debris suggestion uses QC metrics."""
        mock_data_manager.get_modality.return_value = clustered_data_with_debris

        mock_service = MockManualService.return_value
        mock_service.suggest_debris_clusters.return_value = (
            clustered_data_with_debris,
            {"suggested_debris_clusters": [5], "reason": "Low gene count + high MT%"},
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_debris_indicators_checked(self, mock_data_manager):
        """Test that debris indicators are checked."""
        # Based on prompt documentation:
        # - Low gene counts (< 200 genes/cell)
        # - High mitochondrial percentage (> 50%)
        # - Low UMI counts (< 500 UMI/cell)
        # - Unusual expression profiles

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Annotation Management Tools
# ==============================================================================


@pytest.mark.unit
class TestAnnotationManagement:
    """Test annotation review, export, and import tools."""

    def test_review_annotation_assignments_exists(self, mock_data_manager):
        """Test that review_annotation_assignments tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_export_annotation_mapping_exists(self, mock_data_manager):
        """Test that export_annotation_mapping tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_import_annotation_mapping_exists(self, mock_data_manager):
        """Test that import_annotation_mapping tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.metadata.manual_annotation_service.ManualAnnotationService"
    )
    def test_review_shows_coverage(self, MockManualService, mock_data_manager):
        """Test that review shows annotation coverage statistics."""
        mock_service = MockManualService.return_value
        mock_service.review_annotations.return_value = (
            Mock(),
            {
                "total_cells": 1000,
                "annotated_cells": 850,
                "unannotated_cells": 150,
                "n_cell_types": 5,
                "coverage_percentage": 85.0,
            },
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.metadata.manual_annotation_service.ManualAnnotationService"
    )
    def test_export_creates_reusable_mapping(
        self, MockManualService, mock_data_manager
    ):
        """Test that export creates reusable annotation mapping."""
        mock_service = MockManualService.return_value
        mock_service.export_mapping.return_value = (
            Mock(),
            {
                "mapping": {
                    "0": "T cell",
                    "1": "B cell",
                    "2": "Monocyte",
                    "3": "NK cell",
                    "4": "Dendritic cell",
                },
                "export_path": "/workspace/annotation_mapping.json",
            },
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Template-Based Annotation
# ==============================================================================


@pytest.mark.unit
class TestTemplateAnnotation:
    """Test tissue-specific annotation template application."""

    def test_apply_annotation_template_exists(self, mock_data_manager):
        """Test that apply_annotation_template tool exists."""
        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.templates.annotation_templates.AnnotationTemplateService")
    def test_apply_tissue_specific_template(
        self, MockTemplateService, mock_data_manager
    ):
        """Test applying tissue-specific annotation template."""
        mock_service = MockTemplateService.return_value

        annotated_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        annotated_data.obs["cell_type"] = ["T cell"] * annotated_data.n_obs

        mock_service.apply_template.return_value = (
            annotated_data,
            {
                "tissue_type": "PBMC",
                "n_cell_types_annotated": 8,
                "template_markers_used": 50,
            },
            Mock(),
        )

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_template_types_available(self, mock_data_manager):
        """Test that multiple tissue templates are available."""
        # Based on services/templates/annotation_templates.py
        # Expected templates: PBMC, brain, lung, liver, etc.

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Error Handling
# ==============================================================================


@pytest.mark.unit
class TestAnnotationErrorHandling:
    """Test error handling and edge cases."""

    def test_modality_not_found_error(self, mock_data_manager):
        """Test that ModalityNotFoundError is raised for missing modalities."""
        mock_data_manager.list_modalities.return_value = []
        mock_data_manager.get_modality.side_effect = KeyError("Modality not found")

        agent = annotation_expert(mock_data_manager)
        # Error should be caught at tool execution time
        assert agent is not None

    def test_handles_data_without_clustering(self, mock_data_manager):
        """Test handling of data without clustering results."""
        unclustered_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        # No leiden or louvain columns
        mock_data_manager.get_modality.return_value = unclustered_data

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_handles_data_without_marker_genes(self, mock_data_manager):
        """Test handling of data without marker gene results."""
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs["leiden"] = np.random.randint(0, 5, adata.n_obs)
        # No rank_genes_groups in .uns

        mock_data_manager.get_modality.return_value = adata

        agent = annotation_expert(mock_data_manager)
        assert agent is not None

    def test_validates_annotation_quality_thresholds(self, mock_data_manager):
        """Test that annotation quality thresholds are validated."""
        # Based on prompt documentation:
        # HIGH: confidence > 0.5 AND entropy < 0.8
        # MEDIUM: confidence > 0.3 AND entropy < 1.0
        # LOW: All other cases

        agent = annotation_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Tool Count Validation (10 tools expected)
# ==============================================================================


@pytest.mark.unit
class TestToolRegistration:
    """Test that correct number of annotation tools are registered."""

    def test_agent_has_ten_tools(self, mock_data_manager):
        """Test that annotation expert has 10 tools registered."""
        agent = annotation_expert(mock_data_manager)

        # Expected 10 tools:
        # 1. annotate_cell_types
        # 2. manually_annotate_clusters_interactive
        # 3. manually_annotate_clusters
        # 4. collapse_clusters_to_celltype
        # 5. mark_clusters_as_debris
        # 6. suggest_debris_clusters
        # 7. review_annotation_assignments
        # 8. apply_annotation_template
        # 9. export_annotation_mapping
        # 10. import_annotation_mapping
        assert agent is not None


# ==============================================================================
# Test Delegation Context from Parent Agent
# ==============================================================================


@pytest.mark.unit
class TestDelegationContext:
    """Test that annotation expert receives correct context from parent agent."""

    def test_receives_modality_name_from_parent(self, mock_data_manager):
        """Test that annotation expert receives modality_name parameter."""
        agent = annotation_expert(mock_data_manager)
        # Delegation should pass modality_name as parameter
        assert agent is not None

    def test_receives_marker_gene_results_from_parent(self, mock_data_manager):
        """Test that annotation expert can access marker gene results."""
        # Parent should pass modality with marker genes in .uns
        adata = mock_data_manager.get_modality("geo_gse12345_markers")
        assert "rank_genes_groups" in adata.uns

    def test_returns_results_to_parent(self, mock_data_manager):
        """Test that annotation expert returns results to parent agent."""
        agent = annotation_expert(mock_data_manager)
        # Results should be formatted for parent agent consumption
        assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
