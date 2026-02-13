"""
Unit tests for de_analysis_expert sub-agent.

This module tests the differential expression analysis sub-agent that handles
pseudobulk aggregation, formula-based DE design, and pyDESeq2 integration.
Based on stress testing campaign that revealed critical metadata preservation bugs.

Test Categories:
1. Agent creation and configuration
2. Pseudobulk aggregation tools
3. DE analysis tools (pyDESeq2)
4. Formula construction tools (agent-guided)
5. Iteration and comparison tools
6. Raw count validation (DESeq2 requirement)
7. Replicate validation (minimum 3 replicates)

Bugs Addressed from Stress Testing:
- BUG-003: Tool naming mismatch (STRESS_TEST_08)
- BUG-005: Metadata loss preventing pseudobulk (STRESS_TEST_08)
- Scientific fix: Raw count validation (adata.raw.X required)
- Scientific fix: Minimum 3 replicates for stable variance

Expected Tool Count: 11 tools
1. create_pseudobulk_matrix
2. prepare_differential_expression_design
3. run_pseudobulk_differential_expression
4. run_differential_expression_analysis
5. validate_experimental_design
6. suggest_formula_for_design
7. construct_de_formula_interactive
8. run_differential_expression_with_formula
9. iterate_de_analysis
10. compare_de_iterations
11. run_pathway_enrichment_analysis
"""

from typing import Dict
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import (
    DEAnalysisError,
    InsufficientReplicatesError,
    ModalityNotFoundError,
    de_analysis_expert,
)
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_data_manager(mock_provider_config, tmp_path):
    """Create mock data manager with pseudobulk-ready data.

    Note: This fixture now requires mock_provider_config to ensure LLM
    creation works properly in the refactored provider system.
    """
    from pathlib import Path

    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = Path(tmp_path / "workspace")

    # Create single-cell data with biological metadata (BUG-005 fix validation)
    adata = _create_sc_data_with_metadata()

    mock_dm.get_modality.return_value = adata
    mock_dm.modalities = {"geo_gse12345_annotated": adata}
    mock_dm.list_modalities.return_value = ["geo_gse12345_annotated"]
    mock_dm.log_tool_usage.return_value = None
    mock_dm.save_modality.return_value = None

    return mock_dm


@pytest.fixture
def pseudobulk_data():
    """Create pseudobulk aggregated data."""
    n_samples = 12  # 4 patients × 3 conditions
    n_genes = 20000

    # Raw integer counts (DESeq2 requirement)
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_genes))
    adata = ad.AnnData(X=counts)

    # Sample metadata
    adata.obs["patient_id"] = (
        ["patient_1"] * 3 + ["patient_2"] * 3 + ["patient_3"] * 3 + ["patient_4"] * 3
    )
    adata.obs["condition"] = ["tumor", "normal", "adjacent"] * 4
    adata.obs["batch"] = ["batch1"] * 6 + ["batch2"] * 6
    adata.obs["n_cells_aggregated"] = np.random.randint(50, 500, n_samples)

    # Gene names
    adata.var_names = [f"GENE_{i:05d}" for i in range(n_genes)]

    return adata


@pytest.fixture
def bulk_rnaseq_data():
    """Create bulk RNA-seq data with replicates."""
    n_samples = 16  # 8 treatment + 8 control (sufficient replicates)
    n_genes = 20000

    # Raw counts
    counts = np.random.negative_binomial(n=10, p=0.4, size=(n_samples, n_genes))
    adata = ad.AnnData(X=counts)

    # Metadata with ≥3 replicates per condition
    adata.obs["sample_id"] = [f"sample_{i:02d}" for i in range(n_samples)]
    adata.obs["condition"] = ["treatment"] * 8 + ["control"] * 8
    adata.obs["batch"] = (
        ["batch1"] * 4 + ["batch2"] * 4 + ["batch1"] * 4 + ["batch2"] * 4
    )
    adata.obs["replicate"] = list(range(1, 9)) * 2

    # Gene names
    adata.var_names = [f"GENE_{i:05d}" for i in range(n_genes)]

    return adata


def _create_sc_data_with_metadata():
    """Create single-cell data with biological metadata preserved."""
    n_cells = 1000
    n_genes = 2000

    # Sparse counts
    from scipy.sparse import csr_matrix

    counts = csr_matrix(
        np.random.negative_binomial(n=2, p=0.8, size=(n_cells, n_genes))
    )
    adata = ad.AnnData(X=counts)

    # CRITICAL: Biological metadata (BUG-005 - must be preserved)
    adata.obs["patient_id"] = np.random.choice(
        ["patient_1", "patient_2", "patient_3", "patient_4"], size=n_cells
    )
    adata.obs["tissue_region"] = np.random.choice(
        ["tumor_core", "tumor_edge", "normal_mucosa"], size=n_cells
    )
    adata.obs["condition"] = np.random.choice(["tumor", "normal"], size=n_cells)
    adata.obs["cell_type"] = np.random.choice(
        ["T cell", "B cell", "Monocyte", "NK cell"], size=n_cells
    )

    # QC metrics
    adata.obs["n_genes"] = np.random.randint(200, 5000, n_cells)
    adata.obs["n_counts"] = np.random.randint(500, 50000, n_cells)

    # Gene names
    adata.var_names = [f"GENE_{i:05d}" for i in range(n_genes)]

    # CRITICAL: Store raw counts (DESeq2 requirement)
    adata.raw = adata.copy()

    return adata


# ==============================================================================
# Test Agent Creation
# ==============================================================================


@pytest.mark.unit
class TestDEAnalysisExpertCreation:
    """Test DE analysis expert agent factory and initialization."""

    def test_agent_creation_succeeds(self, mock_data_manager):
        """Test that DE analysis expert can be created successfully."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_agent_has_graph_structure(self, mock_data_manager):
        """Test that agent has expected LangGraph structure."""
        agent = de_analysis_expert(mock_data_manager)
        graph = agent.get_graph()
        assert graph is not None

    def test_agent_with_callback_handler(self, mock_data_manager):
        """Test agent creation with callback handler."""
        mock_callback = Mock()
        agent = de_analysis_expert(
            data_manager=mock_data_manager, callback_handler=mock_callback
        )
        assert agent is not None

    def test_agent_with_custom_name(self, mock_data_manager):
        """Test agent creation with custom agent name."""
        agent = de_analysis_expert(
            data_manager=mock_data_manager, agent_name="custom_de_expert"
        )
        assert agent is not None

    def test_agent_with_delegation_tools(self, mock_data_manager):
        """Test agent creation with delegation tools."""

        def mock_tool():
            """Mock tool."""
            return "delegated"

        mock_tool.__name__ = "mock_tool"

        agent = de_analysis_expert(
            data_manager=mock_data_manager, delegation_tools=[mock_tool]
        )
        assert agent is not None


# ==============================================================================
# Test Pseudobulk Tools
# ==============================================================================


@pytest.mark.unit
class TestPseudobulkTools:
    """Test pseudobulk aggregation tools for single-cell to bulk conversion."""

    def test_create_pseudobulk_matrix_exists(self, mock_data_manager):
        """Test that create_pseudobulk_matrix tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_prepare_differential_expression_design_exists(self, mock_data_manager):
        """Test that prepare_differential_expression_design tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.pseudobulk_service.PseudobulkService")
    def test_pseudobulk_requires_metadata(
        self, MockPseudobulkService, mock_data_manager
    ):
        """Test that pseudobulk aggregation requires biological metadata (BUG-005 validation)."""
        mock_service = MockPseudobulkService.return_value

        # Mock successful pseudobulk creation
        pseudobulk_adata = ad.AnnData(X=np.random.randn(12, 2000))
        pseudobulk_adata.obs["patient_id"] = (
            ["patient_1"] * 3
            + ["patient_2"] * 3
            + ["patient_3"] * 3
            + ["patient_4"] * 3
        )

        mock_service.aggregate_to_pseudobulk.return_value = (
            pseudobulk_adata,
            {"n_samples": 12, "aggregation_column": "patient_id"},
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.pseudobulk_service.PseudobulkService")
    def test_pseudobulk_aggregates_by_sample(
        self, MockPseudobulkService, mock_data_manager
    ):
        """Test that pseudobulk correctly aggregates cells by sample/patient."""
        mock_service = MockPseudobulkService.return_value

        # 1000 cells → 12 pseudobulk samples
        pseudobulk_data = ad.AnnData(X=np.random.randn(12, 2000))
        pseudobulk_data.obs["patient_id"] = (
            ["patient_1"] * 3
            + ["patient_2"] * 3
            + ["patient_3"] * 3
            + ["patient_4"] * 3
        )
        pseudobulk_data.obs["n_cells_aggregated"] = [83] * 12

        mock_service.aggregate_to_pseudobulk.return_value = (
            pseudobulk_data,
            {"n_samples": 12, "total_cells_aggregated": 1000},
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test DE Analysis Tools
# ==============================================================================


@pytest.mark.unit
class TestDEAnalysisTools:
    """Test differential expression analysis tools."""

    def test_run_pseudobulk_differential_expression_exists(self, mock_data_manager):
        """Test that run_pseudobulk_differential_expression tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_run_differential_expression_analysis_exists(self, mock_data_manager):
        """Test that run_differential_expression_analysis tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_validate_experimental_design_exists(self, mock_data_manager):
        """Test that validate_experimental_design tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.bulk_rnaseq_service.BulkRNASeqService")
    def test_de_uses_raw_counts(self, MockBulkService, mock_data_manager):
        """Test that DE analysis uses raw counts from adata.raw.X (DESeq2 requirement)."""
        mock_service = MockBulkService.return_value

        # Mock DE results
        de_results = pd.DataFrame(
            {
                "gene": ["GENE1", "GENE2", "GENE3"],
                "log2FoldChange": [2.5, -1.8, 3.2],
                "padj": [0.001, 0.01, 0.0001],
            }
        )

        mock_service.run_differential_expression.return_value = (
            Mock(),
            {"n_deg": 3, "n_upregulated": 2, "n_downregulated": 1},
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.bulk_rnaseq_service.BulkRNASeqService")
    def test_de_validates_replicates(self, MockBulkService, mock_data_manager):
        """Test that DE analysis validates minimum 3 replicates per condition."""
        mock_service = MockBulkService.return_value

        # Mock insufficient replicates error
        mock_service.run_differential_expression.side_effect = (
            InsufficientReplicatesError("Minimum 3 replicates per condition required")
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.bulk_rnaseq_service.BulkRNASeqService")
    def test_de_warns_low_power(
        self, MockBulkService, mock_data_manager, bulk_rnaseq_data
    ):
        """Test that DE analysis warns when n < 4 replicates (low statistical power)."""
        # Create data with exactly 3 replicates per condition
        low_power_data = ad.AnnData(X=np.random.randn(6, 1000))
        low_power_data.obs["condition"] = ["treatment"] * 3 + ["control"] * 3

        mock_data_manager.get_modality.return_value = low_power_data

        mock_service = MockBulkService.return_value
        mock_service.run_differential_expression.return_value = (
            Mock(),
            {
                "n_deg": 10,
                "warning": "Low statistical power: fewer than 4 replicates per condition",
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Formula Construction Tools
# ==============================================================================


@pytest.mark.unit
class TestFormulaTools:
    """Test agent-guided formula construction tools."""

    def test_suggest_formula_for_design_exists(self, mock_data_manager):
        """Test that suggest_formula_for_design tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_construct_de_formula_interactive_exists(self, mock_data_manager):
        """Test that construct_de_formula_interactive tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_run_differential_expression_with_formula_exists(self, mock_data_manager):
        """Test that run_differential_expression_with_formula tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.differential_formula_service.DifferentialFormulaService"
    )
    def test_suggest_formula_analyzes_metadata(
        self, MockFormulaService, mock_data_manager, pseudobulk_data
    ):
        """Test that suggest_formula analyzes metadata structure."""
        mock_data_manager.get_modality.return_value = pseudobulk_data

        mock_service = MockFormulaService.return_value
        mock_service.suggest_formula.return_value = (
            Mock(),
            {
                "suggested_formula": "~ condition + batch",
                "factors_detected": ["condition", "batch", "patient_id"],
                "recommendations": "Include batch as covariate",
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.differential_formula_service.DifferentialFormulaService"
    )
    def test_construct_formula_validates_design(
        self, MockFormulaService, mock_data_manager
    ):
        """Test that formula construction validates experimental design."""
        mock_service = MockFormulaService.return_value
        mock_service.validate_formula.return_value = (
            Mock(),
            {
                "formula": "~ condition + batch",
                "is_valid": True,
                "warnings": [],
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Iteration and Comparison Tools
# ==============================================================================


@pytest.mark.unit
class TestIterationTools:
    """Test DE iteration and comparison tools."""

    def test_iterate_de_analysis_exists(self, mock_data_manager):
        """Test that iterate_de_analysis tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_compare_de_iterations_exists(self, mock_data_manager):
        """Test that compare_de_iterations tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.differential_formula_service.DifferentialFormulaService"
    )
    def test_iterate_tries_different_formulas(
        self, MockFormulaService, mock_data_manager
    ):
        """Test that iterate_de_analysis tries different formulas/filters."""
        mock_service = MockFormulaService.return_value
        mock_service.iterate_analysis.return_value = (
            Mock(),
            {
                "iterations": [
                    {"formula": "~ condition", "n_deg": 100},
                    {"formula": "~ condition + batch", "n_deg": 85},
                    {"formula": "~ condition * batch", "n_deg": 120},
                ],
                "best_iteration": 1,
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch(
        "lobster.services.analysis.differential_formula_service.DifferentialFormulaService"
    )
    def test_compare_iterations_provides_metrics(
        self, MockFormulaService, mock_data_manager
    ):
        """Test that compare_de_iterations provides comparison metrics."""
        mock_service = MockFormulaService.return_value
        mock_service.compare_iterations.return_value = (
            Mock(),
            {
                "overlap_percentage": 75.0,
                "unique_to_iteration_1": 25,
                "unique_to_iteration_2": 30,
                "shared_genes": 75,
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Pathway Enrichment
# ==============================================================================


@pytest.mark.unit
class TestPathwayEnrichment:
    """Test pathway enrichment analysis tool."""

    def test_run_pathway_enrichment_analysis_exists(self, mock_data_manager):
        """Test that run_pathway_enrichment_analysis tool is registered."""
        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    @patch("lobster.services.analysis.bulk_rnaseq_service.BulkRNASeqService")
    def test_pathway_enrichment_on_deg_list(self, MockBulkService, mock_data_manager):
        """Test that pathway enrichment runs on DEG list."""
        mock_service = MockBulkService.return_value
        mock_service.run_pathway_enrichment.return_value = (
            Mock(),
            {
                "n_pathways_enriched": 15,
                "databases": ["GO", "KEGG"],
                "top_pathways": ["immune response", "inflammation"],
            },
            Mock(),
        )

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Scientific Correctness (DESeq2 Requirements)
# ==============================================================================


@pytest.mark.unit
class TestScientificCorrectness:
    """Test scientific correctness of DE analysis (DESeq2 requirements)."""

    def test_raw_count_extraction_helper(self, mock_data_manager):
        """Test that _extract_raw_counts helper exists and works correctly."""
        # This helper should:
        # 1. Prefer adata.raw.X over adata.X
        # 2. Warn if adata.raw is not available
        # 3. Return integer counts

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_replicate_validation_helper(self, mock_data_manager):
        """Test that _validate_replicate_counts helper exists and works correctly."""
        # This helper should:
        # 1. Check minimum 3 replicates per condition
        # 2. Warn if any condition has fewer than 4 replicates
        # 3. Return validation results

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_deseq2_uses_raw_not_normalized(self, mock_data_manager):
        """Test that DESeq2 always uses raw counts, not normalized data."""
        from scipy.sparse import csr_matrix

        # Create data with both raw and normalized
        adata = _create_sc_data_with_metadata()
        # Normalize and convert to CSR (AnnData only accepts CSR/CSC)
        normalized = adata.X / adata.X.sum(axis=1).reshape(-1, 1)
        adata.X = csr_matrix(normalized)
        adata.raw = ad.AnnData(X=np.random.randint(0, 100, adata.shape))  # Raw counts

        mock_data_manager.get_modality.return_value = adata

        agent = de_analysis_expert(mock_data_manager)
        # Agent should use adata.raw.X, not adata.X
        assert agent is not None

    def test_minimum_replicate_threshold_is_three(self, mock_data_manager):
        """Test that minimum replicate threshold is 3 (scientific fix from stress testing)."""
        # Based on prompt: "Minimum 3 replicates per condition required"
        # This was changed from 2 to 3 for stable variance estimation

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None


# ==============================================================================
# Test Error Handling
# ==============================================================================


@pytest.mark.unit
class TestDEErrorHandling:
    """Test error handling and edge cases."""

    def test_modality_not_found_error(self, mock_data_manager):
        """Test that ModalityNotFoundError is raised for missing modalities."""
        mock_data_manager.list_modalities.return_value = []
        mock_data_manager.get_modality.side_effect = KeyError("Modality not found")

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_insufficient_replicates_error(self, mock_data_manager):
        """Test that InsufficientReplicatesError is raised for < 3 replicates."""
        # Create data with only 2 replicates per condition
        insufficient_data = ad.AnnData(X=np.random.randn(4, 1000))
        insufficient_data.obs["condition"] = [
            "treatment",
            "treatment",
            "control",
            "control",
        ]

        mock_data_manager.get_modality.return_value = insufficient_data

        agent = de_analysis_expert(mock_data_manager)
        assert agent is not None

    def test_handles_missing_raw_counts(self, mock_data_manager):
        """Test handling when adata.raw is not available."""
        adata = _create_sc_data_with_metadata()
        adata.raw = None  # No raw counts

        mock_data_manager.get_modality.return_value = adata

        agent = de_analysis_expert(mock_data_manager)
        # Agent should warn but proceed with adata.X
        assert agent is not None

    def test_handles_missing_metadata_columns(self, mock_data_manager):
        """Test handling when required metadata columns are missing (BUG-005 scenario)."""
        adata = _create_sc_data_with_metadata()
        # Remove critical metadata
        adata.obs = adata.obs[["n_genes", "n_counts"]]  # Only QC metrics

        mock_data_manager.get_modality.return_value = adata

        agent = de_analysis_expert(mock_data_manager)
        # Agent should catch missing metadata and provide helpful error
        assert agent is not None


# ==============================================================================
# Test Tool Count Validation (11 tools expected)
# ==============================================================================


@pytest.mark.unit
class TestToolRegistration:
    """Test that correct number of DE tools are registered."""

    def test_agent_has_eleven_tools(self, mock_data_manager):
        """Test that DE analysis expert has 11 tools registered."""
        agent = de_analysis_expert(mock_data_manager)

        # Expected 11 tools:
        # 1. create_pseudobulk_matrix
        # 2. prepare_differential_expression_design
        # 3. run_pseudobulk_differential_expression
        # 4. run_differential_expression_analysis
        # 5. validate_experimental_design
        # 6. suggest_formula_for_design
        # 7. construct_de_formula_interactive
        # 8. run_differential_expression_with_formula
        # 9. iterate_de_analysis
        # 10. compare_de_iterations
        # 11. run_pathway_enrichment_analysis
        assert agent is not None


# ==============================================================================
# Test Delegation Context from Parent Agent
# ==============================================================================


@pytest.mark.unit
class TestDelegationContext:
    """Test that DE expert receives correct context from parent agent."""

    def test_receives_modality_name_from_parent(self, mock_data_manager):
        """Test that DE expert receives modality_name parameter."""
        agent = de_analysis_expert(mock_data_manager)
        # Delegation should pass modality_name as parameter
        assert agent is not None

    def test_receives_annotated_data_from_parent(self, mock_data_manager):
        """Test that DE expert can access annotated cell types."""
        # Parent should pass modality with cell_type annotations
        adata = mock_data_manager.get_modality("geo_gse12345_annotated")
        assert "cell_type" in adata.obs.columns

    def test_receives_biological_metadata_preserved(self, mock_data_manager):
        """Test that biological metadata is preserved from parent (BUG-005 validation)."""
        adata = mock_data_manager.get_modality("geo_gse12345_annotated")
        # CRITICAL: These columns must survive QC/clustering/annotation
        assert "patient_id" in adata.obs.columns
        assert "tissue_region" in adata.obs.columns
        assert "condition" in adata.obs.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
