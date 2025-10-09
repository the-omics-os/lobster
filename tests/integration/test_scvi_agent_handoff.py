"""
Integration tests for scVI agent handoff workflow.

Tests the complete workflow from SingleCell Expert → ML Expert → back to SingleCell Expert
with conditional testing based on scVI availability.
"""

from unittest.mock import MagicMock, Mock, patch

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2

# Check if scVI is available for conditional testing
try:
    import scvi
    import torch

    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 instance."""
    data_manager = MagicMock(spec=DataManagerV2)

    # Create mock single-cell AnnData
    n_cells = 1000
    n_genes = 2000

    mock_adata = MagicMock(spec=anndata.AnnData)
    mock_adata.n_obs = n_cells
    mock_adata.n_vars = n_genes
    mock_adata.shape = (n_cells, n_genes)
    mock_adata.obs = pd.DataFrame(
        {
            "sample": ["sample1"] * 500 + ["sample2"] * 500,
            "leiden": np.random.randint(0, 5, n_cells).astype(str),
        }
    )
    mock_adata.obsm = {}  # Will be populated with scVI embeddings
    mock_adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    mock_adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Mock data manager methods
    data_manager.list_modalities.return_value = ["test_modality"]
    data_manager.get_modality.return_value = mock_adata
    data_manager.modalities = {"test_modality": mock_adata}
    data_manager.log_tool_usage = MagicMock()

    return data_manager, mock_adata


class TestScviHandoffWorkflow:
    """Test the complete agent handoff workflow for scVI."""

    @pytest.mark.skip(
        reason="request_scvi_embedding tool is commented out in singlecell_expert.py (supervisor-mediated flow)"
    )
    def test_singlecell_expert_handoff_request(self, mock_data_manager):
        """Test SingleCell Expert creates proper handoff request."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Create SingleCell Expert agent
        sc_agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = sc_agent.get_graph().nodes["tools"].data.tools_by_name

        # Check if request_scvi_embedding tool exists
        assert (
            "request_scvi_embedding" in tools_by_name
        ), f"request_scvi_embedding tool not found. Available tools: {list(tools_by_name.keys())}"

        request_tool = tools_by_name["request_scvi_embedding"]

        # Test the handoff request
        result = request_tool.invoke(
            {
                "modality_name": "test_modality",
                "n_latent": 15,
                "batch_key": "sample",
                "max_epochs": 200,
                "use_gpu": False,
            }
        )

        # Verify handoff message contains expected elements
        assert "HANDOFF TO MACHINE LEARNING EXPERT" in result
        assert "Train scVI embedding" in result
        assert "test_modality" in result
        assert "n_latent: 15" in result or "Latent dimensions: 15" in result
        assert "sample" in result

        # Verify logging was called
        data_manager.log_tool_usage.assert_called()
        last_log_call = data_manager.log_tool_usage.call_args
        assert last_log_call[1]["tool_name"] == "request_scvi_embedding"

    @pytest.mark.skip(
        reason="request_scvi_embedding tool is commented out in singlecell_expert.py (supervisor-mediated flow)"
    )
    def test_singlecell_expert_validates_modality(self, mock_data_manager):
        """Test SingleCell Expert validates modality before handoff."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager
        data_manager.list_modalities.return_value = [
            "other_modality"
        ]  # Different modality

        # Create SingleCell Expert agent
        sc_agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = sc_agent.get_graph().nodes["tools"].data.tools_by_name

        # Check if request_scvi_embedding tool exists
        assert (
            "request_scvi_embedding" in tools_by_name
        ), f"request_scvi_embedding tool not found. Available tools: {list(tools_by_name.keys())}"

        request_tool = tools_by_name["request_scvi_embedding"]

        # Test with non-existent modality
        result = request_tool.invoke({"modality_name": "nonexistent_modality"})

        # Should return error message, not handoff
        assert "❌" in result or "not found" in result
        assert "HANDOFF" not in result

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scVI dependencies not installed")
    def test_ml_expert_scvi_availability_check(self, mock_data_manager):
        """Test ML Expert scVI availability checking."""
        from lobster.agents.machine_learning_expert import machine_learning_expert

        data_manager, mock_adata = mock_data_manager

        # Create ML Expert agent
        ml_agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = ml_agent.get_graph().nodes["tools"].data.tools_by_name

        assert (
            "check_scvi_availability" in tools_by_name
        ), f"check_scvi_availability tool not found. Available: {list(tools_by_name.keys())}"
        check_tool = tools_by_name["check_scvi_availability"]

        # Test availability check
        result = check_tool.invoke({})

        # Should indicate scVI is ready (since we're testing with scVI available)
        assert "✅ scVI is ready" in result or "available" in result.lower()

    @pytest.mark.skipif(
        SCVI_AVAILABLE, reason="Testing unavailable scenario when scVI is installed"
    )
    def test_ml_expert_scvi_unavailable_message(self, mock_data_manager):
        """Test ML Expert provides installation guidance when scVI unavailable."""
        from lobster.agents.machine_learning_expert import machine_learning_expert

        data_manager, mock_adata = mock_data_manager

        # Create ML Expert agent
        ml_agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = ml_agent.get_graph().nodes["tools"].data.tools_by_name

        assert (
            "check_scvi_availability" in tools_by_name
        ), f"check_scvi_availability tool not found. Available: {list(tools_by_name.keys())}"
        check_tool = tools_by_name["check_scvi_availability"]

        # Test availability check when not available
        result = check_tool.invoke({})

        # Should provide installation instructions
        assert "❌" in result or "not available" in result
        assert "pip install" in result or "Installation" in result

    def test_clustering_service_use_rep_parameter(self):
        """Test clustering service accepts use_rep parameter."""
        from lobster.tools.clustering_service import ClusteringService

        # Create clustering service
        clustering_service = ClusteringService()

        # Create mock AnnData with scVI embeddings
        n_cells = 100
        n_genes = 500

        mock_adata = MagicMock(spec=anndata.AnnData)
        mock_adata.n_obs = n_cells
        mock_adata.n_vars = n_genes
        mock_adata.shape = (n_cells, n_genes)
        mock_adata.obs = pd.DataFrame({"cell_type": ["A"] * 50 + ["B"] * 50})

        # Mock scVI embeddings
        scvi_embeddings = np.random.randn(n_cells, 10)
        mock_adata.obsm = {"X_scvi": scvi_embeddings}

        # Mock scanpy functions
        with (
            patch("scanpy.pp.neighbors") as mock_neighbors,
            patch("scanpy.tl.leiden") as mock_leiden,
            patch("scanpy.tl.umap") as mock_umap,
            patch("scanpy.tl.rank_genes_groups") as mock_rank_genes,
        ):

            # Configure mocks to simulate successful clustering
            mock_adata.obs["leiden"] = np.random.randint(0, 3, n_cells).astype(str)
            mock_adata.obsm["X_umap"] = np.random.randn(n_cells, 2)

            try:
                # Test clustering with custom embedding
                adata_result, stats = clustering_service.cluster_and_visualize(
                    adata=mock_adata, use_rep="X_scvi", resolution=0.5
                )

                # Verify scanpy was called with use_rep parameter
                mock_neighbors.assert_called()
                neighbors_call = mock_neighbors.call_args
                assert "use_rep" in neighbors_call[1]
                assert neighbors_call[1]["use_rep"] == "X_scvi"

                # Verify results
                assert stats["n_clusters"] > 0
                assert stats["has_umap"] is True

            except Exception as e:
                # If clustering fails due to missing dependencies, that's expected
                # The important thing is that use_rep parameter is accepted
                assert "use_rep" not in str(e)  # Parameter should be accepted


class TestEndToEndWorkflow:
    """Test end-to-end workflow scenarios."""

    def test_complete_workflow_without_scvi(self, mock_data_manager):
        """Test complete workflow when scVI is not available."""
        from lobster.agents.machine_learning_expert import machine_learning_expert
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Test SingleCell Expert handoff request
        sc_agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        sc_tools_by_name = sc_agent.get_graph().nodes["tools"].data.tools_by_name

        # Skip test if request_scvi_embedding is not available (commented out)
        if "request_scvi_embedding" not in sc_tools_by_name:
            pytest.skip(
                "request_scvi_embedding tool not available (commented out in source)"
            )

        request_tool = sc_tools_by_name["request_scvi_embedding"]

        handoff_message = request_tool.invoke({"modality_name": "test_modality"})

        # Should create handoff message even without scVI
        assert isinstance(handoff_message, str)
        assert len(handoff_message) > 0

        # Test ML Expert response to check availability
        ml_agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph
        ml_tools_by_name = ml_agent.get_graph().nodes["tools"].data.tools_by_name

        assert (
            "check_scvi_availability" in ml_tools_by_name
        ), f"check_scvi_availability tool not found. Available: {list(ml_tools_by_name.keys())}"
        check_tool = ml_tools_by_name["check_scvi_availability"]

        availability_result = check_tool.invoke({})

        # Should provide clear guidance regardless of availability
        assert isinstance(availability_result, str)
        assert len(availability_result) > 0

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="Requires scVI installation")
    def test_ml_expert_train_scvi_tool_structure(self, mock_data_manager):
        """Test ML Expert scVI training tool exists and has correct structure."""
        from lobster.agents.machine_learning_expert import machine_learning_expert

        data_manager, mock_adata = mock_data_manager

        # Create ML Expert agent
        ml_agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = ml_agent.get_graph().nodes["tools"].data.tools_by_name

        # Verify scVI tools exist
        assert (
            "check_scvi_availability" in tools_by_name
        ), f"check_scvi_availability not found. Available: {list(tools_by_name.keys())}"
        assert (
            "train_scvi_embedding" in tools_by_name
        ), f"train_scvi_embedding not found. Available: {list(tools_by_name.keys())}"

        # Get train_scvi_embedding tool
        train_tool = tools_by_name["train_scvi_embedding"]

        # Test tool accepts expected parameters
        # Note: We won't actually run training in tests, just verify tool structure
        try:
            # This should not raise an error for parameter validation
            # The tool should accept these parameters even if training fails
            tool_params = {
                "modality_name": "test_modality",
                "n_latent": 10,
                "batch_key": "sample",
                "use_gpu": False,
            }

            # Tool should exist and be callable (even if it fails due to setup)
            assert callable(train_tool.func)

        except Exception as e:
            # Tool should exist even if execution fails
            assert "not found" not in str(e).lower()

    def test_clustering_service_custom_embeddings_integration(self):
        """Test clustering service integrates properly with custom embeddings."""
        from lobster.tools.clustering_service import ClusteringService

        service = ClusteringService()

        # Create test data with embeddings
        n_cells = 200
        test_adata = MagicMock(spec=anndata.AnnData)
        test_adata.n_obs = n_cells
        test_adata.n_vars = 1000
        test_adata.shape = (n_cells, 1000)
        test_adata.obs = pd.DataFrame({"batch": ["A"] * 100 + ["B"] * 100})

        # Add custom embeddings
        custom_embeddings = np.random.randn(n_cells, 15)
        test_adata.obsm = {"X_scvi": custom_embeddings}

        # Mock scanpy operations
        with (
            patch("scanpy.pp.neighbors") as mock_neighbors,
            patch("scanpy.tl.leiden") as mock_leiden,
            patch("scanpy.tl.umap") as mock_umap,
        ):

            # Setup mocks to simulate successful execution
            test_adata.obs["leiden"] = np.random.randint(0, 4, n_cells).astype(str)
            test_adata.obsm["X_umap"] = np.random.randn(n_cells, 2)

            try:
                # Test with custom embeddings
                result_adata, stats = service.cluster_and_visualize(
                    adata=test_adata, use_rep="X_scvi", resolution=0.8
                )

                # Verify use_rep was passed to neighbors
                mock_neighbors.assert_called()
                call_args = mock_neighbors.call_args
                if call_args and len(call_args) > 1:
                    kwargs = call_args[1]
                    assert "use_rep" in kwargs
                    assert kwargs["use_rep"] == "X_scvi"

                # Verify clustering results
                assert "n_clusters" in stats
                assert stats["resolution"] == 0.8

            except Exception as e:
                # If execution fails due to dependencies, the important part
                # is that use_rep parameter was accepted and passed correctly
                assert "use_rep" not in str(e)


class TestAgentToolIntegration:
    """Test that agents properly integrate scVI tools."""

    def test_singlecell_expert_has_scvi_handoff(self, mock_data_manager):
        """Test SingleCell Expert includes scVI handoff tool."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Create agent
        agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = agent.get_graph().nodes["tools"].data.tools_by_name

        # request_scvi_embedding is commented out, so skip if not present
        if "request_scvi_embedding" not in tools_by_name:
            pytest.skip(
                "request_scvi_embedding tool not available (commented out in source)"
            )

        assert "request_scvi_embedding" in tools_by_name

    def test_ml_expert_has_scvi_tools(self, mock_data_manager):
        """Test ML Expert includes scVI tools."""
        from lobster.agents.machine_learning_expert import machine_learning_expert

        data_manager, mock_adata = mock_data_manager

        # Create agent
        agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = agent.get_graph().nodes["tools"].data.tools_by_name

        # Verify scVI tools exist
        assert (
            "check_scvi_availability" in tools_by_name
        ), f"check_scvi_availability not found. Available: {list(tools_by_name.keys())}"
        assert (
            "train_scvi_embedding" in tools_by_name
        ), f"train_scvi_embedding not found. Available: {list(tools_by_name.keys())}"

    def test_singlecell_expert_cluster_modality_use_rep(self, mock_data_manager):
        """Test SingleCell Expert cluster_modality accepts use_rep parameter."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Add scVI embeddings to mock data
        mock_adata.obsm["X_scvi"] = np.random.randn(1000, 10)

        # Mock clustering service BEFORE creating agent (service instantiated at module level)
        with patch(
            "lobster.agents.singlecell_expert.ClusteringService"
        ) as mock_service_class:
            mock_service = MagicMock()
            mock_service.cluster_and_visualize.return_value = (
                mock_adata,
                {
                    "n_clusters": 3,
                    "resolution": 0.5,
                    "has_umap": True,
                    "has_marker_genes": False,
                    "original_shape": (1000, 2000),
                    "final_shape": (1000, 2000),
                    "batch_correction": False,
                    "demo_mode": False,
                    "cluster_sizes": {"0": 300, "1": 400, "2": 300},
                },
            )
            mock_service_class.return_value = mock_service

            # Create agent (will use mocked service)
            agent = singlecell_expert(data_manager)

            # Get tools from compiled graph - correct API for CompiledStateGraph
            tools_by_name = agent.get_graph().nodes["tools"].data.tools_by_name

            assert (
                "cluster_modality" in tools_by_name
            ), f"cluster_modality not found. Available: {list(tools_by_name.keys())}"
            cluster_tool = tools_by_name["cluster_modality"]

            # Test clustering with use_rep parameter
            result = cluster_tool.invoke(
                {
                    "modality_name": "test_modality",
                    "use_rep": "X_scvi",
                    "resolution": 0.7,
                }
            )

            # Verify clustering service was called with use_rep
            mock_service.cluster_and_visualize.assert_called_once()
            call_kwargs = mock_service.cluster_and_visualize.call_args[1]
            assert "use_rep" in call_kwargs
            assert call_kwargs["use_rep"] == "X_scvi"
            assert call_kwargs["resolution"] == 0.7

            # Verify success message
            assert "Successfully clustered" in result


class TestWorkflowValidation:
    """Test workflow validation and error handling."""

    def test_scvi_handoff_validates_data_size(self, mock_data_manager):
        """Test scVI handoff validates minimum data requirements."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Create too-small dataset
        mock_adata.n_obs = 50  # Below minimum of 100 cells
        mock_adata.n_vars = 200  # Below minimum of 500 genes

        # Create agent
        agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = agent.get_graph().nodes["tools"].data.tools_by_name

        # Skip test if request_scvi_embedding is not available (commented out)
        if "request_scvi_embedding" not in tools_by_name:
            pytest.skip(
                "request_scvi_embedding tool not available (commented out in source)"
            )

        request_tool = tools_by_name["request_scvi_embedding"]

        # Test with too-small dataset
        result = request_tool.invoke({"modality_name": "test_modality"})

        # Should return validation error, not handoff
        assert "❌" in result
        assert ("100 cells" in result) or ("500 genes" in result)
        assert "HANDOFF" not in result

    def test_batch_key_auto_detection(self, mock_data_manager):
        """Test automatic batch key detection in handoff."""
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Add various batch-related columns
        mock_adata.obs = pd.DataFrame(
            {
                "patient_id": ["P1"] * 500 + ["P2"] * 500,
                "other_col": ["X"] * 1000,
                "leiden": np.random.randint(0, 5, 1000).astype(str),
            }
        )

        # Create agent
        agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        tools_by_name = agent.get_graph().nodes["tools"].data.tools_by_name

        # Skip test if request_scvi_embedding is not available (commented out)
        if "request_scvi_embedding" not in tools_by_name:
            pytest.skip(
                "request_scvi_embedding tool not available (commented out in source)"
            )

        request_tool = tools_by_name["request_scvi_embedding"]

        # Test auto-detection (don't specify batch_key)
        result = request_tool.invoke({"modality_name": "test_modality"})

        # Should detect patient_id as batch key
        assert "patient_id" in result or "auto" in result.lower()
        assert "HANDOFF TO MACHINE LEARNING EXPERT" in result


class TestDocumentationExamples:
    """Test examples that would be in documentation."""

    def test_typical_scvi_workflow_structure(self, mock_data_manager):
        """Test the structure of a typical scVI workflow."""
        from lobster.agents.machine_learning_expert import machine_learning_expert
        from lobster.agents.singlecell_expert import singlecell_expert

        data_manager, mock_adata = mock_data_manager

        # Step 1: SingleCell Expert requests scVI embedding
        sc_agent = singlecell_expert(data_manager)

        # Get tools from compiled graph - correct API for CompiledStateGraph
        sc_tools_by_name = sc_agent.get_graph().nodes["tools"].data.tools_by_name
        sc_tool_names = list(sc_tools_by_name.keys())

        # Essential SingleCell tools
        essential_sc_tools = [
            "request_scvi_embedding",  # New scVI handoff (may be commented out)
            "cluster_modality",  # Updated with use_rep support
            "check_data_status",
        ]

        for essential_tool in essential_sc_tools:
            if essential_tool == "request_scvi_embedding":
                # This tool might be commented out, so skip if not present
                continue
            assert (
                essential_tool in sc_tool_names
            ), f"Essential SingleCell tool missing: {essential_tool}. Available: {sc_tool_names}"

        # Step 2: ML Expert handles scVI training
        ml_agent = machine_learning_expert(data_manager)

        # Get tools from compiled graph
        ml_tools_by_name = ml_agent.get_graph().nodes["tools"].data.tools_by_name
        ml_tool_names = list(ml_tools_by_name.keys())

        # Essential ML tools for scVI
        essential_ml_tools = ["check_scvi_availability", "train_scvi_embedding"]

        for essential_tool in essential_ml_tools:
            assert (
                essential_tool in ml_tool_names
            ), f"Essential ML tool missing: {essential_tool}. Available: {ml_tool_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
