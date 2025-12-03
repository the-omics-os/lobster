"""
Basic integration test to verify tools work with data_expert agent.
"""

from pathlib import Path
from unittest.mock import Mock

import anndata
import numpy as np
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CustomCodeExecutionService, SDKDelegationService


class TestBasicIntegration:
    """Test that services can be instantiated and used."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def data_manager(self, workspace):
        """Create DataManagerV2."""
        dm = DataManagerV2(workspace_path=workspace)

        # Add a test modality
        adata = anndata.AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
        adata.obs["sample"] = ["s1", "s2"]
        adata.var["gene"] = ["g1", "g2", "g3"]
        dm.modalities["test_data"] = adata

        return dm

    def test_custom_code_service_initialization(self, data_manager):
        """Test that CustomCodeExecutionService can be created."""
        service = CustomCodeExecutionService(data_manager)
        assert service is not None
        assert service.data_manager == data_manager

    def test_custom_code_simple_execution(self, data_manager):
        """Test simple code execution."""
        service = CustomCodeExecutionService(data_manager)

        result, stats, ir = service.execute(code="result = 2 + 2", persist=False)

        assert result == 4
        assert stats["success"] is True
        assert ir.operation == "custom_code_execution"

    def test_custom_code_with_modality(self, data_manager):
        """Test code execution with modality access."""
        service = CustomCodeExecutionService(data_manager)

        result, stats, ir = service.execute(
            code="result = adata.n_obs", modality_name="test_data", persist=False
        )

        assert result == 2  # 2 observations
        assert stats["success"] is True

    def test_sdk_delegation_service_initialization_fails_gracefully(self, data_manager):
        """Test that SDK service fails gracefully when SDK not available."""
        # SDK might not be available in test environment
        try:
            service = SDKDelegationService(data_manager)
            # If it succeeds, that's fine
            assert service is not None
        except Exception as e:
            # Should raise SDKDelegationError with helpful message
            assert "SDK not available" in str(e) or "SDK" in str(type(e).__name__)

    def test_data_expert_can_be_imported(self):
        """Test that data_expert module can be imported with new tools."""
        try:
            from lobster.agents.data_expert import data_expert

            assert data_expert is not None
        except ImportError as e:
            pytest.fail(f"Failed to import data_expert: {e}")
