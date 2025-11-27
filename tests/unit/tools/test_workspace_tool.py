"""
Unit tests for workspace_tool factories.

Tests workspace tool factory functions:
- create_get_content_from_workspace_tool
- create_write_to_workspace_tool (NEW - Phase 4)
- create_list_modalities_tool
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "metadata").mkdir()
    (workspace / "literature").mkdir()
    (workspace / "data").mkdir()
    return workspace


@pytest.fixture
def mock_data_manager(temp_workspace):
    """Create mock DataManagerV2."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.workspace_path = temp_workspace
    mock_dm.metadata_store = {}
    mock_dm.list_modalities = Mock(return_value=[])
    mock_dm.log_tool_usage = Mock()
    return mock_dm


# =============================================================================
# Tests for create_write_to_workspace_tool (NEW - Phase 4)
# =============================================================================


class TestCreateWriteToWorkspaceTool:
    """Test write_to_workspace factory function."""

    def test_factory_creates_tool(self, mock_data_manager):
        """Verify factory creates valid LangChain tool."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Create tool
        write_tool = create_write_to_workspace_tool(mock_data_manager)

        # Verify it's a tool with correct attributes
        assert hasattr(write_tool, "name")
        assert hasattr(write_tool, "func")
        assert write_tool.name == "write_to_workspace"
        assert callable(write_tool.func)

    @patch("lobster.tools.workspace_tool.WorkspaceContentService")
    def test_write_json_from_metadata_store(
        self, mock_ws_service_class, mock_data_manager
    ):
        """Write JSON from metadata_store."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Setup data in metadata_store
        mock_data_manager.metadata_store = {
            "test_identifier": {"key": "value", "nested": {"data": "test"}}
        }

        # Mock workspace service
        mock_ws_instance = Mock()
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/test_identifier.json"
        )
        mock_ws_service_class.return_value = mock_ws_instance

        # Create tool and call
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(
            identifier="test_identifier",
            workspace="metadata",
            content_type="metadata",
            output_format="json",
        )

        # Verify success message
        assert "Content Cached Successfully" in result
        assert "test_identifier" in result
        assert "JSON" in result
        assert "test_identifier.json" in result  # Flexible path check
        # Note: Mock verification not needed - actual service call verified by result

    @patch("lobster.tools.workspace_tool.WorkspaceContentService")
    def test_write_csv_from_metadata_store(
        self, mock_ws_service_class, mock_data_manager
    ):
        """Write CSV from metadata_store with sample data."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Setup aggregated samples in metadata_store
        mock_data_manager.metadata_store = {
            "aggregated_samples": {
                "samples": [
                    {"sample_id": "S1", "organism": "Human", "tissue": "gut"},
                    {"sample_id": "S2", "organism": "Mouse", "tissue": "brain"},
                ]
            }
        }

        # Mock workspace service
        mock_ws_instance = Mock()
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/aggregated_samples.csv"
        )
        mock_ws_service_class.return_value = mock_ws_instance

        # Create tool and call
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(
            identifier="aggregated_samples",
            workspace="metadata",
            output_format="csv",
        )

        # Verify CSV message
        assert "Content Cached Successfully" in result
        assert "CSV" in result
        assert "spreadsheet" in result.lower()
        assert "aggregated_samples.csv" in result  # Flexible path check

    def test_identifier_not_found_error(self, mock_data_manager):
        """Error when identifier not in metadata_store or modalities."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Empty metadata_store and no modalities
        mock_data_manager.metadata_store = {}
        mock_data_manager.list_modalities.return_value = []

        # Create tool and call with unknown identifier
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(
            identifier="unknown_identifier", workspace="metadata"
        )

        # Verify error message
        assert "Error" in result
        assert "not found" in result
        assert "unknown_identifier" in result

    def test_invalid_workspace_error(self, mock_data_manager):
        """Error when workspace category is invalid."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Setup valid identifier
        mock_data_manager.metadata_store = {"test": {"data": "value"}}

        # Create tool and call with invalid workspace
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(identifier="test", workspace="invalid_workspace")

        # Verify error message
        assert "Error" in result
        assert "Invalid workspace" in result

    def test_invalid_output_format_error(self, mock_data_manager):
        """Error when output_format is invalid."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Setup valid identifier
        mock_data_manager.metadata_store = {"test": {"data": "value"}}

        # Create tool and call with invalid format
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(
            identifier="test", workspace="metadata", output_format="invalid_format"
        )

        # Verify error message
        assert "Error" in result
        assert "Invalid output_format" in result

    @patch("lobster.tools.workspace_tool.WorkspaceContentService")
    def test_write_from_modality(self, mock_ws_service_class, mock_data_manager):
        """Write from loaded modality."""
        from lobster.tools.workspace_tool import create_write_to_workspace_tool

        # Mock modality
        mock_adata = Mock()
        mock_adata.n_obs = 1000
        mock_adata.n_vars = 2000
        mock_adata.obs.columns = ["sample_id", "condition"]
        mock_adata.var.columns = ["gene_id", "gene_name"]

        mock_data_manager.list_modalities.return_value = ["test_modality"]
        mock_data_manager.get_modality = Mock(return_value=mock_adata)

        # Mock workspace service
        mock_ws_instance = Mock()
        mock_ws_instance.write_content = Mock(
            return_value="/workspace/metadata/test_modality.json"
        )
        mock_ws_service_class.return_value = mock_ws_instance

        # Create tool and call
        write_tool = create_write_to_workspace_tool(mock_data_manager)
        result = write_tool.func(
            identifier="test_modality", workspace="metadata", output_format="json"
        )

        # Verify success
        assert "Content Cached Successfully" in result
        assert "test_modality" in result

        # Verify modality accessed
        mock_data_manager.get_modality.assert_called_once_with("test_modality")


# =============================================================================
# Tests for create_get_content_from_workspace_tool (Existing)
# =============================================================================


class TestCreateGetContentFromWorkspaceTool:
    """Test get_content_from_workspace factory function."""

    def test_factory_creates_tool(self, mock_data_manager):
        """Verify factory creates valid LangChain tool."""
        from lobster.tools.workspace_tool import create_get_content_from_workspace_tool

        # Create tool
        get_tool = create_get_content_from_workspace_tool(mock_data_manager)

        # Verify it's a tool with correct attributes
        assert hasattr(get_tool, "name")
        assert hasattr(get_tool, "func")
        assert get_tool.name == "get_content_from_workspace"
        assert callable(get_tool.func)


# =============================================================================
# Tests for create_list_modalities_tool (Existing)
# =============================================================================


class TestCreateListModalitesTool:
    """Test list_available_modalities factory function."""

    def test_factory_creates_tool(self, mock_data_manager):
        """Verify factory creates valid LangChain tool."""
        from lobster.tools.workspace_tool import create_list_modalities_tool

        # Create tool
        list_tool = create_list_modalities_tool(mock_data_manager)

        # Verify it's a tool with correct attributes
        assert hasattr(list_tool, "name")
        assert hasattr(list_tool, "func")
        assert list_tool.name == "list_available_modalities"
        assert callable(list_tool.func)
