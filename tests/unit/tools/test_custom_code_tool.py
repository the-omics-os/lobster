"""
Unit tests for shared custom code execution tool factory.

Tests the factory pattern for creating agent-specific execute_custom_code tools
with unified signature and optional post-processing hooks.
"""

from unittest.mock import MagicMock, call

import pytest

from lobster.core.analysis_ir import AnalysisStep
from lobster.tools.custom_code_tool import (
    PostProcessor,
    create_execute_custom_code_tool,
    metadata_store_post_processor,
)


class TestCreateExecuteCustomCodeTool:
    """Test factory function for creating execute_custom_code tools."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create mock DataManagerV2."""
        dm = MagicMock()
        dm.workspace_path = tmp_path
        dm.metadata_store = {}
        return dm

    @pytest.fixture
    def mock_service(self):
        """Create mock CustomCodeExecutionService."""
        service = MagicMock()
        mock_ir = MagicMock(spec=AnalysisStep)
        service.execute.return_value = (
            42,  # result
            {"success": True, "duration_seconds": 0.1, "warnings": []},  # stats
            mock_ir,  # ir
        )
        return service

    def test_factory_returns_callable(self, mock_data_manager, mock_service):
        """Factory returns a LangChain StructuredTool with invoke interface."""
        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        # LangChain StructuredTool has invoke() method, not directly callable via __call__
        assert hasattr(tool, "invoke")  # LangChain tool interface
        assert hasattr(tool, "name")
        assert tool.name == "execute_custom_code"

    def test_factory_with_agent_name(self, mock_data_manager, mock_service):
        """Factory accepts agent_name parameter."""
        import json

        tool = create_execute_custom_code_tool(
            mock_data_manager, mock_service, agent_name="test_agent"
        )
        result = tool.invoke({"python_code": "x = 1"})
        result_obj = json.loads(result)
        assert result_obj["success"] is True

        # Verify agent_name logged
        call_args = mock_data_manager.log_tool_usage.call_args
        assert call_args.kwargs["parameters"]["agent"] == "test_agent"

    def test_mutual_exclusivity_validation(self, mock_data_manager, mock_service):
        """Cannot specify both modality_name and workspace_key."""
        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        result = tool.invoke(
            {
                "python_code": "x = 1",
                "modality_name": "test_modality",
                "workspace_key": "test_key",
            }
        )

        # Mutual exclusivity returns a plain error string (not JSON)
        assert "Cannot specify both" in result
        assert "modality_name" in result
        assert "workspace_key" in result
        # Service should NOT be called
        mock_service.execute.assert_not_called()

    def test_workspace_key_converted_to_list(self, mock_data_manager, mock_service):
        """workspace_key string is converted to list for service."""
        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        tool.invoke({"python_code": "x = 1", "workspace_key": "my_key"})

        call_args = mock_service.execute.call_args
        assert call_args.kwargs["workspace_keys"] == ["my_key"]
        assert call_args.kwargs["modality_name"] is None

    def test_modality_name_passed_through(self, mock_data_manager, mock_service):
        """modality_name passed directly to service."""
        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        tool.invoke({"python_code": "x = 1", "modality_name": "my_modality"})

        call_args = mock_service.execute.call_args
        assert call_args.kwargs["modality_name"] == "my_modality"
        assert call_args.kwargs["workspace_keys"] is None

    def test_neither_parameter_passes_none(self, mock_data_manager, mock_service):
        """When neither parameter specified, both are None."""
        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        tool.invoke({"python_code": "x = 1"})

        call_args = mock_service.execute.call_args
        assert call_args.kwargs["modality_name"] is None
        assert call_args.kwargs["workspace_keys"] is None

    def test_provenance_logging(self, mock_data_manager, mock_service):
        """Tool logs to provenance with correct parameters."""
        tool = create_execute_custom_code_tool(
            mock_data_manager, mock_service, agent_name="test_agent"
        )
        tool.invoke(
            {
                "python_code": "x = 1",
                "description": "Test execution",
                "persist": True,
            }
        )

        mock_data_manager.log_tool_usage.assert_called_once()
        call_args = mock_data_manager.log_tool_usage.call_args

        assert call_args.kwargs["tool_name"] == "execute_custom_code"
        assert call_args.kwargs["parameters"]["description"] == "Test execution"
        assert call_args.kwargs["parameters"]["persist"] is True
        assert call_args.kwargs["parameters"]["agent"] == "test_agent"
        assert call_args.kwargs["ir"] is not None

    def test_post_processor_called_on_success(self, mock_data_manager, mock_service):
        """Post-processor called when execution succeeds."""
        post_proc = MagicMock(return_value="Post-processed message!")
        tool = create_execute_custom_code_tool(
            mock_data_manager, mock_service, post_processor=post_proc
        )

        result = tool.invoke({"python_code": "x = 1", "workspace_key": "test_key"})

        # Post-processor should be called
        post_proc.assert_called_once()
        call_args = post_proc.call_args
        assert call_args[0][0] == 42  # result
        assert call_args[0][2] == mock_data_manager  # data_manager
        assert call_args[0][3] == "test_key"  # workspace_key
        assert call_args[0][4] is None  # modality_name

        # Message should be in response
        import json

        result_obj = json.loads(result)
        assert result_obj["success"] is True
        assert result_obj.get("post_processor_message") == "Post-processed message!"

    def test_post_processor_not_called_on_failure(
        self, mock_data_manager, mock_service
    ):
        """Post-processor NOT called when execution fails."""
        mock_service.execute.return_value = (
            None,
            {"success": False, "duration_seconds": 0.1},
            MagicMock(),
        )
        post_proc = MagicMock()
        tool = create_execute_custom_code_tool(
            mock_data_manager, mock_service, post_processor=post_proc
        )

        tool.invoke({"python_code": "x = 1"})

        # Post-processor should NOT be called
        post_proc.assert_not_called()

    def test_post_processor_error_handled(self, mock_data_manager, mock_service):
        """Post-processor errors are caught and logged."""
        post_proc = MagicMock(side_effect=Exception("Post-processor failed"))
        tool = create_execute_custom_code_tool(
            mock_data_manager, mock_service, post_processor=post_proc
        )

        result = tool.invoke({"python_code": "x = 1"})

        # Execution should still succeed (JSON response with success=true)
        import json

        result_obj = json.loads(result)
        assert result_obj["success"] is True
        # Warning about post-processor failure in post_processor_message
        assert "Post-processing failed" in result_obj.get("post_processor_message", "")

    def test_response_formatting(self, mock_data_manager, mock_service):
        """Response includes all expected sections as structured JSON."""
        import json

        mock_service.execute.return_value = (
            {"sample_count": 100},  # result
            {
                "success": True,
                "duration_seconds": 2.5,
                "warnings": ["Import not in standard stack"],
                "stdout_preview": "Processing...",
                "result_type": "dict",
            },
            MagicMock(),
        )

        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        result = tool.invoke({"python_code": "x = 1", "persist": False})

        result_obj = json.loads(result)
        assert result_obj["success"] is True
        assert result_obj["duration_seconds"] == 2.5
        assert "Import not in standard stack" in result_obj["warnings"]
        assert result_obj["result"]["sample_count"] == 100
        assert result_obj["result_type"] == "dict"
        assert result_obj["stdout"] == "Processing..."
        assert result_obj["persisted"] is False

    def test_result_truncation(self, mock_data_manager, mock_service):
        """Large results are truncated in response."""
        large_result = "x" * 1000
        mock_service.execute.return_value = (
            large_result,
            {"success": True, "duration_seconds": 0.1},
            MagicMock(),
        )

        tool = create_execute_custom_code_tool(mock_data_manager, mock_service)
        result = tool.invoke({"python_code": "x = 1"})

        assert "truncated" in result


class TestMetadataStorePostProcessor:
    """Test metadata_assistant-specific post-processor."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManagerV2 with metadata_store."""
        dm = MagicMock()
        dm.metadata_store = {"existing_key": {"samples": []}}
        return dm

    def test_in_place_update(self, mock_data_manager):
        """Updates existing metadata_store key in-place."""
        result = {"samples": [{"id": 1}, {"id": 2}]}
        stats = {}

        msg = metadata_store_post_processor(
            result, stats, mock_data_manager, "existing_key", None
        )

        # Should update existing key
        assert (
            mock_data_manager.metadata_store["existing_key"]["samples"]
            == result["samples"]
        )
        assert msg is not None
        assert "Persisted 2 samples" in msg
        assert "existing_key" in msg

    def test_new_key_creation(self, mock_data_manager):
        """Creates new key with output_key pattern."""
        result = {
            "samples": [{"id": 1}],
            "output_key": "filtered_samples",
            "filter_criteria": "body_site",
            "stats": {"count": 1},
        }

        msg = metadata_store_post_processor(result, {}, mock_data_manager, None, None)

        # Should create new key
        assert "filtered_samples" in mock_data_manager.metadata_store
        assert (
            mock_data_manager.metadata_store["filtered_samples"]["samples"]
            == result["samples"]
        )
        assert (
            mock_data_manager.metadata_store["filtered_samples"]["filter_criteria"]
            == "body_site"
        )

        # Message should include export instructions
        assert "Created metadata_store" in msg
        assert "filtered_samples" in msg
        assert "write_to_workspace" in msg

    def test_no_action_for_non_dict(self, mock_data_manager):
        """Returns None for non-dict results."""
        msg = metadata_store_post_processor(42, {}, mock_data_manager, None, None)
        assert msg is None

    def test_no_action_without_samples_key(self, mock_data_manager):
        """Returns None when result dict lacks 'samples' key."""
        result = {"other_key": "value"}
        msg = metadata_store_post_processor(
            result, {}, mock_data_manager, "existing_key", None
        )
        assert msg is None

    def test_workspace_key_not_in_store(self, mock_data_manager):
        """No action when workspace_key not in metadata_store."""
        result = {"samples": [{"id": 1}]}
        msg = metadata_store_post_processor(
            result, {}, mock_data_manager, "nonexistent_key", None
        )
        assert msg is None

    def test_output_key_without_samples(self, mock_data_manager):
        """No action when output_key present but samples missing."""
        result = {"output_key": "new_key", "data": "value"}
        msg = metadata_store_post_processor(result, {}, mock_data_manager, None, None)
        assert msg is None


class TestIntegration:
    """Integration tests combining factory and post-processor."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create mock DataManagerV2."""
        dm = MagicMock()
        dm.workspace_path = tmp_path
        dm.metadata_store = {"existing_samples": {"samples": []}}
        return dm

    @pytest.fixture
    def mock_service(self):
        """Create mock service that returns metadata result."""
        service = MagicMock()
        service.execute.return_value = (
            {"samples": [{"id": 1}, {"id": 2}], "output_key": "filtered"},
            {"success": True, "duration_seconds": 0.5},
            MagicMock(),
        )
        return service

    def test_metadata_assistant_workflow(self, mock_data_manager, mock_service):
        """Test complete metadata_assistant workflow."""
        tool = create_execute_custom_code_tool(
            data_manager=mock_data_manager,
            custom_code_service=mock_service,
            agent_name="metadata_assistant",
            post_processor=metadata_store_post_processor,
        )

        result = tool.invoke(
            {
                "python_code": "samples = [...]; result = {'samples': samples, 'output_key': 'filtered'}",
                "workspace_key": "source_key",
            }
        )

        # Should create new metadata_store entry
        assert "filtered" in mock_data_manager.metadata_store

        # Response should be structured JSON with success
        import json

        result_obj = json.loads(result)
        assert result_obj["success"] is True
        # Post-processor message should mention the created store entry
        assert result_obj.get("post_processor_message") is not None

    def test_data_expert_workflow(self, mock_data_manager, mock_service):
        """Test complete data_expert workflow (no post-processing)."""
        mock_service.execute.return_value = (
            np_mean_result := 3.5,
            {"success": True, "duration_seconds": 0.2, "result_type": "float"},
            MagicMock(),
        )

        tool = create_execute_custom_code_tool(
            data_manager=mock_data_manager,
            custom_code_service=mock_service,
            agent_name="data_expert",
            post_processor=None,  # No post-processing for data_expert
        )

        result = tool.invoke(
            {
                "python_code": "result = adata.obs['n_genes'].mean()",
                "modality_name": "my_modality",
            }
        )

        # Should NOT modify metadata_store
        assert "filtered" not in mock_data_manager.metadata_store

        # Response should be structured JSON with execution result
        import json

        result_obj = json.loads(result)
        assert result_obj["success"] is True
        assert result_obj["result"] == 3.5
        assert result_obj["post_processor_message"] is None  # No post-processing
