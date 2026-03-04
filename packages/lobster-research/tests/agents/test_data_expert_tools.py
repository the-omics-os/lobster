"""Test that data_expert factory includes filesystem tools."""
import pytest
from unittest.mock import MagicMock, patch


class TestDataExpertToolWiring:
    """Verify filesystem tools are wired into data_expert agent."""

    @patch("lobster.agents.data_expert.data_expert.create_llm")
    @patch("lobster.agents.data_expert.data_expert.get_settings")
    def test_filesystem_tools_included(self, mock_settings, mock_create_llm, tmp_path):
        """data_expert factory should include 6 filesystem tools."""
        from lobster.core.runtime.data_manager import DataManagerV2

        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.get_agent_llm_params.return_value = {}
        mock_llm = MagicMock()
        mock_llm.with_config.return_value = mock_llm
        mock_create_llm.return_value = mock_llm

        # Create minimal DataManagerV2
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Patch create_react_agent to capture tools
        captured_tools = {}

        def capture_agent(**kwargs):
            captured_tools["tools"] = kwargs.get("tools", [])
            return MagicMock()

        with patch(
            "lobster.agents.data_expert.data_expert.create_react_agent",
            side_effect=capture_agent,
        ):
            from lobster.agents.data_expert.data_expert import data_expert

            data_expert(data_manager=dm, workspace_path=tmp_path)

        tool_names = {t.name for t in captured_tools["tools"]}
        # Verify filesystem tools are present
        assert "list_files" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "glob_files" in tool_names
        assert "grep_files" in tool_names
        assert "shell_execute" in tool_names
        # Verify existing tools still present
        assert "execute_download_from_queue" in tool_names
        assert "load_modality" in tool_names
