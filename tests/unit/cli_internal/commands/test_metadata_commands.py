"""
Unit tests for metadata commands (extracted from cli.py).

Tests metadata_list, metadata_clear, metadata_clear_exports, and metadata_clear_all
functions using mock client and output adapter.
"""

from unittest.mock import MagicMock, PropertyMock
from pathlib import Path
import pytest

from lobster.cli_internal.commands import (
    metadata_list,
    metadata_clear,
    metadata_clear_exports,
    metadata_clear_all,
)
from lobster.cli_internal.commands.output_adapter import OutputAdapter


class TestMetadataList:
    """Tests for metadata_list function."""

    def test_empty_metadata_store(self):
        """Test metadata_list with empty metadata store."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.metadata_store = {}
        client.data_manager.current_metadata = None
        client.data_manager.workspace_path = Path("/tmp/test_workspace")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_list(client, output)

        # Verify
        assert result == "No metadata available"
        # Should print but no tables
        assert output.print.call_count >= 1
        assert output.print_table.call_count == 0

    def test_with_metadata_store_entries(self):
        """Test metadata_list with populated metadata store."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.metadata_store = {
            "GSE12345": {
                "metadata": {
                    "title": "Test Dataset 1",
                    "samples": {"sample1": {}, "sample2": {}},
                },
                "validation": {"predicted_data_type": "single_cell_rna_seq"},
                "fetch_timestamp": "2024-01-01T00:00:00",
            },
            "GSE67890": {
                "metadata": {
                    "title": "Test Dataset 2",
                    "samples": {"sample1": {}},
                },
                "validation": {"predicted_data_type": "bulk_rna_seq"},
                "fetch_timestamp": "2024-01-02T00:00:00",
            },
        }
        client.data_manager.current_metadata = None
        client.data_manager.workspace_path = Path("/tmp/test_workspace")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_list(client, output)

        # Verify
        assert "2 entries" in result
        # Should print metadata store table
        assert output.print_table.call_count >= 1
        # Check table was called with proper structure
        table_call = output.print_table.call_args[0][0]
        assert "columns" in table_call
        assert "rows" in table_call
        assert len(table_call["rows"]) == 2

    def test_with_current_metadata(self):
        """Test metadata_list with current_metadata."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.metadata_store = {}
        client.data_manager.current_metadata = {
            "dataset_id": "GSE12345",
            "title": "Test Dataset",
            "samples": 100,
        }
        client.data_manager.workspace_path = Path("/tmp/test_workspace")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_list(client, output)

        # Verify
        assert "3 entries" in result  # 3 keys in current_metadata
        assert output.print_table.call_count >= 1


class TestMetadataClear:
    """Tests for metadata_clear function."""

    def test_clear_empty_store(self):
        """Test metadata_clear with empty store and current_metadata."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.metadata_store = {}
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_clear(client, output)

        # Verify - updated message reflects clearing memory + disk
        assert result == "Metadata already empty"
        output.print.assert_called()
        # confirm should NOT be called for empty store
        output.confirm.assert_not_called()

    def test_clear_with_confirmation_yes(self):
        """Test metadata_clear with user confirmation (yes)."""
        # Setup mocks
        client = MagicMock()
        metadata_store = {"GSE12345": {}, "GSE67890": {}}
        client.data_manager.metadata_store = metadata_store
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)
        output.confirm.return_value = True

        # Execute
        result = metadata_clear(client, output)

        # Verify - updated return message reflects memory + disk clearing
        assert "Cleared 2 metadata items (memory + disk)" in result
        assert len(metadata_store) == 0  # Store was cleared
        output.confirm.assert_called_once()

    def test_clear_with_confirmation_no(self):
        """Test metadata_clear with user confirmation (no)."""
        # Setup mocks
        client = MagicMock()
        metadata_store = {"GSE12345": {}, "GSE67890": {}}
        client.data_manager.metadata_store = metadata_store
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)
        output.confirm.return_value = False

        # Execute
        result = metadata_clear(client, output)

        # Verify
        assert result is None  # Operation cancelled
        assert len(metadata_store) == 2  # Store NOT cleared
        output.confirm.assert_called_once()

    def test_no_metadata_store_attribute(self):
        """Test metadata_clear when metadata_store doesn't exist."""
        # Setup mocks
        client = MagicMock()
        del client.data_manager.metadata_store  # Simulate missing attribute
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_clear(client, output)

        # Verify - with no store AND no current_metadata AND no disk files, returns empty message
        assert result == "Metadata already empty"
        output.print.assert_called()


class TestMetadataClearExports:
    """Tests for metadata_clear_exports function."""

    def test_clear_exports_no_files(self):
        """Test metadata_clear_exports with no export files."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_clear_exports(client, output)

        # Verify
        assert result == "No export files to clear"
        output.confirm.assert_not_called()


class TestMetadataClearAll:
    """Tests for metadata_clear_all function."""

    def test_clear_all_nothing_to_clear(self):
        """Test metadata_clear_all with nothing to clear."""
        # Setup mocks
        client = MagicMock()
        client.data_manager.metadata_store = {}
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)

        # Execute
        result = metadata_clear_all(client, output)

        # Verify
        assert result == "Nothing to clear"
        output.confirm.assert_not_called()

    def test_clear_all_with_confirmation_yes(self):
        """Test metadata_clear_all with user confirmation (yes)."""
        # Setup mocks
        client = MagicMock()
        metadata_store = {"GSE12345": {}}
        current_metadata = {"key": "value"}
        client.data_manager.metadata_store = metadata_store
        client.data_manager.current_metadata = current_metadata
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)
        output.confirm.return_value = True

        # Execute
        result = metadata_clear_all(client, output)

        # Verify
        assert "Cleared all metadata" in result
        assert len(metadata_store) == 0  # Store was cleared
        assert len(current_metadata) == 0  # Current metadata was cleared
        output.confirm.assert_called_once()

    def test_clear_all_with_confirmation_no(self):
        """Test metadata_clear_all with user confirmation (no)."""
        # Setup mocks
        client = MagicMock()
        metadata_store = {"GSE12345": {}}
        client.data_manager.metadata_store = metadata_store
        client.data_manager.current_metadata = {}
        client.data_manager.workspace_path = Path("/tmp/nonexistent")
        output = MagicMock(spec=OutputAdapter)
        output.confirm.return_value = False

        # Execute
        result = metadata_clear_all(client, output)

        # Verify
        assert result is None  # Operation cancelled
        assert len(metadata_store) == 1  # Store NOT cleared
        output.confirm.assert_called_once()
