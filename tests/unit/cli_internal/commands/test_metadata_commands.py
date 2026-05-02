"""
Unit tests for metadata commands (extracted from cli.py).

Tests metadata_list, metadata_clear, metadata_clear_exports, and metadata_clear_all
functions using mock client and output adapter.
"""

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from lobster.cli_internal.commands import (
    metadata_clear,
    metadata_clear_all,
    metadata_clear_exports,
    metadata_list,
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
        # Should render blocks (no tables — only section/hint blocks emitted)
        assert output.render_blocks.call_count >= 1
        # No table blocks should appear in any render_blocks call
        all_blocks = [
            block
            for call in output.render_blocks.call_args_list
            for block in call[0][0]
        ]
        assert not any(b.kind == "table" for b in all_blocks)

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
        # Should render at least one call containing a table block
        assert output.render_blocks.call_count >= 1
        all_blocks = [
            block
            for call in output.render_blocks.call_args_list
            for block in call[0][0]
        ]
        table_blocks = [b for b in all_blocks if b.kind == "table"]
        assert len(table_blocks) >= 1
        # Check the metadata-store table has the right structure and row count
        metadata_table = table_blocks[0]
        assert "columns" in metadata_table.data
        assert "rows" in metadata_table.data
        assert len(metadata_table.data["rows"]) == 2

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
        # Should render blocks; current_metadata is shown as a kv block (not table kind)
        assert output.render_blocks.call_count >= 1
        all_blocks = [
            block
            for call in output.render_blocks.call_args_list
            for block in call[0][0]
        ]
        assert any(b.kind in ("table", "kv") for b in all_blocks)


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
        output.render_blocks.assert_called()
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
        output.render_blocks.assert_called()


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
