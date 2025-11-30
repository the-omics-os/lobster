"""
Unit tests for ExecutionContextBuilder.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.execution_context_builder import ExecutionContextBuilder


class TestExecutionContextBuilder:
    """Test suite for ExecutionContextBuilder."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace with sample files."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Create sample CSV
        df = pd.DataFrame({'sample': ['A', 'B', 'C'], 'value': [1, 2, 3]})
        df.to_csv(workspace / "test_metadata.csv", index=False)

        # Create sample CSV with special characters in name
        df2 = pd.DataFrame({'x': [10, 20]})
        df2.to_csv(workspace / "sample-data.csv", index=False)

        # Create sample JSON
        with open(workspace / "config.json", 'w') as f:
            json.dump({'param1': 42, 'param2': 'test'}, f)

        # Create hidden JSON (should be skipped)
        with open(workspace / ".session.json", 'w') as f:
            json.dump({'hidden': True}, f)

        # Create download queue JSONL
        with open(workspace / "download_queue.jsonl", 'w') as f:
            f.write(json.dumps({'entry_id': 'entry1', 'status': 'PENDING'}) + '\n')
            f.write(json.dumps({'entry_id': 'entry2', 'status': 'COMPLETED'}) + '\n')

        # Create publication queue JSONL
        with open(workspace / "publication_queue.jsonl", 'w') as f:
            f.write(json.dumps({'pmid': '12345', 'status': 'PENDING'}) + '\n')

        return workspace

    @pytest.fixture
    def mock_data_manager(self, workspace):
        """Create mock DataManagerV2 with test workspace."""
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = workspace
        dm.list_modalities.return_value = ['modality1', 'modality2']
        return dm

    @pytest.fixture
    def builder(self, mock_data_manager):
        """Create builder instance."""
        return ExecutionContextBuilder(mock_data_manager)

    def test_initialization(self, builder, mock_data_manager, workspace):
        """Test builder initialization."""
        assert builder.data_manager == mock_data_manager
        assert builder.workspace_path == workspace

    def test_build_context_basic(self, builder):
        """Test basic context building without modality."""
        context = builder.build_context(
            modality_name=None,
            load_workspace_files=False
        )

        # Check core keys
        assert 'data_manager' in context
        assert 'workspace_path' in context
        assert 'modalities' in context
        assert 'pd' in context
        assert 'Path' in context

        # Check modalities list
        assert context['modalities'] == ['modality1', 'modality2']

        # Check no adata loaded
        assert 'adata' not in context

    def test_build_context_with_modality(self, builder, mock_data_manager):
        """Test context building with specific modality."""
        # Mock modality loading
        import anndata
        import numpy as np
        mock_adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        mock_data_manager.get_modality.return_value = mock_adata

        context = builder.build_context(
            modality_name='modality1',
            load_workspace_files=False
        )

        # Check adata is loaded
        assert 'adata' in context
        assert context['adata'] is mock_adata  # Use identity check, not equality
        mock_data_manager.get_modality.assert_called_once_with('modality1')

    def test_build_context_modality_not_found(self, builder, mock_data_manager):
        """Test context building with non-existent modality."""
        context = builder.build_context(
            modality_name='nonexistent',
            load_workspace_files=False
        )

        # adata should not be in context
        assert 'adata' not in context
        # Should not have tried to load
        mock_data_manager.get_modality.assert_not_called()

    def test_build_context_with_workspace_files(self, builder):
        """Test context building with workspace file loading."""
        context = builder.build_context(
            modality_name=None,
            load_workspace_files=True
        )

        # Check CSV files loaded
        assert 'test_metadata' in context
        assert isinstance(context['test_metadata'], pd.DataFrame)
        assert list(context['test_metadata'].columns) == ['sample', 'value']

        # Check CSV with special characters sanitized
        assert 'sample_data' in context
        assert isinstance(context['sample_data'], pd.DataFrame)

        # Check JSON files loaded
        assert 'config' in context
        assert context['config'] == {'param1': 42, 'param2': 'test'}

        # Check hidden JSON not loaded
        assert 'session' not in context
        assert '.session' not in context

        # Check queue files loaded
        assert 'download_queue' in context
        assert isinstance(context['download_queue'], list)
        assert len(context['download_queue']) == 2
        assert context['download_queue'][0]['entry_id'] == 'entry1'

        assert 'publication_queue' in context
        assert isinstance(context['publication_queue'], list)
        assert len(context['publication_queue']) == 1

    def test_load_workspace_files(self, builder):
        """Test _load_workspace_files method."""
        csv_data, json_data = builder._load_workspace_files()

        # Check CSV data
        assert 'test_metadata' in csv_data
        assert 'sample_data' in csv_data

        # Check JSON data
        assert 'config' in json_data
        assert 'download_queue' in json_data
        assert 'publication_queue' in json_data

        # Check hidden file not loaded
        assert 'session' not in json_data

    def test_sanitize_filename_basic(self, builder):
        """Test filename sanitization - basic cases."""
        assert builder._sanitize_filename("valid_name") == "valid_name"
        assert builder._sanitize_filename("geo_gse12345") == "geo_gse12345"

    def test_sanitize_filename_hyphens(self, builder):
        """Test filename sanitization - hyphens."""
        assert builder._sanitize_filename("sample-data") == "sample_data"
        assert builder._sanitize_filename("my-test-file") == "my_test_file"

    def test_sanitize_filename_spaces(self, builder):
        """Test filename sanitization - spaces."""
        assert builder._sanitize_filename("my data") == "my_data"
        assert builder._sanitize_filename("test file 2024") == "test_file_2024"

    def test_sanitize_filename_starts_with_digit(self, builder):
        """Test filename sanitization - starts with digit."""
        assert builder._sanitize_filename("2024_results") == "data_2024_results"
        assert builder._sanitize_filename("123abc") == "data_123abc"

    def test_sanitize_filename_special_characters(self, builder):
        """Test filename sanitization - special characters."""
        assert builder._sanitize_filename("file!name@test") == "filenametest"
        assert builder._sanitize_filename("data(2024)") == "data2024"

    def test_sanitize_filename_empty(self, builder):
        """Test filename sanitization - empty or all-special."""
        assert builder._sanitize_filename("!!!") == "data"
        assert builder._sanitize_filename("") == "data"

    def test_context_contains_pd_and_path(self, builder):
        """Test that convenience imports are available."""
        context = builder.build_context(load_workspace_files=False)

        assert context['pd'] == pd
        assert context['Path'] == Path

    def test_empty_workspace(self, tmp_path, mock_data_manager):
        """Test context building with empty workspace."""
        empty_workspace = tmp_path / "empty"
        empty_workspace.mkdir()
        mock_data_manager.workspace_path = empty_workspace

        builder = ExecutionContextBuilder(mock_data_manager)
        context = builder.build_context(load_workspace_files=True)

        # Should still have core keys
        assert 'data_manager' in context
        assert 'workspace_path' in context
        assert 'modalities' in context

        # But no loaded files
        assert 'test_metadata' not in context
        assert 'config' not in context

    def test_malformed_csv_handling(self, tmp_path, mock_data_manager):
        """Test handling of malformed CSV file."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_data_manager.workspace_path = workspace

        # Create malformed CSV
        with open(workspace / "bad.csv", 'w') as f:
            f.write("invalid,csv\ndata")

        builder = ExecutionContextBuilder(mock_data_manager)

        # Should not crash
        csv_data, _ = builder._load_workspace_files()

        # Malformed file might be loaded with errors or skipped
        # Either way, should not raise exception

    def test_malformed_json_handling(self, tmp_path, mock_data_manager):
        """Test handling of malformed JSON file."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_data_manager.workspace_path = workspace

        # Create malformed JSON
        with open(workspace / "bad.json", 'w') as f:
            f.write("{invalid json")

        builder = ExecutionContextBuilder(mock_data_manager)

        # Should not crash
        _, json_data = builder._load_workspace_files()

        # Malformed file should be skipped
        assert 'bad' not in json_data

    def test_empty_jsonl_lines_skipped(self, tmp_path, mock_data_manager):
        """Test that empty lines in JSONL are skipped."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        mock_data_manager.workspace_path = workspace

        # Create JSONL with empty lines
        with open(workspace / "download_queue.jsonl", 'w') as f:
            f.write(json.dumps({'id': 1}) + '\n')
            f.write('\n')  # Empty line
            f.write(json.dumps({'id': 2}) + '\n')

        builder = ExecutionContextBuilder(mock_data_manager)
        _, json_data = builder._load_workspace_files()

        # Should load only 2 entries (empty line skipped)
        assert len(json_data['download_queue']) == 2
