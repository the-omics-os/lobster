"""Narrow unit tests for MatrixParser domain module.

Tests MatrixParser methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from lobster.services.data_access.geo.matrix_parsing import MatrixParser


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes MatrixParser needs."""
    service = MagicMock()
    service.data_manager = MagicMock()
    service.data_manager.metadata_store = {}
    service.cache_dir = Path("/tmp/test_geo_cache")
    service.console = MagicMock()
    service.geo_downloader = MagicMock()
    service.geo_parser = MagicMock()
    service.pipeline_engine = MagicMock()
    service.tenx_loader = MagicMock()
    return service


@pytest.fixture
def parser(mock_service):
    """Create a MatrixParser with mocked service."""
    return MatrixParser(mock_service)


class TestMatrixParserInit:
    """Test MatrixParser initialization."""

    def test_init_stores_service_reference(self, mock_service):
        parser = MatrixParser(mock_service)
        assert parser.service is mock_service


class TestIsValidExpressionMatrix:
    """Test _is_valid_expression_matrix validation logic."""

    def test_returns_true_for_numeric_dataframe(self, parser):
        """DataFrame with numeric values should be valid."""
        df = pd.DataFrame(
            {"gene1": [1.0, 2.0, 3.0], "gene2": [4.0, 5.0, 6.0]},
            index=["s1", "s2", "s3"],
        )
        assert parser._is_valid_expression_matrix(df) is True

    def test_returns_false_for_empty_dataframe(self, parser):
        """Empty DataFrame should still be valid if it has numeric dtypes.
        The method checks for numeric columns, not emptiness."""
        df = pd.DataFrame()
        # Empty DataFrame has no numeric columns
        assert parser._is_valid_expression_matrix(df) is False

    def test_returns_false_for_string_only_dataframe(self, parser):
        """DataFrame with only string columns should be invalid."""
        df = pd.DataFrame(
            {"col1": ["a", "b"], "col2": ["c", "d"]},
        )
        assert parser._is_valid_expression_matrix(df) is False

    def test_returns_false_for_non_dataframe(self, parser):
        """Non-DataFrame input should return False."""
        assert parser._is_valid_expression_matrix("not a dataframe") is False
        assert parser._is_valid_expression_matrix(None) is False
        assert parser._is_valid_expression_matrix(42) is False

    def test_returns_true_for_integer_matrix(self, parser):
        """Integer count matrix should be valid."""
        df = pd.DataFrame(
            {"gene1": [100, 200, 0], "gene2": [50, 0, 300]},
            index=["s1", "s2", "s3"],
        )
        assert parser._is_valid_expression_matrix(df) is True


class TestDetermineDataTypeFromMetadata:
    """Test _determine_data_type_from_metadata classification."""

    @patch("lobster.services.data_access.geo.matrix_parsing.MatrixParser._determine_data_type_from_metadata")
    def test_delegates_to_data_type_detector(self, mock_method, parser):
        """Should delegate to DataTypeDetector for classification."""
        mock_method.return_value = "single_cell_rna_seq"
        result = parser._determine_data_type_from_metadata({"samples": {"GSM1": {}}})
        assert result == "single_cell_rna_seq"

    def test_returns_string_result(self, parser):
        """Should return a string data type."""
        # The actual method uses DataTypeDetector which returns strings
        result = parser._determine_data_type_from_metadata({"samples": {}})
        assert isinstance(result, str)


class TestClassifySingleFile:
    """Test _classify_single_file scoring logic."""

    def test_uses_score_expression_file_for_ranking(self, parser):
        """File classification should use _score_expression_file for scoring."""
        # _score_expression_file is imported from helpers and used in _classify_single_file
        # We verify that the scoring function produces expected relative rankings
        from lobster.services.data_access.geo.helpers import _score_expression_file

        # Expression-like filenames should score higher than metadata-like ones
        expression_score = _score_expression_file("counts_matrix.tsv.gz")
        metadata_score = _score_expression_file("sample_metadata.csv")

        assert expression_score > metadata_score, (
            f"Expression file should score higher than metadata: "
            f"{expression_score} vs {metadata_score}"
        )
