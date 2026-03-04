"""Tests for ParseResult dataclass and chunk parser partial signaling.

Verifies that:
- ParseResult carries integrity metadata (is_partial, rows_read, truncation_reason)
- parse_large_file_in_chunks returns ParseResult with correct flags
- parse_expression_file returns ParseResult instead of bare DataFrame
- parse_supplementary_file returns ParseResult when delegating
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from lobster.services.data_access.geo.parser import GEOParser, ParseResult


# ---------------------------------------------------------------------------
# ParseResult dataclass property tests
# ---------------------------------------------------------------------------


class TestParseResultProperties:
    """Test ParseResult dataclass fields and convenience properties."""

    def test_complete_result(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ParseResult(data=df, is_partial=False, rows_read=3)
        assert result.is_complete is True
        assert result.is_empty is False
        assert result.truncation_reason is None

    def test_partial_result(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = ParseResult(
            data=df,
            is_partial=True,
            rows_read=2,
            truncation_reason="Memory limit reached after 5 chunks (2 rows)",
        )
        assert result.is_complete is False
        assert result.is_empty is False
        assert result.truncation_reason is not None

    def test_empty_none_data(self):
        result = ParseResult(data=None)
        assert result.is_empty is True
        assert result.is_complete is False

    def test_empty_dataframe(self):
        result = ParseResult(data=pd.DataFrame())
        assert result.is_empty is True
        # Not partial, but data is empty so not complete either
        assert result.is_complete is False

    def test_defaults(self):
        result = ParseResult(data=None)
        assert result.is_partial is False
        assert result.rows_read == 0
        assert result.truncation_reason is None


# ---------------------------------------------------------------------------
# parse_large_file_in_chunks tests
# ---------------------------------------------------------------------------


class TestParseLargeFileInChunksPartial:
    """Test that parse_large_file_in_chunks returns ParseResult with partial flags."""

    def setup_method(self):
        self.parser = GEOParser()

    @patch("lobster.services.data_access.geo.parser.pd.read_csv")
    @patch("lobster.services.data_access.geo.parser.psutil.virtual_memory")
    def test_memory_limit_returns_partial(self, mock_vm, mock_read_csv):
        """When memory limit triggers break, result is_partial=True."""
        chunk1 = pd.DataFrame({"gene": range(100)})
        chunk2 = pd.DataFrame({"gene": range(100, 200)})

        # First chunk: memory OK. Second chunk: memory NOT available.
        mock_read_csv.return_value = iter([chunk1, chunk2])

        # _check_memory_availability: first call True (chunk1 accepted),
        # second call False (chunk2 triggers break)
        with patch.object(
            self.parser,
            "_check_memory_availability",
            side_effect=[False],  # First chunk fails memory check -> break
        ), patch.object(
            self.parser, "_get_adaptive_chunk_size", return_value=100
        ), patch.object(
            self.parser, "_log_system_memory"
        ):
            # mock vm for the percent check (should not reach it since we break first)
            mock_vm.return_value = MagicMock(percent=50)

            from pathlib import Path

            result = self.parser.parse_large_file_in_chunks(
                Path("/fake/file.csv"), "\t", None
            )

        assert isinstance(result, ParseResult)
        # Broke on first chunk -> no data collected
        assert result.is_partial is True
        assert "Memory limit" in result.truncation_reason

    @patch("lobster.services.data_access.geo.parser.pd.read_csv")
    @patch("lobster.services.data_access.geo.parser.psutil.virtual_memory")
    def test_normal_completion_returns_complete(self, mock_vm, mock_read_csv):
        """Normal completion returns is_partial=False with correct rows_read."""
        chunk1 = pd.DataFrame({"gene": range(50)})
        chunk2 = pd.DataFrame({"gene": range(50, 100)})

        mock_read_csv.return_value = iter([chunk1, chunk2])
        mock_vm.return_value = MagicMock(percent=50)

        with patch.object(
            self.parser, "_check_memory_availability", return_value=True
        ), patch.object(
            self.parser, "_get_adaptive_chunk_size", return_value=50
        ), patch.object(
            self.parser, "_log_system_memory"
        ):
            from pathlib import Path

            result = self.parser.parse_large_file_in_chunks(
                Path("/fake/file.csv"), "\t", None
            )

        assert isinstance(result, ParseResult)
        assert result.is_partial is False
        assert result.rows_read == 100
        assert result.data is not None
        assert len(result.data) == 100

    @patch("lobster.services.data_access.geo.parser.pd.read_csv")
    def test_no_chunks_returns_empty(self, mock_read_csv):
        """No data read returns ParseResult with data=None."""
        mock_read_csv.return_value = iter([])  # Empty iterator

        with patch.object(
            self.parser, "_get_adaptive_chunk_size", return_value=100
        ), patch.object(
            self.parser, "_log_system_memory"
        ):
            from pathlib import Path

            result = self.parser.parse_large_file_in_chunks(
                Path("/fake/file.csv"), "\t", None
            )

        assert isinstance(result, ParseResult)
        assert result.data is None
        assert result.is_partial is False
        assert result.rows_read == 0

    @patch("lobster.services.data_access.geo.parser.pd.read_csv")
    def test_memory_error_returns_partial_none(self, mock_read_csv):
        """MemoryError during parsing returns ParseResult with is_partial=True and data=None."""
        mock_read_csv.side_effect = MemoryError("out of memory")

        with patch.object(
            self.parser, "_get_adaptive_chunk_size", return_value=100
        ), patch.object(
            self.parser, "_log_system_memory"
        ):
            from pathlib import Path

            result = self.parser.parse_large_file_in_chunks(
                Path("/fake/file.csv"), "\t", None
            )

        assert isinstance(result, ParseResult)
        assert result.data is None
        assert result.is_partial is True
        assert "MemoryError" in result.truncation_reason


# ---------------------------------------------------------------------------
# parse_expression_file returns ParseResult
# ---------------------------------------------------------------------------


class TestParseExpressionFileReturnsParseResult:
    """Test that parse_expression_file returns ParseResult."""

    def setup_method(self):
        self.parser = GEOParser()

    @patch.object(GEOParser, "_estimate_dimensions_from_file", return_value=(None, None))
    @patch.object(GEOParser, "_estimate_dataframe_memory", return_value=None)
    @patch.object(GEOParser, "sniff_delimiter", return_value="\t")
    def test_returns_parse_result_on_success(self, mock_sniff, mock_est, mock_dim):
        """parse_expression_file wraps DataFrame in ParseResult."""
        df = pd.DataFrame({"gene1": [1.0, 2.0], "gene2": [3.0, 4.0]})

        with patch("lobster.services.data_access.geo.parser.pd.read_csv", return_value=df):
            with patch.object(self.parser, "_log_system_memory"):
                from pathlib import Path

                result = self.parser.parse_expression_file(Path("/fake/small_file.csv"))

        assert isinstance(result, ParseResult)
        assert result.data is not None
        assert result.is_partial is False

    def test_returns_parse_result_on_rds(self):
        """RDS files return ParseResult with data=None."""
        from pathlib import Path

        result = self.parser.parse_expression_file(Path("/fake/file.rds"))
        assert isinstance(result, ParseResult)
        assert result.data is None

    @patch.object(GEOParser, "_estimate_dimensions_from_file", return_value=(None, None))
    @patch.object(GEOParser, "_estimate_dataframe_memory", return_value=None)
    @patch.object(GEOParser, "sniff_delimiter", return_value="\t")
    def test_returns_parse_result_on_exception(self, mock_sniff, mock_est, mock_dim):
        """General exception returns ParseResult with data=None."""
        with patch(
            "lobster.services.data_access.geo.parser.pd.read_csv",
            side_effect=Exception("bad file"),
        ):
            from pathlib import Path

            # Force into the basic_pandas fallback that also raises
            with patch.object(
                self.parser, "parse_with_basic_pandas", side_effect=Exception("still bad")
            ):
                result = self.parser.parse_expression_file(Path("/fake/bad_file.csv"))

        assert isinstance(result, ParseResult)
        assert result.data is None


# ---------------------------------------------------------------------------
# parse_supplementary_file returns ParseResult
# ---------------------------------------------------------------------------


class TestParseSupplementaryFileReturnsParseResult:
    """Test that parse_supplementary_file returns ParseResult."""

    def setup_method(self):
        self.parser = GEOParser()

    def test_delegates_to_parse_expression_file(self):
        """For .txt files, delegates to parse_expression_file which returns ParseResult."""
        expected = ParseResult(data=pd.DataFrame({"a": [1]}), rows_read=1)

        with patch.object(
            self.parser, "parse_expression_file", return_value=expected
        ):
            from pathlib import Path

            result = self.parser.parse_supplementary_file(Path("/fake/data.txt"))

        assert isinstance(result, ParseResult)
        assert result is expected

    def test_unsupported_format_returns_parse_result(self):
        """Unsupported format returns ParseResult with data=None."""
        from pathlib import Path

        result = self.parser.parse_supplementary_file(Path("/fake/data.xyz"))
        assert isinstance(result, ParseResult)
        assert result.data is None

    def test_exception_returns_parse_result(self):
        """Exception during parsing returns ParseResult with data=None."""
        with patch.object(
            self.parser,
            "_parse_h5ad_with_fallback",
            side_effect=Exception("h5ad error"),
        ):
            from pathlib import Path

            result = self.parser.parse_supplementary_file(Path("/fake/data.h5ad"))

        assert isinstance(result, ParseResult)
        assert result.data is None
