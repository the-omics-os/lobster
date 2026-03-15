"""Tests for geo_service.py call sites handling ParseResult from parser.

Verifies that all 5 call sites in GEOService:
- Extract .data from ParseResult
- Log a warning when .is_partial is True
- Handle ParseResult with data=None gracefully (same as old None)
- Do not log warnings for complete results
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lobster.services.data_access.geo.parser import ParseResult

# The lobster logger sets propagate=False and has its own StreamHandler,
# so pytest's caplog cannot capture messages. We use a custom handler.
# After Phase 4 decomposition, methods live in domain modules under geo/.
_GEO_LOGGER_NAMES = [
    "lobster.services.data_access.geo_service",
    "lobster.services.data_access.geo.archive_processing",
    "lobster.services.data_access.geo.matrix_parsing",
    "lobster.services.data_access.geo.download_execution",
    "lobster.services.data_access.geo.metadata_fetch",
    "lobster.services.data_access.geo.concatenation",
]


class _WarningCapture(logging.Handler):
    """Simple handler to capture WARNING+ log messages in tests."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def log_capture():
    """Attach a capturing handler to the geo_service and domain module loggers."""
    handler = _WarningCapture()
    loggers = [logging.getLogger(name) for name in _GEO_LOGGER_NAMES]
    for lgr in loggers:
        lgr.addHandler(handler)
    yield handler
    for lgr in loggers:
        lgr.removeHandler(handler)


@pytest.fixture
def geo_service():
    """Create a minimally-initialized GEOService for call site testing."""
    from lobster.services.data_access.geo_service import GEOService

    with patch.object(GEOService, "__init__", lambda self: None):
        svc = GEOService()
        svc.geo_parser = MagicMock()
        svc.geo_downloader = MagicMock()
        svc.cache_dir = Path("/tmp/test_cache")
        return svc


# ---------------------------------------------------------------------------
# _download_and_parse_file (line 4004: parse_expression_file)
# ---------------------------------------------------------------------------


class TestDownloadAndParseFilePartialHandling:
    """Test call site at line 4004 in _download_and_parse_file."""

    def test_partial_result_logs_warning_and_returns_data(
        self, geo_service, log_capture
    ):
        """parse_expression_file returning partial ParseResult triggers warning."""
        df = pd.DataFrame({"a": [1]})
        partial = ParseResult(
            data=df,
            is_partial=True,
            rows_read=50,
            truncation_reason="Memory limit reached after 3 chunks (50 rows)",
        )
        geo_service.geo_parser.parse_expression_file.return_value = partial

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_and_parse_file(
                "ftp://example.com/file.csv", "GSE12345"
            )

        # Should extract .data from ParseResult and return it
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        # Should have logged a warning about partial parse
        assert any("Partial parse" in msg for msg in log_capture.messages)

    def test_complete_result_no_warning(self, geo_service, log_capture):
        """Complete ParseResult does not trigger a warning."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        complete = ParseResult(data=df, is_partial=False, rows_read=3)
        geo_service.geo_parser.parse_expression_file.return_value = complete

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_and_parse_file(
                "ftp://example.com/file.csv", "GSE12345"
            )

        assert isinstance(result, pd.DataFrame)
        assert not any("Partial parse" in msg for msg in log_capture.messages)

    def test_none_data_returns_none(self, geo_service):
        """ParseResult with data=None behaves like old None return."""
        empty = ParseResult(data=None)
        geo_service.geo_parser.parse_expression_file.return_value = empty

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_and_parse_file(
                "ftp://example.com/file.csv", "GSE12345"
            )

        assert result is None


# ---------------------------------------------------------------------------
# _download_h5_file (line 4849: parse_supplementary_file)
# ---------------------------------------------------------------------------


class TestDownloadH5FilePartialHandling:
    """Test call site at line 4849 in _download_h5_file."""

    def test_partial_result_logs_warning(self, geo_service, log_capture):
        """Partial ParseResult from H5 parse triggers warning."""
        df = pd.DataFrame({"gene": [1, 2]}, index=["cell1", "cell2"])
        partial = ParseResult(
            data=df,
            is_partial=True,
            rows_read=2,
            truncation_reason="Memory limit reached",
        )
        geo_service.geo_parser.parse_supplementary_file.return_value = partial

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_h5_file(
                "ftp://example.com/file.h5", "GSM123"
            )

        # Should extract data and still return it
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert any("Partial parse" in msg for msg in log_capture.messages)

    def test_none_data_returns_none(self, geo_service):
        """ParseResult with data=None returns None (same as old behavior)."""
        empty = ParseResult(data=None)
        geo_service.geo_parser.parse_supplementary_file.return_value = empty

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_h5_file(
                "ftp://example.com/file.h5", "GSM123"
            )

        assert result is None


# ---------------------------------------------------------------------------
# _download_single_expression_file (line 5004: parse_supplementary_file)
# ---------------------------------------------------------------------------


class TestDownloadSingleExpressionFilePartialHandling:
    """Test call site at line 5004 in _download_single_expression_file."""

    def test_partial_result_logs_warning(self, geo_service, log_capture):
        """Partial ParseResult triggers warning."""
        df = pd.DataFrame({"gene": [1]}, index=["cell1"])
        partial = ParseResult(
            data=df,
            is_partial=True,
            rows_read=1,
            truncation_reason="Memory limit",
        )
        geo_service.geo_parser.parse_supplementary_file.return_value = partial
        geo_service._determine_transpose_biologically = MagicMock(
            return_value=(False, "no transpose needed")
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_single_expression_file(
                "ftp://example.com/file.txt", "GSM456", "GSE789"
            )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert any("Partial parse" in msg for msg in log_capture.messages)

    def test_complete_result_no_warning(self, geo_service, log_capture):
        """Complete result does not log warning."""
        df = pd.DataFrame({"gene": [1, 2]}, index=["cell1", "cell2"])
        complete = ParseResult(data=df, is_partial=False, rows_read=2)
        geo_service.geo_parser.parse_supplementary_file.return_value = complete
        geo_service._determine_transpose_biologically = MagicMock(
            return_value=(False, "no transpose needed")
        )

        with patch("pathlib.Path.exists", return_value=True):
            result = geo_service._download_single_expression_file(
                "ftp://example.com/file.txt", "GSM456", "GSE789"
            )

        assert result is not None
        assert not any("Partial parse" in msg for msg in log_capture.messages)
