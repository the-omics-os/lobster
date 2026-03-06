"""
Tests for SOFT pre-download deduplication and shared helpers.

Covers GDEC-03: SOFT download logic extracted to single location.
Tests build_soft_url (GSE + GSM + short accession) and
pre_download_soft_file (cached, download, failure modes).
Also tests helpers.py exports for shared utilities.
"""

import ssl
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# soft_download.py tests
# ---------------------------------------------------------------------------


class TestBuildSoftUrl:
    """Test build_soft_url constructs correct NCBI HTTPS URLs."""

    def test_gse_series_standard(self):
        """GSE194247 -> series/GSE194nnn/GSE194247/soft/..."""
        from lobster.services.data_access.geo.soft_download import build_soft_url

        url = build_soft_url("GSE194247")
        assert url == (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/"
            "GSE194247/soft/GSE194247_family.soft.gz"
        )

    def test_gsm_sample(self):
        """GSM1234567 -> samples/GSM1234nnn/GSM1234567/soft/..."""
        from lobster.services.data_access.geo.soft_download import build_soft_url

        url = build_soft_url("GSM1234567")
        assert url == (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1234nnn/"
            "GSM1234567/soft/GSM1234567_family.soft.gz"
        )

    def test_short_accession_gse(self):
        """GSE12 (< 3 digits) -> series/GSEnnn/GSE12/soft/..."""
        from lobster.services.data_access.geo.soft_download import build_soft_url

        url = build_soft_url("GSE12")
        assert url == (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/"
            "GSE12/soft/GSE12_family.soft.gz"
        )

    def test_short_accession_gsm(self):
        """GSM99 (< 3 digits) -> samples/GSMnnn/GSM99/soft/..."""
        from lobster.services.data_access.geo.soft_download import build_soft_url

        url = build_soft_url("GSM99")
        assert url == (
            "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/"
            "GSM99/soft/GSM99_family.soft.gz"
        )

    def test_gse_with_three_digits(self):
        """GSE123 -> series/GSEnnn/GSE123/soft/..."""
        from lobster.services.data_access.geo.soft_download import build_soft_url

        url = build_soft_url("GSE123")
        assert url == (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/"
            "GSE123/soft/GSE123_family.soft.gz"
        )


class TestPreDownloadSoftFile:
    """Test pre_download_soft_file caching and download behavior."""

    def test_returns_cached_path(self, tmp_path):
        """If SOFT file already exists, return path without network call."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        cached = tmp_path / "GSE194247_family.soft.gz"
        cached.write_bytes(b"cached content")

        result = pre_download_soft_file("GSE194247", tmp_path)
        assert result == cached

    @patch("lobster.services.data_access.geo.soft_download.urllib.request.urlopen")
    @patch("lobster.services.data_access.geo.soft_download.create_ssl_context")
    def test_downloads_when_not_cached(self, mock_ssl, mock_urlopen, tmp_path):
        """Downloads via HTTPS when file not in cache, returns path."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        # Set up mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b"soft file content"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_ssl.return_value = MagicMock(spec=ssl.SSLContext)

        result = pre_download_soft_file("GSE194247", tmp_path)
        expected = tmp_path / "GSE194247_family.soft.gz"

        assert result == expected
        assert expected.exists()
        assert expected.read_bytes() == b"soft file content"
        mock_urlopen.assert_called_once()

    @patch("lobster.services.data_access.geo.soft_download.urllib.request.urlopen")
    @patch("lobster.services.data_access.geo.soft_download.create_ssl_context")
    @patch("lobster.services.data_access.geo.soft_download.logger")
    def test_returns_none_on_non_ssl_failure(
        self, mock_logger, mock_ssl, mock_urlopen, tmp_path
    ):
        """Non-SSL download failure returns None (GEOparse will retry via FTP)."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        mock_ssl.return_value = MagicMock(spec=ssl.SSLContext)
        mock_urlopen.side_effect = ConnectionError("Connection refused")

        result = pre_download_soft_file("GSE194247", tmp_path)
        assert result is None
        mock_logger.warning.assert_not_called()
        mock_logger.debug.assert_called()

    @patch("lobster.services.data_access.geo.soft_download.urllib.request.urlopen")
    @patch("lobster.services.data_access.geo.soft_download.create_ssl_context")
    def test_raises_on_ssl_certificate_failure(
        self, mock_ssl, mock_urlopen, tmp_path
    ):
        """SSL certificate failure raises (not recoverable by FTP retry)."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        mock_ssl.return_value = MagicMock(spec=ssl.SSLContext)
        mock_urlopen.side_effect = Exception("CERTIFICATE_VERIFY_FAILED")

        with pytest.raises(Exception, match="SSL certificate verification failed"):
            pre_download_soft_file("GSE194247", tmp_path)


# ---------------------------------------------------------------------------
# helpers.py export tests
# ---------------------------------------------------------------------------


class TestHelpersExports:
    """Test that helpers.py exports all required shared symbols."""

    def test_exports_retry_outcome(self):
        from lobster.services.data_access.geo.helpers import RetryOutcome

        assert hasattr(RetryOutcome, "SUCCESS")
        assert hasattr(RetryOutcome, "EXHAUSTED")
        assert hasattr(RetryOutcome, "SOFT_FILE_MISSING")

    def test_exports_retry_result(self):
        from lobster.services.data_access.geo.helpers import RetryResult

        r = RetryResult(
            outcome=__import__(
                "lobster.services.data_access.geo.helpers", fromlist=["RetryOutcome"]
            ).RetryOutcome.SUCCESS,
            value="data",
            retries_used=0,
        )
        assert r.succeeded is True
        assert r.needs_fallback is False

    def test_exports_archive_extensions(self):
        from lobster.services.data_access.geo.helpers import ARCHIVE_EXTENSIONS

        assert isinstance(ARCHIVE_EXTENSIONS, tuple)
        assert ".tar" in ARCHIVE_EXTENSIONS
        assert ".tar.gz" in ARCHIVE_EXTENSIONS
        assert ".tgz" in ARCHIVE_EXTENSIONS
        assert ".tar.bz2" in ARCHIVE_EXTENSIONS

    def test_exports_is_archive_url(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/file.tar.gz") is True
        assert _is_archive_url("https://example.com/file.csv") is False

    def test_exports_score_expression_file(self):
        from lobster.services.data_access.geo.helpers import _score_expression_file

        # Expression file -> positive score
        assert _score_expression_file("GSE123_counts.csv.gz") > 0
        # Feature list -> negative score
        assert _score_expression_file("barcodes.tsv.gz") < 0


class TestIsArchiveUrlFromHelpers:
    """Test _is_archive_url from helpers module."""

    def test_identifies_tar(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.tar") is True

    def test_identifies_tar_gz(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.tar.gz") is True

    def test_identifies_tgz(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.tgz") is True

    def test_identifies_tar_bz2(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.tar.bz2") is True

    def test_rejects_csv(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.csv") is False

    def test_rejects_h5ad(self):
        from lobster.services.data_access.geo.helpers import _is_archive_url

        assert _is_archive_url("https://example.com/data.h5ad") is False


class TestScoreExpressionFileFromHelpers:
    """Test _score_expression_file from helpers module."""

    def test_counts_file_positive(self):
        from lobster.services.data_access.geo.helpers import _score_expression_file

        assert _score_expression_file("GSE123_counts.csv.gz") > 0

    def test_barcodes_file_negative(self):
        from lobster.services.data_access.geo.helpers import _score_expression_file

        assert _score_expression_file("barcodes.tsv.gz") < 0

    def test_expression_matrix_highly_positive(self):
        from lobster.services.data_access.geo.helpers import _score_expression_file

        score = _score_expression_file("GSE123_expression_matrix.h5ad")
        assert score > 2.0  # expression + matrix + h5ad bonus


class TestIsDataValid:
    """Test _is_data_valid standalone function from helpers."""

    def test_none_is_invalid(self):
        from lobster.services.data_access.geo.helpers import _is_data_valid

        assert _is_data_valid(None) is False

    def test_empty_dataframe_is_invalid(self):
        import pandas as pd

        from lobster.services.data_access.geo.helpers import _is_data_valid

        assert _is_data_valid(pd.DataFrame()) is False

    def test_nonempty_dataframe_is_valid(self):
        import pandas as pd

        from lobster.services.data_access.geo.helpers import _is_data_valid

        assert _is_data_valid(pd.DataFrame({"a": [1]})) is True

    def test_other_type_is_valid_if_not_none(self):
        from lobster.services.data_access.geo.helpers import _is_data_valid

        assert _is_data_valid("some_data") is True
