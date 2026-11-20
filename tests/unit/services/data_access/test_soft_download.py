"""
Tests for SOFT pre-download deduplication and shared helpers.

Covers GDEC-03: SOFT download logic extracted to single location.
Tests build_soft_url (GSE + GSM + short accession) and
pre_download_soft_file (cached, download, failure modes).
Also tests helpers.py exports for shared utilities.
"""

import gzip
import http.client
import socket
import ssl
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# soft_download.py tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_backoff_waits(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)


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


def _soft_fixture(geo_id="GSE194247"):
    return gzip.compress(
        (f"^SERIES = {geo_id}\n!Series_geo_accession = {geo_id}\n").encode(),
        mtime=0,
    )


class TestPreDownloadSoftFile:
    """Test pre_download_soft_file caching and download behavior."""

    def test_returns_cached_path(self, tmp_path):
        """If SOFT file already exists, return path without network call."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        cached = tmp_path / "GSE194247_family.soft.gz"
        cached.write_bytes(_soft_fixture())

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
        mock_response.read.return_value = _soft_fixture()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_ssl.return_value = MagicMock(spec=ssl.SSLContext)

        result = pre_download_soft_file("GSE194247", tmp_path)
        expected = tmp_path / "GSE194247_family.soft.gz"

        assert result == expected
        assert expected.exists()
        assert expected.read_bytes() == _soft_fixture()
        mock_urlopen.assert_called_once_with(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194247/soft/GSE194247_family.soft.gz",
            context=mock_ssl.return_value,
            timeout=60.0,
        )
        mock_ssl.assert_called_once()

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
    def test_raises_on_ssl_certificate_failure(self, mock_ssl, mock_urlopen, tmp_path):
        """SSL certificate failure raises (not recoverable by FTP retry)."""
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        mock_ssl.return_value = MagicMock(spec=ssl.SSLContext)
        mock_urlopen.side_effect = Exception("CERTIFICATE_VERIFY_FAILED")

        with pytest.raises(Exception, match="SSL certificate verification failed"):
            pre_download_soft_file("GSE194247", tmp_path)


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if isinstance(self.body, BaseException):
            raise self.body
        return self.body


class TestPreDownloadOnASlowRoute:
    """A slow or filtered route to NCBI: each request is bounded and retried, so the
    metadata stays on HTTPS instead of falling back to GEOparse's FTP."""

    @pytest.fixture
    def opened(self, monkeypatch):
        from lobster.services.data_access.geo import soft_download

        calls, sleeps = [], []
        monkeypatch.setattr(soft_download, "create_ssl_context", lambda: None)
        monkeypatch.setattr("time.sleep", sleeps.append)

        def serve(*results):
            def urlopen(url, context=None, timeout=None):
                calls.append(timeout)
                result = results[min(len(calls), len(results)) - 1]
                if isinstance(result, BaseException) and not isinstance(
                    result, socket.timeout
                ):
                    raise result
                return _Response(result)

            monkeypatch.setattr(soft_download.urllib.request, "urlopen", urlopen)
            return calls, sleeps

        return serve

    @pytest.mark.parametrize("idle, timeout", [(None, 60.0), ("240", 240.0)])
    def test_each_request_waits_as_long_as_the_idle_bound(
        self, opened, tmp_path, monkeypatch, idle, timeout
    ):
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        if idle is None:
            monkeypatch.delenv("LOBSTER_DOWNLOAD_IDLE_TIMEOUT_S", raising=False)
        else:
            monkeypatch.setenv("LOBSTER_DOWNLOAD_IDLE_TIMEOUT_S", idle)
        calls, _ = opened(_soft_fixture())

        assert pre_download_soft_file("GSE194247", tmp_path) is not None
        assert calls == [timeout]

    @pytest.mark.parametrize(
        "failure",
        [
            urllib.error.URLError(socket.timeout("timed out")),
            ConnectionResetError("reset"),
            socket.timeout("the read stalled"),
            http.client.IncompleteRead(b"partial"),
            urllib.error.HTTPError("u", 503, "busy", {}, None),
            urllib.error.HTTPError("u", 500, "error", {}, None),
            b"not gzip",
        ],
        ids=["connect-timeout", "reset", "read-stall", "cut", "503", "500", "garbled"],
    )
    def test_a_transient_failure_is_tried_again(self, opened, tmp_path, failure):
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        calls, sleeps = opened(failure, _soft_fixture())

        result = pre_download_soft_file("GSE194247", tmp_path)

        assert result == tmp_path / "GSE194247_family.soft.gz"
        assert result.read_bytes() == _soft_fixture()
        assert len(calls) == 2 and len(sleeps) == 1
        assert not list(tmp_path.glob(".GSE194247_soft_*"))

    def test_it_gives_up_after_three_attempts(self, opened, tmp_path):
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        calls, sleeps = opened(ConnectionResetError("reset"))

        assert pre_download_soft_file("GSE194247", tmp_path) is None
        assert len(calls) == 3
        assert len(sleeps) == 2 and sleeps[0] < sleeps[1]

    def test_a_file_the_server_does_not_have_is_not_asked_for_again(
        self, opened, tmp_path
    ):
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        calls, sleeps = opened(urllib.error.HTTPError("u", 404, "missing", {}, None))

        assert pre_download_soft_file("GSE194247", tmp_path) is None
        assert len(calls) == 1 and not sleeps

    @pytest.mark.parametrize(
        "failure",
        [
            Exception("CERTIFICATE_VERIFY_FAILED"),
            ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number"),
        ],
        ids=["certificate", "ssl"],
    )
    def test_a_certificate_failure_is_not_tried_again(self, opened, tmp_path, failure):
        from lobster.services.data_access.geo.soft_download import (
            pre_download_soft_file,
        )

        calls, _ = opened(failure)

        with pytest.raises(Exception, match="SSL certificate verification failed"):
            pre_download_soft_file("GSE194247", tmp_path)
        assert len(calls) == 1


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
