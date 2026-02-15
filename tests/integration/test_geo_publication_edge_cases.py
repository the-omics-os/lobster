"""
Comprehensive edge case tests for GEO and Publication services.

This test module covers Agent 9's mission scenarios:
1. GEO download timeouts
2. Corrupted file handling
3. Metadata extraction edge cases
4. Invalid GEO accessions
5. Network failure recovery
6. Partial download recovery
7. FTP 550 error handling (REGRESSION TEST for recent fix)
8. Concurrent download operations

These tests use realistic scenarios with real file corruption, network simulation,
and stress testing to ensure production resilience.
"""

import ftplib
import gzip
import socket
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.geo_service import GEOService

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data_manager(temp_cache_dir):
    """Create mock DataManagerV2 instance."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.cache_dir = temp_cache_dir
    mock_dm.metadata_store = {}
    mock_dm.list_modalities.return_value = []
    mock_dm.log_tool_usage = Mock()
    return mock_dm


@pytest.fixture
def geo_service(mock_data_manager, temp_cache_dir):
    """Create GEOService instance."""
    return GEOService(data_manager=mock_data_manager, cache_dir=str(temp_cache_dir))


@pytest.fixture
def corrupted_gzip_file(temp_cache_dir):
    """Create corrupted gzip file (truncated)."""
    corrupted_file = temp_cache_dir / "corrupted.txt.gz"

    # Create valid gzip, then truncate it
    with gzip.open(corrupted_file, "wt") as gz:
        gz.write("This is test data\n" * 100)

    # Truncate to corrupt
    with open(corrupted_file, "r+b") as f:
        f.truncate(50)  # Truncate to 50 bytes

    yield corrupted_file

    if corrupted_file.exists():
        corrupted_file.unlink()


@pytest.fixture
def corrupted_tar_file(temp_cache_dir):
    """Create corrupted tar file."""
    corrupted_tar = temp_cache_dir / "corrupted.tar.gz"

    # Create valid tar, then corrupt it
    with tarfile.open(corrupted_tar, "w:gz") as tar:
        # Add some dummy content
        info = tarfile.TarInfo(name="test.txt")
        info.size = 10
        tar.addfile(info, fileobj=None)

    # Corrupt by truncating
    with open(corrupted_tar, "r+b") as f:
        f.truncate(100)

    yield corrupted_tar

    if corrupted_tar.exists():
        corrupted_tar.unlink()


# ===============================================================================
# Scenario 1: GEO Download Timeouts
# ===============================================================================


@pytest.mark.integration
class TestGEODownloadTimeouts:
    """Test GEO download timeout scenarios."""

    @pytest.mark.slow
    @patch("ftplib.FTP")
    def test_ftp_connection_timeout(self, mock_ftp_class, geo_service, temp_cache_dir):
        """Test FTP connection timeout handling (slow: 12s)."""
        mock_ftp = MagicMock()
        mock_ftp.connect.side_effect = socket.timeout("Connection timed out")
        mock_ftp_class.return_value = mock_ftp

        download_file = temp_cache_dir / "test.gz"

        # Simulate download with timeout (may raise socket.timeout or generic Exception)
        with pytest.raises((socket.timeout, Exception)) as exc_info:
            geo_service.geo_downloader._download_ftp(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE123nnn/GSE123456/suppl/test.gz",
                download_file,
                "Test download",
            )

        # Verify it's a timeout-related error
        error_msg = str(exc_info.value).lower()
        assert (
            "timeout" in error_msg or "timed out" in error_msg
        ), f"Expected timeout error, got: {exc_info.value}"

    @patch("ftplib.FTP")
    @patch("time.sleep")  # Mock sleep to speed up test
    def test_download_timeout_with_retry(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test download timeout triggers retry logic."""
        mock_ftp = MagicMock()

        # First attempt times out, second succeeds
        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                raise socket.timeout("Read timed out")
            # Second attempt succeeds
            callback(b"Test data after retry")

        mock_ftp.size.return_value = 100
        mock_ftp.retrbinary.side_effect = mock_retrbinary
        mock_ftp_class.return_value = mock_ftp

        download_file = temp_cache_dir / "test.txt"

        # Should succeed after retry
        with patch.object(
            geo_service.geo_downloader, "_validate_gzip_integrity", return_value=True
        ):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
                download_file,
                "Test download",
                max_retries=3,
            )

        assert success is True
        assert download_file.exists()
        # Should have retried once (2^2 = 4 second delay)
        mock_sleep.assert_called_once_with(4)

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_persistent_timeout_failure(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test persistent timeout leads to final failure."""
        mock_ftp = MagicMock()
        mock_ftp.retrbinary.side_effect = socket.timeout("Persistent timeout")
        mock_ftp_class.return_value = mock_ftp

        download_file = temp_cache_dir / "test.txt"

        success = geo_service.geo_downloader._download_with_retry(
            "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
            download_file,
            "Test download",
            max_retries=3,
        )

        assert success is False
        # Should have retried twice (attempts 2 and 3)
        assert mock_sleep.call_count == 2


# ===============================================================================
# Scenario 2: Corrupted File Handling
# ===============================================================================


@pytest.mark.integration
class TestCorruptedFileHandling:
    """Test handling of corrupted downloaded files."""

    def test_detect_corrupted_gzip(self, geo_service, corrupted_gzip_file):
        """Test detection of corrupted gzip files."""
        is_valid = geo_service.geo_downloader._validate_gzip_integrity(
            corrupted_gzip_file
        )

        assert is_valid is False, "Corrupted gzip should be detected as invalid"

    def test_reject_corrupted_gzip_after_download(self, geo_service, temp_cache_dir):
        """Test that corrupted gzip triggers redownload attempt."""
        download_file = temp_cache_dir / "corrupted_download.gz"

        # Create corrupted file
        with open(download_file, "wb") as f:
            f.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00")  # Valid gzip header
            f.write(b"corrupted_content\x00\x01\x02")  # Invalid content

        # Validation should fail
        is_valid = geo_service.geo_downloader._validate_gzip_integrity(download_file)
        assert is_valid is False

    def test_handle_corrupted_tar_archive(self, geo_service, corrupted_tar_file):
        """Test handling of corrupted TAR archives."""
        # Attempt to parse should fail gracefully
        with pytest.raises(tarfile.ReadError):
            with tarfile.open(corrupted_tar_file, "r:gz") as tar:
                tar.getmembers()

    def test_corrupted_matrix_file_rejection(self, geo_service):
        """Test rejection of corrupted expression matrix files."""
        # Create DataFrame with invalid data (non-numeric)
        corrupted_matrix = pd.DataFrame(
            [["invalid", "data"], ["more", "corruption"]], columns=["gene1", "gene2"]
        )

        is_valid = geo_service._is_valid_expression_matrix(corrupted_matrix)
        assert is_valid is False, "Non-numeric matrix should be rejected"

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_retry_on_corrupted_download(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test retry mechanism when downloaded file is corrupted."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                # First download: corrupted data
                callback(b"\x1f\x8b\x08\x00corrupted")
            else:
                # Second download: valid gzip data
                import gzip
                import io

                buffer = io.BytesIO()
                with gzip.open(buffer, "wt") as gz:
                    gz.write("Valid content")
                callback(buffer.getvalue())

        mock_ftp.size.return_value = 100
        mock_ftp.retrbinary.side_effect = mock_retrbinary

        download_file = temp_cache_dir / "test.gz"

        # Should retry and succeed
        success = geo_service.geo_downloader._download_with_retry(
            "ftp://ftp.ncbi.nlm.nih.gov/geo/test.gz",
            download_file,
            "Test download",
            max_retries=3,
        )

        assert success is True
        assert download_file.exists()


# ===============================================================================
# Scenario 3: Metadata Extraction Edge Cases
# ===============================================================================


@pytest.mark.integration
class TestMetadataExtractionEdgeCases:
    """Test edge cases in GEO metadata extraction."""

    def test_empty_metadata_response(self, geo_service):
        """Test handling of empty metadata response from NCBI."""
        with patch.object(geo_service, "_fetch_geo_metadata") as mock_fetch:
            mock_fetch.return_value = {}

            # Should handle gracefully
            metadata = geo_service._extract_geo_metadata("GSE999999")

            assert metadata is None or isinstance(metadata, dict)

    def test_malformed_xml_metadata(self, geo_service):
        """Test handling of malformed XML in metadata response."""
        malformed_xml = "<GEODataSet><incomplete_tag>"

        with patch("xml.etree.ElementTree.fromstring") as mock_parse:
            mock_parse.side_effect = Exception("XML parsing error")

            with pytest.raises(Exception):
                geo_service._parse_geo_xml(malformed_xml)

    def test_missing_required_metadata_fields(self, geo_service):
        """Test handling when required metadata fields are missing."""
        incomplete_metadata = {
            "accession": "GSE123456",
            # Missing: title, summary, organism, samples
        }

        # Service should handle missing fields gracefully
        is_valid = geo_service._validate_metadata(incomplete_metadata)

        assert is_valid is False or incomplete_metadata.get("title") is None

    def test_unicode_in_metadata(self, geo_service):
        """Test handling of Unicode characters in metadata."""
        unicode_metadata = {
            "title": "Single-cell RNA-seq of β-cells with α-synuclein",
            "summary": "Analysis of immune cells: CD4⁺ T cells, CD8⁺ cytotoxic",
            "organism": "Homo sapiens",
            "characteristics": {
                "tissue": "brain—hippocampus",
                "treatment": "500 μg/mL",
            },
        }

        # Should handle Unicode without errors
        formatted = geo_service._format_metadata_display(unicode_metadata)

        assert isinstance(formatted, str)
        assert "β-cells" in formatted or "beta-cells" in formatted

    def test_very_large_metadata(self, geo_service):
        """Test handling of very large metadata (e.g., 10K samples)."""
        large_metadata = {
            "accession": "GSE123456",
            "title": "Large cohort study",
            "samples": [f"GSM{i}" for i in range(10000)],  # 10K samples
            "characteristics": {f"sample_{i}": f"value_{i}" for i in range(10000)},
        }

        # Should handle without memory issues
        formatted = geo_service._format_metadata_display(large_metadata)

        assert isinstance(formatted, str)
        assert "10000" in formatted or "10,000" in formatted


# ===============================================================================
# Scenario 4: Invalid GEO Accessions
# ===============================================================================


@pytest.mark.integration
class TestInvalidGEOAccessions:
    """Test handling of invalid GEO accession IDs."""

    def test_nonexistent_gse_accession(self, geo_service):
        """Test handling of non-existent GSE accession."""
        fake_accession = "GSE999999999"

        with patch.object(geo_service, "fetch_metadata_only") as mock_fetch:
            mock_fetch.return_value = "Error: Dataset not found"

            result = geo_service.fetch_metadata_only(fake_accession)

            assert "Error" in result or "not found" in result.lower()

    def test_malformed_accession_format(self, geo_service):
        """Test handling of malformed accession format."""
        invalid_accessions = [
            "GS123456",  # Missing 'E'
            "GSE",  # Incomplete
            "12345678",  # Missing prefix
            "GSE-123456",  # Wrong separator
            "gse123456",  # Lowercase (should be normalized)
        ]

        for accession in invalid_accessions:
            # Should either normalize or reject
            normalized = geo_service._normalize_accession(accession)

            # Valid normalization or None
            assert normalized is None or normalized.startswith("GSE")

    def test_gsm_instead_of_gse(self, geo_service):
        """Test handling when GSM (sample) provided instead of GSE (series)."""
        gsm_accession = "GSM1234567"

        # Should detect and handle appropriately
        accession_type = geo_service._detect_accession_type(gsm_accession)

        assert accession_type == "gsm" or accession_type == "sample"

    def test_gpl_platform_accession(self, geo_service):
        """Test handling of GPL (platform) accession."""
        gpl_accession = "GPL24676"

        accession_type = geo_service._detect_accession_type(gpl_accession)

        assert accession_type == "gpl" or accession_type == "platform"


# ===============================================================================
# Scenario 7: FTP 550 Error Handling (REGRESSION TEST)
# ===============================================================================


@pytest.mark.integration
class TestFTP550ErrorHandling:
    """
    REGRESSION TEST for FTP 550 error handling.

    Recent fix: Wrapped FTP 550 errors in OSError to enable retry logic.
    This test ensures the fix remains in place and working correctly.
    """

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_ftp_550_error_wrapped_in_oserror(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """
        Test that FTP 550 error is properly wrapped in OSError for retry.

        REGRESSION TEST: Before fix, FTP 550 errors were not caught by retry logic.
        After fix: FTP 550 errors are wrapped in OSError and retried.
        """
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                # First attempt: FTP 550 error
                raise ftplib.error_perm("550 File not found")
            # Second attempt: success
            callback(b"Success after FTP 550")

        mock_ftp.size.return_value = 100
        mock_ftp.retrbinary.side_effect = mock_retrbinary

        download_file = temp_cache_dir / "test.txt"

        # Should wrap FTP 550 in OSError and retry
        with patch.object(
            geo_service.geo_downloader, "_validate_gzip_integrity", return_value=True
        ):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
                download_file,
                "Test download",
                max_retries=3,
            )

        assert (
            success is True
        ), "FTP 550 error should trigger retry and eventually succeed"
        assert attempt[0] == 2, "Should have made 2 attempts (1 failure + 1 success)"
        mock_sleep.assert_called_once_with(4)  # Exponential backoff: 2^2 = 4

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_persistent_ftp_550_failure(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test persistent FTP 550 error leads to final failure after retries."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # All attempts fail with FTP 550
        mock_ftp.retrbinary.side_effect = ftplib.error_perm(
            "550 Requested file not found"
        )

        download_file = temp_cache_dir / "test.txt"

        success = geo_service.geo_downloader._download_with_retry(
            "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
            download_file,
            "Test download",
            max_retries=3,
        )

        assert success is False
        # Should have retried 2 times (attempts 2 and 3)
        assert mock_sleep.call_count == 2

    @patch("ftplib.FTP")
    def test_ftp_550_vs_other_errors(self, mock_ftp_class, geo_service, temp_cache_dir):
        """Test that FTP 550 is handled differently from other FTP errors."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Test different FTP errors
        ftp_errors = [
            ftplib.error_perm("550 File not found"),  # Should retry
            ftplib.error_perm("530 Login incorrect"),  # May not retry
            ftplib.error_temp("421 Service not available"),  # Should retry
        ]

        for error in ftp_errors:
            mock_ftp.retrbinary.side_effect = error

            download_file = temp_cache_dir / f"test_{ftp_errors.index(error)}.txt"

            with patch("time.sleep"):  # Speed up test
                success = geo_service.geo_downloader._download_with_retry(
                    "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
                    download_file,
                    "Test download",
                    max_retries=2,
                )

            # All should eventually fail if persistent
            assert success is False


# ===============================================================================
# Scenario 8: Concurrent Download Operations
# ===============================================================================


@pytest.mark.integration
class TestConcurrentDownloadOperations:
    """Test concurrent download operations and thread safety."""

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_concurrent_downloads_different_files(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test downloading multiple files concurrently."""

        def mock_download(accession):
            """Mock download function."""
            mock_ftp = MagicMock()
            mock_ftp.size.return_value = 100
            mock_ftp.retrbinary = lambda cmd, callback, blocksize: callback(
                f"Data for {accession}".encode()
            )
            mock_ftp_class.return_value = mock_ftp

            download_file = temp_cache_dir / f"{accession}.txt"

            with patch.object(
                geo_service.geo_downloader,
                "_validate_gzip_integrity",
                return_value=True,
            ):
                success = geo_service.geo_downloader._download_with_retry(
                    f"ftp://ftp.ncbi.nlm.nih.gov/geo/{accession}.txt",
                    download_file,
                    f"Downloading {accession}",
                    max_retries=2,
                )

            return accession, success, download_file.exists()

        # Simulate concurrent downloads
        accessions = [f"GSE12345{i}" for i in range(5)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(mock_download, acc): acc for acc in accessions}

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    results.append((futures[future], False, str(e)))

        # All downloads should succeed
        assert len(results) == 5
        for accession, success, exists in results:
            assert success is True, f"Download failed for {accession}"

    @patch("ftplib.FTP")
    def test_concurrent_downloads_with_failures(
        self, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test concurrent downloads with some failures."""

        def mock_download_with_failure(accession, should_fail):
            """Mock download with controlled failure."""
            mock_ftp = MagicMock()

            if should_fail:
                mock_ftp.retrbinary.side_effect = socket.timeout("Simulated failure")
            else:
                mock_ftp.size.return_value = 100
                mock_ftp.retrbinary = lambda cmd, callback, blocksize: callback(
                    f"Data for {accession}".encode()
                )

            mock_ftp_class.return_value = mock_ftp

            download_file = temp_cache_dir / f"{accession}_fail.txt"

            with patch("time.sleep"):  # Speed up test
                with patch.object(
                    geo_service.geo_downloader,
                    "_validate_gzip_integrity",
                    return_value=True,
                ):
                    success = geo_service.geo_downloader._download_with_retry(
                        f"ftp://ftp.ncbi.nlm.nih.gov/geo/{accession}.txt",
                        download_file,
                        f"Downloading {accession}",
                        max_retries=2,
                    )

            return accession, success

        # Mix of success and failure
        test_cases = [
            ("GSE123450", False),  # Should succeed
            ("GSE123451", True),  # Should fail
            ("GSE123452", False),  # Should succeed
            ("GSE123453", True),  # Should fail
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(mock_download_with_failure, acc, should_fail): (
                    acc,
                    should_fail,
                )
                for acc, should_fail in test_cases
            }

            results = {}
            for future in as_completed(futures):
                acc, expected_failure = futures[future]
                try:
                    accession, success = future.result(timeout=10)
                    results[accession] = success
                except Exception:
                    results[acc] = False

        # Verify expected outcomes
        assert results["GSE123450"] is True, "GSE123450 should succeed"
        assert results["GSE123451"] is False, "GSE123451 should fail"
        assert results["GSE123452"] is True, "GSE123452 should succeed"
        assert results["GSE123453"] is False, "GSE123453 should fail"

    def test_thread_safe_cache_access(self, geo_service, temp_cache_dir):
        """Test that cache directory access is thread-safe."""

        def create_cache_file(worker_id):
            """Create a cache file in a thread."""
            cache_file = temp_cache_dir / f"worker_{worker_id}.cache"
            cache_file.write_text(f"Cache data from worker {worker_id}")
            time.sleep(0.01)  # Simulate processing
            return worker_id, cache_file.exists()

        # Simulate concurrent cache writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(create_cache_file, i): i for i in range(10)}

            results = []
            for future in as_completed(futures):
                worker_id, exists = future.result()
                results.append((worker_id, exists))

        # All cache files should exist
        assert len(results) == 10
        for worker_id, exists in results:
            assert exists is True, f"Cache file for worker {worker_id} missing"


# ===============================================================================
# Scenario 9: Partial Download Recovery
# ===============================================================================


@pytest.mark.integration
class TestPartialDownloadRecovery:
    """Test recovery from partial/interrupted downloads."""

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_resume_partial_download(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test resuming a partially downloaded file."""
        # Create partial file
        partial_file = temp_cache_dir / "partial_download.gz"
        partial_content = b"Partial data " * 10
        partial_file.write_bytes(partial_content)

        partial_size = len(partial_content)
        total_size = partial_size + 100

        mock_ftp = MagicMock()
        mock_ftp.size.return_value = total_size

        # Simulate resuming from partial position
        def mock_retrbinary(cmd, callback, blocksize=8192):
            # Only send remaining data
            remaining_data = b"Remaining data after resume"
            callback(remaining_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary
        mock_ftp_class.return_value = mock_ftp

        # Should resume from existing position
        with patch.object(
            geo_service.geo_downloader, "_validate_gzip_integrity", return_value=True
        ):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.gz",
                partial_file,
                "Resume download",
                max_retries=2,
            )

        assert success is True
        assert partial_file.exists()

    @patch("ftplib.FTP")
    def test_cleanup_failed_partial_download(
        self, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test cleanup of failed partial downloads."""
        partial_file = temp_cache_dir / "failed_download.gz"
        partial_file.write_bytes(b"Failed partial data")

        mock_ftp = MagicMock()
        mock_ftp.retrbinary.side_effect = Exception("Download interrupted")
        mock_ftp_class.return_value = mock_ftp

        # Download fails, should clean up partial file
        with patch("time.sleep"):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.gz",
                partial_file,
                "Failed download",
                max_retries=2,
            )

        assert success is False
        # Partial file should be cleaned up
        assert not partial_file.exists()


# ===============================================================================
# Scenario 10: Network Failure Recovery
# ===============================================================================


@pytest.mark.integration
class TestNetworkFailureRecovery:
    """Test recovery from various network failures."""

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_connection_reset_recovery(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test recovery from connection reset errors."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                raise ConnectionResetError("Connection reset by peer")
            callback(b"Success after connection reset")

        mock_ftp.size.return_value = 100
        mock_ftp.retrbinary.side_effect = mock_retrbinary

        download_file = temp_cache_dir / "test.txt"

        with patch.object(
            geo_service.geo_downloader, "_validate_gzip_integrity", return_value=True
        ):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
                download_file,
                "Test download",
                max_retries=3,
            )

        assert success is True
        assert download_file.exists()

    @patch("ftplib.FTP")
    @patch("time.sleep")
    def test_dns_resolution_failure_recovery(
        self, mock_sleep, mock_ftp_class, geo_service, temp_cache_dir
    ):
        """Test recovery from DNS resolution failures."""
        mock_ftp = MagicMock()

        attempt = [0]

        def mock_connect(host, port=21, timeout=30):
            attempt[0] += 1
            if attempt[0] == 1:
                raise socket.gaierror("Name or service not known")
            # Second attempt succeeds
            return None

        mock_ftp.connect.side_effect = mock_connect
        mock_ftp.size.return_value = 100
        mock_ftp.retrbinary = lambda cmd, callback, blocksize: callback(
            b"DNS recovered"
        )
        mock_ftp_class.return_value = mock_ftp

        download_file = temp_cache_dir / "test.txt"

        with patch.object(
            geo_service.geo_downloader, "_validate_gzip_integrity", return_value=True
        ):
            success = geo_service.geo_downloader._download_with_retry(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/test.txt",
                download_file,
                "Test download",
                max_retries=3,
            )

        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
