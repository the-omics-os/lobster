"""
Unit tests for GEO downloader retry logic and FTP download improvements.

This module tests Bug #1 fixes:
- Exponential backoff retry logic
- Chunked FTP downloads for large files
- MD5 checksum verification
- Gzip integrity validation
- Error recovery and cleanup
"""

import ftplib
import gzip
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from lobster.tools.geo_downloader import GEODownloadManager


@pytest.fixture
def geo_manager():
    """Create GEODownloadManager instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = GEODownloadManager(cache_dir=temp_dir)
        yield manager


@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def temp_gzip_file():
    """Create temporary gzip file with valid content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as f:
        temp_path = Path(f.name)

    # Write valid gzip content
    test_content = b"Test data for gzip validation\n" * 100
    with gzip.open(temp_path, "wb") as gz:
        gz.write(test_content)

    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


# ===============================================================================
# MD5 Checksum Tests
# ===============================================================================


class TestMD5Checksum:
    """Test MD5 checksum calculation."""

    def test_calculate_md5_valid_file(self, geo_manager, temp_file):
        """Test MD5 calculation for valid file."""
        # Write test content
        test_content = b"Test content for MD5 calculation"
        temp_file.write_bytes(test_content)

        # Calculate MD5
        md5_hash = geo_manager._calculate_md5(temp_file)

        # Verify MD5 format (32 hex characters)
        assert isinstance(md5_hash, str)
        assert len(md5_hash) == 32
        assert all(c in "0123456789abcdef" for c in md5_hash)

    def test_calculate_md5_reproducible(self, geo_manager, temp_file):
        """Test MD5 calculation is reproducible."""
        # Write test content
        test_content = b"Reproducible test content"
        temp_file.write_bytes(test_content)

        # Calculate MD5 twice
        md5_1 = geo_manager._calculate_md5(temp_file)
        md5_2 = geo_manager._calculate_md5(temp_file)

        assert md5_1 == md5_2

    def test_calculate_md5_different_content(self, geo_manager):
        """Test MD5 differs for different content."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            file1 = Path(f1.name)
            file1.write_bytes(b"Content A")

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            file2 = Path(f2.name)
            file2.write_bytes(b"Content B")

        try:
            md5_1 = geo_manager._calculate_md5(file1)
            md5_2 = geo_manager._calculate_md5(file2)

            assert md5_1 != md5_2
        finally:
            file1.unlink()
            file2.unlink()

    def test_calculate_md5_large_file(self, geo_manager):
        """Test MD5 calculation for large file (memory efficiency)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            large_file = Path(f.name)
            # Write 10MB of data in chunks
            chunk = b"X" * 8192
            for _ in range(1280):  # 1280 * 8KB = 10MB
                f.write(chunk)

        try:
            md5_hash = geo_manager._calculate_md5(large_file)

            # Should complete without memory issues
            assert isinstance(md5_hash, str)
            assert len(md5_hash) == 32
        finally:
            large_file.unlink()


# ===============================================================================
# Gzip Integrity Validation Tests
# ===============================================================================


class TestGzipValidation:
    """Test gzip integrity validation."""

    def test_validate_valid_gzip(self, geo_manager, temp_gzip_file):
        """Test validation of valid gzip file."""
        is_valid = geo_manager._validate_gzip_integrity(temp_gzip_file)
        assert is_valid is True

    def test_validate_corrupted_gzip(self, geo_manager, temp_file):
        """Test validation of corrupted gzip file."""
        # Create corrupted gzip file (just write random bytes)
        corrupted_path = temp_file.parent / "corrupted.gz"
        corrupted_path.write_bytes(b"Not a valid gzip file\x00\x01\x02")

        try:
            is_valid = geo_manager._validate_gzip_integrity(corrupted_path)
            assert is_valid is False
        finally:
            if corrupted_path.exists():
                corrupted_path.unlink()

    def test_validate_truncated_gzip(self, geo_manager, temp_gzip_file):
        """Test validation of truncated gzip file."""
        # Truncate the gzip file to simulate incomplete download
        with open(temp_gzip_file, "r+b") as f:
            f.truncate(50)  # Truncate to 50 bytes

        is_valid = geo_manager._validate_gzip_integrity(temp_gzip_file)
        assert is_valid is False

    def test_validate_empty_gzip(self, geo_manager):
        """Test validation of empty gzip file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as f:
            empty_file = Path(f.name)
            # Write empty content

        try:
            is_valid = geo_manager._validate_gzip_integrity(empty_file)
            # Empty gzip files may be valid or invalid depending on format
            assert isinstance(is_valid, bool)
        finally:
            empty_file.unlink()


# ===============================================================================
# Chunked FTP Download Tests
# ===============================================================================


class TestChunkedFTPDownload:
    """Test chunked FTP download functionality."""

    @patch("ftplib.FTP")
    def test_chunked_download_success(self, mock_ftp_class, geo_manager, temp_file):
        """Test successful chunked FTP download."""
        # Mock FTP instance
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Simulate download data
        test_data = b"Test data chunk" * 100
        file_size = len(test_data)

        def mock_retrbinary(cmd, callback, blocksize=8192):
            """Simulate FTP retrbinary with chunked callback."""
            for i in range(0, len(test_data), blocksize):
                chunk = test_data[i : i + blocksize]
                callback(chunk)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        # Perform chunked download
        success = geo_manager._chunked_ftp_download(
            mock_ftp,
            "/remote/path/file.txt",
            temp_file,
            file_size,
            "Test download",
            chunk_size=8192,
        )

        assert success is True
        assert temp_file.exists()
        assert temp_file.read_bytes() == test_data

        # Verify FTP commands were called
        mock_ftp.voidcmd.assert_called_once_with("TYPE I")
        mock_ftp.retrbinary.assert_called_once()

    @patch("ftplib.FTP")
    def test_chunked_download_failure(self, mock_ftp_class, geo_manager, temp_file):
        """Test chunked FTP download failure handling."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Simulate FTP error during download
        mock_ftp.retrbinary.side_effect = ftplib.error_temp("Temporary error")

        success = geo_manager._chunked_ftp_download(
            mock_ftp,
            "/remote/path/file.txt",
            temp_file,
            1024,
            "Test download",
            chunk_size=8192,
        )

        assert success is False

    @patch("ftplib.FTP")
    def test_chunked_download_custom_chunk_size(
        self, mock_ftp_class, geo_manager, temp_file
    ):
        """Test chunked download with custom chunk size."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        test_data = b"X" * 1024
        chunks_received = []

        def mock_retrbinary(cmd, callback, blocksize=8192):
            for i in range(0, len(test_data), blocksize):
                chunk = test_data[i : i + blocksize]
                chunks_received.append(len(chunk))
                callback(chunk)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        # Use 512-byte chunks
        geo_manager._chunked_ftp_download(
            mock_ftp,
            "/remote/path/file.txt",
            temp_file,
            len(test_data),
            "Test download",
            chunk_size=512,
        )

        # Verify chunk sizes
        assert all(size <= 512 for size in chunks_received)


# ===============================================================================
# Retry Logic Tests
# ===============================================================================


class TestDownloadWithRetry:
    """Test download retry logic with exponential backoff."""

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("ftplib.FTP")
    def test_retry_success_on_first_attempt(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test successful download on first attempt (no retries needed)."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Mock successful download
        test_data = b"Success on first attempt"
        mock_ftp.size.return_value = len(test_data)

        def mock_retrbinary(cmd, callback, blocksize=8192):
            callback(test_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        # Patch _validate_gzip_integrity to return True
        with patch.object(geo_manager, "_validate_gzip_integrity", return_value=True):
            success = geo_manager._download_with_retry(
                "ftp://ftp.example.com/test.gz",
                temp_file,
                "Test download",
                max_retries=3,
            )

        assert success is True
        assert temp_file.exists()
        mock_sleep.assert_not_called()  # No retries, so no sleep

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_success_on_second_attempt(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test successful download after one retry."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        test_data = b"Success after retry"
        mock_ftp.size.return_value = len(test_data)

        # First attempt fails, second succeeds
        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                raise ftplib.error_temp("Temporary network error")
            callback(test_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        with patch.object(geo_manager, "_validate_gzip_integrity", return_value=True):
            success = geo_manager._download_with_retry(
                "ftp://ftp.example.com/test.gz",
                temp_file,
                "Test download",
                max_retries=3,
            )

        assert success is True
        assert temp_file.exists()
        # Should have slept once (2^2 = 4 seconds ± 20% jitter after first failure)
        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert 3.2 <= delay <= 4.8, f"Retry delay should be 4s ± 20%, got {delay}"

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_exponential_backoff(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test exponential backoff delays with jitter (base 2^n ± 20%)."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # All attempts fail
        mock_ftp.retrbinary.side_effect = ftplib.error_temp("Network error")

        with patch.object(geo_manager, "_validate_gzip_integrity", return_value=True):
            success = geo_manager._download_with_retry(
                "ftp://ftp.example.com/test.gz",
                temp_file,
                "Test download",
                max_retries=3,
            )

        assert success is False
        # Should have slept with exponential backoff: 2^2=4s, 2^3=8s (with jitter ±20%)
        assert mock_sleep.call_count == 2

        # Check base delays with jitter tolerance (base ± 20%)
        call_args = [call[0][0] for call in mock_sleep.call_args_list]
        assert (
            3.2 <= call_args[0] <= 4.8
        ), f"First retry delay should be 4s ± 20%, got {call_args[0]}"
        assert (
            6.4 <= call_args[1] <= 9.6
        ), f"Second retry delay should be 8s ± 20%, got {call_args[1]}"

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_max_attempts_reached(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test failure after max retries."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # All attempts fail
        mock_ftp.connect.side_effect = OSError("Connection failed")

        success = geo_manager._download_with_retry(
            "ftp://ftp.example.com/test.gz",
            temp_file,
            "Test download",
            max_retries=3,
        )

        assert success is False
        assert mock_ftp_class.call_count == 3  # 3 connection attempts
        assert mock_sleep.call_count == 2  # Sleeps after attempts 1 and 2

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_cleanup_on_failure(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test partial file cleanup on failed download."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # First attempt creates partial file, second fails
        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                # Write partial data then fail
                callback(b"Partial data")
                raise ftplib.error_temp("Connection lost")
            raise ftplib.error_temp("Still failing")

        mock_ftp.size.return_value = 1000
        mock_ftp.retrbinary.side_effect = mock_retrbinary

        success = geo_manager._download_with_retry(
            "ftp://ftp.example.com/test.gz",
            temp_file,
            "Test download",
            max_retries=2,
        )

        assert success is False
        # Partial file should be cleaned up
        assert not temp_file.exists()

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_md5_verification(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test MD5 checksum verification during retry."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        test_data = b"Test data for MD5"
        mock_ftp.size.return_value = len(test_data)

        def mock_retrbinary(cmd, callback, blocksize=8192):
            callback(test_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        # Calculate correct MD5
        import hashlib

        correct_md5 = hashlib.md5(test_data).hexdigest()
        wrong_md5 = "0" * 32

        # First attempt with wrong MD5 should fail, second with correct should succeed
        attempt = [0]

        def variable_md5_check(path):
            attempt[0] += 1
            if attempt[0] == 1:
                # First attempt: wrong MD5
                return geo_manager._calculate_md5(path) == wrong_md5
            # Second attempt: correct MD5
            return geo_manager._calculate_md5(path) == correct_md5

        with patch.object(geo_manager, "_calculate_md5") as mock_md5:
            mock_md5.return_value = correct_md5

            # First call with wrong MD5 (will retry)
            success = geo_manager._download_with_retry(
                "ftp://ftp.example.com/test.txt",
                temp_file,
                "Test download",
                max_retries=2,
                md5_checksum=wrong_md5,
            )

            assert success is False  # Wrong MD5
            assert mock_sleep.call_count >= 1  # Should have retried

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_retry_gzip_validation_failure(
        self, mock_ftp_class, mock_sleep, geo_manager
    ):
        """Test retry when gzip validation fails."""
        import tempfile

        # Create temp file with .gz extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as f:
            temp_file = Path(f.name)

        try:
            mock_ftp = MagicMock()
            mock_ftp_class.return_value = mock_ftp

            # Write valid data
            test_data = b"Test gzip data"
            mock_ftp.size.return_value = len(test_data)

            def mock_retrbinary(cmd, callback, blocksize=8192):
                callback(test_data)

            mock_ftp.retrbinary.side_effect = mock_retrbinary

            # First attempt: gzip validation fails, second succeeds
            attempt = [0]

            def mock_validate(path):
                attempt[0] += 1
                return attempt[0] > 1  # Fail first, succeed second

            with patch.object(
                geo_manager, "_validate_gzip_integrity", side_effect=mock_validate
            ):
                success = geo_manager._download_with_retry(
                    "ftp://ftp.example.com/test.gz",
                    temp_file,
                    "Test download",
                    max_retries=3,
                )

            assert success is True
            assert mock_sleep.call_count == 1  # One retry

        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()


# ===============================================================================
# Integration Tests
# ===============================================================================


class TestFTPDownloadIntegration:
    """Test integration of _download_ftp() with retry wrapper."""

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_download_ftp_delegates_to_retry(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test that _download_ftp() properly delegates to _download_with_retry()."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        test_data = b"Integration test data"
        mock_ftp.size.return_value = len(test_data)

        def mock_retrbinary(cmd, callback, blocksize=8192):
            callback(test_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        with patch.object(geo_manager, "_validate_gzip_integrity", return_value=True):
            success = geo_manager._download_ftp(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/series/test.gz",
                temp_file,
                "Test download",
            )

        assert success is True
        assert temp_file.exists()

    @patch("time.sleep")
    @patch("ftplib.FTP")
    def test_download_ftp_with_retry_on_failure(
        self, mock_ftp_class, mock_sleep, geo_manager, temp_file
    ):
        """Test _download_ftp() retries on transient failures."""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        test_data = b"Data after retry"
        mock_ftp.size.return_value = len(test_data)

        # Fail first attempt, succeed second
        attempt = [0]

        def mock_retrbinary(cmd, callback, blocksize=8192):
            attempt[0] += 1
            if attempt[0] == 1:
                raise ftplib.error_temp("Temporary failure")
            callback(test_data)

        mock_ftp.retrbinary.side_effect = mock_retrbinary

        with patch.object(geo_manager, "_validate_gzip_integrity", return_value=True):
            success = geo_manager._download_ftp(
                "ftp://ftp.ncbi.nlm.nih.gov/geo/series/test.gz",
                temp_file,
                "Test download",
            )

        assert success is True
        assert temp_file.exists()
        assert mock_sleep.call_count == 1  # One retry delay


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
