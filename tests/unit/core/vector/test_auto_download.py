"""
Unit tests for ChromaDB backend ontology auto-download functionality.

Tests cover: tarball URL correctness, cache directory location, cache
hit/miss paths, graceful network failure, corrupt tarball handling,
and atomic download cleanup.  All tests use mocked dependencies with
no real network calls.

The ChromaDB module is mocked via method-level patches so tests run
regardless of whether chromadb is installed.
"""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.core.vector.backends.chromadb_backend import (
    ONTOLOGY_CACHE_DIR,
    ONTOLOGY_TARBALLS,
    _download_with_progress,
)


# ---------------------------------------------------------------------------
# TestOntologyTarballs -- static constant validation
# ---------------------------------------------------------------------------


class TestOntologyTarballs:
    """Validate ONTOLOGY_TARBALLS constants are correct."""

    def test_tarball_urls_cover_all_three_ontologies(self):
        """ONTOLOGY_TARBALLS has exactly 3 entries for mondo, uberon, cell_ontology."""
        assert len(ONTOLOGY_TARBALLS) == 3
        assert "mondo_v2024_01" in ONTOLOGY_TARBALLS
        assert "uberon_v2024_01" in ONTOLOGY_TARBALLS
        assert "cell_ontology_v2024_01" in ONTOLOGY_TARBALLS

    def test_tarball_urls_use_correct_s3_bucket(self):
        """All URLs start with the lobster-ontology-data S3 bucket prefix."""
        expected_prefix = "https://lobster-ontology-data.s3.amazonaws.com/v1/"
        for url in ONTOLOGY_TARBALLS.values():
            assert url.startswith(expected_prefix), f"URL does not match: {url}"

    def test_tarball_urls_use_correct_filenames(self):
        """URLs end with the expected tarball filenames."""
        expected_filenames = {
            "mondo_v2024_01": "mondo_sapbert_768.tar.gz",
            "uberon_v2024_01": "uberon_sapbert_768.tar.gz",
            "cell_ontology_v2024_01": "cell_ontology_sapbert_768.tar.gz",
        }
        for key, url in ONTOLOGY_TARBALLS.items():
            assert url.endswith(expected_filenames[key]), (
                f"{key}: URL should end with {expected_filenames[key]}, got {url}"
            )

    def test_cache_dir_is_under_home_lobster(self):
        """ONTOLOGY_CACHE_DIR ends with .lobster/ontology_cache."""
        parts = ONTOLOGY_CACHE_DIR.parts
        assert parts[-2] == ".lobster"
        assert parts[-1] == "ontology_cache"


# ---------------------------------------------------------------------------
# Helper: create a ChromaDBBackend with the chromadb client pre-mocked
# ---------------------------------------------------------------------------


def _make_backend_with_mock_client(tmp_path, collection_count=0):
    """
    Create a ChromaDBBackend that does NOT require real chromadb.

    The internal _client is replaced with a MagicMock after construction.
    collection_exists / count are patched to return values based on
    *collection_count*.

    Returns:
        tuple: (backend, mock_client, mock_collection)
    """
    from lobster.core.vector.backends.chromadb_backend import ChromaDBBackend

    persist = tmp_path / "vector_store"
    persist.mkdir(parents=True, exist_ok=True)
    backend = ChromaDBBackend(persist_path=str(persist))

    # Replace the lazy client with a mock
    mock_collection = MagicMock()
    mock_collection.count.return_value = collection_count
    mock_collection.add = MagicMock()

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    if collection_count > 0:
        mock_client.get_collection.return_value = mock_collection
    else:
        mock_client.get_collection.side_effect = ValueError("not found")
    mock_client.list_collections.return_value = []

    backend._client = mock_client
    return backend, mock_client, mock_collection


# ---------------------------------------------------------------------------
# TestEnsureOntologyData -- auto-download logic
# ---------------------------------------------------------------------------


class TestEnsureOntologyData:
    """Tests for ChromaDBBackend._ensure_ontology_data() method."""

    def test_returns_false_for_non_ontology_collection(self, tmp_path):
        """Non-ontology collection names skip auto-download immediately."""
        backend, _, _ = _make_backend_with_mock_client(tmp_path)
        result = backend._ensure_ontology_data("my_custom_collection")
        assert result is False

    def test_returns_true_if_collection_already_populated(self, tmp_path):
        """If collection already has data, no download is attempted."""
        backend, mock_client, mock_coll = _make_backend_with_mock_client(
            tmp_path, collection_count=5000
        )

        with patch(
            "lobster.core.vector.backends.chromadb_backend._download_with_progress"
        ) as mock_download:
            result = backend._ensure_ontology_data("mondo_v2024_01")
            assert result is True
            mock_download.assert_not_called()

    def test_downloads_tarball_on_cache_miss(self, tmp_path):
        """When no cached tarball exists, downloads from S3 URL."""
        backend, _, _ = _make_backend_with_mock_client(tmp_path)
        cache_dir = tmp_path / "ontology_cache"

        with (
            patch(
                "lobster.core.vector.backends.chromadb_backend.ONTOLOGY_CACHE_DIR",
                cache_dir,
            ),
            patch(
                "lobster.core.vector.backends.chromadb_backend._download_with_progress"
            ) as mock_download,
        ):
            # fake_download creates a file that will fail tarball extraction
            def fake_download(url, dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake")

            mock_download.side_effect = fake_download

            # Will fail during tarball extraction but download IS called
            result = backend._ensure_ontology_data("mondo_v2024_01")

            mock_download.assert_called_once()
            call_url = mock_download.call_args[0][0]
            assert call_url == ONTOLOGY_TARBALLS["mondo_v2024_01"]

    def test_skips_download_if_tarball_cached(self, tmp_path):
        """If tarball file exists in cache, download is not called."""
        backend, _, _ = _make_backend_with_mock_client(tmp_path)
        cache_dir = tmp_path / "ontology_cache"
        cache_dir.mkdir(parents=True)

        # Create a fake cached tarball
        tarball_path = cache_dir / "mondo_sapbert_768.tar.gz"
        tarball_path.write_bytes(b"fake tarball data")

        with (
            patch(
                "lobster.core.vector.backends.chromadb_backend.ONTOLOGY_CACHE_DIR",
                cache_dir,
            ),
            patch(
                "lobster.core.vector.backends.chromadb_backend._download_with_progress"
            ) as mock_download,
        ):
            # Will fail at extraction (corrupt data) but download must NOT be called
            result = backend._ensure_ontology_data("mondo_v2024_01")

            mock_download.assert_not_called()

    def test_graceful_failure_on_network_error(self, tmp_path):
        """Network error returns False (no crash) and logs a warning."""
        import requests

        backend, _, _ = _make_backend_with_mock_client(tmp_path)
        cache_dir = tmp_path / "ontology_cache"

        with (
            patch(
                "lobster.core.vector.backends.chromadb_backend.ONTOLOGY_CACHE_DIR",
                cache_dir,
            ),
            patch(
                "lobster.core.vector.backends.chromadb_backend._download_with_progress",
                side_effect=requests.ConnectionError("Network unreachable"),
            ),
            patch(
                "lobster.core.vector.backends.chromadb_backend.logger"
            ) as mock_logger,
        ):
            result = backend._ensure_ontology_data("mondo_v2024_01")

            assert result is False
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "Failed to download" in warning_msg

    def test_graceful_failure_on_corrupt_tarball(self, tmp_path):
        """Corrupt tarball returns False and removes the bad file from cache."""
        backend, _, _ = _make_backend_with_mock_client(tmp_path)
        cache_dir = tmp_path / "ontology_cache"
        cache_dir.mkdir(parents=True)

        # Create a corrupt cached tarball
        tarball_path = cache_dir / "mondo_sapbert_768.tar.gz"
        tarball_path.write_bytes(b"this is not a valid tarball")

        with (
            patch(
                "lobster.core.vector.backends.chromadb_backend.ONTOLOGY_CACHE_DIR",
                cache_dir,
            ),
            patch(
                "lobster.core.vector.backends.chromadb_backend.logger"
            ) as mock_logger,
        ):
            result = backend._ensure_ontology_data("mondo_v2024_01")

            assert result is False
            # Corrupt tarball should be removed from cache
            assert not tarball_path.exists()
            mock_logger.warning.assert_called()

    def test_atomic_download_cleans_up_tmp_on_failure(self, tmp_path):
        """When download fails mid-stream, no .tmp file is left behind."""
        import requests

        dest = tmp_path / "ontology_cache" / "test_file.tar.gz"
        tmp_file = dest.with_suffix(".tmp")

        with patch(
            "requests.get",
            side_effect=requests.ConnectionError("Connection reset"),
        ):
            with pytest.raises(requests.ConnectionError):
                _download_with_progress(
                    "https://example.com/test.tar.gz", dest
                )

        assert not tmp_file.exists()
        assert not dest.exists()


# ---------------------------------------------------------------------------
# TestDownloadWithProgress -- download function tests
# ---------------------------------------------------------------------------


class TestDownloadWithProgress:
    """Tests for the _download_with_progress() module-level function."""

    def test_downloads_to_temp_then_renames(self, tmp_path):
        """File is written to .tmp first, then renamed to final dest."""
        dest = tmp_path / "test_download.tar.gz"

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "12"}
        mock_response.iter_content.return_value = [b"hello", b" world!"]
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            _download_with_progress("https://example.com/file.tar.gz", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"hello world!"
        # .tmp should not exist after successful download
        assert not dest.with_suffix(".tmp").exists()

    def test_http_error_raises(self, tmp_path):
        """HTTP errors from raise_for_status are propagated."""
        import requests

        dest = tmp_path / "test_download.tar.gz"

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found"
        )

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError, match="404"):
                _download_with_progress(
                    "https://example.com/missing.tar.gz", dest
                )

        assert not dest.exists()
        assert not dest.with_suffix(".tmp").exists()

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories are created if they don't exist."""
        dest = tmp_path / "deep" / "nested" / "dir" / "file.tar.gz"

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content.return_value = [b"hello"]
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            _download_with_progress("https://example.com/file.tar.gz", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"hello"
