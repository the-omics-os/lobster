"""Tests for temp file cleanup on parser failure in _process_tar_file.

Verifies that extraction directories are cleaned up when _process_tar_file
raises an exception, but preserved on successful processing.
"""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from lobster.services.data_access.geo_service import GEOService


@pytest.fixture
def geo_service(tmp_path):
    """Create a GEOService with mocked dependencies and tmp_path as cache_dir."""
    with patch.object(GEOService, "__init__", lambda self: None):
        service = GEOService()
        service.cache_dir = tmp_path
        service.geo_downloader = MagicMock()
        service.geo_parser = MagicMock()
        return service


def _create_valid_tar(tar_path: Path, filename: str = "data.txt", content: str = "data"):
    """Create a minimal valid tar file."""
    tmp_file = tar_path.parent / filename
    tmp_file.write_text(content)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(tmp_file, arcname=filename)
    tmp_file.unlink()


class TestTempCleanupOnFailure:
    """Tests that extraction directories are cleaned up on exception."""

    def test_extract_dir_removed_on_download_exception(self, geo_service, tmp_path):
        """When download raises, {gse_id}_extracted should be removed if it exists."""
        gse_id = "GSE12345"
        extract_dir = tmp_path / f"{gse_id}_extracted"
        extract_dir.mkdir()
        (extract_dir / "some_file.txt").write_text("data")

        # Mock download to raise an exception
        geo_service.geo_downloader.download_file.side_effect = RuntimeError("Network error")

        result = geo_service._process_tar_file(
            f"https://ftp.ncbi.nlm.nih.gov/{gse_id}_RAW.tar", gse_id
        )

        assert result is None
        assert not extract_dir.exists(), "extract_dir should be cleaned up on failure"

    def test_both_dirs_removed_on_exception(self, geo_service, tmp_path):
        """When exception occurs, both extract_dir and nested_extract_dir are removed."""
        gse_id = "GSE12345"
        extract_dir = tmp_path / f"{gse_id}_extracted"
        nested_dir = tmp_path / f"{gse_id}_nested_extracted"
        extract_dir.mkdir()
        nested_dir.mkdir()
        (nested_dir / "nested_file.txt").write_text("data")

        # Create a valid tar so extraction starts, then mock parser to raise
        _create_valid_tar(tmp_path / f"{gse_id}_RAW.tar", "big_file.txt", "x" * 200000)

        geo_service.geo_parser.parse_expression_file.side_effect = RuntimeError("Parse crash")

        # Also need to mock parse_10x_data to avoid it succeeding
        geo_service.geo_parser.parse_10x_data = MagicMock(return_value=None)

        # Patch BulkRNASeqService to raise ValueError (no quant files)
        with patch(
            "lobster.services.data_access.geo_service.BulkRNASeqService"
        ) as mock_bulk:
            mock_bulk.return_value._detect_quantification_tool.side_effect = ValueError(
                "No quant files"
            )
            result = geo_service._process_tar_file(
                f"https://ftp.ncbi.nlm.nih.gov/{gse_id}_RAW.tar", gse_id
            )

        assert result is None
        assert not extract_dir.exists(), "extract_dir should be cleaned up"
        assert not nested_dir.exists(), "nested_extract_dir should be cleaned up"


class TestTempPreservationOnSuccess:
    """Tests that extraction directories are preserved on successful processing."""

    def test_extract_dir_preserved_on_success(self, geo_service, tmp_path):
        """On successful return, extraction directories should remain for caching."""
        gse_id = "GSE99999"
        extract_dir = tmp_path / f"{gse_id}_extracted"
        # Don't pre-create -- let the method create it during extraction

        # Create a valid tar with a large expression file (>100KB to pass size check)
        _create_valid_tar(
            tmp_path / f"{gse_id}_RAW.tar",
            "expression_matrix.txt",
            "gene\tsample1\tsample2\n" + "GENE1\t1.0\t2.0\n" * 10000,
        )

        # Mock the parser to return valid data
        mock_df = pd.DataFrame(
            {"sample1": [1.0, 3.0], "sample2": [2.0, 4.0]},
            index=["GENE1", "GENE2"],
        )
        geo_service.geo_parser.parse_expression_file.return_value = mock_df

        # Patch BulkRNASeqService to raise ValueError (no quant files)
        with patch(
            "lobster.services.data_access.geo_service.BulkRNASeqService"
        ) as mock_bulk:
            mock_bulk.return_value._detect_quantification_tool.side_effect = ValueError(
                "No quant files"
            )
            result = geo_service._process_tar_file(
                f"https://ftp.ncbi.nlm.nih.gov/{gse_id}_RAW.tar", gse_id
            )

        assert result is not None
        assert extract_dir.exists(), "extract_dir should be preserved on success"


class TestCleanupGracefulHandling:
    """Tests that cleanup handles edge cases gracefully."""

    def test_cleanup_handles_nonexistent_dirs(self, geo_service, tmp_path):
        """Cleanup should not fail if directories don't exist."""
        gse_id = "GSE12345"
        # Do NOT create any dirs

        # Mock download to raise so we hit the exception path
        geo_service.geo_downloader.download_file.side_effect = RuntimeError("Network error")

        # Should not raise even though dirs don't exist
        result = geo_service._process_tar_file(
            f"https://ftp.ncbi.nlm.nih.gov/{gse_id}_RAW.tar", gse_id
        )
        assert result is None
