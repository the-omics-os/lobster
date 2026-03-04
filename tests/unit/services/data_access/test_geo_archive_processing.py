"""Narrow unit tests for ArchiveProcessor domain module.

Tests ArchiveProcessor methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from lobster.services.data_access.geo.archive_processing import ArchiveProcessor


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes ArchiveProcessor needs."""
    service = MagicMock()
    service.data_manager = MagicMock()
    service.cache_dir = Path("/tmp/test_geo_cache")
    service.console = MagicMock()
    service.geo_downloader = MagicMock()
    service.geo_parser = MagicMock()
    service.pipeline_engine = MagicMock()
    service.tenx_loader = MagicMock()
    return service


@pytest.fixture
def processor(mock_service):
    """Create an ArchiveProcessor with mocked service."""
    return ArchiveProcessor(mock_service)


class TestArchiveProcessorInit:
    """Test ArchiveProcessor initialization."""

    def test_init_stores_service_reference(self, mock_service):
        processor = ArchiveProcessor(mock_service)
        assert processor.service is mock_service


class TestDetectKallistoSalmonFiles:
    """Test _detect_kallisto_salmon_files detection logic."""

    def test_identifies_salmon_output_patterns(self, processor):
        """Salmon quant.sf files should be detected."""
        files = [
            "/data/sample1/quant.sf",
            "/data/sample2/quant.sf",
            "/data/readme.txt",
        ]
        has_quant, tool_type, matched, estimated = (
            processor._detect_kallisto_salmon_files(files)
        )
        assert has_quant is True
        assert tool_type == "salmon"
        assert len(matched) == 2

    def test_identifies_kallisto_output_patterns(self, processor):
        """Kallisto abundance.tsv files should be detected."""
        files = [
            "/data/sample1/abundance.tsv",
            "/data/sample2/abundance.h5",
        ]
        has_quant, tool_type, matched, estimated = (
            processor._detect_kallisto_salmon_files(files)
        )
        assert has_quant is True
        assert tool_type == "kallisto"
        assert len(matched) == 2

    def test_returns_false_for_unrecognized_patterns(self, processor):
        """Non-quantification files should return no detection."""
        files = [
            "/data/expression_matrix.csv",
            "/data/metadata.txt",
            "/data/barcodes.tsv.gz",
        ]
        has_quant, tool_type, matched, estimated = (
            processor._detect_kallisto_salmon_files(files)
        )
        assert has_quant is False
        assert tool_type == ""
        assert matched == []
        assert estimated == 0

    def test_detects_mixed_kallisto_salmon(self, processor):
        """Mix of Kallisto and Salmon files should be detected as mixed."""
        files = [
            "/data/sample1/abundance.tsv",
            "/data/sample2/quant.sf",
        ]
        has_quant, tool_type, matched, estimated = (
            processor._detect_kallisto_salmon_files(files)
        )
        assert has_quant is True
        assert tool_type == "mixed"
        assert len(matched) == 2

    def test_empty_file_list_returns_false(self, processor):
        """Empty file list should return no detection."""
        has_quant, tool_type, matched, estimated = (
            processor._detect_kallisto_salmon_files([])
        )
        assert has_quant is False
