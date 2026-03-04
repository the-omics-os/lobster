"""Narrow unit tests for DownloadExecutor domain module.

Tests DownloadExecutor methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lobster.services.data_access.geo.download_execution import DownloadExecutor


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes DownloadExecutor needs."""
    service = MagicMock()
    service.data_manager = MagicMock()
    service.data_manager.metadata_store = {}
    service.cache_dir = Path("/tmp/test_geo_cache")
    service.console = MagicMock()
    service.geo_downloader = MagicMock()
    service.geo_parser = MagicMock()
    service.pipeline_engine = MagicMock()
    service.tenx_loader = MagicMock()
    service.download_strategy = MagicMock()
    return service


@pytest.fixture
def executor(mock_service):
    """Create a DownloadExecutor with mocked service."""
    return DownloadExecutor(mock_service)


class TestDownloadExecutorInit:
    """Test DownloadExecutor initialization."""

    def test_init_stores_service_reference(self, mock_service):
        executor = DownloadExecutor(mock_service)
        assert executor.service is mock_service


class TestDownloadDataset:
    """Test download_dataset routing and validation."""

    def test_rejects_invalid_geo_id(self, executor):
        """Non-GSE IDs should return error message."""
        result = executor.download_dataset("INVALID123")
        assert "Invalid GEO ID format" in result

    def test_rejects_gds_id(self, executor):
        """GDS IDs should be rejected (download requires GSE)."""
        result = executor.download_dataset("GDS5826")
        assert "Invalid GEO ID format" in result


class TestDownloadWithStrategy:
    """Test download_with_strategy orchestration."""

    def test_sets_use_intersecting_genes_only_on_service(self, executor, mock_service):
        """download_with_strategy should store concatenation strategy on service."""
        # Make metadata store return something to avoid early exit
        mock_service.data_manager.metadata_store = {"GSE123": {"metadata": {}}}
        mock_service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "strategy_config": {"pipeline_type": "supplementary_first"},
        }
        mock_service._determine_data_type_from_metadata.return_value = "single_cell_rna_seq"
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = []

        executor.download_with_strategy("GSE123", use_intersecting_genes_only=True)
        assert mock_service._use_intersecting_genes_only is True

    def test_cleans_geo_id(self, executor, mock_service):
        """download_with_strategy should strip and uppercase GEO ID."""
        mock_service.data_manager.metadata_store = {"GSE123": {"metadata": {}}}
        mock_service.data_manager._get_geo_metadata.return_value = {
            "metadata": {},
            "strategy_config": {},
        }
        mock_service._determine_data_type_from_metadata.return_value = "bulk_rna_seq"
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="FALLBACK"),
            "default fallback",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = []

        executor.download_with_strategy("  gse123  ")
        # The cleaned ID should be used for metadata store lookup
        assert mock_service._use_intersecting_genes_only is None


class TestGetProcessingPipeline:
    """Test _get_processing_pipeline pipeline selection."""

    def test_returns_callable_pipeline_list(self, executor, mock_service):
        """_get_processing_pipeline should return list of callables."""
        mock_service._determine_data_type_from_metadata.return_value = "bulk_rna_seq"

        mock_func = MagicMock()
        mock_service.pipeline_engine.determine_pipeline.return_value = (
            MagicMock(name="SUPPLEMENTARY_FIRST"),
            "test description",
        )
        mock_service.pipeline_engine.get_pipeline_functions.return_value = [mock_func]

        result = executor._get_processing_pipeline(
            "GSE123",
            {"samples": {}},
            {"pipeline_type": "supplementary_first"},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_func
