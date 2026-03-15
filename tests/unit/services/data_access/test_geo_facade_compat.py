"""Facade backward compatibility tests for GEOService decomposition.

Verifies that all import patterns, mock patterns, and API surface is preserved
after the GEOService monolith was decomposed into 5 domain modules.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestImportCompatibility:
    """Verify all import patterns used by consumers still work."""

    def test_geo_service_importable(self):
        """GEOService must be importable from geo_service module."""
        from lobster.services.data_access.geo_service import GEOService

        assert GEOService is not None

    def test_geo_data_source_importable(self):
        """GEODataSource enum must be importable from geo_service."""
        from lobster.services.data_access.geo_service import GEODataSource

        assert hasattr(GEODataSource, "GEOPARSE")
        assert hasattr(GEODataSource, "SUPPLEMENTARY")

    def test_geo_result_importable(self):
        """GEOResult must be importable from geo_service."""
        from lobster.services.data_access.geo_service import GEOResult

        assert GEOResult is not None

    def test_retry_outcome_importable(self):
        """RetryOutcome must be importable from geo_service (re-export from helpers)."""
        from lobster.services.data_access.geo_service import RetryOutcome

        assert hasattr(RetryOutcome, "SUCCESS")
        assert hasattr(RetryOutcome, "EXHAUSTED")
        assert hasattr(RetryOutcome, "SOFT_FILE_MISSING")

    def test_retry_result_importable(self):
        """RetryResult must be importable from geo_service (re-export from helpers)."""
        from lobster.services.data_access.geo_service import RetryResult

        assert RetryResult is not None

    def test_archive_extensions_importable(self):
        """ARCHIVE_EXTENSIONS must be importable from geo_service."""
        from lobster.services.data_access.geo_service import ARCHIVE_EXTENSIONS

        assert isinstance(ARCHIVE_EXTENSIONS, tuple)
        assert ".tar" in ARCHIVE_EXTENSIONS

    def test_is_archive_url_importable(self):
        """_is_archive_url must be importable from geo_service."""
        from lobster.services.data_access.geo_service import _is_archive_url

        assert callable(_is_archive_url)
        assert _is_archive_url("file.tar.gz") is True
        assert _is_archive_url("file.csv") is False

    def test_score_expression_file_importable(self):
        """_score_expression_file must be importable from geo_service."""
        from lobster.services.data_access.geo_service import _score_expression_file

        assert callable(_score_expression_file)


class TestFacadeIdentity:
    """Verify facade module provides the same class."""

    def test_facade_is_same_class_as_direct(self):
        """GEOService from geo.facade must be the same class as from geo_service."""
        from lobster.services.data_access.geo.facade import GEOService as FacadeGEO
        from lobster.services.data_access.geo_service import GEOService as DirectGEO

        assert FacadeGEO is DirectGEO


class TestDomainModuleComposition:
    """Verify GEOService creates all 5 domain module instances."""

    def test_init_creates_all_domain_modules(self, tmp_path):
        """GEOService.__init__ should create all 5 domain module instances."""
        from lobster.services.data_access.geo.archive_processing import ArchiveProcessor
        from lobster.services.data_access.geo.concatenation import SampleConcatenator
        from lobster.services.data_access.geo.download_execution import DownloadExecutor
        from lobster.services.data_access.geo.matrix_parsing import MatrixParser
        from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
        from lobster.services.data_access.geo_service import GEOService

        # Create mock data_manager with real temp path for cache_dir
        mock_dm = MagicMock()
        mock_dm.cache_dir = tmp_path

        service = GEOService(data_manager=mock_dm)

        assert isinstance(service._metadata_fetcher, MetadataFetcher)
        assert isinstance(service._download_executor, DownloadExecutor)
        assert isinstance(service._archive_processor, ArchiveProcessor)
        assert isinstance(service._matrix_parser, MatrixParser)
        assert isinstance(service._concatenator, SampleConcatenator)


class TestPublicAPIDelegation:
    """Verify public methods delegate to domain modules."""

    @pytest.fixture
    def service_with_mocked_modules(self, tmp_path):
        """Create GEOService with mocked domain modules."""
        from lobster.services.data_access.geo_service import GEOService

        mock_dm = MagicMock()
        mock_dm.cache_dir = tmp_path

        service = GEOService(data_manager=mock_dm)
        return service

    def test_fetch_metadata_only_delegates_to_fetcher(
        self, service_with_mocked_modules
    ):
        """fetch_metadata_only should delegate to MetadataFetcher."""
        service = service_with_mocked_modules
        service._metadata_fetcher.fetch_metadata_only = MagicMock(
            return_value=({"title": "test"}, {})
        )

        result = service.fetch_metadata_only("GSE123")
        service._metadata_fetcher.fetch_metadata_only.assert_called_once_with("GSE123")
        assert result == ({"title": "test"}, {})

    def test_download_dataset_delegates_to_executor(self, service_with_mocked_modules):
        """download_dataset should delegate to DownloadExecutor."""
        service = service_with_mocked_modules
        service._download_executor.download_dataset = MagicMock(
            return_value="Downloaded successfully"
        )

        result = service.download_dataset("GSE123")
        service._download_executor.download_dataset.assert_called_once_with(
            "GSE123", None
        )
        assert result == "Downloaded successfully"


class TestGetAttrForwarding:
    """Verify __getattr__ forwards private methods to domain modules."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create GEOService for getattr testing."""
        from lobster.services.data_access.geo_service import GEOService

        mock_dm = MagicMock()
        mock_dm.cache_dir = tmp_path
        return GEOService(data_manager=mock_dm)

    def test_forwards_fetch_gse_metadata_to_fetcher(self, service):
        """__getattr__ should forward _fetch_gse_metadata to MetadataFetcher."""
        service._metadata_fetcher._fetch_gse_metadata = MagicMock(
            return_value=({"title": "test"}, {})
        )

        result = service._fetch_gse_metadata("GSE123")
        assert result == ({"title": "test"}, {})

    def test_raises_attribute_error_for_nonexistent(self, service):
        """__getattr__ should raise AttributeError for nonexistent attributes."""
        with pytest.raises(AttributeError, match="has no attribute"):
            service._completely_nonexistent_method_xyz()

    def test_mock_patching_private_method_works(self, service):
        """Existing test patterns using mock.patch.object must still work.

        This is critical -- many existing tests mock private methods like
        _process_supplementary_files via the GEOService object.
        """
        mock_return = MagicMock()
        with patch.object(
            service._archive_processor,
            "_process_supplementary_files",
            return_value=mock_return,
        ):
            result = service._process_supplementary_files("gse_obj", "GSE123")
            assert result is mock_return

    def test_forwards_process_tar_file_to_archive_processor(self, service):
        """__getattr__ should forward _process_tar_file to ArchiveProcessor."""
        # Verify the method is accessible through the facade
        assert hasattr(service._archive_processor, "_process_tar_file")
        method = service._process_tar_file
        assert callable(method)

    def test_forwards_is_valid_expression_matrix_to_parser(self, service):
        """__getattr__ should forward _is_valid_expression_matrix to MatrixParser."""
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = service._is_valid_expression_matrix(df)
        assert result is True
