"""Narrow unit tests for MetadataFetcher domain module.

Tests MetadataFetcher methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes MetadataFetcher needs."""
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
def fetcher(mock_service):
    """Create a MetadataFetcher with mocked service."""
    return MetadataFetcher(mock_service)


class TestMetadataFetcherInit:
    """Test MetadataFetcher initialization."""

    def test_init_stores_service_reference(self, mock_service):
        fetcher = MetadataFetcher(mock_service)
        assert fetcher.service is mock_service


class TestFetchMetadataOnly:
    """Test fetch_metadata_only routing logic."""

    def test_routes_gse_to_fetch_gse_metadata(self, fetcher):
        """GSE IDs should be routed to _fetch_gse_metadata."""
        fetcher._fetch_gse_metadata = MagicMock(return_value=({"title": "test"}, {}))
        result = fetcher.fetch_metadata_only("GSE194247")
        fetcher._fetch_gse_metadata.assert_called_once_with("GSE194247")
        assert result == ({"title": "test"}, {})

    def test_routes_gds_to_fetch_gds_metadata(self, fetcher):
        """GDS IDs should be routed to _fetch_gds_metadata_and_convert."""
        fetcher._fetch_gds_metadata_and_convert = MagicMock(
            return_value=({"title": "test"}, {})
        )
        result = fetcher.fetch_metadata_only("GDS5826")
        fetcher._fetch_gds_metadata_and_convert.assert_called_once_with("GDS5826")

    def test_returns_none_tuple_for_invalid_ids(self, fetcher):
        """Invalid IDs (not GSE/GDS) should return (None, None)."""
        result = fetcher.fetch_metadata_only("INVALID123")
        assert result == (None, None)

    def test_cleans_and_uppercases_geo_id(self, fetcher):
        """GEO IDs should be stripped and uppercased."""
        fetcher._fetch_gse_metadata = MagicMock(return_value=({"title": "test"}, {}))
        fetcher.fetch_metadata_only("  gse194247  ")
        fetcher._fetch_gse_metadata.assert_called_once_with("GSE194247")


class TestSafelyExtractMetadataField:
    """Test _safely_extract_metadata_field edge cases."""

    def test_returns_default_for_missing_metadata_attr(self, fetcher):
        """Objects without metadata attr should return default."""
        obj = MagicMock(spec=[])  # No metadata attribute
        result = fetcher._safely_extract_metadata_field(obj, "title", "fallback")
        assert result == "fallback"

    def test_returns_default_for_missing_key(self, fetcher):
        """Missing keys should return default."""
        obj = MagicMock()
        obj.metadata = {"other_field": ["value"]}
        result = fetcher._safely_extract_metadata_field(obj, "title", "N/A")
        assert result == "N/A"

    def test_joins_list_values(self, fetcher):
        """List values should be joined with comma separator."""
        obj = MagicMock()
        obj.metadata = {"title": ["Part 1", "Part 2"]}
        result = fetcher._safely_extract_metadata_field(obj, "title")
        assert result == "Part 1, Part 2"

    def test_returns_string_for_scalar_value(self, fetcher):
        """Scalar values should be returned as strings."""
        obj = MagicMock()
        obj.metadata = {"title": "Single Title"}
        result = fetcher._safely_extract_metadata_field(obj, "title")
        assert result == "Single Title"

    def test_returns_default_for_empty_list(self, fetcher):
        """Empty list values should return default."""
        obj = MagicMock()
        obj.metadata = {"title": []}
        result = fetcher._safely_extract_metadata_field(obj, "title", "default")
        assert result == "default"


class TestDetectSampleTypes:
    """Test _detect_sample_types classification logic."""

    def test_detects_rna_from_library_strategy(self, fetcher):
        """RNA-Seq library strategy should classify as rna."""
        metadata = {
            "samples": {
                "GSM1": {"library_strategy": "RNA-Seq"},
                "GSM2": {"library_strategy": "RNA-Seq"},
            }
        }
        result = fetcher._detect_sample_types(metadata)
        assert "rna" in result
        assert "GSM1" in result["rna"]
        assert "GSM2" in result["rna"]

    def test_detects_protein_from_characteristics(self, fetcher):
        """Antibody capture in characteristics should classify as protein."""
        metadata = {
            "samples": {
                "GSM3": {"characteristics_ch1": ["library type: antibody capture"]},
            }
        }
        result = fetcher._detect_sample_types(metadata)
        assert "protein" in result
        assert "GSM3" in result["protein"]

    def test_returns_empty_dict_for_no_samples(self, fetcher):
        """Empty samples dict should return empty result."""
        result = fetcher._detect_sample_types({"samples": {}})
        assert result == {}

    def test_returns_empty_dict_for_missing_samples_key(self, fetcher):
        """Missing 'samples' key should return empty result."""
        result = fetcher._detect_sample_types({})
        assert result == {}
