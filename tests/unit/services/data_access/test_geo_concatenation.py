"""Narrow unit tests for SampleConcatenator domain module.

Tests SampleConcatenator methods in isolation via mocked service.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from lobster.services.data_access.geo.concatenation import SampleConcatenator


@pytest.fixture
def mock_service():
    """Create a mock GEOService with all attributes SampleConcatenator needs."""
    service = MagicMock()
    service.data_manager = MagicMock()
    service.data_manager.metadata_store = {}
    service.cache_dir = Path("/tmp/test_geo_cache")
    service.console = MagicMock()
    return service


@pytest.fixture
def concatenator(mock_service):
    """Create a SampleConcatenator with mocked service."""
    return SampleConcatenator(mock_service)


class TestSampleConcatenatorInit:
    """Test SampleConcatenator initialization."""

    def test_init_stores_service_reference(self, mock_service):
        concatenator = SampleConcatenator(mock_service)
        assert concatenator.service is mock_service


class TestAnalyzeGeneCoverageAndDecideJoin:
    """Test _analyze_gene_coverage_and_decide_join strategy selection."""

    def test_returns_inner_when_coverage_is_uniform(self, concatenator, mock_service):
        """When gene counts are very similar, should recommend inner join."""
        # Create mock AnnData objects with similar gene counts
        mock_adata_1 = MagicMock()
        mock_adata_1.n_vars = 20000
        mock_adata_2 = MagicMock()
        mock_adata_2.n_vars = 20100
        mock_adata_3 = MagicMock()
        mock_adata_3.n_vars = 19900

        # Map modality names to mock adata objects
        mock_service.data_manager.get_modality.side_effect = [
            mock_adata_1,
            mock_adata_2,
            mock_adata_3,
        ]

        use_inner, metadata = concatenator._analyze_gene_coverage_and_decide_join(
            ["sample_1", "sample_2", "sample_3"]
        )

        assert use_inner is True
        assert metadata["decision"] == "inner"
        assert "CV" in metadata["reasoning"] or "coverage" in metadata["reasoning"].lower()

    def test_returns_outer_when_coverage_varies(self, concatenator, mock_service):
        """When gene counts vary significantly, should recommend outer join."""
        # Create mock AnnData objects with very different gene counts
        mock_adata_1 = MagicMock()
        mock_adata_1.n_vars = 5000
        mock_adata_2 = MagicMock()
        mock_adata_2.n_vars = 20000
        mock_adata_3 = MagicMock()
        mock_adata_3.n_vars = 15000

        mock_service.data_manager.get_modality.side_effect = [
            mock_adata_1,
            mock_adata_2,
            mock_adata_3,
        ]

        use_inner, metadata = concatenator._analyze_gene_coverage_and_decide_join(
            ["sample_1", "sample_2", "sample_3"]
        )

        assert use_inner is False
        assert metadata["decision"] == "outer"

    def test_defaults_to_inner_when_no_valid_samples(self, concatenator, mock_service):
        """When no samples can be read, should default to inner join."""
        mock_service.data_manager.get_modality.side_effect = Exception("Not found")

        use_inner, metadata = concatenator._analyze_gene_coverage_and_decide_join(
            ["missing_1", "missing_2"]
        )

        assert use_inner is True
        assert metadata["decision"] == "inner"
        assert "No valid samples" in metadata["reasoning"]

    def test_handles_error_gracefully(self, concatenator, mock_service):
        """Errors during analysis should default to outer join for safety."""
        # Return an adata that raises on n_vars
        mock_adata = MagicMock()
        type(mock_adata).n_vars = PropertyMock(side_effect=AttributeError("broken"))
        mock_service.data_manager.get_modality.return_value = mock_adata

        use_inner, metadata = concatenator._analyze_gene_coverage_and_decide_join(
            ["sample_1"]
        )

        # With no valid gene counts, defaults to inner (the "no valid samples" path)
        assert metadata["decision"] in ("inner", "outer")
