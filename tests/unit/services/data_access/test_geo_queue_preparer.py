"""
Tests for GDS-to-GSE canonicalization in GEOQueuePreparer.

Verifies that GDS accessions are resolved to their canonical GSE ID
at queue preparation time, with original_accession preserved in metadata.
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer


@pytest.fixture
def mock_data_manager():
    """Create a mocked DataManagerV2."""
    dm = MagicMock()
    dm.metadata_store = {}
    dm.workspace_dir = "/tmp/test_workspace"
    return dm


@pytest.fixture
def preparer(mock_data_manager):
    """Create a GEOQueuePreparer with mocked dependencies."""
    return GEOQueuePreparer(data_manager=mock_data_manager)


class TestGDSCanonToGSE:
    """Test GDS-to-GSE canonicalization in prepare_queue_entry."""

    def test_gds_accession_canonicalized_to_gse(self, preparer):
        """GDS accession is resolved to GSE in the queue entry dataset_id."""
        with patch.object(preparer, "_resolve_gds_to_gse", return_value="GSE67835"):
            # Mock the parent prepare_queue_entry to avoid real API calls
            with patch(
                "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
            ) as mock_super:
                # Create a mock queue entry result
                mock_entry = MagicMock()
                mock_entry.queue_entry.dataset_id = "GSE67835"
                mock_entry.queue_entry.metadata = {}
                mock_super.return_value = mock_entry

                result = preparer.prepare_queue_entry("GDS5826")

                # Verify _resolve_gds_to_gse was called
                preparer._resolve_gds_to_gse.assert_called_once_with("GDS5826")

                # Verify super was called with the canonical GSE
                mock_super.assert_called_once_with("GSE67835", 5)

    def test_original_accession_preserved_in_metadata(self, preparer):
        """When GDS is canonicalized, original_accession is stored in metadata."""
        with patch.object(preparer, "_resolve_gds_to_gse", return_value="GSE67835"):
            with patch(
                "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
            ) as mock_super:
                mock_entry = MagicMock()
                mock_entry.queue_entry.dataset_id = "GSE67835"
                mock_entry.queue_entry.metadata = {"title": "test"}
                mock_super.return_value = mock_entry

                result = preparer.prepare_queue_entry("GDS5826")

                # original_accession should be preserved
                assert result.queue_entry.metadata["original_accession"] == "GDS5826"

    def test_gse_accession_passes_through_unchanged(self, preparer):
        """GSE accession passes through without canonicalization."""
        with patch(
            "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
        ) as mock_super:
            mock_entry = MagicMock()
            mock_entry.queue_entry.dataset_id = "GSE180759"
            mock_entry.queue_entry.metadata = {}
            mock_super.return_value = mock_entry

            result = preparer.prepare_queue_entry("GSE180759")

            # Super should be called with original GSE (no canonicalization)
            mock_super.assert_called_once_with("GSE180759", 5)

            # No original_accession should be set
            assert "original_accession" not in result.queue_entry.metadata

    def test_gpl_accession_passes_through_unchanged(self, preparer):
        """GPL accession passes through -- only GDS triggers canonicalization."""
        with patch(
            "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
        ) as mock_super:
            mock_entry = MagicMock()
            mock_entry.queue_entry.dataset_id = "GPL570"
            mock_entry.queue_entry.metadata = {}
            mock_super.return_value = mock_entry

            result = preparer.prepare_queue_entry("GPL570")

            mock_super.assert_called_once_with("GPL570", 5)
            assert "original_accession" not in result.queue_entry.metadata

    def test_case_insensitive_gds_detection(self, preparer):
        """Lowercase 'gds' prefix also triggers canonicalization."""
        with patch.object(preparer, "_resolve_gds_to_gse", return_value="GSE67835"):
            with patch(
                "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
            ) as mock_super:
                mock_entry = MagicMock()
                mock_entry.queue_entry.dataset_id = "GSE67835"
                mock_entry.queue_entry.metadata = {"title": "test"}
                mock_super.return_value = mock_entry

                result = preparer.prepare_queue_entry("gds5826")

                preparer._resolve_gds_to_gse.assert_called_once_with("gds5826")
                mock_super.assert_called_once_with("GSE67835", 5)

    def test_gds_resolution_failure_passes_through(self, preparer):
        """If GDS resolution fails, the original accession is used as-is."""
        with patch.object(
            preparer,
            "_resolve_gds_to_gse",
            side_effect=Exception("Network error"),
        ):
            with patch(
                "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
            ) as mock_super:
                mock_entry = MagicMock()
                mock_entry.queue_entry.dataset_id = "GDS5826"
                mock_entry.queue_entry.metadata = {}
                mock_super.return_value = mock_entry

                result = preparer.prepare_queue_entry("GDS5826")

                # Falls back to original accession
                mock_super.assert_called_once_with("GDS5826", 5)
                # No original_accession in metadata since no canonicalization happened
                assert "original_accession" not in result.queue_entry.metadata

    def test_gds_resolution_returns_none_passes_through(self, preparer):
        """If GDS resolution returns None, the original accession is used."""
        with patch.object(preparer, "_resolve_gds_to_gse", return_value=None):
            with patch(
                "lobster.core.interfaces.queue_preparer.IQueuePreparer.prepare_queue_entry"
            ) as mock_super:
                mock_entry = MagicMock()
                mock_entry.queue_entry.dataset_id = "GDS5826"
                mock_entry.queue_entry.metadata = {}
                mock_super.return_value = mock_entry

                result = preparer.prepare_queue_entry("GDS5826")

                mock_super.assert_called_once_with("GDS5826", 5)
                assert "original_accession" not in result.queue_entry.metadata
