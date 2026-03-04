"""
Tests for centralized metadata write enforcement (GSAF-03).

Ensures _enrich_geo_metadata correctly updates existing entries,
handles missing entries gracefully, and preserves existing fields.
"""

import logging

import pytest

from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def dm(tmp_path):
    """Create a DataManagerV2 with temp workspace for testing."""
    return DataManagerV2(workspace_path=tmp_path, auto_scan=False)


@pytest.fixture
def dm_with_entry(dm):
    """DataManagerV2 with a pre-stored metadata entry."""
    dm._store_geo_metadata(
        geo_id="GSE11111",
        metadata={"samples": {"GSM1": {"title": "Sample 1"}}},
        stored_by="test_setup",
        validation_result={"valid": True},
    )
    return dm


class TestEnrichGeoMetadata:
    """_enrich_geo_metadata must update entries without recreating them."""

    def test_enrich_updates_existing_entry_fields(self, dm_with_entry):
        """Enrichment adds new fields to existing entry."""
        result = dm_with_entry._enrich_geo_metadata(
            "GSE11111",
            modality_detection={"modality": "single_cell", "confidence": 0.95},
        )
        assert result is not None
        assert result["modality_detection"]["modality"] == "single_cell"
        assert result["modality_detection"]["confidence"] == 0.95

    def test_enrich_returns_none_for_missing_entry(self, dm, caplog):
        """Enrichment returns None and warns if entry does not exist."""
        with caplog.at_level(logging.WARNING):
            result = dm._enrich_geo_metadata(
                "GSE_NONEXISTENT",
                status="validated",
            )
        assert result is None
        assert any("Cannot enrich" in r.message for r in caplog.records)

    def test_enrich_preserves_existing_fields(self, dm_with_entry):
        """Enrichment preserves existing fields while adding new ones."""
        result = dm_with_entry._enrich_geo_metadata(
            "GSE11111",
            concatenation_decision={"strategy": "outer_join"},
        )
        assert result is not None
        # Original fields preserved
        assert result["metadata"]["samples"]["GSM1"]["title"] == "Sample 1"
        assert result["stored_by"] == "test_setup"
        assert result["validation_result"] == {"valid": True}
        # New field added
        assert result["concatenation_decision"]["strategy"] == "outer_join"

    def test_enrich_updates_metadata_store(self, dm_with_entry):
        """After enrichment, metadata_store contains the merged entry."""
        dm_with_entry._enrich_geo_metadata(
            "GSE11111",
            status="validated",
            validation_timestamp="2026-03-04T00:00:00",
        )
        entry = dm_with_entry.metadata_store["GSE11111"]
        assert entry["status"] == "validated"
        assert entry["validation_timestamp"] == "2026-03-04T00:00:00"
        # Original field still present
        assert entry["metadata"]["samples"]["GSM1"]["title"] == "Sample 1"

    def test_enrich_overwrites_existing_field_values(self, dm_with_entry):
        """Enrichment can overwrite existing field values."""
        result = dm_with_entry._enrich_geo_metadata(
            "GSE11111",
            validation_result={"valid": False, "reason": "platform_error"},
        )
        assert result is not None
        assert result["validation_result"]["valid"] is False
        assert result["validation_result"]["reason"] == "platform_error"

    def test_enrich_returns_updated_entry(self, dm_with_entry):
        """Enrichment returns the updated entry object."""
        result = dm_with_entry._enrich_geo_metadata(
            "GSE11111",
            multimodal_info={"is_multimodal": True},
        )
        assert result is not None
        assert result["multimodal_info"]["is_multimodal"] is True
        # Verify it matches what is in the store
        stored = dm_with_entry.metadata_store["GSE11111"]
        assert stored["multimodal_info"]["is_multimodal"] is True
