"""
Tests for metadata key standardization (GSAF-02).

Ensures MetadataEntry TypedDict uses 'validation_result' consistently
and _store_geo_metadata handles the key contract correctly.
"""

import logging
import tempfile
from typing import get_type_hints

import pytest

from lobster.core.data_manager_v2 import DataManagerV2, MetadataEntry


@pytest.fixture
def dm(tmp_path):
    """Create a DataManagerV2 with temp workspace for testing."""
    return DataManagerV2(workspace_path=tmp_path, auto_scan=False)


class TestMetadataEntryTypeContract:
    """MetadataEntry TypedDict must use 'validation_result' key."""

    def test_metadata_entry_has_validation_result_key(self):
        """MetadataEntry has 'validation_result' field, not 'validation'."""
        hints = get_type_hints(MetadataEntry)
        assert (
            "validation_result" in hints
        ), "MetadataEntry must have 'validation_result' key"

    def test_metadata_entry_does_not_have_old_validation_key(self):
        """MetadataEntry must NOT have the old 'validation' field."""
        hints = get_type_hints(MetadataEntry)
        assert (
            "validation" not in hints
        ), "MetadataEntry must not have old 'validation' key -- use 'validation_result'"


class TestStoreGeoMetadataKeyContract:
    """_store_geo_metadata must accept and store 'validation_result' kwarg."""

    def test_store_with_validation_result_kwarg(self, dm):
        """_store_geo_metadata stores validation_result kwarg under the correct key."""
        validation_data = {"valid": True, "checks": ["platform_ok"]}
        entry = dm._store_geo_metadata(
            geo_id="GSE99999",
            metadata={"samples": {}},
            stored_by="test",
            validation_result=validation_data,
        )
        assert "validation_result" in entry
        assert entry["validation_result"] == validation_data

    def test_store_does_not_accept_old_validation_kwarg(self, dm):
        """_store_geo_metadata ignores the old 'validation' kwarg (stores nothing for it)."""
        entry = dm._store_geo_metadata(
            geo_id="GSE99998",
            metadata={"samples": {}},
            stored_by="test",
            validation={"valid": True},  # old kwarg name
        )
        # The old 'validation' kwarg should NOT create a 'validation' key
        assert "validation" not in entry, "Old 'validation' kwarg should not be stored"

    def test_store_raises_on_none_metadata(self, dm):
        """_store_geo_metadata raises ValueError if metadata is None."""
        with pytest.raises(ValueError, match="metadata dict is required"):
            dm._store_geo_metadata(
                geo_id="GSE99997",
                metadata=None,
                stored_by="test",
            )

    def test_store_warns_on_none_validation_result(self, dm, caplog):
        """_store_geo_metadata warns if validation_result is explicitly None."""
        with caplog.at_level(logging.WARNING):
            entry = dm._store_geo_metadata(
                geo_id="GSE99996",
                metadata={"samples": {}},
                stored_by="test",
                validation_result=None,
            )
        assert any("validation_result is None" in r.message for r in caplog.records)

    def test_get_geo_metadata_returns_validation_result(self, dm):
        """_get_geo_metadata returns entry with accessible 'validation_result' key."""
        validation_data = {"valid": True}
        dm._store_geo_metadata(
            geo_id="GSE99995",
            metadata={"samples": {}},
            stored_by="test",
            validation_result=validation_data,
        )
        entry = dm._get_geo_metadata("GSE99995")
        assert entry is not None
        assert entry.get("validation_result") == validation_data

    def test_roundtrip_without_validation_result(self, dm):
        """Entry without validation_result returns None on .get()."""
        dm._store_geo_metadata(
            geo_id="GSE99994",
            metadata={"samples": {}},
            stored_by="test",
        )
        entry = dm._get_geo_metadata("GSE99994")
        assert entry is not None
        assert entry.get("validation_result") is None
