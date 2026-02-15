"""
Unit tests for the lineage tracking module.

Tests cover:
- Base name extraction from modality names
- Processing step inference from suffixes
- Lineage metadata creation and attachment
- Version incrementing from parent
- Legacy lineage migration
"""

from datetime import datetime
from unittest.mock import MagicMock

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.lineage import (
    CANONICAL_STEPS,
    LINEAGE_KEY,
    SUFFIX_PATTERNS,
    SUFFIX_TO_STEP,
    LineageMetadata,
    attach_lineage,
    create_lineage_metadata,
    ensure_lineage,
    extract_base_name,
    get_lineage,
    get_lineage_dict,
    has_lineage,
    infer_processing_step,
)


class TestExtractBaseName:
    """Tests for extract_base_name function."""

    def test_no_suffix(self):
        """Test that names without suffixes are returned unchanged."""
        assert extract_base_name("geo_gse12345") == "geo_gse12345"
        assert (
            extract_base_name("pride_pxd12345_proteomics")
            == "pride_pxd12345_proteomics"
        )

    def test_single_suffix(self):
        """Test extraction with single suffix."""
        assert extract_base_name("geo_gse12345_clustered") == "geo_gse12345"
        assert extract_base_name("geo_gse12345_normalized") == "geo_gse12345"
        assert extract_base_name("geo_gse12345_quality_assessed") == "geo_gse12345"

    def test_chained_suffixes(self):
        """Test extraction with multiple chained suffixes."""
        assert extract_base_name("geo_gse12345_filtered_normalized") == "geo_gse12345"
        assert (
            extract_base_name("geo_gse12345_filtered_normalized_clustered")
            == "geo_gse12345"
        )
        assert (
            extract_base_name(
                "geo_gse12345_quality_assessed_filtered_normalized_clustered_annotated"
            )
            == "geo_gse12345"
        )

    def test_preserves_non_suffix_parts(self):
        """Test that non-suffix parts of the name are preserved."""
        # 'proteomics' is not a suffix, so it should be kept
        assert (
            extract_base_name("pride_pxd12345_proteomics")
            == "pride_pxd12345_proteomics"
        )
        assert (
            extract_base_name("pride_pxd12345_proteomics_normalized")
            == "pride_pxd12345_proteomics"
        )

    def test_handles_empty_and_edge_cases(self):
        """Test edge cases."""
        assert extract_base_name("") == ""
        assert extract_base_name("x") == "x"
        # Just a suffix should strip to empty
        assert extract_base_name("_clustered") == ""


class TestInferProcessingStep:
    """Tests for infer_processing_step function."""

    def test_raw_for_no_suffix(self):
        """Test that names without suffixes return RAW."""
        assert infer_processing_step("geo_gse12345") == "raw"
        assert infer_processing_step("pride_pxd12345") == "raw"

    def test_known_suffixes(self):
        """Test inference for known suffixes."""
        assert infer_processing_step("geo_gse12345_clustered") == "clustered"
        assert infer_processing_step("geo_gse12345_normalized") == "normalized"
        assert (
            infer_processing_step("geo_gse12345_quality_assessed") == "quality_assessed"
        )
        assert infer_processing_step("geo_gse12345_filtered") == "filtered"
        assert infer_processing_step("geo_gse12345_annotated") == "annotated"
        assert infer_processing_step("geo_gse12345_markers") == "markers"

    def test_compound_suffix(self):
        """Test inference for compound suffixes."""
        assert (
            infer_processing_step("geo_gse12345_filtered_normalized")
            == "filtered_normalized"
        )

    def test_takes_last_suffix(self):
        """Test that chained suffixes return the last one."""
        # Note: the current implementation matches the first suffix found in SUFFIX_PATTERNS
        # which is sorted by length. So _filtered_normalized matches first
        result = infer_processing_step("geo_gse12345_filtered_normalized_clustered")
        # After extracting _clustered, we'd be left with _filtered_normalized
        # But infer_processing_step checks against the FULL name
        assert result == "clustered"  # _clustered is matched


class TestLineageMetadata:
    """Tests for LineageMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        lineage = LineageMetadata(
            base_name="geo_gse12345",
            version=2,
            processing_step="clustered",
            parent_modality="geo_gse12345_normalized",
            step_summary="Clustered into 10 clusters",
            created_at="2026-02-08T10:00:00",
        )
        d = lineage.to_dict()
        assert d["base_name"] == "geo_gse12345"
        assert d["version"] == 2
        assert d["processing_step"] == "clustered"
        assert d["parent_modality"] == "geo_gse12345_normalized"
        assert d["step_summary"] == "Clustered into 10 clusters"
        assert d["created_at"] == "2026-02-08T10:00:00"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "base_name": "geo_gse12345",
            "version": 3,
            "processing_step": "annotated",
            "parent_modality": "geo_gse12345_clustered",
            "step_summary": "Cell types annotated",
            "created_at": "2026-02-08T11:00:00",
        }
        lineage = LineageMetadata.from_dict(d)
        assert lineage.base_name == "geo_gse12345"
        assert lineage.version == 3
        assert lineage.processing_step == "annotated"
        assert lineage.parent_modality == "geo_gse12345_clustered"

    def test_from_dict_ignores_extra_keys(self):
        """Test that extra keys (like _synthetic) are ignored."""
        d = {
            "base_name": "geo_gse12345",
            "version": 1,
            "processing_step": "raw",
            "parent_modality": None,
            "step_summary": "Loaded",
            "created_at": "2026-02-08T10:00:00",
            "_synthetic": True,  # Extra key
            "_extra_field": "ignored",
        }
        lineage = LineageMetadata.from_dict(d)
        assert lineage.base_name == "geo_gse12345"
        assert lineage.version == 1


class TestCreateLineageMetadata:
    """Tests for create_lineage_metadata factory function."""

    def test_auto_extract_base_name(self):
        """Test auto-extraction of base name from modality name."""
        lineage = create_lineage_metadata(modality_name="geo_gse12345_clustered")
        assert lineage.base_name == "geo_gse12345"

    def test_auto_infer_step(self):
        """Test auto-inference of processing step."""
        lineage = create_lineage_metadata(modality_name="geo_gse12345_clustered")
        assert lineage.processing_step == "clustered"

    def test_version_starts_at_1(self):
        """Test that version starts at 1 for raw data."""
        lineage = create_lineage_metadata(modality_name="geo_gse12345")
        assert lineage.version == 1

    def test_version_increments_from_parent(self):
        """Test that version increments from parent."""
        # Create parent AnnData with lineage
        parent_adata = anndata.AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
        )
        parent_adata.uns[LINEAGE_KEY] = {
            "base_name": "geo_gse12345",
            "version": 2,
            "processing_step": "normalized",
            "parent_modality": "geo_gse12345_raw",
            "step_summary": "Normalized",
            "created_at": "2026-02-08T10:00:00",
        }

        # Create child lineage
        lineage = create_lineage_metadata(
            modality_name="geo_gse12345_clustered",
            parent_modality="geo_gse12345_normalized",
            parent_adata=parent_adata,
        )

        assert lineage.version == 3  # Incremented from parent
        assert lineage.base_name == "geo_gse12345"  # Inherited from parent
        assert lineage.parent_modality == "geo_gse12345_normalized"

    def test_explicit_values_override_inference(self):
        """Test that explicit values override inference."""
        lineage = create_lineage_metadata(
            modality_name="geo_gse12345_clustered",
            step="custom",
            base_name="my_custom_base",
            version=99,
        )
        assert lineage.processing_step == "custom"
        assert lineage.base_name == "my_custom_base"
        assert lineage.version == 99

    def test_step_summary_is_stored(self):
        """Test that step summary is stored."""
        lineage = create_lineage_metadata(
            modality_name="geo_gse12345_clustered",
            step_summary="Clustered into 15 groups",
        )
        assert lineage.step_summary == "Clustered into 15 groups"

    def test_created_at_is_set(self):
        """Test that created_at is set to current time."""
        lineage = create_lineage_metadata(modality_name="geo_gse12345")
        # Just verify it's a valid ISO timestamp
        datetime.fromisoformat(lineage.created_at)


class TestAttachAndGetLineage:
    """Tests for attach_lineage and get_lineage functions."""

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object."""
        return anndata.AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
        )

    def test_attach_lineage(self, sample_adata):
        """Test attaching lineage to AnnData."""
        lineage = LineageMetadata(
            base_name="test",
            version=1,
            processing_step="raw",
            parent_modality=None,
            step_summary="Test",
            created_at="2026-02-08T10:00:00",
        )

        result = attach_lineage(sample_adata, lineage)

        assert result is sample_adata  # Returns same object
        assert LINEAGE_KEY in sample_adata.uns
        assert sample_adata.uns[LINEAGE_KEY]["base_name"] == "test"

    def test_get_lineage(self, sample_adata):
        """Test extracting lineage from AnnData."""
        sample_adata.uns[LINEAGE_KEY] = {
            "base_name": "geo_gse12345",
            "version": 2,
            "processing_step": "clustered",
            "parent_modality": "geo_gse12345_normalized",
            "step_summary": "Clustered",
            "created_at": "2026-02-08T10:00:00",
        }

        lineage = get_lineage(sample_adata)

        assert lineage is not None
        assert lineage.base_name == "geo_gse12345"
        assert lineage.version == 2
        assert lineage.processing_step == "clustered"

    def test_get_lineage_returns_none_if_missing(self, sample_adata):
        """Test that get_lineage returns None if no lineage."""
        assert get_lineage(sample_adata) is None

    def test_has_lineage(self, sample_adata):
        """Test has_lineage function."""
        assert has_lineage(sample_adata) is False

        sample_adata.uns[LINEAGE_KEY] = {"version": 1}
        assert has_lineage(sample_adata) is True


class TestEnsureLineage:
    """Tests for ensure_lineage function (legacy migration)."""

    @pytest.fixture
    def legacy_adata(self):
        """Create an AnnData without lineage (legacy file)."""
        return anndata.AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
        )

    def test_creates_synthetic_lineage_for_legacy(self, legacy_adata):
        """Test that synthetic lineage is created for legacy files."""
        result = ensure_lineage(legacy_adata, "geo_gse12345_clustered")

        assert result is legacy_adata
        assert LINEAGE_KEY in legacy_adata.uns
        lineage = legacy_adata.uns[LINEAGE_KEY]
        assert lineage["base_name"] == "geo_gse12345"
        assert lineage["processing_step"] == "clustered"
        assert lineage["_synthetic"] is True  # Marked as synthetic

    def test_preserves_existing_lineage(self, legacy_adata):
        """Test that existing lineage is preserved."""
        # Add existing lineage
        legacy_adata.uns[LINEAGE_KEY] = {
            "base_name": "original",
            "version": 5,
            "processing_step": "raw",
            "parent_modality": None,
            "step_summary": "Original",
            "created_at": "2026-02-08T10:00:00",
        }

        result = ensure_lineage(legacy_adata, "geo_gse12345_clustered")

        # Existing lineage should be preserved
        assert legacy_adata.uns[LINEAGE_KEY]["base_name"] == "original"
        assert legacy_adata.uns[LINEAGE_KEY]["version"] == 5

    def test_custom_step_summary(self, legacy_adata):
        """Test custom step summary for legacy lineage."""
        ensure_lineage(
            legacy_adata,
            "geo_gse12345",
            step_summary="Loaded from workspace",
        )

        assert legacy_adata.uns[LINEAGE_KEY]["step_summary"] == "Loaded from workspace"


class TestGetLineageDict:
    """Tests for get_lineage_dict function."""

    def test_returns_raw_dict(self):
        """Test that raw dict is returned including extra fields."""
        adata = anndata.AnnData(
            X=np.random.rand(5, 3),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(3)]),
        )
        adata.uns[LINEAGE_KEY] = {
            "base_name": "test",
            "version": 1,
            "processing_step": "raw",
            "parent_modality": None,
            "step_summary": "Test",
            "created_at": "2026-02-08T10:00:00",
            "_synthetic": True,
            "_extra_field": "accessible",
        }

        result = get_lineage_dict(adata)

        assert result["_synthetic"] is True
        assert result["_extra_field"] == "accessible"

    def test_returns_none_if_missing(self):
        """Test that None is returned if no lineage."""
        adata = anndata.AnnData(X=np.random.rand(5, 3))
        assert get_lineage_dict(adata) is None


class TestProcessingSteps:
    """Tests for canonical steps and suffix mappings."""

    def test_all_suffix_patterns_have_mappings(self):
        """Test that all suffix patterns have step mappings."""
        for suffix in SUFFIX_PATTERNS:
            assert suffix in SUFFIX_TO_STEP, f"Missing mapping for suffix: {suffix}"

    def test_step_values_are_strings(self):
        """Test that all step values are strings."""
        for suffix, step in SUFFIX_TO_STEP.items():
            assert isinstance(step, str), f"Invalid step for {suffix}: {type(step)}"

    def test_canonical_steps_cover_suffix_values(self):
        """Test that suffix mappings use canonical steps."""
        for suffix, step in SUFFIX_TO_STEP.items():
            assert (
                step in CANONICAL_STEPS
            ), f"Step '{step}' for suffix '{suffix}' not in CANONICAL_STEPS"

    def test_suffix_patterns_sorted_by_length(self):
        """Test that suffix patterns are sorted by length (longest first)."""
        lengths = [len(s) for s in SUFFIX_PATTERNS]
        # Should be sorted in descending order
        assert lengths == sorted(lengths, reverse=True)
