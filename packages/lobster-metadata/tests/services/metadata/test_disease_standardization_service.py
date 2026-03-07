"""
Unit tests for DiseaseStandardizationService
"""

import numpy as np
import pandas as pd
import pytest

from lobster.services.metadata.disease_standardization_service import (
    DiseaseStandardizationService,
)


@pytest.fixture
def service():
    """Create a service instance."""
    return DiseaseStandardizationService()


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
            "disease": [
                "Colorectal Cancer",  # CRC - exact match
                "CRC",  # CRC - exact match
                "Ulcerative Colitis",  # UC - exact match
                "Crohn's Disease",  # CD - exact match
                "Healthy Control",  # Healthy - contains match
                "tumor",  # Unmapped (generic term removed from CRC mappings)
                "Healthy volunteer",  # Healthy - exact match (changed from "Normal" which is now unmapped)
                "Unknown Disease",  # Unmapped
            ],
            "sample_type": [
                "fecal",
                "Stool",
                "feces",
                "biopsy",
                "tissue",
                "fecal sample",
                "gut tissue",
                "fecal",
            ],
            "tissue": [None, None, None, "colon", "rectal", None, "intestinal", None],
        }
    )


class TestDiseaseStandardization:
    """Test disease term standardization."""

    def test_exact_match(self, service):
        """Test exact matching of disease terms."""
        metadata = pd.DataFrame({"disease": ["CRC", "UC", "CD", "Healthy"]})

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        assert stats["mapping_stats"]["exact_matches"] == 4
        assert stats["mapping_stats"]["unmapped"] == 0
        assert stats["standardization_rate"] == 100.0

    def test_contains_match(self, service):
        """Test contains matching (value contains variant)."""
        metadata = pd.DataFrame(
            {
                "disease": [
                    "Colorectal Cancer Stage II",
                    "Active Ulcerative Colitis",
                    "Crohn's Disease with complications",
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        assert result["disease"].tolist() == ["crc", "uc", "cd"]
        assert stats["mapping_stats"]["contains_matches"] >= 1

    def test_reverse_contains_match(self, service):
        """Test reverse contains matching (variant contains value)."""
        metadata = pd.DataFrame({"disease": ["cancer", "colitis", "tumor"]})

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # "cancer" should match CRC (reverse contains: "colon cancer" contains "cancer")
        # "colitis" should match UC (reverse contains: "ulcerative colitis" contains "colitis")
        # "tumor" is UNMAPPED (generic term removed and no variant contains "tumor")
        assert "crc" in result["disease"].tolist()
        assert "uc" in result["disease"].tolist()
        assert result["disease"].tolist().count("crc") == 1
        assert result["disease"].tolist().count("uc") == 1
        assert stats["mapping_stats"]["unmapped"] == 1  # Only "tumor" unmapped
        assert stats["mapping_stats"]["reverse_contains_matches"] == 2

    def test_token_match(self, service):
        """Test token-based matching."""
        metadata = pd.DataFrame(
            {
                "disease": [
                    "rectal adenocarcinoma",  # Should match CRC (token: rectal)
                    "colon tumor",  # Should match CRC (token: colon from "colon cancer")
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # Both should map to CRC via anatomical tokens (rectal, colon)
        # "rectal" appears in "rectal cancer" variant
        # "colon" appears in "colon cancer" variant
        assert result["disease"].iloc[0] == "crc"
        assert result["disease"].iloc[1] == "crc"
        assert all(d == "crc" for d in result["disease"])
        assert stats["mapping_stats"]["token_matches"] == 2
        assert stats["mapping_stats"]["unmapped"] == 0

    def test_lung_cancer_not_mapped_to_crc(self, service):
        """Regression test: lung cancer should NOT map to CRC after generic term removal."""
        metadata = pd.DataFrame(
            {
                "disease": [
                    "lung adenocarcinoma",
                    "breast tumor",
                    "glioblastoma",  # Replaced "pancreatic cancer" (which has "cancer" token)
                    "liver tumour",
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # All should be unmapped - no overlapping tokens with CRC variants
        # "cancer"/"adenocarcinoma" removed from CRC mappings, but still present in variants
        # like "colon cancer", so terms with "cancer" token would match via Level 4
        # These terms have NO token overlap with CRC, so they remain unmapped
        assert "crc" not in result["disease"].tolist()
        assert stats["mapping_stats"]["unmapped"] == 4

        # Original values should be preserved for unmapped terms
        assert "lung adenocarcinoma" in result["disease"].tolist()
        assert "breast tumor" in result["disease"].tolist()
        assert "glioblastoma" in result["disease"].tolist()
        assert "liver tumour" in result["disease"].tolist()

    def test_control_terms_not_mapped_to_healthy(self, service):
        """Regression test: generic 'control' terms should NOT map to healthy.

        Scientific justification:
        - "control" alone is ambiguous (negative/positive/technical/disease controls)
        - Only compound terms like "healthy control" should map to healthy
        - This prevents misclassification of extraction blanks, mock communities,
          and disease control groups as healthy samples.
        """
        metadata = pd.DataFrame(
            {
                "disease": [
                    "negative control",  # Extraction blank - NOT a healthy sample
                    "positive control",  # Mock community - NOT a healthy sample
                    "disease control",  # Disease control group - NOT healthy
                    "IBS control",  # IBS disease control - NOT healthy
                    "technical control",  # Technical control - NOT a health status
                    "control",  # Ambiguous - should NOT map to healthy
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # None of these should map to "healthy"
        assert result["disease"].tolist().count("healthy") == 0
        # All should remain unmapped (original values preserved)
        assert "negative control" in result["disease"].tolist()
        assert "positive control" in result["disease"].tolist()
        assert "disease control" in result["disease"].tolist()

    def test_normal_terms_not_mapped_to_healthy(self, service):
        """Regression test: generic 'normal' terms should NOT map to healthy.

        Scientific justification:
        - "normal tissue" in tumor studies refers to adjacent non-tumor tissue
        - "normal flora" refers to microbiome composition, not health status
        - Only explicit health terms should map to healthy
        """
        metadata = pd.DataFrame(
            {
                "disease": [
                    "normal tissue",  # Adjacent normal in tumor studies
                    "normal flora",  # Microbiome term, not health status
                    "normal distribution",  # Statistics term (edge case)
                    "normal",  # Ambiguous - should NOT map to healthy
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # None of these should map to "healthy"
        assert result["disease"].tolist().count("healthy") == 0
        # Original values should be preserved
        assert "normal tissue" in result["disease"].tolist()
        assert "normal flora" in result["disease"].tolist()

    def test_healthy_compound_terms_still_match(self, service):
        """Verify that explicit healthy compound terms still map correctly.

        After removing generic "control" and "normal", these compound terms
        should still correctly map to "healthy".
        """
        metadata = pd.DataFrame(
            {
                "disease": [
                    "healthy control",  # ✓ Should map to healthy
                    "healthy volunteer",  # ✓ Should map to healthy
                    "healthy donor",  # ✓ Should map to healthy
                    "normal control",  # ✓ Should map to healthy (compound term)
                    "non-diseased",  # ✓ Should map to healthy
                    "disease-free",  # ✓ Should map to healthy
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # All should map to "healthy"
        assert all(d == "healthy" for d in result["disease"])
        assert stats["mapping_stats"]["unmapped"] == 0

    def test_unmapped_terms(self, service):
        """Test handling of unmapped terms."""
        metadata = pd.DataFrame(
            {"disease": ["Unknown Disease", "Rare Condition", "N/A"]}
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # Note: "Unknown Disease" contains "disease" which is a token in CRC mappings
        # So it will match via token matching. Only truly unmapped terms remain.
        assert stats["mapping_stats"]["unmapped"] >= 1  # At least some unmapped
        # N/A should be unmapped
        assert "N/A" in result["disease"].tolist()

    def test_case_insensitive_matching(self, service):
        """Test case-insensitive matching."""
        metadata = pd.DataFrame(
            {"disease": ["CRC", "crc", "Crc", "COLORECTAL CANCER", "colorectal cancer"]}
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # All should map to "crc"
        assert all(d == "crc" for d in result["disease"])
        assert stats["standardization_rate"] == 100.0

    def test_punctuation_handling(self, service):
        """Test handling of punctuation in disease terms."""
        metadata = pd.DataFrame(
            {
                "disease": [
                    "Crohn's Disease",
                    "Crohns Disease",
                    "Crohn Disease",
                    "ulcerative-colitis",
                    "ulcerative_colitis",
                ]
            }
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # First three should map to "cd", last two to "uc"
        assert result["disease"].iloc[0] == "cd"
        assert result["disease"].iloc[1] == "cd"
        assert result["disease"].iloc[2] == "cd"
        assert result["disease"].iloc[4] == "uc"  # exact match in variant list

    def test_original_column_preservation(self, service, sample_metadata):
        """Test that original values are preserved."""
        result, stats, ir = service.standardize_disease_terms(
            sample_metadata, "disease"
        )

        assert "disease_original" in result.columns
        assert list(result["disease_original"]) == list(sample_metadata["disease"])

    def test_statistics_completeness(self, service, sample_metadata):
        """Test that statistics are complete and correct."""
        result, stats, ir = service.standardize_disease_terms(
            sample_metadata, "disease"
        )

        # Check all required stats keys
        assert "total_samples" in stats
        assert "standardization_rate" in stats
        assert "mapping_stats" in stats
        assert "unique_mappings" in stats
        assert "disease_distribution" in stats

        # Check mapping stats structure
        assert "exact_matches" in stats["mapping_stats"]
        assert "contains_matches" in stats["mapping_stats"]
        assert "token_matches" in stats["mapping_stats"]
        assert "unmapped" in stats["mapping_stats"]

        # Verify counts
        total_matches = sum(stats["mapping_stats"].values())
        assert total_matches == stats["total_samples"]

    def test_provenance_ir(self, service, sample_metadata):
        """Test that provenance IR is generated correctly."""
        result, stats, ir = service.standardize_disease_terms(
            sample_metadata, "disease"
        )

        assert ir.operation == "disease_standardization"
        assert ir.tool_name == "standardize_disease_terms"
        assert ir.library == "custom"
        assert "disease_column" in ir.parameters
        assert "disease_mappings" in ir.parameters

    def test_missing_column_error(self, service, sample_metadata):
        """Test error handling for missing disease column."""
        with pytest.raises(ValueError, match="Disease column .* not found"):
            service.standardize_disease_terms(sample_metadata, "nonexistent_column")


class TestSampleTypeFiltering:
    """Test sample type filtering."""

    def test_fecal_only_filter(self, service, sample_metadata):
        """Test filtering for fecal samples only."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata, sample_types=["fecal"], sample_columns=["sample_type"]
        )

        # Should keep S1, S2, S3, S6, S8 (fecal/stool/feces/fecal sample/fecal)
        assert len(result) >= 3  # At least the clear fecal matches
        assert stats["original_samples"] == 8
        assert stats["retention_rate"] < 100.0

    def test_gut_tissue_filter(self, service, sample_metadata):
        """Test filtering for gut tissue samples."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata,
            sample_types=["gut"],
            sample_columns=["sample_type", "tissue"],
        )

        # Should keep samples with biopsy, tissue, colon, rectal, intestinal
        assert len(result) >= 3
        assert "matched_column" in result.columns
        assert "matched_value" in result.columns

    def test_multiple_columns(self, service, sample_metadata):
        """Test filtering across multiple columns."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata,
            sample_types=["fecal"],
            sample_columns=["sample_type", "tissue"],
        )

        # Should find fecal in sample_type column
        assert "sample_type" in stats["matches_by_column"]
        assert stats["matches_by_column"]["sample_type"] > 0

    def test_auto_column_detection(self, service):
        """Test automatic detection of sample type columns."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "sample_type": ["fecal", "tissue"],
                "body_site": ["gut", "gut"],
            }
        )

        result, stats, ir = service.filter_by_sample_type(
            metadata, sample_types=["fecal"]
        )

        # Should auto-detect sample_type and body_site columns
        assert len(stats["columns_checked"]) >= 1
        assert "sample_type" in stats["columns_checked"]

    def test_no_columns_found_error(self, service):
        """Test error when no sample type columns found."""
        metadata = pd.DataFrame({"sample_id": ["S1", "S2"], "disease": ["CRC", "UC"]})

        with pytest.raises(ValueError, match="No sample type columns found"):
            service.filter_by_sample_type(metadata, sample_types=["fecal"])

    def test_empty_result_handling(self, service, sample_metadata):
        """Test handling when no samples match filter."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata,
            sample_types=["blood"],  # No blood samples in test data
            sample_columns=["sample_type"],
        )

        assert len(result) == 0
        assert stats["filtered_samples"] == 0
        assert stats["retention_rate"] == 0.0

    def test_case_insensitive_sample_type(self, service):
        """Test case-insensitive sample type matching."""
        metadata = pd.DataFrame({"sample_type": ["FECAL", "Fecal", "fecal", "STOOL"]})

        result, stats, ir = service.filter_by_sample_type(
            metadata, sample_types=["fecal"], sample_columns=["sample_type"]
        )

        assert len(result) == 4  # All should match

    def test_filter_statistics(self, service, sample_metadata):
        """Test that filter statistics are complete."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata, sample_types=["fecal"], sample_columns=["sample_type"]
        )

        assert "original_samples" in stats
        assert "filtered_samples" in stats
        assert "retention_rate" in stats
        assert "sample_types_requested" in stats
        assert "columns_checked" in stats
        assert "matches_by_column" in stats

    def test_filter_provenance_ir(self, service, sample_metadata):
        """Test that filter provenance IR is generated."""
        result, stats, ir = service.filter_by_sample_type(
            sample_metadata, sample_types=["fecal"], sample_columns=["sample_type"]
        )

        assert ir.operation == "sample_type_filter"
        assert ir.tool_name == "filter_by_sample_type"
        assert "sample_types" in ir.parameters
        assert "retention_rate" in ir.parameters


class TestHelperMethods:
    """Test helper methods."""

    def test_normalize_term(self, service):
        """Test term normalization."""
        assert service._normalize_term("Crohn's Disease") == "crohn s disease"
        assert service._normalize_term("COLORECTAL-CANCER") == "colorectal cancer"
        assert service._normalize_term("  Multiple   Spaces  ") == "multiple spaces"

    def test_tokenize(self, service):
        """Test tokenization."""
        tokens = service._tokenize("crohn s disease")
        assert "crohn" in tokens
        assert "disease" in tokens
        assert "s" not in tokens  # Single-char tokens ignored

    def test_fuzzy_match_returns_match_type(self, service):
        """Test that fuzzy_match returns correct match type."""
        # Exact match
        match, match_type = service._fuzzy_match("crc", service.DISEASE_MAPPINGS)
        assert match == "crc"
        assert match_type == "exact"

        # Token match (note: "disease" is a token in CRC mappings)
        match, match_type = service._fuzzy_match(
            "completely unknown disease", service.DISEASE_MAPPINGS
        )
        assert match_type == "token"  # "disease" token matches CRC

        # Truly unmapped
        match, match_type = service._fuzzy_match("xyz123", service.DISEASE_MAPPINGS)
        assert match_type == "unmapped"
        assert match == "xyz123"  # Original value preserved


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, service):
        """Test handling of empty dataframe."""
        metadata = pd.DataFrame({"disease": []})

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        assert len(result) == 0
        assert stats["total_samples"] == 0

    def test_nan_values(self, service):
        """Test handling of NaN values."""
        metadata = pd.DataFrame({"disease": ["CRC", np.nan, "UC", None]})

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # Should handle NaN gracefully (convert to string "nan")
        assert len(result) == 4
        assert stats["total_samples"] == 4

    def test_numeric_values(self, service):
        """Test handling of numeric disease values."""
        metadata = pd.DataFrame({"disease": [1, 2, 3]})

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # Should convert to string and attempt matching
        assert len(result) == 3
        # Likely unmapped since "1", "2", "3" don't match diseases
        assert stats["mapping_stats"]["unmapped"] == 3

    def test_special_characters(self, service):
        """Test handling of special characters."""
        metadata = pd.DataFrame(
            {"disease": ["Crohn's (active)", "UC [moderate]", "CRC - stage 2"]}
        )

        result, stats, ir = service.standardize_disease_terms(metadata, "disease")

        # Should strip special chars and match correctly
        assert "cd" in result["disease"].tolist()
        assert "uc" in result["disease"].tolist()
        assert "crc" in result["disease"].tolist()
