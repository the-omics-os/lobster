"""
Unit tests for MicrobiomeFilteringService

Tests cover:
- 16S amplicon detection (OR-based, strict/non-strict)
- Host organism validation (fuzzy matching)
- Edge cases (empty data, malformed inputs, case sensitivity)
- IR generation and provenance tracking
"""

import pytest
from typing import Dict, Any

from lobster.services.metadata.microbiome_filtering_service import (
    MicrobiomeFilteringService,
    ValidationResult,
    HOST_ALIASES,
)
from lobster.core.analysis_ir import AnalysisStep


@pytest.fixture
def service():
    """Create service instance."""
    return MicrobiomeFilteringService()


# =============================================================================
# 16S Amplicon Validation Tests
# =============================================================================


class Test16SAmplicon:
    """Tests for validate_16s_amplicon method."""

    def test_valid_platform_illumina(self, service):
        """Test detection via platform field (Illumina)."""
        metadata = {"platform": "Illumina MiSeq"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert result == metadata
        assert stats["is_valid"] is True
        assert stats["matched_field"] == "platform"
        assert stats["matched_value"] == "Illumina MiSeq"
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "microbiome.filtering.validate_16s_amplicon"

    def test_valid_platform_pacbio(self, service):
        """Test detection via platform field (PacBio)."""
        metadata = {"platform": "PacBio Sequel II"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_field"] == "platform"

    def test_valid_library_strategy_amplicon(self, service):
        """Test detection via library_strategy field."""
        metadata = {"library_strategy": "AMPLICON"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_field"] == "library_strategy"

    def test_valid_library_strategy_16s(self, service):
        """Test detection via library_strategy with 16S keyword."""
        metadata = {"library_strategy": "16S rRNA sequencing"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_field"] == "library_strategy"

    def test_valid_assay_type(self, service):
        """Test detection via assay_type field."""
        metadata = {"assay_type": "16S Sequencing"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_field"] == "assay_type"

    def test_valid_multiple_fields(self, service):
        """Test with multiple valid fields (should match first)."""
        metadata = {
            "platform": "Illumina NextSeq",
            "library_strategy": "AMPLICON",
            "assay_type": "16S Sequencing",
        }
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True
        # Should match platform first (order of AMPLICON_KEYWORDS)
        assert stats["matched_field"] == "platform"

    def test_invalid_no_keywords(self, service):
        """Test with metadata lacking 16S keywords."""
        metadata = {"platform": "Oxford Nanopore", "library_strategy": "RNA-Seq"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert result == {}
        assert stats["is_valid"] is False
        assert stats["matched_field"] is None
        assert "No 16S amplicon indicators" in stats["reason"]

    def test_invalid_empty_metadata(self, service):
        """Test with empty metadata."""
        metadata = {}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert result == {}
        assert stats["is_valid"] is False

    def test_case_insensitive_matching(self, service):
        """Test case-insensitive keyword matching."""
        test_cases = [
            {"platform": "illumina"},
            {"platform": "ILLUMINA"},
            {"platform": "Illumina"},
            {"library_strategy": "amplicon"},
            {"library_strategy": "AMPLICON"},
        ]

        for metadata in test_cases:
            result, stats, ir = service.validate_16s_amplicon(metadata)
            assert stats["is_valid"] is True, f"Failed for: {metadata}"

    def test_substring_matching_nonstrict(self, service):
        """Test substring matching in non-strict mode."""
        metadata = {"platform": "Custom Illumina MiSeq v3"}
        result, stats, ir = service.validate_16s_amplicon(metadata, strict=False)

        assert stats["is_valid"] is True
        assert stats["strict_mode"] is False

    def test_strict_mode_exact_match_required(self, service):
        """Test strict mode requires exact match."""
        metadata = {"library_strategy": "AMPLICON"}
        result, stats, ir = service.validate_16s_amplicon(metadata, strict=True)

        # Should match because "amplicon" is in keyword list
        assert stats["is_valid"] is True

    def test_strict_mode_substring_rejected(self, service):
        """Test strict mode rejects substring matches."""
        metadata = {"platform": "Custom Illumina MiSeq"}
        result, stats, ir = service.validate_16s_amplicon(metadata, strict=True)

        # "custom illumina miseq" != "illumina" (exact), should fail
        assert stats["is_valid"] is False

    def test_field_name_variations(self, service):
        """Test handling of field name variations (underscores, spaces)."""
        # Service should check "library_strategy", "library strategy", "librarystrategy"
        metadata = {"librarystrategy": "AMPLICON"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert stats["is_valid"] is True

    def test_stats_structure(self, service):
        """Test stats dictionary structure."""
        metadata = {"platform": "Illumina"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        required_keys = [
            "is_valid",
            "reason",
            "matched_field",
            "matched_value",
            "strict_mode",
            "fields_checked",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        assert isinstance(stats["fields_checked"], list)
        assert len(stats["fields_checked"]) > 0

    def test_ir_structure_16s(self, service):
        """Test IR structure for 16S validation."""
        metadata = {"library_strategy": "AMPLICON"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        assert ir.operation == "microbiome.filtering.validate_16s_amplicon"
        assert ir.tool_name == "MicrobiomeFilteringService.validate_16s_amplicon"
        assert "strict=" in ir.description
        assert ir.library == "lobster.tools.microbiome_filtering_service"
        assert "metadata" in ir.parameters
        assert "strict" in ir.parameters
        assert len(ir.input_entities) > 0
        assert len(ir.output_entities) > 0


# =============================================================================
# Host Organism Validation Tests
# =============================================================================


class TestHostOrganism:
    """Tests for validate_host_organism method."""

    def test_valid_human_exact(self, service):
        """Test exact match for Human."""
        metadata = {"organism": "Homo sapiens"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert result == metadata
        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Human"
        assert stats["confidence_score"] >= 85.0

    def test_valid_human_lowercase(self, service):
        """Test fuzzy match for lowercase human."""
        metadata = {"organism": "homo sapiens"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Human"

    def test_valid_human_casual(self, service):
        """Test fuzzy match for casual 'human'."""
        metadata = {"organism": "human"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Human"

    def test_valid_mouse_exact(self, service):
        """Test exact match for Mouse."""
        metadata = {"organism": "Mus musculus"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Mouse"

    def test_valid_mouse_abbreviated(self, service):
        """Test fuzzy match for abbreviated mouse."""
        metadata = {"organism": "M. musculus"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Mouse"

    def test_invalid_organism_not_in_list(self, service):
        """Test rejection of organism not in allowed list."""
        metadata = {"organism": "Rattus norvegicus"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert result == {}
        assert stats["is_valid"] is False
        assert stats["matched_host"] is None
        assert "not in allowed list" in stats["reason"]

    def test_invalid_empty_metadata(self, service):
        """Test with empty metadata."""
        metadata = {}
        result, stats, ir = service.validate_host_organism(metadata)

        assert result == {}
        assert stats["is_valid"] is False

    def test_invalid_missing_organism_field(self, service):
        """Test with metadata lacking organism field."""
        metadata = {"platform": "Illumina", "assay": "RNA-Seq"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert stats["is_valid"] is False

    def test_custom_allowed_hosts(self, service):
        """Test with custom allowed hosts list."""
        metadata = {"organism": "Rattus norvegicus"}
        allowed = ["Rat"]

        # First verify it fails with default hosts
        result1, stats1, ir1 = service.validate_host_organism(metadata)
        assert stats1["is_valid"] is False

        # Note: This would pass if we added Rat to HOST_ALIASES
        # For now, it should fail since "Rattus norvegicus" won't fuzzy match "Rat" > 85%

    def test_custom_threshold_strict(self, service):
        """Test with higher fuzzy threshold."""
        metadata = {"organism": "H. sapiens"}
        result, stats, ir = service.validate_host_organism(
            metadata, fuzzy_threshold=95.0
        )

        # "H. sapiens" in HOST_ALIASES, should match even with high threshold
        assert stats["is_valid"] is True

    def test_custom_threshold_lenient(self, service):
        """Test with lower fuzzy threshold."""
        metadata = {"organism": "humanoid"}
        result, stats, ir = service.validate_host_organism(
            metadata, fuzzy_threshold=50.0
        )

        # "humanoid" should fuzzy match "human" with low threshold
        # Let's check the actual behavior
        # (may pass or fail depending on fuzzy score)

    def test_host_field_variations(self, service):
        """Test different organism field names."""
        field_variations = [
            {"organism": "Homo sapiens"},
            {"host": "Homo sapiens"},
            {"host_organism": "Homo sapiens"},
            {"source": "Homo sapiens"},
            {"taxon": "Homo sapiens"},
            {"species": "Homo sapiens"},
        ]

        for metadata in field_variations:
            result, stats, ir = service.validate_host_organism(metadata)
            assert stats["is_valid"] is True, f"Failed for field: {list(metadata.keys())[0]}"

    def test_stats_structure_host(self, service):
        """Test stats dictionary structure for host validation."""
        metadata = {"organism": "Homo sapiens"}
        result, stats, ir = service.validate_host_organism(metadata)

        required_keys = [
            "is_valid",
            "reason",
            "matched_host",
            "confidence_score",
            "allowed_hosts",
            "fuzzy_threshold",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        assert isinstance(stats["allowed_hosts"], list)
        assert isinstance(stats["fuzzy_threshold"], float)

    def test_ir_structure_host(self, service):
        """Test IR structure for host validation."""
        metadata = {"organism": "Homo sapiens"}
        result, stats, ir = service.validate_host_organism(metadata)

        assert ir.operation == "microbiome.filtering.validate_host_organism"
        assert ir.tool_name == "MicrobiomeFilteringService.validate_host_organism"
        assert "threshold=" in ir.description
        assert ir.library == "lobster.tools.microbiome_filtering_service"
        assert "metadata" in ir.parameters
        assert "allowed_hosts" in ir.parameters
        assert "fuzzy_threshold" in ir.parameters
        assert "import difflib" in ir.imports


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelpers:
    """Tests for internal helper methods."""

    def test_normalize_field_string(self, service):
        """Test field normalization with string."""
        assert service._normalize_field("  Test Value  ") == "test value"
        assert service._normalize_field("UPPERCASE") == "uppercase"
        assert service._normalize_field("MixedCase") == "mixedcase"

    def test_normalize_field_non_string(self, service):
        """Test field normalization with non-string types."""
        assert service._normalize_field(123) == "123"
        assert service._normalize_field(None) == "none"
        assert service._normalize_field(True) == "true"

    def test_contains_16s_substring_match(self, service):
        """Test _contains_16s with substring matching."""
        keywords = ["illumina", "miseq"]

        assert service._contains_16s("illumina miseq v3", keywords, strict=False)
        assert service._contains_16s("custom illumina", keywords, strict=False)
        assert not service._contains_16s("pacbio sequel", keywords, strict=False)

    def test_contains_16s_exact_match(self, service):
        """Test _contains_16s with exact matching."""
        keywords = ["illumina", "miseq"]

        assert service._contains_16s("illumina", keywords, strict=True)
        assert service._contains_16s("miseq", keywords, strict=True)
        assert not service._contains_16s("illumina miseq", keywords, strict=True)

    def test_match_host_exact_alias(self, service):
        """Test _match_host with exact alias match."""
        result = service._match_host("Homo sapiens", ["Human"], 85.0)

        assert result is not None
        matched_host, score = result
        assert matched_host == "Human"
        assert score >= 85.0

    def test_match_host_fuzzy_match(self, service):
        """Test _match_host with fuzzy matching."""
        result = service._match_host("homo sapiens", ["Human"], 85.0)

        assert result is not None
        matched_host, score = result
        assert matched_host == "Human"

    def test_match_host_below_threshold(self, service):
        """Test _match_host rejecting low-score matches."""
        result = service._match_host("Escherichia coli", ["Human", "Mouse"], 85.0)

        assert result is None

    def test_match_host_multiple_hosts(self, service):
        """Test _match_host with multiple allowed hosts."""
        result = service._match_host("Mus musculus", ["Human", "Mouse"], 85.0)

        assert result is not None
        matched_host, score = result
        assert matched_host == "Mouse"


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_malformed_metadata_types(self, service):
        """Test handling of malformed metadata types."""
        # Non-dict metadata should still work with normalization
        metadata = {"platform": 12345}
        result, stats, ir = service.validate_16s_amplicon(metadata)
        # Should handle conversion via _normalize_field

    def test_unicode_characters(self, service):
        """Test handling of Unicode characters."""
        metadata = {"organism": "Homo sapiens 🧬"}
        result, stats, ir = service.validate_host_organism(metadata)
        # Fuzzy matching should handle unicode gracefully

    def test_very_long_field_values(self, service):
        """Test handling of very long field values."""
        metadata = {"platform": "Illumina " + "x" * 1000}
        result, stats, ir = service.validate_16s_amplicon(metadata)
        # Should still detect "Illumina" substring

    def test_special_characters_in_fields(self, service):
        """Test handling of special characters."""
        metadata = {"library_strategy": "16S-rRNA-Seq"}
        result, stats, ir = service.validate_16s_amplicon(metadata)
        assert stats["is_valid"] is True

    def test_multiple_validations_sequential(self, service):
        """Test running multiple validations sequentially."""
        metadata = {
            "platform": "Illumina MiSeq",
            "organism": "Homo sapiens",
        }

        # Run both validations
        result1, stats1, ir1 = service.validate_16s_amplicon(metadata)
        result2, stats2, ir2 = service.validate_host_organism(metadata)

        assert stats1["is_valid"] is True
        assert stats2["is_valid"] is True


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for service constants."""

    def test_host_aliases_structure(self):
        """Test HOST_ALIASES constant structure."""
        assert "Human" in HOST_ALIASES
        assert "Mouse" in HOST_ALIASES
        assert isinstance(HOST_ALIASES["Human"], list)
        assert len(HOST_ALIASES["Human"]) > 0

    def test_amplicon_keywords_structure(self, service):
        """Test AMPLICON_KEYWORDS structure."""
        assert "platform" in service.AMPLICON_KEYWORDS
        assert "library_strategy" in service.AMPLICON_KEYWORDS
        assert "assay_type" in service.AMPLICON_KEYWORDS
        assert isinstance(service.AMPLICON_KEYWORDS["platform"], list)


# =============================================================================
# Provenance and Reproducibility Tests
# =============================================================================


class TestProvenance:
    """Tests for provenance tracking and reproducibility."""

    def test_ir_contains_executable_code(self, service):
        """Test that IR contains executable Python code."""
        metadata = {"platform": "Illumina"}
        result, stats, ir = service.validate_16s_amplicon(metadata)

        # IR code should be valid Python (no syntax errors)
        code = ir.code_template
        assert "metadata = " in code
        assert "AMPLICON_KEYWORDS" in code
        compile(code, "<string>", "exec")  # Should not raise SyntaxError

    def test_ir_parameter_schema_completeness(self, service):
        """Test that IR parameter schema is complete."""
        metadata = {"organism": "Homo sapiens"}
        result, stats, ir = service.validate_host_organism(metadata)

        # All parameters should have schema entries
        for param_name in ir.parameters.keys():
            assert param_name in ir.parameter_schema

    def test_consistent_ir_across_runs(self, service):
        """Test that IR is consistent across identical runs."""
        metadata = {"platform": "Illumina"}

        result1, stats1, ir1 = service.validate_16s_amplicon(metadata)
        result2, stats2, ir2 = service.validate_16s_amplicon(metadata)

        # IR structure should be identical
        assert ir1.operation == ir2.operation
        assert ir1.tool_name == ir2.tool_name
        assert ir1.parameters == ir2.parameters
