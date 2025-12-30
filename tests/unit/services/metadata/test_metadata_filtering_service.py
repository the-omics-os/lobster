"""
Unit tests for MetadataFilteringService.

Tests the natural language criteria parsing and filter application logic
extracted from metadata_assistant.py.
"""

import pytest
from unittest.mock import Mock, MagicMock

from lobster.services.metadata.metadata_filtering_service import MetadataFilteringService


class TestParseCriteria:
    """Tests for natural language criteria parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = MetadataFilteringService()

    def test_parse_16s_keywords(self):
        """Test 16S amplicon detection."""
        # Direct keyword
        result = self.service.parse_criteria("16S human fecal")
        assert result["check_16s"] is True

        # Alternative keyword
        result = self.service.parse_criteria("amplicon sequencing")
        assert result["check_16s"] is True

        # Not present
        result = self.service.parse_criteria("human fecal")
        assert result["check_16s"] is False

    def test_parse_shotgun_keywords(self):
        """Test shotgun metagenomic detection."""
        # Direct keyword
        result = self.service.parse_criteria("shotgun sequencing")
        assert result["check_shotgun"] is True

        # WGS keyword
        result = self.service.parse_criteria("wgs data")
        assert result["check_shotgun"] is True

        # Metagenomic keyword
        result = self.service.parse_criteria("metagenomic analysis")
        assert result["check_shotgun"] is True

    def test_parse_host_organisms(self):
        """Test host organism detection."""
        result = self.service.parse_criteria("16S human fecal")
        assert "Human" in result["host_organisms"]

        result = self.service.parse_criteria("mouse gut samples")
        assert "Mouse" in result["host_organisms"]

        result = self.service.parse_criteria("homo sapiens samples")
        assert "Human" in result["host_organisms"]

    def test_parse_sample_types(self):
        """Test sample type detection."""
        result = self.service.parse_criteria("fecal samples")
        assert "fecal" in result["sample_types"]

        result = self.service.parse_criteria("stool microbiome")
        assert "fecal" in result["sample_types"]

        result = self.service.parse_criteria("gut tissue biopsy")
        assert "gut" in result["sample_types"]

    def test_parse_disease_keywords(self):
        """Test disease standardization detection."""
        for keyword in ["CRC", "UC", "CD", "cancer", "colitis", "crohn", "healthy", "control"]:
            result = self.service.parse_criteria(f"samples with {keyword}")
            assert result["standardize_disease"] is True, f"Failed for keyword: {keyword}"

        # Bug 3 fix (DataBioMix): Disease extraction now ALWAYS enabled
        # Even without disease keywords, we extract disease for valuable metadata
        result = self.service.parse_criteria("16S human fecal")
        assert result["standardize_disease"] is True  # Changed from False

    def test_parse_combined_criteria(self):
        """Test parsing combined criteria string."""
        result = self.service.parse_criteria("16S shotgun human fecal CRC")

        assert result["check_16s"] is True
        assert result["check_shotgun"] is True
        assert "Human" in result["host_organisms"]
        assert "fecal" in result["sample_types"]
        assert result["standardize_disease"] is True

    def test_parse_case_insensitive(self):
        """Test case insensitivity."""
        result = self.service.parse_criteria("16S HUMAN FECAL CRC")

        assert result["check_16s"] is True
        assert "Human" in result["host_organisms"]
        assert "fecal" in result["sample_types"]
        assert result["standardize_disease"] is True


class TestApplyFilters:
    """Tests for filter application."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock services
        self.mock_microbiome = Mock()
        self.mock_disease = Mock()

        self.service = MetadataFilteringService(
            microbiome_service=self.mock_microbiome,
            disease_service=self.mock_disease,
        )

    def test_service_unavailable_returns_unfiltered(self):
        """Test that missing services returns samples unchanged."""
        service = MetadataFilteringService()  # No services
        samples = [{"id": 1}, {"id": 2}]
        parsed = {"check_16s": True}

        result, stats, ir = service.apply_filters(samples, parsed)

        assert result == samples
        assert stats["skipped"] is True

    def test_sequencing_filter_or_logic(self):
        """Test OR logic for sequencing filters."""
        # Set up mocks - first sample is 16S, second is shotgun
        self.mock_microbiome.validate_16s_amplicon.side_effect = [
            (True, {}, None),  # Sample 1 is 16S
            (False, {}, None),  # Sample 2 is not 16S
        ]
        self.mock_microbiome.validate_shotgun.side_effect = [
            (False, {}, None),  # Sample 1 is not shotgun (but already matched 16S)
            (True, {}, None),   # Sample 2 is shotgun
        ]

        samples = [{"id": 1}, {"id": 2}]
        parsed = {"check_16s": True, "check_shotgun": True}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        # Both samples should be retained (OR logic)
        assert len(result) == 2
        assert result[0]["_matched_sequencing_method"] == "16S_amplicon"
        assert result[1]["_matched_sequencing_method"] == "shotgun_metagenomic"

    def test_host_filter(self):
        """Test host organism filtering."""
        self.mock_microbiome.validate_host_organism.side_effect = [
            (True, {}, None),   # Sample 1 matches
            (False, {}, None),  # Sample 2 doesn't match
        ]

        samples = [{"id": 1}, {"id": 2}]
        parsed = {"host_organisms": ["Human"]}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_stats_calculation(self):
        """Test that stats are calculated correctly."""
        samples = [{"id": i} for i in range(10)]
        parsed = {}  # No filters

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert stats["samples_original"] == 10
        assert stats["samples_retained"] == 10
        assert stats["retention_rate"] == 100.0

    def test_ir_generation(self):
        """Test that IR is generated with correct structure."""
        samples = [{"id": 1}]
        parsed = {"check_16s": True}

        self.mock_microbiome.validate_16s_amplicon.return_value = (True, {}, None)

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert ir.operation == "metadata_filtering"
        assert ir.tool_name == "MetadataFilteringService.apply_filters"
        assert "16S amplicon" in ir.description
        assert ir.parameters["check_16s"] is True


class TestIsAvailable:
    """Tests for service availability check."""

    def test_available_with_microbiome_service(self):
        """Test service is available when microbiome service is provided."""
        service = MetadataFilteringService(microbiome_service=Mock())
        assert service.is_available is True

    def test_unavailable_without_services(self):
        """Test service is unavailable without dependencies."""
        service = MetadataFilteringService()
        assert service.is_available is False
