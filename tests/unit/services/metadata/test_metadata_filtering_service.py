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


class TestSampleTypeFilter:
    """Tests for sample type filtering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_microbiome = Mock()
        self.mock_disease = Mock()

        self.service = MetadataFilteringService(
            microbiome_service=self.mock_microbiome,
            disease_service=self.mock_disease,
        )

    def test_sample_type_filter_applied(self):
        """Test that sample type filter is applied when sample_types in criteria."""
        # Set up mock - first sample matches fecal, second doesn't
        self.mock_microbiome.validate_sample_type.side_effect = [
            ({}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "isolation_source"}, None),
            ({}, {"is_valid": False, "matched_sample_type": None, "matched_field": None}, None),
        ]

        samples = [
            {"id": 1, "isolation_source": "human feces"},
            {"id": 2, "isolation_source": "blood"},
        ]
        parsed = {"sample_types": ["fecal"]}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["_matched_sample_type"] == "fecal"
        assert "sample_type(1/2)" in stats["filters_applied"]

    def test_sample_type_filter_or_logic(self):
        """Test OR logic when multiple sample types allowed."""
        # Set up mock - sample 1 is fecal, sample 2 is gut
        self.mock_microbiome.validate_sample_type.side_effect = [
            ({}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "isolation_source"}, None),
            ({}, {"is_valid": True, "matched_sample_type": "gut", "matched_field": "body_site"}, None),
        ]

        samples = [
            {"id": 1, "isolation_source": "stool"},
            {"id": 2, "body_site": "colon biopsy"},
        ]
        parsed = {"sample_types": ["fecal", "gut"]}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        # Both samples should be retained (OR logic across sample types)
        assert len(result) == 2
        assert result[0]["_matched_sample_type"] == "fecal"
        assert result[1]["_matched_sample_type"] == "gut"

    def test_sample_type_filter_stats(self):
        """Test stats include sample type filtering info."""
        self.mock_microbiome.validate_sample_type.side_effect = [
            ({}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "isolation_source"}, None),
            ({}, {"is_valid": False, "matched_sample_type": None}, None),
            ({}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "body_site"}, None),
        ]

        samples = [{"id": i} for i in range(3)]
        parsed = {"sample_types": ["fecal"]}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert stats["samples_original"] == 3
        assert stats["samples_retained"] == 2
        assert stats["retention_rate"] == pytest.approx(66.67, rel=0.01)
        assert any("sample_type" in step for step in stats["filters_applied"])

    def test_sample_type_combined_with_other_filters(self):
        """Test sample type filter combined with sequencing and host filters."""
        # All filters pass for sample 1, sample 2 fails host
        self.mock_microbiome.validate_16s_amplicon.return_value = (True, {}, None)
        self.mock_microbiome.validate_host_organism.side_effect = [
            (True, {}, None),
            (False, {}, None),
        ]
        self.mock_microbiome.validate_sample_type.return_value = (
            {}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "isolation_source"}, None
        )

        samples = [{"id": 1}, {"id": 2}]
        parsed = {
            "check_16s": True,
            "host_organisms": ["Human"],
            "sample_types": ["fecal"],
        }

        result, stats, ir = self.service.apply_filters(samples, parsed)

        # Only sample 1 should pass (sample 2 fails host filter before reaching sample_type)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_sample_type_not_applied_when_empty(self):
        """Test sample type filter not called when sample_types is empty."""
        samples = [{"id": 1}]
        parsed = {"sample_types": []}  # Empty list

        result, stats, ir = self.service.apply_filters(samples, parsed)

        # Should not call validate_sample_type
        self.mock_microbiome.validate_sample_type.assert_not_called()
        assert len(result) == 1

    def test_sample_type_ir_includes_filter(self):
        """Test that IR description includes sample type filter."""
        self.mock_microbiome.validate_sample_type.return_value = (
            {}, {"is_valid": True, "matched_sample_type": "fecal", "matched_field": "isolation_source"}, None
        )

        samples = [{"id": 1}]
        parsed = {"sample_types": ["fecal"]}

        result, stats, ir = self.service.apply_filters(samples, parsed)

        assert "sample_type" in ir.description
        assert ir.parameters["sample_types"] == ["fecal"]


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
