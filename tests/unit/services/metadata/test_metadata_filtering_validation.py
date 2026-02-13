"""
Unit tests for disease data validation in MetadataFilteringService.

Tests the 50% coverage threshold logic and validation modes:
- Happy path (high coverage passes)
- Validation failure (low coverage with strict mode)
- Warning only (low coverage with permissive mode)
- Edge cases (exact threshold, custom threshold, disabled validation)
- Error message quality (actionable recommendations)

Author: ultrathink
Date: 2026-01-09
"""

import logging

import pytest

from lobster.services.metadata.metadata_filtering_service import (
    MetadataFilteringService,
)


@pytest.fixture
def service():
    """
    Create MetadataFilteringService instance with mock dependencies.

    Returns a minimal service for testing validation logic without
    requiring full microbiome/disease service implementations.
    """

    # Mock disease service with standardize_disease_terms method
    class MockDiseaseService:
        def standardize_disease_terms(self, df, disease_column=None):
            """Mock disease standardization that preserves input data."""
            # Count mapped vs unmapped samples
            mapped_count = df[disease_column].notna().sum()
            total_count = len(df)

            # Simple mapping: known diseases → self, unknown → None
            known_diseases = {
                "colorectal cancer": "crc",
                "ulcerative colitis": "uc",
                "crohn's disease": "cd",
                "healthy": "healthy",
                "crc": "crc",
                "uc": "uc",
                "cd": "cd",
            }

            # Apply simple mapping
            df["disease_standardized"] = df[disease_column].apply(
                lambda x: known_diseases.get(str(x).lower()) if x else None
            )

            # Calculate standardization rate
            standardized_count = df["disease_standardized"].notna().sum()
            standardization_rate = (
                (standardized_count / total_count * 100) if total_count > 0 else 0.0
            )

            unmapped_count = total_count - standardized_count

            stats = {
                "total_samples": total_count,
                "standardization_rate": standardization_rate,
                "mapping_stats": {
                    "unmapped": unmapped_count,
                },
            }

            return df, stats, None

    # Mock microbiome service (minimal - just needs to exist)
    class MockMicrobiomeService:
        pass

    # Mock disease extractor
    def mock_disease_extractor(df, study_context=None):
        """Mock extractor that looks for 'disease' column."""
        if "disease" in df.columns:
            return "disease"
        return None

    return MetadataFilteringService(
        microbiome_service=MockMicrobiomeService(),
        disease_service=MockDiseaseService(),
        disease_extractor=mock_disease_extractor,
    )


class TestDiseaseDataValidation:
    """
    Test suite for disease data validation with coverage threshold logic.

    Tests verify that the 50% threshold validation works correctly in both
    strict mode (raise error) and permissive mode (warn only).
    """

    def test_high_coverage_passes(self, service):
        """Test that high disease coverage (>50%) passes validation."""
        samples = [
            {"disease": "colorectal cancer"},  # Maps to crc
            {"disease": "ulcerative colitis"},  # Maps to uc
            {"disease": "crohn's disease"},  # Maps to cd
            {"disease": "healthy"},  # Maps to healthy
            {
                "disease": "unknown"
            },  # Unmapped (1 out of 5 = 20% unmapped, 80% coverage)
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": True,
        }

        # Should NOT raise exception (80% coverage > 50% threshold)
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)

        assert len(filtered) == 5
        assert "samples_retained" in stats
        assert stats["samples_retained"] == 5

    def test_low_coverage_fails_strict_mode(self, service):
        """Test that low disease coverage (<50%) fails in strict mode."""
        samples = [
            {"disease": "colorectal cancer"},  # Maps to crc (1 mapped)
            {"disease": "unknown1"},  # Unmapped
            {"disease": "unknown2"},  # Unmapped
            {"disease": "unknown3"},  # Unmapped
            {"disease": "unknown4"},  # Unmapped (4 unmapped = 20% coverage)
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,  # 50% required
            "strict_disease_validation": True,  # Fail hard
        }

        # Should raise ValueError with specific message
        with pytest.raises(ValueError) as exc_info:
            service.apply_filters(samples, parsed_criteria)

        error_msg = str(exc_info.value)
        assert "Insufficient disease data" in error_msg
        assert "20.0% coverage" in error_msg
        assert "(1/5 samples)" in error_msg
        assert "Recommendations:" in error_msg

    def test_low_coverage_warns_permissive_mode(self, service, caplog):
        """Test that low coverage logs warning in permissive mode."""
        samples = [
            {"disease": "colorectal cancer"},  # 1 mapped, 4 unmapped = 20%
            {"disease": "unknown1"},
            {"disease": "unknown2"},
            {"disease": "unknown3"},
            {"disease": "unknown4"},
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": False,  # Permissive mode
        }

        # Should NOT raise exception, but should log warning
        with caplog.at_level(logging.WARNING):
            filtered, stats, ir = service.apply_filters(samples, parsed_criteria)

        assert len(filtered) == 5  # All samples retained
        assert "Insufficient disease data" in caplog.text
        assert "20.0% coverage" in caplog.text

    def test_exact_threshold_passes(self, service):
        """Test that exactly 50% coverage passes validation."""
        samples = [
            {"disease": "crc"},  # Mapped
            {"disease": "uc"},  # Mapped (2 mapped, 2 unmapped = 50% exact)
            {"disease": "unknown1"},  # Unmapped
            {"disease": "unknown2"},  # Unmapped
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,  # Exactly 50%
            "strict_disease_validation": True,
        }

        # Should pass (50.0% >= 50.0%)
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 4

    def test_custom_threshold(self, service):
        """Test that custom threshold (30%) works correctly."""
        samples = [
            {"disease": "crc"},  # 1 mapped, 2 unmapped = 33.3%
            {"disease": "unknown1"},
            {"disease": "unknown2"},
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.3,  # 30% threshold
            "strict_disease_validation": True,
        }

        # Should pass (33.3% > 30%)
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 3

    def test_disabled_validation(self, service):
        """Test that validation can be disabled (threshold=0.0)."""
        samples = [
            {"disease": "unknown1"},  # 0% coverage
            {"disease": "unknown2"},
            {"disease": "unknown3"},
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.0,  # Disabled
            "strict_disease_validation": True,
        }

        # Should NOT raise exception even with 0% coverage
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 3

    def test_100_percent_coverage(self, service):
        """Test that 100% coverage passes validation."""
        samples = [
            {"disease": "crc"},
            {"disease": "uc"},
            {"disease": "cd"},
            {"disease": "healthy"},
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": True,
        }

        # Should pass (100% coverage)
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 4

    def test_error_message_actionable(self, service):
        """Test that error message contains actionable recommendations."""
        samples = [{"disease": "unknown"} for _ in range(10)]  # 0% coverage

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": True,
        }

        with pytest.raises(ValueError) as exc_info:
            service.apply_filters(samples, parsed_criteria)

        error_msg = str(exc_info.value)
        # Check for 4 recommendations
        assert "1. Use different datasets" in error_msg
        assert "2. Manually enrich samples" in error_msg
        assert "3. Lower threshold" in error_msg
        assert "4. Skip disease filtering" in error_msg

    def test_empty_samples_no_error(self, service):
        """Test that empty samples list doesn't crash validation."""
        samples = []

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": True,
        }

        # Should handle gracefully
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 0

    def test_missing_disease_column_no_error(self, service):
        """Test that samples without disease column don't crash validation."""
        samples = [
            {"other_field": "value1"},
            {"other_field": "value2"},
        ]

        parsed_criteria = {
            "standardize_disease": True,
            "min_disease_coverage": 0.5,
            "strict_disease_validation": True,
        }

        # Should handle gracefully (no disease column found)
        filtered, stats, ir = service.apply_filters(samples, parsed_criteria)
        assert len(filtered) == 2  # Samples retained (no disease filter applied)
