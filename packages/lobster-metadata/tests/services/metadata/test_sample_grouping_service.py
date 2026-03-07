"""
Smoke tests for SampleGroupingService.

Tests basic service instantiation and method availability.
"""

import pytest


class TestSampleGroupingService:
    """Test SampleGroupingService basic functionality."""

    def test_service_imports(self):
        """Test that service can be imported."""
        from lobster.services.metadata.sample_grouping_service import (
            SampleGroupingService,
        )

        assert SampleGroupingService is not None

    def test_service_instantiation(self):
        """Test that service can be instantiated."""
        from lobster.services.metadata.sample_grouping_service import (
            SampleGroupingService,
        )

        service = SampleGroupingService()
        assert service is not None
        assert hasattr(service, "group_samples")
