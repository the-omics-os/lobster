"""
Smoke tests for ModalityDetectionService.

Tests basic service instantiation and method availability.
"""

import pytest


class TestModalityDetectionService:
    """Test ModalityDetectionService basic functionality."""

    def test_service_imports(self):
        """Test that service can be imported."""
        from lobster.services.data_management.modality_detection_service import (
            ModalityDetectionService,
        )

        assert ModalityDetectionService is not None

    def test_service_instantiation(self):
        """Test that service can be instantiated."""
        from lobster.services.data_management.modality_detection_service import (
            ModalityDetectionService,
        )

        service = ModalityDetectionService()
        assert service is not None
        assert hasattr(service, "detect_modality")
