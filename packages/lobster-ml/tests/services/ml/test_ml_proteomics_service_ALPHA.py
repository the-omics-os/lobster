"""
Smoke tests for MLProteomicsService (experimental ALPHA).

Note: This is an ALPHA service marked as experimental.
Full integration tests will be added as the API stabilizes.
"""

import pytest


def test_service_import():
    """Verify MLProteomicsService is importable."""
    from lobster.services.ml.ml_proteomics_service_ALPHA import (
        MLProteomicsService,
    )

    assert MLProteomicsService is not None
    assert hasattr(MLProteomicsService, "name")
    assert hasattr(MLProteomicsService, "version")


def test_service_instantiation():
    """Verify MLProteomicsService can be instantiated."""
    from lobster.services.ml.ml_proteomics_service_ALPHA import (
        MLProteomicsService,
    )

    service = MLProteomicsService()
    assert service is not None
    assert service.name == "ml_proteomics"
