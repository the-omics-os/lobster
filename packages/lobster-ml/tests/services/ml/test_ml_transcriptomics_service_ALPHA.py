"""
Smoke tests for MLTranscriptomicsService (experimental ALPHA).

Note: This is an ALPHA service marked as experimental.
Full integration tests will be added as the API stabilizes.
"""

import pytest


def test_service_import():
    """Verify MLTranscriptomicsService is importable."""
    from lobster.services.ml.ml_transcriptomics_service_ALPHA import (
        MLTranscriptomicsService,
    )

    assert MLTranscriptomicsService is not None
    assert hasattr(MLTranscriptomicsService, "name")
    assert hasattr(MLTranscriptomicsService, "version")


def test_service_instantiation():
    """Verify MLTranscriptomicsService can be instantiated."""
    from lobster.services.ml.ml_transcriptomics_service_ALPHA import (
        MLTranscriptomicsService,
    )

    service = MLTranscriptomicsService()
    assert service is not None
    assert service.name == "ml_transcriptomics"
