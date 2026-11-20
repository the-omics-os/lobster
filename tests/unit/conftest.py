"""
Unit-test directory conftest.

Activates fixtures that should be autouse within tests/unit/ only,
keeping them out of integration/system tests that need real config resolution.
"""

import sys

import pytest


@pytest.fixture(autouse=True)
def auto_mock_provider_config_for_unit(auto_mock_provider_config):
    """Auto-apply provider config mocking for all unit tests."""
    return auto_mock_provider_config


@pytest.fixture(autouse=True)
def restore_model_pricing():
    """Undo in-place changes to the shared pricing table after each test.

    Listing a live catalog (OpenRouter) merges its models into MODEL_PRICING;
    left there, they change the table for every later test in the worker.
    """
    settings = sys.modules.get("lobster.config.settings")
    saved = dict(settings.MODEL_PRICING) if settings else None
    yield
    settings = sys.modules.get("lobster.config.settings")
    if settings is None:
        return
    if saved is None:  # first imported by this test: rebuild the static table
        from lobster.config.providers.registry import ProviderRegistry

        saved = ProviderRegistry.get_all_models_with_pricing()
    table = settings.MODEL_PRICING
    table.clear()
    table.update(saved)
