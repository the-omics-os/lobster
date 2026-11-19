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
    if settings is None:
        yield
        return
    table = settings.MODEL_PRICING
    saved = dict(table)
    yield
    table.clear()
    table.update(saved)
