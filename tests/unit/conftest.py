"""
Unit-test directory conftest.

Activates fixtures that should be autouse within tests/unit/ only,
keeping them out of integration/system tests that need real config resolution.
"""

import pytest


@pytest.fixture(autouse=True)
def auto_mock_provider_config_for_unit(auto_mock_provider_config):
    """Auto-apply provider config mocking for all unit tests."""
    return auto_mock_provider_config
