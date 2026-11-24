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


@pytest.fixture(autouse=True)
def no_omics_os_key_check_network(monkeypatch):
    """`lobster init` checks a saved Omics-OS key over the network; unit
    tests answer "could not reach" instead. Tests of the check replace it."""
    from lobster.config import cloud_setup

    monkeypatch.setattr(cloud_setup, "_probe_key", lambda server, key: None)
