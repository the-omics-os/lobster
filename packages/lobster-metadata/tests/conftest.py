"""
Pytest configuration and fixtures for lobster-metadata package tests.

Provides the mock_provider_config fixture required by metadata_assistant tests
to ensure LLM creation is properly mocked, preventing "No provider configured" errors.

This mirrors the auto_mock_provider_config / mock_provider_config pattern from
the root tests/conftest.py, scoped for package-level test isolation.
"""

from typing import Any, Dict
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="function", autouse=True)
def auto_mock_provider_config(request, monkeypatch) -> Dict[str, Any]:
    """AUTOUSE: Automatically mock provider configuration for ALL tests.

    This fixture runs automatically for every test to ensure ConfigResolver
    is properly mocked, preventing "No provider configured" errors.

    Tests can opt-out using @pytest.mark.no_auto_config marker.
    """
    if "no_auto_config" in request.keywords:
        return {}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "anthropic")

    # Mock ConfigResolver.get_instance()
    mock_resolver_instance = Mock()
    mock_resolver_instance.resolve_provider.return_value = ("anthropic", "env")
    mock_resolver_instance.resolve_model.return_value = ("claude-sonnet-4", "default")
    mock_resolver_instance.resolve_profile.return_value = ("production", "default")
    mock_resolver_instance.is_configured.return_value = True

    try:
        from unittest.mock import patch

        patcher_resolver = patch(
            "lobster.core.config_resolver.ConfigResolver.get_instance",
            return_value=mock_resolver_instance,
        )
        patcher_resolver.start()
        request.addfinalizer(patcher_resolver.stop)
    except Exception:
        pass

    # Mock LLM creation
    mock_llm = Mock()
    mock_llm.with_config.return_value = mock_llm
    mock_llm.invoke.return_value = Mock(content="Mock LLM response")

    try:
        from unittest.mock import patch

        patcher_create_llm = patch(
            "lobster.config.llm_factory.create_llm",
            return_value=mock_llm,
        )
        patcher_create_llm.start()
        request.addfinalizer(patcher_create_llm.stop)

        patcher_factory = patch(
            "lobster.config.llm_factory.LLMFactory.create_llm",
            return_value=mock_llm,
        )
        patcher_factory.start()
        request.addfinalizer(patcher_factory.stop)
    except Exception:
        pass

    # Mock agent configurator
    mock_agent_config_instance = Mock()
    mock_agent_config_instance.get_agent_llm_params.return_value = {
        "temperature": 0.1,
    }

    try:
        from unittest.mock import patch

        patcher_agent_config = patch(
            "lobster.config.agent_config.initialize_configurator",
            return_value=mock_agent_config_instance,
        )
        patcher_agent_config.start()
        request.addfinalizer(patcher_agent_config.stop)
    except Exception:
        pass

    return {
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "config_resolver": mock_resolver_instance,
        "llm": mock_llm,
        "agent_config": mock_agent_config_instance,
    }


@pytest.fixture(scope="function")
def mock_provider_config(auto_mock_provider_config) -> Dict[str, Any]:
    """Legacy fixture that delegates to auto_mock_provider_config.

    This fixture exists for backward compatibility with tests that
    explicitly request mock_provider_config. It simply returns the
    auto-configured mocks from auto_mock_provider_config.
    """
    return auto_mock_provider_config
