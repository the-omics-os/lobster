"""
Pytest configuration for lobster-structural-viz package tests.

Provides necessary fixtures for testing structural visualization components.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="function", autouse=True)
def auto_mock_provider_config(request):
    """
    Automatically mock LLM provider configuration for all tests.

    This fixture:
    - Mocks the provider setup functions to avoid real LLM API calls
    - Sets environment variables for API keys
    - Provides a mock LLM instance

    Applied automatically to all tests unless test has 'no_auto_config' marker.
    """
    # Check if test explicitly requests no auto-config
    if "no_auto_config" in request.keywords:
        yield {}
        return

    # Mock environment variables
    env_patches = {
        "ANTHROPIC_API_KEY": "test-anthropic-key-12345",
        "AWS_BEDROCK_ACCESS_KEY": "test-bedrock-access-12345",
        "AWS_BEDROCK_SECRET_ACCESS_KEY": "test-bedrock-secret-12345",
        "NCBI_API_KEY": "test-ncbi-key-12345",
        "LOBSTER_LLM_PROVIDER": "anthropic",  # Set explicit provider
    }

    with patch.dict(os.environ, env_patches, clear=False):
        # Mock LLM creation functions at multiple levels
        mock_llm = MagicMock()
        mock_llm.model_name = "claude-3-opus-20240229"
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.with_structured_output = MagicMock(return_value=mock_llm)

        # Patch both the factory and the LLM classes
        patches = [
            patch("lobster.config.llm_factory.create_llm", return_value=mock_llm),
            patch("lobster.config.llm_factory.LLMFactory.create_llm", return_value=mock_llm),
        ]

        for p in patches:
            p.start()

        try:
            yield {
                "mock_llm": mock_llm,
                "env_patches": env_patches,
            }
        finally:
            for p in patches:
                p.stop()


@pytest.fixture(scope="function")
def mock_provider_config(auto_mock_provider_config) -> Dict[str, Any]:
    """Legacy fixture that delegates to auto_mock_provider_config.

    This fixture exists for backward compatibility with tests that
    explicitly request mock_provider_config. It simply returns the
    auto-configured mocks from auto_mock_provider_config.

    Args:
        auto_mock_provider_config: The automatically applied config fixture

    Returns:
        Dict[str, Any]: Provider configuration including mock LLM and env patches
    """
    return auto_mock_provider_config
