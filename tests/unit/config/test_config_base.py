"""Tests for shared configuration infrastructure."""

import pytest

from lobster.config.base_config import ProviderConfigBase
from lobster.config.constants import VALID_PROFILES, VALID_PROVIDERS


def test_valid_providers_includes_all():
    """Ensure all expected providers are in VALID_PROVIDERS."""
    assert "anthropic" in VALID_PROVIDERS
    assert "bedrock" in VALID_PROVIDERS
    assert "ollama" in VALID_PROVIDERS
    assert "gemini" in VALID_PROVIDERS
    assert "azure" in VALID_PROVIDERS
    assert len(VALID_PROVIDERS) == 5


def test_valid_profiles_includes_all():
    """Ensure all expected profiles are in VALID_PROFILES."""
    assert "development" in VALID_PROFILES
    assert "production" in VALID_PROFILES
    assert len(VALID_PROFILES) == 5


def test_workspace_config_inherits_base():
    """Verify WorkspaceProviderConfig inherits from ProviderConfigBase."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    assert issubclass(WorkspaceProviderConfig, ProviderConfigBase)


def test_global_config_inherits_base():
    """Verify GlobalProviderConfig inherits from ProviderConfigBase."""
    from lobster.config.global_config import GlobalProviderConfig

    assert issubclass(GlobalProviderConfig, ProviderConfigBase)


def test_workspace_config_model_suffix():
    """Verify workspace config uses '_model' suffix."""
    from lobster.config.workspace_config import WorkspaceProviderConfig

    config = WorkspaceProviderConfig()
    assert config.model_field_suffix == "_model"


def test_global_config_model_suffix():
    """Verify global config uses '_default_model' suffix."""
    from lobster.config.global_config import GlobalProviderConfig

    config = GlobalProviderConfig()
    assert config.model_field_suffix == "_default_model"
