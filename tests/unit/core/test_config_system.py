"""
Unit tests for unified configuration system.

Tests cover:
- WorkspaceProviderConfig: save/load, validation, defaults
- GlobalProviderConfig: save/load, validation, defaults
- ConfigResolver: 6-layer priority resolution
- Error handling: corrupted JSON, invalid schemas, missing files
- Edge cases: empty configs, invalid provider names, permission errors
- Priority resolution: runtime > workspace > global > env > auto-detect > default
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from lobster.core.config_resolver import ConfigResolver
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.workspace_config import WorkspaceProviderConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def temp_global_config_dir(tmp_path, monkeypatch):
    """Create temporary global config directory and override home."""
    config_dir = tmp_path / ".config" / "lobster"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Patch the CONFIG_DIR in global_config module
    import lobster.core.global_config as gc
    monkeypatch.setattr(gc, "CONFIG_DIR", config_dir)

    return config_dir


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for tests."""
    env_vars = [
        "LOBSTER_LLM_PROVIDER",
        "OLLAMA_DEFAULT_MODEL",
        "ANTHROPIC_API_KEY",
        "AWS_BEDROCK_ACCESS_KEY",
        "AWS_BEDROCK_SECRET_ACCESS_KEY",
        "LOBSTER_PROFILE",
        "OLLAMA_BASE_URL",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# WorkspaceProviderConfig Tests
# =============================================================================


class TestWorkspaceProviderConfig:
    """Test workspace configuration functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkspaceProviderConfig()

        assert config.global_provider is None
        assert config.ollama_model is None
        assert config.ollama_host == "http://localhost:11434"
        assert config.per_agent_providers == {}
        assert config.per_agent_models == {}
        assert config.profile == "production"

    def test_valid_provider_validation(self):
        """Test valid provider names."""
        for provider in ["bedrock", "anthropic", "ollama"]:
            config = WorkspaceProviderConfig(global_provider=provider)
            assert config.global_provider == provider

    def test_invalid_provider_validation(self):
        """Test invalid provider name raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            WorkspaceProviderConfig(global_provider="invalid_provider")

        assert "Invalid provider" in str(excinfo.value)

    def test_valid_profile_validation(self):
        """Test valid profile names."""
        for profile in ["development", "production", "ultra", "godmode", "hybrid"]:
            config = WorkspaceProviderConfig(profile=profile)
            assert config.profile == profile

    def test_invalid_profile_validation(self):
        """Test invalid profile name raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            WorkspaceProviderConfig(profile="invalid_profile")

        assert "Invalid profile" in str(excinfo.value)

    def test_per_agent_provider_validation(self):
        """Test per-agent provider validation."""
        # Valid per-agent providers
        config = WorkspaceProviderConfig(
            per_agent_providers={
                "supervisor": "ollama",
                "data_expert": "bedrock"
            }
        )
        assert config.per_agent_providers["supervisor"] == "ollama"

        # Invalid per-agent provider
        with pytest.raises(ValidationError) as excinfo:
            WorkspaceProviderConfig(
                per_agent_providers={"supervisor": "invalid"}
            )
        assert "Invalid provider" in str(excinfo.value)

    def test_save_and_load(self, temp_workspace):
        """Test saving and loading configuration."""
        # Create config
        config = WorkspaceProviderConfig(
            global_provider="ollama",
            ollama_model="llama3:70b-instruct",
            profile="ultra",
            per_agent_providers={"supervisor": "ollama"}
        )

        # Save
        config.save(temp_workspace)

        # Verify file exists
        config_path = temp_workspace / "provider_config.json"
        assert config_path.exists()

        # Load
        loaded_config = WorkspaceProviderConfig.load(temp_workspace)

        # Verify values
        assert loaded_config.global_provider == "ollama"
        assert loaded_config.ollama_model == "llama3:70b-instruct"
        assert loaded_config.profile == "ultra"
        assert loaded_config.per_agent_providers["supervisor"] == "ollama"

    def test_load_missing_file(self, temp_workspace):
        """Test loading when file doesn't exist returns defaults."""
        config = WorkspaceProviderConfig.load(temp_workspace)

        assert config.global_provider is None
        assert config.profile == "production"

    def test_load_corrupted_json(self, temp_workspace):
        """Test loading corrupted JSON returns defaults."""
        config_path = temp_workspace / "provider_config.json"
        config_path.write_text("{ invalid json }}")

        config = WorkspaceProviderConfig.load(temp_workspace)

        # Should return defaults without crashing
        assert config.global_provider is None
        assert config.profile == "production"

    def test_load_invalid_schema(self, temp_workspace):
        """Test loading invalid schema returns defaults."""
        config_path = temp_workspace / "provider_config.json"
        config_path.write_text('{"global_provider": "invalid_provider"}')

        config = WorkspaceProviderConfig.load(temp_workspace)

        # Should return defaults without crashing
        assert config.global_provider is None

    def test_exists(self, temp_workspace):
        """Test checking if config file exists."""
        assert not WorkspaceProviderConfig.exists(temp_workspace)

        config = WorkspaceProviderConfig(global_provider="ollama")
        config.save(temp_workspace)

        assert WorkspaceProviderConfig.exists(temp_workspace)

    def test_reset(self):
        """Test resetting configuration to defaults."""
        config = WorkspaceProviderConfig(
            global_provider="ollama",
            ollama_model="llama3:70b",
            profile="ultra",
            per_agent_providers={"supervisor": "ollama"}
        )

        config.reset()

        assert config.global_provider is None
        assert config.ollama_model is None
        assert config.profile == "production"
        assert config.per_agent_providers == {}


# =============================================================================
# GlobalProviderConfig Tests
# =============================================================================


@pytest.mark.no_auto_config
class TestGlobalProviderConfig:
    """Test global user configuration functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GlobalProviderConfig()

        assert config.default_provider is None
        assert config.default_profile == "production"
        assert config.ollama_default_model is None
        assert config.ollama_default_host == "http://localhost:11434"

    def test_valid_provider_validation(self):
        """Test valid provider names."""
        for provider in ["bedrock", "anthropic", "ollama"]:
            config = GlobalProviderConfig(default_provider=provider)
            assert config.default_provider == provider

    def test_invalid_provider_validation(self):
        """Test invalid provider name raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            GlobalProviderConfig(default_provider="invalid_provider")

        assert "Invalid provider" in str(excinfo.value)

    def test_save_and_load(self, temp_global_config_dir):
        """Test saving and loading global configuration."""
        # Create config
        config = GlobalProviderConfig(
            default_provider="ollama",
            ollama_default_model="mixtral:8x7b-instruct",
            default_profile="development"
        )

        # Save
        config.save()

        # Verify file exists
        config_path = temp_global_config_dir / "providers.json"
        assert config_path.exists()

        # Load
        loaded_config = GlobalProviderConfig.load()

        # Verify values
        assert loaded_config.default_provider == "ollama"
        assert loaded_config.ollama_default_model == "mixtral:8x7b-instruct"
        assert loaded_config.default_profile == "development"

    def test_load_missing_file(self, temp_global_config_dir):
        """Test loading when file doesn't exist returns defaults."""
        config = GlobalProviderConfig.load()

        assert config.default_provider is None
        assert config.default_profile == "production"

    def test_exists(self, temp_global_config_dir):
        """Test checking if global config file exists."""
        assert not GlobalProviderConfig.exists()

        config = GlobalProviderConfig(default_provider="ollama")
        config.save()

        assert GlobalProviderConfig.exists()

    def test_reset(self):
        """Test resetting global configuration to defaults."""
        config = GlobalProviderConfig(
            default_provider="ollama",
            ollama_default_model="llama3:70b",
            default_profile="ultra"
        )

        config.reset()

        assert config.default_provider is None
        assert config.ollama_default_model is None
        assert config.default_profile == "production"

    def test_get_config_path(self, temp_global_config_dir):
        """Test getting config file path."""
        path = GlobalProviderConfig.get_config_path()
        assert path == temp_global_config_dir / "providers.json"


# =============================================================================
# ConfigResolver Tests
# =============================================================================


@pytest.mark.no_auto_config
class TestConfigResolver:
    """Test configuration resolution with priority hierarchy."""

    def test_resolve_provider_runtime_override(self, temp_workspace, clean_env):
        """Test Layer 1: Runtime override has highest priority."""
        # Setup workspace config
        workspace_config = WorkspaceProviderConfig(global_provider="bedrock")
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider(runtime_override="ollama")

        assert provider == "ollama"
        assert source == "runtime flag --provider"

    def test_resolve_provider_workspace_config(self, temp_workspace, clean_env):
        """Test Layer 2: Workspace config."""
        workspace_config = WorkspaceProviderConfig(global_provider="ollama")
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider()

        assert provider == "ollama"
        assert source == "workspace config"

    def test_resolve_provider_global_config(
        self, temp_workspace, temp_global_config_dir, clean_env
    ):
        """Test Layer 3: Global user config."""
        global_config = GlobalProviderConfig(default_provider="anthropic")
        global_config.save()

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider()

        assert provider == "anthropic"
        assert source == "global user config"

    def test_resolve_provider_env_variable(self, temp_workspace, monkeypatch):
        """Test Layer 4: Environment variable."""
        monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "bedrock")

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider()

        assert provider == "bedrock"
        assert source == "environment variable LOBSTER_LLM_PROVIDER"

    @patch("lobster.config.llm_factory.LLMFactory._is_ollama_running")
    def test_resolve_provider_auto_detect_ollama(
        self, mock_ollama, temp_workspace, clean_env
    ):
        """Test Layer 5: Auto-detection (Ollama running)."""
        mock_ollama.return_value = True

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider()

        assert provider == "ollama"
        assert source == "auto-detected (Ollama running)"

    def test_resolve_provider_auto_detect_anthropic(
        self, temp_workspace, monkeypatch, clean_env
    ):
        """Test Layer 5: Auto-detection (Anthropic API key)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        # Mock Ollama not running
        with patch("lobster.config.llm_factory.LLMFactory._is_ollama_running", return_value=False):
            resolver = ConfigResolver(temp_workspace)
            provider, source = resolver.resolve_provider()

        assert provider == "anthropic"
        assert source == "auto-detected (ANTHROPIC_API_KEY set)"

    def test_resolve_provider_default(self, temp_workspace, clean_env):
        """Test Layer 6: Default fallback."""
        # Mock Ollama not running
        with patch("lobster.config.llm_factory.LLMFactory._is_ollama_running", return_value=False):
            resolver = ConfigResolver(temp_workspace)
            provider, source = resolver.resolve_provider()

        assert provider == "bedrock"
        assert source == "default (no configuration found)"

    def test_resolve_model_workspace_per_agent(self, temp_workspace):
        """Test per-agent model resolution from workspace."""
        workspace_config = WorkspaceProviderConfig(
            per_agent_models={"supervisor": "llama3:70b-instruct"}
        )
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        model, source = resolver.resolve_model("supervisor", provider="ollama")

        assert model == "llama3:70b-instruct"
        assert "workspace config (agent 'supervisor')" in source

    def test_resolve_model_workspace_global(self, temp_workspace):
        """Test global Ollama model from workspace."""
        workspace_config = WorkspaceProviderConfig(
            ollama_model="mixtral:8x7b-instruct"
        )
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        model, source = resolver.resolve_model(provider="ollama")

        assert model == "mixtral:8x7b-instruct"
        assert source == "workspace config (ollama model)"

    def test_resolve_profile_workspace(self, temp_workspace):
        """Test profile resolution from workspace."""
        workspace_config = WorkspaceProviderConfig(profile="ultra")
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        profile, source = resolver.resolve_profile()

        assert profile == "ultra"
        assert source == "workspace config"

    def test_resolve_profile_env_variable(self, tmp_path, monkeypatch):
        """Test profile resolution from environment."""
        # Use tmp_path without creating workspace config
        # to test env variable priority (layer 4)
        monkeypatch.setenv("LOBSTER_PROFILE", "development")

        # Create empty workspace dir without config file
        empty_workspace = tmp_path / "empty_workspace"
        empty_workspace.mkdir()

        resolver = ConfigResolver(empty_workspace)
        profile, source = resolver.resolve_profile()

        assert profile == "development"
        assert source == "environment variable LOBSTER_PROFILE"

    def test_resolve_per_agent_provider(self, temp_workspace):
        """Test per-agent provider override."""
        workspace_config = WorkspaceProviderConfig(
            per_agent_providers={"supervisor": "ollama"}
        )
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_per_agent_provider("supervisor", "bedrock")

        assert provider == "ollama"
        assert "workspace config (agent 'supervisor')" in source

    def test_resolve_per_agent_provider_fallback(self, temp_workspace):
        """Test fallback to global provider for agent without override."""
        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_per_agent_provider("data_expert", "bedrock")

        assert provider == "bedrock"
        assert source == "global provider"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_workspace_config(self, temp_workspace):
        """Test loading empty workspace config file."""
        config_path = temp_workspace / "provider_config.json"
        config_path.write_text("{}")

        config = WorkspaceProviderConfig.load(temp_workspace)

        # Should load with defaults
        assert config.global_provider is None
        assert config.profile == "production"

    def test_resolver_without_workspace(self):
        """Test resolver without workspace path."""
        resolver = ConfigResolver(workspace_path=None)

        # Should still work with defaults
        provider, source = resolver.resolve_provider()
        assert provider in ["bedrock", "anthropic", "ollama"]

    def test_invalid_runtime_provider_fallback(self, temp_workspace):
        """Test invalid runtime provider falls back to next layer."""
        workspace_config = WorkspaceProviderConfig(global_provider="ollama")
        workspace_config.save(temp_workspace)

        resolver = ConfigResolver(temp_workspace)
        provider, source = resolver.resolve_provider(runtime_override="invalid")

        # Should fallback to workspace config
        assert provider == "ollama"
        assert source == "workspace config"

    def test_permission_error_on_save(self, temp_workspace):
        """Test permission error during save."""
        config = WorkspaceProviderConfig(global_provider="ollama")

        # Make workspace read-only
        temp_workspace.chmod(0o444)

        try:
            with pytest.raises(Exception):  # Should raise IOError or PermissionError
                config.save(temp_workspace)
        finally:
            # Restore permissions
            temp_workspace.chmod(0o755)
