"""
Unit tests for lobster.config.agent_defaults.

Covers:
- get_agent_params() default behaviour (no error for unknown agents)
- get_agent_params() environment variable overrides
- get_current_profile() default and env var override
- ThinkingConfig.to_dict() enabled vs disabled
- THINKING_PRESETS contents
"""

import os
from unittest.mock import patch

import pytest

from lobster.config.agent_defaults import (
    THINKING_PRESETS,
    ThinkingConfig,
    get_agent_params,
    get_current_profile,
)


# =============================================================================
# get_agent_params tests
# =============================================================================


class TestGetAgentParams:
    """Test get_agent_params() resolution logic."""

    def test_unknown_agent_returns_default_temperature(self):
        """Any unknown agent name must return temperature=1.0 without raising."""
        params = get_agent_params("totally_unknown_agent_xyz")
        assert params == {"temperature": 1.0}

    def test_known_agent_returns_default_temperature(self):
        """A real agent name with no overrides also returns temperature=1.0."""
        params = get_agent_params("supervisor")
        assert params["temperature"] == 1.0

    def test_env_var_overrides_temperature(self):
        """LOBSTER_{AGENT}_TEMPERATURE env var should override default temperature."""
        with patch.dict(os.environ, {"LOBSTER_SUPERVISOR_TEMPERATURE": "0.5"}):
            params = get_agent_params("supervisor")
        assert params["temperature"] == 0.5

    def test_env_var_temperature_invalid_value_falls_back(self):
        """A non-numeric LOBSTER_{AGENT}_TEMPERATURE must be silently ignored."""
        with patch.dict(os.environ, {"LOBSTER_SUPERVISOR_TEMPERATURE": "not_a_float"}):
            params = get_agent_params("supervisor")
        assert params["temperature"] == 1.0

    def test_env_var_thinking_preset_applies(self):
        """LOBSTER_{AGENT}_THINKING env var enables the specified preset."""
        with patch.dict(os.environ, {"LOBSTER_SUPERVISOR_THINKING": "light"}):
            params = get_agent_params("supervisor")
        assert "additional_model_request_fields" in params
        assert params["additional_model_request_fields"]["thinking"]["budget_tokens"] == 1000

    def test_global_thinking_override_applies(self):
        """LOBSTER_GLOBAL_THINKING applies thinking preset to all agents."""
        with patch.dict(os.environ, {"LOBSTER_GLOBAL_THINKING": "extended"}):
            params = get_agent_params("any_agent")
        assert "additional_model_request_fields" in params
        assert params["additional_model_request_fields"]["thinking"]["budget_tokens"] == 5000

    def test_disabled_thinking_preset_produces_no_key(self):
        """When no thinking env var is set the returned dict has no additional_model_request_fields."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure no LOBSTER_* thinking vars are set
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("LOBSTER_") or "_THINKING" not in k
            }
            with patch.dict(os.environ, env, clear=True):
                params = get_agent_params("supervisor")
        assert "additional_model_request_fields" not in params

    def test_unknown_thinking_preset_is_ignored(self):
        """An unrecognised LOBSTER_GLOBAL_THINKING value must be silently ignored."""
        with patch.dict(os.environ, {"LOBSTER_GLOBAL_THINKING": "nonexistent_preset"}):
            params = get_agent_params("supervisor")
        assert "additional_model_request_fields" not in params

    def test_returns_dict(self):
        """Return value must always be a dict."""
        result = get_agent_params("whatever")
        assert isinstance(result, dict)


# =============================================================================
# get_current_profile tests
# =============================================================================


class TestGetCurrentProfile:
    """Test get_current_profile() resolution logic."""

    def test_default_profile_is_production(self):
        """Without any configuration the profile must be 'production'."""
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items() if k != "LOBSTER_PROFILE"}
            with patch.dict(os.environ, env, clear=True):
                profile = get_current_profile()
        assert profile == "production"

    def test_env_var_overrides_profile(self):
        """LOBSTER_PROFILE env var must override the default."""
        with patch.dict(os.environ, {"LOBSTER_PROFILE": "development"}):
            profile = get_current_profile()
        assert profile == "development"

    def test_env_var_arbitrary_profile_name(self):
        """LOBSTER_PROFILE can be any string value."""
        with patch.dict(os.environ, {"LOBSTER_PROFILE": "staging"}):
            profile = get_current_profile()
        assert profile == "staging"

    def test_returns_string(self):
        """Return value must always be a str."""
        result = get_current_profile()
        assert isinstance(result, str)


# =============================================================================
# ThinkingConfig tests
# =============================================================================


class TestThinkingConfigAgentDefaults:
    """Test ThinkingConfig as exported from agent_defaults."""

    def test_to_dict_when_disabled(self):
        """Disabled ThinkingConfig must produce an empty dict."""
        config = ThinkingConfig(enabled=False, budget_tokens=5000)
        assert config.to_dict() == {}

    def test_to_dict_when_enabled(self):
        """Enabled ThinkingConfig must produce the correct Bedrock structure."""
        config = ThinkingConfig(enabled=True, budget_tokens=3000)
        result = config.to_dict()
        assert result == {"thinking": {"type": "enabled", "budget_tokens": 3000}}

    def test_default_type_is_enabled_string(self):
        """Default type field must be the string 'enabled'."""
        config = ThinkingConfig()
        assert config.type == "enabled"

    def test_default_budget_tokens(self):
        """Default budget_tokens must be 2000."""
        config = ThinkingConfig()
        assert config.budget_tokens == 2000

    def test_default_enabled_is_false(self):
        """Default enabled must be False."""
        config = ThinkingConfig()
        assert config.enabled is False


# =============================================================================
# THINKING_PRESETS tests
# =============================================================================


class TestThinkingPresetsAgentDefaults:
    """Test THINKING_PRESETS as exported from agent_defaults."""

    def test_required_keys_present(self):
        """All required preset names must exist."""
        for key in ("disabled", "light", "standard", "extended", "deep"):
            assert key in THINKING_PRESETS, f"Missing preset: '{key}'"

    def test_budget_token_values(self):
        """Each preset must carry the correct budget_tokens value."""
        expected = {
            "light": 1000,
            "standard": 2000,
            "extended": 5000,
            "deep": 10000,
        }
        for name, expected_budget in expected.items():
            assert THINKING_PRESETS[name].budget_tokens == expected_budget, (
                f"Preset '{name}' has wrong budget_tokens"
            )

    def test_disabled_preset_is_not_enabled(self):
        """The 'disabled' preset must have enabled=False."""
        assert THINKING_PRESETS["disabled"].enabled is False

    def test_non_disabled_presets_are_enabled(self):
        """All presets except 'disabled' must have enabled=True."""
        for name, preset in THINKING_PRESETS.items():
            if name != "disabled":
                assert preset.enabled is True, (
                    f"Preset '{name}' should be enabled but is not"
                )

    def test_all_values_are_thinking_config_instances(self):
        """Every value in THINKING_PRESETS must be a ThinkingConfig."""
        for name, value in THINKING_PRESETS.items():
            assert isinstance(value, ThinkingConfig), (
                f"THINKING_PRESETS['{name}'] is not a ThinkingConfig"
            )
