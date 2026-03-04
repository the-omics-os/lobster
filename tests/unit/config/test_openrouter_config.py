"""Tests for OpenRouter config schema fields."""


def test_global_config_has_openrouter_model_field():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="anthropic/claude-sonnet-4-5")
    assert config.openrouter_default_model == "anthropic/claude-sonnet-4-5"


def test_global_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_default_model."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(openrouter_default_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_global_config_openrouter_model_defaults_none():
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig()
    assert config.openrouter_default_model is None


def test_workspace_config_has_openrouter_model_field():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="meta-llama/llama-3.3-70b-instruct")
    assert config.openrouter_model == "meta-llama/llama-3.3-70b-instruct"


def test_workspace_config_get_model_for_openrouter():
    """base_config.get_model_for_provider() auto-resolves openrouter_model."""
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(openrouter_model="openai/gpt-4o")
    assert config.get_model_for_provider("openrouter") == "openai/gpt-4o"


def test_workspace_config_accepts_openrouter_provider():
    from lobster.config.workspace_config import WorkspaceProviderConfig
    config = WorkspaceProviderConfig(global_provider="openrouter")
    assert config.global_provider == "openrouter"


def test_global_config_roundtrip_json():
    """OpenRouter config field survives JSON serialization."""
    from lobster.config.global_config import GlobalProviderConfig
    config = GlobalProviderConfig(
        default_provider="openrouter",
        openrouter_default_model="deepseek/deepseek-r1"
    )
    json_str = config.model_dump_json()
    reloaded = GlobalProviderConfig.model_validate_json(json_str)
    assert reloaded.default_provider == "openrouter"
    assert reloaded.openrouter_default_model == "deepseek/deepseek-r1"


def test_create_openrouter_config_valid():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("sk-or-test-key-123")
    assert result.success is True
    assert result.provider_type == "openrouter"
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test-key-123"
    assert result.env_vars["LOBSTER_LLM_PROVIDER"] == "openrouter"


def test_create_openrouter_config_strips_whitespace():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("  sk-or-test  ")
    assert result.env_vars["OPENROUTER_API_KEY"] == "sk-or-test"


def test_create_openrouter_config_empty_key_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("")
    assert result.success is False
    assert result.message is not None


def test_create_openrouter_config_whitespace_only_fails():
    from lobster.config.provider_setup import create_openrouter_config
    result = create_openrouter_config("   ")
    assert result.success is False
