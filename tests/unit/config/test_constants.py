"""
Tests for lobster/config/constants.py.

Validates that the shared constant lists (VALID_PROVIDERS, PROVIDER_DISPLAY_NAMES, etc.)
contain the expected entries as the provider roster evolves.
"""


def test_openrouter_in_valid_providers():
    from lobster.config.constants import VALID_PROVIDERS

    assert "openrouter" in VALID_PROVIDERS


def test_openrouter_in_display_names():
    from lobster.config.constants import PROVIDER_DISPLAY_NAMES

    assert "openrouter" in PROVIDER_DISPLAY_NAMES
    assert "OpenRouter" in PROVIDER_DISPLAY_NAMES["openrouter"]


def test_valid_providers_contains_all_core_providers():
    """Guard against accidental removals from VALID_PROVIDERS."""
    from lobster.config.constants import VALID_PROVIDERS

    required = {"anthropic", "bedrock", "ollama", "gemini", "azure", "openai", "openrouter"}
    missing = required - set(VALID_PROVIDERS)
    assert not missing, f"Missing providers: {missing}"


def test_display_names_covers_all_valid_providers():
    """Every VALID_PROVIDER must have a display name."""
    from lobster.config.constants import PROVIDER_DISPLAY_NAMES, VALID_PROVIDERS

    for provider in VALID_PROVIDERS:
        assert provider in PROVIDER_DISPLAY_NAMES, (
            f"Provider '{provider}' is in VALID_PROVIDERS but missing from PROVIDER_DISPLAY_NAMES"
        )
