"""The merged price table is the same whichever provider module was imported first.

Importing a provider module registers it; if that happens before the registry
loads the rest, the provider that wins a model id two providers list used to
change (Bedrock's "Claude Sonnet 4.5 (Bedrock)" instead of Omics-OS Cloud's,
Azure's GPT-4o price instead of OpenAI's).
"""

import pytest

from lobster.config.providers.registry import ProviderRegistry


@pytest.fixture
def fresh_registry():
    saved, initialized = (
        dict(ProviderRegistry._providers),
        ProviderRegistry._initialized,
    )
    ProviderRegistry.reset()
    yield
    ProviderRegistry._providers.clear()
    ProviderRegistry._providers.update(saved)
    ProviderRegistry._initialized = initialized


def _merged_after_importing_first(provider_cls):
    ProviderRegistry.register(provider_cls())  # what a module-level import does
    return ProviderRegistry.get_all_models_with_pricing()


def test_omics_os_imported_first_still_wins_its_shared_ids(fresh_registry):
    from lobster.config.providers.omics_os_provider import OmicsOSProvider

    table = _merged_after_importing_first(OmicsOSProvider)
    for model in (
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-5-5",
    ):
        assert table[model]["display_name"].endswith("(Omics-OS Cloud)")


def test_openai_imported_first_keeps_its_gpt_4o_price(fresh_registry):
    from lobster.config.providers.openai_provider import OpenAIProvider

    table = _merged_after_importing_first(OpenAIProvider)
    assert table["gpt-4o"]["input_per_million"] == 2.5


def test_a_provider_outside_the_built_in_list_comes_last(fresh_registry):
    from types import SimpleNamespace

    class PluginProvider:
        name = "plugin"
        KNOWN_MODELS = [
            SimpleNamespace(
                name="gpt-4o",
                input_cost_per_million=9.0,
                output_cost_per_million=9.0,
                display_name="GPT-4o (plugin)",
            )
        ]

    table = _merged_after_importing_first(PluginProvider)
    assert table["gpt-4o"]["display_name"] == "GPT-4o (plugin)"
