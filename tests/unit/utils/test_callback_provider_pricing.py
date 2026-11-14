"""The token tracker prices a model by the ACTIVE provider's catalog row.

MODEL_PRICING is keyed by model id alone and the last registered provider wins, so two
providers that list the same id at different prices shadow each other: Bedrock lists
``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` at 3.30/16.50 (regional premium) and
the Omics-OS provider at 3.00/15.00; Azure lists ``gpt-4o`` at 5/15, OpenAI at 2.5/10.
A tracker given ``provider_for_pricing`` prefers the active provider's row and falls back
to the merged table; without it nothing changes (Cloud builds its tracker that way).
"""

import logging
from types import SimpleNamespace

import pytest

from lobster.config.providers.registry import ProviderRegistry
from lobster.config.settings import MODEL_PRICING
from lobster.utils.callbacks import TokenTrackingCallback

pytestmark = pytest.mark.no_auto_config

S45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
USAGE = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}


def end(tracker, model):
    tracker.on_llm_end(
        SimpleNamespace(
            generations=[[]], llm_output={"usage": USAGE, "model_name": model}
        )
    )
    return tracker.get_latest_cost()


def cost(model, provider=None, **kwargs):
    tracker = TokenTrackingCallback(
        "pricing",
        MODEL_PRICING,
        **({"provider_for_pricing": provider} if provider else {}),
        **kwargs,
    )
    return end(tracker, model)["latest_cost_usd"]


def test_the_registry_can_list_one_providers_rows():
    rows = ProviderRegistry.get_all_models_with_pricing("bedrock")
    assert rows[S45]["input_per_million"] == 3.3
    assert ProviderRegistry.get_all_models_with_pricing("omics-os")[S45][
        "input_per_million"
    ] == pytest.approx(3.0)
    assert ProviderRegistry.get_all_models_with_pricing("no-such-provider") == {}


def test_without_a_resolver_the_merged_table_decides_as_today():
    # Same construction as lobster-cloud api/routes/chat_stream.py: no new kwarg.
    assert cost(S45) == pytest.approx(18.0)  # omics-os row 3 + 15
    assert cost("gpt-4o") == pytest.approx(12.5)  # openai row 2.5 + 10


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("bedrock", S45, 19.8),  # 3.3 + 16.5
        ("omics-os", S45, 18.0),
        ("azure", "gpt-4o", 20.0),  # 5 + 15
        ("openai", "gpt-4o", 12.5),
    ],
)
def test_the_active_providers_row_prices_the_call(provider, model, expected):
    assert cost(model, lambda: provider) == pytest.approx(expected)


def test_a_provider_without_a_row_for_the_model_falls_back_to_the_merged_table():
    assert cost("gpt-4o", lambda: "bedrock") == pytest.approx(12.5)


@pytest.mark.parametrize("resolver", ["boom", None, "no-such-provider", 7])
def test_a_failing_or_unknown_resolver_never_breaks_the_run(resolver, caplog):
    def provider():
        if resolver == "boom":
            raise RuntimeError("config unreadable")
        return resolver

    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        assert cost(S45, provider) == pytest.approx(18.0)  # today's row


def test_the_resolver_is_read_at_call_time():
    active = ["bedrock"]
    tracker = TokenTrackingCallback(
        "switch", MODEL_PRICING, provider_for_pricing=lambda: active[0]
    )
    assert end(tracker, S45)["latest_cost_usd"] == pytest.approx(19.8)
    active[0] = "omics-os"  # runtime provider switch
    assert end(tracker, S45)["latest_cost_usd"] == pytest.approx(18.0)


def test_a_model_known_only_to_the_active_provider_is_priced_not_unpriced():
    # Not in the passed pricing_config at all, but present in the provider's catalog.
    tracker = TokenTrackingCallback("only", {}, provider_for_pricing=lambda: "bedrock")
    latest = end(tracker, S45)
    assert latest["latest_cost_usd"] == pytest.approx(19.8)
    assert latest["unpriced_invocations"] == 0


def test_the_client_builds_its_tracker_with_the_active_provider(tmp_path, monkeypatch):
    from lobster.core import client as client_module

    seen = {}

    class Spy(TokenTrackingCallback):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(client_module, "TokenTrackingCallback", Spy)
    try:
        client_module.AgentClient(workspace_path=tmp_path, provider_override="bedrock")
    except Exception:
        pass  # only the tracker construction matters here
    resolver = seen["provider_for_pricing"]
    assert resolver() == "bedrock"


def test_the_catalog_is_read_once_per_provider_and_unknown_names_are_not_cached(
    monkeypatch,
):
    calls = []
    real = ProviderRegistry.get_all_models_with_pricing

    def spy(provider=None):
        calls.append(provider)
        return real(provider)

    monkeypatch.setattr(ProviderRegistry, "get_all_models_with_pricing", spy)
    tracker = TokenTrackingCallback(
        "cache", MODEL_PRICING, provider_for_pricing=lambda: "bedrock"
    )
    end(tracker, S45)
    end(tracker, S45)
    assert calls == ["bedrock"]

    calls.clear()
    unknown = TokenTrackingCallback(
        "cache", MODEL_PRICING, provider_for_pricing=lambda: "late-provider"
    )
    end(unknown, S45)
    end(unknown, S45)
    assert calls == ["late-provider", "late-provider"]


def test_the_client_falls_back_to_the_configured_provider(tmp_path, monkeypatch):
    from lobster.config.llm_factory import LLMFactory
    from lobster.core import client as client_module

    seen = {}

    class Spy(TokenTrackingCallback):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(client_module, "TokenTrackingCallback", Spy)
    monkeypatch.setattr(
        LLMFactory, "get_current_provider", classmethod(lambda cls, path=None: "azure")
    )
    try:
        client_module.AgentClient(workspace_path=tmp_path)
    except Exception:
        pass  # only the tracker construction matters here
    assert seen["provider_for_pricing"]() == "azure"
