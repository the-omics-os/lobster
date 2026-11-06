"""Native SDK injection seams with real model serialization and fake HTTP.

These are request-interception feasibility tests, not an authorization or
accounting implementation. A host must still supply its own bound factory.
"""

import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

import pytest

from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from lobster.core.exceptions import InferenceRefused
from tests.unit.config._native_inference_helpers import (
    ANTHROPIC_MODEL,
    BEDROCK_MODEL,
    HookProbe,
    NativeFactory,
    anthropic_model,
    bedrock_model,
    require_agents,
    require_bedrock,
)


@pytest.fixture(autouse=True)
def isolate_transport_and_configuration(monkeypatch):
    import dotenv

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    def no_dotenv(*args, **kwargs):
        return False

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(dotenv, "load_dotenv", no_dotenv)
    if settings_module := sys.modules.get("lobster.config.settings"):
        monkeypatch.setattr(settings_module, "load_dotenv", no_dotenv)


@pytest.fixture(params=["bedrock", "anthropic"])
def native_model(request):
    probe = HookProbe()
    model = (
        bedrock_model(probe) if request.param == "bedrock" else anthropic_model(probe)
    )
    return model, probe, request.param


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_native_hook_covers_every_execution_mode(native_model, operation):
    model, probe, provider = native_model
    if operation == "invoke":
        model.invoke("hello")
    elif operation == "ainvoke":
        await model.ainvoke("hello")
    elif operation == "stream":
        list(model.stream("hello"))
    else:
        _ = [chunk async for chunk in model.astream("hello")]
    assert len(probe.hooks) == len(probe.sent) == 1
    request = probe.sent[0]
    payload = json.loads(request.body if provider == "bedrock" else request.content)
    if provider == "bedrock":
        assert BEDROCK_MODEL in unquote(request.url)
        assert payload["inferenceConfig"]["maxTokens"] == 137
        if "stream" in operation:
            assert probe.closed == 1
    else:
        assert payload["model"] == ANTHROPIC_MODEL
        assert payload["max_tokens"] == 137
        # anthropic>=2 sends the key as a Bearer token; older SDKs as x-api-key.
        credential = request.headers.get("x-api-key") or request.headers.get(
            "authorization", ""
        ).removeprefix("Bearer ")
        assert credential == "fake-anthropic-key"


@pytest.mark.no_auto_config
@pytest.mark.parametrize("cap", [None, 1, 8192], ids=["unset", "one", "large"])
@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_bound_cap_and_unconfigured_default_reach_native_request(
    cap, operation, monkeypatch
):
    """Bound caps survive native construction; unbound defaults add no limit."""
    require_bedrock()
    from unittest.mock import Mock, patch

    from lobster.config.inference_context import InferenceContext
    from lobster.config.llm_factory import LLMFactory

    factory = None
    if cap is None:
        from lobster.config.providers.bedrock_provider import BedrockProvider

        monkeypatch.setattr(
            "lobster.config.providers.get_provider", lambda _: BedrockProvider()
        )
        monkeypatch.setenv("AWS_BEDROCK_ACCESS_KEY", "fake-access-key")
        monkeypatch.setenv("AWS_BEDROCK_SECRET_ACCESS_KEY", "fake-secret-key")
        probe = HookProbe()
        native = bedrock_model(probe)
        # Keep real factory/provider/builder serialization, replacing only IO.
        resolver = Mock()
        resolver.resolve_provider.return_value = ("bedrock", "test")
        with patch("lobster.core.config_resolver.ConfigResolver") as resolver_cls:
            resolver_cls.get_instance.return_value = resolver
            model = LLMFactory.create_llm(
                {
                    "model_id": BEDROCK_MODEL,
                    "additional_model_request_fields": {
                        "client": native.client,
                        "bedrock_client": StubBedrockControlClient(),
                        "cache": False,
                    },
                },
                agent_name="test_agent",
            )
        assert model.max_tokens is None
    else:
        context = InferenceContext(
            "bedrock", BEDROCK_MODEL, "synthetic-binding", "version-a", 200000, cap
        )
        factory = NativeFactory()
        # An ambient profile cap must not override the authoritative run binding.
        model = LLMFactory.create_llm(
            {"max_tokens": 4096},
            agent_name="test-component",
            inference_context=context,
            model_factory=factory,
        )
        _, actual_context, _, probe, authorizer, _ = factory.records[0]
        assert actual_context is context
        assert model.max_tokens == cap

    try:
        if operation == "invoke":
            model.invoke("hello")
        elif operation == "ainvoke":
            await model.ainvoke("hello")
        elif operation == "stream":
            list(model.stream("hello"))
        else:
            _ = [chunk async for chunk in model.astream("hello")]
        assert len(probe.hooks) == len(probe.sent) == 1
        payload = json.loads(probe.sent[0].body)
        if cap is None:
            assert "maxTokens" not in payload.get("inferenceConfig", {})
            assert "max_tokens" not in payload
        else:
            assert payload["inferenceConfig"]["maxTokens"] == cap
            assert len(authorizer.descriptors) == len(authorizer.outcomes) == 1
            assert authorizer.descriptors[0].max_output_tokens == cap
            assert authorizer.outcomes[0][1].completeness == "complete"
        if "stream" in operation:
            assert probe.closed == 1
    finally:
        if factory is not None:
            await factory.aclose()


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_native_hook_refuses_before_transport(native_model, operation):
    model, probe, _ = native_model
    refusal = InferenceRefused("budget")
    probe.error = refusal
    with pytest.raises(InferenceRefused) as caught:
        if operation == "invoke":
            model.invoke("hello")
        elif operation == "ainvoke":
            await model.ainvoke("hello")
        elif operation == "stream":
            list(model.stream("hello"))
        else:
            _ = [chunk async for chunk in model.astream("hello")]
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []


def test_native_sdk_does_not_retry_500(native_model):
    model, probe, _ = native_model
    probe.status = 500
    with pytest.raises(Exception):
        model.invoke("hello")
    assert len(probe.hooks) == len(probe.sent) == 1


def test_cached_helper_model_still_passes_native_hook_in_thread(native_model):
    require_agents("data_expert_agent")
    from lobster.agents.data_expert.assistant import DataExpertAssistant

    model, probe, _ = native_model
    helper = DataExpertAssistant()
    helper._llm = model
    helper.detect_modality({}, "GSE1")  # Exercise and retain the same model instance.
    refusal = InferenceRefused("policy")
    probe.error = refusal
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(InferenceRefused) as caught:
            executor.submit(helper.extract_strategy_config, {}, "GSE1").result()
    assert caught.value is refusal
    assert len(probe.hooks) == 2
    assert len(probe.sent) == 1


def test_legacy_bulk_entry_preserves_native_refusal(native_model, tmp_path):
    from types import SimpleNamespace

    from lobster.services.data_access.geo_fallback_service import GEOFallbackService

    model, probe, _ = native_model
    refusal = InferenceRefused("policy")
    probe.error = refusal
    service = SimpleNamespace(
        data_manager=SimpleNamespace(metadata_store={}),
        cache_dir=tmp_path,
        console=None,
        geo_downloader=None,
        geo_parser=None,
        fetch_metadata_only=lambda _: model.invoke("hello"),
    )
    with pytest.raises(InferenceRefused) as caught:
        GEOFallbackService(service).download_bulk_dataset("GSE1")
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []


def test_legacy_sample_entry_does_not_retry_native_refusal(
    native_model, monkeypatch, tmp_path
):
    require_agents("data_expert_agent")
    from types import SimpleNamespace

    import pandas as pd

    from lobster.agents.data_expert.assistant import DataExpertAssistant
    from lobster.services.data_access import geo_fallback_service
    from lobster.services.data_access.geo.concatenation import SampleConcatenator

    model, probe, _ = native_model
    refusal = InferenceRefused("budget")
    probe.error = refusal
    monkeypatch.setattr(DataExpertAssistant, "llm", property(lambda self: model))
    gsm = SimpleNamespace(metadata={}, table=pd.DataFrame([[1, 2]]))
    monkeypatch.setattr(
        geo_fallback_service, "pre_download_soft_file", lambda *a, **k: None
    )
    monkeypatch.setattr(geo_fallback_service.GEOparse, "get_GEO", lambda **k: gsm)
    service = SimpleNamespace(
        data_manager=SimpleNamespace(metadata_store={}, workspace_path=tmp_path),
        cache_dir=tmp_path,
        console=None,
        geo_downloader=None,
        geo_parser=None,
    )
    service._store_single_sample_as_modality = SampleConcatenator(
        service
    )._store_single_sample_as_modality
    with pytest.raises(InferenceRefused) as caught:
        geo_fallback_service.GEOFallbackService(service).download_single_cell_sample(
            "GSM1"
        )
    assert caught.value is refusal
    assert len(probe.hooks) == 1
    assert probe.sent == []
