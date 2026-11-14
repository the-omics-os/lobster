"""A provider that returns no usage must not read as a complete, free run.

Found in a real-user MCP run on Ollama Cloud: a 28 s multi-agent run reported
total_tokens 0 and $0.0 with cost_complete true. Two causes: langchain-openai leaves
streamed usage off when a custom base_url is set (so OpenAI-compatible hosts send none),
and a call without usage left cost_complete untouched.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from lobster.utils.callbacks import TokenTrackingCallback

pytestmark = pytest.mark.no_auto_config

PRICES = {"model": {"input_per_million": 3, "output_per_million": 15}}


@pytest.fixture
def tracker():
    return TokenTrackingCallback("usage", PRICES)


def with_usage(input_tokens=1000, output_tokens=100):
    return SimpleNamespace(
        generations=[[]],
        llm_output={
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "model_name": "model",
        },
    )


NO_USAGE = SimpleNamespace(generations=[[]], llm_output={})


def test_a_call_without_usage_makes_the_cost_incomplete(tracker):
    tracker.on_llm_end(NO_USAGE)
    latest = tracker.get_latest_cost()
    assert latest["cost_complete"] is False
    assert latest["usage_complete"] is False
    assert latest["missing_usage_invocations"] == 1
    assert latest["total_tokens"] == 0
    assert tracker.get_usage_summary()["cost_complete"] is False


def test_a_failed_call_without_usage_makes_the_cost_incomplete(tracker):
    tracker.on_llm_error(RuntimeError("mock failure"))
    assert tracker.get_latest_cost()["cost_complete"] is False


def test_one_missing_call_among_priced_ones_makes_the_total_a_lower_bound(tracker):
    tracker.on_llm_end(with_usage())
    tracker.on_llm_end(NO_USAGE)
    latest = tracker.get_latest_cost()
    assert latest["session_total_usd"] == 0.0045
    assert latest["cost_complete"] is False
    assert latest["missing_usage_invocations"] == 1


def test_a_run_with_usage_on_every_call_stays_complete(tracker):
    tracker.on_llm_end(with_usage())
    latest = tracker.get_latest_cost()
    assert latest["cost_complete"] is True
    assert latest["usage_complete"] is True
    assert latest["missing_usage_invocations"] == 0


def test_the_session_line_does_not_call_a_missing_usage_run_local(tracker):
    tracker.on_llm_end(NO_USAGE)
    line = tracker.get_minimal_summary()
    assert "local" not in line
    assert "not reported" in line


def test_reset_clears_the_missing_count(tracker):
    tracker.on_llm_end(NO_USAGE)
    tracker.reset()
    assert tracker.get_latest_cost()["cost_complete"] is True


# ---- OpenAI-compatible hosts: streamed usage is requested ---------------------
HOSTS = [
    ("ollama-cloud", "OLLAMA_API_KEY", "glm-5.2"),
    ("nebius", "NEBIUS_API_KEY", None),
    ("openrouter", "OPENROUTER_API_KEY", "anthropic/claude-sonnet-4-5"),
]


def host_model(name, key, model, monkeypatch, **kwargs):
    import lobster.config.providers  # noqa: F401 -- registers providers
    from lobster.config.providers.registry import ProviderRegistry

    monkeypatch.setenv(key, "test-key")
    provider = ProviderRegistry.get(name)
    return provider.create_chat_model(model or provider.get_default_model(), **kwargs)


@pytest.mark.parametrize("name,key,model", HOSTS)
def test_openai_compatible_hosts_request_streamed_usage(name, key, model, monkeypatch):
    llm = host_model(name, key, model, monkeypatch)
    assert llm._should_stream_usage() is True


def sse(chunks):
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def compatible_host(seen):
    """An OpenAI-compatible host that, like Ollama, sends usage only when asked."""

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "hi"}}
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks.append(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "model": body["model"],
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                    },
                }
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse(chunks)
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_streamed_ollama_cloud_call_counts_the_usage_the_host_returns(monkeypatch):
    seen = []
    llm = host_model(
        "ollama-cloud",
        "OLLAMA_API_KEY",
        "glm-5.2",
        monkeypatch,
        http_client=compatible_host(seen),
    )
    tracker = TokenTrackingCallback("usage", {})
    list(llm.stream("hello", config={"callbacks": [tracker]}))
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert tracker.total_tokens == 150
    assert tracker.get_latest_cost()["usage_complete"] is True


def test_the_openai_provider_asks_for_streamed_usage_behind_a_custom_base_url(
    monkeypatch,
):
    """OPENAI_BASE_URL points the client at a proxy, which switches default usage off."""
    from lobster.config.providers.openai_provider import OpenAIProvider

    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    llm = OpenAIProvider().create_chat_model("any-model", api_key="test-key")
    assert llm._should_stream_usage() is True


def strict_host(seen):
    """A compatible host that rejects the stream_options field with a 400."""

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "stream_options" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "Unknown field: stream_options"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                [
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "hi"},
                            }
                        ],
                    },
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ]
            ),
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("name,key,model", HOSTS)
def test_a_host_that_rejects_stream_options_gets_one_retry_without_it(
    name, key, model, monkeypatch, caplog
):
    """A run that worked before the usage request must still complete (once)."""
    seen = []
    llm = host_model(name, key, model, monkeypatch, http_client=strict_host(seen))
    tracker = TokenTrackingCallback("usage", {})
    with caplog.at_level("WARNING"):
        chunks = list(llm.stream("hello", config={"callbacks": [tracker]}))
    assert "".join(c.content for c in chunks) == "hi"
    assert len(seen) == 2  # exactly one retry
    assert "stream_options" in seen[0] and "stream_options" not in seen[1]
    # One warning carries the hint: the config switch and what it costs.
    warnings = [
        r.getMessage() for r in caplog.records if "stream_usage" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "false" in warnings[0].lower() and "not counted" in warnings[0]
    # The run says its usage is unavailable instead of "complete, free".
    latest = tracker.get_latest_cost()
    assert latest["cost_complete"] is False and latest["usage_complete"] is False
    assert latest["missing_usage_invocations"] == 1
    assert "not reported" in tracker.get_minimal_summary()
    # The model remembers: the next call sends no stream_options and does not retry.
    seen.clear()
    list(llm.stream("again"))
    assert len(seen) == 1 and "stream_options" not in seen[0]


@pytest.mark.parametrize("name,key,model", HOSTS)
def test_a_host_that_accepts_stream_options_is_asked_once(
    name, key, model, monkeypatch
):
    seen = []
    llm = host_model(name, key, model, monkeypatch, http_client=compatible_host(seen))
    list(llm.stream("hello"))
    assert len(seen) == 1 and seen[0]["stream_options"] == {"include_usage": True}


def test_the_explicit_override_sends_no_stream_options_and_reports_unavailable(
    monkeypatch,
):
    seen = []
    llm = host_model(
        "ollama-cloud",
        "OLLAMA_API_KEY",
        "glm-5.2",
        monkeypatch,
        http_client=strict_host(seen),
        stream_usage=False,
    )
    tracker = TokenTrackingCallback("usage", {})
    list(llm.stream("hello", config={"callbacks": [tracker]}))
    assert len(seen) == 1 and "stream_options" not in seen[0]
    latest = tracker.get_latest_cost()
    assert latest["cost_complete"] is False and latest["usage_complete"] is False


@pytest.mark.asyncio
async def test_the_async_stream_retries_once_the_same_way(monkeypatch):
    seen = []

    async def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "stream_options" in body:
            return httpx.Response(
                400, json={"error": {"message": "Unknown field: stream_options"}}
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                [
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "hi"},
                            }
                        ],
                    },
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ]
            ),
        )

    llm = host_model(
        "ollama-cloud",
        "OLLAMA_API_KEY",
        "glm-5.2",
        monkeypatch,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tracker = TokenTrackingCallback("usage", {})
    chunks = [c async for c in llm.astream("hello", config={"callbacks": [tracker]})]
    assert "".join(c.content for c in chunks) == "hi"
    assert len(seen) == 2 and "stream_options" not in seen[1]
    assert tracker.get_latest_cost()["usage_complete"] is False


def test_a_second_rejection_is_not_retried_again_and_carries_the_hint(monkeypatch):
    import openai

    seen = []
    llm = host_model(
        "ollama-cloud",
        "OLLAMA_API_KEY",
        "glm-5.2",
        monkeypatch,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: seen.append(request)
                or httpx.Response(
                    400, json={"error": {"message": "Unknown field: stream_options"}}
                )
            )
        ),
    )
    with pytest.raises(openai.BadRequestError, match="stream_usage") as refused:
        list(llm.stream("hello"))
    assert "Unknown field: stream_options" in str(refused.value)  # host text kept
    assert len(seen) == 2  # one retry, never more


def test_a_rejection_after_chunks_is_raised_not_retried():
    import openai

    from lobster.config.providers.stream_usage_hint import with_stream_usage_hint

    calls = []

    class Stub:
        stream_usage = True

        def _stream(self, *args, **kwargs):
            calls.append(1)
            yield "chunk"
            raise openai.BadRequestError(
                "Unknown field: stream_options",
                response=httpx.Response(400, request=httpx.Request("POST", "http://h")),
                body=None,
            )

        async def _astream(self, *args, **kwargs):
            yield "chunk"

    stub = with_stream_usage_hint(Stub())
    got = []
    with pytest.raises(openai.BadRequestError, match="stream_usage"):
        for chunk in stub._stream():
            got.append(chunk)
    assert got == ["chunk"] and len(calls) == 1
    assert stub.stream_usage is True  # untouched: nothing was learned


def test_an_unrelated_400_is_left_untouched(monkeypatch):
    import openai

    llm = host_model(
        "ollama-cloud",
        "OLLAMA_API_KEY",
        "glm-5.2",
        monkeypatch,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    400, json={"error": {"message": "context too long"}}
                )
            )
        ),
    )
    with pytest.raises(openai.BadRequestError) as refused:
        list(llm.stream("hello"))
    assert "stream_usage" not in str(refused.value)


@pytest.fixture()
def clean_singletons():
    """The factory goes through ConfigResolver; leave no singleton behind."""
    from lobster.config.providers.registry import ProviderRegistry
    from lobster.core.config_resolver import ConfigResolver

    ConfigResolver.reset_instance()
    ProviderRegistry.reset()
    try:
        yield
    finally:
        ConfigResolver.reset_instance()
        ProviderRegistry.reset()


def test_model_config_turns_streamed_usage_off_through_the_llm_factory(
    monkeypatch, clean_singletons
):
    """additional_model_request_fields reaches the provider as kwargs (llm_factory)."""
    from lobster.config.llm_factory import LLMFactory

    seen = []
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    llm = LLMFactory.create_llm(
        {
            "additional_model_request_fields": {
                "stream_usage": False,
                "http_client": strict_host(seen),
            }
        },
        agent_name="supervisor",
        provider_override="ollama-cloud",
        model_override="glm-5.2",
    )
    assert llm.stream_usage is False
    list(llm.stream("hello"))
    assert "stream_options" not in seen[0]
    # Without the override the default asks for usage.
    default = LLMFactory.create_llm(
        {},
        agent_name="supervisor",
        provider_override="ollama-cloud",
        model_override="glm-5.2",
    )
    assert default.stream_usage is True


def _rejection():
    import openai

    return openai.BadRequestError(
        "Unknown field: stream_options",
        response=httpx.Response(400, request=httpx.Request("POST", "http://h")),
        body=None,
    )


@pytest.mark.asyncio
async def test_an_async_rejection_after_chunks_is_raised_not_retried():
    import openai

    from lobster.config.providers.stream_usage_hint import with_stream_usage_hint

    calls = []

    class Stub:
        stream_usage = True

        def _stream(self, *args, **kwargs):
            yield "x"

        async def _astream(self, *args, **kwargs):
            calls.append(1)
            yield "chunk"
            raise _rejection()

    stub = with_stream_usage_hint(Stub())
    got = []
    with pytest.raises(openai.BadRequestError, match="stream_usage"):
        async for chunk in stub._astream():
            got.append(chunk)
    assert got == ["chunk"] and len(calls) == 1 and stub.stream_usage is True


@pytest.mark.asyncio
async def test_an_async_second_rejection_is_not_retried_again():
    import openai

    from lobster.config.providers.stream_usage_hint import with_stream_usage_hint

    calls = []

    class Stub:
        stream_usage = True

        def _stream(self, *args, **kwargs):
            yield "x"

        async def _astream(self, *args, **kwargs):
            calls.append(1)
            raise _rejection()
            yield  # pragma: no cover -- makes this an async generator

    stub = with_stream_usage_hint(Stub())
    with pytest.raises(openai.BadRequestError, match="stream_usage"):
        async for _ in stub._astream():
            pass
    assert len(calls) == 2 and stub.stream_usage is False
