"""Exercise output bounds at the real Converse serialization boundary."""

import socket
import sys
import types
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from lobster.config.providers.bedrock_builder import build_bedrock_converse
from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient
from tests.unit.config._native_inference_helpers import require_bedrock

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class FakeRuntime:
    """In-memory provider transport; model conversion remains real."""

    def __init__(self):
        self.requests = []

    def converse(self, **request):
        self.requests.append(deepcopy(request))
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }

    def converse_stream(self, **request):
        self.requests.append(deepcopy(request))
        return {
            "stream": iter(
                [
                    {"messageStart": {"role": "assistant"}},
                    {
                        "contentBlockDelta": {
                            "contentBlockIndex": 0,
                            "delta": {"text": "ok"},
                        }
                    },
                    {"contentBlockStop": {"contentBlockIndex": 0}},
                    {"messageStop": {"stopReason": "end_turn"}},
                    {
                        "metadata": {
                            "usage": {
                                "inputTokens": 4,
                                "outputTokens": 1,
                                "totalTokens": 5,
                            },
                            "metrics": {"latencyMs": 1},
                        }
                    },
                ]
            )
        }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def model_and_runtime():
    require_bedrock()
    runtime = FakeRuntime()
    model = build_bedrock_converse(
        model_id=MODEL_ID,
        temperature=0.5,
        max_tokens=137,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )
    return model, runtime


def assert_request(runtime):
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request["modelId"] == MODEL_ID
    assert request["inferenceConfig"]["maxTokens"] == 137
    assert request["inferenceConfig"]["temperature"] == 0.5


def test_invoke_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert model.invoke("hello").content == "ok"
    assert_request(runtime)


@pytest.mark.asyncio
async def test_ainvoke_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert (await model.ainvoke("hello")).content == "ok"
    assert_request(runtime)


def test_stream_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    assert "".join(chunk.text for chunk in model.stream("hello")) == "ok"
    assert_request(runtime)


@pytest.mark.asyncio
async def test_astream_serializes_max_tokens(model_and_runtime):
    model, runtime = model_and_runtime
    chunks = [chunk async for chunk in model.astream("hello")]
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert_request(runtime)


def test_tool_binding_preserves_output_bound(model_and_runtime):
    model, runtime = model_and_runtime
    bound = model.bind_tools(
        [
            {
                "name": "sample",
                "description": "A sample tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )
    bound.invoke("hello")
    assert_request(runtime)
    assert runtime.requests[0]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "sample"


PLACEHOLDER = "fake"  # stands in for a credential; never used against AWS


@pytest.fixture
def constructor(monkeypatch):
    """Stand-in ChatBedrockConverse, so these tests need no optional langchain-aws."""
    module = types.ModuleType("langchain_aws")
    module.ChatBedrockConverse = MagicMock(name="ChatBedrockConverse")
    monkeypatch.setitem(sys.modules, "langchain_aws", module)
    return module.ChatBedrockConverse


def _client_kwargs(mode):
    if mode == "gateway":
        return {"client": object(), "bedrock_client": object()}
    return {
        "region_name": "us-east-1",
        "aws_access_key_id": PLACEHOLDER,
        "aws_secret_access_key": PLACEHOLDER,
    }


@pytest.mark.parametrize("mode", ["gateway", "direct"])
@pytest.mark.parametrize("max_tokens", [1, 4096, 8192])
def test_forward_output_token_limit(mode, max_tokens, constructor):
    fields = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    kwargs = _client_kwargs(mode)
    result = build_bedrock_converse(
        model_id="test-model",
        temperature=0.5,
        max_tokens=max_tokens,
        additional_model_request_fields=fields,
        **kwargs,
    )
    assert result is constructor.return_value
    params = constructor.call_args.kwargs
    assert params["max_tokens"] == max_tokens
    assert params["model_id"] == "test-model"
    assert params["temperature"] == 0.5
    assert params["additional_model_request_fields"] == fields
    for key, value in kwargs.items():
        assert params[key] is value or params[key] == value


@pytest.mark.parametrize("mode", ["gateway", "direct"])
@pytest.mark.parametrize(
    "requested", [{}, {"max_tokens": None}], ids=["default", "explicit-none"]
)
def test_send_the_model_bound_unless_requested(mode, requested, constructor):
    """Without an explicit limit the documented model bound is sent, never nothing."""
    kwargs = _client_kwargs(mode)
    build_bedrock_converse(
        model_id="test-model", temperature=0.5, **requested, **kwargs
    )
    params = constructor.call_args.kwargs
    assert params["max_tokens"] == 8192
    assert params["model_id"] == "test-model"
    assert params["temperature"] == 0.5
    for key, value in kwargs.items():
        assert params[key] is value or params[key] == value
