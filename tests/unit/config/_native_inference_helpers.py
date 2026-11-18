"""Provider-neutral fixtures; import real SDKs only at the native boundary.

Only an absent optional package is skipped. An installed package that cannot
import (or an unqualified Anthropic SDK) is a test failure, not missing coverage.
"""

import asyncio
import importlib
import json
import struct
import time
import tomllib
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4
from zlib import crc32

import pytest
from packaging.specifiers import SpecifierSet

from lobster.config.inference_context import (
    INFERENCE_BINDING_VERSION,
    RequestDescriptor,
)

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def require_optional(module, *, extra):
    if find_spec(module) is None:
        return pytest.importorskip(
            module, reason=f"{module} is absent; requires lobster-ai[{extra}]"
        )
    # Do not catch ImportError: installed but broken dependencies must fail.
    return importlib.import_module(module)


def require_bedrock():
    require_optional("boto3", extra="bedrock")
    require_optional("langchain_aws", extra="bedrock")


def require_anthropic():
    require_optional("anthropic", extra="anthropic")
    require_optional("langchain_anthropic", extra="anthropic")
    require_optional("httpx2", extra="anthropic")
    installed = version("anthropic")
    if installed not in SpecifierSet(">=1.7.0"):
        pytest.fail(
            f"Native middleware requires anthropic>=1.7.0; "
            f"installed {installed} (lobster-ai[anthropic])"
        )


def official_agents():
    root = Path(__file__).resolve().parents[3]
    agents = {}
    for path in sorted((root / "packages").glob("*/pyproject.toml")):
        project = tomllib.loads(path.read_text())["project"]
        for name in project.get("entry-points", {}).get("lobster.agents", {}):
            agents[name] = project["name"]
    assert len(agents) == 22
    return agents


def pytest_generate_tests(metafunc):
    if "name" in metafunc.fixturenames:
        metafunc.parametrize("name", sorted(official_agents()))


def require_agents(*names):
    agents = official_agents()
    for name in names:
        package = agents[name]
        try:
            distribution(package)
        except PackageNotFoundError:
            pytest.skip(
                f"{package} is absent; required official package for {name}",
                allow_module_level=True,
            )
        # Factory/config imports happen in the test and are deliberately not
        # caught: an installed but broken plugin must not disappear as a skip.


class Authorizer:
    def __init__(self):
        self.descriptors = []
        self.outcomes = []
        self.refusal = None
        self.finalization_refusal = None

    def before_request(self, descriptor):
        self.descriptors.append(descriptor)
        if self.refusal is not None:
            raise self.refusal
        return object()

    def after_request(self, permit, outcome):
        self.outcomes.append((permit, outcome))
        if self.finalization_refusal is not None:
            raise self.finalization_refusal


class AsyncAuthorizer(Authorizer):
    async def abefore_request(self, descriptor):
        return self.before_request(descriptor)

    async def aafter_request(self, permit, outcome):
        return self.after_request(permit, outcome)


class HookProbe:
    """Record native request-hook execution; never implements policy branches."""

    def __init__(self):
        self.hooks = []
        self.sent = []
        self.error = None
        self.status = 200
        self.closed = 0

    def before(self, request, **kwargs):
        self.hooks.append(request)
        if self.error is not None:
            raise self.error


class FakeRawResponse:
    def __init__(self, payload, probe):
        self.payload = payload
        self.probe = probe

    def stream(self, amt=None, decode_content=False):
        yield self.payload

    def close(self):
        self.probe.closed += 1


def encode_event(kind, payload):
    """Encode an AWS EventStream fixture, including the real wire CRCs."""
    headers = b""
    for name, value in (
        (":message-type", "event"),
        (":event-type", kind),
        (":content-type", "application/json"),
    ):
        name, value = name.encode(), value.encode()
        headers += (
            bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(value)) + value
        )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", crc32(prelude)) + headers + body
    return message + struct.pack(">I", crc32(message))


def bedrock_model(probe):
    require_bedrock()
    import boto3
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config

    from lobster.config.providers.bedrock_builder import build_bedrock_converse
    from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient

    runtime = boto3.client(  # nosec B106 # fake fixture credentials; native HTTP IO is replaced below.
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="fake-access-id",
        aws_secret_access_key="fake-access-secret",
        config=Config(retries={"total_max_attempts": 1}),
    )
    runtime.meta.events.register("before-send.bedrock-runtime.*", probe.before)

    def send(request):
        probe.sent.append(request)
        if probe.status != 200:
            payload = json.dumps({"message": "Synthetic failure"}).encode()
            headers = {
                "content-type": "application/json",
                "x-amzn-errortype": "InternalServerException",
            }
        elif request.url.endswith("converse-stream"):
            events = [
                ("messageStart", {"role": "assistant"}),
                (
                    "contentBlockDelta",
                    {"contentBlockIndex": 0, "delta": {"text": "ok"}},
                ),
                ("contentBlockStop", {"contentBlockIndex": 0}),
                ("messageStop", {"stopReason": "end_turn"}),
                (
                    "metadata",
                    {
                        "usage": {
                            "inputTokens": 4,
                            "outputTokens": 1,
                            "totalTokens": 5,
                        },
                        "metrics": {"latencyMs": 1},
                    },
                ),
            ]
            payload = b"".join(encode_event(kind, value) for kind, value in events)
            headers = {"content-type": "application/vnd.amazon.eventstream"}
        else:
            payload = json.dumps(
                {
                    "output": {
                        "message": {"role": "assistant", "content": [{"text": "ok"}]}
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 4, "outputTokens": 1, "totalTokens": 5},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
            headers = {"content-type": "application/json"}
        return AWSResponse(
            request.url, probe.status, headers, FakeRawResponse(payload, probe)
        )

    runtime._endpoint.http_session.send = send
    return build_bedrock_converse(
        model_id=BEDROCK_MODEL,
        max_tokens=137,
        client=runtime,
        bedrock_client=StubBedrockControlClient(),
        cache=False,
    )


def anthropic_model(probe):
    require_anthropic()
    import anthropic
    import httpx2

    from lobster.config.providers.anthropic_provider import AnthropicProvider

    message = {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }

    def send(request):
        probe.sent.append(request)
        if probe.status != 200:
            return httpx2.Response(
                probe.status,
                json={
                    "type": "error",
                    "error": {"type": "api_error", "message": "Synthetic failure"},
                },
            )
        if json.loads(request.content).get("stream"):
            start = {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 4, "output_tokens": 0},
            }
            events = [
                {"type": "message_start", "message": start},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            stream = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            )
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream.encode(),
            )
        return httpx2.Response(200, json=message)

    def before(request, next_handler):
        probe.before(request)
        return next_handler(request)

    async def abefore(request, next_handler):
        probe.before(request)
        return await next_handler(request)

    model = AnthropicProvider().create_chat_model(
        ANTHROPIC_MODEL,
        api_key="fake-anthropic-key",
        max_tokens=137,
        base_url="https://api.anthropic.com",
        max_retries=0,
        cache=False,
    )
    # The host supplies controlled native clients before exposing the model.
    # Do not put http_client in model_kwargs: it is not a Messages body field.
    model.__dict__["_client"] = anthropic.Client(
        api_key="fake-anthropic-key",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(send)),
        middleware=[before],
    )
    model.__dict__["_async_client"] = anthropic.AsyncClient(
        api_key="fake-anthropic-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(send)),
        middleware=[abefore],
    )
    return model


class NativeFactory:
    capability_version = INFERENCE_BINDING_VERSION

    def __init__(self):
        self.records = []
        self.models = {}

    def create(self, *, component, context):
        probe = HookProbe()
        authorizer = AsyncAuthorizer()
        views = []

        def validate(view):
            views.append(view)

        def describe(view):
            return RequestDescriptor(
                context.provider,
                context.model_id,
                component,
                context.binding_id,
                context.binding_version,
                str(uuid4()),
                str(uuid4()),
                context.max_output_tokens,
                16000,
                time.time() + 60,
            )

        kwargs = dict(
            context=context,
            component=component,
            authorizer=authorizer,
            validate_request=validate,
            describe_request=describe,
        )
        if context.provider == "bedrock":
            require_bedrock()
            from lobster.config.providers.controlled_bedrock_client import (
                ControlledBedrockClient,
                ControlledChatBedrockConverse,
            )
            from lobster.config.providers.gateway_bedrock_stub import (
                StubBedrockControlClient,
            )

            native = bedrock_model(probe)
            owner = ControlledBedrockClient(client=native.client, **kwargs)
            model = ControlledChatBedrockConverse(
                model_id=context.model_id,
                max_tokens=context.max_output_tokens,
                client=owner,
                bedrock_client=StubBedrockControlClient(),
                cache=False,
            )
        else:
            require_anthropic()
            from lobster.config.providers.controlled_anthropic_client import (
                ControlledAnthropicClient,
                ControlledChatAnthropic,
            )

            native = anthropic_model(probe)
            owner = ControlledAnthropicClient(
                **kwargs,
                api_key="fake-anthropic-key",
                base_url="https://api.anthropic.com",
                http_client=native._client._client,
                http_async_client=native._async_client._client,
            )
            model = ControlledChatAnthropic(
                model=context.model_id,
                max_tokens=context.max_output_tokens,
                api_key="fake-anthropic-key",
                base_url="https://api.anthropic.com",
                max_retries=0,
                cache=False,
            ).attach_request_owner(owner)
        self.records.append((component, context, owner, probe, authorizer, views))
        self.models[component] = model
        return model

    def close(self):
        for _, context, owner, *_ in self.records:
            owner.close()
            if context.provider == "anthropic":
                asyncio.run(owner.aclose())

    async def aclose(self):
        for _, context, owner, *_ in self.records:
            owner.close()
            if context.provider == "anthropic":
                await owner.aclose()
