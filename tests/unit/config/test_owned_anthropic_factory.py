"""Real owned Messages clients against synthetic native endpoints, sockets blocked."""

import asyncio
import hmac
import json
import pickle  # nosec B403 # dumps only tests serialization refusal; no untrusted loads.
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
import pytest_asyncio

from lobster.config.inference_context import InferenceContext, RequestDescriptor
from lobster.config.providers.request_auth import ExpiringRequestAuth, RequestAuth
from lobster.core.exceptions import InferenceRefused
from tests.unit.config._native_inference_helpers import require_anthropic

pytestmark = pytest.mark.no_auto_config
MODEL = "claude-sonnet-4-5-20250929"


class LocalHooks:
    def __init__(self):
        self.starts, self.ends = [], []

    def before_request(self, descriptor):
        self.starts.append(descriptor)
        return descriptor.attempt_id

    async def abefore_request(self, descriptor):
        return self.before_request(descriptor)

    def after_request(self, marker, outcome):
        self.ends.append((marker, outcome))

    async def aafter_request(self, marker, outcome):
        self.after_request(marker, outcome)


class Endpoint:
    def __init__(self):
        self.secrets = [uuid4().hex, uuid4().hex]
        self.expected = 0
        self.calls = []
        self.category = None
        self.status = 403
        self.streaming_error = False
        self.force_error = False

    def send(self, request):
        import httpx2

        # Keep only nonsecret evidence; do not retain request/header objects.
        assert hmac.compare_digest(
            request.headers["x-api-key"], self.secrets[self.expected]
        )
        self.calls.append((str(request.url), request.headers["x-request-nonce"]))
        payload = json.loads(request.content)
        message = {
            "id": "msg_synthetic",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 1},
        }
        error = {
            "type": "error",
            "error": {"type": "api_error", "message": self.category},
        }
        if (self.category or self.force_error) and not self.streaming_error:
            return httpx2.Response(self.status, json=error)
        if payload.get("stream"):
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
            ]
            if self.category:
                events.append(error)
            else:
                events.extend(
                    [
                        {"type": "content_block_stop", "index": 0},
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": 1},
                        },
                        {"type": "message_stop"},
                    ]
                )
            data = "".join(
                f"event: {item['type']}\ndata: {json.dumps(item)}\n\n"
                for item in events
            )
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=data.encode(),
            )
        return httpx2.Response(200, json=message)


@pytest_asyncio.fixture
async def bound(monkeypatch):
    require_anthropic()
    import httpx2

    from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory

    endpoint, hooks = Endpoint(), LocalHooks()
    clock = [time.time()]
    renewals = []
    context = InferenceContext(
        "anthropic", MODEL, "binding-a", "version-a", 200000, 137
    )

    def refresh(current):
        renewals.append(current.generation)
        endpoint.expected = 1
        return RequestAuth(
            endpoint.secrets[1], expires_at=clock[0] + 300.0, generation=2
        )

    async def arefresh(current):
        return refresh(current)

    auth = ExpiringRequestAuth(
        initial=RequestAuth(endpoint.secrets[0], expires_at=clock[0] + 300.0),
        refresh=refresh,
        arefresh=arefresh,
        clock=lambda: clock[0],
    )

    def describe(component, view):
        return RequestDescriptor(
            "anthropic",
            MODEL,
            component,
            "binding-a",
            "version-a",
            str(uuid4()),
            str(uuid4()),
            137,
            199863,
            time.time() + 60,
        )

    def decode(view):
        assert view.endpoint == "https://binding.example/native/v1/messages"
        body = view.parameters["body"]
        category = body.get("error", {}).get("message")
        if category in {
            "context",
            "auth",
            "policy",
            "budget",
            "provider",
            "accounting",
        }:
            return InferenceRefused(category)
        return None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "poisoned-discovery")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "poisoned-oauth")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("HTTPS_PROXY", "http://wrong.example:3128")
    factory = OwnedAnthropicFactory(
        context=context,
        base_url="https://binding.example/native",
        request_auth=auth,
        authorizer_factory=lambda component, context: hooks,
        validate_request=lambda component, view: None,
        describe_request=describe,
        error_decoder=decode,
        request_headers=lambda descriptor: {"X-Request-Nonce": uuid4().hex},
        transport_factory=lambda component: (
            httpx2.MockTransport(endpoint.send),
            httpx2.MockTransport(endpoint.send),
        ),
    )
    yield factory, context, endpoint, hooks, auth, clock, renewals
    try:
        await factory.aclose()
    except InferenceRefused as error:
        if error.category != getattr(factory, "_expected_fixture_close_refusal", None):
            raise


async def invoke(model, mode):
    if mode == "invoke":
        return model.invoke("synthetic prompt")
    if mode == "ainvoke":
        return await model.ainvoke("synthetic prompt")
    if mode == "stream":
        return list(model.stream("synthetic prompt"))
    return [chunk async for chunk in model.astream("synthetic prompt")]


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_owned_factory_actual_sdk_refreshes_at_request_boundary(bound, mode):
    factory, context, endpoint, hooks, auth, clock, renewals = bound
    model = factory.create(component="supervisor", context=context)
    await invoke(model, mode)
    clock[0] += 181
    await invoke(model, mode)
    assert len(endpoint.calls) == len(hooks.starts) == len(hooks.ends) == 2
    assert renewals == [1]
    assert len({nonce for _, nonce in endpoint.calls}) == 2
    assert all(
        url == "https://binding.example/native/v1/messages" for url, _ in endpoint.calls
    )
    assert all(result.outcome == "completed" for _, result in hooks.ends)
    for token in endpoint.secrets:
        assert token not in repr(model)
        assert token not in repr(factory)
        assert token not in repr(auth)
    assert model.cache is False
    assert model.max_retries == 0


@pytest.mark.parametrize("status", [400, 401, 403, 409, 422, 429, 500, 502, 503])
@pytest.mark.parametrize(
    "category", ["context", "auth", "policy", "budget", "provider", "accounting"]
)
@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_host_decoder_preserves_every_safe_native_error_category(
    bound, mode, category, status, caplog, capsys
):
    factory, context, endpoint, hooks, _, _, _ = bound
    endpoint.category, endpoint.status = category, status
    model = factory.create(component="specialist", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await invoke(model, mode)
    assert caught.value.category == category
    assert len(endpoint.calls) == len(hooks.ends) == 1
    captured = capsys.readouterr()
    for token in endpoint.secrets:
        assert token not in caplog.text + captured.out + captured.err + repr(
            caught.value
        )


@pytest.mark.parametrize(
    "category", ["context", "auth", "policy", "budget", "provider", "accounting"]
)
@pytest.mark.parametrize("mode", ["stream", "astream"])
@pytest.mark.asyncio
async def test_native_sse_error_after_content_preserves_category(bound, category, mode):
    factory, context, endpoint, hooks, _, _, _ = bound
    endpoint.category, endpoint.streaming_error = category, True
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await invoke(model, mode)
    assert caught.value.category == category
    assert len(endpoint.calls) == len(hooks.ends) == 1
    assert hooks.ends[0][1].outcome == "failed"


@pytest.mark.asyncio
async def test_shared_factory_models_execute_on_real_raw_threads(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    models = [
        factory.create(component=name, context=context)
        for name in ("supervisor", "helper")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.invoke, "synthetic") for model in models]
        assert all(future.result().content == "ok" for future in futures)
    assert len(endpoint.calls) == len(hooks.starts) == len(hooks.ends) == 2
    assert {descriptor.component for descriptor in hooks.starts} == {
        "supervisor",
        "helper",
    }
    assert len({nonce for _, nonce in endpoint.calls}) == 2


@pytest.mark.asyncio
async def test_disposed_retained_models_never_send_or_renew(bound):
    factory, context, endpoint, hooks, auth, _, renewals = bound
    model = factory.create(component="helper", context=context)
    await model.ainvoke("before")
    await factory.aclose()
    with pytest.raises(InferenceRefused):
        await model.ainvoke("after")
    with pytest.raises(InferenceRefused):
        factory.create(component="helper", context=context)
    assert len(endpoint.calls) == 1
    assert not renewals


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_actual_native_raw_and_beta_paths(bound, beta, asynchronous):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="raw-helper", context=context)
    owner = model._request_owner
    client = owner.async_view if asynchronous else owner.sync_view
    if beta:
        client = client.beta
    raw = client.messages.with_raw_response.create(
        model=MODEL,
        max_tokens=137,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    if asynchronous:
        raw = await raw
        message = await raw.parse()
    else:
        message = raw.parse()
    assert message.content[0].text == "ok"
    assert len(endpoint.calls) == len(hooks.ends) == 1
    route = "https://binding.example/native/v1/messages"
    # anthropic<2 posts beta requests to ?beta=true; anthropic>=2 to the plain route.
    assert endpoint.calls[0][0] in ({route, route + "?beta=true"} if beta else {route})


@pytest.mark.asyncio
async def test_terminal_refusal_latches_all_retained_components_before_send(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    first = factory.create(component="supervisor", context=context)
    retained = factory.create(component="helper", context=context)
    endpoint.category = "budget"
    with pytest.raises(InferenceRefused) as initial:
        await first.ainvoke("refuse")
    with pytest.raises(InferenceRefused) as later:
        await retained.ainvoke("must not send")
    assert later.value.category == initial.value.category
    assert later.value.retryable == initial.value.retryable
    assert later.value is not initial.value
    assert factory._refusal == ("budget", False)
    assert len(endpoint.calls) == len(hooks.starts) == 1


@pytest.mark.parametrize("malformed", ["unexpected-marker", "", None])
@pytest.mark.asyncio
async def test_unknown_error_is_provider_not_message_authority(bound, malformed):
    factory, context, endpoint, _, _, _, _ = bound
    endpoint.category, endpoint.force_error = malformed, True
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic")
    assert caught.value.category == "provider"


@pytest.mark.asyncio
async def test_factory_close_waits_for_async_finalizer_under_repeated_cancellation(
    bound,
):
    factory, context, endpoint, hooks, _, _, _ = bound
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(marker, outcome):
        entered.set()
        await release.wait()
        hooks.after_request(marker, outcome)

    hooks.aafter_request = delayed
    model = factory.create(component="helper", context=context)
    request = asyncio.create_task(model.ainvoke("synthetic"))
    await entered.wait()
    close = asyncio.create_task(factory.aclose())
    await asyncio.sleep(0)
    close.cancel()
    await asyncio.sleep(0)
    close.cancel()
    assert not close.done()
    with pytest.raises(InferenceRefused):
        factory.create(component="another", context=context)
    release.set()
    await request
    with pytest.raises(asyncio.CancelledError):
        await close
    assert len(endpoint.calls) == len(hooks.ends) == 1


@pytest.mark.asyncio
async def test_factory_disposes_retained_native_stream_once(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="helper", context=context)
    raw = await model._request_owner.async_view.messages.create(
        model=MODEL,
        max_tokens=137,
        stream=True,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    stream = await raw.parse()
    await anext(stream)
    await factory.aclose()
    assert len(endpoint.calls) == len(hooks.ends) == 1
    assert hooks.ends[0][1].outcome == "cancelled"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_raw_reparse_never_bypasses_failed_finalizer(bound, asynchronous):
    factory, context, endpoint, hooks, _, _, _ = bound
    attempts = []

    def fail(marker, outcome):
        attempts.append(marker)
        raise InferenceRefused("accounting")

    async def afail(marker, outcome):
        fail(marker, outcome)

    hooks.after_request, hooks.aafter_request = fail, afail
    model = factory.create(component="raw-helper", context=context)
    client = (
        model._request_owner.async_view
        if asynchronous
        else model._request_owner.sync_view
    )
    raw = client.messages.create(
        model=MODEL, max_tokens=137, messages=[{"role": "user", "content": "synthetic"}]
    )
    if asynchronous:
        raw = await raw
    for _ in range(2):
        with pytest.raises(InferenceRefused) as caught:
            if asynchronous:
                await raw.parse()
            else:
                raw.parse()
        assert caught.value.category == "accounting"
    assert len(attempts) == len(endpoint.calls) == 1


@pytest.mark.asyncio
async def test_final_issue_gate_blocks_request_racing_terminal_refusal(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause(descriptor):
        marker = hooks.before_request(descriptor)
        if descriptor.component == "helper":
            entered.set()
            await release.wait()
        return marker

    hooks.abefore_request = pause
    helper = factory.create(component="helper", context=context)
    supervisor = factory.create(component="supervisor", context=context)
    pending = asyncio.create_task(helper.ainvoke("paused"))
    await entered.wait()
    endpoint.category = "budget"
    with pytest.raises(InferenceRefused):
        await supervisor.ainvoke("refuse")
    release.set()
    with pytest.raises(InferenceRefused) as caught:
        await pending
    assert caught.value.category == "budget"
    assert len(endpoint.calls) == 1
    assert sum(outcome.request_issued for _, outcome in hooks.ends) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_close_before_native_response_assignment_cannot_abandon_obligation(
    bound, asynchronous
):
    from threading import Event

    import httpx2

    factory, context, endpoint, hooks, _, _, _ = bound
    entered, release = Event(), Event()

    def blocked(request):
        entered.set()
        assert release.wait(5)
        return endpoint.send(request)

    async def ablocked(request):
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return endpoint.send(request)

    factory._transports = lambda component: (
        httpx2.MockTransport(blocked),
        httpx2.MockTransport(ablocked),
    )
    model = factory.create(component="raw-helper", context=context)
    payload = dict(
        model=MODEL, max_tokens=137, messages=[{"role": "user", "content": "synthetic"}]
    )
    owner = model._request_owner
    if asynchronous:
        pending = asyncio.create_task(owner.async_view.messages.create(**payload))
    else:
        pending = asyncio.create_task(
            asyncio.to_thread(owner.sync_view.messages.create, **payload)
        )
    assert await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(factory.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(InferenceRefused):
        await pending
    await asyncio.wait_for(closing, 5)
    assert len(endpoint.calls) == len(hooks.ends) == 1
    assert hooks.ends[0][1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_native_error_exception_chain_does_not_retain_sdk_error_body(bound):
    factory, context, endpoint, _, _, _, _ = bound
    endpoint.category = endpoint.secrets[0]
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic")
    assert caught.value.category == "provider"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert endpoint.secrets[0] not in str(caught.value)
    assert not isinstance(factory._refusal, BaseException)


@pytest.mark.asyncio
async def test_multiple_mixed_streams_cleanup_even_first_finalizer_refuses(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    calls = []

    def finalize(marker, outcome):
        calls.append(marker)
        if len(calls) == 1:
            raise InferenceRefused("accounting")
        hooks.ends.append((marker, outcome))

    async def afinalize(marker, outcome):
        finalize(marker, outcome)

    hooks.after_request, hooks.aafter_request = finalize, afinalize
    model = factory.create(component="helper", context=context)
    owner = model._request_owner
    payload = dict(
        model=MODEL,
        max_tokens=137,
        stream=True,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    first = owner.sync_view.messages.create(**payload)
    second = await owner.async_view.messages.create(**payload)
    next(first.parse())
    await anext(await second.parse())
    with pytest.raises(InferenceRefused) as caught:
        await factory.aclose()
    assert caught.value.category == "accounting"
    assert len(calls) == 2
    assert len(endpoint.calls) == 2
    assert not owner._pending
    assert owner._sync.is_closed()
    assert owner._async.is_closed()
    factory._expected_fixture_close_refusal = "accounting"


@pytest.mark.asyncio
async def test_concurrent_finalizers_wait_and_repeat_terminal_result(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def finalize(marker, outcome):
        calls.append(marker)
        entered.set()
        await release.wait()
        raise InferenceRefused("accounting")

    hooks.aafter_request = finalize
    model = factory.create(component="helper", context=context)
    raw = await model._request_owner.async_view.messages.create(
        model=MODEL,
        max_tokens=137,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    first = asyncio.create_task(raw.parse())
    await entered.wait()
    second = asyncio.create_task(raw.close())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    for task in (first, second):
        with pytest.raises(InferenceRefused) as caught:
            await task
        assert caught.value.category == "accounting"
    assert len(calls) == len(endpoint.calls) == 1


@pytest.mark.asyncio
async def test_cancellation_during_native_raw_parse_is_cancelled_outcome(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="raw-helper", context=context)
    raw = await model._request_owner.async_view.messages.create(
        model=MODEL,
        max_tokens=137,
        messages=[{"role": "user", "content": "synthetic"}],
    )
    entered = asyncio.Event()

    async def delayed(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    raw.attempt.raw.parse = delayed
    task = asyncio.create_task(raw.parse())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(endpoint.calls) == len(hooks.ends) == 1
    assert hooks.ends[0][1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_replay_capacity_refuses_without_erasing_evidence(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="helper", context=context)
    owner = model._request_owner
    owner._attempt_ids = {str(uuid4()) for _ in range(8192)}
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic")
    assert caught.value.category == "context"
    assert len(owner._attempt_ids) == 8192
    assert not endpoint.calls
    assert not hooks.starts


@pytest.mark.asyncio
async def test_host_native_validation_is_mandatory_before_authority_or_send(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    calls = []

    def deny(component, view):
        calls.append(component)
        raise InferenceRefused("policy")

    factory._validate = deny
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic")
    assert caught.value.category == "policy"
    assert calls == ["helper"]
    assert not hooks.starts
    assert not endpoint.calls


@pytest.mark.asyncio
async def test_caller_cannot_override_explicit_auth_channel(bound):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic", extra_headers={"x-api-key": "unapproved"})
    assert caught.value.category == "context"
    assert not endpoint.calls
    assert not hooks.starts


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_closed_raw_never_reparses_or_releases_buffered_output(
    bound, asynchronous
):
    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="raw-helper", context=context)
    client = (
        model._request_owner.async_view
        if asynchronous
        else model._request_owner.sync_view
    )
    raw = client.messages.create(
        model=MODEL, max_tokens=137, messages=[{"role": "user", "content": "synthetic"}]
    )
    if asynchronous:
        raw = await raw
        await raw.close()
    else:
        raw.close()
    parser_calls = []

    def forbidden(**kwargs):
        parser_calls.append(True)
        raise AssertionError("closed native SDK parser must never run")

    async def aforbidden(**kwargs):
        forbidden(**kwargs)

    raw.attempt.raw.parse = aforbidden if asynchronous else forbidden
    with pytest.raises(InferenceRefused) as caught:
        if asynchronous:
            await raw.parse()
        else:
            raw.parse()
    assert caught.value.category == "context"
    assert not parser_calls
    assert len(endpoint.calls) == len(hooks.ends) == 1
    assert hooks.ends[0][1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_drain_reports_late_finalizer_failure_before_raw_assignment(bound):
    from threading import Event

    import httpx2

    factory, context, endpoint, hooks, _, _, _ = bound
    entered, release = Event(), Event()

    async def blocked(request):
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return endpoint.send(request)

    async def fail(marker, outcome):
        raise InferenceRefused("accounting")

    hooks.aafter_request = fail
    factory._transports = lambda component: (
        httpx2.MockTransport(endpoint.send),
        httpx2.MockTransport(blocked),
    )
    model = factory.create(component="helper", context=context)
    pending = asyncio.create_task(
        model._request_owner.async_view.messages.create(
            model=MODEL,
            max_tokens=137,
            messages=[{"role": "user", "content": "synthetic"}],
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(factory.aclose())
    await asyncio.sleep(0)
    release.set()
    for task in (pending, closing):
        with pytest.raises(InferenceRefused) as caught:
            await task
        assert caught.value.category == "accounting"
    assert factory._closing.result() == ("accounting", False)
    factory._expected_fixture_close_refusal = "accounting"


@pytest.mark.asyncio
async def test_sdk_close_error_shared_state_is_canonical_facts_not_traceback(bound):
    factory, context, _, _, _, _, _ = bound
    model = factory.create(component="helper", context=context)
    original = ValueError(uuid4().hex)
    calls = []

    def fail():
        calls.append("sync")
        raise original

    model._request_owner._sync.close = fail
    with pytest.raises(InferenceRefused) as caught:
        await factory.aclose()
    assert caught.value.category == "context"
    assert factory._closing.exception() is None
    assert factory._closing.result() == ("context", False)
    assert model._request_owner._async.is_closed()
    assert calls == ["sync"]
    factory._expected_fixture_close_refusal = "context"


@pytest.mark.parametrize("attribute", ["category", "retryable"])
@pytest.mark.asyncio
async def test_decoder_cannot_latch_object_graph_as_safe_fact(bound, attribute):
    factory, context, endpoint, _, _, _, _ = bound
    original = InferenceRefused("budget")
    if attribute == "retryable":
        original.retryable = ValueError(uuid4().hex)
    else:

        class UnsafeCategory(str):
            pass

        original.category = UnsafeCategory("budget")
        original.category.payload = ValueError(uuid4().hex)
    factory._decode = lambda view: original
    endpoint.category = "budget"
    model = factory.create(component="helper", context=context)
    with pytest.raises(InferenceRefused) as caught:
        await model.ainvoke("synthetic")
    assert caught.value.category == "provider"
    assert type(caught.value.category) is str
    assert type(caught.value.retryable) is bool
    assert factory._refusal == ("provider", False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_close_wins_during_native_parse_and_loser_never_publishes(
    bound, asynchronous
):
    from threading import Event

    factory, context, endpoint, hooks, _, _, _ = bound
    model = factory.create(component="raw-helper", context=context)
    client = (
        model._request_owner.async_view
        if asynchronous
        else model._request_owner.sync_view
    )
    raw = client.messages.create(
        model=MODEL, max_tokens=137, messages=[{"role": "user", "content": "synthetic"}]
    )
    if asynchronous:
        raw = await raw
    entered, release = Event(), Event()
    parser = raw.attempt.raw.parse

    def delayed(**kwargs):
        value = parser(**kwargs)
        entered.set()
        assert release.wait(5)
        return value

    async def adelayed(**kwargs):
        value = await parser(**kwargs)
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return value

    raw.attempt.raw.parse = adelayed if asynchronous else delayed
    task = asyncio.create_task(
        raw.parse() if asynchronous else asyncio.to_thread(raw.parse)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    if asynchronous:
        await raw.close()
    else:
        raw.close()
    assert hooks.ends[0][1].outcome == "cancelled"
    release.set()
    with pytest.raises(InferenceRefused) as caught:
        await task
    assert caught.value.category == "context"
    assert raw.parsed is None
    assert len(endpoint.calls) == len(hooks.ends) == 1


@pytest.mark.asyncio
async def test_auth_views_are_immutable_and_not_serializable():
    from lobster.config.providers.owned_anthropic_factory import OwnedAnthropicFactory

    # No native model/transport is needed to prove authority immutability.
    secrets = [uuid4().hex, uuid4().hex]

    def refresh(current):
        return RequestAuth(secrets[1], expires_at=time.time() + 300.0, generation=2)

    async def arefresh(current):
        return refresh(current)

    auth = ExpiringRequestAuth(
        initial=RequestAuth(secrets[0], expires_at=time.time() + 300.0),
        refresh=refresh,
        arefresh=arefresh,
    )
    factory = OwnedAnthropicFactory(
        context=InferenceContext(
            "anthropic", MODEL, "binding-a", "version-a", 200000, 137
        ),
        base_url="https://binding.example/native",
        request_auth=auth,
        authorizer_factory=lambda component, context: LocalHooks(),
        validate_request=lambda component, view: None,
        describe_request=lambda component, view: None,
        error_decoder=lambda view: None,
        request_headers=lambda descriptor: {},
    )
    try:
        value = auth.get()
        for obj in (value, auth, factory):
            with pytest.raises(TypeError):
                pickle.dumps(obj)
            for token in secrets:
                assert token not in repr(obj)
        with pytest.raises(TypeError):
            value.expires_at = time.time() + 900
    finally:
        await factory.aclose()
