"""Credential admission is identical for ``x-api-key`` and ``Bearer`` SDKs.

anthropic>=2 sends the API key as ``Authorization: Bearer <key>``; older SDKs
send ``x-api-key``. The owned client accepts exactly those two equivalent forms
and normalizes the request to the legacy wire form before any other check.
"""

import json

import pytest

from lobster.core.exceptions import InferenceRefused
from tests.unit.config._native_inference_helpers import ANTHROPIC_MODEL
from tests.unit.config.test_controlled_anthropic_client import (  # noqa: F401
    controlled,
)

pytestmark = pytest.mark.no_auto_config

KEY = "fake-anthropic-key"
URL = "https://api.anthropic.com/v1/messages"


def _send(owner, headers, *, url=URL, method="POST"):
    """Run the owner's pre-send hook on a hand-built request."""
    import httpx2

    request = httpx2.Request(
        method,
        url,
        headers=headers,
        content=json.dumps(
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 137,
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode(),
    )
    attempt = owner._start()
    token = owner._attempt.set(attempt)
    try:
        owner._before_http(request)
    finally:
        owner._attempt.reset(token)
        owner.finalize(attempt, "completed")
    return request


def _assert_wire(request):
    assert request.headers.get_list("x-api-key") == [KEY]
    assert "authorization" not in request.headers


def test_legacy_x_api_key_is_accepted(controlled):
    _, owner, _, authorizer, _ = controlled
    request = _send(owner, {"x-api-key": KEY})
    _assert_wire(request)
    assert len(authorizer.descriptors) == 1


@pytest.mark.parametrize("name", ["Authorization", "authorization", "AUTHORIZATION"])
def test_bearer_is_accepted_and_normalized_to_legacy_wire_form(controlled, name):
    _, owner, _, authorizer, _ = controlled
    request = _send(owner, {name: f"Bearer {KEY}"})
    _assert_wire(request)
    assert len(authorizer.descriptors) == 1


def test_bearer_and_matching_x_api_key_collapse_to_one_credential(controlled):
    _, owner, _, authorizer, _ = controlled
    request = _send(owner, {"x-api-key": KEY, "Authorization": f"Bearer {KEY}"})
    _assert_wire(request)
    assert len(authorizer.descriptors) == 1


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Authorization": "Bearer different-key"}, id="other-bearer"),
        pytest.param({"Authorization": f"Bearer {KEY}x"}, id="bearer-prefix-extra"),
        pytest.param({"Authorization": "Bearer "}, id="empty-bearer"),
        pytest.param({"Authorization": f"Bearer  {KEY}"}, id="double-space"),
        pytest.param({"Authorization": f"bearer {KEY}"}, id="lowercase-scheme"),
        pytest.param({"Authorization": f"Basic {KEY}"}, id="basic-scheme"),
        pytest.param({"Authorization": f"Token {KEY}"}, id="token-scheme"),
        pytest.param({"Authorization": KEY}, id="no-scheme"),
        pytest.param({"x-api-key": "different-key"}, id="other-api-key"),
        pytest.param({}, id="no-credential"),
    ],
)
def test_wrong_or_missing_credential_is_refused_before_admission(controlled, headers):
    _, owner, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused) as caught:
        _send(owner, headers)
    assert caught.value.category == "context"
    assert KEY not in str(caught.value) and KEY not in repr(caught.value)
    assert not authorizer.descriptors and not probe.sent


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(
            {"x-api-key": "different-key", "Authorization": f"Bearer {KEY}"},
            id="api-key-disagrees",
        ),
        pytest.param(
            {"x-api-key": KEY, "Authorization": "Bearer different-key"},
            id="bearer-disagrees",
        ),
        pytest.param(
            {"x-api-key": KEY, "Authorization": f"Basic {KEY}"}, id="scheme-conflict"
        ),
    ],
)
def test_conflicting_credentials_are_refused(controlled, headers):
    _, owner, _, authorizer, _ = controlled
    with pytest.raises(InferenceRefused) as caught:
        _send(owner, headers)
    assert caught.value.category == "context"
    assert not authorizer.descriptors


@pytest.mark.parametrize(
    "headers",
    [
        [("Authorization", f"Bearer {KEY}"), ("authorization", f"Bearer {KEY}")],
        [("Authorization", f"Bearer {KEY}"), ("Authorization", "Bearer other")],
        [("x-api-key", KEY), ("X-Api-Key", KEY)],
    ],
    ids=["duplicate-bearer", "duplicate-bearer-conflict", "duplicate-api-key"],
)
def test_repeated_credential_headers_are_refused(controlled, headers):
    _, owner, _, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        _send(owner, headers)
    assert not authorizer.descriptors


@pytest.mark.parametrize(
    "credential",
    [{"x-api-key": KEY}, {"Authorization": f"Bearer {KEY}"}],
    ids=["legacy", "bearer"],
)
def test_cookie_is_refused_with_either_credential(controlled, credential):
    _, owner, _, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        _send(owner, {**credential, "Cookie": "session=1"})
    assert not authorizer.descriptors


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://api.anthropic.com/v1/messages/batches"},
        {"url": "https://evil.example/v1/messages"},
        {"method": "GET"},
    ],
    ids=["wrong-path", "wrong-host", "wrong-method"],
)
def test_url_and_method_still_checked_with_bearer(controlled, kwargs):
    _, owner, _, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        _send(owner, {"Authorization": f"Bearer {KEY}"}, **kwargs)
    assert not authorizer.descriptors


@pytest.mark.parametrize("suffix", ["", "?beta=true"], ids=["plain", "beta-query"])
def test_beta_route_accepts_both_sdk_spellings(controlled, suffix):
    _, owner, _, authorizer, _ = controlled
    attempt = owner._start()
    attempt.beta = True
    token = owner._attempt.set(attempt)
    import httpx2

    request = httpx2.Request(
        "POST",
        URL + suffix,
        headers={"Authorization": f"Bearer {KEY}"},
        content=json.dumps({"model": ANTHROPIC_MODEL, "max_tokens": 137}).encode(),
    )
    try:
        owner._before_http(request)
    finally:
        owner._attempt.reset(token)
        owner.finalize(attempt, "completed")
    _assert_wire(request)
    assert len(authorizer.descriptors) == 1


def test_non_beta_route_rejects_beta_query(controlled):
    _, owner, _, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        _send(owner, {"x-api-key": KEY}, url=URL + "?beta=true")
    assert not authorizer.descriptors


def test_sdk_native_credential_form_reaches_the_wire_normalized(controlled):
    """Whatever this installed SDK emits, the sent request is x-api-key only."""
    model, _, probe, authorizer, _ = controlled
    model.invoke("hello")
    assert len(probe.sent) == len(authorizer.descriptors) == 1
    _assert_wire(probe.sent[0])


def test_mismatched_bearer_override_is_refused_end_to_end(controlled):
    model, _, probe, authorizer, _ = controlled
    with pytest.raises(InferenceRefused):
        model.invoke("hello", extra_headers={"Authorization": "Bearer other-key"})
    assert not probe.sent and not authorizer.descriptors


def test_gateway_request_auth_still_replaces_credential(controlled):
    import time

    from lobster.config.providers.request_auth import RequestAuth

    _, owner, _, _, _ = controlled
    import httpx2

    request = httpx2.Request("POST", URL, headers={"Authorization": f"Bearer {KEY}"})
    owner._normalize_credential(request)
    owner._apply_auth(
        request, RequestAuth("gateway-token", expires_at=time.time() + 60.0)
    )
    assert request.headers.get_list("x-api-key") == ["gateway-token"]
    assert "authorization" not in request.headers
