"""Bounded waits for the OpenAI-compatible providers.

A stalled host must raise a named error after the idle bound, a slow but steady
stream must pass, and a timed-out request is never sent twice.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("langchain_openai")

from lobster.config.providers import (  # noqa: E402
    nebius_provider,
    ollama_cloud_provider,
    openrouter_provider,
)
from lobster.config.providers.openai_provider import OpenAIProvider  # noqa: E402
from lobster.config.providers.request_bounds import ProviderIdleTimeout  # noqa: E402


def _chunk(text):
    body = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


class _Host:
    """Fake chat-completions host; `script` decides what each request gets."""

    def __init__(self, script):
        self.requests = 0
        host = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                host.requests += 1
                script(host.requests, self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _start_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()


def _stall_after_first_chunk(n, handler):
    _start_stream(handler)
    handler.wfile.write(_chunk("a"))
    handler.wfile.flush()
    time.sleep(10)


def _steady(n, handler):
    _start_stream(handler)
    for i in range(8):
        handler.wfile.write(_chunk(str(i)))
        handler.wfile.flush()
        time.sleep(0.4)
    handler.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", "1")
    monkeypatch.setenv("LOBSTER_LLM_RESPONSE_TIMEOUT_S", "5")


def _model(host):
    return OpenAIProvider().create_chat_model("m", base_url=host.url)


def test_stalled_stream_raises_named_error_once(env):
    host = _Host(_stall_after_first_chunk)
    try:
        start = time.monotonic()
        with pytest.raises(ProviderIdleTimeout) as err:
            list(_model(host).stream("hi"))
        assert (
            time.monotonic() - start < 8
        )  # ends at the 1 s bound, not the host's 10 s
        assert err.value.code == "provider_idle_timeout"
        assert host.requests == 1
    finally:
        host.close()


def test_stalled_async_stream_raises_named_error_once(env):
    host = _Host(_stall_after_first_chunk)

    async def run():
        return [c async for c in _model(host).astream("hi")]

    try:
        with pytest.raises(ProviderIdleTimeout) as err:
            asyncio.run(run())
        assert err.value.code == "provider_idle_timeout"
        assert host.requests == 1
    finally:
        host.close()


def test_slow_but_steady_stream_passes(env):
    host = _Host(_steady)
    try:
        start = time.monotonic()
        chunks = list(_model(host).stream("hi"))
        assert time.monotonic() - start > 2.5  # longer than the 1 s idle bound
        assert "".join(c.content for c in chunks) == "01234567"
    finally:
        host.close()


def test_unary_response_timeout_is_named_and_not_retried(env, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_RESPONSE_TIMEOUT_S", "1")
    host = _Host(lambda n, h: time.sleep(4))
    try:
        with pytest.raises(ProviderIdleTimeout) as err:
            _model(host).invoke("hi")
        assert err.value.code == "provider_response_timeout"
        assert host.requests == 1
    finally:
        host.close()


def _status_then_ok(status, failures):
    def script(n, handler):
        if n <= failures:
            handler.send_response(status)
            handler.send_header("Retry-After", "0")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        _start_stream(handler)
        handler.wfile.write(_chunk("ok"))
        handler.wfile.write(b"data: [DONE]\n\n")

    return script


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retryable_status_is_retried_twice_before_any_body(env, status):
    host = _Host(_status_then_ok(status, 2))
    try:
        chunks = list(_model(host).stream("hi"))
        assert "".join(c.content for c in chunks) == "ok"
        assert host.requests == 3
    finally:
        host.close()


def test_retries_stop_after_two(env):
    host = _Host(_status_then_ok(503, 99))
    try:
        with pytest.raises(Exception):
            list(_model(host).stream("hi"))
        assert host.requests == 3
    finally:
        host.close()


def test_other_status_is_not_retried(env):
    host = _Host(_status_then_ok(400, 99))
    try:
        with pytest.raises(Exception):
            list(_model(host).stream("hi"))
        assert host.requests == 1
    finally:
        host.close()


def test_zero_disables_the_idle_bound(env, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", "0")

    def pause_then_finish(n, handler):
        _start_stream(handler)
        time.sleep(2)
        handler.wfile.write(_chunk("late"))
        handler.wfile.write(b"data: [DONE]\n\n")

    host = _Host(pause_then_finish)
    try:
        chunks = list(_model(host).stream("hi"))
        assert "".join(c.content for c in chunks) == "late"
    finally:
        host.close()


@pytest.mark.parametrize(
    "module,provider_cls,key_env",
    [
        (nebius_provider, "NebiusProvider", "NEBIUS_API_KEY"),
        (openrouter_provider, "OpenRouterProvider", "OPENROUTER_API_KEY"),
        (ollama_cloud_provider, "OllamaCloudProvider", "OLLAMA_API_KEY"),
    ],
)
def test_every_openai_compatible_provider_is_bounded(
    env, monkeypatch, module, provider_cls, key_env
):
    host = _Host(_stall_after_first_chunk)
    monkeypatch.setenv(key_env, "k")
    monkeypatch.setattr(module, "_BASE_URL", host.url)
    try:
        model = getattr(module, provider_cls)().create_chat_model("m")
        with pytest.raises(ProviderIdleTimeout):
            list(model.stream("hi"))
        assert host.requests == 1
    finally:
        host.close()


def test_defaults_and_retry_after_cap(monkeypatch):
    from lobster.config.providers import request_bounds

    monkeypatch.delenv("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("LOBSTER_LLM_RESPONSE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("LOBSTER_LLM_CONNECT_RETRIES", raising=False)
    assert request_bounds._setting("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", 180) == 180
    assert request_bounds._setting("LOBSTER_LLM_RESPONSE_TIMEOUT_S", 600) == 600
    assert request_bounds._retries() == 2

    class Resp:
        def __init__(self, value):
            self.headers = {"Retry-After": value}

    assert request_bounds._delay(Resp("120"), 0) == 60
    assert request_bounds._delay(Resp("3"), 0) == 3


def test_stalled_local_ollama_raises_named_error_once(monkeypatch):
    pytest.importorskip("langchain_ollama")
    from lobster.config.providers.ollama_provider import OllamaProvider

    monkeypatch.setenv("LOBSTER_LLM_RESPONSE_TIMEOUT_S", "1")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    def script(n, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.end_headers()
        line = {
            "model": "m",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "a"},
            "done": False,
        }
        handler.wfile.write((json.dumps(line) + "\n").encode())
        handler.wfile.flush()
        time.sleep(6)

    host = _Host(script)
    try:
        provider = OllamaProvider()
        provider._base_url = host.url.removesuffix("/v1")
        with pytest.raises(ProviderIdleTimeout) as err:
            list(provider.create_chat_model("m").stream("hi"))
        assert err.value.code == "provider_idle_timeout"
        assert host.requests == 1
    finally:
        host.close()


def test_idle_off_still_bounds_a_silent_stream_by_the_response_bound(env, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", "0")
    monkeypatch.setenv("LOBSTER_LLM_RESPONSE_TIMEOUT_S", "1")
    host = _Host(lambda n, h: time.sleep(4))
    try:
        with pytest.raises(ProviderIdleTimeout):
            list(_model(host).stream("hi"))
        assert host.requests == 1
    finally:
        host.close()


def test_error_names_a_disabled_bound_without_crashing():
    err = ProviderIdleTimeout("provider_response_timeout", None)
    assert err.code == "provider_response_timeout"


def test_callers_request_timeout_wins(env):
    model = OpenAIProvider().create_chat_model(
        "m", base_url="http://127.0.0.1:1/v1", request_timeout=77
    )
    assert model.request_timeout == 77
