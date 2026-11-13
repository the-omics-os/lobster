"""Gateway client against a real local fault server: drops, silent hangs, stalls."""

import json
import logging
import socket
import threading
import time
import urllib.request
from types import SimpleNamespace

import httpx
import pytest
from botocore.exceptions import ClientError

from lobster.config.providers import gateway_bedrock_client as gbc
from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

REPLY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
}
EVENTS = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
    {"messageStop": {"stopReason": "end_turn"}},
]


def _read_request(conn):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    while len(body) < length:
        chunk = conn.recv(65536)
        if not chunk:
            return
        body += chunk


def _send(conn, status, body, content_type="application/json"):
    conn.sendall(
        f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
    )


class FaultServer:
    """Serves one scripted behaviour per accepted connection, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.accepted = 0
        self._release = threading.Event()
        self._sock = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        self._threads = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            action = self.script[min(self.accepted, len(self.script) - 1)]
            self.accepted += 1
            thread = threading.Thread(
                target=self._handle, args=(conn, action), daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _handle(self, conn, action):
        with conn:
            _read_request(conn)
            if action == "close":  # disconnect before any response byte
                return
            if action == "hang":  # accept, read, never answer
                self._release.wait()
                return
            if action == "json":
                _send(conn, "200 OK", json.dumps(REPLY).encode())
                return
            if action in ("stream", "stream_stall"):
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                    b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                )
                events = EVENTS if action == "stream" else EVENTS[:1]
                lines = [json.dumps(e) + "\n" for e in events]
                if action == "stream":
                    lines.append("[DONE]\n")
                for line in lines:
                    raw = line.encode()
                    conn.sendall(b"%x\r\n%s\r\n" % (len(raw), raw))
                if action == "stream_stall":
                    self._release.wait()
                    return
                conn.sendall(b"0\r\n\r\n")
                return
            raise AssertionError(action)

    def stop(self):
        self._release.set()
        self._sock.close()
        for thread in list(self._threads):
            thread.join(timeout=5)


@pytest.fixture
def fault_server(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(urllib.request, "getproxies", dict)  # no system proxy
    sleeps = []
    monkeypatch.setattr(gbc, "time", SimpleNamespace(sleep=sleeps.append))
    servers = []

    def start(*script):
        server = FaultServer(script)
        servers.append(server)
        return server, sleeps

    yield start
    for server in servers:
        server.stop()


def _client(server):
    return GatewayBedrockClient(server.url, lambda: "synthetic-token")


def _converse(client):
    return client.converse(modelId="synthetic", messages=[])


def _stream(client):
    return client.converse_stream(modelId="synthetic", messages=[])["stream"]


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        # The hosted endpoint enforces a shorter response limit, so nothing
        # legitimate can arrive later.
        ("https://app.omics-os.com", 190.0),
        ("https://APP.omics-os.com/", 190.0),
        # Other endpoints keep the longer 600 s wait.
        ("https://stream.omics-os.com", 600.0),
        ("https://tenant.omics-os.com", 600.0),
        ("http://localhost:8000", 600.0),
        ("https://example.invalid", 600.0),
    ],
)
def test_unary_read_bound_depends_on_the_host(endpoint, expected):
    client = GatewayBedrockClient(endpoint, lambda: "t")
    assert client._timeout == expected
    assert client._stream_idle_timeout == GatewayBedrockClient._STREAM_IDLE_TIMEOUT
    assert GatewayBedrockClient._STREAM_IDLE_TIMEOUT == 90.0


def test_explicit_timeout_wins_over_the_host_bound():
    for endpoint in ("https://app.omics-os.com", "https://stream.omics-os.com"):
        assert GatewayBedrockClient(endpoint, lambda: "t", timeout=42)._timeout == 42


class _ScriptedClient:
    """Stands in for httpx.Client; every request fails with ``error``."""

    error = None
    instances = []

    def __init__(self, timeout):
        self.timeout = timeout
        self.closed = False
        _ScriptedClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def close(self):
        self.closed = True

    def build_request(self, method, url, **kwargs):
        return httpx.Request(method, url)

    def post(self, url, **kwargs):
        raise self.error

    def send(self, request, stream=False):
        raise self.error


@pytest.fixture
def scripted_client(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gbc, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(httpx, "Client", _ScriptedClient)
    _ScriptedClient.instances = []

    def start(error):
        _ScriptedClient.error = error
        return _ScriptedClient.instances, sleeps

    return start


@pytest.mark.parametrize(
    "endpoint, read_bound",
    [("https://app.omics-os.com", 190.0), ("https://stream.omics-os.com", 600.0)],
)
@pytest.mark.parametrize("call", [_converse, _stream])
def test_read_timeout_after_sending_is_not_retried(
    scripted_client, endpoint, read_bound, call
):
    clients, sleeps = scripted_client(httpx.ReadTimeout("timed out"))
    with pytest.raises(ClientError) as caught:
        call(GatewayBedrockClient(endpoint, lambda: "t"))
    error = caught.value.response["Error"]
    assert error["Code"] == "ResponseTimeout"
    assert "not retried" in error["Message"]
    # The gateway may still be running, and billing, the request.
    assert len(clients) == 1
    assert sleeps == []
    assert all(client.closed for client in clients)
    if call is _converse:
        assert clients[0].timeout.read == read_bound
    else:
        assert clients[0].timeout.read == GatewayBedrockClient._STREAM_IDLE_TIMEOUT


@pytest.mark.parametrize(
    "endpoint", ["https://app.omics-os.com", "https://stream.omics-os.com"]
)
@pytest.mark.parametrize("call", [_converse, _stream])
@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("timed out"),
        # The request was not fully sent, so the gateway cannot have run it.
        httpx.WriteTimeout("timed out"),
        httpx.PoolTimeout("timed out"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_connection_failure_is_retried_on_every_host(
    scripted_client, endpoint, call, error
):
    clients, sleeps = scripted_client(error)
    with pytest.raises(ClientError) as caught:
        call(GatewayBedrockClient(endpoint, lambda: "t"))
    assert caught.value.response["Error"]["Code"] == "ConnectionError"
    assert len(clients) == 3
    assert len(sleeps) == 2
    assert all(client.closed for client in clients)


def test_unary_close_before_headers_is_retried(fault_server, caplog):
    server, sleeps = fault_server("close", "json")
    with caplog.at_level(logging.WARNING, logger=gbc.__name__):
        assert _converse(_client(server)) == REPLY
    assert server.accepted == 2
    assert len(sleeps) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RemoteProtocolError" in warnings[0].getMessage()
    assert "retrying (attempt 2 of 3)" in warnings[0].getMessage()


def test_unary_silent_hang_is_bounded_and_not_retried(fault_server):
    server, sleeps = fault_server("hang", "json")
    client = GatewayBedrockClient(server.url, lambda: "synthetic-token", timeout=0.3)
    started = time.monotonic()
    with pytest.raises(ClientError) as caught:
        _converse(client)
    assert time.monotonic() - started < 5
    assert caught.value.response["Error"]["Code"] == "ResponseTimeout"
    assert server.accepted == 1
    assert sleeps == []


def test_stream_silent_hang_before_headers_is_bounded_and_not_retried(fault_server):
    server, sleeps = fault_server("hang", "stream")
    client = GatewayBedrockClient(
        server.url, lambda: "synthetic-token", stream_idle_timeout=0.3
    )
    started = time.monotonic()
    with pytest.raises(ClientError) as caught:
        _stream(client)
    assert time.monotonic() - started < 5
    assert caught.value.response["Error"]["Code"] == "ResponseTimeout"
    assert caught.value.operation_name == "ConverseStream"
    assert server.accepted == 1
    assert sleeps == []


def test_unary_gives_up_after_three_drops_with_a_clear_error(fault_server):
    server, sleeps = fault_server("close")
    with pytest.raises(ClientError) as caught:
        _converse(_client(server))
    error = caught.value.response["Error"]
    assert error["Code"] == "ConnectionError"
    assert "RemoteProtocolError" in error["Message"]
    assert "3 attempts" in error["Message"]
    assert server.accepted == 3
    assert len(sleeps) == 2


def test_malformed_request_is_not_retried(fault_server):
    server, sleeps = fault_server("json")
    client = GatewayBedrockClient(server.url, lambda: "bad\r\ntoken")
    with pytest.raises(httpx.LocalProtocolError):
        _converse(client)
    with pytest.raises(httpx.LocalProtocolError):
        _stream(client)
    assert sleeps == []


def test_stream_gives_up_after_three_drops_with_a_clear_error(fault_server):
    server, sleeps = fault_server("close")
    with pytest.raises(ClientError) as caught:
        _stream(_client(server))
    assert caught.value.response["Error"]["Code"] == "ConnectionError"
    assert caught.value.operation_name == "ConverseStream"
    assert server.accepted == 3
    assert len(sleeps) == 2


def test_stream_close_before_headers_is_retried(fault_server):
    server, _ = fault_server("close", "stream")
    assert list(_stream(_client(server))) == EVENTS
    assert server.accepted == 2


def test_stream_stall_mid_events_fails_after_idle_timeout_without_replay(
    fault_server, monkeypatch, caplog
):
    monkeypatch.setattr(GatewayBedrockClient, "_STREAM_IDLE_TIMEOUT", 0.3)
    server, sleeps = fault_server("stream_stall", "stream")
    stream = _stream(_client(server))
    assert next(stream) == EVENTS[0]
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=gbc.__name__):
        with pytest.raises(httpx.ReadTimeout):
            next(stream)
    assert time.monotonic() - started < 5
    # Output already reached the caller: a replay would duplicate it.
    assert server.accepted == 1
    assert sleeps == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "after 1 event" in warnings[0].getMessage()
    with pytest.raises(StopIteration):
        next(stream)
