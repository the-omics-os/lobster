"""`lobster serve`: loopback default, token for other binds, host check, safe errors."""

import inspect

import pytest
import typer
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import serve_app, slash_commands
from lobster.cli_internal.commands.heavy.serve_app import (
    INTERNAL_ERROR_DETAIL,
    build_serve_app,
    is_loopback_host,
)

TOKEN = "s3cret-token"
LEAK = "boom at /Users/someone/private/path"


class FakeClient:
    def query(self, question, stream=False):
        return {"answer": question}

    def get_status(self):
        return {"ok": True}


class FailingClient:
    def query(self, question, stream=False):
        raise RuntimeError(LEAK)

    def get_status(self):
        raise RuntimeError(LEAK)


def _client(host, token=None, factory=FakeClient, base_url="http://localhost:8000"):
    return TestClient(build_serve_app(host, token, factory), base_url=base_url)


@pytest.fixture
def no_uvicorn(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append(host))
    monkeypatch.setattr(serve_app, "build_serve_app", lambda host, token: object())
    monkeypatch.delenv("LOBSTER_SERVE_TOKEN", raising=False)
    return calls


# 1. Default bind is loopback


def test_serve_impl_defaults_to_loopback():
    default = inspect.signature(slash_commands.serve_impl).parameters["host"].default
    assert default == "127.0.0.1"


def test_cli_serve_defaults_to_loopback(monkeypatch):
    from lobster.cli import app

    seen = {}
    monkeypatch.setattr(
        slash_commands, "serve_impl", lambda **kwargs: seen.update(kwargs)
    )
    monkeypatch.delenv("LOBSTER_SERVE_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert seen == {"port": 8000, "host": "127.0.0.1", "token": None}


def test_cli_serve_reads_token_from_env(monkeypatch):
    from lobster.cli import app

    seen = {}
    monkeypatch.setattr(
        slash_commands, "serve_impl", lambda **kwargs: seen.update(kwargs)
    )
    monkeypatch.setenv("LOBSTER_SERVE_TOKEN", TOKEN)
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert seen["token"] == TOKEN


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1", "[::1]"]
)
def test_loopback_hosts(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com", ""])
def test_non_loopback_hosts(host):
    assert not is_loopback_host(host)


# 2. Warning for non-loopback binds


def test_loopback_bind_has_no_warning(no_uvicorn, capsys):
    slash_commands.serve_impl()
    assert no_uvicorn == ["127.0.0.1"]
    assert "Warning" not in capsys.readouterr().err


def test_non_loopback_bind_warns(no_uvicorn, capsys):
    slash_commands.serve_impl(host="0.0.0.0", token=TOKEN)
    assert no_uvicorn == ["0.0.0.0"]
    err = capsys.readouterr().err
    assert "Warning: 0.0.0.0 is reachable from other machines" in err
    assert TOKEN not in err


def test_non_loopback_bind_without_token_refuses(no_uvicorn, capsys):
    with pytest.raises(typer.Exit) as exc:
        slash_commands.serve_impl(host="0.0.0.0")
    assert exc.value.exit_code == 1
    assert no_uvicorn == []
    assert "needs a token" in capsys.readouterr().err


def test_non_loopback_bind_uses_env_token(no_uvicorn, monkeypatch):
    monkeypatch.setenv("LOBSTER_SERVE_TOKEN", TOKEN)
    slash_commands.serve_impl(host="0.0.0.0")
    assert no_uvicorn == ["0.0.0.0"]


# 3. Bearer token on non-loopback binds


def test_build_app_requires_token_off_loopback():
    with pytest.raises(ValueError):
        build_serve_app("0.0.0.0", None, FakeClient)
    with pytest.raises(ValueError):
        build_serve_app("0.0.0.0", "", FakeClient)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
    ],
)
def test_missing_or_wrong_token_is_401(headers):
    client = _client("0.0.0.0", TOKEN, base_url="http://192.168.1.10:8000")
    for response in (
        client.get("/status", headers=headers),
        client.post("/query", json={"question": "hi"}, headers=headers),
        client.get("/openapi.json", headers=headers),
    ):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_correct_token_is_accepted():
    client = _client("0.0.0.0", TOKEN, base_url="http://192.168.1.10:8000")
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/status", headers=auth).json() == {"ok": True}
    response = client.post("/query", json={"question": "hi"}, headers=auth)
    assert response.json() == {"answer": "hi"}


def test_token_compare_is_constant_time(monkeypatch):
    calls = []
    real = serve_app.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(serve_app.hmac, "compare_digest", spy)
    client = _client("0.0.0.0", TOKEN, base_url="http://192.168.1.10:8000")
    client.get("/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert calls == [(TOKEN.encode(), TOKEN.encode())]


def test_loopback_needs_no_token():
    assert _client("127.0.0.1").get("/status").status_code == 200


def test_loopback_enforces_a_configured_token():
    client = _client("127.0.0.1", TOKEN)
    assert client.get("/status").status_code == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/status", headers=auth).status_code == 200
    # The Host check still applies with a valid token.
    bad_host = {**auth, "Host": "evil.example"}
    assert client.get("/status", headers=bad_host).status_code == 400


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/docs"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
        ("GET", "/no-such-route"),
        ("OPTIONS", "/query"),
        ("HEAD", "/status"),
    ],
)
def test_every_route_needs_the_token(method, path):
    client = _client("0.0.0.0", TOKEN, base_url="http://192.168.1.10:8000")
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_cli_help_does_not_echo_env_token(monkeypatch):
    from lobster.cli import app

    monkeypatch.setenv("LOBSTER_SERVE_TOKEN", TOKEN)
    result = CliRunner().invoke(app, ["serve", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "LOBSTER_SERVE_TOKEN" in result.output
    assert TOKEN not in result.output


# 4. Host check on loopback binds


@pytest.mark.parametrize(
    "host_header",
    [
        "localhost:8000",
        "127.0.0.1:8000",
        "[::1]:8000",
        "::1",
        "LOCALHOST:8000",
        "localhost",
    ],
)
def test_loopback_host_headers_allowed(host_header):
    response = _client("127.0.0.1").get("/status", headers={"Host": host_header})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "host_header", ["evil.example:8000", "evil.example", "192.168.1.10:8000", ""]
)
def test_foreign_host_headers_rejected(host_header):
    client = _client("127.0.0.1")
    assert client.get("/status", headers={"Host": host_header}).status_code == 400
    response = client.post(
        "/query", json={"question": "hi"}, headers={"Host": host_header}
    )
    assert response.status_code == 400
    assert client.get("/openapi.json", headers={"Host": host_header}).status_code == 400


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:3000", "http://127.0.0.1:8000", "https://[::1]:5173"],
)
def test_loopback_origins_allowed(origin):
    client = _client("127.0.0.1")
    response = client.post(
        "/query", json={"question": "hi"}, headers={"Origin": origin}
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://192.168.1.10:8000",
        "http://localhost.evil.com",
        "http://127.0.0.1.evil.com",
        "http://localhost:notaport",
        "http://localhost/path",
        "http://localhost?q=1",
        "http://localhost#frag",
        "null",
        "file://",
        "garbage",
    ],
)
def test_foreign_origins_rejected(origin):
    client = _client("127.0.0.1")
    for response in (
        client.post("/query", json={"question": "hi"}, headers={"Origin": origin}),
        client.get("/status", headers={"Origin": origin}),
        client.options("/query", headers={"Origin": origin}),
    ):
        assert response.status_code == 403


def test_token_bind_does_not_check_origin():
    # Intended: a browser cannot attach the bearer cross-site without CORS,
    # which the app does not enable.
    client = _client("0.0.0.0", TOKEN, base_url="http://192.168.1.10:8000")
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"}
    assert client.get("/status", headers=headers).status_code == 200


# 5. No exception text in 500 responses


@pytest.mark.parametrize("host,token", [("127.0.0.1", None), ("0.0.0.0", TOKEN)])
def test_errors_do_not_leak_exception_text(host, token):
    base = "http://localhost:8000" if token is None else "http://10.0.0.5:8000"
    client = _client(host, token, factory=FailingClient, base_url=base)
    headers = {"Authorization": f"Bearer {TOKEN}"} if token else {}
    for response in (
        client.get("/status", headers=headers),
        client.post("/query", json={"question": "hi"}, headers=headers),
    ):
        assert response.status_code == 500
        assert response.json() == {"detail": INTERNAL_ERROR_DETAIL}
        assert LEAK not in response.text


def test_errors_are_logged(caplog):
    client = _client("127.0.0.1", factory=FailingClient)
    with caplog.at_level("ERROR", logger=serve_app.__name__):
        client.post("/query", json={"question": "hi"})
        client.get("/status")
    failures = [r for r in caplog.records if r.name == serve_app.__name__]
    assert [r.getMessage() for r in failures] == [
        "lobster serve: /query failed",
        "lobster serve: /status failed",
    ]
    assert all(LEAK in str(r.exc_info[1]) for r in failures)
