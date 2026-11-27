"""A provider that rejects the API key is reported as that, without the
provider's own text, which can carry part of the key."""

from __future__ import annotations

import json

import pytest

from lobster.cli_internal import mcp_record, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)

SECRET = "sk-0123456789abcdef"  # nosec B105 # fake test key


def rejected(
    error_type="OpenAIAuthenticationError", error="Error code: 401 - Unauthorized"
):
    return {"type": "error", "error": error, "error_type": error_type}


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setattr(mcp_stdio, "_provider_name", lambda: "ollama-cloud")


@pytest.mark.parametrize(
    "event",
    [
        rejected(),
        rejected("AuthenticationError", f"Incorrect API key provided: {SECRET}"),
        rejected("exception", f"Error code: 401 - key {SECRET} is not valid"),
        rejected("exception", f"Incorrect API key provided: {SECRET}"),
        rejected("ProviderError", f"The request failed (HTTP 401) for key {SECRET}"),
        rejected("ProviderError", f"401 Unauthorized: {SECRET}"),
        rejected(
            "exception",
            "Error code: 401 - {'type': 'error', 'error': {'type': "
            f"'authentication_error', 'message': 'invalid x-api-key {SECRET}'}}}}",
        ),
        rejected("ResponseError", "unauthorized (status code: 401)"),
    ],
    ids=[
        "openai client",
        "key echoed",
        "plain 401",
        "api key text",
        "401 inside",
        "401 unauthorized",
        "anthropic shape",
        "ollama client",
    ],
)
def test_a_rejected_key_is_reported_by_provider_without_the_providers_text(
    engine, tmp_path, data, event, capfd
):
    _opened, script = engine
    script["events"] = [event]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "failed"
    assert done["error"] == {
        "code": "provider_key_rejected",
        "message": "The ollama-cloud provider rejected the API key it was given. "
        "Check that the key is valid for it, then start a new run.",
        "retryable": False,
    }
    assert done["summary"] == f"provider_key_rejected: {done['error']['message']}"
    everything = json.dumps(done) + capfd.readouterr().err
    assert SECRET not in everything and "OpenAIAuthenticationError" not in everything
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()


def test_a_provider_that_cannot_be_named_is_called_the_configured_one(
    engine, tmp_path, data, monkeypatch
):
    monkeypatch.setattr(mcp_stdio, "_provider_name", lambda: None)
    _opened, script = engine
    script["events"] = [rejected()]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    done = wait_terminal(server, run["run_id"])
    assert done["error"]["message"].startswith(
        "The configured provider rejected the API key"
    )


@pytest.mark.parametrize(
    "error",
    [
        {"code": "provider_error", "message": "boom"},
        {"code": "RateLimitError", "message": "Error code: 429 - slow down"},
        {"code": "exception", "message": "disk error code 4010"},
        {"code": "exception", "message": "wrote 14010 rows"},
        {"code": "exception", "message": "the key column is missing"},
        # a bare number or word is not a provider refusing the key
        {"code": "exception", "message": "Only 401 cells passed the filter"},
        {"code": "exception", "message": "Loaded 401 rows but the matrix is empty"},
        {
            "code": "exception",
            "message": "GEO download needs authentication for dbGaP controlled access",
        },
        {"code": "exception", "message": "Column 'api_key' not found in obs"},
        {"code": "exception", "message": "Unauthorized access to /data/restricted"},
        {"code": "exception", "message": "Authentication failed for the FTP mirror"},
        {"code": "exception", "message": "Error code: 401"},
        # a 401 of a tool or a data source is not the model provider's
        {
            "code": "RuntimeError",
            "message": "HTTP 401 Unauthorized from the PubMed mirror",
        },
        {"code": "PMCProviderError", "message": "HTTP 401 Unauthorized"},
        {"code": "HTTPError", "message": "401 Client Error: Unauthorized for url"},
        {"code": "exception", "message": "Invalid API key for NCBI E-utilities"},
    ],
)
def test_other_errors_are_left_alone(error):
    assert not mcp_record.rejected_key(error)


def test_a_rejection_raised_as_an_exception_is_reported_the_same_way(
    engine, tmp_path, data, monkeypatch, capfd
):
    class AuthenticationError(Exception):
        pass

    def raising(self, prompt, stream=True, cancel_event=None):
        raise AuthenticationError(f"Incorrect API key provided: {SECRET}")

    from tests.unit.mcp.test_run import ScriptedClient

    monkeypatch.setattr(ScriptedClient, "query", raising)
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    done = wait_terminal(server, run["run_id"])
    assert done["error"]["code"] == "provider_key_rejected"
    assert SECRET not in json.dumps(done) + capfd.readouterr().err


def test_a_data_sources_401_stays_a_plain_failure(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [
        rejected("RuntimeError", "HTTP 401 Unauthorized from the PubMed mirror")
    ]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "failed"
    assert done["error"]["code"] != "provider_key_rejected"
    assert "PubMed mirror" in done["error"]["message"]


def _gateway_error(status):
    """What the Omics-OS Cloud gateway client raises for *status*, as a run
    records it (class name and text)."""
    from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

    try:
        client = GatewayBedrockClient(
            endpoint="https://app.omics-os.com", token_fn=lambda: None
        )
        client._handle_error(status, "{}")
    except Exception as exc:
        return {"code": type(exc).__name__, "message": str(exc)}
    raise AssertionError("no error raised")


def test_the_cloud_gateways_401_is_a_rejected_key():
    assert mcp_record.rejected_key(_gateway_error(401))


@pytest.mark.parametrize("status", [402, 403, 422])
def test_the_cloud_gateways_other_refusals_are_not(status):
    assert not mcp_record.rejected_key(_gateway_error(status))


@pytest.mark.parametrize(
    "message",
    [
        "An error occurred (AccessDenied) when calling the GetObject operation: "
        "Access Denied",
        "An error occurred (403) when calling the HeadObject operation: Forbidden",
        "An error occurred (ExpiredToken) when calling the GetObject operation: "
        "The provided token has expired.",
    ],
)
def test_a_data_downloads_aws_error_is_not(message):
    assert not mcp_record.rejected_key({"code": "ClientError", "message": message})
