"""A rejected Omics-OS key names the server that refused it.

A key can be valid on one Omics-OS server and unknown on another, so the
message must say which server was asked and how to point Lobster at the
right one, not only "create a new key". Placeholder keys; no network.
"""

import json

import httpx
import pytest
from botocore.exceptions import ClientError

TENANT = "https://tenant.omics-os.com"


def _assert_names_server(message: str) -> None:
    # The prefix is what the chat UI matches on (go_tui_launcher._friendly_error).
    assert message.startswith("Omics-OS Cloud API key rejected or expired")
    assert TENANT in message
    assert "OMICS_OS_ENDPOINT" in message
    assert "lobster-cli cloud login --device" not in message
    # how to enter the new key; the variable wins over the saved key
    assert "lobster cloud login --api-key <key>" in message
    assert "OMICS_OS_API_KEY" in message
    assert "npm" not in message


def test_gateway_401_names_the_server(monkeypatch):
    from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

    client = GatewayBedrockClient(endpoint=TENANT, token_fn=lambda: "omk_PLACEHOLDER")
    with pytest.raises(ClientError) as error:
        client._handle_error(401, json.dumps({"detail": "Not authenticated"}))
    _assert_names_server(error.value.response["Error"]["Message"])


def test_cloud_session_401_names_the_server(monkeypatch):
    from lobster.cli_internal.commands.light import cloud_query

    transport = httpx.MockTransport(lambda request: httpx.Response(401))
    real_client = httpx.Client
    monkeypatch.setattr(
        cloud_query.httpx,
        "Client",
        lambda *a, **k: real_client(transport=transport),
    )
    with pytest.raises(cloud_query.CloudQueryError) as error:
        cloud_query.create_cloud_session(TENANT, {"X-API-Key": "omk_PLACEHOLDER"})
    _assert_names_server(str(error.value))
