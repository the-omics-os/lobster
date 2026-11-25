"""`lobster init` checks a saved Omics-OS key with the server it will use.

Placeholder keys; the HTTP request is faked, no network.
"""

import httpx
import pytest

from lobster.config import cloud_setup
from lobster.config.cloud_setup import _probe_key as REAL_PROBE

PLATFORM = "https://app.omics-os.com"


@pytest.fixture
def fake_server(monkeypatch):
    """Use the real probe with a faked HTTP answer; record what was asked."""
    asked = {}

    def answer(status=None, error=None):
        def fake_get(url, headers, timeout):
            asked["url"] = url
            if error:
                raise error
            return httpx.Response(status)

        monkeypatch.setattr(cloud_setup, "_probe_key", REAL_PROBE)
        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)

    answer.asked = asked
    return answer


def test_a_working_key_says_so(fake_server):
    fake_server(200)
    message = cloud_setup.check_omics_os_key("omk_PLACEHOLDER")
    assert message == f"✓ Omics-OS Cloud API key works with {PLATFORM}."
    assert fake_server.asked["url"] == f"{PLATFORM}/api/v1/gateway/usage"


def test_a_rejected_key_names_the_server_and_stays_saved(fake_server):
    fake_server(401)
    message = cloud_setup.check_omics_os_key("omk_PLACEHOLDER")
    assert message.startswith("Omics-OS Cloud API key rejected or expired by")
    assert PLATFORM in message and "OMICS_OS_ENDPOINT" in message
    assert "The key is saved" in message


def test_no_network_never_fails_setup(fake_server):
    fake_server(error=httpx.ConnectError("offline"))
    message = cloud_setup.check_omics_os_key("omk_PLACEHOLDER")
    assert "could not check it with" in message and PLATFORM in message


def test_the_key_goes_to_the_endpoint_the_user_set(fake_server, monkeypatch):
    fake_server(200)
    monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://tenant.omics-os.com/")
    cloud_setup.check_omics_os_key("omk_PLACEHOLDER")
    assert fake_server.asked["url"] == (
        "https://tenant.omics-os.com/api/v1/gateway/usage"
    )


def test_a_disallowed_endpoint_gets_no_key(fake_server, monkeypatch):
    fake_server(200)
    monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://attacker.example.com")
    message = cloud_setup.check_omics_os_key("omk_PLACEHOLDER")
    assert "url" not in fake_server.asked
    assert "could not check it with" in message
