"""An Omics-OS key and the server it is sent to come from Lobster's settings.

A key in OMICS_OS_API_KEY (Lobster's own credentials.env is loaded into it)
belongs to the platform host unless OMICS_OS_ENDPOINT names another. The
omics-os CLI's sign-in may point at a tenant host; Lobster reads neither its
token nor its endpoint. All tokens are placeholders; no network.
"""

import json

import pytest

TENANT = "https://tenant.omics-os.com"
PLATFORM = "https://app.omics-os.com"


@pytest.fixture
def creds(tmp_path, monkeypatch):
    from lobster.config import credentials

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
    sign_in = tmp_path / ".config" / "omics-os" / "credentials.json"
    sign_in.parent.mkdir(parents=True)
    sign_in.write_text(
        json.dumps(
            {
                "version": 2,
                "active_profile": "tenant",
                "profiles": {
                    "tenant": {
                        "credential_type": "omics_cli",
                        "cli_token": "omc_PLACEHOLDER_TENANT",
                        "endpoint": TENANT,
                        "auth_mode": "oauth",
                        "token_expiry": "2099-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )
    return credentials


def test_lobster_key_goes_to_platform_not_cli_tenant(creds, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_PLACEHOLDER_APP")
    assert creds.get_api_key() == "omk_PLACEHOLDER_APP"
    assert creds.get_endpoint() == PLATFORM


def test_explicit_endpoint_still_wins_for_lobster_key(creds, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_PLACEHOLDER_APP")
    monkeypatch.setenv("OMICS_OS_ENDPOINT", TENANT)
    assert creds.get_endpoint() == TENANT


def test_the_cli_sign_in_is_neither_key_nor_endpoint(creds):
    assert creds.get_api_key() is None
    assert creds.get_endpoint() == PLATFORM
