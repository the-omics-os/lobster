"""An Omics-OS key and the server it is sent to come from the same source.

A key in OMICS_OS_API_KEY (Lobster's own credentials.env is loaded into it)
belongs to the platform host. The omics-os CLI's active profile may point at
a tenant host; that profile's endpoint is only for the profile's own token.
All tokens are placeholders; no network.
"""

import pytest

TENANT = "https://tenant.omics-os.com"
PLATFORM = "https://app.omics-os.com"


@pytest.fixture
def creds(tmp_path, monkeypatch):
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
    credentials.save_credentials(
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
    return credentials


def test_lobster_key_goes_to_platform_not_cli_tenant(creds, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_PLACEHOLDER_APP")
    assert creds.get_api_key() == "omk_PLACEHOLDER_APP"
    assert creds.get_endpoint() == PLATFORM


def test_explicit_endpoint_still_wins_for_lobster_key(creds, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_PLACEHOLDER_APP")
    monkeypatch.setenv("OMICS_OS_ENDPOINT", TENANT)
    assert creds.get_endpoint() == TENANT


def test_cli_profile_token_keeps_its_own_endpoint(creds):
    assert creds.get_api_key() == "omc_PLACEHOLDER_TENANT"
    assert creds.get_endpoint() == TENANT
