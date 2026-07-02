"""E2E refresh — omc_ refresh drives a real HTTP POST against the fake gateway.

Test-seam (PHASE_E option a): refresh is hard-pinned to ``PLATFORM_ENDPOINT``.
We monkeypatch that constant to the ``fake_gateway`` base URL (in-process, zero
production surface) so the real ``refresh_token()`` code path issues a real POST
we can inspect: URL path, body shape, and the rotation/sync it persists.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def creds_on(tmp_path, monkeypatch):
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
    return credentials


def test_omc_refresh_hits_fake_gateway_platform_path(
    creds_on, cred_factory, fake_gateway, monkeypatch
):
    # Pin the platform (broker) endpoint at the recording fake gateway.
    monkeypatch.setattr(creds_on, "PLATFORM_ENDPOINT", fake_gateway.base_url)
    creds_on.save_credentials(cred_factory.v2_multi())

    out = creds_on.refresh_token()

    # Rotated token returned + persisted.
    assert out == "omc_ROTATEDxxxxxxxxxxxxxxxxxxxxxxxx"

    # Exactly one POST, to the oauth/cli/token path, with the exact body shape.
    # refresh_token is the ACTIVE (databiomix) profile's stored omr_ token.
    active_refresh = cred_factory.v2_multi()["profiles"]["databiomix"]["refresh_token"]
    posts = [r for r in fake_gateway.requests if r["method"] == "POST"]
    assert len(posts) == 1, posts
    assert posts[0]["path"].endswith("/api/v1/gateway/oauth/cli/token")
    assert posts[0]["body"] == {
        "grant_type": "refresh_token",
        "credential_id": "clicred_SHARED",
        "refresh_token": active_refresh,
    }


def test_omc_refresh_persists_and_syncs(
    creds_on, cred_factory, fake_gateway, monkeypatch
):
    monkeypatch.setattr(creds_on, "PLATFORM_ENDPOINT", fake_gateway.base_url)
    creds_on.save_credentials(cred_factory.v2_multi())

    creds_on.refresh_token()
    raw = creds_on.load_credentials()

    # Active profile rotated (cli_token + rotated refresh + expiry).
    active = raw["profiles"]["databiomix"]
    assert active["cli_token"] == "omc_ROTATEDxxxxxxxxxxxxxxxxxxxxxxxx"
    assert active["refresh_token"] == "omr_ROTATEDxxxxxxxxxxxxxxxxxxxxxx"
    assert active["refresh_expiry"] == "2099-06-01T00:00:00+00:00"
    assert active["endpoint"] == "https://databiomix.omics-os.com"  # unchanged

    # Sibling (shared credential_id) rotated too, but keeps its OWN endpoint.
    sib = raw["profiles"]["default"]
    assert sib["cli_token"] == "omc_ROTATEDxxxxxxxxxxxxxxxxxxxxxxxx"
    assert sib["endpoint"] == "https://app.omics-os.com"  # NOT clobbered


def test_omc_refresh_never_hits_tenant_host(
    creds_on, cred_factory, fake_gateway, monkeypatch
):
    """The tenant endpoint must never receive the refresh POST."""
    monkeypatch.setattr(creds_on, "PLATFORM_ENDPOINT", fake_gateway.base_url)
    creds_on.save_credentials(cred_factory.v2_multi())
    creds_on.refresh_token()
    for r in fake_gateway.requests:
        assert "databiomix" not in r["path"]
        assert "databiomix" not in r.get("host", "")


def test_credential_id_never_sent_as_bearer(
    creds_on, cred_factory, fake_gateway, monkeypatch
):
    """clicred_ is an identifier, never a bearer secret — must not appear in headers."""
    monkeypatch.setattr(creds_on, "PLATFORM_ENDPOINT", fake_gateway.base_url)
    creds_on.save_credentials(cred_factory.v2_multi())
    creds_on.refresh_token()
    for r in fake_gateway.requests:
        for hv in r["headers"].values():
            assert "clicred_" not in hv
