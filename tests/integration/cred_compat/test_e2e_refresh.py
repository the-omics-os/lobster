"""E2E refresh — Lobster never refreshes the omics-os CLI's token.

Lobster sends only an API key (OMICS_OS_API_KEY), which does not expire on a
schedule. An expired omics-os CLI sign-in with a refresh token gets no refresh
POST from Lobster and stays byte-identical; the command says no key is set.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("command", [["cloud", "status"], ["cloud", "account"]])
def test_an_expired_sign_in_is_not_refreshed(
    isolated_home, cred_factory, fake_gateway, command
):
    shape = cred_factory.v2_multi(
        default_endpoint=fake_gateway.base_url, tenant_endpoint=fake_gateway.base_url
    )
    for profile in shape["profiles"].values():
        profile["token_expiry"] = "2000-01-01T00:00:00+00:00"
    cred_file = isolated_home.write(shape)
    before = cred_file.read_bytes()
    res = isolated_home.run_cli(command, endpoint_override=fake_gateway.base_url)
    assert res.returncode == 0, res.stderr
    assert fake_gateway.requests == [], "Lobster contacted the gateway"
    assert cred_file.read_bytes() == before
    assert "lobster cloud login --api-key" in " ".join(res.stdout.split())
