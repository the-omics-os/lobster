"""E2E read path — the real CLI never takes the omics-os CLI's sign-in.

Under an isolated HOME, `python -m lobster.cli cloud account` with only an
omics-os CLI sign-in (any shape it has had) says it is not connected, sends
nothing and leaves the file byte-identical. A key Lobster saved in its own
credentials.env is the key it sends.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

SHAPES = ["v1_oauth", "v1_apikey", "v2_single", "v2_multi", "malformed_v2"]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_omics_os_cli_sign_in_is_not_read(
    isolated_home, cred_factory, fake_gateway, shape
):
    cred_file = isolated_home.write(getattr(cred_factory, shape)())
    before = cred_file.read_bytes()
    res = isolated_home.run_cli(
        ["cloud", "account"], endpoint_override=fake_gateway.base_url
    )
    assert res.returncode == 0, res.stderr
    assert "Traceback" not in res.stderr
    assert "Not connected" in res.stdout
    assert fake_gateway.requests == []
    assert cred_file.read_bytes() == before


def test_no_credentials_reports_not_connected(isolated_home):
    res = isolated_home.run_cli(["cloud", "account"])
    assert res.returncode == 0
    assert "Not connected" in res.stdout


def test_the_key_lobster_saved_is_the_key_it_sends(
    isolated_home, cred_factory, fake_gateway
):
    isolated_home.write(cred_factory.v2_multi())
    isolated_home.save_lobster_key("omk_LOBSTERxxxxxxxxxxxxxxxxxxxxxx")
    res = isolated_home.run_cli(
        ["cloud", "account"], endpoint_override=fake_gateway.base_url
    )
    assert res.returncode == 0, res.stderr
    assert "u_TEST" in res.stdout
    sent = fake_gateway.token_bearing_requests()
    assert [r["headers"]["authorization"] for r in sent] == [
        "Bearer omk_LOBSTERxxxxxxxxxxxxxxxxxxxxxx"
    ]
