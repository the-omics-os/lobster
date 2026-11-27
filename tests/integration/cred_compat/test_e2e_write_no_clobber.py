"""E2E write safety — no Lobster command writes the omics-os CLI's sign-in.

`lobster cloud login --api-key` saves OMICS_OS_API_KEY in Lobster's own
credentials.env; `lobster cloud logout` removes only that line. Through login,
status, account and logout the omics-os CLI's file stays byte-identical, so
logging out of Lobster never signs the omics-os CLI out.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

KEY = "omk_NEWxxxxxxxxxxxxxxxxxxxxxxxxxx"


def test_login_status_account_logout_leave_the_sign_in_alone(
    isolated_home, cred_factory, fake_gateway
):
    cred_file = isolated_home.write(cred_factory.v2_multi())
    before = cred_file.read_bytes()
    for args in (
        ["cloud", "login", "--api-key", KEY],
        ["cloud", "status"],
        ["cloud", "account"],
        ["cloud", "logout"],
    ):
        res = isolated_home.run_cli(args, endpoint_override=fake_gateway.base_url)
        assert res.returncode == 0, (args, res.stdout, res.stderr)
        assert cred_file.read_bytes() == before, f"{args} changed the sign-in"


def test_login_saves_the_key_in_lobsters_own_settings(isolated_home, fake_gateway):
    res = isolated_home.run_cli(
        ["cloud", "login", "--api-key", KEY], endpoint_override=fake_gateway.base_url
    )
    assert res.returncode == 0, res.stderr
    saved = isolated_home.lobster_saved.read_text(encoding="utf-8").splitlines()
    assert f"OMICS_OS_API_KEY={KEY}" in saved
    assert not isolated_home.cred_file.exists()


def test_logout_removes_only_lobsters_key_line(isolated_home, cred_factory):
    cred_file = isolated_home.write(cred_factory.v2_single())
    before = cred_file.read_bytes()
    saved = isolated_home.save_lobster_key(KEY, "ANTHROPIC_API_KEY=sk-placeholder")
    res = isolated_home.run_cli(["cloud", "logout"])
    assert res.returncode == 0, res.stderr
    text = saved.read_text(encoding="utf-8")
    assert "OMICS_OS_API_KEY" not in text
    assert "ANTHROPIC_API_KEY=sk-placeholder" in text.splitlines()
    assert cred_file.read_bytes() == before
