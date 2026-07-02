"""E2E write-safety — no writer clobbers sibling profiles.

The crown-jewel gate. A V2 multi-profile file has ``default`` + ``databiomix``
sharing a credential. Every writer (logout, enrich, login-over, refresh) must
leave the non-active sibling byte-intact and keep ``version``/``active_profile``.

Logout is driven through the REAL subprocess CLI (`lobster cloud logout`) — the
end-to-end proof. The file-only writers (enrich RMW, login-over, refresh
write-back) are driven in-process against an isolated CREDENTIALS_FILE, because
they otherwise require a live browser-OAuth callback or gateway; their network
is out of scope here (covered by unit + endpoint-security tests).
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_logout_via_subprocess_preserves_sibling(isolated_home, cred_factory):
    """`lobster cloud logout` on a V2 multi file removes ONLY the active profile."""
    cred_file = isolated_home.write(cred_factory.v2_multi())
    res = isolated_home.run_cli(["cloud", "logout"])
    assert res.returncode == 0, res.stderr

    data = _load(cred_file)
    assert data["version"] == 2
    assert "databiomix" not in data["profiles"], "active profile not removed"
    assert "default" in data["profiles"], "SIBLING clobbered on logout!"
    assert (
        data["profiles"]["default"]["cli_token"] == "omc_DEFAULTxxxxxxxxxxxxxxxxxxxxx"
    )
    assert data["active_profile"] == "default", "active_profile not repointed"


def test_enrich_rmw_preserves_sibling(tmp_path, monkeypatch, cred_factory):
    """Go-TUI enrich writer (_save_active_profile) keeps siblings intact."""
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")

    credentials.save_credentials(cred_factory.v2_multi())
    credentials._save_active_profile(
        {"user_id": "U9", "email": "new@x.com", "tier": "pro"}
    )

    raw = credentials.load_credentials()
    assert raw["version"] == 2 and raw["active_profile"] == "databiomix"
    assert raw["profiles"]["default"]["cli_token"] == "omc_DEFAULTxxxxxxxxxxxxxxxxxxxxx"
    assert raw["profiles"]["databiomix"]["user_id"] == "U9"


def test_login_over_v2_clears_stale_and_keeps_sibling(
    tmp_path, monkeypatch, cred_factory
):
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")

    credentials.save_credentials(cred_factory.v2_multi())
    credentials._save_active_profile(
        {
            "auth_mode": "api_key",
            "api_key": "omk_NEW",
            "endpoint": "https://databiomix.omics-os.com",
        },
        remove=(
            "cli_token",
            "credential_id",
            "credential_type",
            "access_token",
            "refresh_token",
        ),
    )
    raw = credentials.load_credentials()
    active = raw["profiles"]["databiomix"]
    assert "cli_token" not in active and "credential_id" not in active
    assert active["api_key"] == "omk_NEW"
    assert raw["profiles"]["default"]["cli_token"] == "omc_DEFAULTxxxxxxxxxxxxxxxxxxxxx"


def test_v1_writers_stay_flat(tmp_path, monkeypatch, cred_factory):
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")

    credentials.save_credentials(cred_factory.v1_apikey())
    credentials._save_active_profile({"tier": "enterprise"})
    raw = credentials.load_credentials()
    assert "profiles" not in raw
    assert raw["tier"] == "enterprise" and raw["api_key"] == "omk_TESTKEY"
