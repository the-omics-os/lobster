"""CRED_COMPAT_V2 regression + contract tests for lobster.config.credentials.

Covers the three break axes fixed by the initiative:
  1. V2 profile-nesting     — reader must descend into profiles[active_profile]
  2. cli_token field        — omics_cli creds expose cli_token as the Bearer token
  3. omc_ refresh endpoint  — refresh must pin to the PLATFORM host, never a tenant

Plus write-safety (no writer clobbers sibling profiles), the endpoint allowlist
bypass matrix, and the Codex-review-added regressions (stale alias, logout siblings,
login-over-V2 stale fields, broker platform-pin, read/write resolver parity).

All network is mocked — no test makes a live call. All tokens are PLACEHOLDERS.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def creds_env(tmp_path, monkeypatch):
    """Isolate credentials.py onto a tmp file. Returns the credentials module."""
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    cred_file = cred_dir / "credentials.json"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_file)
    # Ensure env overrides never leak in from the host.
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
    return credentials


def _v2_multi():
    """V2 file: default (platform) + databiomix (tenant), shared credential_id."""
    return {
        "version": 2,
        "active_profile": "databiomix",
        "profiles": {
            "default": {
                "credential_type": "omics_cli",
                "cli_token": "omc_DEFAULT",
                "refresh_token": "omr_DEFAULT",
                "credential_id": "clicred_SHARED",
                "endpoint": "https://app.omics-os.com",
                "auth_mode": "oauth",
                "token_expiry": "2099-01-01T00:00:00+00:00",
                "email": "user@x.com",
                "label": "default",
            },
            "databiomix": {
                "credential_type": "omics_cli",
                "cli_token": "omc_TENANT",
                "refresh_token": "omr_TENANT",
                "credential_id": "clicred_SHARED",
                "endpoint": "https://databiomix.omics-os.com",
                "auth_mode": "oauth",
                "token_expiry": "2099-01-01T00:00:00+00:00",
                "email": "user@x.com",
                "label": "databiomix",
            },
        },
    }


def _v1_oauth():
    return {
        "auth_mode": "oauth",
        "access_token": "COGNITO_ACCESS",
        "refresh_token": "COGNITO_REFRESH",
        "client_id": "cid",
        "endpoint": "https://app.omics-os.com",
        "token_expiry": "2099-01-01T00:00:00+00:00",
    }


def _v1_apikey():
    return {
        "auth_mode": "api_key",
        "api_key": "omk_V1KEY",
        "endpoint": "https://app.omics-os.com",
    }


def _fake_httpx_client(capture: dict, response_json: dict, status: int = 200):
    """Return a fake httpx.Client class that records the POST and returns response_json."""

    def _post(url, json=None, **kwargs):
        capture["url"] = url
        capture["body"] = json
        resp = MagicMock()
        resp.status_code = status
        resp.text = ""
        resp.json = lambda: response_json
        return resp

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        post = staticmethod(_post)

    return _FakeClient


# ---------------------------------------------------------------------------
# Axis 1 — V2 nesting: reader descends into the active profile
# ---------------------------------------------------------------------------


class TestAxis1V2Nesting:
    def test_get_api_key_reads_active_profile_token(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        # active_profile is databiomix → its cli_token, NOT default's
        assert creds_env.get_api_key() == "omc_TENANT"

    def test_get_endpoint_reads_active_profile_endpoint(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        assert creds_env.get_endpoint() == "https://databiomix.omics-os.com"

    def test_load_active_profile_returns_flat_active(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        view = creds_env.load_active_profile()
        assert view["label"] == "databiomix"
        assert view["cli_token"] == "omc_TENANT"
        # default profile fields must NOT bleed into the active view
        assert view["cli_token"] != "omc_DEFAULT"

    def test_load_credentials_stays_raw_on_v2(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        raw = creds_env.load_credentials()
        assert raw["version"] == 2
        assert set(raw["profiles"]) == {"default", "databiomix"}


# ---------------------------------------------------------------------------
# Axis 2 — cli_token field: omics_cli exposes cli_token as access_token
# ---------------------------------------------------------------------------


class TestAxis2CliTokenField:
    def test_cli_token_aliased_to_access_token(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        view = creds_env.load_active_profile()
        assert view["access_token"] == view["cli_token"] == "omc_TENANT"

    def test_get_api_key_returns_cli_token_when_no_access_token(self, creds_env):
        f = {
            "version": 2,
            "active_profile": "default",
            "profiles": {
                "default": {
                    "credential_type": "omics_cli",
                    "cli_token": "omc_ONLY",
                    "auth_mode": "oauth",
                    "token_expiry": "2099-01-01T00:00:00+00:00",
                    "endpoint": "https://app.omics-os.com",
                }
            },
        }
        creds_env.save_credentials(f)
        assert creds_env.get_api_key() == "omc_ONLY"

    def test_stale_access_token_masked_by_cli_token(self, creds_env):
        """Codex P2: unconditional alias, NOT setdefault — cli_token authoritative."""
        f = {
            "version": 2,
            "active_profile": "default",
            "profiles": {
                "default": {
                    "credential_type": "omics_cli",
                    "cli_token": "omc_NEW",
                    "access_token": "omc_OLD_STALE",
                    "auth_mode": "oauth",
                    "token_expiry": "2099-01-01T00:00:00+00:00",
                    "endpoint": "https://app.omics-os.com",
                }
            },
        }
        creds_env.save_credentials(f)
        assert creds_env.get_api_key() == "omc_NEW"


# ---------------------------------------------------------------------------
# Axis 3 — omc_ refresh endpoint: pins to PLATFORM, never tenant
# ---------------------------------------------------------------------------


class TestAxis3RefreshEndpoint:
    def test_omc_refresh_pins_to_platform(self, creds_env):
        creds_env.save_credentials(_v2_multi())  # active = tenant profile
        cap = {}
        fake = _fake_httpx_client(
            cap, {"cli_token": "omc_NEW", "refresh_token": "omr_NEW"}
        )
        with patch("httpx.Client", fake):
            out = creds_env.refresh_token()
        assert out == "omc_NEW"
        # MUST hit the platform host, NOT databiomix.omics-os.com
        assert cap["url"] == "https://app.omics-os.com/api/v1/gateway/oauth/cli/token"
        assert "databiomix" not in cap["url"]

    def test_omc_refresh_body_shape(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        cap = {}
        fake = _fake_httpx_client(cap, {"cli_token": "omc_NEW"})
        with patch("httpx.Client", fake):
            creds_env.refresh_token()
        assert cap["body"] == {
            "grant_type": "refresh_token",
            "credential_id": "clicred_SHARED",
            "refresh_token": "omr_TENANT",
        }

    def test_omc_refresh_persists_rotated_refresh(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        fake = _fake_httpx_client(
            {},
            {
                "cli_token": "omc_NEW",
                "refresh_token": "omr_ROTATED",
                "refresh_expiry": "2099-06-01T00:00:00+00:00",
            },
        )
        with patch("httpx.Client", fake):
            creds_env.refresh_token()
        active = creds_env.load_credentials()["profiles"]["databiomix"]
        assert active["cli_token"] == "omc_NEW"
        assert active["refresh_token"] == "omr_ROTATED"
        assert active["refresh_expiry"] == "2099-06-01T00:00:00+00:00"

    def test_omc_refresh_missing_cli_token_fails(self, creds_env):
        """npm parses cli_token ONLY — no access_token fallback."""
        creds_env.save_credentials(_v2_multi())
        fake = _fake_httpx_client({}, {"access_token": "should_be_ignored"})
        with patch("httpx.Client", fake):
            assert creds_env.refresh_token() is None

    def test_legacy_cognito_refresh_path(self, creds_env):
        creds_env.save_credentials(_v1_oauth())
        cap = {}
        fake = _fake_httpx_client(cap, {"access_token": "COGNITO_NEW"})
        with patch("httpx.Client", fake):
            out = creds_env.refresh_token()
        assert out == "COGNITO_NEW"
        assert cap["url"] == "https://app.omics-os.com/api/v1/gateway/token/refresh"
        assert cap["body"] == {"refresh_token": "COGNITO_REFRESH", "client_id": "cid"}


# ---------------------------------------------------------------------------
# Cross-profile sync
# ---------------------------------------------------------------------------


class TestCrossProfileSync:
    def test_sync_fans_by_credential_id_keeps_endpoints(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        fake = _fake_httpx_client(
            {}, {"cli_token": "omc_NEW", "refresh_token": "omr_NEW"}
        )
        with patch("httpx.Client", fake):
            creds_env.refresh_token()
        raw = creds_env.load_credentials()
        # BOTH profiles rotated (shared credential_id)...
        assert raw["profiles"]["default"]["cli_token"] == "omc_NEW"
        assert raw["profiles"]["databiomix"]["cli_token"] == "omc_NEW"
        # ...but each keeps its own endpoint
        assert raw["profiles"]["default"]["endpoint"] == "https://app.omics-os.com"
        assert (
            raw["profiles"]["databiomix"]["endpoint"]
            == "https://databiomix.omics-os.com"
        )

    def test_sync_skips_unrelated_credential_id(self, creds_env):
        f = _v2_multi()
        f["profiles"]["default"]["credential_id"] = "clicred_OTHER"
        creds_env.save_credentials(f)
        fake = _fake_httpx_client(
            {}, {"cli_token": "omc_NEW", "refresh_token": "omr_NEW"}
        )
        with patch("httpx.Client", fake):
            creds_env.refresh_token()
        raw = creds_env.load_credentials()
        # default has a DIFFERENT credential_id → must NOT be rotated
        assert raw["profiles"]["default"]["cli_token"] == "omc_DEFAULT"
        assert raw["profiles"]["databiomix"]["cli_token"] == "omc_NEW"


# ---------------------------------------------------------------------------
# Write safety — no writer clobbers siblings
# ---------------------------------------------------------------------------


class TestWriteSafety:
    def test_save_active_profile_preserves_siblings(self, creds_env):
        creds_env.save_credentials(_v2_multi())
        creds_env._save_active_profile({"user_id": "U1", "tier": "pro"})
        raw = creds_env.load_credentials()
        assert raw["version"] == 2
        assert raw["active_profile"] == "databiomix"
        assert raw["profiles"]["default"]["cli_token"] == "omc_DEFAULT"  # untouched
        assert raw["profiles"]["databiomix"]["user_id"] == "U1"

    def test_save_active_profile_v1_stays_flat(self, creds_env):
        creds_env.save_credentials(_v1_apikey())
        creds_env._save_active_profile({"tier": "enterprise"})
        raw = creds_env.load_credentials()
        assert "profiles" not in raw
        assert raw["tier"] == "enterprise"
        assert raw["api_key"] == "omk_V1KEY"

    def test_login_over_v2_clears_stale_omics_cli_fields(self, creds_env):
        """Codex P1: Cognito login over omics_cli profile must not leave stale fields."""
        creds_env.save_credentials(_v2_multi())
        creds_env._save_active_profile(
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
        active = creds_env.load_credentials()["profiles"]["databiomix"]
        assert "cli_token" not in active
        assert "credential_id" not in active
        assert active.get("credential_type") is None
        assert active["api_key"] == "omk_NEW"
        # sibling still intact
        assert (
            creds_env.load_credentials()["profiles"]["default"]["cli_token"]
            == "omc_DEFAULT"
        )

    def test_logout_removes_active_preserves_sibling(self, creds_env):
        """Codex P1: logout must not unlink the whole V2 file."""
        creds_env.save_credentials(_v2_multi())
        creds_env._clear_active_profile()
        raw = creds_env.load_credentials()
        assert "databiomix" not in raw["profiles"]
        assert "default" in raw["profiles"]
        assert raw["active_profile"] == "default"

    def test_logout_last_profile_deletes_file(self, creds_env):
        f = {
            "version": 2,
            "active_profile": "only",
            "profiles": {"only": {"cli_token": "omc_X", "auth_mode": "oauth"}},
        }
        creds_env.save_credentials(f)
        creds_env._clear_active_profile()
        assert not creds_env.CREDENTIALS_FILE.exists()

    def test_logout_v1_deletes_file(self, creds_env):
        creds_env.save_credentials(_v1_apikey())
        creds_env._clear_active_profile()
        assert not creds_env.CREDENTIALS_FILE.exists()

    def test_save_credentials_is_atomic_and_secure(self, creds_env):
        creds_env.save_credentials(_v1_apikey())
        assert creds_env.CREDENTIALS_FILE.exists()
        assert oct(creds_env.CREDENTIALS_FILE.stat().st_mode & 0o777) == oct(0o600)
        # no leftover temp files
        leftovers = list(creds_env.CREDENTIALS_DIR.glob("*.tmp.*"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# Read/write resolver parity (Codex P2)
# ---------------------------------------------------------------------------


class TestResolverParity:
    def test_read_and_write_agree_on_malformed_active_profile(self, creds_env):
        """Missing active_profile → read view and write target the SAME profile."""
        f = {
            "version": 2,
            # no active_profile key
            "profiles": {
                "default": {
                    "cli_token": "omc_D",
                    "auth_mode": "oauth",
                    "endpoint": "https://app.omics-os.com",
                },
            },
        }
        creds_env.save_credentials(f)
        view = creds_env.load_active_profile()
        assert view["cli_token"] == "omc_D"  # resolved to default
        creds_env._save_active_profile({"marker": "written"})
        assert (
            creds_env.load_credentials()["profiles"]["default"]["marker"] == "written"
        )


# ---------------------------------------------------------------------------
# V1 passthrough — existing users must not regress
# ---------------------------------------------------------------------------


class TestV1Passthrough:
    def test_v1_oauth_read(self, creds_env):
        creds_env.save_credentials(_v1_oauth())
        assert creds_env.get_api_key() == "COGNITO_ACCESS"
        assert creds_env.get_endpoint() == "https://app.omics-os.com"

    def test_v1_apikey_read(self, creds_env):
        creds_env.save_credentials(_v1_apikey())
        assert creds_env.get_api_key() == "omk_V1KEY"

    def test_v1_load_active_profile_passthrough(self, creds_env):
        creds_env.save_credentials(_v1_apikey())
        view = creds_env.load_active_profile()
        assert view["api_key"] == "omk_V1KEY"
        assert "profiles" not in view


# ---------------------------------------------------------------------------
# is_token_expired reads nested auth_mode (D10)
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    def test_expired_on_v2(self, creds_env):
        f = _v2_multi()
        f["profiles"]["databiomix"]["token_expiry"] = "2000-01-01T00:00:00+00:00"
        creds_env.save_credentials(f)
        assert creds_env.is_token_expired() is True

    def test_not_expired_on_v2(self, creds_env):
        f = _v2_multi()
        f["profiles"]["databiomix"]["token_expiry"] = "2099-01-01T00:00:00+00:00"
        creds_env.save_credentials(f)
        assert creds_env.is_token_expired() is False

    def test_stale_on_refresh_failure_returns_none(self, creds_env):
        """D9: expired + refresh fails → get_api_key None (no dead-token serving)."""
        f = _v2_multi()
        f["profiles"]["databiomix"]["token_expiry"] = "2000-01-01T00:00:00+00:00"
        creds_env.save_credentials(f)
        fake = _fake_httpx_client({}, {}, status=500)
        with patch("httpx.Client", fake):
            assert creds_env.get_api_key() is None


# ---------------------------------------------------------------------------
# Endpoint allowlist bypass matrix (PHASE 4 + Codex edge rows)
# ---------------------------------------------------------------------------


class TestEndpointAllowlist:
    @pytest.mark.parametrize(
        "url,allowed",
        [
            ("https://databiomix.omics-os.com", True),
            ("https://app.omics-os.com", True),
            ("https://stream.omics-os.com", True),
            ("http://localhost:8000", True),
            ("http://127.0.0.1", True),
            ("https://DATABIOMIX.OMICS-OS.COM", True),  # uppercase
            (
                "https://user:pass@app.omics-os.com",
                True,
            ),  # userinfo, host resolves clean
            ("https://app.omics-os.com.evil.com", False),  # suffix append
            ("https://evil-omics-os.com", False),  # no leading dot
            ("https://app.omics-os.com@evil.com", False),  # userinfo host-spoof
            ("https://evil.com/?x=app.omics-os.com", False),  # query smuggle
            ("http://databiomix.omics-os.com", False),  # http non-localhost
            ("https://omics-os.com", False),  # apex rejected
            ("https://x.omics-os.com.", False),  # trailing dot
            ("https://app.omics-os.com/path", False),  # path present
        ],
    )
    def test_validate_endpoint_matrix(self, url, allowed):
        from lobster.config.endpoint_policy import (
            EndpointPolicyError,
            validate_endpoint,
        )

        if allowed:
            validate_endpoint(url)  # must not raise
        else:
            with pytest.raises(EndpointPolicyError):
                validate_endpoint(url)

    def test_malformed_ipv6_raises_policy_error_not_valueerror(self):
        from lobster.config.endpoint_policy import (
            EndpointPolicyError,
            validate_endpoint,
        )

        for bad in ("https://[::1", "https://[gggg::1]"):
            with pytest.raises(EndpointPolicyError):
                validate_endpoint(bad)

    def test_cloud_query_wrapper_maps_to_cloudqueryerror(self):
        from lobster.cli_internal.commands.light.cloud_query import (
            CloudQueryError,
            _validate_endpoint,
        )

        with pytest.raises(CloudQueryError):
            _validate_endpoint("https://evil.com")
        # legit tenant passes
        _validate_endpoint("https://databiomix.omics-os.com")


# ---------------------------------------------------------------------------
# Endpoint validation accessors + broker platform-pin (PHASE 5)
# ---------------------------------------------------------------------------


class TestEndpointAccessors:
    def test_get_validated_endpoint_rejects_poison(self, creds_env, monkeypatch):
        from lobster.config.endpoint_policy import EndpointPolicyError

        monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://app.omics-os.com@evil.com")
        with pytest.raises(EndpointPolicyError):
            creds_env.get_validated_endpoint()

    def test_get_validated_endpoint_allows_tenant(self, creds_env, monkeypatch):
        monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://databiomix.omics-os.com")
        assert creds_env.get_validated_endpoint() == "https://databiomix.omics-os.com"

    def test_get_platform_endpoint_always_platform(self, creds_env):
        creds_env.save_credentials(_v2_multi())  # active = tenant
        # broker/auth pin ignores the active (tenant) endpoint
        assert creds_env.get_platform_endpoint() == "https://app.omics-os.com"
