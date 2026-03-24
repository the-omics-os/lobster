"""Tests for Anthropic OAuth backend (PKCE, token exchange, credential store)."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.config.auth.pkce import generate_pkce
from lobster.config.auth.oauth_store import OAuthCredentials, save, load, delete, has_credentials
from lobster.config.auth.anthropic_oauth import (
    PROVIDER_ID,
    build_authorize_url,
    exchange_authorization_code,
    get_access_token,
    is_authenticated,
    login_interactive,
    logout,
    refresh_access_token,
    _parse_authorization_input,
    OAuthError,
)


# ---------------------------------------------------------------------------
# PKCE tests
# ---------------------------------------------------------------------------


class TestPKCE:
    def test_verifier_length(self):
        verifier, _ = generate_pkce()
        # 32 bytes base64url → 43 chars (no padding)
        assert len(verifier) == 43

    def test_challenge_is_sha256(self):
        import base64
        import hashlib

        verifier, challenge = generate_pkce()
        expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_unique_per_call(self):
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2
        assert c1 != c2

    def test_base64url_no_padding(self):
        verifier, challenge = generate_pkce()
        assert "=" not in verifier
        assert "=" not in challenge
        assert "+" not in verifier
        assert "/" not in verifier


# ---------------------------------------------------------------------------
# OAuth credential store tests
# ---------------------------------------------------------------------------


class TestOAuthStore:
    @pytest.fixture(autouse=True)
    def _tmp_auth_dir(self, tmp_path, monkeypatch):
        """Redirect AUTH_DIR to a temp directory."""
        monkeypatch.setattr("lobster.config.auth.oauth_store.AUTH_DIR", tmp_path / "auth")

    def _make_creds(self, **overrides) -> OAuthCredentials:
        defaults = {
            "access_token": "acc_test_123",
            "refresh_token": "ref_test_456",
            "expires_at": time.time() + 3600,
            "provider": "anthropic",
        }
        defaults.update(overrides)
        return OAuthCredentials(**defaults)

    def test_save_and_load(self):
        creds = self._make_creds()
        save(creds)
        loaded = load("anthropic")
        assert loaded is not None
        assert loaded.access_token == "acc_test_123"
        assert loaded.refresh_token == "ref_test_456"

    def test_load_missing_returns_none(self):
        assert load("nonexistent") is None

    def test_delete(self):
        creds = self._make_creds()
        save(creds)
        assert delete("anthropic") is True
        assert load("anthropic") is None

    def test_delete_missing_returns_false(self):
        assert delete("nonexistent") is False

    def test_has_credentials(self):
        assert has_credentials("anthropic") is False
        save(self._make_creds())
        assert has_credentials("anthropic") is True

    def test_is_expired_with_buffer(self):
        # Expires in 200 seconds, buffer is 300 → should be expired
        creds = self._make_creds(expires_at=time.time() + 200)
        assert creds.is_expired(buffer_seconds=300) is True

    def test_is_not_expired(self):
        creds = self._make_creds(expires_at=time.time() + 3600)
        assert creds.is_expired() is False

    def test_preserves_metadata(self):
        creds = self._make_creds(email="user@example.com", account_id="acct_123")
        save(creds)
        loaded = load("anthropic")
        assert loaded.email == "user@example.com"
        assert loaded.account_id == "acct_123"

    def test_file_permissions(self, tmp_path, monkeypatch):
        import os
        import stat

        auth_dir = tmp_path / "auth"
        monkeypatch.setattr("lobster.config.auth.oauth_store.AUTH_DIR", auth_dir)
        save(self._make_creds())
        cred_file = auth_dir / "anthropic.json"
        mode = stat.S_IMODE(os.stat(cred_file).st_mode)
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Anthropic OAuth flow tests
# ---------------------------------------------------------------------------


class TestBuildAuthorizeURL:
    def test_contains_required_params(self):
        url = build_authorize_url("test_verifier", "test_challenge")
        assert "claude.ai/oauth/authorize" in url
        assert "client_id=" in url
        assert "code_challenge=test_challenge" in url
        assert "code_challenge_method=S256" in url
        assert "state=test_verifier" in url
        assert "response_type=code" in url
        assert "redirect_uri=" in url
        assert "scope=" in url


class TestParseAuthorizationInput:
    def test_full_url(self):
        url = "http://localhost:53692/callback?code=abc123&state=verifier1"
        result = _parse_authorization_input(url, "verifier1")
        assert result == {"code": "abc123", "state": "verifier1"}

    def test_code_hash_state(self):
        result = _parse_authorization_input("abc123#verifier1", "verifier1")
        assert result == {"code": "abc123", "state": "verifier1"}

    def test_bare_code(self):
        result = _parse_authorization_input("abc123", "verifier1")
        assert result == {"code": "abc123", "state": "verifier1"}

    def test_state_mismatch_url(self):
        url = "http://localhost:53692/callback?code=abc&state=wrong"
        result = _parse_authorization_input(url, "verifier1")
        assert result is None

    def test_empty_input(self):
        result = _parse_authorization_input("", "verifier1")
        assert result is None


class TestTokenExchange:
    @patch("lobster.config.auth.anthropic_oauth._post_token")
    def test_exchange_code_success(self, mock_post):
        mock_post.return_value = {
            "access_token": "acc_new",
            "refresh_token": "ref_new",
            "expires_in": 3600,
        }
        creds = exchange_authorization_code("code1", "state1", "verifier1")
        assert creds.access_token == "acc_new"
        assert creds.refresh_token == "ref_new"
        assert creds.provider == "anthropic"
        # Verify PKCE params were sent
        call_args = mock_post.call_args[0][0]
        assert call_args["grant_type"] == "authorization_code"
        assert call_args["code_verifier"] == "verifier1"

    @patch("lobster.config.auth.anthropic_oauth._post_token")
    def test_refresh_success(self, mock_post):
        mock_post.return_value = {
            "access_token": "acc_refreshed",
            "refresh_token": "ref_rotated",
            "expires_in": 3600,
        }
        creds = refresh_access_token("ref_old")
        assert creds.access_token == "acc_refreshed"
        assert creds.refresh_token == "ref_rotated"
        call_args = mock_post.call_args[0][0]
        assert call_args["grant_type"] == "refresh_token"

    @patch("lobster.config.auth.anthropic_oauth._post_token")
    def test_refresh_preserves_old_refresh_token(self, mock_post):
        """If server doesn't rotate refresh token, keep the old one."""
        mock_post.return_value = {
            "access_token": "acc_new",
            "expires_in": 3600,
        }
        creds = refresh_access_token("ref_old")
        assert creds.refresh_token == "ref_old"


class TestGetAccessToken:
    @pytest.fixture(autouse=True)
    def _tmp_auth_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lobster.config.auth.oauth_store.AUTH_DIR", tmp_path / "auth")

    def test_no_credentials(self):
        assert get_access_token() is None

    def test_valid_token(self):
        creds = OAuthCredentials(
            access_token="acc_valid",
            refresh_token="ref_1",
            expires_at=time.time() + 3600,
            provider="anthropic",
        )
        save(creds)
        assert get_access_token() == "acc_valid"

    @patch("lobster.config.auth.anthropic_oauth.refresh_access_token")
    def test_expired_token_refreshes(self, mock_refresh):
        mock_refresh.return_value = OAuthCredentials(
            access_token="acc_refreshed",
            refresh_token="ref_2",
            expires_at=time.time() + 3600,
            provider="anthropic",
        )
        creds = OAuthCredentials(
            access_token="acc_expired",
            refresh_token="ref_1",
            expires_at=time.time() - 100,  # Already expired
            provider="anthropic",
        )
        save(creds)
        token = get_access_token()
        assert token == "acc_refreshed"
        mock_refresh.assert_called_once_with("ref_1")


class TestIsAuthenticated:
    @pytest.fixture(autouse=True)
    def _tmp_auth_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lobster.config.auth.oauth_store.AUTH_DIR", tmp_path / "auth")

    def test_not_authenticated(self):
        assert is_authenticated() is False

    def test_authenticated(self):
        save(OAuthCredentials(
            access_token="a", refresh_token="r",
            expires_at=time.time() + 3600, provider="anthropic",
        ))
        assert is_authenticated() is True


class TestLogout:
    @pytest.fixture(autouse=True)
    def _tmp_auth_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lobster.config.auth.oauth_store.AUTH_DIR", tmp_path / "auth")

    def test_logout_clears_credentials(self):
        save(OAuthCredentials(
            access_token="a", refresh_token="r",
            expires_at=time.time() + 3600, provider="anthropic",
        ))
        assert logout() is True
        assert is_authenticated() is False

    def test_logout_no_credentials(self):
        assert logout() is False
