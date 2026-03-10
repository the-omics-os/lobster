"""Tests for the Omics-OS Cloud provider, credentials manager, and CLI commands."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Credentials tests
# ---------------------------------------------------------------------------

class TestCredentials:
    """Tests for lobster.config.credentials module."""

    def test_load_credentials_missing_file(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        assert credentials.load_credentials() is None

    def test_save_and_load_credentials(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        cred_dir = tmp_path / "omics-os"
        cred_file = cred_dir / "credentials.json"
        monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_file)

        data = {"auth_mode": "api_key", "api_key": "omk_test123", "endpoint": "https://example.com"}
        credentials.save_credentials(data)

        assert cred_file.exists()
        assert oct(cred_file.stat().st_mode & 0o777) == oct(0o600)

        loaded = credentials.load_credentials()
        assert loaded["api_key"] == "omk_test123"

    def test_clear_credentials(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        cred_file = tmp_path / "credentials.json"
        cred_file.write_text("{}")
        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_file)

        credentials.clear_credentials()
        assert not cred_file.exists()

    def test_clear_credentials_no_file(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        credentials.clear_credentials()  # Should not raise

    def test_get_api_key_env_priority(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        cred_dir = tmp_path / "omics-os"
        cred_file = cred_dir / "credentials.json"
        monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_file)

        data = {"api_key": "omk_from_file"}
        credentials.save_credentials(data)

        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_from_env")
        assert credentials.get_api_key() == "omk_from_env"

    def test_get_api_key_file_fallback(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        cred_dir = tmp_path / "omics-os"
        cred_file = cred_dir / "credentials.json"
        monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_file)

        data = {"api_key": "omk_from_file"}
        credentials.save_credentials(data)

        monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
        assert credentials.get_api_key() == "omk_from_file"

    def test_get_api_key_none(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
        assert credentials.get_api_key() is None

    def test_get_endpoint_default(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
        assert credentials.get_endpoint() == "https://app.omics-os.com"

    def test_get_endpoint_env_override(self, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://custom.example.com/")
        assert credentials.get_endpoint() == "https://custom.example.com"

    def test_get_auth_headers_with_key(self, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_test")
        headers = credentials.get_auth_headers()
        assert headers == {"Authorization": "Bearer omk_test"}

    def test_get_auth_headers_empty(self, tmp_path, monkeypatch):
        from lobster.config import credentials

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
        assert credentials.get_auth_headers() == {}


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------

class TestOmicsOSProvider:
    """Tests for OmicsOSProvider."""

    def test_provider_properties(self):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        p = OmicsOSProvider()
        assert p.name == "omics-os"
        assert p.display_name == "Omics-OS Cloud"

    def test_list_models(self):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        p = OmicsOSProvider()
        models = p.list_models()
        assert len(models) >= 1
        assert models[0].name == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert models[0].is_default is True

    def test_get_default_model(self):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        p = OmicsOSProvider()
        assert p.get_default_model() == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_is_configured_false(self, tmp_path, monkeypatch):
        from lobster.config import credentials
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)

        p = OmicsOSProvider()
        assert p.is_configured() is False

    def test_is_configured_true(self, monkeypatch):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_test123")

        p = OmicsOSProvider()
        assert p.is_configured() is True

    def test_create_chat_model_no_key_raises(self, tmp_path, monkeypatch):
        from lobster.config import credentials
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "nope.json")
        monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)

        p = OmicsOSProvider()
        with pytest.raises(ValueError, match="not configured"):
            p.create_chat_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    def test_create_chat_model_returns_bedrock_converse(self, monkeypatch):
        """create_chat_model should return a ChatBedrockConverse via gateway shim."""
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_test123")

        p = OmicsOSProvider()
        model = p.create_chat_model(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0.5,
            max_tokens=1024,
        )
        from langchain_aws import ChatBedrockConverse

        assert isinstance(model, ChatBedrockConverse)
        assert model.model_id == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert model.temperature == 0.5

    def test_create_chat_model_alias_resolution(self, monkeypatch):
        """Friendly model names should be resolved to canonical Bedrock IDs."""
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        monkeypatch.setenv("OMICS_OS_API_KEY", "omk_test123")

        p = OmicsOSProvider()
        model = p.create_chat_model("claude-sonnet-4-5-20250514")
        from langchain_aws import ChatBedrockConverse

        assert isinstance(model, ChatBedrockConverse)
        assert model.model_id == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_configuration_help(self):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        p = OmicsOSProvider()
        help_text = p.get_configuration_help()
        assert "lobster cloud login" in help_text

    def test_model_alias_resolution_static(self):
        from lobster.config.providers.omics_os_provider import OmicsOSProvider

        assert OmicsOSProvider._to_bedrock_model_id("claude-sonnet-4-5-20250514") == \
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert OmicsOSProvider._to_bedrock_model_id("claude-sonnet-4-5-20250929") == \
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        # Already canonical — pass through
        assert OmicsOSProvider._to_bedrock_model_id("us.anthropic.claude-sonnet-4-5-20250929-v1:0") == \
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# ---------------------------------------------------------------------------
# Provider registration tests
# ---------------------------------------------------------------------------

class TestProviderRegistration:
    """Tests that omics-os provider integrates with the registry."""

    def test_omics_os_in_valid_providers(self):
        from lobster.config.constants import VALID_PROVIDERS

        assert "omics-os" in VALID_PROVIDERS

    def test_omics_os_display_name(self):
        from lobster.config.constants import PROVIDER_DISPLAY_NAMES

        assert PROVIDER_DISPLAY_NAMES["omics-os"] == "Omics-OS Cloud"

    def test_registry_discovers_provider(self):
        from lobster.config.providers.registry import ProviderRegistry

        ProviderRegistry.reset()
        provider = ProviderRegistry.get("omics-os")
        assert provider is not None
        assert provider.name == "omics-os"


# ---------------------------------------------------------------------------
# Gateway shim tests
# ---------------------------------------------------------------------------

class TestGatewayBedrockClient:
    """Tests for the GatewayBedrockClient shim."""

    def test_meta_region(self):
        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        assert client.meta.region_name == "us-east-1"

    def test_should_retry(self):
        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        assert client._should_retry(429) is True
        assert client._should_retry(503) is True
        assert client._should_retry(200) is False
        assert client._should_retry(401) is False

    def test_retry_delay_increases(self):
        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        d0 = client._retry_delay(0)
        d1 = client._retry_delay(1)
        d2 = client._retry_delay(2)
        # Base: 1, 2, 4 -- with jitter up to 0.5
        assert 1.0 <= d0 <= 1.5
        assert 2.0 <= d1 <= 2.5
        assert 4.0 <= d2 <= 4.5

    def test_handle_error_401(self):
        from botocore.exceptions import ClientError

        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        with pytest.raises(ClientError) as exc_info:
            client._handle_error(401, '{"detail":"Auth failed"}')
        assert "UnauthorizedAccess" in str(exc_info.value)

    def test_handle_error_402(self):
        from botocore.exceptions import ClientError

        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        body = json.dumps({
            "detail": "Monthly budget exhausted",
        })
        with pytest.raises(ClientError) as exc_info:
            client._handle_error(402, body)
        assert "BudgetExceeded" in str(exc_info.value)

    def test_handle_error_422(self):
        from botocore.exceptions import ClientError

        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        with pytest.raises(ClientError) as exc_info:
            client._handle_error(422, '{"detail":"Validation error"}')
        assert "ValidationException" in str(exc_info.value)

    def test_handle_error_500(self):
        from botocore.exceptions import ClientError

        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )
        with pytest.raises(ClientError) as exc_info:
            client._handle_error(500, '{"detail":"Internal error"}')
        assert "ServiceError" in str(exc_info.value)

    def test_build_headers(self):
        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "my_token",
        )
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my_token"
        assert headers["X-Omics-Gateway-Contract"] == "bedrock-converse-v1"

    def test_converse_raises_rate_limit_after_retries(self):
        """After max retries on 429, should raise RateLimitError."""
        from unittest.mock import patch, MagicMock
        from lobster.config.providers.gateway_bedrock_client import GatewayBedrockClient
        from lobster.config.providers.omics_os_provider import RateLimitError

        client = GatewayBedrockClient(
            endpoint="https://example.com",
            token_fn=lambda: "tok",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"

        with patch("httpx.Client") as mock_client_cls, \
             patch("lobster.config.providers.gateway_bedrock_client.time.sleep"):
            mock_http = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_http)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_http.post.return_value = mock_resp

            with pytest.raises(RateLimitError, match="(?i)rate limit"):
                client.converse(
                    modelId="test-model",
                    messages=[{"role": "user", "content": [{"text": "hi"}]}],
                )


class TestStubBedrockControlClient:
    """Tests for the StubBedrockControlClient."""

    def test_meta_region(self):
        from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient

        stub = StubBedrockControlClient()
        assert stub.meta.region_name == "us-east-1"

    def test_get_inference_profile_raises(self):
        from lobster.config.providers.gateway_bedrock_stub import StubBedrockControlClient

        stub = StubBedrockControlClient()
        with pytest.raises(NotImplementedError):
            stub.get_inference_profile(inferenceProfileIdentifier="test")


# ---------------------------------------------------------------------------
# BudgetExhaustedError tests
# ---------------------------------------------------------------------------

class TestBudgetExhaustedError:
    """Tests that BudgetExhaustedError still works after the rewrite."""

    def test_basic(self):
        from lobster.config.providers.omics_os_provider import BudgetExhaustedError

        err = BudgetExhaustedError(
            "Budget exhausted",
            usage={"spent_usd": 2.00, "budget_usd": 2.00},
            upgrade_url="https://app.omics-os.com/settings/billing",
        )
        assert str(err) == "Budget exhausted"
        assert err.usage["spent_usd"] == 2.00
        assert err.upgrade_url is not None


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_basic(self):
        from lobster.config.providers.omics_os_provider import RateLimitError

        err = RateLimitError("Rate limit exceeded", retry_after_seconds=10.0)
        assert str(err) == "Rate limit exceeded"
        assert err.retry_after_seconds == 10.0

    def test_default_retry_after(self):
        from lobster.config.providers.omics_os_provider import RateLimitError

        err = RateLimitError("Rate limit exceeded")
        assert err.retry_after_seconds is None


# ---------------------------------------------------------------------------
# Provider setup config test
# ---------------------------------------------------------------------------

class TestProviderSetupConfig:
    """Tests for create_omics_os_config in provider_setup."""

    def test_create_config_success(self):
        from lobster.config.provider_setup import create_omics_os_config

        config = create_omics_os_config("omk_test123")
        assert config.success is True
        assert config.provider_type == "omics-os"
        assert config.env_vars["OMICS_OS_API_KEY"] == "omk_test123"
        assert config.env_vars["LOBSTER_LLM_PROVIDER"] == "omics-os"

    def test_create_config_empty_key(self):
        from lobster.config.provider_setup import create_omics_os_config

        config = create_omics_os_config("")
        assert config.success is False

    def test_create_config_none_key(self):
        from lobster.config.provider_setup import create_omics_os_config

        config = create_omics_os_config(None)
        assert config.success is False
