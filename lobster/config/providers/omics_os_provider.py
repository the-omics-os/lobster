"""
Omics-OS Cloud LLM Gateway provider.

Routes LLM calls through the Omics-OS Cloud gateway (app.omics-os.com),
enabling managed billing, usage tracking, and model access control.

Usage:
    Set provider: omics-os in lobster config, or:
      export OMICS_OS_API_KEY=omk_...
      export LOBSTER_LLM_PROVIDER=omics-os

    Or authenticate via CLI:
      lobster cloud login
"""

import logging
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when the user's monthly LLM budget is exhausted."""

    def __init__(
        self,
        message: str,
        usage: Optional[dict] = None,
        upgrade_url: Optional[str] = None,
    ):
        super().__init__(message)
        self.usage = usage or {}
        self.upgrade_url = upgrade_url


class RateLimitError(Exception):
    """Raised when the gateway rate limit is exceeded after retries."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


# Friendly name -> canonical Bedrock model ID
_MODEL_ALIASES = {
    "claude-sonnet-4-5-20250514": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}


class OmicsOSProvider(ILLMProvider):
    """
    Omics-OS Cloud LLM Gateway provider.

    Routes all LLM calls through the Omics-OS managed gateway, providing
    usage tracking, billing, and model access management.

    Returns a real ChatBedrockConverse instance with a gateway transport client,
    giving full tool calling and structured output support.
    """

    _default_context_window = 200_000

    KNOWN_MODELS = [
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            display_name="Claude Sonnet 4.5 (Omics-OS Cloud)",
            description="Default gateway model — best balance of speed and capability",
            provider="omics-os",
            context_window=200000,
            is_default=True,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-5-20250929-v1:0[1m]",
            display_name="Claude Sonnet 4.5 Extended (Omics-OS Cloud)",
            description="Gateway model — 1M extended context",
            provider="omics-os",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        ),
    ]

    @property
    def name(self) -> str:
        return "omics-os"

    @property
    def display_name(self) -> str:
        return "Omics-OS Cloud"

    def is_configured(self) -> bool:
        from lobster.config.credentials import get_api_key

        return get_api_key() is not None

    def is_available(self) -> bool:
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        return self.KNOWN_MODELS.copy()

    def get_default_model(self) -> str:
        return "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    @staticmethod
    def _to_bedrock_model_id(model_id: str) -> str:
        """Resolve friendly name or canonical ID to Bedrock model ID."""
        return _MODEL_ALIASES.get(model_id, model_id)

    def _get_token_fn(self):
        """Return a callable that always returns the current valid token."""

        def _token():
            from lobster.config.credentials import get_api_key

            return get_api_key()

        return _token

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        from lobster.config.credentials import get_api_key, get_endpoint
        from lobster.config.providers.bedrock_builder import build_bedrock_converse
        from lobster.config.providers.gateway_bedrock_client import (
            GatewayBedrockClient,
        )
        from lobster.config.providers.gateway_bedrock_stub import (
            StubBedrockControlClient,
        )

        api_key = kwargs.pop("api_key", None) or get_api_key()
        if not api_key:
            raise ValueError(
                "Omics-OS Cloud not configured. "
                "Run 'lobster cloud login' or set OMICS_OS_API_KEY."
            )

        endpoint = kwargs.pop("endpoint", None) or get_endpoint()
        bedrock_model_id = self._to_bedrock_model_id(model_id)

        gateway_client = GatewayBedrockClient(
            endpoint=endpoint,
            token_fn=self._get_token_fn(),
        )

        return build_bedrock_converse(
            model_id=bedrock_model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            client=gateway_client,
            bedrock_client=StubBedrockControlClient(),
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        return (
            "Configure Omics-OS Cloud (managed Bedrock via gateway):\n\n"
            "1. Authenticate via CLI:\n"
            "   lobster cloud login\n\n"
            "2. Or set environment variable:\n"
            "   export OMICS_OS_API_KEY=omk_...\n\n"
            "Get your API key at: https://app.omics-os.com/settings/api-keys\n\n"
            f"Default model: {self.get_default_model()}"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OmicsOSProvider())
