"""
Anthropic Direct API provider implementation.

This module provides the Anthropic Direct API provider for Lobster's LLM system,
enabling access to Claude models via the official Anthropic API.

Architecture:
    - Implements ILLMProvider interface for consistency
    - Uses static model catalog with pricing
    - Integrates with ProviderRegistry for discovery
    - Creates ChatAnthropic instances via LangChain

Example:
    >>> from lobster.config.providers.anthropic_provider import AnthropicProvider
    >>> provider = AnthropicProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model("claude-sonnet-4-20250514")
"""

import logging
import os
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class AnthropicProvider(ILLMProvider):
    """
    Anthropic Direct API provider.

    Provides access to Claude models through Anthropic's official API.
    Requires ANTHROPIC_API_KEY environment variable.

    Features:
        - Static model catalog with pricing
        - Automatic model validation
        - ChatAnthropic integration
        - Cost tracking support

    Usage:
        >>> provider = AnthropicProvider()
        >>> if not provider.is_configured():
        ...     print("Set ANTHROPIC_API_KEY in .env")
        >>> models = provider.list_models()
        >>> llm = provider.create_chat_model(models[0].name)
    """

    _default_context_window = 200_000

    # Static model catalog with pricing
    # Source: https://www.anthropic.com/pricing (January 2025)
    KNOWN_MODELS = [
        ModelInfo(
            name="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            description="Latest Sonnet - best balance of speed and capability",
            provider="anthropic",
            context_window=200000,
            is_default=True,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        ),
        ModelInfo(
            name="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            description="Most capable model - complex reasoning and analysis",
            provider="anthropic",
            context_window=200000,
            is_default=False,
            input_cost_per_million=15.00,  # Estimated (Opus pricing)
            output_cost_per_million=75.00,
        ),
        ModelInfo(
            name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            description="Previous generation Sonnet - fast and capable",
            provider="anthropic",
            context_window=200000,
            is_default=False,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        ),
        ModelInfo(
            name="claude-3-5-haiku-20241022",
            display_name="Claude 3.5 Haiku",
            description="Fastest model - quick tasks and high throughput",
            provider="anthropic",
            context_window=200000,
            is_default=False,
            input_cost_per_million=1.00,
            output_cost_per_million=5.00,
        ),
    ]

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: "anthropic"
        """
        return "anthropic"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: "Anthropic Direct API"
        """
        return "Anthropic Direct API"

    def check_dependencies(self) -> None:
        try:
            import langchain_anthropic  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("anthropic", is_extra=True)
            raise ImportError(
                f"langchain-anthropic package not installed. Install with: {cmd}"
            )

    def is_configured(self) -> bool:
        """
        Check if Anthropic credentials are present (API key OR OAuth).

        Priority: ANTHROPIC_API_KEY env var > OAuth credentials.

        Returns:
            bool: True if any valid credential source exists
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and api_key.strip():
            return True
        # Check for OAuth credentials (lazy import to avoid circular deps)
        try:
            from lobster.config.auth import oauth_store

            return oauth_store.has_credentials("anthropic")
        except Exception:
            return False

    def is_available(self) -> bool:
        """
        Check if Anthropic API is accessible.

        For cloud providers like Anthropic, availability equals configuration
        (no local service to health-check). Actual key validation happens
        at first API call.

        Returns:
            bool: True if configured (cloud always available if configured)

        Note:
            Unlike Ollama, we don't ping the API here to avoid unnecessary
            latency and API charges. Invalid keys fail gracefully at runtime.
        """
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        """
        List all available Anthropic models.

        Returns static catalog of Claude models with pricing information.
        Models are ordered by capability (most capable first).

        Returns:
            List[ModelInfo]: Available Claude models

        Example:
            >>> provider = AnthropicProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.display_name}: ${model.input_cost_per_million}/M tokens")
        """
        return self.KNOWN_MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the recommended default model.

        Returns:
            str: "claude-sonnet-4-20250514" (Claude Sonnet 4)

        Example:
            >>> provider = AnthropicProvider()
            >>> model_id = provider.get_default_model()
            >>> llm = provider.create_chat_model(model_id)
        """
        for model in self.KNOWN_MODELS:
            if model.is_default:
                return model.name
        return self.KNOWN_MODELS[0].name  # Fallback to first model

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatAnthropic instance.

        Instantiates a LangChain ChatAnthropic model with the specified
        parameters. Uses ANTHROPIC_API_KEY from environment.

        Args:
            model_id: Model identifier (must be valid Anthropic model)
            temperature: Sampling temperature (0.0-2.0, default 1.0)
            max_tokens: Maximum tokens in response (default 4096)
            **kwargs: Additional ChatAnthropic parameters (api_key, etc.)

        Returns:
            ChatAnthropic: Configured LangChain chat model

        Raises:
            ImportError: If langchain-anthropic not installed
            ValueError: If model_id is invalid

        Example:
            >>> provider = AnthropicProvider()
            >>> llm = provider.create_chat_model(
            ...     "claude-sonnet-4-20250514",
            ...     temperature=0.7,
            ...     max_tokens=8192
            ... )
            >>> response = llm.invoke("Hello!")

        Notes:
            - Reads ANTHROPIC_API_KEY from environment by default
            - Override with api_key kwarg if needed
            - See ChatAnthropic docs for additional parameters
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("anthropic", is_extra=True)
            raise ImportError(
                f"langchain-anthropic package not installed. " f"Install with: {cmd}"
            )

        # Resolve API key: kwarg > env var > OAuth token
        api_key = kwargs.pop("api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = self._get_oauth_token()
        if not api_key:
            raise ValueError(
                "No Anthropic credentials found. Either:\n"
                "  1. Login with: lobster auth login anthropic\n"
                "  2. Set env var: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        # Create ChatAnthropic instance
        return ChatAnthropic(
            model=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _get_oauth_token() -> Optional[str]:
        """Attempt to get an access token from Anthropic OAuth credentials."""
        try:
            from lobster.config.auth.anthropic_oauth import get_access_token

            return get_access_token()
        except Exception as e:
            logger.debug("Anthropic OAuth token retrieval failed: %s", e)
            return None

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Anthropic provider.

        Returns:
            str: Configuration instructions for the user

        Example:
            >>> provider = AnthropicProvider()
            >>> if not provider.is_configured():
            ...     print(provider.get_configuration_help())
        """
        return (
            "Configure Anthropic:\n\n"
            "Option 1 — Login with Claude account (Pro/Max):\n"
            "   lobster auth login anthropic\n\n"
            "Option 2 — API key:\n"
            "   export ANTHROPIC_API_KEY=sk-ant-...\n"
            "   (Get key from https://console.anthropic.com/)\n\n"
            f"Default model: {self.get_default_model()}\n"
            f"Available models: {', '.join(self.get_model_names())}"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(AnthropicProvider())
