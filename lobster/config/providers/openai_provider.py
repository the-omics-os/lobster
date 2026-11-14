"""
OpenAI API provider implementation.

This module provides the OpenAI API provider for Lobster's LLM system,
enabling access to GPT-4o and o-series reasoning models via the official
OpenAI API.

Architecture:
    - Implements ILLMProvider interface for consistency
    - Uses static model catalog with pricing
    - Creates ChatOpenAI instances via LangChain
    - Temperature and max_tokens sent only when explicitly set

Example:
    >>> from lobster.config.providers.openai_provider import OpenAIProvider
    >>> provider = OpenAIProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model("gpt-4o")
"""

import logging
import os
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo
from lobster.config.providers.stream_usage_hint import with_stream_usage_hint

logger = logging.getLogger(__name__)


class OpenAIProvider(ILLMProvider):
    """
    OpenAI API provider.

    Provides access to GPT-4o, GPT-4o Mini, and o-series reasoning models
    through the official OpenAI API. Requires OPENAI_API_KEY environment variable.

    Features:
        - Static model catalog with pricing
        - Automatic model validation
        - ChatOpenAI integration via LangChain
        - Special handling for o1/o3 reasoning models

    Usage:
        >>> provider = OpenAIProvider()
        >>> if not provider.is_configured():
        ...     print("Set OPENAI_API_KEY in .env")
        >>> models = provider.list_models()
        >>> llm = provider.create_chat_model(models[0].name)
    """

    _default_context_window = 128_000

    # Static model catalog with pricing
    # Source: https://openai.com/api/pricing/ (GPT-6 rows: October 2026)
    KNOWN_MODELS = [
        ModelInfo(
            name="gpt-6.1-sol",
            display_name="GPT-6.1 Sol",
            description="Default - newest GPT; price shown for prompts up to 272K tokens",
            provider="openai",
            context_window=272000,
            is_default=True,
            input_cost_per_million=2.0,
            output_cost_per_million=10.0,
        ),
        ModelInfo(
            name="gpt-6-sol",
            display_name="GPT-6 Sol",
            description="Price shown for prompts up to 272K tokens",
            provider="openai",
            context_window=272000,
            is_default=False,
            input_cost_per_million=2.0,
            output_cost_per_million=10.0,
        ),
        ModelInfo(
            name="gpt-6-luna",
            display_name="GPT-6 Luna",
            description="Fast and low-cost; price shown for prompts up to 272K tokens",
            provider="openai",
            context_window=272000,
            is_default=False,
            input_cost_per_million=0.1,
            output_cost_per_million=0.5,
        ),
        ModelInfo(
            name="gpt-6-astra",
            display_name="GPT-6 Astra",
            description="Most capable GPT; price shown for prompts up to 272K tokens",
            provider="openai",
            context_window=272000,
            is_default=False,
            input_cost_per_million=10.0,
            output_cost_per_million=50.0,
        ),
        ModelInfo(
            name="gpt-4o",
            display_name="GPT-4o",
            description="Most capable GPT model - best for complex tasks",
            provider="openai",
            context_window=128000,
            is_default=False,
            input_cost_per_million=2.50,
            output_cost_per_million=10.00,
        ),
        ModelInfo(
            name="gpt-4o-mini",
            display_name="GPT-4o Mini",
            description="Fast and affordable - good for simple tasks",
            provider="openai",
            context_window=128000,
            is_default=False,
            input_cost_per_million=0.15,
            output_cost_per_million=0.60,
        ),
        ModelInfo(
            name="o1",
            display_name="o1 (Reasoning)",
            description="Advanced reasoning model - complex problem solving",
            provider="openai",
            context_window=200000,
            is_default=False,
            input_cost_per_million=15.00,
            output_cost_per_million=60.00,
        ),
        ModelInfo(
            name="o1-mini",
            display_name="o1 Mini",
            description="Smaller reasoning model - fast reasoning tasks",
            provider="openai",
            context_window=128000,
            is_default=False,
            input_cost_per_million=3.00,
            output_cost_per_million=12.00,
        ),
        ModelInfo(
            name="o3-mini",
            display_name="o3 Mini",
            description="Latest compact reasoning model - efficient reasoning",
            provider="openai",
            context_window=200000,
            is_default=False,
            input_cost_per_million=1.10,
            output_cost_per_million=4.40,
        ),
        ModelInfo(
            name="gpt-5.5-2026-04-23",
            display_name="GPT-5.5",
            description="Most capable GPT model - multimodal reasoning",
            provider="openai",
            context_window=1048576,
            is_default=False,
            input_cost_per_million=5.00,
            output_cost_per_million=30.00,
        ),
    ]

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: "openai"
        """
        return "openai"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: "OpenAI"
        """
        return "OpenAI"

    def check_dependencies(self) -> None:
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("openai", is_extra=True)
            raise ImportError(
                f"langchain-openai package not installed. Install with: {cmd}"
            )

    def is_configured(self) -> bool:
        """
        Check if OpenAI API key is present.

        Checks for OPENAI_API_KEY environment variable.
        Does NOT validate the key (use is_available() for that).

        Returns:
            bool: True if OPENAI_API_KEY is set

        Example:
            >>> provider = OpenAIProvider()
            >>> if not provider.is_configured():
            ...     print("Set OPENAI_API_KEY=sk-... in .env")
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        return bool(api_key and api_key.strip())

    def is_available(self) -> bool:
        """
        Check if OpenAI API is accessible.

        For cloud providers like OpenAI, availability equals configuration
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
        List all available OpenAI models.

        Returns static catalog of OpenAI models with pricing information.
        Models are ordered by capability (most capable first).

        Returns:
            List[ModelInfo]: Available OpenAI models

        Example:
            >>> provider = OpenAIProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.display_name}: ${model.input_cost_per_million}/M tokens")
        """
        return self.KNOWN_MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the recommended default model.

        Returns:
            str: "gpt-4o"

        Example:
            >>> provider = OpenAIProvider()
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatOpenAI instance.

        Instantiates a LangChain ChatOpenAI model with the specified
        parameters. Uses OPENAI_API_KEY from environment.

        Temperature and max_tokens are sent only when set. Reasoning models
        (o-series, GPT-5.5 and GPT-6) accept only their default sampling, so
        leaving them unset keeps every model valid.

        Args:
            model_id: Model identifier (must be valid OpenAI model)
            temperature: Sampling temperature. None (default) sends none and
                the model default applies.
            max_tokens: Maximum tokens in response. None (default) sends none.
            **kwargs: Additional ChatOpenAI parameters (api_key, etc.)

        Returns:
            ChatOpenAI: Configured LangChain chat model

        Raises:
            ImportError: If langchain-openai not installed
            ValueError: If API key not found

        Example:
            >>> provider = OpenAIProvider()
            >>> llm = provider.create_chat_model(
            ...     "gpt-4o",
            ...     temperature=0.7,
            ...     max_tokens=8192
            ... )
            >>> response = llm.invoke("Hello!")

        Notes:
            - Reads OPENAI_API_KEY from environment by default
            - Override with api_key kwarg if needed
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("openai", is_extra=True)
            raise ImportError(
                f"langchain-openai package not installed. " f"Install with: {cmd}"
            )

        # Get API key (prefer kwarg, fallback to environment)
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment or kwargs. "
                "Set it with: export OPENAI_API_KEY=sk-..."
            )

        # Build model kwargs
        model_kwargs = {
            "model": model_id,
            "api_key": api_key,
        }

        # Send sampling and output limits only when explicitly requested.
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        # Merge any additional kwargs (excluding already-handled ones)
        model_kwargs.update(kwargs)
        # langchain-openai asks for streamed usage only on the default OpenAI base URL;
        # behind OPENAI_BASE_URL (a proxy) it would return none.
        model_kwargs.setdefault("stream_usage", True)

        return with_stream_usage_hint(ChatOpenAI(**model_kwargs))

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring OpenAI provider.

        Returns:
            str: Configuration instructions for the user

        Example:
            >>> provider = OpenAIProvider()
            >>> if not provider.is_configured():
            ...     print(provider.get_configuration_help())
        """
        return (
            "Configure OpenAI:\n\n"
            "1. Get API key from: https://platform.openai.com/api-keys\n"
            "2. Set environment variable:\n"
            "   export OPENAI_API_KEY=sk-...\n\n"
            "Or add to .env file:\n"
            "   OPENAI_API_KEY=sk-...\n\n"
            f"Default model: {self.get_default_model()}\n"
            f"Available models: {', '.join(self.get_model_names())}\n\n"
            "Note: reasoning models accept only their default temperature."
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OpenAIProvider())
