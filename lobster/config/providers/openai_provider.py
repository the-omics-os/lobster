"""
OpenAI API provider implementation.

This module provides the OpenAI API provider for Lobster's LLM system,
enabling access to GPT-4o and o-series reasoning models via the official
OpenAI API.

Architecture:
    - Implements ILLMProvider interface for consistency
    - Uses static model catalog with pricing
    - Creates ChatOpenAI instances via LangChain
    - Special handling for o1/o3 reasoning models (no temperature/max_tokens)

Example:
    >>> from lobster.config.providers.openai_provider import OpenAIProvider
    >>> provider = OpenAIProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model("gpt-4o")
"""

import logging
import os
from typing import Any, List

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

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

    # Static model catalog with pricing
    # Source: https://openai.com/api/pricing/ (February 2026)
    MODELS = [
        ModelInfo(
            name="gpt-4o",
            display_name="GPT-4o",
            description="Most capable GPT model - best for complex tasks",
            provider="openai",
            context_window=128000,
            is_default=True,
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
        return self.MODELS.copy()

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
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name  # Fallback to first model

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid for OpenAI.

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "o1")

        Returns:
            bool: True if model exists in catalog

        Example:
            >>> provider = OpenAIProvider()
            >>> if provider.validate_model("gpt-4o"):
            ...     llm = provider.create_chat_model("gpt-4o")
        """
        return model_id in [m.name for m in self.MODELS]

    def _is_reasoning_model(self, model_id: str) -> bool:
        """
        Check if model is an o-series reasoning model.

        Reasoning models (o1, o3) don't support temperature or max_tokens
        parameters. They use different completion behavior.

        Args:
            model_id: Model identifier

        Returns:
            bool: True if model is an o1/o3 reasoning model
        """
        return model_id.startswith("o1") or model_id.startswith("o3")

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatOpenAI instance.

        Instantiates a LangChain ChatOpenAI model with the specified
        parameters. Uses OPENAI_API_KEY from environment.

        For o1/o3 reasoning models, temperature and max_tokens parameters
        are omitted as these models don't support them.

        Args:
            model_id: Model identifier (must be valid OpenAI model)
            temperature: Sampling temperature (0.0-2.0, default 1.0)
                Ignored for o1/o3 reasoning models.
            max_tokens: Maximum tokens in response (default 4096)
                Ignored for o1/o3 reasoning models.
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
            - o1/o3 models ignore temperature and max_tokens
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package not installed. "
                "Install with: pip install lobster-ai[openai]"
            )

        # Validate model ID
        if not self.validate_model(model_id):
            logger.warning(
                f"Model '{model_id}' not in OpenAI catalog. "
                f"Proceeding anyway (may fail at runtime)."
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

        # o1/o3 reasoning models don't support temperature or max_tokens
        if self._is_reasoning_model(model_id):
            logger.debug(
                f"Model '{model_id}' is a reasoning model - "
                "skipping temperature and max_tokens parameters."
            )
        else:
            model_kwargs["temperature"] = temperature
            model_kwargs["max_tokens"] = max_tokens

        # Merge any additional kwargs (excluding already-handled ones)
        model_kwargs.update(kwargs)

        return ChatOpenAI(**model_kwargs)

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
            "Note: o1/o3 reasoning models don't support temperature parameter."
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OpenAIProvider())
