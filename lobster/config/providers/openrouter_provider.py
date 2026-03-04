"""
OpenRouter provider implementation.

OpenRouter is a unified API gateway for 600+ LLM models across providers.
Uses the OpenAI API protocol — no new LangChain packages required.

API key setup: https://openrouter.ai/keys
Model catalog:  https://openrouter.ai/models

Architecture:
    - Implements ILLMProvider interface
    - Uses ChatOpenAI with base_url override (OpenAI-compatible API)
    - Lazy live catalog fetched from openrouter.ai/api/v1/models on first list_models() call
    - Class-level in-process cache — fetched once, never re-fetched
    - Curated fallback catalog used when network is unavailable
    - Hardcoded Lobster AI branding headers for OpenRouter leaderboard

Model naming convention: "provider/model-name" (e.g., "anthropic/claude-sonnet-4-5")

Example:
    >>> provider = OpenRouterProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()  # fetches live catalog on first call
    ...     llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")
"""

import logging
import os
from typing import Any, ClassVar, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)

# OpenRouter API constants
_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://lobsterbio.com",
    "X-Title": "Lobster AI",
}
_ENV_VAR = "OPENROUTER_API_KEY"
_DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
_CATALOG_TIMEOUT = 5.0  # seconds


# Curated fallback catalog — representative models from major families.
# Used when live catalog fetch fails or httpx is unavailable.
_FALLBACK_MODELS: List[ModelInfo] = [
    ModelInfo(
        name="anthropic/claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        description="Anthropic's latest Sonnet — fast, capable, excellent for complex tasks",
        provider="openrouter",
        context_window=200000,
        is_default=True,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    ModelInfo(
        name="anthropic/claude-3-5-haiku",
        display_name="Claude 3.5 Haiku",
        description="Anthropic's fastest model — ideal for lightweight tasks",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    ModelInfo(
        name="openai/gpt-4o",
        display_name="GPT-4o",
        description="OpenAI's flagship multimodal model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="openai/gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Fast and affordable GPT-4 variant",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    ModelInfo(
        name="openai/o1",
        display_name="o1 (Reasoning)",
        description="Advanced reasoning model for complex problem-solving",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=15.00,
        output_cost_per_million=60.00,
    ),
    ModelInfo(
        name="openai/o3-mini",
        display_name="o3 Mini",
        description="Efficient compact reasoning model",
        provider="openrouter",
        context_window=200000,
        is_default=False,
        input_cost_per_million=1.10,
        output_cost_per_million=4.40,
    ),
    ModelInfo(
        name="meta-llama/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        description="Meta's most capable open-source model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="meta-llama/llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B Instruct",
        description="Compact, fast open-source model",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.06,
        output_cost_per_million=0.06,
    ),
    ModelInfo(
        name="deepseek/deepseek-r1",
        display_name="DeepSeek R1",
        description="Strong reasoning model, open-source",
        provider="openrouter",
        context_window=164000,
        is_default=False,
        input_cost_per_million=0.55,
        output_cost_per_million=2.19,
    ),
    ModelInfo(
        name="deepseek/deepseek-chat-v3-0324",
        display_name="DeepSeek Chat V3",
        description="DeepSeek's top conversational model",
        provider="openrouter",
        context_window=164000,
        is_default=False,
        input_cost_per_million=0.27,
        output_cost_per_million=1.10,
    ),
    ModelInfo(
        name="google/gemini-2.0-flash-001",
        display_name="Gemini 2.0 Flash",
        description="Google's fastest multimodal model",
        provider="openrouter",
        context_window=1000000,
        is_default=False,
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="google/gemini-2.5-pro-preview",
        display_name="Gemini 2.5 Pro Preview",
        description="Google's most capable model with 1M context",
        provider="openrouter",
        context_window=1000000,
        is_default=False,
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="mistralai/mistral-large",
        display_name="Mistral Large",
        description="Mistral's flagship model for complex tasks",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.00,
        output_cost_per_million=6.00,
    ),
    ModelInfo(
        name="mistralai/mistral-nemo",
        display_name="Mistral Nemo",
        description="Compact Mistral model, great cost/quality ratio",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=0.13,
        output_cost_per_million=0.13,
    ),
    ModelInfo(
        name="qwen/qwen-2.5-72b-instruct",
        display_name="Qwen 2.5 72B Instruct",
        description="Alibaba's top open model, strong at coding and science",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="x-ai/grok-2-1212",
        display_name="Grok 2",
        description="xAI's flagship model",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=2.00,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="cohere/command-r-plus",
        display_name="Command R+",
        description="Cohere's enterprise-grade RAG model",
        provider="openrouter",
        context_window=128000,
        is_default=False,
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    ModelInfo(
        name="microsoft/phi-4",
        display_name="Phi-4",
        description="Microsoft's compact 14B reasoning model",
        provider="openrouter",
        context_window=16384,
        is_default=False,
        input_cost_per_million=0.07,
        output_cost_per_million=0.14,
    ),
    ModelInfo(
        name="nvidia/llama-3.1-nemotron-70b-instruct",
        display_name="Nemotron 70B",
        description="NVIDIA-tuned Llama with superior instruction following",
        provider="openrouter",
        context_window=131072,
        is_default=False,
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    ModelInfo(
        name="amazon/nova-pro-v1",
        display_name="Amazon Nova Pro",
        description="Amazon's multimodal frontier model",
        provider="openrouter",
        context_window=300000,
        is_default=False,
        input_cost_per_million=0.80,
        output_cost_per_million=3.20,
    ),
]


class OpenRouterProvider(ILLMProvider):
    """
    OpenRouter provider — unified API gateway to 600+ LLM models.

    OpenRouter accepts requests in OpenAI API format and routes them to the
    appropriate upstream model. A single OPENROUTER_API_KEY gives access to
    models from Anthropic, OpenAI, Meta, Google, Mistral, DeepSeek, and more.

    Model names use the "provider/model-name" format:
        - "anthropic/claude-sonnet-4-5"
        - "openai/gpt-4o"
        - "meta-llama/llama-3.3-70b-instruct"

    Features:
        - Live model catalog fetched from openrouter.ai/api/v1/models (lazy, cached)
        - 20-model curated fallback catalog for offline use
        - Hardcoded Lobster AI branding headers (https://openrouter.ai/leaderboard)
        - No new dependencies — uses ChatOpenAI with base_url override

    Usage:
        >>> provider = OpenRouterProvider()
        >>> if not provider.is_configured():
        ...     print("Set OPENROUTER_API_KEY in .env")
        >>> models = provider.list_models()  # fetches live catalog on first call
        >>> llm = provider.create_chat_model("anthropic/claude-sonnet-4-5")
    """

    # Class-level in-process cache for live model catalog
    _models_cache: ClassVar[Optional[List[ModelInfo]]] = None

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter (600+ models)"

    def is_configured(self) -> bool:
        """Check if OPENROUTER_API_KEY is present and non-empty."""
        api_key = os.environ.get(_ENV_VAR)
        return bool(api_key and api_key.strip())

    def is_available(self) -> bool:
        """Check if OpenRouter is accessible (equals is_configured for cloud providers)."""
        return self.is_configured()

    def get_default_model(self) -> str:
        """Get the recommended default model."""
        return _DEFAULT_MODEL

    def list_models(self) -> List[ModelInfo]:
        """
        List available models from OpenRouter.

        On first call, fetches the live catalog from openrouter.ai/api/v1/models
        and caches it in the class for the process lifetime. On network failure
        or if httpx is unavailable, returns the curated fallback catalog.
        """
        if OpenRouterProvider._models_cache is not None:
            return OpenRouterProvider._models_cache

        try:
            import httpx

            api_key = os.environ.get(_ENV_VAR, "")
            response = httpx.get(
                f"{_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_CATALOG_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [self._parse_model(m) for m in data if m.get("id")]
            OpenRouterProvider._models_cache = models
            logger.debug(f"Fetched {len(models)} models from OpenRouter API")
            return models

        except ImportError:
            logger.debug("httpx not available, using OpenRouter fallback catalog")
        except Exception as e:
            logger.debug(f"OpenRouter model fetch failed ({e}), using fallback catalog")

        return list(_FALLBACK_MODELS)

    def _parse_model(self, data: dict) -> ModelInfo:
        """Parse a single model entry from the OpenRouter API response."""
        pricing = data.get("pricing", {})
        input_cost = None
        output_cost = None
        try:
            prompt_price = float(pricing.get("prompt", 0))
            completion_price = float(pricing.get("completion", 0))
            if prompt_price > 0:
                input_cost = prompt_price * 1_000_000
            if completion_price > 0:
                output_cost = completion_price * 1_000_000
        except (ValueError, TypeError):
            pass

        return ModelInfo(
            name=data["id"],
            display_name=data.get("name", data["id"]),
            description=data.get("description", ""),
            provider="openrouter",
            context_window=data.get("context_length"),
            is_default=data["id"] == _DEFAULT_MODEL,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid.

        If the live catalog has been fetched, checks against it.
        Without cache, accepts any non-empty string (OpenRouter validates at inference).
        """
        if not model_id:
            return False
        if OpenRouterProvider._models_cache is not None:
            return model_id in [m.name for m in OpenRouterProvider._models_cache]
        logger.debug(
            f"OpenRouter model '{model_id}' not validated against catalog (cache not loaded)"
        )
        return True

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatOpenAI instance routed through OpenRouter.

        Uses the OpenAI-compatible API with OpenRouter's base URL and
        hardcoded Lobster AI branding headers.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package not installed. "
                "Install with: uv pip install langchain-openai"
            )

        api_key = kwargs.pop("api_key", None) or os.environ.get(_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"{_ENV_VAR} not found in environment. "
                f"Set it with: export {_ENV_VAR}=sk-or-..."
            )

        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=_BASE_URL,
            default_headers=_DEFAULT_HEADERS,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        return (
            "Configure OpenRouter:\n\n"
            "1. Get API key from: https://openrouter.ai/keys\n"
            "2. Set environment variable:\n"
            "   export OPENROUTER_API_KEY=sk-or-...\n\n"
            "Or add to .env file:\n"
            "   OPENROUTER_API_KEY=sk-or-...\n\n"
            f"Default model: {_DEFAULT_MODEL}\n"
            "Model format: provider/model-name (e.g., anthropic/claude-sonnet-4-5)\n"
            "Browse 600+ models at: https://openrouter.ai/models\n"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OpenRouterProvider())
