"""
Nebius AI Studio provider implementation.

Nebius AI Studio (Token Factory) is an OpenAI-wire-compatible inference
platform hosting open-weight models (Qwen, DeepSeek, Kimi, GLM, MiniMax,
Hermes, NVIDIA Nemotron/Cosmos, Gemma). It speaks the OpenAI API protocol,
so no new LangChain packages are required — ChatOpenAI with a base_url
override is sufficient.

API key setup: https://nebius.com (Token Factory / AI Studio dashboard)

Architecture:
    - Implements ILLMProvider interface
    - Uses ChatOpenAI with base_url override (OpenAI-compatible API)
    - Static KNOWN_MODELS catalog carrying pricing (no live catalog fetch)
    - No new dependencies — uses ChatOpenAI, no default_headers/branding

Model naming convention: "org/model-name" (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")

Example:
    >>> provider = NebiusProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model("Qwen/Qwen3-30B-A3B-Instruct-2507")
"""

import os
from typing import Any, ClassVar, List

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

# Nebius AI Studio (Token Factory) API constants
_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
_ENV_VAR = "NEBIUS_API_KEY"
_DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class NebiusProvider(ILLMProvider):
    """
    Nebius AI Studio provider — OpenAI-compatible open-weight model host.

    Nebius AI Studio (Token Factory) accepts requests in OpenAI API format
    and serves open-weight models from Qwen, DeepSeek, Moonshot, Zhipu,
    MiniMax, Nous Research, NVIDIA, and Google. A single NEBIUS_API_KEY
    gives access to all catalog models.

    Model names use the "org/model-name" format:
        - "Qwen/Qwen3-30B-A3B-Instruct-2507"
        - "deepseek-ai/DeepSeek-V4-Pro"
        - "google/gemma-3-27b-it"

    Features:
        - Static KNOWN_MODELS catalog with pricing (no network on list)
        - No new dependencies — uses ChatOpenAI with base_url override
        - No branding headers

    Usage:
        >>> provider = NebiusProvider()
        >>> if not provider.is_configured():
        ...     print("Set NEBIUS_API_KEY in .env")
        >>> models = provider.list_models()
        >>> llm = provider.create_chat_model("Qwen/Qwen3-30B-A3B-Instruct-2507")
    """

    # Conservative window for Nebius model IDs absent from KNOWN_MODELS.
    # Open-weight windows vary from 8k to 1M; the inherited 200k default would
    # over-promise and invite context overflow on a small unlisted model.
    _default_context_window = 128_000

    # Static catalog — pricing (USD per 1M tokens) verified against Nebius.
    KNOWN_MODELS: ClassVar[List[ModelInfo]] = [
        ModelInfo(
            name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            display_name="Qwen3 235B A22B Instruct",
            description="Qwen3 flagship for general reasoning",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.20,
            output_cost_per_million=0.60,
        ),
        ModelInfo(
            name="Qwen/Qwen3-32B",
            display_name="Qwen3 32B",
            description="Qwen3 generalist: multilingual and coding",
            provider="nebius",
            context_window=40960,
            is_default=False,
            input_cost_per_million=0.10,
            output_cost_per_million=0.30,
        ),
        ModelInfo(
            name="Qwen/Qwen3-30B-A3B-Instruct-2507",
            display_name="Qwen3 30B A3B Instruct",
            description="Versatile 30B instruct model, 262k context",
            provider="nebius",
            context_window=262144,
            is_default=True,
            input_cost_per_million=0.10,
            output_cost_per_million=0.30,
        ),
        ModelInfo(
            name="Qwen/Qwen3-Next-80B-A3B-Thinking",
            display_name="Qwen3 Next 80B A3B Thinking",
            description="Qwen3 thinking-optimized for multi-step reasoning",
            provider="nebius",
            context_window=128000,
            is_default=False,
            input_cost_per_million=0.15,
            output_cost_per_million=1.20,
        ),
        ModelInfo(
            name="Qwen/Qwen3.5-397B-A17B",
            display_name="Qwen3.5 397B A17B",
            description="Qwen3.5 MoE multimodal flagship",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.60,
            output_cost_per_million=3.60,
        ),
        ModelInfo(
            name="moonshotai/Kimi-K2.6",
            display_name="Kimi K2.6",
            description="Moonshot multimodal agentic model",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.95,
            output_cost_per_million=4.00,
        ),
        ModelInfo(
            name="moonshotai/Kimi-K2.7-Code",
            display_name="Kimi K2.7 Code",
            description="Moonshot code-focused reasoning model",
            provider="nebius",
            context_window=8000,
            is_default=False,
            input_cost_per_million=0.95,
            output_cost_per_million=4.00,
        ),
        ModelInfo(
            name="zai-org/GLM-5.1",
            display_name="GLM-5.1",
            description="Zhipu flagship multimodal bilingual model",
            provider="nebius",
            context_window=202752,
            is_default=False,
            input_cost_per_million=1.40,
            output_cost_per_million=4.40,
        ),
        ModelInfo(
            name="zai-org/GLM-5.2",
            display_name="GLM-5.2",
            description="Zhipu latest flagship multimodal model",
            provider="nebius",
            context_window=8000,
            is_default=False,
            input_cost_per_million=1.40,
            output_cost_per_million=4.40,
        ),
        ModelInfo(
            name="deepseek-ai/DeepSeek-V4-Pro",
            display_name="DeepSeek V4 Pro",
            description="DeepSeek advanced reasoning, 1M context",
            provider="nebius",
            context_window=1048576,
            is_default=False,
            input_cost_per_million=1.75,
            output_cost_per_million=3.50,
        ),
        ModelInfo(
            name="MiniMaxAI/MiniMax-M3",
            display_name="MiniMax M3",
            description="MiniMax 428B MoE reasoning model",
            provider="nebius",
            context_window=8000,
            is_default=False,
            input_cost_per_million=0.30,
            output_cost_per_million=1.20,
        ),
        ModelInfo(
            name="MiniMaxAI/MiniMax-M2.5",
            display_name="MiniMax M2.5",
            description="MiniMax agentic coding model",
            provider="nebius",
            context_window=196608,
            is_default=False,
            input_cost_per_million=0.30,
            output_cost_per_million=1.20,
        ),
        ModelInfo(
            name="NousResearch/Hermes-4-70B",
            display_name="Hermes 4 70B",
            description="Nous Hermes-4 compact reasoning model",
            provider="nebius",
            context_window=131072,
            is_default=False,
            input_cost_per_million=0.13,
            output_cost_per_million=0.40,
        ),
        ModelInfo(
            name="NousResearch/Hermes-4-405B",
            display_name="Hermes 4 405B",
            description="Nous Hermes-4 hybrid reasoning model",
            provider="nebius",
            context_window=131072,
            is_default=False,
            input_cost_per_million=1.00,
            output_cost_per_million=3.00,
        ),
        ModelInfo(
            name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            display_name="Nemotron 3 Nano 30B A3B",
            description="NVIDIA Nemotron-3 nano, low cost",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.06,
            output_cost_per_million=0.24,
        ),
        ModelInfo(
            name="nvidia/Nemotron-3-Nano-Omni",
            display_name="Nemotron 3 Nano Omni",
            description="NVIDIA Nemotron-3 nano omni model",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.06,
            output_cost_per_million=0.24,
        ),
        ModelInfo(
            name="nvidia/Nemotron-3-Ultra-550b-a55b",
            display_name="Nemotron 3 Ultra 550B A55B",
            description="NVIDIA Nemotron-3 ultra, 1M context",
            provider="nebius",
            context_window=1048576,
            is_default=False,
            input_cost_per_million=1.00,
            output_cost_per_million=3.00,
        ),
        ModelInfo(
            name="nvidia/nemotron-3-super-120b-a12b",
            display_name="Nemotron 3 Super 120B A12B",
            description="NVIDIA Nemotron-3 super model",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.30,
            output_cost_per_million=0.90,
        ),
        ModelInfo(
            name="nvidia/Cosmos3-Super-Reasoner",
            display_name="Cosmos3 Super Reasoner",
            description="NVIDIA Cosmos3 reasoning model",
            provider="nebius",
            context_window=262144,
            is_default=False,
            input_cost_per_million=0.10,
            output_cost_per_million=0.30,
        ),
        ModelInfo(
            name="google/gemma-3-27b-it",
            display_name="Gemma 3 27B IT",
            description="Google Gemma 3 27B instruct model",
            provider="nebius",
            context_window=110000,
            is_default=False,
            input_cost_per_million=0.10,
            output_cost_per_million=0.30,
        ),
    ]

    @property
    def name(self) -> str:
        return "nebius"

    @property
    def display_name(self) -> str:
        return "Nebius AI Studio"

    def check_dependencies(self) -> None:
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("nebius", is_extra=True)
            raise ImportError(
                f"langchain-openai package not installed. Install with: {cmd}"
            )

    def is_configured(self) -> bool:
        """Check if NEBIUS_API_KEY is present and non-empty."""
        api_key = os.environ.get(_ENV_VAR)
        return bool(api_key and api_key.strip())

    def is_available(self) -> bool:
        """Check if Nebius is accessible (equals is_configured for cloud providers)."""
        return self.is_configured()

    def get_default_model(self) -> str:
        """Get the recommended default model."""
        return _DEFAULT_MODEL

    def list_models(self) -> List[ModelInfo]:
        """Return the static Nebius catalog (no network call)."""
        return list(self.KNOWN_MODELS)

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatOpenAI instance routed through Nebius AI Studio.

        Uses the OpenAI-compatible API with Nebius's base URL. No branding
        headers are attached.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from lobster.core.component_registry import get_install_command

            raise ImportError(
                "langchain-openai package not installed. "
                f"Install with: {get_install_command('langchain-openai')}"
            )

        api_key = kwargs.pop("api_key", None) or os.environ.get(_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"{_ENV_VAR} not found in environment. "
                f"Set it with: export {_ENV_VAR}=..."
            )

        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        return (
            "Configure Nebius AI Studio:\n\n"
            "1. Get an API key from Nebius AI Studio / Token Factory:\n"
            "   https://nebius.com\n"
            "2. Set environment variable:\n"
            "   export NEBIUS_API_KEY=...\n\n"
            "Or add to .env file:\n"
            "   NEBIUS_API_KEY=...\n\n"
            f"Default model: {_DEFAULT_MODEL}\n"
            "Model format: org/model-name (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)\n"
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(NebiusProvider())
