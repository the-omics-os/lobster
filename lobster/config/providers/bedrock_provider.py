"""
AWS Bedrock provider implementation.

This module provides AWS Bedrock integration for Lobster, enabling Claude models
via Amazon's Bedrock service with cross-region support and IAM credentials.

Key features:
- Static model catalog (Claude models via Bedrock)
- AWS IAM credential management
- Cross-region support (default: us-east-1)
- ChatBedrockConverse LangChain integration

Example:
    >>> from lobster.config.providers.bedrock_provider import BedrockProvider
    >>>
    >>> provider = BedrockProvider()
    >>> if provider.is_configured():
    ...     llm = provider.create_chat_model("anthropic.claude-sonnet-4-20250514-v1:0")
"""

import logging
import os
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


# === TEMPORARY FIX FOR CHATBEDROCKCONVERSE STREAMING BUG ===
# See: https://github.com/langchain-ai/langchain-aws/issues/239
# Issue: ChatBedrockConverse._should_stream() always returns False, preventing
#        LangGraph from detecting streaming support. This causes LangGraph to
#        buffer entire responses instead of yielding token-by-token.
class BedrockProvider(ILLMProvider):
    """
    AWS Bedrock provider for Claude models.

    Provides access to Claude models through Amazon Bedrock with IAM credential
    management and cross-region support.

    Configuration:
        AWS_BEDROCK_ACCESS_KEY: AWS access key ID
        AWS_BEDROCK_SECRET_ACCESS_KEY: AWS secret access key
        AWS_REGION: AWS region (optional, default: us-east-1)
        LOBSTER_BEDROCK_MODEL_ID: Default model ID (optional). Used when no
            explicit model, workspace config or global config chooses one.

    Attributes:
        KNOWN_MODELS: Static catalog of Claude models available via Bedrock
        DEFAULT_REGION: Default AWS region for Bedrock API
    """

    _default_context_window = 200_000

    # Static model catalog (single source of truth for Bedrock models)
    # ``us.`` profiles (geo / in-region) cost 10% more than ``global.``. Claude 4.x ``us.``
    # rates below are global x 1.1. READ from the AWS Bedrock price page (Geo and
    # In-region rows, 2026-11-09 feed): Opus 4.5/4.7/4.8/5, Sonnet 4.5, Opus 4.6 input.
    # INFERRED (Anthropic: premium applies to Opus 4.7, Haiku 4.5 and later; every
    # published geo row is exactly 1.1x): Haiku 4.5, Sonnet 4.6, Opus 4.6 output.
    # All cross-region IDs for maximum availability
    KNOWN_MODELS = [
        ModelInfo(
            name="global.anthropic.claude-sonnet-5-5",
            display_name="Claude Sonnet 5.5 (Bedrock global)",
            description="Claude Sonnet 5.5 - best balance of capability, speed and cost",
            provider="bedrock",
            context_window=1000000,
            is_default=True,
            input_cost_per_million=2.0,
            output_cost_per_million=10.0,
        ),
        ModelInfo(
            name="global.anthropic.claude-haiku-5-5",
            display_name="Claude Haiku 5.5 (Bedrock global)",
            description="Fast and low-cost; price shown for prompts up to 100K tokens",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=0.1,
            output_cost_per_million=0.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-5-5",
            display_name="Claude Opus 5.5 (Bedrock global)",
            description="Most capable Claude model for complex reasoning",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=4.0,
            output_cost_per_million=20.0,
        ),
        ModelInfo(
            name="global.anthropic.claude-fable-5-1",
            display_name="Claude Fable 5.1 (Bedrock global)",
            description="Frontier Claude model",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=10.0,
            output_cost_per_million=50.0,
        ),
        ModelInfo(
            name="global.openai.gpt-6.1-sol",
            display_name="GPT-6.1 Sol (Bedrock global)",
            description="OpenAI GPT-6.1 Sol; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=2.0,
            output_cost_per_million=10.0,
        ),
        ModelInfo(
            name="global.openai.gpt-6-sol",
            display_name="GPT-6 Sol (Bedrock global)",
            description="OpenAI GPT-6 Sol; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=2.0,
            output_cost_per_million=10.0,
        ),
        ModelInfo(
            name="global.openai.gpt-6-luna",
            display_name="GPT-6 Luna (Bedrock global)",
            description="OpenAI GPT-6 Luna, fast and low-cost; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=0.1,
            output_cost_per_million=0.5,
        ),
        ModelInfo(
            name="global.openai.gpt-6-astra",
            display_name="GPT-6 Astra (Bedrock global)",
            description="OpenAI GPT-6 Astra, most capable; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=10.0,
            output_cost_per_million=50.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-5-5",
            display_name="Claude Sonnet 5.5 (Bedrock US)",
            description="Claude Sonnet 5.5 - best balance of capability, speed and cost",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=2.2,
            output_cost_per_million=11.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-haiku-5-5",
            display_name="Claude Haiku 5.5 (Bedrock US)",
            description="Fast and low-cost; price shown for prompts up to 100K tokens",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=0.11,
            output_cost_per_million=0.55,
        ),
        ModelInfo(
            name="us.anthropic.claude-opus-5-5",
            display_name="Claude Opus 5.5 (Bedrock US)",
            description="Most capable Claude model for complex reasoning",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=4.4,
            output_cost_per_million=22.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-fable-5-1",
            display_name="Claude Fable 5.1 (Bedrock US)",
            description="Frontier Claude model",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=11.0,
            output_cost_per_million=55.0,
        ),
        ModelInfo(
            name="us.openai.gpt-6.1-sol",
            display_name="GPT-6.1 Sol (Bedrock US)",
            description="OpenAI GPT-6.1 Sol; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=2.2,
            output_cost_per_million=11.0,
        ),
        ModelInfo(
            name="us.openai.gpt-6-sol",
            display_name="GPT-6 Sol (Bedrock US)",
            description="OpenAI GPT-6 Sol; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=2.2,
            output_cost_per_million=11.0,
        ),
        ModelInfo(
            name="us.openai.gpt-6-luna",
            display_name="GPT-6 Luna (Bedrock US)",
            description="OpenAI GPT-6 Luna, fast and low-cost; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=0.11,
            output_cost_per_million=0.55,
        ),
        ModelInfo(
            name="us.openai.gpt-6-astra",
            display_name="GPT-6 Astra (Bedrock US)",
            description="OpenAI GPT-6 Astra, most capable; price shown for prompts up to 272K tokens",
            provider="bedrock",
            context_window=1050000,
            is_default=False,
            input_cost_per_million=11.0,
            output_cost_per_million=55.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-20250514-v1:0",
            display_name="Claude Sonnet 4 (Bedrock)",
            description="Claude 4 Sonnet - balanced quality and speed",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            display_name="Claude Sonnet 4.5 (Bedrock)",
            description="Claude 4.5 Sonnet - highest quality Sonnet",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=3.3,
            output_cost_per_million=16.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-4-5-20251101-v1:0",
            display_name="Claude Opus 4.5 (Bedrock)",
            description="Claude 4.5 Opus - most capable model",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-6",
            display_name="Claude Sonnet 4.6 (Bedrock)",
            description="Claude 4.6 Sonnet - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=3.3,
            output_cost_per_million=16.5,  # inferred
        ),
        ModelInfo(
            name="anthropic.claude-opus-4-6-v1",
            display_name="Claude Opus 4.6 (Bedrock)",
            description="Claude 4.6 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        # Cross-region inference profiles.
        #
        # These are not duplicates of the entries above. Bedrock REQUIRES a 'us.'/'global.'
        # prefixed inference-profile ID for on-demand invocation of current Claude models:
        #
        #   ValidationException: Invocation of model ID anthropic.claude-opus-4-8 with
        #   on-demand throughput isn't supported. Retry your request with the ID or ARN of
        #   an inference profile that contains this model.
        #
        # This catalog is also the pricing source of truth (see
        # `get_all_models_with_pricing`), so a missing profile ID can leave real
        # production calls without a pricing entry.
        # 'us.' and 'global.' variants are priced identically.
        #
        # Keep the provider catalog's model prices aligned with current list rates.
        # A rate that is wrong in the expensive direction is not "conservative":
        # it silently misprices usage. Provider model catalogs are covered by a
        # pricing consistency test.
        ModelInfo(
            name="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            display_name="Claude Haiku 4.5 (Bedrock)",
            description="Claude 4.5 Haiku - fastest, lowest cost",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=1.1,
            output_cost_per_million=5.5,  # inferred
        ),
        ModelInfo(
            name="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            display_name="Claude Haiku 4.5 (Bedrock, global)",
            description="Claude 4.5 Haiku - fastest, lowest cost",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=1.0,
            output_cost_per_million=5.0,
        ),
        ModelInfo(
            name="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            display_name="Claude Sonnet 4.5 (Bedrock, global)",
            description="Claude 4.5 Sonnet - highest quality Sonnet",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-opus-4-6-v1",
            display_name="Claude Opus 4.6 (Bedrock, us)",
            description="Claude 4.6 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.5,
            output_cost_per_million=27.5,  # output inferred
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-4-6-v1",
            display_name="Claude Opus 4.6 (Bedrock, global)",
            description="Claude 4.6 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-opus-4-8",
            display_name="Claude Opus 4.8 (Bedrock, us)",
            description="Claude 4.8 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.5,
            output_cost_per_million=27.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-4-8",
            display_name="Claude Opus 4.8 (Bedrock, global)",
            description="Claude 4.8 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-opus-5",
            display_name="Claude Opus 5 (Bedrock, us)",
            description="Claude 5 Opus - most capable model",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.5,
            output_cost_per_million=27.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-5",
            display_name="Claude Opus 5 (Bedrock, global)",
            description="Claude 5 Opus - most capable model",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        # Opus 4.7 completes the generation; the reference prices it identically, and
        # omitting it would leave the same unpriced hole that Opus 4.8/5 had.
        ModelInfo(
            name="us.anthropic.claude-opus-4-7",
            display_name="Claude Opus 4.7 (Bedrock, us)",
            description="Claude 4.7 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.5,
            output_cost_per_million=27.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-4-7",
            display_name="Claude Opus 4.7 (Bedrock, global)",
            description="Claude 4.7 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=5.0,
            output_cost_per_million=25.0,
        ),
        # Opus 4.5 'us.' profile: the 'global.' variant was already present above, but
        # the 'us.' one was not -- and the reference lists both.
        ModelInfo(
            name="us.anthropic.claude-opus-4-5-20251101-v1:0",
            display_name="Claude Opus 4.5 (Bedrock, us)",
            description="Claude 4.5 Opus - most capable model",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=5.5,
            output_cost_per_million=27.5,
        ),
        ModelInfo(
            name="global.anthropic.claude-sonnet-4-6",
            display_name="Claude Sonnet 4.6 (Bedrock, global)",
            description="Claude 4.6 Sonnet - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
    ]

    DEFAULT_REGION = "us-east-1"

    @property
    def name(self) -> str:
        """Return provider identifier."""
        return "bedrock"

    @property
    def display_name(self) -> str:
        """Return human-friendly provider name."""
        return "AWS Bedrock"

    def check_dependencies(self) -> None:
        try:
            import langchain_aws  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("bedrock", is_extra=True)
            raise ImportError(
                f"langchain-aws package not installed. Install with: {cmd}"
            )

    def is_configured(self) -> bool:
        """
        Check if AWS Bedrock credentials are configured.

        Returns:
            bool: True if both AWS_BEDROCK_ACCESS_KEY and
                  AWS_BEDROCK_SECRET_ACCESS_KEY are set

        Example:
            >>> provider = BedrockProvider()
            >>> if not provider.is_configured():
            ...     print("Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY")
        """
        return bool(
            os.environ.get("AWS_BEDROCK_ACCESS_KEY")
            and os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")
        )

    def is_available(self) -> bool:
        """
        Check if AWS Bedrock is accessible.

        For Bedrock, this is equivalent to is_configured() since we cannot
        easily test connectivity without making an API call.

        Returns:
            bool: True if credentials are configured

        Note:
            Full connectivity test would require boto3 call, which is expensive.
            We rely on credential presence as proxy for availability.
        """
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        """
        List all available Bedrock models.

        Returns:
            List[ModelInfo]: Static catalog of Claude models via Bedrock

        Example:
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: {model.description}")
        """
        return self.KNOWN_MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the default Bedrock model.

        ``LOBSTER_BEDROCK_MODEL_ID`` wins when set, so a deployment can pin
        its model without a config file. Otherwise the catalog default.

        Returns:
            str: Default model ID

        Example:
            >>> provider = BedrockProvider()
            >>> model_id = provider.get_default_model()
            >>> # "global.anthropic.claude-sonnet-5-5"
        """
        env_model = (os.environ.get("LOBSTER_BEDROCK_MODEL_ID") or "").strip()
        if env_model:
            logger.debug(f"Using LOBSTER_BEDROCK_MODEL_ID: {env_model}")
            return env_model

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
        Create a ChatBedrockConverse instance.

        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")
            temperature: Sampling temperature (None = model default)
            max_tokens: Maximum tokens in response. Default None sends the model's
                documented output bound (see ``lobster.config.output_bound``).
            **kwargs: Additional parameters:
                - region_name: AWS region (default: us-east-1)
                - aws_access_key_id: Override AWS access key
                - aws_secret_access_key: Override AWS secret key
                - additional_model_request_fields: Extended thinking config
                - thinking: Extended thinking config (auto-wrapped)

        Returns:
            ChatBedrockConverse: LangChain chat model instance

        Raises:
            ImportError: If langchain-aws not installed
            ValueError: If model_id is invalid or credentials missing
        """
        # Import check
        try:
            from langchain_aws import ChatBedrockConverse  # noqa: F401
        except ImportError:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("bedrock", is_extra=True)
            raise ImportError(
                f"langchain-aws package not installed. Install with: {cmd}"
            )

        # Check credentials
        if not self.is_configured():
            raise ValueError(
                "AWS Bedrock credentials not configured. "
                "Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY."
            )

        from lobster.config.providers.bedrock_builder import build_bedrock_converse

        region = kwargs.pop("region_name", None) or os.environ.get(
            "AWS_REGION", self.DEFAULT_REGION
        )
        aws_access_key = kwargs.pop("aws_access_key_id", None) or os.environ.get(
            "AWS_BEDROCK_ACCESS_KEY"
        )
        aws_secret_key = kwargs.pop("aws_secret_access_key", None) or os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY"
        )

        if not (aws_access_key and aws_secret_key):
            raise ValueError(
                "AWS credentials required but not found. "
                "Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY."
            )

        # Handle thinking parameter specially for ChatBedrockConverse
        if "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")
            additional = kwargs.get("additional_model_request_fields", {}) or {}
            additional["thinking"] = thinking_config
            kwargs["additional_model_request_fields"] = additional

        # Generous timeouts for multi-tool agent turns
        # Default boto3 read_timeout (60s) is too short when agents make
        # multiple external API calls (Open Targets, ChEMBL, PubChem) in one turn
        try:
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                read_timeout=600,
                connect_timeout=10,
                retries={"max_attempts": 3},
            )
        except ImportError:
            boto_config = None

        logger.debug(
            f"Creating ChatBedrockConverse with model={model_id}, "
            f"region={region}, temperature={temperature}"
        )

        return build_bedrock_converse(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            config=boto_config,
            **kwargs,
        )

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Bedrock.

        Returns:
            str: Configuration instructions

        Example:
            >>> print(provider.get_configuration_help())
        """
        return (
            "Configure AWS Bedrock by setting environment variables:\n\n"
            "Required:\n"
            "  AWS_BEDROCK_ACCESS_KEY=your_access_key_id\n"
            "  AWS_BEDROCK_SECRET_ACCESS_KEY=your_secret_access_key\n\n"
            "Optional:\n"
            "  AWS_REGION=us-east-1  # Default region for Bedrock API\n\n"
            "Note: Model availability varies by region. Claude Sonnet 4 is available in us-east-1."
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(BedrockProvider())
