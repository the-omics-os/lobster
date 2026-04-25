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
from typing import Any, List

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

    Attributes:
        KNOWN_MODELS: Static catalog of Claude models available via Bedrock
        DEFAULT_REGION: Default AWS region for Bedrock API
    """

    _default_context_window = 200_000

    # Static model catalog (single source of truth for Bedrock models)
    # All cross-region IDs for maximum availability
    KNOWN_MODELS = [
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
            is_default=True,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="global.anthropic.claude-opus-4-5-20251101-v1:0",
            display_name="Claude Opus 4.5 (Bedrock)",
            description="Claude 4.5 Opus - most capable model",
            provider="bedrock",
            context_window=200000,
            is_default=False,
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-6",
            display_name="Claude Sonnet 4.6 (Bedrock)",
            description="Claude 4.6 Sonnet - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="anthropic.claude-opus-4-6-v1",
            display_name="Claude Opus 4.6 (Bedrock)",
            description="Claude 4.6 Opus - 1M native context",
            provider="bedrock",
            context_window=1000000,
            is_default=False,
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
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

        Returns:
            str: Default model ID (Claude Sonnet 4 via Bedrock)

        Example:
            >>> provider = BedrockProvider()
            >>> model_id = provider.get_default_model()
            >>> # "anthropic.claude-sonnet-4-20250514-v1:0"
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
        Create a ChatBedrockConverse instance.

        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")
            temperature: Sampling temperature (0.0-2.0, default: 1.0)
            max_tokens: Maximum tokens in response (default: 4096)
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
