"""Shared ChatBedrockConverse construction for direct-AWS and gateway paths."""

import logging
import os
from typing import Any, Optional

from lobster.config.output_bound import above_thinking_budget, default_output_bound

logger = logging.getLogger(__name__)


def build_bedrock_converse(
    *,
    model_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    client: Any = None,
    bedrock_client: Any = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    config: Any = None,
    additional_model_request_fields: Optional[dict] = None,
    **kwargs: Any,
) -> Any:
    """Build a ChatBedrockConverse instance.

    Two modes:
    - Gateway path: pass `client` (runtime) + `bedrock_client` (control stub)
    - Direct AWS path: pass AWS credentials + region

    ``max_tokens`` is the output-token limit. An explicit value is forwarded
    unchanged. When it is ``None`` (the default) the model's documented output
    bound is sent (``lobster.config.output_bound``): Bedrock's own default, which
    applies when nothing is sent, is far below the model maximum and cut long
    answers off mid-sentence.

    Returns:
        ChatBedrockConverse instance with full tool calling support
    """
    from langchain_aws import ChatBedrockConverse

    params: dict[str, Any] = {"model_id": model_id}

    # Send a temperature only when set; newer models accept only their default.
    if temperature is not None:
        params["temperature"] = temperature

    # Always send an output-token bound. Without one Bedrock applies a default far
    # below the model maximum. The default is the documented model maximum and
    # stays above an extended-thinking budget.
    if max_tokens is None:
        max_tokens = above_thinking_budget(
            default_output_bound(model_id), additional_model_request_fields
        )
    params["max_tokens"] = max_tokens

    # Gateway path: inject pre-built clients
    if client is not None:
        params["client"] = client
    if bedrock_client is not None:
        params["bedrock_client"] = bedrock_client

    # Direct AWS path: pass credentials
    if region_name:
        params["region_name"] = region_name
    if aws_access_key_id:
        params["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        params["aws_secret_access_key"] = aws_secret_access_key

    # Extended thinking and other model-specific fields
    if additional_model_request_fields:
        params["additional_model_request_fields"] = additional_model_request_fields

    # Boto config for timeouts
    if config:
        params["config"] = config

    # Pass through remaining kwargs
    params.update(kwargs)

    logger.debug(
        f"Building ChatBedrockConverse: model={model_id}, gateway={client is not None}"
    )

    if os.getenv("LOBSTER_BEDROCK_PROMPT_CACHE") == "1" and "anthropic." in model_id:
        from lobster.config.providers.caching_bedrock_client import (
            CachingChatBedrockConverse,
        )

        return CachingChatBedrockConverse(**params)

    return ChatBedrockConverse(**params)
