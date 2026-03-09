"""Shared ChatBedrockConverse construction for direct-AWS and gateway paths."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_bedrock_converse(
    *,
    model_id: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
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

    Returns:
        ChatBedrockConverse instance with full tool calling support
    """
    from langchain_aws import ChatBedrockConverse

    params: dict[str, Any] = {
        "model_id": model_id,
        "temperature": temperature,
    }

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

    logger.debug(f"Building ChatBedrockConverse: model={model_id}, gateway={client is not None}")

    return ChatBedrockConverse(**params)
