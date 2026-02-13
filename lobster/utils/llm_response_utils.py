"""Utilities for extracting content from LLM responses.

Provides provider-agnostic text extraction from LangChain AIMessage objects,
handling extended thinking blocks across all supported providers (Anthropic,
Bedrock, Gemini, Ollama, Azure).

Pattern follows client.py's content_blocks approach — see
core/client.py:_extract_from_content_blocks for the full reasoning-aware version.
"""

from typing import Any

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_llm_response(response: Any) -> str:
    """
    Extract text content from an LLM response, skipping thinking/reasoning blocks.

    Uses LangChain's content_blocks abstraction (preferred) with fallback to
    raw content parsing. Works across all providers.

    Args:
        response: LangChain AIMessage from llm.invoke()

    Returns:
        Extracted text string

    Raises:
        ValueError: If no text content could be extracted
    """
    # Strategy 1: content_blocks (preferred — normalized by LangChain)
    if hasattr(response, "content_blocks"):
        try:
            text_parts = []
            for block in response.content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
            if text_parts:
                return "\n\n".join(text_parts).strip()
        except Exception as e:
            logger.debug(f"content_blocks extraction failed: {e}, trying fallback")

    # Strategy 2: raw content parsing
    if hasattr(response, "content"):
        content = response.content

        # String content (simple case — Ollama, some Bedrock)
        if isinstance(content, str):
            return content.strip()

        # List of blocks (Anthropic, Gemini with thinking)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str) and block.strip():
                    text_parts.append(block.strip())
            if text_parts:
                return "\n\n".join(text_parts).strip()

    # Strategy 3: last resort
    raw = str(response.content) if hasattr(response, "content") else str(response)
    if raw.strip():
        return raw.strip()

    raise ValueError("Could not extract text from LLM response")
