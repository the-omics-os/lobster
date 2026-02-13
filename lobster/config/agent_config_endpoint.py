"""
Omics-OS endpoint client for LLM-assisted agent configuration.

This module provides the suggest_agents() function that calls the Omics-OS
hosted endpoint to get agent suggestions based on a user's workflow description.
The endpoint is hosted by Omics-OS (not the user's LLM provider), so no API key
is required from the user.

Part of Phase 8 (CLI & Init Flow) - CONF-08 requirement.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, TypedDict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OMICS_OS_SUGGEST_ENDPOINT = os.environ.get(
    "OMICS_OS_SUGGEST_ENDPOINT",
    "https://api.omics-os.com/v1/suggest-agents",
)

# 30 second timeout prevents blocking init flow (user decision from CONTEXT.md)
TIMEOUT_SECONDS = 30


# =============================================================================
# Response Types
# =============================================================================


class AgentSuggestion(TypedDict):
    """Response from the Omics-OS suggest-agents endpoint.

    Attributes:
        agents: List of suggested agent names (e.g., ["research_agent", "transcriptomics_expert"])
        reasoning: Per-agent reasoning explaining why each was suggested
    """

    agents: list[str]
    reasoning: dict[str, str]


# =============================================================================
# Main Function
# =============================================================================


def suggest_agents(description: str) -> Optional[AgentSuggestion]:
    """
    Call Omics-OS endpoint to suggest agents based on workflow description.

    This function calls the Omics-OS hosted endpoint (not user's LLM provider)
    to get agent suggestions. No API key is required from the user.

    Args:
        description: User's description of their analysis workflow and data types.
                    Example: "I have single-cell RNA-seq data and want to identify
                    cell types and perform differential expression analysis."

    Returns:
        AgentSuggestion dict with:
            - agents: List of suggested agent names
            - reasoning: Dict mapping agent names to 1-liner explanations
        Returns None on ANY error (timeout, network error, HTTP error, invalid response).
        Callers should fall back to manual selection when None is returned.

    Example:
        >>> result = suggest_agents("I need to analyze scRNA-seq data from GEO")
        >>> if result is None:
        ...     # Fall back to manual selection
        ...     print("Can't reach Omics-OS service. Falling back to manual selection.")
        ... else:
        ...     print(f"Suggested agents: {result['agents']}")
        ...     for agent, reason in result['reasoning'].items():
        ...         print(f"  {agent}: {reason}")
    """
    # Lazy import httpx - return None if not installed
    try:
        import httpx
    except ImportError:
        logger.warning(
            "httpx not installed. Cannot call Omics-OS endpoint. "
            "Install with: pip install httpx"
        )
        return None

    if not description or not description.strip():
        logger.warning("Empty description provided to suggest_agents()")
        return None

    try:
        response = httpx.post(
            OMICS_OS_SUGGEST_ENDPOINT,
            json={"description": description.strip()},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        data = response.json()

        # Validate response format
        if not isinstance(data, dict):
            logger.warning(
                "Invalid response format from Omics-OS endpoint: expected dict, got %s",
                type(data).__name__,
            )
            return None

        agents = data.get("agents")
        reasoning = data.get("reasoning")

        if not isinstance(agents, list):
            logger.warning(
                "Invalid 'agents' field in response: expected list, got %s",
                type(agents).__name__ if agents is not None else "None",
            )
            return None

        if not isinstance(reasoning, dict):
            logger.warning(
                "Invalid 'reasoning' field in response: expected dict, got %s",
                type(reasoning).__name__ if reasoning is not None else "None",
            )
            return None

        # Validate agents are strings
        if not all(isinstance(a, str) for a in agents):
            logger.warning("Invalid agent names in response: expected all strings")
            return None

        return AgentSuggestion(agents=agents, reasoning=reasoning)

    except httpx.TimeoutException:
        logger.warning(
            "Timeout calling Omics-OS endpoint (exceeded %ds). "
            "Falling back to manual selection.",
            TIMEOUT_SECONDS,
        )
        return None

    except httpx.RequestError as e:
        logger.warning(
            "Network error calling Omics-OS endpoint: %s. "
            "Falling back to manual selection.",
            str(e),
        )
        return None

    except httpx.HTTPStatusError as e:
        logger.warning(
            "HTTP error from Omics-OS endpoint: %s. Falling back to manual selection.",
            str(e),
        )
        return None

    except ValueError as e:
        # JSON decode error
        logger.warning(
            "Invalid JSON response from Omics-OS endpoint: %s. "
            "Falling back to manual selection.",
            str(e),
        )
        return None

    except Exception as e:
        # Catch-all for unexpected errors
        logger.warning(
            "Unexpected error calling Omics-OS endpoint: %s. "
            "Falling back to manual selection.",
            str(e),
        )
        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "suggest_agents",
    "AgentSuggestion",
    "OMICS_OS_SUGGEST_ENDPOINT",
    "TIMEOUT_SECONDS",
]
