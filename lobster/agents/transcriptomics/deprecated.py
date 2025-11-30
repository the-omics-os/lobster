"""Deprecated agent aliases for backwards compatibility.

These wrappers emit deprecation warnings and delegate to transcriptomics_expert.
They will be removed in a future version.
"""

import logging
import warnings

from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert

logger = logging.getLogger(__name__)


def singlecell_alias(
    data_manager,
    callback_handler=None,
    agent_name: str = "singlecell_expert_agent",
    delegation_tools: list = None,
):
    """DEPRECATED: Routes to transcriptomics_expert.

    This agent is deprecated and will be removed in a future version.
    Use transcriptomics_expert instead, which handles both single-cell
    and bulk RNA-seq analysis.

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler
        agent_name: Agent name (preserved for compatibility)
        delegation_tools: Optional delegation tools

    Returns:
        Agent instance (transcriptomics_expert)
    """
    warnings.warn(
        "singlecell_expert is deprecated and will be removed in a future version. "
        "Use transcriptomics_expert instead, which handles both single-cell and bulk RNA-seq.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "DEPRECATED: singlecell_expert called. Routing to transcriptomics_expert."
    )

    return transcriptomics_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name=agent_name,  # Preserve original name for logging
        delegation_tools=delegation_tools,
    )


def bulk_alias(
    data_manager,
    callback_handler=None,
    agent_name: str = "bulk_rnaseq_expert_agent",
    delegation_tools: list = None,
):
    """DEPRECATED: Routes to transcriptomics_expert.

    This agent is deprecated and will be removed in a future version.
    Use transcriptomics_expert instead, which handles both single-cell
    and bulk RNA-seq analysis.

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler
        agent_name: Agent name (preserved for compatibility)
        delegation_tools: Optional delegation tools

    Returns:
        Agent instance (transcriptomics_expert)
    """
    warnings.warn(
        "bulk_rnaseq_expert is deprecated and will be removed in a future version. "
        "Use transcriptomics_expert instead, which handles both single-cell and bulk RNA-seq.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "DEPRECATED: bulk_rnaseq_expert called. Routing to transcriptomics_expert."
    )

    return transcriptomics_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name=agent_name,  # Preserve original name for logging
        delegation_tools=delegation_tools,
    )


__all__ = ["singlecell_alias", "bulk_alias"]
