"""Deprecated agent aliases for backwards compatibility.

These wrappers emit deprecation warnings and delegate to proteomics_expert.
They will be removed in a future version.
"""

import logging
import warnings

from lobster.agents.proteomics.proteomics_expert import proteomics_expert

logger = logging.getLogger(__name__)


def ms_proteomics_alias(
    data_manager,
    callback_handler=None,
    agent_name: str = "ms_proteomics_expert_agent",
    handoff_tools: list = None,
):
    """DEPRECATED: Routes to proteomics_expert.

    This agent is deprecated and will be removed in a future version.
    Use proteomics_expert instead, which handles both mass spectrometry
    and affinity proteomics analysis.

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler
        agent_name: Agent name (preserved for compatibility)
        handoff_tools: Optional handoff tools (renamed from delegation_tools)

    Returns:
        Agent instance (proteomics_expert)
    """
    warnings.warn(
        "ms_proteomics_expert is deprecated and will be removed in a future version. "
        "Use proteomics_expert instead, which handles both mass spectrometry and affinity proteomics.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "DEPRECATED: ms_proteomics_expert called. Routing to proteomics_expert."
    )

    return proteomics_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name=agent_name,  # Preserve original name for logging
        delegation_tools=handoff_tools,
        force_platform_type="mass_spec",  # Force MS mode for backward compatibility
    )


def affinity_proteomics_alias(
    data_manager,
    callback_handler=None,
    agent_name: str = "affinity_proteomics_expert_agent",
    handoff_tools: list = None,
):
    """DEPRECATED: Routes to proteomics_expert.

    This agent is deprecated and will be removed in a future version.
    Use proteomics_expert instead, which handles both mass spectrometry
    and affinity proteomics analysis.

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler
        agent_name: Agent name (preserved for compatibility)
        handoff_tools: Optional handoff tools (renamed from delegation_tools)

    Returns:
        Agent instance (proteomics_expert)
    """
    warnings.warn(
        "affinity_proteomics_expert is deprecated and will be removed in a future version. "
        "Use proteomics_expert instead, which handles both mass spectrometry and affinity proteomics.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "DEPRECATED: affinity_proteomics_expert called. Routing to proteomics_expert."
    )

    return proteomics_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name=agent_name,  # Preserve original name for logging
        delegation_tools=handoff_tools,
        force_platform_type="affinity",  # Force affinity mode for backward compatibility
    )


__all__ = ["ms_proteomics_alias", "affinity_proteomics_alias"]
