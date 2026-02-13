"""
Subscription tier definitions for feature gating.

All official Lobster agents are free and open source. The tier system exists
for Omics-OS Cloud features and custom enterprise packages.

This module defines the three-tier subscription model:
- FREE: All official agents (11 agents, all open source)
- PREMIUM: Cloud features, priority support
- ENTERPRISE: Custom packages, dedicated compute, SLA

Tier Configuration:
- agents: List of agent names available at this tier
- restricted_handoffs: Dict mapping agent_name -> list of blocked handoff targets
- features: List of feature flags enabled at this tier
- compute_limits: Resource constraints for this tier

Runtime Tier Checking:
- TierRestrictedError: Exception raised when tier is insufficient
- check_tier_access(): Function to validate tier at agent factory invocation
"""

from typing import Any, Dict, List, Optional

# =============================================================================
# SUBSCRIPTION TIER DEFINITIONS
# =============================================================================
#
# DEPRECATION NOTE: The 'agents' lists below are maintained for backward
# compatibility. New code should use is_agent_available() which checks
# AgentRegistryConfig.tier_requirement dynamically via entry points.
# New agents only need tier_requirement in their AGENT_CONFIG - no need
# to update these hardcoded lists.
#

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "display_name": "Free",
        "agents": [
            # All official agents are free and open source (11 agents)
            # Note: Names must match AGENT_REGISTRY keys exactly
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",  # Unified single-cell + bulk RNA-seq
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            "metadata_assistant",  # Publication queue filtering, ID mapping
            "proteomics_expert",  # MS proteomics (DDA/DIA)
            "genomics_expert",  # Genomics analysis and interpretation
            "machine_learning_expert_agent",
            "protein_structure_visualization_expert_agent",
        ],
        "restricted_handoffs": {},  # No restrictions — all agents free
        "features": [
            "local_only",
            "community_support",
        ],
        "compute_limits": {
            "queries_per_day": 50,
            "max_datasets": 5,
        },
    },
    "premium": {
        "display_name": "Premium",
        "agents": [
            # All official agents (same as free — premium adds cloud features)
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            "metadata_assistant",
            "proteomics_expert",
            "genomics_expert",
            "machine_learning_expert_agent",
            "protein_structure_visualization_expert_agent",
        ],
        "restricted_handoffs": {},  # No restrictions
        "features": [
            "local_only",
            "cloud_compute",
            "email_support",
            "priority_processing",
        ],
        "compute_limits": {
            "queries_per_day": 500,
            "max_datasets": 50,
        },
    },
    "enterprise": {
        "display_name": "Enterprise",
        "agents": ["*"],  # Wildcard: all agents including custom packages
        "custom_packages": True,  # Allow lobster-custom-{customer} packages
        "restricted_handoffs": {},  # No restrictions
        "features": [
            "local_only",
            "cloud_compute",
            "dedicated_compute",
            "sla",
            "custom_development",
            "priority_support",
        ],
        "compute_limits": {
            "queries_per_day": None,  # Unlimited
            "max_datasets": None,  # Unlimited
        },
    },
}

# =============================================================================
# TIER HELPER FUNCTIONS
# =============================================================================


def get_tier_config(tier: str) -> Dict[str, Any]:
    """
    Get the full configuration for a subscription tier.

    Args:
        tier: Tier name (free, premium, enterprise)

    Returns:
        Tier configuration dict, defaults to 'free' if tier not found
    """
    return SUBSCRIPTION_TIERS.get(tier.lower(), SUBSCRIPTION_TIERS["free"])


def get_tier_agents(tier: str) -> List[str]:
    """
    Get list of agents available for a subscription tier.

    Args:
        tier: Tier name (free, premium, enterprise)

    Returns:
        List of agent names available at this tier
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("agents", [])


def get_restricted_handoffs(tier: str, agent_name: str) -> List[str]:
    """
    Get list of handoffs restricted for an agent in a given tier.

    Args:
        tier: Tier name (free, premium, enterprise)
        agent_name: Name of the agent to check restrictions for

    Returns:
        List of agent names that cannot be handed off to
    """
    tier_config = get_tier_config(tier)
    restrictions = tier_config.get("restricted_handoffs", {})
    return restrictions.get(agent_name, [])


def is_agent_available(agent_name: str, tier: str) -> bool:
    """
    Check if an agent is available for a subscription tier.

    Uses dynamic tier checking based on AgentRegistryConfig.tier_requirement.
    Falls back to hardcoded lists for unknown agents (backward compatibility).

    Args:
        agent_name: Name of the agent to check
        tier: Tier name (free, premium, enterprise)

    Returns:
        True if agent is available at this tier
    """
    # First, try dynamic checking via component_registry
    try:
        from lobster.core.component_registry import component_registry

        agent_config = component_registry.get_agent(agent_name)

        if agent_config is not None:
            required_tier = getattr(agent_config, "tier_requirement", "free")
            # Enterprise tier has access to all agents
            if tier.lower() == "enterprise":
                return True
            return is_tier_at_least(tier, required_tier)
    except Exception:
        pass  # Fall through to static check

    # Fallback: check hardcoded SUBSCRIPTION_TIERS (backward compatibility)
    tier_config = get_tier_config(tier)
    agents = tier_config.get("agents", [])
    # Wildcard "*" means all agents are available (enterprise tier)
    return "*" in agents or agent_name in agents


def is_handoff_allowed(source_agent: str, target_agent: str, tier: str) -> bool:
    """
    Check if a handoff from source to target agent is allowed at given tier.

    Args:
        source_agent: Agent initiating the handoff
        target_agent: Agent being handed off to
        tier: Subscription tier

    Returns:
        True if handoff is allowed
    """
    restricted = get_restricted_handoffs(tier, source_agent)
    return target_agent not in restricted


def get_tier_features(tier: str) -> List[str]:
    """
    Get list of features enabled for a subscription tier.

    Args:
        tier: Tier name

    Returns:
        List of feature flags
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("features", [])


def get_compute_limits(tier: str) -> Dict[str, Optional[int]]:
    """
    Get compute limits for a subscription tier.

    Args:
        tier: Tier name

    Returns:
        Dict of limit_name -> limit_value (None means unlimited)
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("compute_limits", {})


def get_all_tiers() -> List[str]:
    """Get list of all available tier names."""
    return list(SUBSCRIPTION_TIERS.keys())


def get_tier_display_name(tier: str) -> str:
    """Get human-readable display name for a tier."""
    tier_config = get_tier_config(tier)
    return tier_config.get("display_name", tier.title())


# =============================================================================
# TIER COMPARISON UTILITIES
# =============================================================================

# Tier hierarchy for comparison (higher index = higher tier)
_TIER_HIERARCHY = ["free", "premium", "enterprise"]


def compare_tiers(tier1: str, tier2: str) -> int:
    """
    Compare two tiers.

    Args:
        tier1: First tier name
        tier2: Second tier name

    Returns:
        -1 if tier1 < tier2, 0 if equal, 1 if tier1 > tier2
    """
    idx1 = (
        _TIER_HIERARCHY.index(tier1.lower()) if tier1.lower() in _TIER_HIERARCHY else 0
    )
    idx2 = (
        _TIER_HIERARCHY.index(tier2.lower()) if tier2.lower() in _TIER_HIERARCHY else 0
    )

    if idx1 < idx2:
        return -1
    elif idx1 > idx2:
        return 1
    return 0


def is_tier_at_least(current_tier: str, required_tier: str) -> bool:
    """
    Check if current tier meets or exceeds required tier.

    Args:
        current_tier: User's current tier
        required_tier: Minimum required tier

    Returns:
        True if current_tier >= required_tier
    """
    return compare_tiers(current_tier, required_tier) >= 0


# =============================================================================
# DYNAMIC TIER CHECKING (Entry Point Based)
# =============================================================================


def is_agent_available_dynamic(agent_name: str, user_tier: str) -> bool:
    """
    Check if an agent is available for a user's subscription tier (dynamic).

    This function queries the agent's tier_requirement field from
    AgentRegistryConfig, which is discovered via entry points.

    Args:
        agent_name: Name of the agent to check
        user_tier: User's subscription tier (free, premium, enterprise)

    Returns:
        True if the agent is available at the user's tier
    """
    from lobster.core.component_registry import component_registry

    agent_config = component_registry.get_agent(agent_name)
    if agent_config is None:
        return False

    required_tier = getattr(agent_config, "tier_requirement", "free")

    # Enterprise tier has access to all agents
    if user_tier.lower() == "enterprise":
        return True

    return is_tier_at_least(user_tier, required_tier)


def get_available_agents_dynamic(user_tier: str) -> List[str]:
    """
    Get list of all agents available at a given tier (dynamic).

    This function discovers all agents via entry points and filters
    them based on their tier_requirement field.

    Args:
        user_tier: User's subscription tier (free, premium, enterprise)

    Returns:
        List of agent names available at the specified tier
    """
    from lobster.core.component_registry import component_registry

    available = []
    for name, config in component_registry.list_agents().items():
        required_tier = getattr(config, "tier_requirement", "free")
        if is_tier_at_least(user_tier, required_tier):
            available.append(name)
    return available


# =============================================================================
# RUNTIME TIER CHECKING
# =============================================================================


class TierRestrictedError(Exception):
    """Raised when user's subscription tier doesn't allow access to a feature.

    This exception provides clear guidance for upgrading, including:
    - The specific agent that requires a higher tier
    - The user's current tier
    - URL to pricing page
    - CLI command for activation

    Attributes:
        agent_name: Name of the restricted agent
        required_tier: Minimum tier required for access
        current_tier: User's current subscription tier
    """

    def __init__(self, agent_name: str, required_tier: str, current_tier: str):
        self.agent_name = agent_name
        self.required_tier = required_tier
        self.current_tier = current_tier

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        """Build a user-friendly error message with upgrade guidance."""
        return (
            f"Agent '{self.agent_name}' requires {self.required_tier.title()} tier.\n"
            f"Your current tier: {self.current_tier.title()}.\n\n"
            f"Upgrade at https://omics-os.com/pricing\n"
            f"Or activate with: lobster activate <your-key>"
        )


def check_tier_access(agent_name: str, required_tier: str = "free") -> None:
    """
    Check if the current user's tier allows access to a feature.

    This function is designed to be called at agent factory invocation time
    (not at discovery or graph creation) to enforce runtime tier gating.

    The check passes silently if access is granted. If access is denied,
    a TierRestrictedError is raised with clear upgrade guidance.

    Special cases:
    - Enterprise tier bypasses all checks (always returns without error)
    - Free tier requirement always passes regardless of user tier

    Args:
        agent_name: Name of the agent being accessed (for error message)
        required_tier: Minimum tier required ("free", "premium", "enterprise")

    Raises:
        TierRestrictedError: If user's tier is insufficient for access

    Example:
        >>> from lobster.config.subscription_tiers import check_tier_access
        >>> # In premium agent factory:
        >>> check_tier_access("proteomics_expert", required_tier="premium")
        >>> # Raises TierRestrictedError if user is on free tier
    """
    # Lazy import to avoid circular dependencies
    from lobster.core.license_manager import get_current_tier

    current_tier = get_current_tier()

    # Enterprise tier bypasses all tier checks (per STATE.md decision)
    if current_tier.lower() == "enterprise":
        return

    # Check if current tier meets the requirement
    if is_tier_at_least(current_tier, required_tier):
        return

    # Access denied - raise with helpful message
    raise TierRestrictedError(agent_name, required_tier, current_tier)
