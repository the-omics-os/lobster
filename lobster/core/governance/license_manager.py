"""
License manager — minimal stub.

All Lobster AI agents are free and open source. This module provides
tier resolution for display purposes and Cloud integration only.

Tier resolution order:
1. LOBSTER_SUBSCRIPTION_TIER env var (dev/testing override)
2. LOBSTER_CLOUD_KEY env var (implies premium — Cloud CLI auth)
3. Default: "free"
"""

import os
from typing import Any, Dict, List

TIER_ENV_VAR = "LOBSTER_SUBSCRIPTION_TIER"
CLOUD_KEY_ENV_VAR = "LOBSTER_CLOUD_KEY"


def get_current_tier() -> str:
    """Get current subscription tier."""
    env_tier = os.environ.get(TIER_ENV_VAR)
    if env_tier:
        return env_tier.lower()
    if os.environ.get(CLOUD_KEY_ENV_VAR):
        return "premium"
    return "free"


def get_custom_packages() -> List[str]:
    """Get list of authorized custom packages (always empty for local)."""
    return []


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled. Cloud features require LOBSTER_CLOUD_KEY."""
    if feature == "local_only":
        return True
    if feature in ("cloud_compute", "priority_processing"):
        return bool(os.environ.get(CLOUD_KEY_ENV_VAR))
    return False


def is_premium() -> bool:
    """Check if current tier is premium or higher."""
    return get_current_tier() in ("premium", "enterprise")


def is_enterprise() -> bool:
    """Check if current tier is enterprise."""
    return get_current_tier() == "enterprise"


def get_entitlement_status() -> Dict[str, Any]:
    """Get entitlement status summary for display."""
    tier = get_current_tier()
    source = "default"
    if os.environ.get(TIER_ENV_VAR):
        source = "environment"
    elif os.environ.get(CLOUD_KEY_ENV_VAR):
        source = "cloud_key"

    return {
        "tier": tier,
        "tier_display": tier.title(),
        "source": source,
        "valid": True,
        "custom_packages": [],
        "features": ["local_only"] if tier == "free" else ["local_only", "cloud_compute"],
    }


def load_entitlement() -> Dict[str, Any]:
    """Load entitlement (returns status dict for backward compatibility)."""
    return get_entitlement_status()
