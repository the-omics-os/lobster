"""
License manager for premium entitlement handling.

This module manages license/entitlement files that control access to
premium and custom features. Entitlements are stored in ~/.lobster/license.json
and are issued by the Omics-OS license service during activation.

Entitlement File Structure:
{
    "tier": "premium",
    "customer_id": "cust_abc123",
    "issued_at": "2024-12-01T00:00:00Z",
    "expires_at": "2025-12-01T00:00:00Z",
    "custom_packages": ["lobster-custom-databiomix"],
    "features": ["cloud_compute", "priority_support"],
    "signature": "base64_encoded_signature..."
}

Usage:
    from lobster.core.license_manager import (
        load_entitlement,
        get_current_tier,
        is_feature_enabled,
    )

    entitlement = load_entitlement()
    tier = get_current_tier()
    if is_feature_enabled("cloud_compute"):
        # Enable cloud features
        pass
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Default location for license/entitlement file
DEFAULT_LICENSE_PATH = Path.home() / ".lobster" / "license.json"

# Environment variable to override license path (useful for testing)
LICENSE_PATH_ENV_VAR = "LOBSTER_LICENSE_PATH"

# Environment variable to set tier directly (for development/testing)
TIER_ENV_VAR = "LOBSTER_SUBSCRIPTION_TIER"

# Default entitlement for free tier users
DEFAULT_ENTITLEMENT: Dict[str, Any] = {
    "tier": "free",
    "customer_id": None,
    "issued_at": None,
    "expires_at": None,
    "custom_packages": [],
    "features": ["local_only", "community_support"],
    "valid": True,
    "source": "default",
}

# =============================================================================
# LICENSE FILE OPERATIONS
# =============================================================================


def get_license_path() -> Path:
    """
    Get the path to the license file.

    Checks environment variable first, then uses default location.

    Returns:
        Path to license.json file
    """
    env_path = os.environ.get(LICENSE_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_LICENSE_PATH


def load_entitlement() -> Dict[str, Any]:
    """
    Load and validate entitlement from license file.

    Checks in order:
    1. LOBSTER_SUBSCRIPTION_TIER environment variable (dev override)
    2. License file at ~/.lobster/license.json
    3. Falls back to free tier defaults

    Returns:
        Entitlement dict with tier, features, custom_packages, etc.
    """
    # Check for environment variable override (development/testing)
    env_tier = os.environ.get(TIER_ENV_VAR)
    if env_tier:
        logger.debug(f"Using tier from environment variable: {env_tier}")
        return {
            **DEFAULT_ENTITLEMENT,
            "tier": env_tier.lower(),
            "source": "environment",
            "valid": True,
        }

    # Try to load from license file
    license_path = get_license_path()

    if not license_path.exists():
        logger.debug(f"No license file found at {license_path}, using free tier")
        return DEFAULT_ENTITLEMENT

    try:
        with open(license_path, "r") as f:
            data = json.load(f)

        # Validate required fields
        if "tier" not in data:
            logger.warning("License file missing 'tier' field, using free tier")
            return DEFAULT_ENTITLEMENT

        # Check expiration
        expires_at = data.get("expires_at")
        if expires_at:
            try:
                expiry_date = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expiry_date < datetime.now(expiry_date.tzinfo):
                    logger.warning("License has expired, falling back to free tier")
                    return {
                        **DEFAULT_ENTITLEMENT,
                        "expired": True,
                        "expired_tier": data.get("tier"),
                        "source": "expired_license",
                    }
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse expiry date: {e}")

        # Verify signature if present (placeholder for future cryptographic verification)
        if "signature" in data:
            if not _verify_signature(data):
                logger.warning("License signature verification failed, using free tier")
                return {
                    **DEFAULT_ENTITLEMENT,
                    "signature_invalid": True,
                    "source": "invalid_signature",
                }

        # Valid entitlement
        return {
            "tier": data.get("tier", "free").lower(),
            "customer_id": data.get("customer_id"),
            "issued_at": data.get("issued_at"),
            "expires_at": data.get("expires_at"),
            "custom_packages": data.get("custom_packages", []),
            "features": data.get("features", []),
            "valid": True,
            "source": "license_file",
        }

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in license file: {e}")
        return {**DEFAULT_ENTITLEMENT, "parse_error": str(e)}
    except Exception as e:
        logger.error(f"Error reading license file: {e}")
        return {**DEFAULT_ENTITLEMENT, "read_error": str(e)}


def _verify_signature(data: Dict[str, Any]) -> bool:
    """
    Verify the cryptographic signature of an entitlement.

    This is a placeholder for future implementation. The actual
    verification would use public key cryptography to verify
    that the entitlement was issued by the Omics-OS license service.

    Args:
        data: Entitlement data including signature

    Returns:
        True if signature is valid (currently always returns True)
    """
    # TODO: Implement actual signature verification using public key
    # For now, accept all signatures during development
    signature = data.get("signature")
    if signature:
        logger.debug("Signature verification placeholder - accepting signature")
    return True


def save_entitlement(entitlement: Dict[str, Any]) -> bool:
    """
    Save entitlement data to license file.

    This is called by the CLI during activation to persist
    the entitlement received from the license service.

    Args:
        entitlement: Entitlement data to save

    Returns:
        True if save was successful
    """
    license_path = get_license_path()

    try:
        # Ensure directory exists
        license_path.parent.mkdir(parents=True, exist_ok=True)

        # Write entitlement
        with open(license_path, "w") as f:
            json.dump(entitlement, f, indent=2, default=str)

        # Set restrictive permissions (owner read/write only)
        os.chmod(license_path, 0o600)

        logger.info(f"Saved entitlement to {license_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save entitlement: {e}")
        return False


def clear_entitlement() -> bool:
    """
    Remove the license file, reverting to free tier.

    Returns:
        True if file was removed or didn't exist
    """
    license_path = get_license_path()

    try:
        if license_path.exists():
            license_path.unlink()
            logger.info(f"Removed license file at {license_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove license file: {e}")
        return False


# =============================================================================
# TIER AND FEATURE ACCESSORS
# =============================================================================


def get_current_tier() -> str:
    """
    Get the current subscription tier.

    Returns:
        Tier name: "free", "premium", or "enterprise"
    """
    entitlement = load_entitlement()
    return entitlement.get("tier", "free")


def get_custom_packages() -> List[str]:
    """
    Get list of authorized custom packages.

    Returns:
        List of package names (e.g., ["lobster-custom-databiomix"])
    """
    entitlement = load_entitlement()
    return entitlement.get("custom_packages", [])


def is_feature_enabled(feature: str) -> bool:
    """
    Check if a specific feature is enabled for current entitlement.

    Args:
        feature: Feature name to check (e.g., "cloud_compute")

    Returns:
        True if feature is enabled
    """
    entitlement = load_entitlement()
    features = entitlement.get("features", [])
    return feature in features


def is_premium() -> bool:
    """Check if current tier is premium or higher."""
    tier = get_current_tier()
    return tier in ("premium", "enterprise")


def is_enterprise() -> bool:
    """Check if current tier is enterprise."""
    return get_current_tier() == "enterprise"


def get_entitlement_status() -> Dict[str, Any]:
    """
    Get a summary of current entitlement status for display.

    Returns:
        Dict with tier, validity, expiry, and feature summary
    """
    entitlement = load_entitlement()

    status = {
        "tier": entitlement.get("tier", "free"),
        "tier_display": entitlement.get("tier", "free").title(),
        "valid": entitlement.get("valid", False),
        "source": entitlement.get("source", "unknown"),
    }

    # Add expiry info if present
    expires_at = entitlement.get("expires_at")
    if expires_at:
        try:
            expiry_date = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            status["expires_at"] = expires_at
            status["days_until_expiry"] = (expiry_date - datetime.now(expiry_date.tzinfo)).days
        except (ValueError, TypeError):
            status["expires_at"] = expires_at
            status["days_until_expiry"] = None

    # Add feature summary
    status["features"] = entitlement.get("features", [])
    status["custom_packages"] = entitlement.get("custom_packages", [])

    # Add any warnings
    warnings = []
    if entitlement.get("expired"):
        warnings.append(f"License expired (was {entitlement.get('expired_tier')})")
    if entitlement.get("signature_invalid"):
        warnings.append("License signature invalid")
    if entitlement.get("parse_error"):
        warnings.append(f"License file parse error: {entitlement.get('parse_error')}")

    if warnings:
        status["warnings"] = warnings

    return status


# =============================================================================
# ACTIVATION HELPERS (for CLI integration)
# =============================================================================


def activate_license(
    access_code: str,
    license_server_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Activate a license using an access code.

    This function contacts the license server to exchange an access
    code for an entitlement. The entitlement is then saved locally.

    Args:
        access_code: The activation code provided to the customer
        license_server_url: Optional override for license server URL

    Returns:
        Dict with activation result (success, entitlement, error)
    """
    # Default license server URL (can be overridden via env var or param)
    server_url = (
        license_server_url
        or os.environ.get("LOBSTER_LICENSE_SERVER_URL")
        or "https://licenses.omics-os.com"
    )

    try:
        import httpx

        # Call license server activation endpoint
        response = httpx.post(
            f"{server_url}/api/v1/activate",
            json={
                "access_code": access_code,
                "machine_id": _get_machine_fingerprint(),
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            entitlement = data.get("entitlement", {})

            # Save entitlement locally
            if save_entitlement(entitlement):
                return {
                    "success": True,
                    "entitlement": entitlement,
                    "message": f"Successfully activated {entitlement.get('tier', 'premium')} license",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save entitlement locally",
                }

        elif response.status_code == 401:
            return {
                "success": False,
                "error": "Invalid access code",
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "error": "Access code already used or revoked",
            }
        else:
            return {
                "success": False,
                "error": f"License server error: {response.status_code}",
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed - run 'pip install httpx' for license activation",
        }
    except Exception as e:
        logger.error(f"License activation failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _get_machine_fingerprint() -> str:
    """
    Generate a machine fingerprint for license binding.

    This creates a semi-stable identifier for the machine that
    can be used to bind licenses to specific installations.

    Returns:
        Machine fingerprint string
    """
    import hashlib
    import platform

    # Combine various system identifiers
    components = [
        platform.node(),  # Hostname
        platform.machine(),  # Architecture
        platform.system(),  # OS
    ]

    # Hash for privacy
    fingerprint = hashlib.sha256(":".join(components).encode()).hexdigest()[:32]
    return fingerprint


def refresh_entitlement() -> Dict[str, Any]:
    """
    Refresh the current entitlement from the license server.

    This can be used to check for updates to the entitlement
    (e.g., tier upgrades, package additions) without re-activating.

    Returns:
        Dict with refresh result
    """
    entitlement = load_entitlement()

    if entitlement.get("source") != "license_file":
        return {
            "success": False,
            "error": "No active license to refresh",
        }

    customer_id = entitlement.get("customer_id")
    if not customer_id:
        return {
            "success": False,
            "error": "No customer ID in current license",
        }

    server_url = os.environ.get(
        "LOBSTER_LICENSE_SERVER_URL",
        "https://licenses.omics-os.com"
    )

    try:
        import httpx

        response = httpx.post(
            f"{server_url}/api/v1/refresh",
            json={
                "customer_id": customer_id,
                "machine_id": _get_machine_fingerprint(),
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            new_entitlement = data.get("entitlement", {})

            if save_entitlement(new_entitlement):
                return {
                    "success": True,
                    "entitlement": new_entitlement,
                    "message": "Entitlement refreshed successfully",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save refreshed entitlement",
                }
        else:
            return {
                "success": False,
                "error": f"Refresh failed: {response.status_code}",
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
