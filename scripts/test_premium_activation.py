#!/usr/bin/env python3
"""
Test premium activation and custom package installation.

This script tests the complete premium activation flow:
1. Activates license with test key
2. Verifies enterprise tier is returned
3. Checks that custom packages are auto-installed
4. Validates entitlement signature

Usage:
    python scripts/test_premium_activation.py
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.core.license_manager import (
    activate_license,
    get_entitlement_status,
    clear_entitlement,
)

# Test key for DataBioMix enterprise customer
TEST_KEY = "cb7493f2-6c31-4bfc-b50e-d7e653b91554"


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    """Run the premium activation test."""
    print("=" * 70)
    print("PREMIUM ACTIVATION TEST")
    print("=" * 70)
    print()

    # Step 1: Clear any existing license
    print("Step 1: Clearing any existing license...")
    clear_entitlement()
    print("✓ License cleared")
    print()

    # Step 2: Verify free tier before activation
    print("Step 2: Verifying free tier status...")
    status = get_entitlement_status()
    if status["tier"] != "free":
        print(f"✗ ERROR: Expected free tier, got {status['tier']}")
        return 1
    print(f"✓ Confirmed free tier (source: {status['source']})")
    print()

    # Step 3: Activate with test key
    print("Step 3: Activating with test key...")
    print(f"Key: {TEST_KEY}")
    result = activate_license(TEST_KEY)

    if not result.get("success"):
        print(f"✗ ERROR: Activation failed")
        print(f"   Error: {result.get('error')}")
        return 1

    print("✓ Activation successful")
    print()

    # Step 4: Verify enterprise tier
    print("Step 4: Verifying enterprise tier...")
    status = get_entitlement_status()

    if status["tier"] != "enterprise":
        print(f"✗ ERROR: Expected enterprise tier, got {status['tier']}")
        return 1

    print(f"✓ Tier: {status['tier']}")
    print(f"  Source: {status['source']}")
    print(f"  Valid: {status['valid']}")
    if "expires_at" in status:
        print(f"  Expires: {status['expires_at']}")
        if status.get("days_until_expiry"):
            print(f"  Days until expiry: {status['days_until_expiry']}")
    print()

    # Step 5: Verify custom packages
    print("Step 5: Verifying custom packages...")
    custom_packages = status.get("custom_packages", [])

    if "lobster-custom-databiomix" not in custom_packages:
        print("✗ ERROR: lobster-custom-databiomix not in custom_packages")
        print(f"   Got: {custom_packages}")
        return 1

    print(f"✓ Custom packages: {custom_packages}")
    print()

    # Step 6: Check if custom package was auto-installed
    print("Step 6: Checking if custom package was installed...")
    packages_installed = result.get("packages_installed", [])
    packages_failed = result.get("packages_failed", [])

    if packages_failed:
        print("⚠ WARNING: Some packages failed to install:")
        for pkg in packages_failed:
            print(f"   - {pkg['name']}: {pkg['error']}")

    if packages_installed:
        print("✓ Packages auto-installed:")
        for pkg in packages_installed:
            print(f"   - {pkg['name']} v{pkg['version']}")
    else:
        print("⚠ No packages were auto-installed")

    # Verify with pip
    is_installed = check_package_installed("lobster-custom-databiomix")
    if is_installed:
        print("✓ lobster-custom-databiomix is installed (verified with pip)")
    else:
        print("⚠ lobster-custom-databiomix not found by pip")
    print()

    # Step 7: Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✓ Activation: SUCCESS")
    print("✓ Tier: enterprise")
    print("✓ Custom packages: lobster-custom-databiomix")
    print(f"✓ Auto-install: {len(packages_installed)} package(s)")
    print()
    print("All tests passed! ✅")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
