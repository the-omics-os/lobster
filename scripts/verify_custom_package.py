#!/usr/bin/env python3
"""
Verify custom package integration with Lobster.

This script dynamically discovers and verifies ALL installed lobster-custom-*
packages, checking:
1. Package installation (pip)
2. Plugin discovery (entry points)
3. Agent registration (AGENT_REGISTRY merge)
4. Service imports (all custom services)
5. Factory functions (can actually create agents)

Usage:
    python scripts/verify_custom_package.py                # Check all installed packages
    python scripts/verify_custom_package.py customer     # Check specific package
    python scripts/verify_custom_package.py --list         # List available packages
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_custom_packages() -> List[str]:
    """Discover all installed lobster-custom-* packages."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True,
    )

    packages = []
    for line in result.stdout.split("\n"):
        if line.startswith("lobster-custom-"):
            # Extract package name: "lobster-custom-customer" -> "customer"
            pkg_name = line.split()[0]  # "lobster-custom-customer 2.0.1"
            package_suffix = pkg_name.replace("lobster-custom-", "")
            packages.append(package_suffix)

    return packages


def verify_package_installed(package_name: str) -> Dict[str, Any]:
    """Check if custom package is installed via pip."""
    print(f"\n{'='*70}")
    print(f"1. PACKAGE INSTALLATION CHECK")
    print(f"{'='*70}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", f"lobster-custom-{package_name}"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✅ lobster-custom-{package_name} is installed")

        info = {}
        # Parse version and location
        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                info['version'] = line.split(":")[1].strip()
                print(f"   Version: {info['version']}")
            if line.startswith("Location:"):
                info['location'] = line.split(":")[1].strip()
                print(f"   Location: {info['location']}")

        info['installed'] = True
        return info
    else:
        print(f"❌ lobster-custom-{package_name} NOT installed")
        return {'installed': False}


def verify_entry_points(package_name: str) -> bool:
    """Check if entry points are registered."""
    print(f"\n{'='*70}")
    print(f"2. ENTRY POINT DISCOVERY")
    print(f"{'='*70}")

    try:
        import importlib.metadata as metadata

        eps = metadata.entry_points()

        # Python 3.10+
        if hasattr(eps, 'select'):
            lobster_plugins = eps.select(group='lobster.plugins')
        else:
            # Python 3.9
            lobster_plugins = eps.get('lobster.plugins', [])

        found = False
        for ep in lobster_plugins:
            if package_name in ep.name:
                print(f"✅ Entry point found: {ep.name}")
                print(f"   Module: {ep.value}")
                found = True

        if not found:
            print(f"❌ No entry point found for {package_name}")
            print(f"   Available: {[ep.name for ep in lobster_plugins]}")

        return found
    except Exception as e:
        print(f"❌ Error checking entry points: {e}")
        return False


def verify_custom_registry_import(package_name: str) -> dict:
    """Check if CUSTOM_REGISTRY can be imported."""
    print(f"\n{'='*70}")
    print(f"3. CUSTOM_REGISTRY IMPORT")
    print(f"{'='*70}")

    try:
        module = importlib.import_module(f"lobster_custom_{package_name}")
        registry = getattr(module, "CUSTOM_REGISTRY", None)

        if registry:
            print(f"✅ CUSTOM_REGISTRY imported successfully")
            print(f"   Agents registered: {len(registry)}")
            for name, config in registry.items():
                print(f"   - {name}: {config.factory_function}")
            return registry
        else:
            print(f"❌ CUSTOM_REGISTRY not found in module")
            return {}
    except ImportError as e:
        print(f"❌ Failed to import lobster_custom_{package_name}")
        print(f"   Error: {e}")
        return {}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {}


def verify_agent_registry_merge() -> dict:
    """Check if custom agents are in main AGENT_REGISTRY."""
    print(f"\n{'='*70}")
    print(f"4. AGENT REGISTRY MERGE")
    print(f"{'='*70}")

    try:
        from lobster.config.agent_registry import AGENT_REGISTRY, get_all_agent_names

        all_agents = get_all_agent_names()
        print(f"✅ Total agents in registry: {len(all_agents)}")
        print(f"   Agents: {sorted(all_agents)}")

        # Check for custom agents
        custom_agents = [a for a in all_agents if 'metadata_assistant' in a]
        if custom_agents:
            print(f"\n✅ Custom agent(s) found in registry:")
            for agent in custom_agents:
                config = AGENT_REGISTRY.get(agent)
                if config:
                    print(f"   - {agent}")
                    print(f"     Factory: {config.factory_function}")
        else:
            print(f"\n⚠️  metadata_assistant not in registry (plugin discovery may not have run)")

        return AGENT_REGISTRY
    except Exception as e:
        print(f"❌ Error accessing AGENT_REGISTRY: {e}")
        import traceback
        traceback.print_exc()
        return {}


def discover_package_services(package_name: str) -> List[str]:
    """Dynamically discover all services in custom package."""
    import pkgutil
    import os

    try:
        # Import package
        package = importlib.import_module(f"lobster_custom_{package_name}")
        package_path = os.path.dirname(package.__file__)

        services = []

        # Walk services directory if it exists
        services_dir = Path(package_path) / "services"
        if services_dir.exists():
            for root, dirs, files in os.walk(services_dir):
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        # Convert path to module notation
                        rel_path = Path(root).relative_to(package_path)
                        module_path = f"lobster_custom_{package_name}.{'.'.join(rel_path.parts)}.{file[:-3]}"
                        services.append(module_path)

        return services
    except Exception as e:
        print(f"   Warning: Could not discover services: {e}")
        return []


def verify_service_imports(package_name: str) -> bool:
    """Check if custom services can be imported."""
    print(f"\n{'='*70}")
    print(f"5. SERVICE IMPORTS")
    print(f"{'='*70}")

    services = discover_package_services(package_name)

    if not services:
        print(f"⚠️  No services found in package")
        return True  # Not a failure if package has no services

    print(f"Found {len(services)} service(s) to check:\n")

    success = True
    for service_path in services:
        service_name = service_path.split('.')[-1]
        try:
            importlib.import_module(service_path)
            print(f"✅ {service_name}")
        except ImportError as e:
            print(f"❌ {service_name}")
            print(f"   Path: {service_path}")
            print(f"   Error: {e}")
            success = False

    return success


def verify_agent_factory(package_name: str, registry: dict) -> bool:
    """Check if agent factories can actually create agents."""
    print(f"\n{'='*70}")
    print(f"6. AGENT FACTORY TEST")
    print(f"{'='*70}")

    if not registry:
        print(f"⚠️  No agents in CUSTOM_REGISTRY to test")
        return True

    try:
        from lobster.core.data_manager_v2 import DataManagerV2
        import tempfile

        # Create temp workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace=Path(tmpdir))

            all_passed = True
            for agent_name, config in registry.items():
                try:
                    # Parse factory function path
                    factory_path = config.factory_function
                    module_path, func_name = factory_path.rsplit('.', 1)

                    # Import and call factory
                    module = importlib.import_module(module_path)
                    factory = getattr(module, func_name)
                    agent = factory(data_manager=dm)

                    print(f"✅ {agent_name}")
                    print(f"   Factory: {factory_path}")
                    print(f"   Type: {type(agent).__name__}")

                except Exception as e:
                    print(f"❌ {agent_name}")
                    print(f"   Factory: {config.factory_function}")
                    print(f"   Error: {e}")
                    all_passed = False

            return all_passed

    except Exception as e:
        print(f"❌ Failed to initialize test environment")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_license_entitlement() -> dict:
    """Check license status and custom packages."""
    print(f"\n{'='*70}")
    print(f"7. LICENSE ENTITLEMENT")
    print(f"{'='*70}")

    try:
        from lobster.core.license_manager import get_entitlement_status

        status = get_entitlement_status()

        print(f"Tier: {status.get('tier')}")
        print(f"Source: {status.get('source')}")
        print(f"Valid: {status.get('valid')}")

        custom_packages = status.get('custom_packages', [])
        if custom_packages:
            print(f"✅ Custom packages: {custom_packages}")
        else:
            print(f"⚠️  No custom packages in entitlement")

        return status
    except Exception as e:
        print(f"❌ Error checking license: {e}")
        return {}


def verify_single_package(package_name: str) -> bool:
    """Run all verification checks for a single package."""
    print(f"\n{'='*70}")
    print(f"CUSTOM PACKAGE VERIFICATION: lobster-custom-{package_name}")
    print(f"{'='*70}")

    pkg_info = verify_package_installed(package_name)
    if not pkg_info.get('installed'):
        print(f"\n❌ Package not installed. Skipping remaining checks.")
        return False

    registry = verify_custom_registry_import(package_name)

    checks = [
        ("Package Installation", True),  # Already checked above
        ("Entry Points", verify_entry_points(package_name)),
        ("Custom Registry Import", bool(registry)),
        ("Agent Registry Merge", bool(verify_agent_registry_merge())),
        ("Service Imports", verify_service_imports(package_name)),
        ("Agent Factory", verify_agent_factory(package_name, registry)),
        ("License Entitlement", bool(verify_license_entitlement())),
    ]

    # Summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY: lobster-custom-{package_name}")
    print(f"{'='*70}")

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")

    print(f"\nResult: {passed}/{total} checks passed")
    return passed == total


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify custom package integration with Lobster"
    )
    parser.add_argument(
        "package",
        nargs="?",
        help="Package name (e.g., 'customer'). If omitted, checks all installed packages."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all installed lobster-custom-* packages"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        packages = discover_custom_packages()
        print(f"{'='*70}")
        print(f"INSTALLED CUSTOM PACKAGES")
        print(f"{'='*70}")
        if packages:
            for pkg in packages:
                print(f"  - lobster-custom-{pkg}")
        else:
            print("  (none)")
        return 0

    # Specific package mode
    if args.package:
        success = verify_single_package(args.package)
        return 0 if success else 1

    # Auto-discover mode
    packages = discover_custom_packages()

    if not packages:
        print(f"{'='*70}")
        print(f"NO CUSTOM PACKAGES INSTALLED")
        print(f"{'='*70}")
        print("\nNo lobster-custom-* packages found.")
        print("Install a custom package and try again.")
        return 0

    print(f"{'='*70}")
    print(f"AUTO-DISCOVERED {len(packages)} CUSTOM PACKAGE(S)")
    print(f"{'='*70}")
    for pkg in packages:
        print(f"  - lobster-custom-{pkg}")

    # Check each package
    results = {}
    for pkg in packages:
        success = verify_single_package(pkg)
        results[pkg] = success

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY - ALL PACKAGES")
    print(f"{'='*70}")

    for pkg, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: lobster-custom-{pkg}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All custom packages verified successfully!")
        return 0
    else:
        print("\n⚠️  Some packages have issues. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
