#!/bin/bash
# Quick verification of custom package integration
# Run this in your activated venv to check everything is working

set -e

echo "=========================================="
echo "QUICK CUSTOM PACKAGE VERIFICATION"
echo "=========================================="
echo ""

echo "1. Installed custom packages:"
pip list | grep "lobster-custom-" || echo "  (none)"
echo ""

echo "2. Agent count in registry:"
python -c "from lobster.config.agent_registry import get_all_agent_names; agents = get_all_agent_names(); print(f'  Total: {len(agents)}'); print(f'  Names: {sorted(agents)}')"
echo ""

echo "3. Custom agents registered:"
python -c "from lobster.config.agent_registry import AGENT_REGISTRY; custom = [a for a in AGENT_REGISTRY.keys() if 'metadata_assistant' in a or 'proteomics' in a or 'machine_learning' in a or 'protein_structure' in a]; print(f'  Custom agents: {custom if custom else \"(none)\"}')"
echo ""

echo "4. License status:"
python -c "from lobster.core.license_manager import get_entitlement_status; s = get_entitlement_status(); print(f'  Tier: {s[\"tier\"]}'); print(f'  Custom packages: {s.get(\"custom_packages\", [])}')"
echo ""

echo "5. Test agent import:"
python -c "
try:
    from lobster.config.agent_registry import AGENT_REGISTRY
    if 'metadata_assistant' in AGENT_REGISTRY:
        config = AGENT_REGISTRY['metadata_assistant']
        print(f'  ✅ metadata_assistant in registry')
        print(f'  Factory: {config.factory_function}')

        # Try to import factory
        module_path, func = config.factory_function.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(module_path)
        factory = getattr(mod, func)
        print(f'  ✅ Factory function importable')
    else:
        print('  ❌ metadata_assistant NOT in registry')
except Exception as e:
    print(f'  ❌ Error: {e}')
"
echo ""

echo "=========================================="
echo "Quick check complete!"
echo ""
echo "For detailed verification, run:"
echo "  python scripts/verify_custom_package.py"
echo "=========================================="
