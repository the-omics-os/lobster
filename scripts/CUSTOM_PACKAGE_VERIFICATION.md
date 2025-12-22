# Custom Package Verification Guide

This guide explains how to verify custom package integration with Lobster across agents, services, and tools.

## Quick Verification (30 seconds)

```bash
cd /your/test/environment
source .venv/bin/activate

# Run quick check
bash /path/to/lobster/scripts/quick_verify.sh
```

**Expected Output:**
```
1. Installed custom packages:
  lobster-custom-databiomix  2.0.1

2. Agent count in registry:
  Total: 7
  Names: ['annotation_expert', 'data_expert_agent', 'de_analysis_expert',
          'metadata_assistant', 'research_agent', 'transcriptomics_expert',
          'visualization_expert_agent']

3. Custom agents registered:
  Custom agents: ['metadata_assistant']

4. License status:
  Tier: enterprise
  Custom packages: ['databiomix']

5. Test agent import:
  ✅ metadata_assistant in registry
  Factory: lobster_custom_databiomix.agents.metadata_assistant.metadata_assistant
  ✅ Factory function importable
```

---

## Comprehensive Verification (2-3 minutes)

```bash
# Check all installed custom packages
python /path/to/lobster/scripts/verify_custom_package.py

# Check specific package
python /path/to/lobster/scripts/verify_custom_package.py databiomix

# List available packages
python /path/to/lobster/scripts/verify_custom_package.py --list
```

**Expected Output:**
```
======================================================================
CUSTOM PACKAGE VERIFICATION: lobster-custom-databiomix
======================================================================

======================================================================
1. PACKAGE INSTALLATION CHECK
======================================================================
✅ lobster-custom-databiomix is installed
   Version: 2.0.1
   Location: /path/to/site-packages

======================================================================
2. ENTRY POINT DISCOVERY
======================================================================
✅ Entry point found: databiomix
   Module: lobster_custom_databiomix:register_agents

======================================================================
3. CUSTOM_REGISTRY IMPORT
======================================================================
✅ CUSTOM_REGISTRY imported successfully
   Agents registered: 1
   - metadata_assistant: lobster_custom_databiomix.agents.metadata_assistant.metadata_assistant

======================================================================
4. AGENT REGISTRY MERGE
======================================================================
✅ Total agents in registry: 7
   Agents: ['annotation_expert', 'data_expert_agent', 'de_analysis_expert',
            'metadata_assistant', 'research_agent', 'transcriptomics_expert',
            'visualization_expert_agent']

✅ Custom agent(s) found in registry:
   - metadata_assistant
     Factory: lobster_custom_databiomix.agents.metadata_assistant.metadata_assistant

======================================================================
5. SERVICE IMPORTS
======================================================================
Found 5 service(s) to check:

✅ sample_mapping_service
✅ disease_standardization_service
✅ microbiome_filtering_service
✅ identifier_provenance_service
✅ publication_processing_service

======================================================================
6. AGENT FACTORY TEST
======================================================================
✅ metadata_assistant
   Factory: lobster_custom_databiomix.agents.metadata_assistant.metadata_assistant
   Type: CompiledGraph

======================================================================
7. LICENSE ENTITLEMENT
======================================================================
Tier: enterprise
Source: license_file
Valid: True
✅ Custom packages: ['databiomix']

======================================================================
VERIFICATION SUMMARY: lobster-custom-databiomix
======================================================================
✅ PASS: Package Installation
✅ PASS: Entry Points
✅ PASS: Custom Registry Import
✅ PASS: Agent Registry Merge
✅ PASS: Service Imports
✅ PASS: Agent Factory
✅ PASS: License Entitlement

Result: 7/7 checks passed

🎉 All checks passed! Custom package is fully integrated.
```

---

## Manual Verification Steps

### 1. Check Package Structure

```bash
# Find where package is installed
python -c "
import lobster_custom_databiomix
import os
print(os.path.dirname(lobster_custom_databiomix.__file__))
"

# List package contents
python -c "
import lobster_custom_databiomix
import os
package_dir = os.path.dirname(lobster_custom_databiomix.__file__)
for root, dirs, files in os.walk(package_dir):
    level = root.replace(package_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in [f for f in files if f.endswith('.py')]:
        print(f'{subindent}{file}')
"
```

**Expected Structure:**
```
lobster_custom_databiomix/
  __init__.py
  agents/
    metadata_assistant.py
  services/
    metadata/
      sample_mapping_service.py
      disease_standardization_service.py
      microbiome_filtering_service.py
      identifier_provenance_service.py
    orchestration/
      publication_processing_service.py
```

### 2. Check Agent Registration

```bash
# Check CUSTOM_REGISTRY export
python -c "
from lobster_custom_databiomix import CUSTOM_REGISTRY
print(f'Agents in CUSTOM_REGISTRY: {list(CUSTOM_REGISTRY.keys())}')
for name, config in CUSTOM_REGISTRY.items():
    print(f'\\n{name}:')
    print(f'  Display: {config.display_name}')
    print(f'  Factory: {config.factory_function}')
    print(f'  Handoff: {config.handoff_tool_name}')
"

# Check merged into main registry
python -c "
from lobster.config.agent_registry import AGENT_REGISTRY
if 'metadata_assistant' in AGENT_REGISTRY:
    config = AGENT_REGISTRY['metadata_assistant']
    print(f'✅ metadata_assistant in main registry')
    print(f'   Factory: {config.factory_function}')
else:
    print('❌ metadata_assistant NOT in main registry')
"
```

### 3. Test Service Imports

```bash
# Test each service individually
python -c "
services = [
    'lobster_custom_databiomix.services.metadata.sample_mapping_service',
    'lobster_custom_databiomix.services.metadata.disease_standardization_service',
    'lobster_custom_databiomix.services.metadata.microbiome_filtering_service',
    'lobster_custom_databiomix.services.orchestration.publication_processing_service',
]

import importlib
for service_path in services:
    service_name = service_path.split('.')[-1]
    try:
        mod = importlib.import_module(service_path)
        # Check for expected class
        class_name = ''.join(word.capitalize() for word in service_name.split('_'))
        if hasattr(mod, class_name):
            print(f'✅ {service_name} ({class_name} found)')
        else:
            print(f'⚠️  {service_name} (imported but class {class_name} not found)')
    except ImportError as e:
        print(f'❌ {service_name}: {e}')
"
```

### 4. Test Agent Creation

```bash
# Create agent instance
python -c "
from lobster.core.data_manager_v2 import DataManagerV2
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    dm = DataManagerV2(workspace=Path(tmpdir))

    # Import factory from custom package
    from lobster_custom_databiomix.agents.metadata_assistant import metadata_assistant

    # Create agent
    agent = metadata_assistant(data_manager=dm)

    print('✅ Agent created successfully')
    print(f'   Type: {type(agent).__name__}')
    print(f'   Has get_graph: {hasattr(agent, \"get_graph\")}')

    # Try to get tools (if it's a CompiledGraph)
    if hasattr(agent, 'get_graph'):
        graph = agent.get_graph()
        print(f'   Graph type: {type(graph).__name__}')
"
```

### 5. Check License Entitlement

```bash
# Verify license includes custom packages
python -c "
from lobster.core.license_manager import get_entitlement_status
status = get_entitlement_status()

print(f'Tier: {status[\"tier\"]}')
print(f'Source: {status[\"source\"]}')
print(f'Valid: {status[\"valid\"]}')

custom_pkgs = status.get('custom_packages', [])
if custom_pkgs:
    print(f'✅ Custom packages: {custom_pkgs}')
else:
    print('❌ No custom packages in entitlement')
"

# Check license file directly
cat ~/.lobster/license.json | python -m json.tool | grep -A 2 "custom_packages"
```

---

## Functional Testing (End-to-End)

### Test 1: Agent Available in Chat

```bash
lobster chat
```

Then:
```
> help
```

Look for `metadata_assistant` in the agent list.

### Test 2: Agent Routing

```
> I need to harmonize metadata across multiple datasets
```

**Expected**: Should route to `metadata_assistant` (check the agent name in response).

### Test 3: Service Usage

```
> Process my publication queue and filter for 16S human samples
```

**Expected**:
- Uses `PublicationProcessingService`
- Uses `MicrobiomeFilteringService`
- No import errors

---

## Troubleshooting

### ❌ "metadata_assistant not in registry"

**Cause**: Plugin discovery didn't run or failed silently

**Debug**:
```bash
# Check if package is installed
pip show lobster-custom-databiomix

# Check entry points
python -c "
import importlib.metadata as metadata
eps = metadata.entry_points()
if hasattr(eps, 'select'):
    plugins = list(eps.select(group='lobster.plugins'))
else:
    plugins = list(eps.get('lobster.plugins', []))
print(f'Lobster plugins: {[ep.name for ep in plugins]}')
"

# Force plugin discovery
python -c "
from lobster.core.plugin_loader import discover_plugins
plugins = discover_plugins()
print(f'Discovered: {list(plugins.keys())}')
"
```

### ❌ "ModuleNotFoundError: No module named 'lobster_custom_databiomix.agents'"

**Cause**: Package structure doesn't match factory_function paths

**Debug**:
```bash
# Check actual structure
python -c "
import lobster_custom_databiomix
import os
pkg_dir = os.path.dirname(lobster_custom_databiomix.__file__)
print(f'Package dir: {pkg_dir}')
print(f'Has agents/: {os.path.exists(os.path.join(pkg_dir, \"agents\"))}')
print(f'Has agents/metadata_assistant.py: {os.path.exists(os.path.join(pkg_dir, \"agents\", \"metadata_assistant.py\"))}')
"

# Check factory path
python -c "
from lobster_custom_databiomix import CUSTOM_REGISTRY
print(f'Factory: {CUSTOM_REGISTRY[\"metadata_assistant\"].factory_function}')
"
```

**Fix**: Ensure factory path matches actual structure (should be `lobster_custom_databiomix.agents.X`, not `lobster_custom_databiomix.lobster.agents.X`)

### ❌ "Service import failed"

**Cause**: Service dependencies missing or service file has errors

**Debug**:
```bash
# Try importing service directly
python -c "
import traceback
try:
    from lobster_custom_databiomix.services.metadata.sample_mapping_service import SampleMappingService
    print('✅ Import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    traceback.print_exc()
"
```

---

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Verify Custom Package Integration
  run: |
    # Install custom package
    pip install dist/lobster_custom_*.whl

    # Run verification
    python scripts/verify_custom_package.py

    # Should exit 0 if all checks pass
```

---

## Command Reference

### Discovery
```bash
# List all custom packages
python scripts/verify_custom_package.py --list

# Auto-discover and check all
python scripts/verify_custom_package.py
```

### Specific Package
```bash
# Check DataBioMix package
python scripts/verify_custom_package.py databiomix

# Check hypothetical Anto package
python scripts/verify_custom_package.py anto
```

### Quick Check
```bash
# One-liner: check if any custom packages loaded
python -c "from lobster.config.agent_registry import get_all_agent_names; print([a for a in get_all_agent_names() if a in ['metadata_assistant', 'proteomics_expert', 'machine_learning_expert_agent', 'protein_structure_visualization_expert_agent']])"

# Shell script (multiple checks)
bash scripts/quick_verify.sh
```

---

## What Gets Verified

### 1. Package Installation
- ✅ Package installed via pip
- ✅ Correct version
- ✅ Installation location

### 2. Entry Point Discovery
- ✅ Entry point registered in `lobster.plugins` group
- ✅ Points to `register_agents()` function

### 3. Custom Registry Import
- ✅ `CUSTOM_REGISTRY` dict exported
- ✅ Contains expected agents
- ✅ AgentRegistryConfig format correct

### 4. Agent Registry Merge
- ✅ Custom agents merged into main `AGENT_REGISTRY`
- ✅ Factory functions stored correctly
- ✅ Total agent count correct

### 5. Service Imports
- ✅ All services in `services/` directory importable
- ✅ No missing dependencies
- ✅ Service classes available

### 6. Agent Factory
- ✅ Factory function callable
- ✅ Can create agent instance
- ✅ Agent has expected type (CompiledGraph)

### 7. License Entitlement
- ✅ Correct tier (enterprise)
- ✅ Custom packages listed
- ✅ License valid

---

## Success Criteria

For a custom package to be fully integrated:

| Check | Requirement |
|-------|-------------|
| Installation | Package installed via pip |
| Entry Points | Registered in `lobster.plugins` |
| Registry | `CUSTOM_REGISTRY` with ≥1 agent |
| Merge | Custom agents in main `AGENT_REGISTRY` |
| Services | All services importable |
| Factory | Can create agent instances |
| License | Tier supports custom packages |

**All 7 checks must pass** for successful integration.

---

## For New Custom Packages

When creating a new custom package, use this checklist:

### Package Structure
```bash
lobster-custom-{name}/
├── pyproject.toml                    # version, dependencies, entry_points
├── README.md
├── CHANGELOG.md
└── lobster_custom_{name}/
    ├── __init__.py                   # CUSTOM_REGISTRY export
    ├── agents/                       # Agent implementations
    │   └── {agent_name}.py
    └── services/                     # Supporting services
        └── {category}/
            └── {service_name}.py
```

### Required in __init__.py
```python
from lobster.config.agent_registry import AgentRegistryConfig

CUSTOM_REGISTRY = {
    "agent_name": AgentRegistryConfig(
        name="agent_name",
        display_name="Agent Display Name",
        description="What this agent does",
        factory_function="lobster_custom_{name}.agents.{agent_name}.{agent_name}",
        handoff_tool_name="handoff_to_agent_name",
        handoff_tool_description="When to use this agent",
    ),
}

def register_agents():
    return CUSTOM_REGISTRY
```

### Required in pyproject.toml
```toml
[project.entry-points."lobster.plugins"]
{name} = "lobster_custom_{name}:register_agents"
```

### Verify New Package
```bash
# After installation
python scripts/verify_custom_package.py {name}

# Should see 7/7 checks passed
```

---

## Integration with CI/CD

For automated testing of custom packages:

```yaml
# .github/workflows/test-custom-package.yml
name: Test Custom Package

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install lobster-ai>=3.4.0

      - name: Run verification
        run: |
          # Copy verification script
          curl -O https://raw.githubusercontent.com/the-omics-os/lobster/main/scripts/verify_custom_package.py

          # Install package
          pip install dist/*.whl

          # Verify (exits non-zero on failure)
          python verify_custom_package.py {package_name}
```

---

## Examples for Different Packages

### DataBioMix (metadata focus)
```bash
python scripts/verify_custom_package.py databiomix
# Expects: 1 agent (metadata_assistant), 5 services
```

### Hypothetical Anto (proteomics focus)
```bash
python scripts/verify_custom_package.py anto
# Expects: 1 agent (proteomics_expert), proteomics services
```

### Hypothetical Multi-Agent Package
```bash
python scripts/verify_custom_package.py enterprise
# Expects: Multiple agents, diverse services
```

---

## See Also

- **Sync Infrastructure**: `scripts/README_SYNC_INFRASTRUCTURE.md`
- **Premium Licensing**: `docs/PREMIUM_LICENSING.md`
- **Plugin Loader**: `lobster/core/plugin_loader.py`
- **Subscription Tiers**: `lobster/config/subscription_tiers.py`
