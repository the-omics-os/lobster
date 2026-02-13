# Creating Agents

This guide explains how to create new agents for Lobster AI.

## Overview

Agents are specialist AI systems that handle specific domains (transcriptomics, proteomics, etc.). Each agent:
- Lives in its own package under `packages/`
- Registers via Python entry points
- Has access to domain-specific tools and services

## Step 1: Create Package Structure

```bash
packages/
└── lobster-mydomain/
    ├── pyproject.toml
    └── lobster/
        └── agents/
            └── mydomain/
                ├── __init__.py
                └── my_agent.py
```

**Note:** No `lobster/__init__.py` — this is a PEP 420 namespace package.

## Step 2: Define AGENT_CONFIG

At the **top** of your agent file (before heavy imports):

```python
# packages/lobster-mydomain/lobster/agents/mydomain/my_agent.py

# CRITICAL: Define config FIRST for fast entry point discovery (<50ms)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="my_expert_agent",
    display_name="My Expert",
    description="Expert agent for my domain analysis including X, Y, and Z",
    factory_function="lobster.agents.mydomain.my_agent.my_expert_agent",
    handoff_tool_name="handoff_to_my_expert_agent",
    handoff_tool_description="Assign tasks for my domain: analysis type A, capability B",
    tier_requirement="free",  # All official agents are free. Use "premium"/"enterprise" for custom packages.
    # Optional: child_agents=["sub_agent_a", "sub_agent_b"],
)

# Heavy imports AFTER config
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import AnalysisStep
```

## Step 3: Implement Agent Factory

The factory function must use the **standardized signature** (v1.0.0+):

```python
# Continuing my_agent.py
from langgraph.graph.state import CompiledGraph
from langgraph.prebuilt import create_react_agent
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings

def my_expert_agent(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "my_expert_agent",
    delegation_tools: Optional[List] = None,  # NOT handoff_tools!
    workspace_path: Optional[Path] = None,
    **kwargs
) -> CompiledGraph:
    """Factory function for My Expert agent.

    Args:
        data_manager: DataManagerV2 for data operations
        callback_handler: Optional callback for streaming
        agent_name: Instance name for logging
        delegation_tools: Tools for delegating to child agents
        workspace_path: Optional workspace path override
    """
    # 1. Create LLM
    settings = get_settings()
    model_params = settings.get_agent_llm_params(agent_name)
    llm = create_llm(agent_name, model_params, workspace_path=workspace_path)

    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # 2. Define tools
    @tool
    def analyze_my_data(modality_name: str, param1: str, param2: int = 10) -> str:
        """Analyze data with my method.

        Args:
            modality_name: Name of the modality to analyze
            param1: Description of param1
            param2: Description of param2
        """
        # Import service inside tool (lazy loading)
        from lobster_mydomain.services.mydomain.my_service import MyService
        service = MyService()

        adata = data_manager.get_modality(modality_name)
        result, stats, ir = service.analyze(adata, param1=param1, param2=param2)

        data_manager.store_modality(f"{modality_name}_analyzed", result)
        data_manager.log_tool_usage(
            "analyze_my_data",
            {"modality_name": modality_name, "param1": param1, "param2": param2},
            stats,
            ir=ir  # CRITICAL: Always pass ir for provenance
        )

        return f"Analysis complete: {stats}"

    # 3. Collect tools
    tools = [analyze_my_data]

    # 4. CRITICAL: Add delegation tools for child agents
    if delegation_tools:
        tools = tools + delegation_tools

    # 5. Create system prompt
    system_prompt = """You are a My Domain expert agent.

Your capabilities:
- Analyze data using specialized methods
- Provide domain-specific insights

Always use the available tools for analysis."""

    # 6. Return compiled agent
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
    )
```

## Step 4: Register Entry Point

In `packages/lobster-mydomain/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lobster-mydomain"
version = "1.0.0"
description = "My Domain agent for Lobster AI"
requires-python = ">=3.12"
dependencies = [
    "lobster-ai~=1.0.0",  # Use ~= for compatible release
]

[project.entry-points."lobster.agents"]
my_expert_agent = "lobster.agents.mydomain.my_agent:AGENT_CONFIG"

# CRITICAL: Enable PEP 420 namespace merging
[tool.setuptools]
packages.find = {where = ["."], include = ["lobster*"], namespaces = true}
```

## Step 5: Add Contract Tests

Use `AgentContractTestMixin` to validate your agent follows the plugin contract:

```python
# tests/test_contract.py
from lobster.testing import AgentContractTestMixin


class TestMyExpertAgent(AgentContractTestMixin):
    """Validate my_expert_agent follows the plugin contract."""

    # Required: module containing AGENT_CONFIG
    agent_module = "lobster.agents.mydomain.my_agent"

    # Required: factory function name
    factory_name = "my_expert_agent"

    # Optional: verify specific tier
    # expected_tier = "free"
```

The mixin automatically tests:
- Factory has standard params: `data_manager`, `callback_handler`, `agent_name`, `delegation_tools`, `workspace_path`
- Factory doesn't use deprecated `handoff_tools` param
- `AGENT_CONFIG` exists and has required fields
- Tier requirement is valid

Run tests:

```bash
pytest tests/test_contract.py -v

# Expected output:
# test_contract.py::TestMyExpertAgent::test_factory_has_standard_params PASSED
# test_contract.py::TestMyExpertAgent::test_no_deprecated_handoff_tools PASSED
# test_contract.py::TestMyExpertAgent::test_agent_config_exists PASSED
# ...
```

## Step 6: Install and Test

```bash
# Install in dev mode
cd packages/lobster-mydomain
pip install -e .

# Verify registration
python -c "from lobster.core.component_registry import ComponentRegistry; print(ComponentRegistry().get_available_agents())"

# Run tests
pytest tests/unit/agents/test_my_agent.py -v
```

## Custom Enterprise Agents

All official Lobster agents are free and open source. For custom enterprise packages that need access control, set `tier_requirement="premium"` or `"enterprise"` in AGENT_CONFIG:

```python
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="my_enterprise_agent",
    display_name="My Enterprise Agent",
    description="Custom enterprise agent with advanced capabilities",
    factory_function="lobster.agents.mydomain.my_enterprise_agent.my_enterprise_agent",
    handoff_tool_name="handoff_to_my_enterprise_agent",
    handoff_tool_description="Assign enterprise analysis tasks",
    tier_requirement="premium",  # Requires license (for custom packages)
)
```

Custom enterprise agents are gated by ComponentRegistry license checks at runtime.

## Best Practices

1. **Config first** — AGENT_CONFIG before imports
2. **Lazy imports** — Import services inside tools, not at module level
3. **Always pass ir** — For provenance tracking
4. **Clear descriptions** — Tool docstrings are shown to LLM
5. **Test thoroughly** — Unit tests for config, creation, and tools

## Common Patterns

### Multi-tool agents

```python
agent_tools = [
    analyze_data,
    visualize_results,
    export_report,
]
```

### Conditional tools

```python
if data_manager.has_modality("proteomics"):
    agent_tools.append(analyze_proteomics)
```

### Shared tools from core

```python
from lobster.tools import common_tools
agent_tools.extend(common_tools.get_data_tools(data_manager))
```

## Documentation

After creating your agent:
1. Add documentation in `docs-site/content/docs/agents/`
2. Follow the existing agent page patterns
3. Include capabilities, use cases, and examples

See docs.omics-os.com/docs/developer/creating-agents for full reference.
