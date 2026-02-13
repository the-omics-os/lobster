# Agent Packages

This directory contains modular agent packages for Lobster AI.

## Package Architecture

Lobster uses PEP 420 namespace packages for seamless composition. Each agent package
installs into the `lobster.agents.*` namespace, allowing users to compose exactly
the agents they need.

**Key principle:** Do NOT create `lobster/__init__.py` or `lobster/agents/__init__.py`
in any package. This enables Python's implicit namespace package merging.

## Available Packages

| Package | Agents | Description |
|---------|--------|-------------|
| `lobster-transcriptomics` | transcriptomics_expert, annotation_expert, de_analysis_expert | Single-cell and bulk RNA-seq analysis |
| `lobster-research` | research_agent, data_expert_agent | Literature discovery and data management |
| `lobster-proteomics` | proteomics_expert | Mass spectrometry and affinity proteomics |
| `lobster-genomics` | genomics_expert | GWAS, variant annotation, VCF/PLINK |
| `lobster-visualization` | visualization_expert | Single-cell and general visualizations |
| `lobster-ml` | machine_learning_expert | ML data preparation and export |
| `lobster-metadata` | metadata_assistant | Sample ID mapping, schema standardization |
| `lobster-structural-viz` | protein_structure_visualization_expert | PDB fetching and ChimeraX visualization |

## Installation

```bash
# Core SDK only
pip install lobster-ai

# Core + specific agents
pip install lobster-ai lobster-transcriptomics

# Core + all free agents
pip install lobster-ai[full]
```

## Package Structure Pattern

Each package follows this structure:

```
packages/
  lobster-{domain}/
    pyproject.toml          # Package configuration with entry points
    README.md               # PyPI description
    lobster/                # NO __init__.py (PEP 420 namespace)
      agents/               # NO __init__.py (PEP 420 namespace)
        {domain}/
          __init__.py       # Module exports
          {agent}.py        # Agent factory with AGENT_CONFIG at top
          state.py          # Agent state class
          config.py         # Configuration constants
          prompts.py        # System prompts
```

## Entry Points

Packages register agents and states via entry points in `pyproject.toml`:

```toml
[project.entry-points."lobster.agents"]
transcriptomics_expert = "lobster.agents.transcriptomics.transcriptomics_expert:AGENT_CONFIG"

[project.entry-points."lobster.states"]
TranscriptomicsExpertState = "lobster.agents.transcriptomics.state:TranscriptomicsExpertState"
```

## Agent Config Pattern

Every agent module must define `AGENT_CONFIG` at the top of the file, BEFORE
heavy imports. This enables fast entry point discovery (<50ms):

```python
# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="my_agent",
    display_name="My Agent",
    description="What this agent does",
    factory_function="lobster.agents.domain.my_agent.my_agent",
    handoff_tool_name="handoff_to_my_agent",
    handoff_tool_description="When to delegate to this agent",
)

# === Heavy imports below ===
import pandas as pd
from langchain_core.tools import tool
...
```

## Development

To work with workspace packages:

```bash
cd /Users/tyo/omics-os/lobster
uv sync                      # Install all packages
uv sync --package lobster-ai # Install core only
```

## Versioning

All packages use synchronized versioning:
- Version 1.0.0 across all packages
- Agent packages depend on `lobster-ai~=1.0.0` (compatible constraint)
- Patch updates are allowed, major/minor changes sync all packages
