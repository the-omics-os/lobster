---
name: lobster-dev
description: |
  Develop, extend, and contribute to Lobster AI — the multi-agent bioinformatics engine.
  Use when working on Lobster codebase, creating agents/services, understanding architecture,
  fixing bugs, adding features, or contributing to the open-source project.
  
  Trigger phrases: "add agent", "create service", "extend lobster", "contribute",
  "understand architecture", "how does X work in lobster", "fix bug", "add feature",
  "write tests", "lobster development", "agent development", "bioinformatics code"
---

# Lobster AI Development Guide

**Lobster AI** is a multi-agent bioinformatics platform using LangGraph for orchestration.
This skill teaches you how to work with, extend, and contribute to the codebase.

## Quick Navigation

| Task | Documentation |
|------|---------------|
| **Architecture overview** | [references/architecture.md](references/architecture.md) |
| **Creating new agents** | [references/creating-agents.md](references/creating-agents.md) |
| **Creating new services** | [references/creating-services.md](references/creating-services.md) |
| **Code layout & finding files** | [references/code-layout.md](references/code-layout.md) |
| **Testing patterns** | [references/testing.md](references/testing.md) |
| **CLI reference** | [references/cli.md](references/cli.md) |

## Critical Rules

1. **ComponentRegistry is truth** — Agents discovered via entry points, NOT hardcoded
2. **AGENT_CONFIG at module top** — Define before heavy imports for <50ms discovery
3. **Services return 3-tuple** — `(AnnData, Dict, AnalysisStep)` always
4. **Always pass `ir`** — `log_tool_usage(..., ir=ir)` for reproducibility
5. **No `lobster/__init__.py`** — PEP 420 namespace package

## Package Structure

```
lobster/
├── packages/                    # Agent packages (PEP 420)
│   ├── lobster-transcriptomics/ # transcriptomics_expert, annotation_expert, de_analysis_expert
│   ├── lobster-research/        # research_agent, data_expert_agent
│   ├── lobster-visualization/   # visualization_expert
│   ├── lobster-metadata/        # metadata_assistant
│   ├── lobster-structural-viz/  # protein_structure_visualization_expert
│   ├── lobster-genomics/        # genomics_expert
│   ├── lobster-proteomics/      # proteomics_expert
│   └── lobster-ml/              # machine_learning_expert
└── lobster/                     # Core SDK
    ├── agents/supervisor.py     # Supervisor (stays in core)
    ├── agents/graph.py          # LangGraph builder
    ├── core/                    # Infrastructure (registry, data_manager, provenance)
    ├── services/                # Analysis services
    └── tools/                   # Agent tools
```

## Quick Commands

```bash
# Setup (development)
make dev-install              # Full dev setup with editable install
make test                     # Run all tests
make format                   # black + isort

# Setup (end-user testing via uv tool)
uv tool install 'lobster-ai[full,anthropic]'   # Install as users see it
uv tool upgrade lobster-ai                      # Upgrade to latest

# Running
lobster chat                  # Interactive mode
lobster query "your request"  # Single-turn

# Testing
pytest tests/unit/            # Fast unit tests
pytest tests/integration/     # Integration tests
```

## Service Pattern (Essential)

All services return a 3-tuple:

```python
def analyze(self, adata, **params) -> Tuple[AnnData, Dict, AnalysisStep]:
    # Your analysis logic
    stats = {"n_cells": adata.n_obs, "status": "complete"}
    ir = AnalysisStep(
        activity_type="analyze",
        inputs={"n_obs": adata.n_obs},
        outputs=stats,
        params=params
    )
    return processed_adata, stats, ir
```

Tools wrap services:

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    result, stats, ir = service.analyze(adata, **params)
    data_manager.log_tool_usage("analyze", params, stats, ir=ir)  # IR mandatory!
    return f"Complete: {stats}"
```

## Agent Registration (Entry Points)

Agents register via `pyproject.toml`:

```toml
[project.entry-points."lobster.agents"]
my_agent = "lobster.agents.my_domain.my_agent:AGENT_CONFIG"
```

AGENT_CONFIG must be defined at module top (before imports):

```python
# lobster/agents/mydomain/my_agent.py
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="my_agent",
    display_name="My Expert Agent",
    description="What this agent does",
    factory_function="lobster.agents.mydomain.my_agent.my_agent",
    handoff_tool_name="handoff_to_my_agent",
    handoff_tool_description="Assign tasks for my domain analysis",
    tier_requirement="free",  # All official agents are free
)

# Heavy imports AFTER config
from lobster.core.data_manager_v2 import DataManagerV2
# ... rest of implementation
```

## Key Files

| File | Purpose |
|------|---------|
| `lobster/agents/graph.py` | LangGraph orchestration |
| `lobster/core/component_registry.py` | Agent discovery |
| `lobster/core/data_manager_v2.py` | Data/workspace management |
| `lobster/core/provenance.py` | W3C-PROV tracking |
| `lobster/cli.py` | CLI implementation |

## Online Documentation

Full documentation at **docs.omics-os.com** (or local `docs-site/`):

- **Getting Started**: `docs/getting-started/`
- **Core SDK**: `docs/core/`
- **Agents**: `docs/agents/`
- **Developer Guide**: `docs/developer/`
- **API Reference**: `docs/api-reference/`

## Common Tasks

### Adding a New Agent

1. Create package: `packages/lobster-mydomain/`
2. Define AGENT_CONFIG at top of agent file
3. Register entry point in `pyproject.toml`
4. Implement agent with tools
5. Add tests in `tests/unit/agents/`

See [references/creating-agents.md](references/creating-agents.md) for full guide.

### Adding a New Service

1. Create service class in appropriate package
2. Implement 3-tuple return pattern
3. Wrap in tool with `log_tool_usage`
4. Add unit tests

See [references/creating-services.md](references/creating-services.md) for full guide.

### Understanding Data Flow

```
User Query → CLI → LobsterClientAdapter → AgentClient
                                              ↓
                            LangGraph (supervisor → agents)
                                              ↓
                               Services → DataManagerV2
                                              ↓
                                    Results + Provenance
```

## Testing

```bash
# Unit tests (fast, no external deps)
pytest tests/unit/ -v

# Integration tests (may need env vars)
pytest tests/integration/ -v

# Specific test
pytest tests/unit/test_my_feature.py -v

# With coverage
pytest --cov=lobster tests/
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes following patterns above
4. Run tests: `make test`
5. Format code: `make format`
6. Submit PR with clear description
