# Code Layout

Quick reference for finding files in the Lobster AI codebase.

## Top-Level Structure

```
lobster/                          # Repository root
├── lobster/                      # Core SDK (lobster-ai package)
│   ├── agents/                   # Supervisor + graph builder
│   ├── core/                     # Infrastructure
│   ├── services/                 # Core services
│   ├── tools/                    # Agent tools
│   └── cli.py                    # CLI implementation (3,900+ lines)
├── packages/                     # Agent packages (PEP 420)
│   ├── lobster-transcriptomics/
│   ├── lobster-research/
│   ├── lobster-visualization/
│   ├── lobster-metadata/
│   ├── lobster-structural-viz/
│   ├── lobster-genomics/        # [alpha]
│   ├── lobster-proteomics/      # [alpha]
│   └── lobster-ml/              # [alpha]
├── tests/
│   ├── unit/
│   └── integration/
├── .claude/                      # Claude Code integration
│   ├── docs/                     # Internal docs
│   └── skills/                   # Skills for Claude Code
├── wiki/                         # User documentation (58 pages)
└── docs/                         # Developer documentation
```

## Core SDK (`lobster/lobster/`)

### Agents (`lobster/agents/`)

| File | Purpose |
|------|---------|
| `supervisor.py` | Main routing agent |
| `graph.py` | LangGraph builder (`create_bioinformatics_graph`) |
| `state.py` | State definitions |

### Core (`lobster/core/`)

| File | Purpose |
|------|---------|
| `component_registry.py` | Agent discovery via entry points |
| `data_manager_v2.py` | Data/workspace management |
| `provenance.py` | W3C-PROV tracking |
| `protocols.py` | Interface definitions |
| `client.py` | AgentClient, APIAgentClient |
| `config.py` | Configuration management |

### Services (`lobster/services/`)

Core services that stay in the base package:
- Database adapters
- Format parsers
- Shared utilities

### Tools (`lobster/tools/`)

| File | Purpose |
|------|---------|
| `download_orchestrator.py` | 9-step download flow |
| `common_tools.py` | Shared agent tools |
| `data_tools.py` | Data manipulation tools |

## Agent Packages (`packages/`)

Each package follows this structure:

```
lobster-{domain}/
├── pyproject.toml              # Package config + entry points
├── README.md
└── lobster/
    ├── agents/
    │   └── {domain}/
    │       ├── __init__.py
    │       └── {agent_name}.py
    └── services/
        └── {domain}/
            ├── __init__.py
            └── {service_name}.py
```

### Package Contents

| Package | Agents | Services |
|---------|--------|----------|
| `lobster-transcriptomics` | transcriptomics_expert, annotation_expert, de_analysis_expert | 8 services |
| `lobster-research` | research_agent, data_expert_agent | 1 service |
| `lobster-visualization` | visualization_expert | 1 service |
| `lobster-metadata` | metadata_assistant | 8 services |
| `lobster-structural-viz` | protein_structure_visualization_expert | 4 services |
| `lobster-proteomics` [alpha] | proteomics_expert | 11 services |
| `lobster-genomics` [alpha] | genomics_expert | 3 services |
| `lobster-ml` [alpha] | machine_learning_expert | 4 services |

## Tests (`tests/`)

```
tests/
├── unit/
│   ├── agents/               # Agent tests
│   ├── services/             # Service tests
│   ├── core/                 # Core component tests
│   └── tools/                # Tool tests
├── integration/
│   ├── test_workflows.py     # End-to-end workflows
│   └── test_downloads.py     # Download tests
└── conftest.py               # Shared fixtures
```

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Root package config |
| `.env` | Environment variables |
| `.env.example` | Env var template |
| `CLAUDE.md` | Claude Code system prompt |
| `Makefile` | Dev commands |

## Documentation

| Location | Content |
|----------|---------|
| `wiki/` | User guides (58 pages) |
| `docs/` | Developer docs (17 files) |
| `.claude/docs/` | Internal architecture docs |
| `docs-site/` | Online docs (Next.js) |

## Finding Things

### "Where is agent X?"

```bash
# Find agent file
find packages -name "*agent*.py" | grep -i {name}

# Or check entry points
grep -r "lobster.agents" packages/*/pyproject.toml
```

### "Where is service X?"

```bash
# In packages
find packages -path "*/services/*" -name "*.py"

# In core
find lobster/services -name "*.py"
```

### "What tools does agent X have?"

```bash
# Search for @tool decorators
grep -A5 "@tool" packages/lobster-{domain}/lobster/agents/{domain}/*.py
```

### "How is X configured?"

```bash
# Check AGENT_CONFIG
grep -A20 "AGENT_CONFIG" packages/lobster-{domain}/lobster/agents/{domain}/*.py
```

## Key Files Quick Reference

**Start here when...**

| Task | Read This |
|------|-----------|
| Understanding flow | `lobster/agents/graph.py` |
| Agent discovery | `lobster/core/component_registry.py` |
| Data management | `lobster/core/data_manager_v2.py` |
| Provenance | `lobster/core/provenance.py` |
| CLI commands | `lobster/cli.py` |
| Download logic | `lobster/tools/download_orchestrator.py` |
| Agent example | `packages/lobster-transcriptomics/lobster/agents/transcriptomics/transcriptomics_expert.py` |
| Service example | Any file in `packages/*/lobster/services/*/` |
