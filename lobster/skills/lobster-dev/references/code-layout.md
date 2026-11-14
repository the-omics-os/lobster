# Code Layout — Lobster AI 1.2

Paths below are relative to the engine repository root. Installed namespace
packages can occupy multiple source directories; inspect their entry points and
module paths instead of assuming everything is in the core distribution.

## Core SDK

| Path | Responsibility |
|---|---|
| `lobster/cli.py`, `lobster/__main__.py` | Local command entry points |
| `lobster/cli_internal/commands/light/` | Configuration, workspace, files, queue, scaffold and validation commands |
| `lobster/cli_internal/commands/heavy/` | Local chat/query, session setup, slash dispatch, visualization |
| `lobster/agents/supervisor.py`, `graph.py` | Supervisor and graph/delegation construction |
| `lobster/core/component_registry.py` | Lazy installed component discovery |
| `lobster/config/agent_registry.py` | AgentRegistryConfig |
| `lobster/config/llm_factory.py`, `settings.py`, `providers/` | Inference configuration/provider construction |
| `lobster/core/client.py` | AgentClient and session execution |
| `lobster/core/runtime/data_manager.py` | DataManagerV2 (canonical implementation) |
| `lobster/core/adapters/`, `interfaces/` | Data adapters and interfaces |
| `lobster/core/omics_registry.py` | Omics type metadata and detection |
| `lobster/core/provenance/` | Analysis IR, provenance persistence and stage contracts |
| `lobster/core/governance/` | Runtime tool monitoring and result-claim checks |
| `lobster/core/figures/` | Saved revisions, source payloads, saver, listings and custom-code snapshots |
| `lobster/core/plot_manager.py` | Plot staging and saved-figure integration |
| `lobster/figure_templates/` | Public template API and style |
| `lobster/figure_template_providers/` | Built-in template providers |
| `lobster/pricing/` | Offline price data, overrides and usage normalization |
| `lobster/tools/code_tool_switch.py` | Per-agent code-tool exposure, env/workspace precedence and factory helper |
| `lobster/tools/custom_code_tool.py` | Shared execute_custom_code wrapper and explicit figure submission |
| `lobster/tools/download_orchestrator.py` | Queue download orchestration |
| `lobster/services/` | Shared stateless services |
| `lobster/testing/` | Contract mixins and IR replay/comparison helpers |
| `lobster/scaffold/` | Plugin generators, templates and static validators |

`lobster/core/data_manager_v2.py` is a compatibility shim, not the place to add
runtime behavior. Provenance is a package, not an old root `provenance.py` file.

## Domain Packages

```text
packages/lobster-<domain>/
  pyproject.toml                  # entry points and compatibility dependencies
  lobster/                        # PEP 420 shared namespace: no __init__.py
    agents/                       # shared namespace: no __init__.py
      <domain>/                   # package-specific agent modules
    services/                     # shared namespace: no __init__.py
      analysis/ or quality/ ...   # function-oriented scientific services
    figure_template_providers/    # optional provider module; no namespace __init__.py
  tests/                          # contract and scientific tests
```

Do not force existing packages into that illustrative shape: some factories,
including visualization, live directly in the agents namespace. The exact import
path comes from the package's `[project.entry-points."lobster.agents"]` entries.
`lobster-vector` provides cross-cutting services rather than agent factories;
`lobster-ai-tui` packages the Go binary built from `lobster-tui/`.

## Source Discovery

```bash
# Agent registration and factory locations
rg -n 'lobster.agents|AGENT_CONFIG' packages/*/pyproject.toml
find packages -path '*/agents/*' -name '*.py'

# Scientific services and tool definitions
find packages -path '*/services/*' -name '*.py'
rg -n '@tool' packages/lobster-transcriptomics/lobster/agents/

# Installed source, in the same Python environment as the engine
uv run python -c "import lobster; print(list(lobster.__path__))"
uv run python -c "import lobster.agents.transcriptomics as m; print(list(m.__path__))"
```

Reading installed source is fine; editing site-packages is not a development
workflow. Scaffold a plugin or work in an approved source checkout instead.

## Tests and Documentation

- Core tests: `tests/unit/`, `tests/integration/` and feature-specific directories.
- Domain tests: each package's `tests/` directory.
- Contract implementation: `lobster/testing/contract_mixins.py`.
- Generated contract example: `lobster/scaffold/templates/test_contract.py.j2`.
- Developer guidance: this skill's `references/`, plus `CONTRIBUTING.md`.
- Public documentation: `https://docs.omics-os.com` (discover pages through llms.txt).

Do not assume an old wiki, private instructions or a sibling documentation checkout
exists in a contributor's environment. Follow the paths that actually exist.
