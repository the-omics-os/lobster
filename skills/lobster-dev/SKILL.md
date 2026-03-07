---
name: lobster-dev
description: |
  Develop, extend, and contribute to Lobster AI — the multi-agent self-extending bioinformatics engine.
  Use when working on Lobster codebase, creating agents/services, understanding architecture,
  fixing bugs, adding features, or contributing to the open-source project.

  IMPORTANT: Before creating new agents or packages, follow the planning
  workflow first (see "What To Do Based On Your Task" → planning-workflow.md).

  Trigger phrases: "add agent", "create service", "extend lobster", "contribute",
  "understand architecture", "how does X work in lobster", "fix bug", "add feature",
  "write tests", "lobster development", "agent development", "bioinformatics code",
  "build a new agent for", "add support for", "create plugin", "new domain"
---

# Lobster AI Development Guide

**Lobster AI** is an open-source multi-agent bioinformatics engine (LangGraph, Python 3.12+) powering **Omics-OS**. Lobster solves bioinformatics tasks starting from raw data to scientific insights to visualization using supervisor multi-agent architecture. This skill teaches you how to extend it — from adding a single tool to building entire domain agent packages.

## Step 0: Discover Your Environment

Before any work, determine what's available and how you're working:

```bash
# 1. Is lobster installed? Where?
which lobster
lobster --version

# 2. What agents are already installed?
python -c "from lobster.core.component_registry import component_registry; component_registry.reset(); print(component_registry.list_agents())"

# 3. Where is lobster source? (for reading reference implementations)
python -c "import lobster; print(lobster.__path__)"

# 4. Are you in the lobster repo, or building a standalone plugin?
ls packages/lobster-*/pyproject.toml 2>/dev/null && echo "CONTRIBUTOR" || echo "PLUGIN_AUTHOR"
```

**HARD GATE — If `lobster` is not installed, STOP. Install it NOW before doing anything else:**
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'lobster-ai[anthropic]'  # or [openai], [google], depending on provider
lobster --version  # Must succeed before you proceed
```
Do NOT skip this. Do NOT "come back to it later". Do NOT manually create package directories.
`lobster scaffold agent` is the ONLY way to create new agent packages — it generates correct
PEP 420 structure, entry points, AQUADIF metadata, and contract tests that you WILL get wrong
by hand. If scaffold is unavailable, installing lobster-ai is your first task.

**Your development mode determines your workflow:**

| Mode | How you got here | Where you create packages | How you test |
|---|---|---|---|
| **Contributor** | `git clone` + `make dev-install` | Inside `packages/` in the repo | `make test`, full repo access |
| **Plugin author** | `uv pip install lobster-ai` or `uv tool install` | Anywhere — scaffold creates standalone packages | `uv pip install -e ./lobster-<domain>/` then `pytest` |

Both modes produce the same result: a PEP 420 namespace package discovered by `ComponentRegistry` via entry points. The scaffold output is identical — a standalone package that works in either mode.

## What To Do Based On Your Task

| You want to... | Fast path? | Read these references (in order) |
|---|---|---|
| **Create a new agent** for a new domain | No — full workflow | [planning-workflow.md](references/planning-workflow.md) → [scaffold.md](references/scaffold.md) → [creating-agents.md](references/creating-agents.md) → [aquadif-contract.md](references/aquadif-contract.md) |
| **Add a tool** to an existing agent | Yes (contributor only) | [creating-agents.md](references/creating-agents.md) §Tool Design → [aquadif-contract.md](references/aquadif-contract.md) |
| **Extend an agent** with a child agent | No — needs scoping | [creating-agents.md](references/creating-agents.md) §Parent-Child → [scaffold.md](references/scaffold.md) |
| **Add a database provider or adapter** | No | [plugin-architecture.md](references/plugin-architecture.md) |
| **Create or modify a service** | Yes | [creating-services.md](references/creating-services.md) |
| **Fix a bug** | Yes | [code-layout.md](references/code-layout.md) → [architecture.md](references/architecture.md) |
| **Understand the codebase** | — | [architecture.md](references/architecture.md) → [code-layout.md](references/code-layout.md) |
| **Write or fix tests** | Yes | [testing.md](references/testing.md) |
| **Migrate AQUADIF metadata** on existing agent | Yes | [aquadif-contract.md](references/aquadif-contract.md) §Migration |
| **Find domain knowledge** for a new agent | — | [bioskills-bridge.md](references/bioskills-bridge.md) |

**"Fast path"** = skip the planning workflow, go straight to the reference files.

## Examples

### Example 0: THE WORKFLOW FOR EVERYTHING

user requests: "Build a Lobster agent for epigenomics analysis (bisulfite-seq, ChIP-seq, ATAC-seq) because no lobster packages cover this domain"

```
Step 1: lobster --version          # Not found? Install it FIRST (see Step 0 hard gate)
Step 2: Read planning-workflow.md  # Understand need, check what exists, gather domain knowledge
Step 3: lobster scaffold agent ... # Generate correct package structure (NEVER skip this)
Step 4: Fill in real domain logic  # Read creating-agents.md, creating-services.md
Step 5: lobster validate-plugin ./lobster-<domain>/  # Must pass 8/8
Step 6: uv pip install -e ./lobster-<domain>/ && pytest tests/ -m contract
```

**Expected result:** A standalone PEP 420 package at `./lobster-<your implementation>/` that installs with `uv pip install -e ./lobster-<your implementation>/` and registers via entry points, runs 'lobster status' without errors and running 'lobster query "hi"' without any errors.

## Success Criteria

Before calling your work done, verify:

- [ ] `python scripts/validate_plugin.py <plugin-dir>` passes 8/8 checks
- [ ] Contract tests green: `pytest tests/ -m contract`
- [ ] Every `@tool` has `.metadata` and `.tags` assigned immediately after
- [ ] Every provenance-required tool (IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE) calls `log_tool_usage(ir=ir)`
- [ ] No `lobster/__init__.py` or `lobster/agents/__init__.py` (PEP 420)

## Hard Rules

Non-negotiable. Violating these causes runtime failures or contract test failures.

1. **ComponentRegistry is truth** — agents discovered via entry points, NOT hardcoded registries
2. **AGENT_CONFIG at module top** — define before heavy imports for <50ms entry point discovery
3. **Services return 3-tuple** — `(AnnData, Dict, AnalysisStep)` always
4. **Always pass `ir=ir`** — every `log_tool_usage()` call for provenance-required tools
5. **No `lobster/__init__.py`** — PEP 420 namespace package (also no `lobster/agents/__init__.py`)
6. **No `try/except ImportError`** — use entry points for agent discovery, not eager imports
7. **No module-level `component_registry` calls** — causes slow startup; use lazy functions inside factories
8. **All tools MUST have AQUADIF metadata** — `.metadata` and `.tags` assigned after `@tool` decorator
9. **NEVER manually create package directories** — always use `lobster scaffold agent`. If lobster is not installed, install it first. Scaffold gets PEP 420, entry points, and contract tests right; you won't.

## Contributing

After building a plugin, ask your user if they want to submit a PR. If confirmed: fork → feature branch → `make test` → `make format` → PR with clear description.
