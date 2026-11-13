---
name: lobster-dev
description: |
  Develop and extend Lobster AI: agent plugins, stateless services, tool metadata,
  provenance IR, figure templates, contract tests, and local CLI behavior.
  Use for creating an agent or service, fixing Lobster code, adding a tool,
  understanding architecture, migrating a plugin, or writing tests.
  Before adding a new capability, follow the need-first planning workflow.
  Generate new agent packages with lobster scaffold agent, not by hand.
---

# Lobster AI Development Guide — 1.2

Lobster AI is the Python bioinformatics engine powering Omics-OS. Plugins are
namespace packages discovered through entry points. This skill covers engine
implementation; the supported Cloud client is npm `@omicsos/lobster`, a separate
client contract, not Python/Go Cloud compatibility.

## Start with the Installed Environment and Source

```bash
which lobster
lobster --version
uv run python -c "import lobster; print(list(lobster.__path__))"
uv run python -c "from lobster.core.component_registry import component_registry; print(sorted(component_registry.list_agents()))"
```

Run Python inspection in the environment containing the engine and plugin.
Do not modify installed site-packages. If the engine is unavailable, get user
approval before installation or environment changes:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lobster-ai[anthropic]~=1.2'
```

A `uv tool` installation is isolated from a project's Python environment. For
plugin development, keep engine and plugin in the same `.venv`. When npm's
`lobster` wins PATH, `uv run python -m lobster` targets the local engine explicitly.

## Choose the Workflow

| Task | Read |
|---|---|
| New domain/capability | [planning-workflow.md](references/planning-workflow.md) → [scaffold.md](references/scaffold.md) → [creating-agents.md](references/creating-agents.md) |
| Existing agent tool | [creating-agents.md](references/creating-agents.md) → [aquadif-contract.md](references/aquadif-contract.md) |
| Stateless service | [creating-services.md](references/creating-services.md) |
| Adapter/provider/omics extension | [plugin-architecture.md](references/plugin-architecture.md) |
| Figure template or saved-figure behavior | [plugin-architecture.md](references/plugin-architecture.md) → [testing.md](references/testing.md) |
| Contract migration | [aquadif-migration.md](references/aquadif-migration.md) |
| Tests | [testing.md](references/testing.md) |
| Source discovery / architecture | [code-layout.md](references/code-layout.md) → [architecture.md](references/architecture.md) |
| Local developer commands | [cli.md](references/cli.md) |
| A new local UI string | [cli.md](references/cli.md#local-language) and `lobster/i18n/README.md` |
| External domain knowledge | [bioskills-bridge.md](references/bioskills-bridge.md) |

## Scaffold First

```bash
lobster scaffold agent --name domain_expert --display-name "Domain Expert" \
  --description "Analyze the chosen biological domain"
lobster validate-plugin ./lobster-domain/
```

Fill in the generated domain logic and replace placeholder contract fixtures with
small real fixtures and scientific assertions. Structural validation does not
establish that the biology is correct or that a notebook produces the same outputs.

The 1.2 scaffold generates `lobster-ai~=1.2`; keep the plugin's compatibility range aligned with the installed engine.

The bundled `scripts/validate_plugin.py` is **superseded** by
`lobster validate-plugin`. It is retained for compatibility; use the installed
engine's command and current test kit, not that older script as a release gate.

## Hard Rules

1. Discover components through `ComponentRegistry` and entry points, not a
   hardcoded agent catalog or module-level registry calls.
2. Put `AGENT_CONFIG` before heavy imports; load scientific dependencies lazily.
3. Keep analysis services stateless and returning `(AnnData or None, dict, AnalysisStep)`.
4. For provenance-required tools, pass a real `ir` into `log_tool_usage`;
   literal `ir=None` is hollow. Record parameters, transformations and outcomes.
5. Preserve PEP 420 boundaries: no `lobster/__init__.py` or shared namespace
   `__init__.py` in a plugin. Avoid eager cross-agent imports and ImportError guards
   pretending to perform agent discovery.
6. Every tool has AQUADIF `.metadata` and `.tags`, assigned after `@tool`;
   primary category first, parameterized collection types in its schema.
7. Ordinary scientific tools should provide results and template suggestions,
   not silently save figures. Use the shared saved-figure path for a figure producer.
8. Run the current `AgentContractTestMixin` and `FigureContractTestMixin` cases.
   A metadata declaration or placeholder reply assertion is not an execution check.
9. Preserve raw-count source for DESeq2; never round normalized values to satisfy
   count validation. Drawing a figure must not rerun an analysis unnecessarily.
10. Propagate terminal `InferenceRefused` through recovery handlers; do not switch
    inference authority or expose raw SDK errors/request payloads to the user.
11. A new local user-visible string needs easy Simplified Chinese in the same
    change. Follow [local language](references/cli.md#local-language). Do not
    translate command names, flags, JSON keys, tool docstrings, or agent prompts.

Custom-code exposure uses `attach_code_tool` and the canonical registry name. The supervisor, `data_expert_agent`, `metadata_assistant`, and `visualization_expert_agent` default to enabled; other participating built-in experts default to disabled. `research_agent` does not offer this tool. See [CLI configuration](references/cli.md#code-tool-availability) for overrides and precedence. Code runs as your own OS user, not within a confinement boundary; run only code you trust.

## Completion Checklist

- [ ] Canonical `lobster validate-plugin` has no unresolved failures or warnings.
- [ ] Focused contract and scientific tests cover valid inputs, refusals and outputs.
- [ ] Real IR is tested against the recorded transformation, not just its presence.
- [ ] Figure cases cover relevant tools, previews and save outcomes; skip reasons
      are explicit and not represented as tested behavior.
- [ ] Required entry points are discovered in the same environment as the engine.
- [ ] Changes have user-facing documentation with appropriately bounded claims.

Use `uv run pytest` for Python tests. Run only the scope approved for the task;
never turn a documentation audit into installs, network inference or a full suite.
Before contribution, ask whether the user wants publication. A fork, push, PR,
package release or tag is not implied by permission to make a local change.
