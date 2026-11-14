# Scaffold and Validation — Lobster AI 1.2

Generate new agent packages with the installed engine's scaffold command. Do not
manually construct namespace boundaries, entry points or placeholder contract tests.

## Prerequisites

```bash
which lobster
lobster --version
```

Use a compatible 1.2 engine environment. If missing, get permission before setup:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lobster-ai[anthropic]~=1.2'
```

When npm `lobster` is also present, use `uv run python -m lobster` to address the
local engine explicitly. Engine and plugin must share the development environment.

## Generate

```bash
lobster scaffold agent --name domain_expert --display-name "Domain Expert" \
  --description "Analyze the selected biological domain"
lobster validate-plugin ./lobster-domain/
```

| Flag | Meaning |
|---|---|
| `--name` | Required snake_case agent identifier |
| `--display-name` | Required display label |
| `--description` | Required capability description |
| `--children` | Comma-separated child agent names |
| `--output-dir`, `-o` | Destination; defaults to current directory |
| `--author-name` | Package author; default Lobster AI Community |
| `--author-email` | Author email; default community@lobsterbio.com |

`--tier` is a hidden deprecated option, not a plugin access-control mechanism.
For an in-repository plugin, choose `-o packages/`. Child generation wires the
parent config and each child's entry point; inspect the resulting files.

## Generated Files

```text
lobster-<domain>/
  pyproject.toml
  README.md
  lobster/agents/<domain>/
    <agent_name>.py
    shared_tools.py
    state.py
    config.py
    prompts.py
    __init__.py
  tests/
    conftest.py
    test_contract.py
```

Shared `lobster/` and `lobster/agents/` namespace boundaries have no `__init__.py`.
The optional domain initializer must stay cheap and avoid eager agent imports.

The generated `pyproject.toml` uses `lobster-ai~=1.2`. Keep engine and plugin in the same environment and validate against the version that will run the plugin.

## Replace Placeholders, Not the Contract

- Implement real domain services and validate their scientific assumptions.
- Fill tool schemas/docstrings and AQUADIF metadata/tags after `@tool`.
- Produce real AnalysisStep IR; placeholder comments are not replay code.
- Keep AGENT_CONFIG above heavy imports and factory dependencies lazy.
- Customize decision trees, not a duplicate tool-name catalog in the prompt.
- Replace generated fixture data and reply-substring smoke checks with small
  real inputs and assertions against actual scientific output.

Generated `test_contract.py` imports `AgentContractTestMixin`,
`FigureContractTestMixin` and `FigureToolCase`. The figure cases run with real
DataManagerV2 and the real factory; ordinary analysis tools must not stage/save
figures. Add cases/explicit skip reasons for exposed provenance-primary tools.
Transitional producers require saving cases; previews are not saving cases.
See [testing.md](testing.md).

## Install and Discover

After installation permission and dependency review:

```bash
cd lobster-domain
uv pip install -e '.[dev]'
uv run python -c "from lobster.core.component_registry import component_registry; print(component_registry.has_agent('domain_expert'))"
uv run pytest tests/ -m contract -v
lobster validate-plugin .
```

If discovery fails, check that the editable package is installed in this Python
environment, its `lobster.agents` entry point matches AGENT_CONFIG, and namespace
boundaries are not collapsed by `__init__.py`. Do not edit installed site-packages.

## Validation Limits

`lobster validate-plugin` inspects namespace/entry points, config position, factory
parameters, AQUADIF metadata/provenance, import boundaries and figure-contract
declarations. Results vary with package contents; do not hardcode a result count.
It does not execute the science. The static hollow-IR transition may produce a
warning even though strict AgentContractTestMixin rejects `ir=None`.

The skill's bundled `scripts/validate_plugin.py` is superseded by this installed
command and retained only for compatibility. Use current engine validation plus
executed contracts and scientific tests, not the older script as a release gate.
