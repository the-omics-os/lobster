# Migrating an Existing Plugin to the 1.2 Contract

Read [aquadif-contract.md](aquadif-contract.md) first. Migration means matching
metadata, real IR and executed behavior, not merely adding category labels.

## Inventory Exposed Tools

```bash
rg -n '@tool' packages/lobster-domain/lobster/agents/
```

Use the actual package path and inspect factories/shared tool builders. Runtime-
injected delegation tools need to be distinguished from a factory's own tools.
Do not infer a complete inventory from decorator counts alone.

For each exposed tool, record the main operation, required inputs, primary
category, IR source, retained outputs and whether it can stage/save a figure.

## Categorize and Log Real Operations

Assign a primary category first and secondary categories only for substantial
operations. Required categories are IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE,
ANNOTATE and SYNTHESIZE. Do not bury them behind an administrative category.

```python
analyze_domain.metadata = {"categories": ["ANALYZE"], "provenance": True}
analyze_domain.tags = ["ANALYZE"]
```

Use fresh metadata dictionaries, matching tags and parameterized tool schema
collections. A required tool logs a real AnalysisStep from its service;
`ir=None` is hollow. Confirm the template performs the same updates to relevant
values/labels/layers, or explicitly records a non-replayable outcome when that is
the honest contract. Do not invent an IR addition operator for a multi-step tool.

UTILITY/DELEGATE/CODE_EXEC have no category-required IR gate. This does not
prohibit administrative logging or excuse misclassifying a scientific transform.

## Update Contract Test Configuration

```python
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestDomainContract(AgentContractTestMixin):
    agent_module = "lobster.agents.domain.domain_expert"
    factory_name = "domain_expert"
    is_parent_agent = False
    tools_required = True
```

Use optional `factory_module` when config and factory differ. Old attributes
`agent_name`/`factory_function` do not configure this mixin. Select
`is_parent_agent` according to the actual parent contract, not just to make a
failure disappear. Working plugins require real exposed tools.

## Add the Executed Figure Contract

Add `FigureContractTestMixin` and implement `figure_tool_cases()` using
`FigureToolCase` from `lobster.testing.contract_mixins`.

Each case supplies a real tool name/schema, `setup(dm, monkeypatch)` with a small
real fixture and `assert_success(reply, dm)` that checks the scientific output
and rejects returned errors. The harness creates a fresh real DataManagerV2 and
invokes the actual factory/tool. A mocked graph is not equivalent.

- Ordinary provenance-primary tools must have representative valid cases or
  explicit nonempty reasons in `figure_contract_skips`.
- Transitional figure producers require saving cases; they cannot be skipped.
- `save_plot=False` is a preview and never satisfies a producer's saving case.
- Ordinary/preview cases must not save/stage/write a figure.
- Producers must not change analysis modalities while drawing.
- Validate `suggests_templates` names against the installed catalog.

Return scientific results/template suggestions instead of retaining an old
analysis-tool auto-plot side effect. For intentional figure producers, use the
central saved-figure path and explicit outcomes. Custom-code Plotly variables
are no longer auto-captured; callers submit explicitly and use parent refs for
edits. See [testing.md](testing.md).

## Align Versions and Code Access

The 1.2 scaffold generates `lobster-ai~=1.2`. Replace unconditional code-tool
construction with `attach_code_tool` using the canonical registry name and declared
`code_tool_default`; keep its returned tool list. Most participating built-in
experts default to disabled; supervisor/data/metadata/visualization roles default
to enabled. See [cli.md](cli.md#code-tool-availability) for exact identifiers,
overrides and precedence. Code runs as your own OS user, not within a confinement
boundary; run only code you trust.

Keep plugin/core compatibility aligned and use installed discovery, not a fixed
historical package/tool inventory. Do not manually inject unsupported code tools
or switch inference credentials to work around refusals.

## Validate and Report

```bash
lobster validate-plugin ./packages/lobster-domain/
uv run pytest packages/lobster-domain/tests/ -m contract -v
uv run pytest packages/lobster-domain/tests/ -v
```

Run only the scope authorized for the task. The canonical validator is static
and still has a hollow-IR warning transition; strict AgentContractTestMixin
rejects hollow IR. The bundled skill validator script is superseded and should
not be used as a release gate.

Report executed cases, skip reasons, warnings, scientific comparisons and any
notebook execution separately. Do not treat a successful export, metadata check
or lack of raised exceptions as proof that the scientific operation succeeded.
