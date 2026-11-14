# Testing Plugins and Scientific Outputs — Lobster AI 1.2

Passing a structural check is not scientific qualification. Test the actual tool,
data transformation, refusal and saved-file outcome using representative inputs.
Run only the scope approved for the task; keep provider calls and external network
out of unit/contract tests.

## Focused Commands

```bash
uv run pytest packages/lobster-domain/tests/ -m contract -v
uv run pytest tests/unit/test_my_feature.py -v
uv run pytest tests/unit/test_my_feature.py::test_specific_case -v
lobster validate-plugin ./packages/lobster-domain/
```

The domain/test paths are placeholders for your package. External plugin authors
run the generated package's `tests/` from its directory. Use the same environment
for the engine, installed editable plugin and test runner. Do not start with
unbounded parallelism or the full suite when a small focused check is sufficient.

## Static Validator vs Executed Tests

`lobster validate-plugin` inspects package structure, entry points, config position,
factory signatures, metadata/provenance declarations, imports and figure-contract
declarations. It is not an execution test. It may still warn rather than fail for
an old hollow `ir=None` declaration; that does **not** make the IR acceptable.
`AgentContractTestMixin` rejects hollow provenance and unreadable tool source.

The bundled skill `scripts/validate_plugin.py` is superseded by the installed
engine's `lobster validate-plugin`; do not use its older checks as a release gate.
Neither command produces an assurance about scientific correctness.

## Agent/AQUADIF Contract

Use the current attribute names:

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

`agent_module` contains AGENT_CONFIG; optional `factory_module` is used when the
factory lives elsewhere. `factory_name` is the actual function name.
Do not substitute `agent_name` or `factory_function` as test configuration.

The mixin checks factory parameters, AGENT_CONFIG, tool metadata/categories,
provenance calls, independent metadata objects and collection schemas. It rejects
deprecated `handoff_tools`; factories use `delegation_tools`. When
`is_parent_agent=True`, the minimum is IMPORT + QUALITY + (ANALYZE or DELEGATE).
Do not label a missing scientific tool tested by disabling a gate. Generated
placeholder factories can start with
`tools_required=False`, but working plugins should require actual tools.

Mock inference construction through the real factory's source path when needed.
A MagicMock graph or manually fabricated tool list does not demonstrate what the
real factory exposes.

## Executed Figure Contract

Use `FigureContractTestMixin` and `FigureToolCase` from
`lobster.testing.contract_mixins`. The test kit creates a fresh real DataManagerV2,
builds the actual factory, invokes the tool as a tool call and observes figure/
filesystem behavior. A case has:

```python
FigureToolCase(
    tool_name="analyze_domain",
    args={"modality_name": "example"},
    setup=store_small_real_input,
    assert_success=assert_expected_scientific_output,
    label="representative domain input",
)
```

This describes the case shape, not a runnable test: implement both fixture
functions and use an exposed tool's real argument schema.

- `setup(dm, monkeypatch)` runs before factory construction. Store a small real
  modality and patch external services/inference here; patches survive through
  invocation and assertions.
- `assert_success(reply, dm)` must reject returned error strings and verify actual
  expected values, metadata, new modalities or IR. A reply substring alone is a
  scaffold smoke check, not a scientific test.
- `figure_tool_cases()` supplies representative valid invocations. Coverage is
  over tools observed in those setups, not every possible factory configuration.
- Every ordinary tool with a provenance-required primary category needs a case
  or a nonempty reason in `figure_contract_skips`.
- Transitional figure producers cannot be skipped and need a **saving** case.
  `args={"save_plot": False, ...}` is a preview and does not count as that case.
- Preview/ordinary-tool cases must not stage/save/write a figure. Producers also
  must not alter the analysis dataset while drawing.
- `suggests_templates` entries must resolve in the installed template catalog.
  Coverage reports distinguish cases, skips, uncovered categories and excluded
  trusted submitters; a reported skip is not tested behavior.

Use the generated `test_contract.py` as the structural starting point. Replace
its placeholder input and reply checks with domain fixtures and assertions.

For a native visualization tool such as `create_qc_plots`, direct invocation
requires a full ToolCall, not bare `tool.invoke(args)`:

```python
reply = tool.invoke({"name": tool.name, "args": {"modality_name": "qc_assessed"},
                     "id": "draw_qc_1", "type": "tool_call"})
```

Here `tool` is the selected tool from the real factory; the envelope injects the
call ID, not an invented scientific argument. An underlying service callable is
a different interface.

## Scientific and IR Tests

For every service/tool, check:

1. The return tuple contains the expected data, statistics and real AnalysisStep.
2. Input data is unchanged unless mutation is an explicit contract.
3. Values, identifiers, layers and relevant metadata match expected outputs.
4. Parameters in IR correspond to the operation actually executed.
5. Replaying executable IR on a copy produces the same relevant transformation;
   record-only `replayable=False` steps are treated as records, not transformations.
6. Invalid/empty/nonfinite data and missing biological metadata are refused.
7. Randomness is seeded or outcomes retained, and output files are discovered by
   returned paths rather than assumed current-directory globs.

`AnalysisStep` uses `operation` and `parameters`, not `activity_type`/`params`.
Do not invent a `+` operator for composing IR. The helper
`lobster.testing.provenance_replay.replay_ir` executes a recorded template for
comparison; inspect the helper's source/signature before using it in a new test.

### Counts and GEO

Test both valid raw-count input and normalized/log input that must refuse.
When `layers['counts']` exists but is invalid, DESeq2 must refuse rather than use
`X`. Include sparse/dense forms and sample identity preservation where relevant.
Numeric count validity alone does not prove upstream raw-count origin.

For `data_requirement="gene_counts"`, cover no candidate, ambiguous selection,
incomplete/sample-mismatched data and value fidelity. Do not mark a handled
refusal as a successful download.

### Saved Figures and Templates

Assert actual save outcomes and saved revision identity, not object existence.
Check explicit submission, failed-code no-save, preview no-save, edit parent
preservation, source/config behavior and unsubmitted-figure warnings. Include
pending confirmation: it is uncertainty, not a definitive no-save or confirmed
save. Export tests should distinguish view-package contents from omitted lifecycle
artifacts and unsupported legacy files.
Template fixtures should cover Plotly validity, input immutability, declared
parameters, stable IDs where applicable and house style. Drawing must not rerun
an upstream scientific model merely to produce a chart.

## Privacy and Inference Refusals

Test propagation of typed `InferenceRefused` through broad recovery handlers.
Do not log raw SDK bodies or leak credentials when failures are wrapped.
Privacy tests need assertions against their approved outward fields, not a claim
that all system logs are sanitized. Provider requests are never made implicitly
as part of a documentation or contract inspection task.

## Report What Actually Ran

Record commands and outcomes, including skips, warnings and limits. Keep these
categories separate: source inspection, static validation, executed contract,
scientific fixture comparison, notebook execution and live integration. A pass
in one category does not establish another.
