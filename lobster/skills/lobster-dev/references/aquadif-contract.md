# AQUADIF Tool Contract — Lobster AI 1.2

Every tool declares its primary operation, optional secondary categories and
provenance requirement. Category declarations and tests must describe what the
runtime actually does, not merely the intended scientific workflow.

## Categories

| Category | Primary operation | Required IR |
|---|---|---|
| IMPORT | External data into workspace representation | Yes |
| QUALITY | Technical quality assessment/metrics | Yes |
| FILTER | Remove observations or features | Yes |
| PREPROCESS | Transform values/representation | Yes |
| ANALYZE | Statistical tests, embeddings or patterns | Yes |
| ANNOTATE | Biological labels/meaning | Yes |
| SYNTHESIZE | Combine scientific results | Yes |
| DELEGATE | Handoff to specialist | No |
| UTILITY | Session/queue/file administration | No |
| CODE_EXEC | Custom code escape hatch | Exempt from category-required IR |

Use the primary category first. Up to three categories are supported; choose
secondary categories only for substantial operations, not logging or output
formatting. A provenance-required category must not be buried behind an exempt
primary category.

- FILTER removes; PREPROCESS changes representation.
- QUALITY assesses fitness; ANALYZE extracts scientific patterns.
- ANNOTATE adds meaning; IMPORT loads an external representation.
- Preparing a download queue is administrative; executing the import is IMPORT.
- Exporting a table/file is not itself proof of a scientific transformation.

## Metadata Pattern

```python
@tool
def assess_quality(modality_name: str) -> str:
    """Assess the supplied modality's technical quality."""
    adata = data_manager.get_modality(modality_name)
    output, stats, ir = quality_service.assess(adata)
    data_manager.store_modality(f"{modality_name}_quality_assessed", output)
    data_manager.log_tool_usage(
        "assess_quality",
        {"modality_name": modality_name},
        description="Assessed modality quality",
        ir=ir,
    )
    return str(stats)


assess_quality.metadata = {"categories": ["QUALITY"], "provenance": True}
assess_quality.tags = ["QUALITY"]
```

The example assumes factory-local dependencies and a real implemented service.
Assign metadata **after** `@tool`, use category string literals, independent
metadata dictionaries and matching tags. Parameter collections must be typed
(`list[str]`, `dict[str, Any]`), not bare collection classes.

For a read-only utility:

```python
@tool
def list_modalities() -> str:
    """List modality names in this workspace."""
    return str(data_manager.list_modalities())


list_modalities.metadata = {"categories": ["UTILITY"], "provenance": False}
list_modalities.tags = ["UTILITY"]
```

UTILITY/DELEGATE/CODE_EXEC are exempt from the category-required IR gate.
Administrative logging can still exist; do not infer a blanket ban on
`log_tool_usage` from the exemption. Conversely, an exempt category is not a way
to disguise an actual scientific transformation that should have a real IR.

## Real IR, Not a Keyword Placeholder

For required categories:

```python
output, stats, ir = service.analyze(adata, **parameters)
data_manager.log_tool_usage("analyze_data", parameters, description="Analyzed data", ir=ir)
```

`ir=None` records no AnalysisStep and fails the strict contract. A tool's source
must be readable/parseable for the AST check, including asynchronous tool bodies.
A declaration or an AST call is still not proof of scientific fidelity: execute
and compare the recorded transformation.

| Operation | IR treatment |
|---|---|
| Stores/mutates an AnnData modality | Real executable IR for the same change |
| Writes QC columns into AnnData | IR must perform the same column updates, not only report counts |
| Changes metadata outside notebook AnnData | Explicit outcome record with replayable=False when appropriate |
| Read-only scientific check/drawing | Real IR; record-only when it cannot transform notebook data |
| Administrative queue/file/listing operation | No category-required IR |

Record seeds or outcomes for nondeterministic steps. A derived train/test/split
modality requires an IR that selects the same data. Do not assume the notebook's
current `adata` is automatically that split. Log separate real steps or compose
an explicit combined template; `AnalysisStep + AnalysisStep` is not a supported
composition shortcut.

## Figure Contract

Analysis output and figure saving are separate responsibilities:

- Ordinary provenance-primary tools must not stage/save figures as silent side
  effects. Return retained results and optionally declare `suggests_templates`.
- Suggested names must resolve against the installed template catalog.
- Templates return editable Plotly figures from supplied data. They do not mutate
  modalities, fit models or save to disk.
- A figure producer uses the central saved-figure path. Preview is not a revision.
- Custom code explicitly calls `submit_figure`; unsubmitted Figure variables are
  warned about but not saved in 1.2.
- Editing requires an exact saved revision ref and that same `parent` on submit;
  source/config behavior is explicit, not inferred from a variable name.

Use `FigureContractTestMixin` with representative valid `FigureToolCase` entries
and scientific output assertions. Ordinary covered tools need cases or explicit
skip reasons; transitional producers cannot be skipped and require a saving
case. Preview cases (`save_plot=False`) do not satisfy that saving requirement.
The harness observes the real factory/data manager and reports coverage limits.
See [testing.md](testing.md).

## Static vs Runtime Checks

`AgentContractTestMixin` validates factory/config/schema/category/IR contracts.
It rejects literal hollow IR and unreadable tool source.
`lobster validate-plugin` statically inspects package declarations and can still
emit a transitional warning for hollow IR. Do not treat that warning as permission
to ship without real IR. Figure declaration checks are not execution tests.

```bash
lobster validate-plugin ./packages/lobster-domain/
uv run pytest packages/lobster-domain/tests/ -m contract -v
```

Substitute your actual package path. Use focused tests with inference/network
mocked and report skipped cases separately from executed behavior.

`AquadifMonitor` lives in `lobster/core/governance/aquadif_monitor.py`. It observes
invocations/category distribution and provenance calls; monitoring does not turn
an absent IR or unsupported figure behavior into a passing contract.

## Migration and Code Access

See [aquadif-migration.md](aquadif-migration.md) for an existing agent and
[scaffold.md](scaffold.md) for a new package.

Custom-code availability is resolved by `attach_code_tool` for participating
factories; exemption from the IR category gate does not grant code access.
Use the canonical registry name, declare `code_tool_default` as needed, and
retain the helper's returned tool list. See [cli.md](cli.md#code-tool-availability)
for defaults and overrides. Code runs as your own OS user, not within a confinement
boundary; run only code you trust.

Terminal inference refusals must propagate through tool/service recovery. Never
use an arbitrary-code or provider fallback to circumvent auth, policy or budget
boundaries. Notebook export and a passing metadata gate do not establish that
an exported analysis was executed successfully.
