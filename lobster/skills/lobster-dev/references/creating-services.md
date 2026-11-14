# Creating Stateless Services — Lobster AI 1.2

A scientific service performs a bounded operation on supplied data. It does not
own the session, route agents, construct an LLM or silently save a figure.
The wrapping tool handles DataManager storage and provenance logging.

## Return Contract

```python
result_adata, statistics, ir = service.analyze(adata, **parameters)
```

Use `(AnnData or None, dict, AnalysisStep)` for provenance-required operations.
A read-only administrative service can return no IR when its tool category does
not require it. Do not replace a required IR with `None` just to satisfy tuple size.

Services belong in a package's `lobster/services/<function_type>/` namespace,
for example `analysis/`, `quality/`, `data_access/` or `metadata/`. Shared core
services remain in the core SDK. Preserve shared PEP 420 namespaces; a plugin
must not add `lobster/__init__.py` or `lobster/services/__init__.py`.

## Small Illustrative Service

This example filters features by their supplied matrix total. It is not a
biological QC recommendation: the caller must choose an appropriate matrix and
threshold for the assay.

```python
from lobster.core.provenance.analysis_ir import AnalysisStep


class FeatureTotalService:
    def filter_features(self, adata, min_total: int = 1):
        import numpy as np

        if adata is None or adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("A nonempty input modality is required")
        if min_total < 0:
            raise ValueError("min_total must be nonnegative")
        totals = np.asarray(adata.X.sum(axis=0)).ravel()
        keep = totals >= min_total
        output = adata[:, keep].copy()
        stats = {
            "status": "complete",
            "features_before": adata.n_vars,
            "features_after": output.n_vars,
        }
        ir = AnalysisStep(
            operation="filter_features_by_total",
            tool_name="filter_features_by_total",
            description="Keep features meeting the selected matrix-total threshold",
            library="numpy",
            code_template=(
                "totals = np.asarray(adata.X.sum(axis=0)).ravel()\n"
                "adata = adata[:, totals >= {{ min_total }}].copy()\n"
            ),
            imports=["import numpy as np"],
            parameters={"min_total": min_total},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )
        return output, stats, ir
```

The template must perform the actual operation, not just a comment saying
"analysis here". Use typed finite parameters; retain inputs/outputs that a future
notebook cell needs. A full scientific service also validates representation,
missing values, identifier consistency and assay-specific assumptions.

## Wrap with a Tool

Inside the owning agent's tool factory:

```python
from langchain_core.tools import tool


@tool
def filter_features_by_total(modality_name: str, min_total: int = 1) -> str:
    """Keep features meeting the supplied matrix-total threshold."""
    adata = data_manager.get_modality(modality_name)
    output, stats, ir = service.filter_features(adata, min_total=min_total)
    data_manager.store_modality(f"{modality_name}_filtered", output)
    data_manager.log_tool_usage(
        "filter_features_by_total",
        {"modality_name": modality_name, "min_total": min_total},
        description="Filtered features by matrix total",
        ir=ir,
    )
    return str(stats)


filter_features_by_total.metadata = {"categories": ["FILTER"], "provenance": True}
filter_features_by_total.tags = ["FILTER"]
```

`data_manager` and `service` are factory-local dependencies, not module-level
session objects. Match the logged tool name to the exposed function name. The
third argument of `log_tool_usage` is a description, not a statistics dictionary.

## IR Fidelity

- Use a real `AnalysisStep` with the operation's parameters and code.
- If the tool operates on a split/derived modality, its IR must select the same
  observations/features rather than blindly operating on the notebook's `adata`.
- Record random seeds or outcomes for nondeterministic transformations.
- For external metadata/outcome records that cannot transform notebook data, set
  `replayable=False` explicitly. This records the step; it does not rerun it.
- There is no general `AnalysisStep + AnalysisStep` composition API. Log real
  steps separately or construct an explicit combined template that actually
  performs the same operations in the same order.
- Test replay against values, labels, layers and relevant metadata, not merely
  the presence of an IR object or a successful notebook export.

## Raw Counts and Exports

DESeq2 requires a raw-count workflow. Preserve `layers['counts']` through
normalization when appropriate and preserve sample/feature identity through
aggregation. Finite, nonnegative whole-number checks cannot establish the
scientific origin of a matrix. Do not round TPM, CPM or log expression to bypass
validation.

Default/relative analysis output paths should be resolved through
`data_manager.exports_dir` by the tool/session boundary. Do not accidentally
write relative files into the caller's current working directory. An export
path is not evidence that the result was independently checked.

## Figure Boundary

Return retained analysis results suitable for drawing. Ordinary scientific tools
can declare `suggests_templates`; templates draw from those results without
mutating them or rerunning science. A figure producer must use the shared save
path and report preview vs actual save outcomes. Never write figure-store/catalog
files directly from a stateless analysis service.

## Error and Test Discipline

Validate inputs before destructive work; operate on a copy where needed. A domain
refusal should clearly explain the missing assumption or unsupported data.
Terminal `InferenceRefused` must propagate through generic recovery handlers;
do not turn it into a scientific fallback or log raw provider payloads.

Tests should cover tuple shape, statistics, input immutability, meaningful IR,
refusals, empty/invalid inputs and representative assay-specific outputs. Use
small real fixtures with external inference/network calls mocked. See
[testing.md](testing.md) and [aquadif-contract.md](aquadif-contract.md).
