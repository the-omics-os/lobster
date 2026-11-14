# Architecture — Lobster AI 1.2

Read implementation source when changing a contract. This reference describes the
local engine, not a Python/Go Cloud-client contract.

## Execution Flow

```text
Local CLI (lobster/cli.py + cli_internal/commands)
  → AgentClient
  → LangGraph supervisor and installed specialist factories
  → Tools → stateless services
  → DataManagerV2 modalities, metadata, recorded IR and saved figures
```

`ComponentRegistry` discovers installed entry points lazily. Plugins do not add
agents to a central hardcoded catalog. The graph resolves parent/child delegation
from agent configuration; a plugin factory accepts `delegation_tools`, not the
old `handoff_tools` parameter.

## Source Boundaries

| Component | Canonical implementation |
|---|---|
| Agent configuration and discovery | `lobster/config/agent_registry.py`, `lobster/core/component_registry.py` |
| Graph construction/delegation | `lobster/agents/graph.py` |
| Local client/session execution | `lobster/core/client.py` |
| Data manager | `lobster/core/runtime/data_manager.py` |
| Analysis IR | `lobster/core/provenance/analysis_ir.py` |
| Provenance tracker | `lobster/core/provenance/provenance.py` |
| Figure store/saver/snapshot | `lobster/core/figures/` |
| Template discovery/drawing | `lobster/figure_templates/`, `lobster/figure_template_providers/` |
| List-price data and normalization | `lobster/pricing/` |
| Runtime tool monitoring | `lobster/core/governance/aquadif_monitor.py` |
| Plugin contracts | `lobster/testing/contract_mixins.py` |

The `lobster/core/data_manager_v2.py` module is a compatibility shim. Use
`from lobster.core.runtime.data_manager import DataManagerV2` in new code.

## Modality and Service Contract

A modality is a named AnnData object, with observations, features, layers,
embeddings and analysis metadata. Services should not own a session/data manager
or perform agent routing. They return data, statistics and an `AnalysisStep`;
the wrapping tool stores data and logs the step.

```python
result_adata, stats, ir = service.analyze(adata, **parameters)
data_manager.store_modality("input_analyzed", result_adata)
data_manager.log_tool_usage("analyze_input", parameters, description="Analyzed input", ir=ir)
```

This is a pattern, not a complete implementation: services must supply real
scientific logic and an IR describing the transformation. `AnalysisStep` fields
include `operation`, `tool_name`, `description`, `library`, `code_template`,
`imports`, `parameters`, `parameter_schema`, `input_entities`, `output_entities`
and `replayable`. Do not use invented `activity_type` or `params` fields.

Set `replayable=False` explicitly for an outcome record that cannot transform the
notebook's current `adata`, for example retained external metadata. An exported
record is not evidence that the calculation was repeated. Test executable steps
against actual inputs and outputs; export alone does not establish scientific
correctness. Default export curates steps marked exportable; replay mode selects
those marked replayable and refuses failed steps or recorded unreplayable data
changes. Custom code using attempt-only figure helpers is excluded from executable
export. Inspect omission summaries instead of assuming the entire session is in
the notebook.

## Workspace and Session Identity

A local workspace root holds data, cache and `exports/`. Session provenance lives
under its `.lobster/sessions/<session_id>/` directory when a session is attached.
These are different paths and responsibilities. Named local queries should use
the same `-w` and `--session-id` at every step.

Default/relative DE and annotation exports use `data_manager.exports_dir`, not the
current shell directory. Inspect output paths instead of assuming every export
API uses the same destination or forbids absolute paths.

## Saved-Figure Boundary

Scientific analysis and figure rendering are separate operations. Ordinary tools
can suggest matching templates without writing a figure. Templates return
editable Plotly objects from retained data and do not fit a model, mutate a
dataset or save to disk.

The saved-figure path stages a candidate and commits through the trusted saver;
a saved revision has figure/revision identity and associated execution data.
A preview is not a saved revision. Do not directly write catalog files or use a
Plotly variable as evidence that a save happened.

Custom code explicitly calls `submit_figure`. No auto-capture is performed in 1.2.
A saved edit loads an exact revision and submits with that same `parent`; a new
revision preserves the old one. Frozen source data must be supplied explicitly
when regeneration data is needed. Template recipe metadata is a declaration,
not proof of which code was executed. Save confirmation pending is uncertainty,
not a completed save or a definite absence.

Data-package figure export is a view package: revision/figure/config bytes and
views, not a full reloadable store. Referenced source, execution, code and activity
artifacts remain in the workspace and are not copied; packaged provenance is a
session summary. Inspect scope/omission disclosures. Legacy flat plot files are
not supported by this export/open path and remain unchanged.

## Errors and Cost Evidence

Terminal `InferenceRefused` categories are context/auth/policy/budget/provider/
accounting. Recovery code must propagate them before logging raw exceptions or
returning a scientific fallback. Do not change inference authority silently.
The controlled provider path emits fixed categories rather than raw SDK bodies;
this is not a universal log-sanitization claim.

Offline list prices use exact provider/model keys and a packaged snapshot with
manual overrides. Unknown rates are not free inference, and local estimates are
not invoices or Cloud tariffs. Prompt-cache requests are off by default; direct
own-AWS Anthropic Bedrock usage can opt in with `LOBSTER_BEDROCK_PROMPT_CACHE=1`.
This setting does not establish a hit, saving or Cloud activation.

The shared `attach_code_tool` helper resolves availability by canonical registry
name, per-agent environment, global environment, workspace `config.toml`, then
declared default. Disabled factories do not construct the code service.
See [cli.md](cli.md#code-tool-availability) for defaults, exceptions and overrides.
Code runs as your own OS user, not within a confinement boundary; run only code you trust.

`/tokens` displays aggregate/per-agent values, not completeness or pricing snapshots.
`TokenTrackingCallback.get_usage_summary()` exposes `cost_complete`,
`usage_complete`, `unpriced_invocations`, `unpriced_models`, and
`attempt_coverage="callback_only"`, plus per-invocation pricing evidence where
available. Incomplete priced totals are lower bounds; withhold derived cost metrics.
Callback-only coverage does not establish a complete provider bill.

## Extending the Engine

See [creating-agents.md](creating-agents.md),
[creating-services.md](creating-services.md),
[plugin-architecture.md](plugin-architecture.md) and [testing.md](testing.md).
Use source signatures instead of copying old graph/client constructor examples.
