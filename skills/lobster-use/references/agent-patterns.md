# Orchestration Patterns — Lobster AI

Use the local Python engine to offload biological workflows, not to hide a failed
analysis behind plausible prose. **Local Python using an Omics-OS Cloud account**
is still a Python install/setup task; do not replace it with npm. Only managed
Cloud-session workflows use the npm client. Choose the execution location before
installation; see [client selection](../SKILL.md#choose-the-correct-client-before-installing).

## Local Setup and Session Continuity

First complete the [local Python Cloud-account setup](../SKILL.md#local-python-setup-with-an-omics-os-cloud-account)
when that is the task: Python 1.2.2+ engine, default `[full]` analysis bundle,
then supported non-interactive init or environment-only provider selection.
Do not install npm instead, pin an older engine or call internal credential APIs
to work around missing public init flags. A configuration save is not verified
account access, and an empty agent inventory is not analysis-ready.

Assume `lobster` resolves to Python, or use the installed environment's explicit
interpreter, such as `.venv/bin/python -m lobster` in the fresh-project recipe.

```bash
lobster --version
lobster status
lobster agents list
```

For the Cloud-account route, Python `status` should display provider `omics-os`
with its configuration source. `Account Access: Not checked (configuration only)`
is deliberate: provider selection is not authenticated account access.

Do not install packages or write provider credentials automatically when these
checks fail. Obtain approval for the appropriate setup; see [CLI reference](cli-reference.md).
`lobster config-test --json` is **optional paid live verification**, not a passive
setup check: it invokes an LLM, may contact NCBI and probes workspace writes.
Only run after permission, checking exit status, JSON `valid` and the intended
provider. Stop on setup/inventory failures rather than proceeding to analysis.
Keep the same workspace and session on every query and inspection command:

```bash
WORKSPACE="./analysis"
SESSION="study"
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Inspect sample metadata and propose a QC plan before making changes"
lobster command "workspace list" -w "$WORKSPACE" --session-id "$SESSION" --json
```

`$WORKSPACE` is the actual root, not the parent of a nested hidden workspace.
For input loading and explicit saves, follow the
[save/reload requirements](../SKILL.md#local-session-history-is-not-a-data-save);
conversation continuity is not a substitute for those steps.

## Capture Both Failure Signals

A local JSON response can be present even when the process exits nonzero. Capture
that response without allowing shell `set -e` to discard it:

```bash
status=0
RESULT=$(lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Run the approved QC workflow" 2>"$WORKSPACE/query.stderr.log") || status=$?

if [ "$status" -ne 0 ] || ! printf '%s\n' "$RESULT" | jq -e '.success == true' >/dev/null; then
  printf '%s\n' "$RESULT" | jq -r '.error // .response // "No structured result"'
  # Keep stderr for private diagnosis; review/redact it before sharing.
  exit 1
fi
printf '%s\n' "$RESULT" | jq -r '.response'
```

The workspace must exist before redirecting stderr there. Do not infer success
from parseable JSON, reassuring prose or exit status alone. Missing/invalid JSON
is a failure to obtain a structured result, not permission to fabricate one.

## Inspect Without Inference

```bash
lobster command data -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command modalities -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command files -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command plots -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command "queue list" -w "$WORKSPACE" --session-id "$SESSION" --json
```

These are supported local commands without an LLM call. A successful inspection
command can still report an empty workspace. An inventory-read error is not an
empty or complete inventory. Inspect the returned tables, warnings and messages
before scheduling an analysis.

## Count-Safe Bulk RNA Workflow

1. Discover a dataset; inspect its metadata and data representations before
   downloading. Do not assume an accession contains a raw count matrix.
2. Request GEO download through
   `execute_download_from_queue(..., data_requirement="gene_counts")`. This
   tool option requires a suitable complete series-level bulk gene-count matrix
   with exact sample identity; it is not a CLI flag.
3. Read any `Requirement not met` refusal and candidate reasons. Ambiguity,
   missing counts, sample mismatch or changed numeric values require a new choice,
   not repeated unchanged downloads or substituting TPM/CPM.
4. Confirm biological replicate metadata and the contrast before DESeq2.
5. Preserve raw counts through QC/aggregation. Valid `layers['counts']` is
   preferred; absent that layer, `X` must pass count validation. An invalid counts
   layer is refused. Do not round normalized values to circumvent the check.
6. Inspect returned tables, count-source report and paths. Numeric whole-number
   values alone are not proof that an upstream pipeline produced raw counts.

For a **pre-approved simple two-group bulk comparison**, coding agents can load
a reviewed saved input, run dependent steps and request a supported save within
**one** non-interactive query. Set the file path relative to `$WORKSPACE` (or use
an absolute path) and supply the approved design; do not guess these values:

```bash
WORKSPACE="./analysis"
SESSION="study"
: "${COUNTS_FILE:?Set the reviewed saved raw-count file path}"
: "${GROUP_KEY:?Set the approved obs group column}"
: "${REFERENCE:?Set the approved reference group}"
: "${CASE:?Set the approved comparison group}"

lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Verify '$COUNTS_FILE' exists and choose its registered adapter; explicitly load it
   with load_modality in this query. Inspect raw-count origin evidence, replicate
   metadata and the approved contrast: formula='~$GROUP_KEY',
   contrast=['$GROUP_KEY', '$CASE', '$REFERENCE'] (case versus reference).
   Stop if raw-count origin is unestablished or metadata is missing; do not invent
   a design or substitute normalized data. Report any design, count, HDF-read or
   export-path refusal verbatim and stop; do not retry it through custom code.
   Run assess_data_quality and pass its resulting QC-assessed modality as
   pseudobulk_modality to run_de_with_formula with the approved formula/contrast
   and save_results=True. Report input/output modality names, the H5AD and CSV
   paths, save errors and any memory-only intermediates; do not ignore failed saves."

lobster command files -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command modalities -w "$WORKSPACE" --session-id "$SESSION" --json
```

The `modalities` check is a **new-process** view of saved H5ADs by file stem.
Use the failure checks above and inspect the actual saved/loaded outputs, not just
request text or a reassuring response. The final DE modality/table save does not
promise that every QC intermediate was saved. Before a later query, explicitly
load the needed saved file in that query and inspect its metadata/representation.
The formula tool's argument is named `pseudobulk_modality`; supplying a registered
bulk-count modality does not itself aggregate cells or create biological replicates.
`run_bulk_de_direct` still applies marker-style out-group filtering by
default; prefer `run_de_with_formula` for bulk designs. A `DE refused: no genes
remain after filtering.` reply is not a zero-DE biological result. Keep approved
comparison labels free of `/` to avoid the saved-result pointer limitation.
More complex or multi-factor designs require a separately agreed workflow; follow
the [design-question rule](../SKILL.md#scientific-workflow-rules), not this shortcut.

For a human who can operate an interactive terminal, `lobster chat` with `/save`
is the alternative; see the [save/reload requirements](../SKILL.md#local-session-history-is-not-a-data-save).

## Figure Workflow — Local Engine

Use retained analysis results to draw; do not recompute DE, clustering or chemistry
just to change a figure. Ready-made templates return editable Plotly figures and
never implicitly save them. Run `assess_data_quality` before `create_qc_plots` and
draw the full QC-assessed modality; drawing does not repair missing metrics or
accept DE/small-gene subsets as QC inputs.

A custom-code figure is retained only by explicit `submit_figure`. Creating
`fig`, calling a template or using Plotly display is not auto-capture. Check the
submission/save outcome, then inspect `lobster command plots --json` with the same
workspace/session. Preview-only results are not saved revisions.

For an edit request, use the exact saved `<figure_id>@<revision_id>`:

```python
# Inside execute_custom_code; replace the placeholder with a literal saved ref.
fig = load_figure("<figure_id>@<revision_id>")
fig.update_layout(title="Updated title")
submit_figure(fig, parent="<figure_id>@<revision_id>", intent="Change the title")
```

Filtering/restyling an existing figure is still an edit. Preserve its original
revision and check that the reply reports a newly saved revision. To retain
regeneration data for a new figure, supply `source` explicitly. Omitted config on
an edit inherits the parent's config; explicit `{}` clears it.

Interactive `/plots` and `/plot <identifier>` provide saved listing/opening.
Template discovery occurs through `list_templates()`, `get_template(name)` and
`show_template_source(name)` inside custom code, not an invented slash command.
These local figure behaviors do not assert Cloud feature availability.

## Outputs and Notebook Export

```bash
lobster command files -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command "pipeline export" -w "$WORKSPACE" --session-id "$SESSION" --json
find "$WORKSPACE/exports" -type f
```

Default/relative DE and annotation outputs use the workspace `exports/` directory.
Read returned paths for notebooks and other outputs; do not assume a root glob.
Default notebook export curates recorded steps; replay mode selects those marked
replayable and can refuse failed or unreplayable data-changing steps. Custom code
using attempt-only figure helpers is excluded from executable export. Inspect the
omission summary and any record-only steps. Export alone does not mean the notebook
was run or its scientific outputs checked.

## Cost, Privacy and Recovery

Bedrock cache points are opt-in (`LOBSTER_BEDROCK_PROMPT_CACHE=1`), intended for
direct own-AWS Anthropic inference, and off by default. No hit-rate or saving
follows merely from enabling them. The local price snapshot is offline list-price
data; exact provider/model keys and unknown rates matter.

`/tokens` shows aggregate and per-agent values, not completeness or pricing
snapshots. Detailed callback summaries expose `cost_complete`, `usage_complete`,
`unpriced_invocations`, `unpriced_models`, and `attempt_coverage="callback_only"`.
If pricing is incomplete, the priced sum is a lower bound; withhold derived cost
metrics. Check usage completeness separately. Callback-only coverage is not a
complete provider bill. See [cli-reference.md](cli-reference.md#code-tool-and-token-evidence)
for code access and detailed configuration; execution runs as Lobster's own OS user.

Do not recover a terminal inference refusal by silently switching credentials,
providers or authority. Auth/policy/budget/context/accounting failures need an
authorized resolution. Preserve diagnostic logs privately; fixed-category
provider refusals are not a guarantee that every log is safe to share.

## npm Managed Cloud Session Orchestration

Use npm CLI 0.4.2 `@omicsos/lobster` only for managed Cloud sessions, not as a
replacement for local Python using a Cloud account. `lobster-cli` is the npm alias;
its sign-in/status results do not establish a configured Python engine:

```bash
lobster-cli cloud login --device  # human approves the displayed URL/code
lobster-cli cloud status
lobster-cli cloud query --json "Describe the available session data"
lobster-cli cloud query --json --session-id latest "Continue the approved workflow"
lobster-cli cloud chat
```

For headless API-key access, supply `OMICSOS_API_KEY` from an approved secret
source. Browser/device login requires the user's participation. Preserve returned
session UUIDs; prefer an explicit UUID over `latest` when coordinating multiple
sessions. Check JSON `success` and error/finish information as well as process
status. The npm query path is programmatic, not interactive-only. Do not use a
stored credential's existence or a grep for account text as an authentication test.
