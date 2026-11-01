# CLI Reference — Local Lobster AI

Examples here target the Python engine unless marked **npm Cloud sessions**.
An Omics-OS Cloud account can supply inference for **local Python**: install
`lobster-ai` for that task, not npm alone. npm is for managed Cloud sessions.
If npm `lobster` is also installed, use `uv run python -m lobster <command>` in
an environment containing the Python engine to avoid selecting the npm binary.
In particular, npm `lobster init` is not local Python provider configuration.

## Installation and Local Configuration

Install or upgrade only with the user's permission. The Cloud init flags in this
reference require Python engine **1.2.2+**. Use `[full]` by default for local
analysis; provider-only extras can leave **no specialist agents**. `[full]` does
not include every optional domain: inspect installed/enabled task agents.

```bash
# Fresh approved POSIX project/environment; do not replace an existing .venv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lobster-ai[full]~=1.2.2'
.venv/bin/python -m lobster --version
.venv/bin/python -m lobster agents list

# Alternative user-wide tool install (not a project .venv):
uv tool install --python 3.12 'lobster-ai[full]~=1.2.2'
# For an existing tool install, upgrade with permission:
uv tool upgrade lobster-ai
```

In an existing approved `.venv`, use its Python with `uv pip install --python`
instead of recreating it. On Windows its interpreter is `.venv/Scripts/python.exe`.
If `uv` is missing, obtain permission to install it; do not fall back to npm or
bare pip. Do not pin an older public version that lacks these init flags or mix
package versions intentionally to bypass a setup failure.

### Local Python with an Omics-OS Cloud Account

Create a key at `https://app.omics-os.com/account` through the human
account owner. Prefer `OMICS_OS_API_KEY` securely injected into the Python process.
If the user authorizes a key file, Bash can read it without printing the value:

```bash
# Set OMICS_OS_KEY_FILE to the user-authorized path; do not enable set -x
export OMICS_OS_API_KEY="$(<"$OMICS_OS_KEY_FILE")"
```

Never source a key file as executable code. Use the Python interpreter installed
above, not an npm binary. For a fresh workspace config in the current directory:

```bash
(
  set -e
  umask 077
  : "${OMICS_OS_API_KEY:?Supply an authorized Omics-OS Cloud API key securely}"
  .venv/bin/python -m lobster init --non-interactive --use-omics-os \
    --skip-extras --skip-ssl-test </dev/null
  .venv/bin/python -m lobster status
  .venv/bin/python -m lobster agents list
)
```

- `--use-omics-os`: select the gateway; resolve a nonblank `OMICS_OS_API_KEY`, then
  the active saved Python platform profile. If that lookup is missing/blank,
  workspace-mode init tries CWD `.env`, then global Python `credentials.env`
  (global paths below). `--global` skips CWD `.env` and uses only the global file
  for this fallback. Saved files are parsed without exporting their values.
  An approved saved credential can supply init without the fresh-key ENV guard
  in the example. A saved OAuth token may refresh; this is not universally offline.
- `--omics-os-key <key>`: explicit Cloud key, used before lookup even with
  `--use-omics-os`. Blank explicit input fails rather than falling back. This
  flag can expose the key in process arguments; the environment route is preferred.
- Don't pass another provider's flags with either Cloud selector: the other
  explicit provider takes priority. No third-party provider key is required for
  Omics-only setup. `--cloud-key`/`LOBSTER_CLOUD_KEY` is a legacy supplement and
  does not select the `omics-os` gateway.
- Init's success message means **saved, not checked**. It does not verify account
  access, format, validity or inference, even if a nonempty key was accepted.
  Do not report authentication verified from init alone.
- Capture exit status and output. Missing/blank keys fail nonzero without a
  hidden prompt. Stop on failures; do not use private/internal credential APIs
  as a workaround. If `init --help` lacks the flags, upgrade with permission.

**Environment-only alternative (no init credential write):**

```bash
export LOBSTER_LLM_PROVIDER=omics-os
: "${OMICS_OS_API_KEY:?Supply an authorized Omics-OS Cloud API key securely}"
.venv/bin/python -m lobster status
```

Python `status` shows `Initialization`, `Provider` and `Configuration Source`.
For this route, expect provider `omics-os`; `Account Access` remains
`Not checked (configuration only)`. These fields display provider configuration,
not valid keys, authenticated identity or inference success. Stop if the displayed
provider is wrong; do not silently replace configuration.

The provider variable applies when no higher-priority provider configuration
exists. Selection priority is query/chat `--provider` > workspace provider JSON >
global provider JSON > `LOBSTER_LLM_PROVIDER`. For a later authorized query/chat,
`--provider omics-os` explicitly selects Cloud without rewriting configuration.
A subsequent analysis can still write data and use paid inference.

**Human route:** Python `lobster init` accepts a key in a masked Cloud prompt.
Bare Python `lobster cloud login` does not start a supported browser flow; it
returns guidance. Python `lobster cloud login --api-key <key>` is a separate
manual-key login that can validate/store a platform credential, but the flag
exposes the key in argv and login does not select the local provider. npm device
login requires human approval and is not a non-interactive setup shortcut;
configure Python separately rather than assuming shared credentials work.

`--skip-extras` also suppresses the entire end-of-wizard `uv tool` installation
prompt/handoff, not just provider extras. Agent choices can still be recorded and
an optional manual install command printed. Without it, tool-install offers retain
receipt extras and existing `--with` packages, add requested packages and include
`--reinstall`; displayed commands are shell-quoted. Run them only with permission.

### Cloud Init Writes and Overwrite

Workspace non-interactive init writes **CWD** `.env` (includes the key) and
`.lobster_workspace/provider_config.json`. The restrictive umask above protects
new files; it does not repair permissions on existing files or old backups.
Keep credential files/backups private and out of commits/shared logs.

Adding `--global` instead writes `providers.json` and `credentials.env` under
`$XDG_CONFIG_HOME/lobster` or `~/.config/lobster` on Unix, `%APPDATA%\\lobster` on
Windows. The global credential writer requests mode `0600`; inspect permission
warnings rather than assuming enforcement on every platform. No CWD `.env` is
written in global mode, but non-interactive init still creates CWD
`.lobster_workspace`; agent-selection flags can write agent configuration there.
Obtain permission for both global paths and any CWD side effects.

With non-interactive Cloud selected and without `--force`:
- Existing different-provider configuration fails nonzero and asks for `--force`.
  A JSON-only workspace still counts as existing configuration even with no CWD
  `.env`, including when that provider JSON cannot be loaded.
- Existing Cloud configuration with a usable key can return "already configured"
  and exit 0 **without rewriting**. A supplied replacement key is not thereby
  persisted; this is not key rotation or account verification.
- Missing required credentials fail before replacement/backup. Do not interpret
  any other init mode's existing-config exit 0 as a completed reconfiguration.

Use `--force` only after approval to replace provider/credential configuration.
Non-interactive Cloud setup backs up each existing workspace artifact before
replacing either: `.env.backup.YYYYMMDD_HHMMSS` in CWD and
`.lobster_workspace/provider_config.json.backup.YYYYMMDD_HHMMSS`. A failed backup
stops setup before replacement. Global mode retains
`providers.json.backup.YYYYMMDD_HHMMSS`, not a complete backup of all secrets:
global `credentials.env` can be overwritten, and the provider backup is not a
credentials rollback. Read output and inspect nonsecret status, not file values.

Required Cloud provider/credential write failures exit nonzero, without a saved
success claim or restarting another wizard. The diagnostic says **partial
configuration may remain; setup is not complete**. Stop and inspect the reported
failure privately before an approved repair; do not assume transactional rollback
or a complete setup because one file was written.

Keep init key lookup separate from dotenv/runtime loading: global credentials
fill unset environment values; an applicable workspace/fallback `.env` can load
with overwrite enabled. An exported key does not universally defeat an old
`.env`. Do not read or print credential files merely to inspect configuration.

### Other Local Providers and Live Verification

These are alternatives to Omics-OS, not extra requirements for a Cloud account:

```bash
lobster init --non-interactive --anthropic-key "$ANTHROPIC_API_KEY" --profile production
lobster init --non-interactive --gemini-key "$GOOGLE_API_KEY"
lobster init --non-interactive --openai-key "$OPENAI_API_KEY"
lobster init --non-interactive --openrouter-key "$OPENROUTER_API_KEY"
lobster init --non-interactive --nebius-key "$NEBIUS_API_KEY"
lobster init --non-interactive --bedrock-access-key "$AWS_BEDROCK_ACCESS_KEY" \
  --bedrock-secret-key "$AWS_BEDROCK_SECRET_ACCESS_KEY"
lobster init --non-interactive --azure-endpoint "$AZURE_AI_ENDPOINT" \
  --azure-credential "$AZURE_AI_CREDENTIAL"
lobster init --non-interactive --use-ollama --ollama-model "llama3:8b-instruct"
```

A key expanded into a flag can still be visible in process arguments. Do not echo
it or capture it in shared logs. Masked interactive setup is a human option,
not a machine requirement.

`lobster config-test --json` **invokes a paid LLM test**, may contact NCBI and
writes/deletes a workspace probe. Run only after explicit permission for those
operations, not as an automatic setup-only check. Require exit 0 and JSON
`valid: true`; for a Cloud setup also require
`checks.llm_provider.provider: "omics-os"`. Preserve refusals/failures privately
and stop; do not switch provider/credentials or claim a successful analysis from
this small connectivity test.

| Init flag | Meaning |
|---|---|
| `--non-interactive` | Machine mode, no wizard; select a provider explicitly |
| `--use-omics-os` / `--omics-os-key` | Non-interactive Cloud gateway selection/key as above |
| `--model`, `-m` | Selected provider's model ID |
| `--profile` | Anthropic/Bedrock profile: development, production, performance, max |
| `--ncbi-key` | Optional NCBI key |
| `--agents` | Comma-separated enabled agent identifiers |
| `--preset` | Agent selection preset, e.g. scrna-basic |
| `--skip-extras` | Skip optional installs and the end-of-wizard tool handoff; agent choices can still be recorded |
| `--skip-ssl-test` | Skip init SSL/connectivity test, not all network activity |
| `--skip-docling` / `--install-docling` | PDF intelligence installation choice |
| `--global`, `-g` | Global provider/credential writes plus possible CWD side effects |
| `--force`, `-f` | Approved overwrite, with the limited backup described above |
| `--ui` | Interactive init surface: auto, go, classic |

Local Bedrock names are `AWS_BEDROCK_ACCESS_KEY` and
`AWS_BEDROCK_SECRET_ACCESS_KEY`; Azure's canonical pair is `AZURE_AI_ENDPOINT`
and `AZURE_AI_CREDENTIAL`. These are distinct from npm Cloud authentication.

## Local Chat

```bash
lobster chat -w ./analysis --session-id study
lobster chat --ui go           # requires an interactive terminal and Go binary
lobster chat --ui classic      # classic terminal
lobster chat --classic         # equivalent surface selection
lobster chat --no-intro
```

Default `--ui auto` tries Go, then classic. `--ui ink` is not valid for local chat.
Trace options can select the classic path instead of the Go fast path.

Chat is the human/interactive option. See the
[save/reload requirements](../SKILL.md#local-session-history-is-not-a-data-save);
coding agents should use the [non-interactive recipe](agent-patterns.md#count-safe-bulk-rna-workflow).

| Flag | Meaning |
|---|---|
| `--workspace`, `-w` | Workspace root; otherwise default or LOBSTER_WORKSPACE |
| `--session-id`, `-s` | Named session; `latest` selects most recent |
| `--provider`, `-p` | Provider override |
| `--model`, `-m` | Model override |
| `--reasoning` | Reasoning display |
| `--verbose`, `-v` | Tool/agent trace detail |
| `--debug`, `-d` | Debug logging; handle logs as sensitive |
| `--stream` / `--no-stream` | Text streaming; default on in chat |

## Local Single-Turn Query

```bash
lobster query -w ./analysis --session-id study --json "Describe the loaded data"
lobster query -w ./analysis --session-id latest --json "Inspect saved inputs before continuing"
lobster query -w ./analysis --session-id study --output answer.md "Summarize results"
```

Use the same workspace **and** session throughout a multi-step workflow. For
input loading and explicit tool saves, follow the
[save/reload requirements](../SKILL.md#local-session-history-is-not-a-data-save)
and [agent-drivable recipe](agent-patterns.md#count-safe-bulk-rna-workflow).

| Flag | Meaning |
|---|---|
| `--session-id`, `-s` | Conversation/session continuity; not an implicit data save |
| `--workspace`, `-w` | Actual workspace root, not parent of an added hidden directory |
| `--json`, `-j` | Structured output on stdout |
| `--output`, `-o` | Final answer file path, **not** output format |
| `--provider`, `-p`; `--model`, `-m` | Inference overrides |
| `--reasoning`; `--verbose`, `-v`; `--debug`, `-d` | Optional diagnostics |
| `--stream` / `--no-stream` | Default off in query; JSON uses complete output |

Success JSON includes `success`, `response`, `session_id` and may include
`last_agent`, `token_usage`. Failure JSON includes `error`; failed local queries
exit nonzero. A valid JSON document alone is not a successful analysis.

## Local Workspace Commands — No LLM Call

```bash
lobster command "workspace list" -w ./analysis --session-id study --json
lobster command data -w ./analysis --session-id study --json
lobster command modalities -w ./analysis --session-id study --json
lobster command files -w ./analysis --session-id study --json
lobster command plots -w ./analysis --session-id study --json
lobster command "queue list" -w ./analysis --session-id study --json
lobster command "pipeline export" -w ./analysis --session-id study --json
```

`command` takes one quoted command string, optional workspace/session and `--json`.
Leading `/` is stripped, but **only the supported dispatcher commands work**.
Do not assume every interactive slash command has a programmatic counterpart.
For example `/tokens`, `/plot` and `/open` are interactive controls, not
`lobster command` branches. `lobster command "read <file>"` is supported; inspect
any read refusal rather than retrying through custom code.

Unknown dispatcher commands exit 1; with `--json`, `success` is false. Machine
`pipeline run` likewise exits nonzero for missing provenance, prompt EOF or a
failure summary from its shared execution helper. Check both exit status and JSON;
a success summary is not biological validation or proof of notebook reproducibility.

Normal command JSON carries `success`, `command` and `data`. Inspect structured
`data.blocks` (including table/kv blocks) and text in `data.messages` when present;
empty output collections can be omitted. Derived `data.tables` and
`data.code_blocks` may also appear for matching blocks. A top-level `summary` is
optional for some dispatcher results, not the structured output. Inspect the actual blocks
and messages rather than treating command success as proof that a dataset exists.

## Interactive Controls

| Control | Purpose |
|---|---|
| `/data`, `/modalities`, `/describe <name>` | Inspect loaded data |
| `/files`, `/tree`, `/read <file>`, `/open <file>` | File inspection/opening |
| `/workspace list`, `/workspace info <sel>`, `/workspace load <sel>` | Dataset inventory/loading |
| `/workspace remove <sel>`, `/workspace save` | Dataset management; removal needs approval |
| `/queue`, `/queue list`, `/queue clear`, `/queue export` | Download/publication queue management |
| `/metadata`, `/metadata samples`, `/metadata exports` | Metadata overview |
| `/save [--force]`, `/restore [pattern]` | Persist/restore modalities |
| `/pipeline list`, `/pipeline info`, `/pipeline export`, `/pipeline run` | Notebook listing, export and explicit execution |
| `/export [--no-png] [--force]` | Export a data package; inspect warnings and returned path |
| `/plots`, `/plot <identifier>` | Saved figure listing/opening |
| `/config`, `/config provider list`, `/config model list` | Inspect inference configuration |
| `/config provider <name> [--save]`, `/config model <name> [--save]` | Switch provider/model; approval required for new inference authority |
| `/tokens` | Session token/cost display |

Notebook export/run needs a session containing recorded analysis history. Export
is a selected subset, not evidence of execution. `/pipeline export --replay`
selects steps marked replayable; it refuses failed steps and recorded unreplayable
data changes. Custom code using attempt-only figure helpers is excluded from
executable export. Inspect the omission summary, notebook and outputs independently
before using them as evidence.

## Local Figures and Revisions

`/plots` lists saved figures. `/plot` without an identifier opens the plot view
directory; an identifier selects a saved plot by ID/title, including saved
`<figure_id>@<revision_id>` references. Prefer exact references for an edit.

Run `assess_data_quality` before `create_qc_plots`, then draw the full QC-assessed
modality, not a DE/small-gene subset. Drawing neither computes nor repairs missing
QC metrics; do not manufacture columns to bypass a refusal.

Inside `execute_custom_code`, a new Plotly figure is saved only through explicit
submission, after the code succeeds:

```python
# `rows` is the retained input table, not a newly rerun analysis.
fig = get_template("generic.scatter")(rows, x="x", y="y", title="Measurements")
submit_figure(fig, source=rows, intent="Compare x and y", label="Measurements")
```

For a requested edit, replace the placeholders below with the literal ref returned
by the previous save (do not pass a title in place of revision identity):

```python
fig = load_figure("<figure_id>@<revision_id>")
fig.update_layout(title="Revised title")
submit_figure(fig, parent="<figure_id>@<revision_id>", intent="Change the title")
```

The edit creates a revision of the same figure, retaining the original. Omitted
`config` inherits the parent's saved config; `config={}` clears it. Supply a
`source` explicitly when regeneration data is needed. Saved source helpers include
`list_figures()`, `load_figure_config(ref)` and `load_source(source_id)`; they read
the attempt-start snapshot, so retry with a fresh attempt if a ref is not yet
available. Write an older revision ref literally in the code to request it.

`list_templates()`, `get_template(name)` and `show_template_source(name)` expose
the installed template catalog. There is no `/templates` command implied here.
Templates return editable figures and do not save, mutate datasets or fit models.

A plot tool's `save_plot=False` is a preview, not a stored revision. Creating a
`fig` variable or displaying it is not implicit submission: the reply names
unsubmitted Figure variables and they are not saved. Check the returned save
outcome and saved listing before reporting a figure as available. `Save confirmation
pending.` is uncertainty about confirmation, not a definite absence or completed
save. Legacy flat plot files remain unchanged and are not supported by this
`/plot` opening or package export path; inspect the reported warnings.

A custom-code serialized result envelope is limited to **16 MiB**; larger envelopes
are `result_unreadable` and submitted figures are not saved. Write large outputs
to workspace files and return compact summaries/paths. This is separate from
LLM-facing text truncation; always inspect completion and save outcomes.

## Code Tool and Token Evidence

The supervisor, `data_expert_agent`, `metadata_assistant`, and
`visualization_expert_agent` default to custom code enabled; other participating
built-in experts default to disabled. `research_agent` does not offer the tool.

The switch controls only the `execute_custom_code` tool; it does not disable other tools, including tools that run shell commands or write files.

For participating factories, precedence is per-agent environment, global
environment, workspace `config.toml`, then the declared default (otherwise off):

- `LOBSTER_CODE_TOOL_<AGENT_NAME_UPPER>=1|0`, using the canonical registry name;
  values are whitespace-trimmed, not `true`/`false`.
- `LOBSTER_CODE_TOOL=all|none|default`; global modes are whitespace-trimmed and
  case-insensitive. `default` defers to workspace/default settings.
- `[agent_settings.<agent_name>] code_tool = true|false` in the chosen
  workspace's `config.toml`; these must be TOML booleans.

Missing/blank environment values defer; invalid values warn and skip only that
layer. No user/global configuration is read for this switch. A per-agent `1`
overrides global `none`; a factory without code-tool wiring is unaffected by `all`.

Without environment overrides, a workspace setting can disable an agent:

```toml
# ./analysis/config.toml
[agent_settings.data_expert_agent]
code_tool = false
```

Custom code runs as your own OS user, not within a confinement boundary.
Run only code you trust.

Prompt caching is off unless `LOBSTER_BEDROCK_PROMPT_CACHE=1`. The opt-in is intended
for direct own-AWS Bedrock Anthropic inference; it requests cache points, not a
promised hit or saving. No Cloud activation follows from this setting.

Price lookup uses the packaged offline snapshot and explicit manual overrides.
Unknown prices remain unknown. Local list-price estimates are neither invoices
nor Cloud billing tariffs.

`/tokens` displays session totals and per-agent tokens/costs/calls; it does not
show completeness flags or per-invocation pricing snapshots. Detailed callback
usage summaries expose `cost_complete`, `usage_complete`, `unpriced_invocations`,
`unpriced_models`, and `attempt_coverage="callback_only"`. Invocation records can
include `cache_cost_status`, `cache_cost_reason`, and `cache_cost_snapshot` where
available. A total with `cost_complete=False` is a priced lower bound; withhold
derived cost metrics. Check usage and pricing completeness independently;
callback-only coverage does not establish a complete provider bill.

## Exports and Errors

The workspace's `exports/` directory receives default/relative analysis outputs,
including DE/annotation tables. Returned paths are authoritative; do not assume
the shell's current directory or a root-level notebook glob. Native
analysis export tools refuse destinations outside the workspace unless the user
listed an existing absolute folder in `LOBSTER_EXPORT_ROOTS` before startup
(`:`-separated, `;` on Windows). Relative, missing and non-folder entries are
ignored; restart after changing the setting. Absolute dataset-save paths and
custom code are not restricted by this export setting: it is not confinement.

After resuming a saved session, `/pipeline export` can repeat steps;
inspect for duplicated filtering/normalization before execution.

Data-package figure export is a **view package**, not a complete reloadable figure
store. It copies saved revision/figure/config bytes and views, but not the source,
execution, code or activity artifacts those revisions reference. A packaged
`provenance.json` is a session summary, not those artifacts. Inspect package scope,
`Not included`, and `Not copied` disclosures before sharing or assuming restoration.

Keep diagnostic files private. Typed inference refusals distinguish context,
auth, policy, budget, provider and accounting failures. Do not suppress them or
switch credentials to make an analysis continue. This is not a guarantee that
all logs or every exception are free of sensitive content.

## npm Cloud Sessions — Not Local Python Setup

Use this section only for a managed Cloud session. It does not install Python
`lobster-ai` or configure its local `omics-os` provider.
These examples use npm CLI 0.4.2 and its `lobster-cli` alias:

```bash
npm install -g @omicsos/lobster@0.4.2  # permission required
lobster-cli cloud login --device     # human browser authorization, not a Python login
lobster-cli cloud status
lobster-cli cloud chat
lobster-cli cloud query --json --session-id latest "Continue the session"
lobster-cli cloud logout
```

Device login requires human browser approval and can write credentials under
`~/.config/omics-os/`; get permission before those writes. Do not read or print
credential files to inspect configuration. Do not treat successful npm sign-in
as proof of a Python installation, local provider selection or working local
inference; use the local Python setup route for that task.

API-key auth accepts `OMICSOS_API_KEY` from
the environment; `--api-key <key>` also exists but can expose the key in process
arguments. Cloud query supports `--session-id/-s`, `--project-id/-p`, `--json/-j`
and `--stream`. Store returned session UUIDs for continuity. Do not infer Cloud
figure capabilities from this local-engine reference.

**Source:** Python `lobster/cli.py`, `lobster/cli_internal/commands/`,
`lobster/tools/custom_code_tool.py`, `lobster/core/figures/child_api.py`;
Cloud CLI: `@omicsos/lobster` command help and source.
