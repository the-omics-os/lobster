# Developer CLI Reference — Lobster AI 1.2

This is the **local Python engine** surface. npm `@omicsos/lobster` owns
`lobster cloud *` and npm onboarding. When both are installed, use
`uv run python -m lobster` to target local engine commands explicitly.

## Source Map

| Contract | Source |
|---|---|
| Options and entry points | `lobster/cli.py`, `lobster/__main__.py` |
| Chat/query execution | `lobster/cli_internal/commands/heavy/chat_commands.py`, `query_commands.py` |
| Slash/programmatic dispatch | `lobster/cli_internal/commands/heavy/slash_commands.py` |
| Scaffold/validation | `lobster/cli_internal/commands/light/scaffold_commands.py`, `validate_commands.py` |
| Agent inventory | `lobster/cli_internal/commands/light/agent_commands.py` |
| Workspace/session resolution | `lobster/cli_internal/commands/heavy/session_infra.py` |

Read these implementations when changing a command. Old command docstrings are
not proof that every listed slash command has a programmatic dispatcher branch.

## Local language

Chinese (`zh`) is a local UI option. English stays the complete catalog.

```bash
lobster --lang zh --help
lobster --lang zh init
# inside chat:
/lang zh
```

`zh-CN`, `zh-Hans`, `cn`, `chinese`, and `mandarin` select the same catalog. `LOBSTER_LANG` and `lang` in `.lobster_workspace/provider_config.json` do too, after `--lang`. An unknown code stays English. The Go screen reads the language when it starts.

When you add or change text a local user reads, add easy Simplified Chinese in the same change. `lobster/i18n/README.md` is the how-to. Copy `en.json`, `zh.json`, and `phrases.zh.json` into `lobster-tui/internal/i18n/catalogs/`. Leave command names, flags, `--json` keys, tool docstrings, and agent prompts in English.

## Basic Commands

```bash
lobster --version
lobster --help
lobster chat --help
lobster query --help
lobster agents list
lobster config-test --json
lobster query -w ./analysis --session-id study --json "Describe available data"
lobster command "workspace list" -w ./analysis --session-id study --json
```

Local `chat` uses Go-first `--ui auto` with classic fallback. Valid surfaces are
auto/go/classic, not ink. `--ui go` requires a TTY and available Go binary.
`--debug/-d` belongs on chat/query, not a global `lobster --debug` flag.

Local `query` supports `--workspace/-w`, `--session-id/-s`, `--provider/-p`,
`--model/-m`, `--json/-j`, trace options and stream switches. `--output/-o` is
an answer **file path**, not a text/json format selector. JSON queries emit
success/response/session_id and failure error information; failures exit nonzero.
Do not infer invented error codes for provider or scientific failures.

## Configuration

```bash
lobster config show
lobster config list-models
lobster config list-profiles
lobster config show-config
lobster config models
lobster config generate-env
```

`lobster init` is interactive unless `--non-interactive` is supplied with the
selected provider flags. Global settings/credential writes need permission.
There is no generic `lobster config set KEY VALUE` command.
Local Bedrock uses `AWS_BEDROCK_ACCESS_KEY` + `AWS_BEDROCK_SECRET_ACCESS_KEY`;
Azure uses `AZURE_AI_ENDPOINT` + `AZURE_AI_CREDENTIAL`. Do not conflate those
with npm Cloud `OMICSOS_API_KEY`.

## Code-Tool Availability

The supervisor, `data_expert_agent`, `metadata_assistant`, and
`visualization_expert_agent` default to enabled. Other participating built-in
experts default to disabled; `research_agent` does not offer the tool.

The switch controls only the `execute_custom_code` tool; it does not disable other tools, including tools that run shell commands or write files.

For an agent whose factory calls `attach_code_tool`, precedence is:

1. `LOBSTER_CODE_TOOL_<AGENT_NAME_UPPER>=1|0` (canonical registry name).
2. `LOBSTER_CODE_TOOL=all|none`; `default` defers to the layers below.
3. `<workspace>/config.toml`: `[agent_settings.<agent_name>] code_tool = true|false`.
4. The declared `AgentRegistryConfig.code_tool_default`, then `False`.

Per-agent values accept only `1`/`0` after trimming whitespace. Global modes are
trimmed and case-insensitive. Missing/blank environment values defer; invalid
values warn and skip only that layer. Workspace values must be TOML booleans.
No user/global configuration is read for this switch. A per-agent `1` overrides
global `none`; `none` is not a hard stop and the switch is not a security boundary.
A factory that never calls the helper does not gain a tool from `all`.

Without overriding environment variables:

```toml
# ./analysis/config.toml
[agent_settings.data_expert_agent]
code_tool = false
```

Custom code runs as your own OS user, not within a confinement boundary.
Run only code you trust. Its serialized result envelope is limited to **16 MiB**;
larger envelopes are `result_unreadable` and submitted figures are not saved.
Write large outputs to workspace files and return compact summaries/paths.
This limit is separate from LLM-facing text truncation; inspect completion/save outcomes.

**Source:** `lobster/tools/code_tool_switch.py`, `lobster/config/agent_registry.py`,
participating agent factories, and supervisor assembly in `lobster/agents/graph.py`.

## Scaffold and Validation

```bash
lobster scaffold agent --name domain_expert --display-name "Domain Expert" \
  --description "Analyze the selected biological domain"
lobster validate-plugin ./lobster-domain/
```

See [scaffold.md](scaffold.md). Static checks do not execute the scientific
workflow. The skill's bundled validator script is superseded by the installed
command. Use contract/scientific tests in addition to static validation.

## Sessions, Outputs and Figures

Named session continuation uses `--session-id`; `latest` chooses the most recent
local session in the selected workspace. Do not invent top-level `sessions list`
or `sessions resume` commands for the Python engine.

```bash
lobster command data -w ./analysis --session-id study --json
lobster command files -w ./analysis --session-id study --json
lobster command plots -w ./analysis --session-id study --json
lobster command "pipeline export" -w ./analysis --session-id study --json
```

`-w` is the actual workspace root. Session provenance is under
`.lobster/sessions/<session_id>/`; default/relative analysis tables use `exports/`.
Returned paths take precedence over assumed shell-current-directory output.
Exporting a notebook does not mean it was run.

Interactive `/plots` and `/plot <identifier>` list/open saved figures.
Custom-code figures require explicit `submit_figure`; a Plotly variable is not
saved automatically. Edits use an exact saved parent revision, retain originals
and report save outcomes. Preview is not a saved revision. Drawing templates
operate on retained data and do not fit models or save implicitly.

Interactive `/tokens` is distinct from the supported `lobster command`
dispatcher; do not assume `lobster command tokens` exists.

`/tokens` reports session totals and per-agent tokens/costs/calls; it does not
render completeness flags or per-invocation pricing snapshots. The callback's
`TokenTrackingCallback.get_usage_summary()` exposes `cost_complete`,
`usage_complete`, `unpriced_invocations`, `unpriced_models`, and
`attempt_coverage="callback_only"`. Invocation records can include
`cache_cost_status`, `cache_cost_reason`, and `cache_cost_snapshot` where available.
A total with `cost_complete=False` is a priced lower bound; withhold derived cost
metrics. Callback-only coverage does not establish a complete provider bill.
Inspect both usage and pricing completeness before using totals as evidence.

## Privacy and Inference Boundaries

Treat debug/stderr artifacts as sensitive. Typed inference refusals must not be
swallowed by scientific recovery or bypassed using other credentials.
Provider refusal sanitization is not universal log sanitization or code
confinement. Prompt-cache requests are off by default; direct own-AWS Anthropic
Bedrock can opt in with `LOBSTER_BEDROCK_PROMPT_CACHE=1`, without any promise of
cache hits, cost savings or Cloud activation.

For user orchestration/complete setup examples, read the `lobster-use` skill's
CLI and orchestration references. Cloud command syntax comes from the supported
npm client's help/source, not Python/Go compatibility code.
