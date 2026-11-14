# Code Access, Cost and Failure Handling

The supervisor, `data_expert_agent`, `metadata_assistant`, and
`visualization_expert_agent` have custom code enabled by default. Other
participating built-in experts default to disabled; `research_agent` does not
offer the tool. Override with `LOBSTER_CODE_TOOL_<AGENT_NAME_UPPER>=1|0` or global
`LOBSTER_CODE_TOOL=all|none|default`; workspace `config.toml` can also set
`[agent_settings.<agent_name>] code_tool = false`. Per-agent environment beats
global environment, then workspace settings, then the declared default;
`default` defers. Per-agent `1` overrides global `none`.

The switch controls only the `execute_custom_code` tool; it does not disable other tools, including tools that run shell commands or write files.

Custom code runs as your own OS user, not within a confinement boundary.
Run only code you trust. Its serialized result envelope is limited to **16 MiB**;
larger envelopes are unreadable and submitted figures are not saved. Write large
outputs to workspace files and return compact summaries/paths.
See [CLI reference](cli-reference.md#code-tool-and-token-evidence)
for exact value handling and configuration scope.

Bedrock prompt caching is **OFF by default**. For direct own-AWS Bedrock use with
Anthropic models, `LOBSTER_BEDROCK_PROMPT_CACHE=1` opts in to cache-point requests.
This does not promise a cache hit or a saving; do not extend it to a Cloud
activation claim.

Prices come from a packaged offline snapshot with manual overrides, not a live
price fetch. Exact provider/model identity matters; missing rates are unknown,
not zero. Local estimates are not provider invoices or Omics-OS Cloud tariffs.

`/tokens` displays totals and per-agent tokens/costs/calls, not completeness or
pricing snapshots. Detailed callback usage summaries expose `cost_complete`,
`usage_complete`, `unpriced_invocations`, `unpriced_models`, and
`attempt_coverage="callback_only"`. If pricing is incomplete, the priced sum is a
lower bound; withhold derived cost metrics. Usage completeness is separate from
pricing completeness, and callback-only coverage is not a complete provider bill.

Check both process status and JSON `success`. Local `query --json` failures return
an error and exit nonzero. Preserve stderr privately; do not paste raw logs,
credentials, patient data or provider request payloads into external reports.
Typed inference refusals must not be worked around by switching credentials,
authority or an unrelated scientific fallback. Report the limitation and ask for
an authorized next step.

Default/relative analysis exports use the workspace's `exports/` directory; inspect
returned paths rather than assuming files were written to the shell's current
directory. Notebook export is a selected subset of recorded steps, not evidence
of execution. Custom code using attempt-only figure helpers is excluded from
executable export; inspect the omission summary. Replay mode selects steps marked
replayable and can refuse failed or unreplayable data-changing steps. Inspect and
run the notebook separately before making any statement about its results. After
resuming a saved session, `/pipeline export` can contain steps twice; inspect for
repeated filtering/normalization before executing it.
