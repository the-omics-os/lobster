# Charm TUI Implementation — State

**Branch:** `feature/charm-ui` (worktree at `/Users/tyo/Omics-OS/lobster-charm-ui/`)
**Started:** 2026-03-04
**Last updated:** 2026-03-05
**Snapshot basis:** commits (`516186d`, `ac7a60e`, `642ad8d`, `fd142fb`) plus HITL design session 2026-03-05

## Phase Status

| Phase | Name | Status | Commits | Notes |
|-------|------|--------|---------|-------|
| 0 | Init Wizard | **COMPLETE** | `8477338` | Go init wizard + bridge bootstrapping |
| 1 | Protocol Foundation & Streaming Chat | **COMPLETE** | `78f7df8`..`742c99c` | IPC bridge, streaming, startup hardening |
| 2 | Rendering & UX Polish | **COMPLETE** | `78f7df8`..`742c99c` | markdown render, loading/status polish |
| 3 | Protocol Handlers & Native Interactions | **COMPLETE** | `516186d` | forms/select/progress/confirm plumbing captured |
| 4 | Slash Wiring + Autocomplete Baseline | **COMPLETE** | `516186d` | Go-native slash handling + bridge dispatch |
| 4.2 | Parity Hardening | **COMPLETE** | `ac7a60e` | `/open`, `/restore`, `/config` direct switch, protocol-safe branches |
| 5 | Rich CLI Functional/Visual Parity Migration | **IN PROGRESS** | `ac7a60e`, `642ad8d` | history ring, completion scaffold, native `/exit` confirm flow |
| 6A | HITL Interrupt Infrastructure (Python) | NOT STARTED | — | `ask_user` tool, component mapper, interrupt detection, classic fallback |
| 6B | HITL Protocol Extension (Bridge + Go types) | NOT STARTED | — | `component_render/response/close` messages, Go-side routing |
| 6C | BioCharm Go Components | NOT STARTED | — | `BioComponent` interface, registry, individual domain components |
| 7 | Distribution & Cross-Platform | NOT STARTED | — | deferred |
| 8 | SSH & Cloud | FUTURE | — | deferred |

## Current Baseline (2026-03-05)

- Go command history ring is active with draft restore behavior and strict `Up/Down` ownership (history recall only; no transcript scroll side effects) (`lobster-tui/internal/chat/model.go`).
- Path completion protocol is active (`completion_request`/`completion_response`) for `/read`, `/open`, `/workspace load` (`lobster-tui/internal/chat/completions.go`, `lobster/cli_internal/go_tui_launcher.py`).
- Go UX interaction polish landed for daily command entry: history recall now works while slash suggestions are visible, path-tail completion parsing preserves spaces/quotes, local `/clear` and protocol clear paths are unified, and select `Ctrl+C` returns the current selection (`lobster-tui/internal/chat/model.go`, `lobster-tui/internal/chat/completions.go`).
- Streaming viewport behavior now preserves user scroll position while content is streaming (auto-follow only when already at bottom; `TypeDone` still forces bottom) (`lobster-tui/internal/chat/model.go`).
- Go transcript retrieval hardening landed: dedicated `PgUp/PgDn` viewport paging now works even while drafting input, and the viewport now renders a right-edge scrollbar thumb/track when scrollable (`lobster-tui/internal/chat/model.go`). Mouse interaction is now explicit and reversible: inline mode still defaults to selection-first, fullscreen still defaults to scroll-first, `Ctrl+G` toggles live between `select` and `scroll` mouse modes, and the current mode is always visible in the status line (`lobster-tui/internal/chat/model.go`, `lobster-tui/internal/chat/run.go`, `lobster-tui/internal/chat/run_test.go`).
- Go launcher runtime metadata now resolves the active provider via `ConfigResolver` for local clients instead of echoing the raw override placeholder, so inline/provider chrome reflects the actual backend (for example `ollama`) at startup and after `/config provider ...` changes (`lobster/cli_internal/go_tui_launcher.py`, `tests/unit/cli/test_go_tui_launcher_completions.py`).
- Fatal startup diagnostics are now shared across UI paths: provider package/import failures, invalid `--provider`, missing credentials, and no-provider-configured startup failures are classified once, rendered via Rich in classic/query, and emitted as protocol `alert` messages in Go mode without traceback or Rich leakage (`lobster/cli_internal/startup_diagnostics.py`, `lobster/cli_internal/commands/heavy/session_infra.py`, `lobster/cli_internal/go_tui_launcher.py`, `tests/unit/cli/test_startup_diagnostics.py`).
- Go visual/readability polish landed for alert and layout ergonomics: stronger alert hierarchy, adaptive narrow-terminal width clamps, consistent prompt/tool/progress spacing, and theme contrast tuning (`lobster-tui/internal/chat/views.go`, `lobster-tui/internal/theme/theme.go`).
- `/config show` visual polish landed across Python + Go: empty sections now collapse to compact notes, the loose command-hint footer is replaced with a structured reference table, and Go protocol tables render as width-aware preformatted blocks for better long-column alignment (`lobster/cli_internal/commands/light/config_commands.py`, `lobster-tui/internal/chat/model.go`, `tests/golden/slash_commands/config_show.json`).
- `/config model` visual polish landed across Python + Go: the command now leads with a structured active-selection summary, lists provider models in a cleaner status-aware table, and replaces the loose usage footer with a structured command reference table (`lobster/cli_internal/commands/light/config_commands.py`, `tests/golden/slash_commands/config_model.json`).
- Inline startup welcome now follows the `animated_logo.md` blueprint with your custom 4-line glyph logo: fade-in + 50ms left-to-right ATCG scramble, then persistent low-frequency spark flickers while keeping logo/taglines visible during chat (`lobster-tui/internal/chat/views.go`, `lobster-tui/internal/chat/model.go`).
- Go chat now supports a low-noise automation path: `lobster chat --ui go --no-intro` (or `LOBSTER_TUI_NO_INTRO=1`) suppresses the inline welcome block plus pre-ready spinner/tip animation so PTY-driven agents can interact with the TUI without startup redraw noise (`lobster/cli.py`, `lobster/cli_internal/go_tui_launcher.py`, `lobster-tui/internal/chat/run.go`, `lobster-tui/internal/chat/model.go`).
- `/exit` in Go mode now uses native confirm prompt with explicit cancel/quit outcomes (`lobster-tui/internal/chat/model.go`, `lobster-tui/internal/chat/model_test.go`).
- Go mode now prints a post-exit footer after TUI teardown with session resume hint, feedback/issues/contact links, runtime metadata, token summary, and next-step commands for continuity (`lobster/cli_internal/go_tui_launcher.py`, `tests/unit/cli/test_go_tui_exit_footer.py`).
- Protocol table sanitization now strips Rich markup from titles/headers/cells before emitting non-Rich adapters, preventing literal `[cyan]...[/cyan]` leakage in Go transcripts (`lobster/cli_internal/commands/output_adapter.py`).
- Init wizard parity hardening landed for `lobster init --ui go` / questionary fallback: package selections now normalize to canonical agent ids before install/config write, Ollama setup surfaces detected local models plus curated presets and custom entry, and Smart Standardization / vector-search is restored as a native setup step with persisted config and regression coverage.
- Init package installation no longer re-discovers freshly editable-installed agent packages inside the same Python process, which removes the false `Failed to load lobster.agents ... No module named ...` warning burst after agreeing to install agents during init (`lobster/cli_internal/commands/heavy/init_commands.py`, `tests/unit/cli/test_init_tui_ux.py`).
- Adjacent one-shot CLI polish landed for `lobster query`: human mode now renders as a compact report-style response (header + markdown + single-line footer) instead of a large boxed panel, `--json` remains machine-safe and unchanged, and component discovery now lazy-loads entry-point groups so unrelated optional download-service imports do not leak startup warnings into simple query runs (`lobster/cli_internal/commands/heavy/query_commands.py`, `lobster/core/component_registry.py`, `tests/unit/cli/test_query_commands.py`, `tests/unit/core/test_component_registry.py`).
- Query debug-path credibility pass landed: `--reasoning` / `--verbose` runs now emit an immediate `Working…` line instead of dead-air, final human rendering prefers structured `text` over combined `[Thinking: ...]` payloads, live reasoning is not duplicated when callbacks already showed it, and verbose runs append a compact execution summary derived from callback events (`lobster/cli_internal/commands/heavy/query_commands.py`, `lobster/utils/callbacks.py`, `tests/unit/cli/test_query_commands.py`, `tests/unit/utils/test_callbacks.py`).
- Priority slash regressions fixed in Python dispatch path: `/open`, `/restore`, `/config provider <name>`, `/config model <name>`, `/save` protocol-safe branch (`lobster/cli_internal/commands/heavy/slash_commands.py`).
- Protocol safety hardening landed for `/clear` and `/exit` in Python slash dispatch when running under protocol adapter.
- Golden transcript checks now cover core protocol command surfaces (`/help`, `/session`, `/status`, `/tokens`) plus family-baseline commands (`/workspace list`, `/queue`, `/metadata publications`, `/config`, `/config provider`, `/config model`, `/pipeline`) via `tests/integration/test_slash_command_golden_transcripts.py` with committed fixtures in `tests/golden/slash_commands/`.
- Core visual-prettification updates landed in slash dispatch: `/help` now includes explicit admin/UI section; `/session` and `/status` include next-step hints in protocol mode.
- Command matrix maintained in `.planning/charm-ui/PARITY_MATRIX.md` (current snapshot: no `blocked` command rows).
- Regression coverage expanded in Go chat tests for status/spinner parsing, select navigation + `Ctrl+C`, clear-path parity, and streaming autoscroll behavior (`lobster-tui/internal/chat/model_test.go`, `lobster-tui/internal/chat/completions_test.go`).

## Validation Run (2026-03-05)

- `cd lobster-tui && go test ./...` passed.
- `.venv/bin/python -m pytest tests/unit/cli/test_go_tui_launcher_completions.py tests/integration/test_go_tui_protocol_smoke.py tests/unit/cli/test_slash_commands_go_tui_regressions.py -q` passed.
- `uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py tests/unit/cli/test_slash_commands_go_tui_regressions.py tests/integration/test_go_tui_protocol_smoke_script.py` passed.
- `cd lobster-tui && go test ./internal/chat ./internal/protocol` passed.
- `uv run pytest -q tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py` passed.
- `uv run pytest -q tests/integration/test_go_tui_protocol_smoke.py tests/integration/test_go_tui_protocol_smoke_script.py` passed.
- `uv run pytest -q tests/unit/cli_internal/commands/test_output_adapter.py` passed.
- `uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py` passed after intentional golden refresh for sanitized table payloads.
- `uv run pytest -q tests/unit/cli/test_go_tui_exit_footer.py tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py` passed.
- `cd lobster-tui && go test ./internal/chat ./internal/protocol` passed.
- `cd lobster-tui && go test ./internal/chat -run TestShouldEnableMouseCapture` passed.
- `cd lobster-tui && go test ./internal/chat ./internal/protocol` passed after `/config show` table-rendering updates.
- `uv run pytest -q tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py` passed after `/config show` output restructuring.
- `uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py` passed after adding `/config show` golden coverage.
- `cd lobster-tui && go test ./internal/chat ./internal/protocol` passed after `/config model` output cleanup.
- `uv run pytest -q tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py` passed after `/config model` switch-breadth regression coverage was added.
- `uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py` passed after adding `/config model` golden coverage.
- `uv run pytest -q tests/unit/cli/test_query_commands.py tests/unit/core/test_component_registry.py tests/unit/cli/test_cli_decomposition.py` passed after `lobster query` renderer compaction and component-registry lazy-loading changes.
- `uv run pytest -q tests/unit/cli/test_query_commands.py tests/unit/utils/test_callbacks.py::TestTerminalCallbackVisibility::test_reasoning_visibility_flag_tracks_rendered_reasoning` passed after `lobster query` debug-path activity/reasoning separation updates.
- `uv run pytest -q tests/unit/cli/test_startup_diagnostics.py tests/unit/cli/test_go_tui_launcher_completions.py tests/unit/cli/test_slash_commands_go_tui_regressions.py tests/unit/cli/test_query_commands.py tests/unit/cli/test_cli_decomposition.py` passed after startup-diagnostics unification and regression coverage were added.
- Manual PTY validation passed for:
  - `uv run lobster chat --ui go --no-intro --provider invalid`
  - `uv run lobster chat --ui go --no-intro --provider bedrock`
  - clean-room no-config startup via `env -i ... uv run --project /Users/tyo/Omics-OS/lobster-charm-ui lobster chat --ui go --no-intro --workspace <tmp>`
- `cd lobster-tui && go test ./...` passed after initwizard UX parity updates.
- `PYTHONPATH="$PWD" .venv/bin/pytest -q tests/unit/cli/test_init_go_ui.py tests/unit/cli/test_init_tui_ux.py` passed after initwizard/adapter/install-path regressions were added.

## Remaining Work (Phase 5)

- Complete OutputAdapter migration for remaining direct Rich rendering branches.
- Complete visual parity pass for high-frequency command outputs.
- Expand protocol smoke to command-family parity flows.
- Expand CLI-level startup smoke coverage beyond the new unit/PTY checks if more startup failure classes are added.
- Complete native init UX follow-up after the recent parity hardening: port the remaining classic-only setup steps (license activation, custom cloud endpoint, docling, SSL connectivity/fix) and replace the post-wizard Rich install tail with a Charm-native confirmation/progress/summary flow.
- Expand transcript/golden checks from family baselines to broader subcommand coverage (`workspace info/load/remove`, `queue list/clear/export/import`, `metadata overview/samples/workspace/clear`, `pipeline export/run/info`).
- Implement context-compaction visibility MVP for Go mode: emit one informational line when supervisor pre-hook trims context (no new panel/command in first pass).

## Known Gaps / Tech Debt

1. `cancel` protocol event is still placeholder in Python event loop (`go_tui_launcher.py`).
2. `_execute_command()` still contains direct Rich rendering branches for classic-only surfaces; parity-critical Go command branches are protocol-safe.
3. Transcript golden coverage now includes baseline commands per family; deeper subcommand breadth is still pending.
4. Long streaming stress scenarios are not yet exercised by automated tests.
5. Cross-platform packaging/CI remains deferred to later phase.
6. Supervisor context compaction is currently operational but silent to operators; MVP UX surfacing is planned (Go path first).
7. Native init still omits several classic-only setup surfaces: premium/license activation, custom cloud endpoint entry, docling prompt, and SSL connectivity test/fix. Smart Standardization and Ollama selection are now native, but the remaining setup path is not yet parity-complete.
8. Init still renders package-install/progress as a Rich tail after the Charm/questionary wizard instead of keeping the operator inside one coherent native flow. The false post-install `lobster.agents` warning burst was fixed, but broader install-time output polish is still pending.

## Design Documents

| Document | Purpose |
|----------|---------|
| `PARITY_MATRIX.md` | Authoritative command parity tracking (30 commands) |
| `NEXT_PHASE_PARITY_PLAN.md` | Phase 5 workstream plan with execution checklist |
| `HITL_DESIGN.md` | Phase 6 human-in-the-loop architecture: `ask_user` tool, component mapper, interrupt/resume, BioCharm components |

## Phase 6 Summary (HITL)

**Design:** `.planning/charm-ui/HITL_DESIGN.md`

Three sub-phases, independently shippable:

| Sub-phase | Scope | Depends on | Validates with |
|-----------|-------|------------|----------------|
| **6A** | Python interrupt infra (`ask_user` tool, component mapper LLM, interrupt detection in client, classic prompt_toolkit fallback) | Phase 5 stable | Classic CLI end-to-end |
| **6B** | Protocol extension (`component_render/response/close` in Go types.go + Python bridge, resume loop in go_tui_launcher) | 6A | Protocol smoke tests |
| **6C** | BioCharm Go components (registry, confirm, select, cell_type_selector, threshold_slider, ...) | 6B | Per-component Go tests |

**Key files (6A, Python only):**
- `lobster/services/interaction/component_schemas.py` — NEW: component registry
- `lobster/services/interaction/component_mapper.py` — NEW: LLM mapper (structured output)
- `lobster/tools/user_interaction.py` — NEW: `ask_user` tool with `interrupt()`
- `lobster/core/client.py` — MODIFY: `__interrupt__` detection, `resume_from_interrupt()`
- `lobster/agents/graph.py` — MODIFY: wire `ask_user` to supervisor
- `lobster/cli_internal/classic_interaction.py` — NEW: prompt_toolkit fallback renderers

**Key decisions:**
- Supervisor calls `ask_user(question, context)` — sub-agents never talk to users
- Second LLM call maps question to component (inherits session provider)
- LangGraph `interrupt()` inside the tool, `Command(resume=...)` to continue
- Classic CLI always has a text-based fallback via `fallback_prompt`
- DeepAgents' `__interrupt__` detection pattern adopted (validated in production)

## Next Agent Focus (Compaction MVP)

1. Extend supervisor pre-hook return payload with compaction metadata only when trim occurs.
2. Surface that metadata through `LobsterClient._stream_query()` as `context_compaction` stream events.
3. Translate `context_compaction` to existing Go protocol `alert(level=info)` in `go_tui_launcher`.
4. Add dedupe guard so each compaction episode is announced once per turn.
5. Add unit/integration coverage:
   - pre-hook metadata emission under trim/no-trim
   - stream event emission
   - launcher mapping to alert payload
