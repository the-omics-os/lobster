# WS1 Command Parity Matrix (Authoritative)

Date: 2026-03-05
Scope: Derived from Python slash dispatch in `lobster/cli_internal/commands/heavy/slash_commands.py::_execute_command` and Go-native slash handling in `lobster-tui/internal/chat/model.go`.

Snapshot:
- Tracked commands: `30`
- `parity`: `24`
- `degraded-explicit`: `6`
- `blocked`: `0`

Classification legend:
- `parity`: command path exists in Go mode and is expected to behave via Python parity path.
- `degraded-explicit`: reduced behavior is explicit to operator (in-product guidance/warning).
- `blocked`: parity is blocked by missing implementation or missing explicit fallback.

| Command | Go handling path | Classification | Evidence (file:line refs) | Next action |
|---|---|---|---|---|
| `/help` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1635`, `tests/integration/test_slash_command_golden_transcripts.py:63` | Core golden transcript coverage is in place; extend to admin/help variants if added. |
| `/data` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1679` | Transcript parity audit pending for empty/loaded modality states. |
| `/session` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1682`, `tests/integration/test_slash_command_golden_transcripts.py:63` | Core golden transcript coverage is in place; expand with real workspace path variance checks. |
| `/status` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1705`, `tests/integration/test_slash_command_golden_transcripts.py:63` | Core golden transcript coverage is in place; add tier-source variants (license/server). |
| `/tokens` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1732`, `tests/integration/test_slash_command_golden_transcripts.py:63` | Core golden transcript coverage is in place; add ollama/free-mode variant golden. |
| `/reset` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2581` | Add transcript check for reset messaging and retained modalities behavior. |
| `/workspace *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2219`, `tests/golden/slash_commands/workspace_list.json` | Baseline transcript golden exists for `/workspace list`; `/workspace load` path completion now preserves spaced/quoted tails in Go model; expand to `info/load/remove/status/save` transcript flows. |
| `/files` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2131` | Validate category grouping and empty-state behavior in Go transcript. |
| `/tree` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2175` | Keep explicit fallback to `/files`; optional Go-native tree renderer later. |
| `/save [--force]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2421`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:136` | Protocol safety validated; transcript parity for summary text still pending. |
| `/restore [pattern]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2488`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:75` | Handler wiring validated; transcript parity for counts/messages pending. |
| `/read <file/glob>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2319`, `tests/unit/cli/test_go_tui_launcher_completions.py:23`, `lobster-tui/internal/chat/completions_test.go:103` | Completion path validated, including spaced/quoted path-tail parsing in Go completion context; transcript parity for file-read output pending. |
| `/open <path>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2380`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:45`, `lobster-tui/internal/chat/completions_test.go:103` | Command execution regression validated; spaced/quoted path-tail completion parsing validated in Go model; cross-platform messaging parity pending. |
| `/queue *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2255`, `tests/golden/slash_commands/queue.json` | Baseline transcript golden exists for `/queue`; expand to `load/list/clear/export/import` subcommands. |
| `/metadata *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2085`, `tests/golden/slash_commands/metadata_publications.json` | Baseline transcript golden exists for `/metadata publications`; expand to overview/samples/workspace/clear variants. |
| `/config *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2501`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:98`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:117`, `tests/golden/slash_commands/config_show.json`, `tests/golden/slash_commands/config_provider.json`, `tests/golden/slash_commands/config_model.json` | Direct switch forms are validated for provider/model, and transcript goldens now cover `/config show`, `/config provider`, and `/config model`; expand only to additional failure/empty-state variants if visual regressions appear. |
| `/pipeline *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2343`, `tests/golden/slash_commands/pipeline_list.json` | Baseline transcript golden exists for `/pipeline` list output; expand to `export/run/info` coverage. |
| `/export` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2366` | Validate `--no-png` and `--force` branch behavior in Go mode transcripts. |
| `/plots` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2411` | Validate plot list table parity. |
| `/plot` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2496` | Validate identifier resolution and error guidance parity. |
| `/describe` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2417` | Validate modality lookup success/error parity. |
| `/modalities` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2414` | Validate modality table parity in Go transcript. |
| `/vector-search` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2533` | Validate JSON/code-block rendering and `--top-k` behavior in Go mode. |
| `/clear` | `native` | `parity` | `lobster-tui/internal/chat/model.go:727`, `lobster/cli_internal/commands/heavy/slash_commands.py:2575`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:171`, `lobster-tui/internal/chat/model_test.go:241` | Protocol-safe behavior validated; local and protocol clear paths are now unified and regression-tested; UX decision (screen vs history) remains a product choice if behavior changes are desired. |
| `/exit` | `native` | `parity` | `lobster-tui/internal/chat/model.go:744`, `lobster-tui/internal/chat/model_test.go:12`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:192`, `lobster/cli_internal/go_tui_launcher.py:320`, `tests/unit/cli/test_go_tui_exit_footer.py:1` | Native confirm flow is regression-covered and Go launcher now emits a post-exit continuation/footer block (session resume + links + quick next steps); optional copy polish remains. |
| `/dashboard` | `native` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:752`, `lobster/cli_internal/commands/heavy/slash_commands.py:1903` | Keep explicit fallback to classic UI; optional bridge enhancement later. |
| `/status-panel` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1926` | Keep explicit fallback to `/status`; optional native panel later. |
| `/workspace-info` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1961` | Keep explicit fallback to `/workspace`/`/files`; optional native view later. |
| `/analysis-dash` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1996` | Keep explicit fallback to `/plots` + `/metadata`; optional dashboard later. |
| `/progress` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2029` | Keep explicit fallback; optional protocol-native progress dashboard later. |

## Supplemental Runtime UX Surfaces (Non-Slash)

These are not slash commands, but they affect perceived parity/completeness in chat operation.

| Surface | Go handling path | Classification | Evidence (file:line refs) | Next action |
|---|---|---|---|---|
| Supervisor context compaction visibility | `not surfaced` | `degraded-explicit` | `lobster/agents/context_management.py:235`, `lobster/agents/graph.py:819`, `lobster/core/client.py:322`, `lobster/cli_internal/go_tui_launcher.py:458` | Implement MVP: emit `context_compaction` stream event and map to Go `alert(level=info)` so users can see when context was compacted. |
| Streaming viewport follow ergonomics | `native` | `parity` | `lobster-tui/internal/chat/model.go:990`, `lobster-tui/internal/chat/model_test.go:263` | Baseline behavior now preserves user scroll during streaming and auto-follows when at bottom; add long-stream stress validation in later smoke expansion. |
| Transcript scrollback retrieval controls | `native` | `parity` | `lobster-tui/internal/chat/model.go:942`, `lobster-tui/internal/chat/model_test.go:323`, `lobster-tui/internal/chat/run.go:74`, `lobster-tui/internal/chat/run_test.go:1` | `Up`/`Down` are history-only and transcript navigation uses `PgUp`/`PgDn`; scrollbar indicator communicates current scroll position. Mouse interaction is explicit: inline defaults to `select`, fullscreen defaults to `scroll`, `Ctrl+G` toggles live between them, and env override remains available via `LOBSTER_TUI_MOUSE_CAPTURE`. |
| Protocol table markup sanitization/rendering | `bridged` | `parity` | `lobster/cli_internal/commands/output_adapter.py:356`, `lobster-tui/internal/chat/model.go:851`, `tests/unit/cli_internal/commands/test_output_adapter.py:4`, `tests/golden/slash_commands/config_show.json` | Sanitization now strips Rich tags from protocol tables, and Go renders them as width-aware preformatted blocks; extend similar verification to any remaining non-table rich-text payloads if surfaced. |
| Fatal startup diagnostics (`provider`/config/dependency failures) | `bridged` | `parity` | `lobster/cli_internal/startup_diagnostics.py:1`, `lobster/cli_internal/commands/heavy/session_infra.py:565`, `lobster/cli_internal/go_tui_launcher.py:348`, `tests/unit/cli/test_startup_diagnostics.py:1`, `tests/unit/cli/test_go_tui_launcher_completions.py:1` | Shared startup classification now feeds Rich and Go renderers; expand future CLI-level smoke coverage if additional startup failure classes are introduced. |
