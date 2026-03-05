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
| `/workspace *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2219` | Expand smoke coverage to `list/info/load/remove/status/save` command flows. |
| `/files` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2131` | Validate category grouping and empty-state behavior in Go transcript. |
| `/tree` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2175` | Keep explicit fallback to `/files`; optional Go-native tree renderer later. |
| `/save [--force]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2421`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:136` | Protocol safety validated; transcript parity for summary text still pending. |
| `/restore [pattern]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2488`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:75` | Handler wiring validated; transcript parity for counts/messages pending. |
| `/read <file/glob>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2319`, `tests/unit/cli/test_go_tui_launcher_completions.py:23` | Completion path validated; transcript parity for file-read output pending. |
| `/open <path>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2380`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:45` | Command execution regression validated; cross-platform messaging parity pending. |
| `/queue *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2255` | Validate `load/list/clear/export/import` subcommand transcript coverage. |
| `/metadata *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2085` | Validate all metadata subcommands in protocol transcript harness. |
| `/config *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2501`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:98` | Direct switch forms validated; transcript parity for usage/error text pending. |
| `/pipeline *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2343` | Validate `list/export/run/info` transcript parity coverage. |
| `/export` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2366` | Validate `--no-png` and `--force` branch behavior in Go mode transcripts. |
| `/plots` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2411` | Validate plot list table parity. |
| `/plot` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2496` | Validate identifier resolution and error guidance parity. |
| `/describe` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2417` | Validate modality lookup success/error parity. |
| `/modalities` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2414` | Validate modality table parity in Go transcript. |
| `/vector-search` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2533` | Validate JSON/code-block rendering and `--top-k` behavior in Go mode. |
| `/clear` | `native` | `parity` | `lobster-tui/internal/chat/model.go:737`, `lobster/cli_internal/commands/heavy/slash_commands.py:2575`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:171` | Protocol-safe behavior validated; UX parity decision (screen vs history) pending. |
| `/exit` | `native` | `parity` | `lobster-tui/internal/chat/model.go:744`, `lobster-tui/internal/chat/model_test.go:12`, `tests/unit/cli/test_slash_commands_go_tui_regressions.py:192` | Confirm flow regression covered; final transcript UX audit pending. |
| `/dashboard` | `native` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:752`, `lobster/cli_internal/commands/heavy/slash_commands.py:1903` | Keep explicit fallback to classic UI; optional bridge enhancement later. |
| `/status-panel` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1926` | Keep explicit fallback to `/status`; optional native panel later. |
| `/workspace-info` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1961` | Keep explicit fallback to `/workspace`/`/files`; optional native view later. |
| `/analysis-dash` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:1996` | Keep explicit fallback to `/plots` + `/metadata`; optional dashboard later. |
| `/progress` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:763`, `lobster/cli_internal/commands/heavy/slash_commands.py:2029` | Keep explicit fallback; optional protocol-native progress dashboard later. |
