# WS1 Command Parity Matrix (Authoritative)

Date: 2026-03-05
Scope: Derived from Python slash dispatch in `lobster/cli_internal/commands/heavy/slash_commands.py::_execute_command` and Go-native slash handling in `lobster-tui/internal/chat/model.go`.

Classification legend:
- `parity`: command path exists in Go mode and is expected to behave via Python parity path.
- `degraded-explicit`: reduced behavior is explicit to operator (in-product guidance/warning).
- `blocked`: parity is currently blocked by native divergence or missing explicit fallback.

| Command | Go handling path (native/bridged) | Classification | Evidence (file:line refs) | Next Action |
|---|---|---|---|---|
| `/help` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:736-747`, `lobster/cli_internal/commands/heavy/slash_commands.py:1635-1677` | Validate section/table formatting against classic transcript; needs-validation. |
| `/data` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:736-747`, `lobster/cli_internal/commands/heavy/slash_commands.py:1679-1680` | Validate `data_summary` parity for empty/loaded states in Go transcript; needs-validation. |
| `/session` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1682-1704` | Run protocol transcript parity check for session table fields; needs-validation. |
| `/status` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1705-1731` | Validate tier/provider/model field parity in Go transcript; needs-validation. |
| `/tokens` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1732-1810` | Verify large token table rendering in Go transcript; needs-validation. |
| `/reset` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2578-2583` | Add/confirm regression test for reset messaging and retained modalities; needs-validation. |
| `/workspace *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2219-2253` | Validate `list/info/load/remove/status/save` flows in Go protocol smoke tests; needs-validation. |
| `/files` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2131-2134`, `lobster/cli_internal/commands/heavy/slash_commands.py:2619-2661` | Confirm category table output and empty-state parity in Go transcript; needs-validation. |
| `/tree` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2175-2181` | Keep explicit fallback message; optional enhancement is Go-native tree renderer. |
| `/save [--force]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2421-2429`, `lobster/cli_internal/commands/heavy/slash_commands.py:2667-2685` | Validate save summaries and force flag behavior in protocol tests; needs-validation. |
| `/restore [pattern]` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2488-2494`, `lobster/cli_internal/commands/heavy/slash_commands.py:2688-2710` | Validate restore pattern behavior and result counts in Go transcripts; needs-validation. |
| `/read <file/glob>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2319-2335` | Verify quoted paths and glob behavior round-trip through protocol; needs-validation. |
| `/open <path>` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2380-2409` | Validate OS-specific open success/error messaging in Go mode; needs-validation. |
| `/queue *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2255-2314` | Verify `load/list/clear/export/import` branches and usage guidance in Go transcripts; needs-validation. |
| `/metadata *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2085-2129` | Validate all subcommands (`publications/samples/workspace/exports/list/clear`) in smoke tests; needs-validation. |
| `/config *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2501-2531` | Validate direct switch forms (`provider <name>`, `model <name>`) and `--save`; needs-validation. |
| `/pipeline *` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2343-2364` | Verify `list/export/run/info` coverage in transcript harness; needs-validation. |
| `/export` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2366-2378` | Validate `--no-png` and `--force` flags through Go command path; needs-validation. |
| `/plots` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2411-2412` | Confirm structured plot list rendering in Go transcript; needs-validation. |
| `/plot` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2496-2499` | Validate plot identifier resolution and failure guidance; needs-validation. |
| `/describe` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2417-2419` | Validate modality lookup and unknown-modality error messaging; needs-validation. |
| `/modalities` | `bridged` | `parity` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2414-2415` | Validate modality table parity in Go transcript; needs-validation. |
| `/clear` | `native` | `parity` | `lobster-tui/internal/chat/model.go:670-677`, `lobster/cli_internal/commands/heavy/slash_commands.py:2575-2576` | Confirm clear semantics (screen/history expectations) against operator UX; needs-validation. |
| `/exit` | `native` | `blocked` | `lobster-tui/internal/chat/model.go:678-681`, `lobster/cli_internal/commands/heavy/slash_commands.py:2585-2604` | Align behavior (add confirm in Go or remove confirm in Python with explicit policy); needs-validation. |
| `/dashboard` | `native` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:688-696`, `lobster/cli_internal/commands/heavy/slash_commands.py:1903-1914` | Keep explicit fallback; optional bridge to launch classic/Textual flow from Go wrapper. |
| `/status-panel` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1926-1933` | Keep explicit fallback to `/status`; optional Go-native panel implementation later. |
| `/workspace-info` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1961-1967` | Keep explicit fallback to `/workspace`/`/files`; optional Go-native view later. |
| `/analysis-dash` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:1996-2002` | Keep explicit fallback to `/plots` + `/metadata`; optional Go-native dashboard later. |
| `/progress` | `bridged` | `degraded-explicit` | `lobster-tui/internal/chat/model.go:697-703`, `lobster/cli_internal/commands/heavy/slash_commands.py:2029-2035` | Keep explicit fallback; optional protocol-native progress dashboard as follow-up. |
