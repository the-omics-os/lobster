# Phase 5 Plan — Rich CLI Functional + Visual Parity Migration

**Status:** In Progress
**Owner:** Charm UI migration stream
**Date:** 2026-03-05
**Last verified against code/tests:** 2026-03-05 (`ac7a60e`, `642ad8d`, `fd142fb`)

## Objective

Migrate remaining functional and visual parity surfaces from classic Rich CLI to Go TUI so `lobster chat --ui go` is a safe default for daily use.

## Current Completion Snapshot

- WS1: Command parity matrix is in place and current (`.planning/charm-ui/PARITY_MATRIX.md`).
- WS4: Local command history ring (`Up`/`Down`) is implemented in Go model.
- WS4: Protocol completion request/response scaffold is implemented for `/read`, `/open`, `/workspace load`.
- WS5: Regression and smoke test scaffolding landed and passes in targeted slices.
- WS3: Core command visual pass started (`/help`, `/session`, `/status`, `/config show`, `/config model`) with refreshed golden transcripts.
- WS5: Golden transcript family baselines added for `/workspace list`, `/queue`, `/metadata publications`, `/config`/`/config provider`/`/config model`, and `/pipeline`.
- WS3/WS4: Go chat UX polish landed for high-frequency interaction paths: slash history recall under suggestions, spaced/quoted path completion parsing, clear-path parity between local/protocol handling, select `Ctrl+C` current-option behavior, and non-forced streaming autoscroll while user is scrolled up.
- WS3/WS4: Scrollback retrieval hardening landed: `Up`/`Down` are now reserved for input history, dedicated `PgUp`/`PgDn` viewport paging works while drafting input, and a right-edge scrollbar indicator is shown when the transcript is scrollable. Mouse interaction now has an explicit runtime toggle: inline mode defaults to `select`, fullscreen defaults to `scroll`, `Ctrl+G` flips modes live, and the current mode is surfaced in the UI.
- WS3/WS4: Go exit-path parity improved: after leaving Go chat, launcher now emits a classic-style continuation/footer block (session resume hint, feedback/issues/contact links, runtime/token context, and quick next-step commands).
- WS2/WS3: Fatal chat startup diagnostics are now normalized through a shared UI-agnostic layer, so Go mode renders provider/config/dependency failures as protocol alerts instead of leaking Rich output or tracebacks; classic and query startup reuse the same classification/copy.
- WS2/WS3: Protocol table sanitization landed so Rich tags are stripped from titles/headers/cells before Go-protocol emission (prevents literal `[cyan]...[/cyan]` leakage in transcript tables).
- WS3: Go protocol tables now render as width-aware preformatted blocks, improving alignment for long path/config surfaces without changing the protocol contract.
- WS3: Go rendering polish landed for alert readability and narrow-terminal layout resilience (alert hierarchy/chips, width clamps, spacing rhythm, theme contrast tuning).
- Adjacent CLI polish: `lobster query` remains classic/JSON-first by design, but human mode now uses a compact report-style renderer and component discovery no longer eagerly imports unrelated optional plugin groups during simple query startup.
- Adjacent CLI debug-path polish: `lobster query --reasoning/--verbose` now stays visibly alive in non-stream mode, avoids duplicating reasoning inside the final answer block, and appends a compact execution summary for verbose multi-tool runs.
- WS5: Go regression tests expanded for status/spinner lifecycle, select behavior, clear parity, streaming viewport behavior, and spaced completion parsing.
- WS5: Added focused adapter regression coverage for protocol table sanitization and refreshed affected slash transcript goldens.
- Remaining: finish OutputAdapter migration, visual polish pass, and deeper subcommand transcript/golden parity expansion.

## Remaining Workstreams

### WS2 — OutputAdapter Completion
- Convert residual direct Rich rendering branches in slash command dispatch to OutputAdapter-driven protocol output.
- Keep classic-only interactive branches explicit and guarded.

### WS3 — Visual Support + Prettification
- Normalize table/alert formatting across priority commands.
- Complete transcript-based visual parity check for core command families.
- Add first-pass visibility for supervisor context compaction events (new feature, Go-first).

### WS3.1 — Context Compaction Visibility (MVP / easiest path)
- Goal: make silent `pre_model_hook` compaction visible in Go chat with minimal surface area and minimal protocol churn.
- Scope: Go protocol chat path only (`lobster chat --ui go`), supervisor compaction only.
- UX choice (MVP): emit one `info` alert line when compaction occurs (no new panel, no new slash command, no prompt badge yet).
- Message format target: `Context compacted: <before>→<after> messages (budget <tokens>).`
- Follow-up hint target: `Full delegated outputs remain retrievable via store keys/retrieve_agent_result.`

Execution steps:
1. `pre_model_hook` emits compaction metadata in state update only when trimming occurs (include `before_count`, `after_count`, `budget_tokens`).
2. `LobsterClient._stream_query()` detects compaction metadata from update events and yields a `context_compaction` stream event.
3. `go_tui_launcher` maps `context_compaction` to existing protocol `alert` with `level="info"` (reuse existing UI path).
4. Add event dedupe guard per query turn (prevent repeated notice spam).
5. Keep Rich CLI unchanged in MVP (documented as out of scope for this increment).

Acceptance criteria:
1. Compaction notice appears exactly once per compaction episode in Go chat.
2. No notice appears when no trimming occurs.
3. Existing stream order is preserved (assistant text, done/status, tool feed unaffected).
4. Existing slash/golden tests remain green.

### WS5 — Validation + Exit Audit
- Expand smoke coverage from protocol dispatch to multi-command parity flows.
- Expand golden transcript checks from family baselines to broader subcommand coverage.
- Run final parity audit and publish unresolved exceptions (if any).

## Execution Checklist

- [x] Create parity matrix artifact (`.planning/charm-ui/PARITY_MATRIX.md`)
- [x] Complete OutputAdapter conversion for parity-critical Go command branches
- [ ] Apply visual formatting standards to top 20 slash commands
- [x] Implement command history ring in Go model
- [x] Add protocol completion request/response scaffolding
- [x] Add integration smoke test script for Go chat commands (`tests/integration/test_go_tui_protocol_smoke.py`)
- [x] Add targeted Python/Go regression coverage for recent parity fixes
- [x] Normalize fatal startup diagnostics across classic/Go/query paths and add regression coverage for provider/config/dependency failures
- [x] Land Go chat interaction UX polish for completion/history/clear/select/stream-scroll behavior
- [x] Expand Go TUI unit regression coverage for spinner/status/select/clear/streaming viewport behavior
- [x] Add family-baseline golden transcript coverage for `workspace`/`queue`/`metadata`/`config`/`pipeline`
- [x] Fix protocol table markup leakage for Go mode and add adapter regression tests
- [x] Harden transcript scrollback retrieval (`PgUp`/`PgDn` while typing + scrollbar indicator + live select/scroll mouse mode toggle)
- [ ] Add supervisor compaction visibility MVP in Go path via `info` alert emission
- [ ] Add tests for compaction metadata emission + stream event mapping + launcher alert translation
- [ ] Run transcript/golden exit audit and close remaining parity exceptions
