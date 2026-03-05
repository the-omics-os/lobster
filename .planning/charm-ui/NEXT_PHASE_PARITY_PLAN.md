# Phase 5 Plan — Rich CLI Functional + Visual Parity Migration

**Status:** In Progress
**Owner:** Charm UI migration stream
**Date:** 2026-03-05
**Last verified against code/tests:** 2026-03-05 (`ac7a60e`, `642ad8d`)

## Objective

Migrate remaining functional and visual parity surfaces from classic Rich CLI to Go TUI so `lobster chat --ui go` is a safe default for daily use.

## Current Completion Snapshot

- WS1: Command parity matrix is in place and current (`.planning/charm-ui/PARITY_MATRIX.md`).
- WS4: Local command history ring (`Up`/`Down`) is implemented in Go model.
- WS4: Protocol completion request/response scaffold is implemented for `/read`, `/open`, `/workspace load`.
- WS5: Regression and smoke test scaffolding landed and passes in targeted slices.
- Remaining: finish OutputAdapter migration, visual polish pass, and transcript/golden parity audit.

## Remaining Workstreams

### WS2 — OutputAdapter Completion
- Convert residual direct Rich rendering branches in slash command dispatch to OutputAdapter-driven protocol output.
- Keep classic-only interactive branches explicit and guarded.

### WS3 — Visual Support + Prettification
- Normalize table/alert formatting across priority commands.
- Complete transcript-based visual parity check for core command families.

### WS5 — Validation + Exit Audit
- Expand smoke coverage from protocol dispatch to multi-command parity flows.
- Add golden transcript checks for command output shape.
- Run final parity audit and publish unresolved exceptions (if any).

## Execution Checklist

- [x] Create parity matrix artifact (`.planning/charm-ui/PARITY_MATRIX.md`)
- [ ] Complete OutputAdapter conversion for remaining direct-console branches
- [ ] Apply visual formatting standards to top 20 slash commands
- [x] Implement command history ring in Go model
- [x] Add protocol completion request/response scaffolding
- [x] Add integration smoke test script for Go chat commands (`tests/integration/test_go_tui_protocol_smoke.py`)
- [x] Add targeted Python/Go regression coverage for recent parity fixes
- [ ] Run transcript/golden exit audit and close remaining parity exceptions
