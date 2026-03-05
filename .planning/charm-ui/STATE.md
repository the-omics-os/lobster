# Charm TUI Implementation — State

**Branch:** `feature/charm-ui` (worktree at `/Users/tyo/Omics-OS/lobster-charm-ui/`)
**Started:** 2026-03-04
**Last updated:** 2026-03-05

## Environment

- Workspace root: `/Users/tyo/Omics-OS/lobster-charm-ui`
- Python environment: `.venv` at `/Users/tyo/Omics-OS/lobster-charm-ui/.venv`
- Recommended activation: `source .venv/bin/activate`
- Install mode for local parity testing: `uv pip install --python .venv/bin/python -e .`

## Design Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Implementation spec | `kevin_notes/charm_ui_implementation.md` | Original design, architecture, phase plan (Kevin + ultrathink) |
| Architecture reference | `.claude/docs/charm-tui-architecture.md` | Go TUI architecture, BioCharm components, theme system |
| Protocol reference | `.claude/docs/charm-tui-protocol.md` | 28 message types, component protocol, event wiring, bridge details |

## Phase Status

| Phase | Name | Status | Commits | Notes |
|-------|------|--------|---------|-------|
| 0 | Init Wizard | **COMPLETE** | `8477338` | Go `huh` forms, Python bridge, questionary fallback |
| 1 | Protocol Foundation & Streaming Chat | **COMPLETE** | `78f7df8`..`742c99c` | IPC bridge, streaming, agent transitions, tool feed |
| 2 | Rich Rendering & UX Polish | **COMPLETE** | (same session) | Glamour markdown, loading tips, session ID, 6 bug fixes |
| 3 | Protocol Handlers & Native Interactions | **COMPLETE** | uncommitted | 20 protocol handlers, forms, confirm/select, progress |
| 4 | Native Slash Commands & Event Wiring | **COMPLETE** | uncommitted | Go-native commands + Python bridge dispatch |
| 4.1 | Autocomplete | **COMPLETE** | uncommitted | context-aware suggestions, trailing-space handling |
| 4.2 | Parity Hardening (bridge + command fixes) | **COMPLETE** | uncommitted | `/open`, `/restore`, `/config` direct switch, protocol-safe output paths |
| 5 | Rich CLI Functional/Visual Parity Migration | **NEXT** | — | migrate remaining functionality, visual supports, and output polish from Rich CLI |
| 6 | BioCharm Components | NOT STARTED | — | 6 domain components (celltype, threshold, qc, sequence, ontology, dna) |
| 7 | Distribution & Cross-Platform | NOT STARTED | — | CI cross-compile, platform wheels, Homebrew |
| 8 | SSH & Cloud | FUTURE | — | Charm `wish` library, `ssh app.omics-os.com` |

## Current Baseline (2026-03-05)

### Go TUI
- Slash completion supports top-level, subcommand, and deep contexts including trailing-space cases.
- Suggestions expanded to cover parity-critical commands (e.g., `/status-panel`, `/workspace-info`, `/analysis-dash`, `/progress`, `/vector-search`, `/dashboard`).
- Input prompt duplication fixed; completion visibility improved.

### Python bridge / slash dispatch
- Go launcher now prefers local dev `lobster-tui` binary (and supports `LOBSTER_TUI_BINARY` override).
- Slash dispatch keeps command args end-to-end from Go → Python.
- `_execute_command()` now supports protocol-safe output in more branches.
- Concrete command fixes landed:
  - `/open` now executes and returns status
  - `/restore` now performs actual restore path
  - `/config provider <name>` and `/config model <name>` now work without mandatory `switch`
  - `/workspace save` wired
  - `/pipeline` defaults to list

### Validation completed
- `cd lobster-tui && go test ./...` passed.
- Added/ran targeted regression tests for slash-command fixes:
  - `tests/unit/cli/test_slash_commands_go_tui_regressions.py`
- Python syntax compile checks passed for updated launcher/dispatch files.

## Remaining Work

### Phase 5 (Next): Rich CLI Functional/Visual Parity Migration

Goal: migrate all remaining functionality, visual support surfaces, and output prettification from classic Rich CLI to Go TUI while preserving `--ui classic` fallback.

High-level scope:
- Full slash command parity (behavior + output shape)
- Visual parity for high-value command surfaces (session/status/files/workspace/queue/metadata)
- Prettification and consistency pass for Go-rendered tables, alerts, and summaries
- Interaction parity where practical (history, completion depth, prompt ergonomics)
- Golden transcript and integration coverage for regression prevention

Detailed execution plan: see `.planning/charm-ui/NEXT_PHASE_PARITY_PLAN.md`.

### Phase 6: BioCharm Components
- Keep as originally planned after parity baseline is stable.

### Phase 7: Distribution & Cross-Platform
- CI cross-compile + packaging after parity and component stability.

### Phase 8: SSH & Cloud (Post-Funding)
- `wish` + remote TUI sessions.

## Known Gaps & Tech Debt (Updated)

1. Protocol-bridged autocomplete still lacks dynamic file path completion (`completion_request` / `completion_response` not wired yet).
2. Input history recall (Up/Down history ring) not implemented in Go model.
3. `TypeCancel` exists but cancel path in Python event loop still placeholder.
4. Some command branches still use direct `console.*` rendering and need full OutputAdapter parity conversion.
5. No end-to-end automated protocol integration suite for Go TUI yet.
6. No stress validation for long streaming outputs (10K+ token transcript scenarios).
7. CI cross-compilation pipeline remains deferred.
