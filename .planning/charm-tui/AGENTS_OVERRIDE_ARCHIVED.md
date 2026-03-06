# AGENT.md

System brief for Codex agents working in `lobster-charm-ui`.

## Purpose

This repository’s immediate purpose is to make the Go/Charm chat CLI (`lobster chat --ui go`) functionally and visually complete relative to the current Python/Rich chat experience, then safely exceed parity with targeted UX improvements.

This is not a generic CLI testing sandbox. It is an implementation stream for:
1. Go TUI protocol behavior correctness
2. Slash-command parity and transcript stability
3. Operator-visible UX quality (layout, flow, telemetry clarity)

## Primary Objective

Ship `lobster chat --ui go` as a safe daily default without regressing `--ui classic`.

## Source Of Truth

Always start with:
1. `.planning/charm-ui/NEXT_PHASE_PARITY_PLAN.md`
2. `.planning/charm-ui/PARITY_MATRIX.md`
3. `.planning/charm-ui/STATE.md`

If these docs and code diverge, update docs in the same change.

## Current Focus (Phase 5)

1. OutputAdapter parity completion for remaining Rich-only paths
2. Visual/transcript polish across high-frequency slash surfaces
3. Regression and smoke expansion for Go protocol flows
4. Context compaction visibility MVP in Go mode (supervisor pre-hook trimming should not be silent)

## Non-Negotiables

1. Do not break `lobster chat --ui classic` behavior.
2. Keep Go path protocol-safe (no direct Rich-only rendering in parity-critical Go branches).
3. Preserve explicit degraded fallbacks where parity is intentionally not implemented.
4. Do not hide parity gaps; document them in planning artifacts.
5. Prefer incremental, test-backed changes over broad rewrites.

## Architecture Snapshot

- Python orchestrates runtime/session/commands.
- Go binary (`lobster-tui`) renders the chat UX.
- `lobster/cli_internal/go_tui_launcher.py` bridges Python events to Go protocol messages.
- Protocol contracts live in `lobster-tui/internal/protocol/types.go`.
- Go chat model/rendering lives in `lobster-tui/internal/chat/`.

## Working Rules For Agents

1. Implement smallest viable increment first.
2. Keep changes localized; avoid opportunistic refactors unless required.
3. Add/refresh tests with behavior changes.
4. Refresh goldens only when output changes are intentional.
5. Leave a clear handoff in `.planning/charm-ui/STATE.md` when stopping mid-stream.

## Validation Baseline

Run these before handoff when relevant:

```bash
cd lobster-tui && go test ./internal/chat ./internal/protocol
uv run pytest -q \
  tests/unit/cli/test_go_tui_launcher_completions.py \
  tests/unit/cli/test_slash_commands_go_tui_regressions.py
```

When transcript-affecting behavior changes:

```bash
uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py
# only when intentional
LOBSTER_UPDATE_GOLDENS=1 uv run pytest -q tests/integration/test_slash_command_golden_transcripts.py
```

## Go TUI Automation Note

When validating `lobster chat --ui go` from an agent/PTY session, use `--no-intro` by default:

```bash
uv run lobster chat --ui go --no-intro
```

This suppresses the inline welcome animation plus pre-ready spinner/tip churn, which keeps PTY-driven runs stable enough for future agents to inspect and interact with. Only omit `--no-intro` when the task is specifically about the intro animation or startup visual behavior.

If Go-side TUI code changed, rebuild the local dev binary before validating so the CLI does not launch a stale `lobster-tui` executable:

```bash
cd lobster-tui && go build -o lobster-tui ./cmd/lobster-tui
```

## Done Definition (For Any Parity Increment)

1. Behavior works in Go mode.
2. Classic mode is unchanged or explicitly documented.
3. Tests pass for touched surfaces.
4. Planning docs reflect new state and next steps.
