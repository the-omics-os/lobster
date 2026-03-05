# Phase 5 Plan — Rich CLI Functional + Visual Parity Migration

**Status:** Draft (execution-ready)
**Owner:** Charm UI migration stream
**Date:** 2026-03-05

## State Linkage

- Source of truth for current implementation status and resolved gaps:
  - `.planning/charm-ui/STATE.md`
- This plan is intentionally execution-focused and does not duplicate detailed current-state narrative.

## Objective

Migrate remaining functionality, visual supports, and presentation polish from classic Rich CLI to Go TUI so `lobster chat --ui go` can be the primary day-to-day interface without parity regressions.

## Success Criteria (Phase Exit)

1. Command parity: all high-frequency slash commands match classic behavior or have explicit Go-safe fallback messaging.
2. Visual parity: key command outputs use structured tables/alerts in Go TUI (no degraded plain dumps for parity surfaces).
3. Interaction parity: history + improved completion coverage + consistent prompt/status behavior.
4. Stability: protocol integration tests + golden transcript checks for core command paths.
5. Rollback safety: `--ui classic` remains fully functional and untouched.

## Scope

### In scope
- `lobster chat` command surface in Go mode.
- Slash command behavior + render parity.
- OutputAdapter migration for residual direct `console.*` branches.
- Visual prettification for status/session/data/workspace/queue/metadata/config/tokens/help.

### Out of scope
- BioCharm interactive domain components (moved to next phase).
- Cross-platform packaging/distribution work.
- SSH/cloud remote session layer.

## Workstreams

## WS1 — Command Parity Matrix and Gap Closure

Deliverables:
- Build authoritative command matrix from `_execute_command()` and classic output behavior.
- Classify each command as: `parity`, `degraded-but-explicit`, or `blocked`.
- Close all `blocked` commands in Go mode for core command families.

Priority command families:
- Core: `/help`, `/session`, `/status`, `/tokens`, `/data`, `/reset`
- Workspace: `/workspace *`, `/files`, `/tree`, `/save`, `/restore`, `/read`, `/open`
- Queue/Metadata: `/queue *`, `/metadata *`
- Config/Pipeline: `/config *`, `/pipeline *`, `/export`, `/plots`, `/plot`, `/describe`, `/modalities`

Acceptance:
- No silent no-op command branches.
- Unknown subcommand/help guidance always routed through OutputAdapter.

## WS2 — OutputAdapter Completion (Protocol-Safe Rendering)

Deliverables:
- Convert residual `console.print` command paths in slash dispatch to OutputAdapter.
- Ensure alerts/tables/code blocks map cleanly through `ProtocolOutputAdapter`.
- Remove branch-specific rendering drift between classic and go where avoidable.

Acceptance:
- Core command paths emit protocol messages only (except deliberate classic-only branches).
- No direct Rich-only rendering in Go-path critical branches.

## WS3 — Visual Support + Prettification Pass

Deliverables:
- Standardize command output format patterns:
  - Summary line
  - Structured table(s)
  - Next-step hints
- Normalize severity levels (`info`, `warning`, `error`, `success`) for consistent alert style.
- Improve readability for long tables via truncation rules and section labels.

Prettification targets:
- `/help` split sections (core/power/admin)
- `/status` + `/session` with concise key-value tables
- `/files`/`/workspace` consistent columns and ordering
- `/queue` + `/metadata` action feedback consistency

Acceptance:
- Side-by-side transcript comparison shows no major UX regressions vs Rich CLI.

## WS4 — Interaction Parity

Deliverables:
- Implement local input history ring in Go (`Up/Down` recall).
- Expand completion depth where command matrix indicates missing suggestions.
- Add protocol-backed path completion scaffold (`completion_request`/`completion_response`) for `/read`, `/open`, `/workspace load`.

Acceptance:
- Operators can navigate recent command history in Go chat.
- Completion experience covers command + key arg paths.

## WS5 — Validation and Guardrails

Deliverables:
- Python unit tests for fixed command regressions.
- Go completion tests for new suggestion coverage.
- Integration smoke harness for Go protocol command round-trip.
- Golden transcript snapshots for core slash-command flows.

Acceptance:
- CI gate includes Go tests + Python parity regressions.
- No regression in previously fixed bugs (`/open`, `/restore`, config direct switch, trailing-space completion).

## Milestone Breakdown

### M1 — Parity Inventory + High-Severity Fixes (1 session)
- Freeze command matrix and gap labels.
- Fix any remaining no-op or unsafe branches.

### M2 — Adapter Completion + Visual Baseline (1-2 sessions)
- Finish OutputAdapter migration for parity-critical branches.
- Establish unified formatting primitives.

### M3 — Prettification + Interaction Parity (1-2 sessions)
- Command output polish pass.
- History + completion enhancements.

### M4 — Regression Harness + Exit Audit (1 session)
- Golden transcript + smoke automation.
- Publish parity audit checklist and unresolved exceptions.

## Risk Log

1. Drift between classic and Go behavior while rapidly migrating branches.
- Mitigation: command matrix + golden transcripts.

2. Rich-specific components not representable one-to-one in Go.
- Mitigation: explicit degraded mode messaging with suggested alternative commands.

3. Mixed installs causing stale code execution.
- Mitigation: enforce editable install check in dev workflow (`uv pip install -e .`) and verify import path.

## Execution Checklist

- [ ] Create parity matrix artifact (`.planning/charm-ui/PARITY_MATRIX.md`)
- [ ] Complete OutputAdapter conversion for remaining direct-console branches
- [ ] Apply visual formatting standards to top 20 slash commands
- [ ] Implement command history ring in Go model
- [ ] Add protocol completion request/response scaffolding
- [ ] Add integration smoke test script for Go chat commands
- [ ] Run exit audit and update `.planning/charm-ui/STATE.md`
