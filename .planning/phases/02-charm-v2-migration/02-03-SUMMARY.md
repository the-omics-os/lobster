---
phase: 02-charm-v2-migration
plan: 03
subsystem: tui
tags: [charm-v2, bubbletea, huh-removal, initwizard, forms, go]

requires:
  - phase: 02-charm-v2-migration
    plan: 02
    provides: all imports migrated to v2, huh is the last remaining v1 dependency
provides:
  - huh dependency completely removed from go.mod
  - init wizard rewritten as BubbleTea v2 state machine
  - forms use inline model instead of huh tea.Exec suspension
  - zero v1 bubbletea/lipgloss/bubbles/huh imports in any Go file
affects: [03-*-PLAN]

tech-stack:
  removed: [github.com/charmbracelet/huh v0.8.0, github.com/charmbracelet/bubbletea v1 (indirect), github.com/charmbracelet/bubbles v1 (indirect)]
  patterns: [step-based wizard state machine, inline form model, sub-model composition]

key-files:
  modified:
    - lobster-tui/internal/initwizard/wizard.go
    - lobster-tui/internal/initwizard/wizard_test.go
    - lobster-tui/internal/chat/forms.go
    - lobster-tui/internal/chat/model.go
    - lobster-tui/go.mod
    - lobster-tui/go.sum

key-decisions:
  - "Wizard uses step-based state machine (wizardStep enum) with sub-models for each input type"
  - "Forms use inline rendering in chat area instead of tea.Exec suspension (no TUI interruption)"
  - "Sub-models (multiSelectModel, singleSelectModel, confirmModel) are reusable within wizard"
  - "v1 lipgloss remains as indirect dependency via glamour (expected, glamour has no v2)"

requirements-completed: [MIGR-03, MIGR-05, MIGR-06]

duration: 6min
completed: 2026-03-07
---

# Phase 2 Plan 3: Init Wizard Rewrite and huh Removal Summary

**Rewrote 806-line init wizard and 125-line forms.go using BubbleTea v2 primitives, completely removing the huh dependency that pinned the binary to v1.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-07T04:36:38Z
- **Completed:** 2026-03-07T04:43:19Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Init wizard rewritten as BubbleTea v2 Model with step-based state machine (stepAgentPackages -> stepProvider -> stepAPIKey -> stepProfile -> stepOptionalKeys -> stepDone)
- Custom sub-models: multiSelectModel (space toggle, enter confirm), singleSelectModel (cursor navigate, enter select), confirmModel (y/n keys)
- TextInput from bubbles v2 for all text/password fields with EchoPassword mode
- Back-navigation via Esc at every step, cancellation produces {"cancelled":true} JSON
- forms.go rewritten with inlineFormModel supporting text, password, select, and confirm field types
- Inline form renders in chat area (no TUI suspension via tea.Exec)
- huh completely removed from go.mod via `go mod tidy`
- v1 bubbletea and v1 bubbles also removed from go.mod (were indirect deps of huh)
- Zero v1 bubbletea/lipgloss/bubbles/huh imports in any .go file (v1 lipgloss remains as glamour indirect only)
- Full test suite green: 19 initwizard tests + all chat/biocomp tests pass

## Task Commits

1. **Task 1: Rewrite init wizard with BubbleTea v2 state machine** - `2a0f75e` (feat)
2. **Task 2: Rewrite forms.go without huh and remove huh from go.mod** - `81bd07f` (feat)

## Files Modified

- `lobster-tui/internal/initwizard/wizard.go` -- Complete rewrite: 817 lines -> 750 lines, huh imports replaced with v2 primitives
- `lobster-tui/internal/initwizard/wizard_test.go` -- Expanded from 3 to 19 tests covering sub-models, step transitions, provider-specific logic
- `lobster-tui/internal/chat/forms.go` -- Rewritten from huh tea.Exec to inline form model (126 -> 305 lines)
- `lobster-tui/internal/chat/model.go` -- Added activeForm field, inline form rendering, key delegation
- `lobster-tui/go.mod` -- huh, v1 bubbletea, v1 bubbles removed
- `lobster-tui/go.sum` -- Updated checksums

## Decisions Made

- Wizard uses step-based state machine with enum rather than sequential huh.Form.Run() calls
- Sub-models are value types with Update() returning (model, done, cancelled) tuples
- Forms render inline in the chat area rather than suspending the TUI (better UX, no screen flicker)
- v1 lipgloss kept as indirect (glamour dependency) -- expected and documented

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None -- all compilation and tests passed on first attempt.

## Next Phase Readiness

- Charm v2 migration is now complete (all 3 plans finished)
- Zero v1 dependencies remain except glamour v0.10 (which has no v2 release)
- Ready to proceed with Phase 3

## Self-Check: PASSED

- All 6 modified files exist on disk
- Both task commits (2a0f75e, 81bd07f) found in git log
- huh not in go.mod
- Zero v1 bubbletea/lipgloss/bubbles/huh imports (glamour excluded)
- Line counts: wizard.go=1279, wizard_test.go=424, forms.go=369 (all exceed minimums)

---
*Phase: 02-charm-v2-migration*
*Completed: 2026-03-07*
