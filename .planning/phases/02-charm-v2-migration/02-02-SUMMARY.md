---
phase: 02-charm-v2-migration
plan: 02
subsystem: tui
tags: [charm-v2, bubbletea, lipgloss, bubbles, migration, go]

requires:
  - phase: 02-charm-v2-migration
    plan: 01
    provides: validated v2 API assumptions and string representation findings
provides:
  - all imports migrated from github.com/charmbracelet/* to charm.land/*/v2
  - View() returns tea.View across all models
  - color types migrated to color.Color
  - key handling migrated to KeyPressMsg Code/Mod/Text fields
affects: [02-03-PLAN]

tech-stack:
  removed: [github.com/charmbracelet/bubbletea, github.com/charmbracelet/lipgloss, github.com/charmbracelet/bubbles]
  migrated_to: [charm.land/bubbletea/v2, charm.land/lipgloss/v2, charm.land/bubbles/v2]
  patterns: [tea.NewView() wrapping, color.Color interface, KeyPressMsg with Code/Mod/Text]

key-files:
  modified:
    - lobster-tui/go.mod
    - lobster-tui/internal/chat/model.go
    - lobster-tui/internal/chat/views.go
    - lobster-tui/internal/chat/run.go
    - lobster-tui/internal/chat/completions.go
    - lobster-tui/internal/chat/content.go
    - lobster-tui/internal/chat/forms.go
    - lobster-tui/internal/chat/model_test.go
    - lobster-tui/internal/chat/content_test.go
    - lobster-tui/internal/chat/completions_test.go
    - lobster-tui/internal/chat/run_test.go
    - lobster-tui/internal/chat/views_intro_test.go
    - lobster-tui/internal/chat/component_lifecycle_test.go
    - lobster-tui/internal/theme/theme.go
    - lobster-tui/internal/theme/lobster.go
    - lobster-tui/internal/theme/loader.go
    - lobster-tui/internal/biocomp/component.go
    - lobster-tui/internal/biocomp/overlay.go
    - lobster-tui/internal/biocomp/overlay_test.go
    - lobster-tui/internal/biocomp/registry_test.go
    - lobster-tui/internal/biocomp/confirm/confirm.go
    - lobster-tui/internal/biocomp/confirm/confirm_test.go
    - lobster-tui/internal/biocomp/bioselect/select.go
    - lobster-tui/internal/biocomp/bioselect/select_test.go
    - lobster-tui/internal/biocomp/textinput/textinput.go
    - lobster-tui/internal/biocomp/textinput/textinput_test.go
    - lobster-tui/internal/biocomp/threshold/threshold.go
    - lobster-tui/internal/biocomp/threshold/threshold_test.go
    - lobster-tui/internal/biocomp/celltype/celltype.go
    - lobster-tui/internal/biocomp/celltype/celltype_test.go
    - lobster-tui/internal/biocomp/qcdash/qcdash.go
    - lobster-tui/internal/biocomp/qcdash/qcdash_test.go
    - lobster-tui/internal/initwizard/wizard.go

key-decisions:
  - "Applied spike findings: 'esc' not 'escape', 'space' not ' ' in all key switch cases"
  - "View.Content field access pattern for viewport and view output"
  - "viewport.New() with functional options throughout"

requirements-completed: [MIGR-01, MIGR-02, MIGR-03, MIGR-04]

duration: ~137min
completed: 2026-03-07
---

# Phase 2 Plan 2: Mechanical Import Migration Summary

**Migrated all 29 Go files from Charm v1 (github.com/charmbracelet/*) to v2 (charm.land/*/v2) — imports, types, View return type, color types, and key handling.**

## Performance

- **Duration:** ~137 min (connection dropped near end, all work completed)
- **Tasks:** 2
- **Files modified:** 29 (+go.mod, go.sum)

## Accomplishments
- All imports migrated from `github.com/charmbracelet/*` to `charm.land/*/v2`
- `View()` returns `tea.View` (via `tea.NewView()`) across all models and components
- Color types migrated to `color.Color` interface throughout theme system
- Key handling migrated to `KeyPressMsg` with `Code`/`Mod`/`Text` fields
- Applied spike findings: "esc" not "escape", "space" not " " in switch cases
- Full test suite passes (all packages green)
- Build succeeds with no errors
- 326 insertions, 358 deletions (net reduction in code)

## Task Commits

1. **Task 1: Migrate imports, types, and APIs** - `46777f8` (feat)
2. **Task 2: Fix test compilation** - `2504ea7` (fix)

## Files Modified
- 29 Go source files across chat/, biocomp/, theme/, initwizard/
- go.mod — v1 references replaced with v2
- go.sum — updated checksums

## Issues Encountered
- Agent connection dropped after completing all code work but before writing SUMMARY.md
- SUMMARY.md and state updates completed by orchestrator

## Next Phase Readiness
- All v1→v2 import migration complete
- huh dependency still present in initwizard — addressed by Plan 02-03
- Ready for 02-03-PLAN (init wizard rewrite, huh removal)

---
*Phase: 02-charm-v2-migration*
*Completed: 2026-03-07*
