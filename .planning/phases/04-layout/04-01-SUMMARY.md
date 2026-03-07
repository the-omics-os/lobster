---
phase: 04-layout
plan: 01
subsystem: ui
tags: [go, bubbletea, layout-engine, lipgloss, tdd]

requires:
  - phase: 03-rendering-and-style
    provides: theme styles (FooterStatus, FooterToolFeed, FooterComponentFrame)
provides:
  - Layout struct with 4-region height computation
  - FooterMode state machine (Status/ToolFeed/Component)
  - computeLayout() single source of truth
  - Footer rendering functions (renderFooterRegion, renderStatusFooter, renderToolFeedFooter)
  - layoutReservedRows() backward compatibility delegation
affects: [04-02-PLAN, view-refactor, component-lifecycle]

tech-stack:
  added: []
  patterns: [4-layer-layout-engine, footer-state-machine, greedy-viewport]

key-files:
  created:
    - lobster-tui/internal/chat/layout.go
    - lobster-tui/internal/chat/layout_test.go
  modified:
    - lobster-tui/internal/chat/model.go
    - lobster-tui/internal/chat/views.go

key-decisions:
  - "Footer renderers live in views.go (rendering concern), layout computation in layout.go (data concern)"
  - "layoutReservedRows() delegates to computeLayout() for non-inline, preserves legacy path for inline mode"
  - "Component footer defaults to half-height (max 20, min 3) as placeholder for plan 02"

patterns-established:
  - "4-layer layout: Header + Viewport + Input + Footer always sum to m.height"
  - "Footer state machine: footerMode() classifies, footerHeight() sizes, renderFooterRegion() dispatches"
  - "Viewport is greedy: absorbs remaining height after fixed regions, clamped to min 1"

requirements-completed: [LAYO-01, LAYO-02, LAYO-06]

duration: 3min
completed: 2026-03-07
---

# Phase 04 Plan 01: Layout Engine Summary

**4-layer layout engine with Layout struct, computeLayout() single source of truth, FooterMode state machine, and styled footer renderers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T05:45:13Z
- **Completed:** 2026-03-07T05:48:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Layout struct with HeaderHeight, ViewportHeight, InputHeight, FooterHeight -- heights always sum to m.height
- FooterMode state machine (Status/ToolFeed/Component) with footerMode(), footerHeight(), renderFooterRegion()
- computeLayout() replaces scattered height bookkeeping as single source of truth
- layoutReservedRows() delegates to computeLayout() for non-inline mode, preserving backward compatibility
- 9 unit tests covering height invariant, footer modes, viewport greedy absorption, minimum viewport, inline early return

## Task Commits

Each task was committed atomically:

1. **Task 1: Layout engine types, computeLayout, and tests** - `2bdccc6` (feat)
2. **Task 2: Footer rendering functions in views.go** - `22e8873` (feat)

## Files Created/Modified
- `lobster-tui/internal/chat/layout.go` - Layout struct, FooterMode enum, computeLayout(), footerMode(), footerHeight(), layoutReservedRows() delegation
- `lobster-tui/internal/chat/layout_test.go` - 9 unit tests for layout engine
- `lobster-tui/internal/chat/model.go` - layoutReservedRows() renamed to layoutReservedRowsLegacy() for inline mode
- `lobster-tui/internal/chat/views.go` - renderFooterRegion(), renderStatusFooter(), renderToolFeedFooter() with FooterStatus/FooterToolFeed styles

## Decisions Made
- Footer renderers placed in views.go (rendering concern) while layout computation stays in layout.go (data concern)
- layoutReservedRows() delegates to computeLayout() for non-inline mode but preserves the legacy inline path unchanged
- Component footer height uses a reasonable default (half-height, max 20) as placeholder until plan 02 implements LAYO-03/LAYO-04

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Layout engine ready for View() migration in plan 02
- renderFooterRegion can be called from View() to replace scattered tool feed + status bar rendering
- Component footer (FooterModeComponent) falls back to status -- plan 02 implements the real renderer

---
*Phase: 04-layout*
*Completed: 2026-03-07*
