---
phase: 02-charm-v2-migration
plan: 01
subsystem: tui
tags: [charm-v2, bubbletea, lipgloss, bubbles, glamour, spike, go]

requires:
  - phase: 01-foundation
    provides: typed content block model and BioComp lifecycle (prerequisite for v2 migration)
provides:
  - validated v2 API assumptions (View type, color types, KeyPressMsg, viewport, glamour coexistence)
  - documented string representations for key constants (esc not escape, space not " ")
affects: [02-02-PLAN, 02-03-PLAN]

tech-stack:
  added: [charm.land/bubbletea/v2 v2.0.1, charm.land/lipgloss/v2 v2.0.0, charm.land/bubbles/v2 v2.0.0]
  patterns: [v2 spike testing in internal/spike/, functional options for viewport.New()]

key-files:
  created: [lobster-tui/internal/spike/v2_api_test.go]
  modified: [lobster-tui/go.mod, lobster-tui/go.sum]

key-decisions:
  - "KeyEscape.String() returns 'esc' not 'escape' -- migration must use 'esc' in switch cases"
  - "KeySpace.String() returns 'space' not ' ' -- migration must update space bar handling"
  - "View.Content field (not .Body() method) for accessing rendered string"
  - "viewport.New() uses functional options (WithWidth, WithHeight) not direct field assignment"
  - "Glamour v0.10 + lipgloss v2 coexist without module conflict (different module paths)"

patterns-established:
  - "v2 spike tests in internal/spike/ package -- isolated from production code"
  - "tea.NewView(s) wraps string to tea.View for top-level Model.View()"

requirements-completed: [MIGR-07]

duration: 3min
completed: 2026-03-07
---

# Phase 2 Plan 1: API Spike Summary

**Charm v2 API spike validating 6 assumptions: View type, color.Color interface, KeyPressMsg string matching, viewport functional options, glamour coexistence, and View field-based program options**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T01:21:32Z
- **Completed:** 2026-03-07T01:24:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- All 6 v2 API assumptions validated by passing tests
- Discovered key string representation differences: "esc" (not "escape"), "space" (not " ")
- Confirmed glamour v0.10 and lipgloss v2 coexist without module conflict
- v2 dependencies added alongside v1 (both coexist in go.mod)
- Full existing test suite remains green (all packages pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add v2 dependencies to go.mod** - `9296098` (chore)
2. **Task 2: Create v2 API spike tests** - `296a89d` (test)

## Files Created/Modified
- `lobster-tui/internal/spike/v2_api_test.go` - 6 isolated tests validating v2 API assumptions (170 lines)
- `lobster-tui/go.mod` - Added charm.land v2 dependencies alongside existing v1
- `lobster-tui/go.sum` - Updated checksums for v2 packages

## Decisions Made
- KeyEscape strings to "esc" not "escape" -- Plan 02 must use `case "esc":` in all switch statements
- KeySpace strings to "space" not " " -- Plan 02 must update space bar handling from `case " ":` to `case "space":`
- View.Content is a public string field, not accessed via method -- Plan 02 migration is straightforward
- viewport.New() uses functional options pattern (WithWidth, WithHeight) not struct field assignment
- Glamour coexistence confirmed -- no special handling needed in Plan 02

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed View.Body() to View.Content**
- **Found during:** Task 2 (spike test creation)
- **Issue:** Plan assumed tea.View has a `.Body()` method; actual API uses `.Content` field
- **Fix:** Changed test to use `v.Content` instead of `v.Body()`
- **Files modified:** lobster-tui/internal/spike/v2_api_test.go
- **Verification:** Test compiles and passes
- **Committed in:** 296a89d (Task 2 commit)

**2. [Rule 1 - Bug] Fixed viewport field assignment to functional options**
- **Found during:** Task 2 (spike test creation)
- **Issue:** Plan assumed `vp.Width = 40` direct field assignment; v2 viewport uses functional options
- **Fix:** Changed to `viewport.New(viewport.WithWidth(40), viewport.WithHeight(10))`
- **Files modified:** lobster-tui/internal/spike/v2_api_test.go
- **Verification:** Test compiles and passes
- **Committed in:** 296a89d (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs -- plan assumptions vs actual API)
**Impact on plan:** Both fixes were necessary for correctness. Discovered during spike as intended -- this is exactly why we run spikes before migration.

## Issues Encountered
None -- all v2 packages resolved correctly and tests passed after API corrections.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- v2 packages confirmed working alongside v1 in go.mod
- Key string mappings documented for Plan 02 mechanical migration
- Viewport functional options pattern documented for Plan 02
- Ready to proceed with 02-02-PLAN (mechanical import migration)

---
*Phase: 02-charm-v2-migration*
*Completed: 2026-03-07*
