---
phase: 04-layout
plan: 02
subsystem: ui
tags: [go, bubbletea, lipgloss, joinvertical, layout-engine, biocomp, footer]

requires:
  - phase: 04-layout-01
    provides: Layout struct, computeLayout(), FooterMode state machine, footer renderers
provides:
  - View() rewritten with lipgloss.JoinVertical for 4-layer composition
  - renderComponentFooter with biocomp.RenderFrame and safeView fallback
  - componentFooterHeight with per-component sizing and max-half-terminal clamping
  - renderHeaderRegion, renderViewportRegion, renderInputRegion methods
  - Inline mode preserved unchanged in viewInline()
affects: [05-lifecycle, component-hosting, resize-handling]

tech-stack:
  added: []
  patterns: [joinvertical-region-composition, footer-hosted-components, inline-mode-isolation]

key-files:
  created: []
  modified:
    - lobster-tui/internal/chat/model.go
    - lobster-tui/internal/chat/views.go
    - lobster-tui/internal/chat/layout.go
    - lobster-tui/internal/chat/layout_test.go

key-decisions:
  - "View() split into non-inline (JoinVertical) and viewInline() (legacy builder) to preserve inline behavior"
  - "Viewport region skips lipgloss Width constraint to preserve scrollbar column appended by renderViewportWithScrollbar"
  - "componentFooterHeight uses fixed base heights per component name (15/8/10) plus 3 for borders+help, avoiding double-render anti-pattern"
  - "Overlay lipgloss.Place removed entirely; components now render in footer frame via renderComponentFooter"

patterns-established:
  - "Region renderers: renderHeaderRegion, renderViewportRegion, renderInputRegion, renderFooterRegion each accept Layout and return height-constrained strings"
  - "Footer-hosted components: biocomp.RenderFrame within FooterComponentFrame style, fallback to status footer on panic"

requirements-completed: [LAYO-03, LAYO-04, LAYO-05]

duration: 7min
completed: 2026-03-07
---

# Phase 04 Plan 02: View() Rewrite Summary

**4-layer JoinVertical composition replacing string builder, with component footer hosting and overlay removal**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-07T05:50:43Z
- **Completed:** 2026-03-07T05:57:58Z
- **Tasks:** 2 of 3 (task 3 is human verification checkpoint)
- **Files modified:** 4

## Accomplishments
- View() rewritten to use lipgloss.JoinVertical with 4 region renderers (header, viewport, input, footer)
- Component footer renderer with biocomp.RenderFrame, safeView panic fallback, and per-component height sizing
- componentFooterHeight with max-half-terminal clamping (min 5, max min(height/2, 20))
- Inline mode extracted to viewInline() preserving legacy behavior unchanged
- Tool feed removed from mid-view position, now renders in footer via renderFooterRegion
- Overlay lipgloss.Place code removed; components render in expanding footer frame
- recalculateViewportHeight uses computeLayout() for non-inline mode
- 4 new tests: expand, contract, resize invariant, small-terminal clamp

## Task Commits

Each task was committed atomically:

1. **Task 1: Component footer renderer + expand/contract + resize tests** - `6cfca6b` (feat)
2. **Task 2: Rewrite View() with JoinVertical layout composition** - `85fa13e` (feat)

## Files Created/Modified
- `lobster-tui/internal/chat/layout.go` - componentFooterHeight() with per-component base heights and clamping
- `lobster-tui/internal/chat/layout_test.go` - 4 new tests (expand, contract, resize, clamp) + mock BioComponent
- `lobster-tui/internal/chat/views.go` - renderComponentFooter, renderHeaderRegion, renderViewportRegion, renderInputRegion, wired FooterModeComponent dispatch
- `lobster-tui/internal/chat/model.go` - View() split into JoinVertical non-inline and viewInline(), recalculateViewportHeight uses computeLayout()

## Decisions Made
- Split View() into two methods rather than keeping inline conditional blocks -- cleaner separation, inline code preserved verbatim
- Viewport region renders without Width constraint because scrollbar column extends beyond viewport width
- Component footer height uses fixed allocation per component name instead of calling View() to measure (avoids double-render anti-pattern)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Scrollbar thumb stripped by lipgloss Width constraint**
- **Found during:** Task 2 (View() rewrite)
- **Issue:** renderViewportWithScrollbar appends scrollbar column beyond viewport width; lipgloss Width(m.width) truncated the thumb character
- **Fix:** Removed Width constraint from renderViewportRegion, relying on viewport's internal width management
- **Files modified:** lobster-tui/internal/chat/views.go
- **Verification:** TestViewRendersScrollbarWhenViewportScrollable passes
- **Committed in:** 85fa13e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for scrollbar visibility. No scope creep.

## Issues Encountered
None beyond the auto-fixed scrollbar issue.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 4-layer layout system complete and tested
- Task 3 (human verification) pending -- visual check of layout rendering
- Ready for phase 05 (component lifecycle) after verification

---
*Phase: 04-layout*
*Completed: 2026-03-07*
