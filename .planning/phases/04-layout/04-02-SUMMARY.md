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
  - "Viewport width set to m.width-1 to reserve scrollbar column; all regions constrained with Width+MaxWidth to prevent JoinVertical overflow"
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
- **Tasks:** 3 of 3 (task 3 checkpoint: resize fix applied after human verification)
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
3. **Task 3: Resize stability fix (checkpoint)** - `17038b8` (fix) — viewport width-1 for scrollbar, Width+MaxWidth on all regions, geometry-first header, regression tests

## Files Created/Modified
- `lobster-tui/internal/chat/layout.go` - componentFooterHeight() with per-component base heights and clamping
- `lobster-tui/internal/chat/layout_test.go` - 4 new tests (expand, contract, resize, clamp) + mock BioComponent
- `lobster-tui/internal/chat/views.go` - renderComponentFooter, renderHeaderRegion, renderViewportRegion, renderInputRegion, wired FooterModeComponent dispatch
- `lobster-tui/internal/chat/model.go` - View() split into JoinVertical non-inline and viewInline(), recalculateViewportHeight uses computeLayout()

## Decisions Made
- Split View() into two methods rather than keeping inline conditional blocks -- cleaner separation, inline code preserved verbatim
- Viewport width = m.width-1 to reserve scrollbar column within terminal width budget (fixes JoinVertical overflow)
- All 4 regions get Width(m.width).MaxWidth(m.width) to prevent any region from expanding the JoinVertical frame
- Geometry-first header: constant 2 rows instead of render-to-measure lineCount(renderHeader(m))
- Component footer height uses fixed allocation per component name instead of calling View() to measure (avoids double-render anti-pattern)
- Footer status line truncated via MaxWidth to prevent wrapping at narrow terminals

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Resize instability: JoinVertical overflow from scrollbar width**
- **Found during:** Task 3 (human verification checkpoint)
- **Issue:** viewport.View() returned m.width-wide content, renderViewportWithScrollbar appended +1 char, JoinVertical padded all regions to m.width+1, terminal auto-wrapped causing ghosting and duplication
- **Root cause verified by:** Codex GPT-5.4 analysis of BubbleTea v2 / lipgloss v2 / Bubbles v2 source
- **Fix:** 6 changes: (1) viewport width=m.width-1, (2) Width+MaxWidth on all regions, (3) status truncation, (4) fix double-wrap in tool feed footer, (5) geometry-first header, (6) regression tests
- **Files modified:** layout.go, layout_test.go, views.go, model.go, model_test.go
- **Verification:** 4 new regression tests + full suite passes
- **Committed in:** 17038b8

---

**Total deviations:** 1 auto-fixed (1 critical resize bug)
**Impact on plan:** Essential fix for resize stability. No scope creep.

## Issues Encountered
Resize instability found during human checkpoint — fully resolved with 6-part fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 4-layer layout system complete, resize-stable, and regression-tested
- Task 3 checkpoint resolved — resize fix applied and verified
- Ready for phase 05 (component lifecycle)

---
*Phase: 04-layout*
*Completed: 2026-03-07*
