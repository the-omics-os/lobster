# Roadmap: Lobster TUI v2 Overhaul

## Overview

Incremental overhaul of the Go TUI from Charm v1 to v2, introducing typed content blocks, fixing BioComp lifecycle bugs, migrating the framework, and delivering native table/code rendering with a 4-layer layout system. Each phase produces a compiling, testable binary. The critical path is Foundation (data model) -> Migration (v2 gate) -> Rendering (first visible payoff). Layout and Python integration follow once the rendering layer is solid.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Typed content block model + BioComp lifecycle fixes
- [x] **Phase 2: Charm v2 Migration** - Full framework migration from v1 to v2 (imports, key handling, huh removal) (completed 2026-03-07)
- [x] **Phase 3: Rendering and Style** - Native tables, clean messages, code blocks, alerts, render cache, semantic style system (completed 2026-03-07)
- [ ] **Phase 4: Layout** - 4-layer layout system with dynamic footer for components
- [ ] **Phase 5: Python Integration** - Supervisor ask-user tool with LLM-driven component selection

## Phase Details

### Phase 1: Foundation
**Goal**: Messages carry typed structure from protocol to renderer, and BioComp components work correctly
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, COMP-01, COMP-02, COMP-03, COMP-04, COMP-05, COMP-06, COMP-07, COMP-08
**Success Criteria** (what must be TRUE):
  1. Protocol table/code/alert/handoff messages arrive at View() as typed ContentBlocks (not flat strings)
  2. Streaming text followed by a structured block produces both blocks without data loss
  3. BioComp threshold_slider and cell_type_selector instantiate their native components (not text_input fallback)
  4. Pressing Esc on an active component sends a cancel action distinct from submitting empty data
  5. SetData updates only reach the matching active component (stale MsgID updates are discarded)
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Typed content block model (content.go, protocol handler updates, Content() helper)
- [x] 01-02-PLAN.md -- BioComp lifecycle fixes (imports, cancel, replacement, layout math, SetData/ChangeEvent, error boundary, stale guard)

### Phase 2: Charm v2 Migration
**Goal**: Entire TUI runs on Charm v2 packages with no v1 dependencies remaining
**Depends on**: Phase 1
**Requirements**: MIGR-01, MIGR-02, MIGR-03, MIGR-04, MIGR-05, MIGR-06, MIGR-07
**Success Criteria** (what must be TRUE):
  1. All Go imports use charm.land/*/v2 paths (no github.com/charmbracelet/* v1 imports remain)
  2. View() returns tea.View across all models (not string)
  3. Init wizard completes all 5 steps using bubbles v2 primitives (huh dependency removed from go.mod)
  4. All keyboard shortcuts work correctly with v2 KeyPressMsg (space, enter, esc, ctrl+c, arrow keys)
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md -- API spike validating v2 assumptions (View type, color types, KeyPressMsg, viewport, glamour coexistence)
- [x] 02-02-PLAN.md -- Mechanical import migration + View() return type + color type updates + KeyMsg->KeyPressMsg
- [x] 02-03-PLAN.md -- Init wizard rewrite with v2 state machine + forms.go rewrite + huh removal

### Phase 3: Rendering and Style
**Goal**: Users see beautiful tables, clean flowing messages, syntax-highlighted code, and consistent visual styling
**Depends on**: Phase 2
**Requirements**: REND-01, REND-02, REND-03, REND-04, REND-05, REND-06, REND-07, STYL-01, STYL-02, STYL-03, STYL-04, STYL-05, STYL-06
**Success Criteria** (what must be TRUE):
  1. Tables render with rounded borders and themed header/row styles via lipgloss/table (no ASCII art)
  2. Messages flow without box borders (crush-style with padding and margin only)
  3. Code blocks display language labels and syntax highlighting
  4. Resizing the terminal re-renders finalized messages from cache without flickering or delay
  5. All block types (table, code, alert, handoff) have dedicated style tokens in the theme system
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md -- Semantic style expansion + native table rendering + crush-style messages + render cache
- [x] 03-02-PLAN.md -- Code block rendering with chroma + alert/handoff renderers + views.go wiring

### Phase 4: Layout
**Goal**: TUI has a proper layout system with dynamic footer that hosts interactive components
**Depends on**: Phase 2
**Requirements**: LAYO-01, LAYO-02, LAYO-03, LAYO-04, LAYO-05, LAYO-06
**Success Criteria** (what must be TRUE):
  1. Terminal window has distinct header, viewport, input, and footer regions computed by computeLayout()
  2. Footer shows status line (spinner + agent + cost) when no component is active
  3. When a BioCharm component activates, footer expands to host it; when it dismisses, footer contracts back
  4. Resizing the terminal triggers layout recomputation without visual artifacts or height oscillation
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Layout engine core (Layout struct, computeLayout, footer state machine, footer renderers)
- [ ] 04-02-PLAN.md -- View() rewrite with JoinVertical + component footer hosting + resize handling

### Phase 5: Python Integration
**Goal**: Supervisor can ask users interactive questions via BioCharm components selected by the LLM
**Depends on**: Phase 4
**Requirements**: PYTH-01
**Success Criteria** (what must be TRUE):
  1. Supervisor ask-user tool triggers the correct BioCharm component type based on question context
  2. User response from the TUI component flows back to the supervisor as tool result
**Plans**: TBD

Plans:
- [ ] 05-01: Supervisor ask-user tool with LLM-driven component selection

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5
(Phases 3 and 4 both depend on Phase 2 but execute sequentially for solo developer safety)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-03-07 |
| 2. Charm v2 Migration | 3/3 | Complete   | 2026-03-07 |
| 3. Rendering and Style | 2/2 | Complete   | 2026-03-07 |
| 4. Layout | 0/2 | Not started | - |
| 5. Python Integration | 0/1 | Not started | - |
