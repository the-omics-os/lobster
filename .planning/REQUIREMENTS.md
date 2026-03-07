# Requirements: Lobster TUI v2 Overhaul

**Defined:** 2026-03-06
**Core Value:** The TUI renders structured protocol data beautifully and correctly -- typed content blocks survive to render time, components have correct lifecycle semantics, and the layout adapts dynamically.

## v1 Requirements

Requirements for the complete TUI overhaul. Each maps to roadmap phases.

### Data Model

- [x] **DATA-01**: ChatMessage uses typed ContentBlock array instead of flat string
- [x] **DATA-02**: Protocol handlers create BlockTable/BlockCode/BlockAlert/BlockHandoff from typed messages
- [x] **DATA-03**: Streaming flushes text buffer as BlockText before appending typed blocks
- [x] **DATA-04**: Content() helper concatenates text blocks for backward compatibility

### Component Lifecycle

- [x] **COMP-01**: threshold_slider and cell_type_selector instantiate native components (not text_input fallback)
- [x] **COMP-02**: Esc sends cancel action distinct from empty data submission
- [x] **COMP-03**: Active component replacement sends cancel to old component before init new
- [x] **COMP-04**: Layout size hints shared between layoutReservedRows() and View()
- [x] **COMP-05**: ChangeEvent fires after component value changes with 50ms debounce
- [x] **COMP-06**: SetData updates active component when protocol sends updated data
- [x] **COMP-07**: Error boundary in handleComponentRender() -- bad Init sends error response, no crash
- [x] **COMP-08**: Stale SetData guard discards updates for wrong component MsgID

### Framework Migration

- [ ] **MIGR-01**: All imports migrated from github.com/charmbracelet/* to charm.land/*/v2
- [ ] **MIGR-02**: View() returns tea.View (not string) across all models
- [x] **MIGR-03**: All tea.KeyMsg handling rewritten to tea.KeyPressMsg (Code/Mod fields)
- [ ] **MIGR-04**: Color types updated from lipgloss string to image/color.Color
- [x] **MIGR-05**: Init wizard rewritten with bubbles v2 primitives (huh dependency removed)
- [x] **MIGR-06**: Forms rewritten with bubbles v2 primitives
- [x] **MIGR-07**: Phase 2A spike validates v2 API assumptions before full migration

### Rendering

- [x] **REND-01**: Tables render via lipgloss/table with rounded borders, themed header/row styles
- [x] **REND-02**: Messages flow without box borders (crush-style, padding+margin only)
- [x] **REND-03**: Code blocks render with language labels and syntax highlighting
- [x] **REND-04**: Alert blocks render with colored severity indicators
- [x] **REND-05**: Agent handoff blocks render with from/to/reason formatting
- [x] **REND-06**: Width-keyed render cache for finalized messages (invalidate on width/theme change)
- [x] **REND-07**: Streaming messages are never cached

### Layout

- [x] **LAYO-01**: 4-layer layout with header, viewport, input, footer regions via computeLayout()
- [x] **LAYO-02**: Footer shows status line (spinner + agent + cost) when idle
- [ ] **LAYO-03**: Footer expands for active BioCharm component with help bar
- [ ] **LAYO-04**: Footer contracts when component dismisses
- [ ] **LAYO-05**: Terminal resize triggers layout recomputation without visual artifacts
- [x] **LAYO-06**: Tool feed migrates from mid-View to footer region

### Style System

- [x] **STYL-01**: Expanded Styles struct with 40+ semantic tokens
- [x] **STYL-02**: Dedicated table styles (TableHeader, TableRowEven, TableRowOdd, TableBorder)
- [x] **STYL-03**: Dedicated alert styles (AlertSuccess, AlertWarning, AlertError, AlertInfo)
- [x] **STYL-04**: Dedicated code styles (CodeBlock, CodeLabel)
- [x] **STYL-05**: Dedicated footer styles (FooterStatus, FooterToolFeed, FooterComponentFrame)
- [x] **STYL-06**: Chat styles (AgentName, UserName, MessageBody, HandoffPrefix)

### Python Integration

- [ ] **PYTH-01**: Supervisor ask-user tool with LLM-driven component selection

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Components

- **ACOMP-01**: OntologyBrowser -- GO/Reactome/CellOntology tree navigation with lazy-load expand
- **ACOMP-02**: SequenceInput -- DNA/RNA/protein sequence entry with validation
- **ACOMP-03**: DNAAnimation -- ASCII double helix loading animation replacing braille spinner

### Advanced Rendering

- **AREND-01**: Lipgloss v2 compositing/layers for overlay rendering
- **AREND-02**: Theme customization via user config files

## Out of Scope

| Feature | Reason |
|---------|--------|
| Protocol JSON format changes | Cloud UI and Python bridge depend on current format. Block model is Go-internal. |
| Overlay rendering for components | Cannot paint over content in sequential string concat. Footer-based is the solution. |
| Windows support | Not in current platform scope |
| OntologyBrowser / SequenceInput / DNAAnimation | Post-overhaul components -- need solid lifecycle foundation first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| COMP-01 | Phase 1 | Complete |
| COMP-02 | Phase 1 | Complete |
| COMP-03 | Phase 1 | Complete |
| COMP-04 | Phase 1 | Complete |
| COMP-05 | Phase 1 | Complete |
| COMP-06 | Phase 1 | Complete |
| COMP-07 | Phase 1 | Complete |
| COMP-08 | Phase 1 | Complete |
| MIGR-01 | Phase 2 | Pending |
| MIGR-02 | Phase 2 | Pending |
| MIGR-03 | Phase 2 | Complete |
| MIGR-04 | Phase 2 | Pending |
| MIGR-05 | Phase 2 | Complete |
| MIGR-06 | Phase 2 | Complete |
| MIGR-07 | Phase 2 | Complete |
| REND-01 | Phase 3 | Complete |
| REND-02 | Phase 3 | Complete |
| REND-03 | Phase 3 | Complete |
| REND-04 | Phase 3 | Complete |
| REND-05 | Phase 3 | Complete |
| REND-06 | Phase 3 | Complete |
| REND-07 | Phase 3 | Complete |
| STYL-01 | Phase 3 | Complete |
| STYL-02 | Phase 3 | Complete |
| STYL-03 | Phase 3 | Complete |
| STYL-04 | Phase 3 | Complete |
| STYL-05 | Phase 3 | Complete |
| STYL-06 | Phase 3 | Complete |
| LAYO-01 | Phase 4 | Complete |
| LAYO-02 | Phase 4 | Complete |
| LAYO-03 | Phase 4 | Pending |
| LAYO-04 | Phase 4 | Pending |
| LAYO-05 | Phase 4 | Pending |
| LAYO-06 | Phase 4 | Complete |
| PYTH-01 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-07 after 03-01-PLAN.md completion*
