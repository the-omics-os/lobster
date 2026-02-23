---
phase: 06-metabolomics-package
plan: 03
subsystem: metabolomics
tags: [metabolomics, LC-MS, GC-MS, NMR, system-prompt, agent-prompt, tool-selection, workflow-guidance]

# Dependency graph
requires:
  - phase: 06-metabolomics-package-plan-02
    provides: 10 metabolomics tools via create_shared_tools, agent factory with AGENT_CONFIG, minimal prompts.py placeholder
provides:
  - Full metabolomics_expert system prompt with identity, platform detection, 10-tool inventory, standard workflows, tool selection guide, and delegation protocol
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [Metabolomics prompt with platform-specific workflow guidance for LC-MS/GC-MS/NMR]

key-files:
  created: []
  modified:
    - packages/lobster-metabolomics/lobster/agents/metabolomics/prompts.py

key-decisions: []

patterns-established:
  - "Metabolomics prompt follows proteomics prompt structure: XML-tagged sections with Identity, Platform Detection, Tools, Workflows, Tool Selection Guide, Important Rules, Delegation Protocol"

requirements-completed: [DOC-06]

# Metrics
duration: 3min
completed: 2026-02-23
---

# Phase 6 Plan 03: Metabolomics Expert Prompt Summary

**Complete metabolomics_expert system prompt (16K chars) with LC-MS/GC-MS/NMR platform detection, 10 tools organized by workflow stage, standard workflows per platform, and tool selection disambiguation guide**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-23T06:17:22Z
- **Completed:** 2026-02-23T06:20:24Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Expanded minimal prompts.py placeholder into full 16,483-character system prompt covering all 8 required sections
- All 10 tools documented with usage guidance organized by workflow stage (QC, Preprocessing, Analysis, Annotation, Pathway)
- Platform-specific workflows for LC-MS (PQN + KNN + log2), GC-MS (TIC + min + log2), and NMR (PQN + median + no log)
- Tool selection disambiguation table preventing LLM confusion between similar tools (e.g., normalize vs impute)
- 15 important rules enforcing strict preprocessing order and model validation warnings
- Verified full agent stack: AGENT_CONFIG -> factory -> prompt -> METABOLOMICS_EXPERT_AVAILABLE = True

## Task Commits

Each task was committed atomically:

1. **Task 1: Create metabolomics_expert prompt** - `6befdb5` (feat)

## Files Created/Modified
- `packages/lobster-metabolomics/lobster/agents/metabolomics/prompts.py` - Full system prompt with identity, platform detection (LC-MS/GC-MS/NMR), 10-tool inventory, standard workflows, tool selection guide, important rules, delegation protocol, and response format guidance

## Decisions Made
None - followed plan as specified. The prompt structure follows the proteomics prompt pattern exactly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Metabolomics package is now feature-complete: 4 services (Plan 01), 10 tools + agent factory (Plan 02), full system prompt (Plan 03)
- Phase 06 complete: all 3 plans executed
- Agent stack verified end-to-end: AGENT_CONFIG resolves, factory function works, prompt imports correctly, METABOLOMICS_EXPERT_AVAILABLE = True

## Self-Check: PASSED

All files verified present. Task commit (6befdb5) verified in git log.

---
*Phase: 06-metabolomics-package*
*Completed: 2026-02-23*
