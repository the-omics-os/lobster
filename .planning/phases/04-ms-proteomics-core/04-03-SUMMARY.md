---
phase: 04-ms-proteomics-core
plan: 03
subsystem: proteomics-agent-prompt
tags: [proteomics, prompt, llm-guidance, tools, workflows, ptm, tmt, batch-correction, import]

# Dependency graph
requires:
  - phase: 04-ms-proteomics-core
    plan: 02
    provides: 5 new tools (import, PTM, batch correction, rollup, PTM normalization) and add_peptide_mapping deprecation
provides:
  - Rewritten create_proteomics_expert_prompt() with all 15 parent tools organized by workflow stage
  - 3 new workflows (MS Discovery, PTM Phosphoproteomics, TMT)
  - Tool Selection Guide for disambiguation (batch vs plate correction, import vs PTM import)
  - Updated Important Rules (10-12) including deprecation guidance
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Tool listing organized by workflow stage (Import, QC, Preprocessing, Processing, Analysis, Summary, Platform-specific)
    - Tool Selection Guide section for LLM disambiguation between similar tools

key-files:
  created: []
  modified:
    - packages/lobster-proteomics/lobster/agents/proteomics/prompts.py

key-decisions:
  - "No new decisions - executed plan as specified"

patterns-established:
  - "Prompt tool sections organized by workflow stage (not flat list) for better LLM tool selection"
  - "Tool Selection Guide as dedicated prompt section for disambiguating similar-purpose tools"

requirements-completed: [DOC-03]

# Metrics
duration: 2min
completed: 2026-02-23
---

# Phase 4 Plan 3: Proteomics Expert Prompt Rewrite Summary

**Rewritten proteomics expert prompt with 15 parent tools, 3 new workflows (MS import, PTM phosphoproteomics, TMT rollup), and tool selection guide**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-23T03:37:21Z
- **Completed:** 2026-02-23T03:40:01Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Rewrote `create_proteomics_expert_prompt()` with all 15 parent tools organized by workflow stage (was 11 in flat list)
- Added 3 new workflows: MS Discovery (updated with import_proteomics_data as first step), PTM Phosphoproteomics (import protein + PTM -> normalize -> DE handoff), TMT (import -> rollup -> batch correction -> normalize)
- Added Tool Selection Guide section disambiguating import_proteomics_data vs import_ptm_sites, correct_batch_effects vs correct_plate_effects, normalize_proteomics_data vs normalize_ptm_to_protein
- Added 3 new Important Rules: #10 (start MS with import), #11 (always import both protein + PTM data), #12 (never reference deprecated add_peptide_mapping)
- Added PTM Analysis subsection under Platform Considerations

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite create_proteomics_expert_prompt() for Phase 4 tool inventory** - `3011128` (feat)

## Files Created/Modified
- `packages/lobster-proteomics/lobster/agents/proteomics/prompts.py` - Rewrote create_proteomics_expert_prompt() with 15 tools, 3 workflows, tool selection guide, PTM considerations, updated rules (+108 -36 lines)

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 (MS Proteomics Core) is now complete: all 3 plans executed
- Proteomics expert agent has complete tool inventory (15 parent + 2 delegation), service methods (Plan 01), tool wrappers (Plan 02), and LLM guidance prompt (Plan 03)
- Ready for Phase 5 planning

## Self-Check: PASSED

- FOUND: prompts.py
- FOUND: 3011128 (Task 1 commit)
- FOUND: 04-03-SUMMARY.md

---
*Phase: 04-ms-proteomics-core*
*Completed: 2026-02-23*
