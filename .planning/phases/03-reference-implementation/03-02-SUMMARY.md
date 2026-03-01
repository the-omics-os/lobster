---
phase: 03-reference-implementation
plan: 02
subsystem: documentation
tags: [aquadif, migration-guide, skill, lobster-dev, phase-4-rollout]

# Dependency graph
requires:
  - phase: 03-reference-implementation
    plan: 01
    provides: Complete transcriptomics tool categorization mapping table (22 tools with categories, provenance flags, and ambiguous tool rationale)
provides:
  - AQUADIF migration reference document (skills/lobster-dev/references/aquadif-migration.md)
  - Updated MANIFEST for automatic skill distribution
  - Phase 4 rollout guide covering 7-step migration checklist, code patterns, and transcriptomics worked example
affects: [phase-04-rollout, lobster-dev-skill, contributor-onboarding]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Migration checklist pattern: 7-step process from tool inventory to regression testing"
    - "Category decision quick reference: compact lookup table for fast categorization during rollout"

key-files:
  created:
    - skills/lobster-dev/references/aquadif-migration.md
  modified:
    - skills/lobster-dev/MANIFEST

key-decisions:
  - "302-line document balances completeness with agent readability -- slightly over 250 target but all content substantive"
  - "String literals recommended over enum imports in tool files to reduce coupling"
  - "Anti-patterns section included to prevent common mistakes observed in Phase 1 eval"

patterns-established:
  - "Migration guide as skill reference: lives in lobster-dev/references/ for automatic distribution via installer"
  - "Worked example pattern: real mapping table from reference implementation as concrete analogy for rollout"

requirements-completed: [IMPL-04]

# Metrics
duration: 2min
completed: 2026-03-01
---

# Phase 3 Plan 02: Migration Guide Summary

**AQUADIF migration reference with 7-step checklist, 5 annotated code patterns, and transcriptomics worked example (22 tools) for Phase 4 rollout of 156 remaining tools**

## Performance

- **Duration:** 2 min (118s)
- **Started:** 2026-03-01T04:11:39Z
- **Completed:** 2026-03-01T04:13:37Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created 302-line migration reference document covering the complete AQUADIF tagging process
- Included full transcriptomics mapping table (22 tools) with rationale for 5 ambiguous tool categorizations
- Updated MANIFEST so migration guide auto-distributes via skill installer (no Lambda redeploy needed)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AQUADIF migration reference document** - `035a4e8` (feat)
2. **Task 2: Update MANIFEST for skill distribution** - `94fb3a9` (chore)

## Files Created/Modified
- `skills/lobster-dev/references/aquadif-migration.md` - Migration guide with checklist, code patterns, worked example, and quick reference
- `skills/lobster-dev/MANIFEST` - Added aquadif-migration.md entry after aquadif-contract.md

## Decisions Made
- Slightly exceeded 250-line target (302 lines) because all sections carry substantive content needed for Phase 4 rollout
- Included anti-patterns section based on patterns observed in Phase 1 skill evaluation (agents consistently make these mistakes)
- Recommended string literals over enum imports in tool files -- reduces coupling while contract tests still validate

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 complete: both plans (reference implementation + migration guide) delivered
- Phase 4 executor (Claude) can follow the migration guide to tag 156 remaining tools across 8 packages
- Contract test mixin (from Plan 01) + migration guide (this plan) provide everything needed for rollout
- All deliverables distributed via lobster-dev skill installer

---
*Phase: 03-reference-implementation*
*Completed: 2026-03-01*
