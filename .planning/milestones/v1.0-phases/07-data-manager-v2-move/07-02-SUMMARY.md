---
phase: 07-data-manager-v2-move
plan: 02
subsystem: infra
tags: [scaffold, import-linter, ci, data-manager, canonical-path]

requires:
  - phase: 07-data-manager-v2-move plan 01
    provides: data_manager_v2 moved to core/runtime/data_manager.py with backward-compatible shim
provides:
  - Scaffold templates generating code with canonical import path
  - CI check blocking new deprecated-path imports in scaffold/packages
  - Import-linter independence contract referencing canonical module
affects: [new-agent-scaffold, ci-pipeline]

tech-stack:
  added: []
  patterns: [ci-deprecated-import-guard, canonical-import-enforcement]

key-files:
  created: []
  modified:
    - lobster/scaffold/templates/agent.py.j2
    - lobster/scaffold/templates/shared_tools.py.j2
    - .importlinter
    - .github/workflows/ci-basic.yml

key-decisions:
  - "CI grep scope limited to lobster/scaffold/ and packages/ only -- existing 80 importers handled by shim"

patterns-established:
  - "CI deprecated-import guard: grep-based check in quality-and-tests job blocks regression"

requirements-completed: [DMGR-03, DMGR-04]

duration: 1min
completed: 2026-03-04
---

# Phase 7 Plan 02: Scaffold + CI Enforcement Summary

**Scaffold templates, import-linter, and CI updated to enforce canonical lobster.core.runtime.data_manager path**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-04T18:17:53Z
- **Completed:** 2026-03-04T18:19:07Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Both scaffold Jinja2 templates now generate code importing from canonical path
- Import-linter core-independence contract references canonical module
- CI workflow blocks PRs adding new deprecated-path imports in scaffold/packages

## Task Commits

Each task was committed atomically:

1. **Task 1: Update scaffold templates and import-linter to canonical path** - `4270a67` (chore)
2. **Task 2: Add CI check to block new deprecated-path imports** - `c654f88` (chore)

## Files Created/Modified
- `lobster/scaffold/templates/agent.py.j2` - Updated import to lobster.core.runtime.data_manager
- `lobster/scaffold/templates/shared_tools.py.j2` - Updated import to lobster.core.runtime.data_manager
- `.importlinter` - core-independence contract references canonical module path
- `.github/workflows/ci-basic.yml` - New step checking for deprecated data_manager_v2 imports

## Decisions Made
- CI grep scope limited to lobster/scaffold/ and packages/ only -- existing 80 importers in lobster/ and tests/ are handled by the backward-compatible shim and will be migrated later (SHIM-01)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 07 complete -- data_manager_v2 fully moved with shim (plan 01) and enforcement (plan 02)
- Ready for Phase 08

---
*Phase: 07-data-manager-v2-move*
*Completed: 2026-03-04*
