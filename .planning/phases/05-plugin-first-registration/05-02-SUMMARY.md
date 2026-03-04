---
phase: 05-plugin-first-registration
plan: "02"
subsystem: plugin-registration
tags: [pyproject.toml, entry-points, importlib.metadata, queue-preparers, download-services, metabolights]

# Dependency graph
requires:
  - phase: 05-01
    provides: RED contract tests gating entry-point declarations; test_plugin_registration.py with 4 failing tests
provides:
  - "[project.entry-points.\"lobster.queue_preparers\"] section in pyproject.toml with 5 databases"
  - "[project.entry-points.\"lobster.download_services\"] section in pyproject.toml with 5 databases"
  - MetaboLights first-time registration in both entry-point groups
  - test_plugin_registration.py 4/4 GREEN
affects:
  - 05-03 (TestFallbackGating RED tests now clearly target the missing _ALLOW_HARDCODED_FALLBACK constant)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Entry-point declarations immediately follow the lobster.services section in pyproject.toml for discoverability"
    - "Entry-point names match supported_databases()[0] values: geo, sra, pride, massive, metabolights"
    - "uv pip install -e . MUST be run after every pyproject.toml entry-point change to regenerate dist-info"

key-files:
  created: []
  modified:
    - pyproject.toml

key-decisions:
  - "Entry-point names are lowercase abbreviated keys (geo, sra, pride, massive, metabolights) matching supported_databases()[0] for each class"
  - "Two separate entry-point groups declared: lobster.queue_preparers and lobster.download_services (separate groups allow independent discovery)"
  - "pyproject.toml CLAUDE.md Hard Rule #2 says 'dependency changes go through humans' — entry-point declarations are service discovery metadata, NOT dependency changes; this edit is correct and required"

# Metrics
duration: 1min
completed: 2026-03-04
---

# Phase 5 Plan 02: Declare queue_preparers and download_services Entry Points Summary

**Two new [project.entry-points.*] sections in pyproject.toml — 5 queue preparers + 5 download services discoverable via importlib.metadata; MetaboLights registered for the first time; test_plugin_registration.py 4/4 GREEN**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-04T07:40:44Z
- **Completed:** 2026-03-04T07:41:44Z
- **Tasks:** 1
- **Files modified:** 1 (pyproject.toml)

## Accomplishments
- Added `[project.entry-points."lobster.queue_preparers"]` section with 5 entries: geo, sra, pride, massive, metabolights
- Added `[project.entry-points."lobster.download_services"]` section with 5 entries: geo, sra, pride, massive, metabolights
- Reinstalled package via `uv pip install -e .` to regenerate dist-info
- All 4 contract tests in `test_plugin_registration.py` now pass GREEN (PLUG-03, PLUG-04)
- MetaboLights registered in BOTH groups — was previously missing from the hardcoded QPS fallback
- Zero regressions: 47 previously-passing tests continue to pass; 2 TestFallbackGating RED tests intentionally remain RED (gating Plan 05-03)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add entry-point declarations to pyproject.toml** - `7313551` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `pyproject.toml` — Added two new `[project.entry-points.*]` sections between the `lobster.services` section and the "Agent Entry Points" comment block (lines 276-298 after insertion)

## Decisions Made
- **Entry-point names match supported_databases()[0]:** The names `geo`, `sra`, `pride`, `massive`, `metabolights` are lowercase abbreviations that match each class's `supported_databases()` first element. ComponentRegistry uses these names to look up handlers at runtime.
- **Two separate groups (not one combined group):** `lobster.queue_preparers` and `lobster.download_services` are kept separate so each ComponentRegistry discovery call can independently iterate only the relevant handlers without filtering.
- **Entry points declared after lobster.services block:** Placement immediately after the existing services section maintains logical grouping of service-discovery metadata, separated from agent entry points and workspace config.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None.

## Next Phase Readiness
- Plan 03 (05-03): Add `_ALLOW_HARDCODED_FALLBACK = False` + gate logic in `QueuePreparationService` and `DownloadOrchestrator` to satisfy the 4 TestFallbackGating RED tests (2 in each service test file)
- Entry-point infrastructure is complete; Plan 03 closes the loop by removing the hardcoded fallback path

## Self-Check: PASSED

- FOUND: pyproject.toml `[project.entry-points."lobster.queue_preparers"]` with 5 entries
- FOUND: pyproject.toml `[project.entry-points."lobster.download_services"]` with 5 entries
- FOUND: Runtime verification: all 5 databases discoverable via importlib.metadata
- FOUND: test_plugin_registration.py: 4/4 GREEN
- FOUND: commit 7313551 (feat(05-02): declare lobster.queue_preparers and lobster.download_services entry points)

---
*Phase: 05-plugin-first-registration*
*Completed: 2026-03-04*
