---
phase: 05-plugin-first-registration
plan: "01"
subsystem: testing
tags: [pytest, importlib.metadata, entry-points, plugin-registration, tdd]

# Dependency graph
requires:
  - phase: 04-geo-service-decomposition
    provides: Stable GEO service structure; queue preparer and download service classes ready to be wired as entry points
provides:
  - RED contract tests asserting lobster.queue_preparers + lobster.download_services entry-point declarations exist in pyproject.toml
  - TestEntryPointDiscovery classes in both service test files (mocked path — GREEN)
  - TestFallbackGating classes in both service test files (flag missing — RED)
affects:
  - 05-02 (plan 02 must satisfy test_plugin_registration.py RED tests)
  - 05-03 (plan 03 must satisfy TestFallbackGating RED tests)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Patch the singleton instance attribute (lobster.core.component_registry.component_registry.list_X) not the local import (module.component_registry.list_X) because both routers import component_registry lazily inside the method"
    - "TDD RED scaffold: contract tests use real importlib.metadata (no mocks) so they stay RED until pyproject.toml is updated and package is reinstalled"

key-files:
  created:
    - tests/unit/services/data_access/test_plugin_registration.py
  modified:
    - tests/unit/services/data_access/test_queue_preparation_service.py
    - tests/unit/services/data_access/test_download_services.py

key-decisions:
  - "Patch lobster.core.component_registry.component_registry.list_queue_preparers (singleton) not module.component_registry.list_queue_preparers (local import) — routers import component_registry inside the method body, not at module level"
  - "test_fallback_skipped_when_flag_false uses pytest.skip when _ALLOW_HARDCODED_FALLBACK missing rather than failing with AttributeError — test_fallback_flag_is_false_by_default already fails RED for that condition"
  - "TestEntryPointDiscovery tests are GREEN (mocks substitute for missing pyproject.toml declarations) — this is expected per plan spec"

patterns-established:
  - "Entry-point contract tests: use real importlib.metadata with no mocking for contract fidelity"
  - "Fallback gate tests: import module directly and check hasattr before asserting constant value"

requirements-completed:
  - PLUG-01
  - PLUG-02
  - PLUG-03
  - PLUG-04
  - PLUG-05
  - PLUG-06

# Metrics
duration: 7min
completed: 2026-03-04
---

# Phase 5 Plan 01: Plugin-First Registration RED Test Scaffolds Summary

**Three test files with failing RED contracts — TestPluginRegistrationContract (4 tests), TestFallbackGating x2 (2 tests each) — gating Plans 02 and 03 to declare pyproject.toml entry points and add _ALLOW_HARDCODED_FALLBACK constants**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-04T07:30:36Z
- **Completed:** 2026-03-04T07:38:00Z
- **Tasks:** 2
- **Files modified:** 3 (1 created, 2 appended)

## Accomplishments
- Created `test_plugin_registration.py` with `TestPluginRegistrationContract` — 4 real-importlib.metadata tests that fail RED because `pyproject.toml` has no `lobster.queue_preparers` or `lobster.download_services` entry-point declarations
- Appended `TestEntryPointDiscovery` to both service test files — 2 tests each, mocking the component_registry singleton, passing GREEN as designed
- Appended `TestFallbackGating` to both service test files — 2 tests each, failing RED because `_ALLOW_HARDCODED_FALLBACK` constant doesn't exist yet in either module
- Zero existing tests broken (642 passed, same 7 pre-existing network test failures unrelated to this work)

## Task Commits

Each task was committed atomically:

1. **Task 1: test_plugin_registration.py RED contract tests** - `290ecfe` (test)
2. **Task 2: TestEntryPointDiscovery + TestFallbackGating in both files** - `824e1c3` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/unit/services/data_access/test_plugin_registration.py` — New contract tests using real importlib.metadata; 4 tests all fail RED (no entry-point declarations yet)
- `tests/unit/services/data_access/test_queue_preparation_service.py` — TestEntryPointDiscovery (GREEN via mocks) + TestFallbackGating (RED — constant missing)
- `tests/unit/services/data_access/test_download_services.py` — Same pattern; also added `patch` to existing import line

## Decisions Made
- **Patch the singleton instance:** Both `QueuePreparationService._register_default_preparers` and `DownloadOrchestrator._register_default_services` import `component_registry` inside the method via `from lobster.core.component_registry import component_registry`, making it a local variable only. Patching the singleton object's method directly (`lobster.core.component_registry.component_registry.list_queue_preparers`) is the only reliable intercept point.
- **pytest.skip for second fallback test:** `test_fallback_skipped_when_flag_false` skips when `_ALLOW_HARDCODED_FALLBACK` is absent — avoids misleading AttributeError when the primary test (`test_fallback_flag_is_false_by_default`) already covers that RED condition clearly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect patch paths for component_registry**
- **Found during:** Task 2 (TestEntryPointDiscovery verification)
- **Issue:** Plan spec showed patch path as `lobster.services.data_access.queue_preparation_service.component_registry.list_queue_preparers` — but component_registry is NOT a module-level attribute in either service file (it's imported inside the method body). Patch failed with `AttributeError: module has no attribute 'component_registry'`.
- **Fix:** Changed all patch paths to `lobster.core.component_registry.component_registry.list_queue_preparers` (patching the singleton instance's method directly)
- **Files modified:** test_queue_preparation_service.py, test_download_services.py
- **Verification:** TestEntryPointDiscovery — 4 passed GREEN after fix
- **Committed in:** 824e1c3 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in plan's assumed patch path)
**Impact on plan:** Necessary fix to make TestEntryPointDiscovery pass as designed. No scope creep.

## Issues Encountered
- Component_registry lazy import pattern (inside method body) required adjusting patch paths from module-level attribute to singleton instance method. Caught immediately on first test run.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 02 (05-02): Add entry-point declarations to pyproject.toml → satisfies TestPluginRegistrationContract (4 RED → GREEN)
- Plan 03 (05-03): Add `_ALLOW_HARDCODED_FALLBACK = False` + gate logic → satisfies TestFallbackGating (2×2 RED → GREEN)
- All wave-0 test scaffolds are in place; Plans 02 and 03 can now execute with clear unambiguous pass/fail signals

## Self-Check: PASSED

- FOUND: tests/unit/services/data_access/test_plugin_registration.py
- FOUND: .planning/phases/05-plugin-first-registration/05-01-SUMMARY.md
- FOUND: commit 290ecfe (test_plugin_registration.py RED contract tests)
- FOUND: commit 824e1c3 (TestEntryPointDiscovery + TestFallbackGating)

---
*Phase: 05-plugin-first-registration*
*Completed: 2026-03-04*
