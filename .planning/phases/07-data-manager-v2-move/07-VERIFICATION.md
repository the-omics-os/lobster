---
phase: 07-data-manager-v2-move
verified: 2026-03-04T18:30:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 7: data_manager_v2 Move — Verification Report

**Phase Goal:** Move data_manager_v2.py into core/runtime/ subpackage with backward-compatible shim and enforce canonical imports
**Verified:** 2026-03-04T18:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `from lobster.core.runtime.data_manager import DataManagerV2` works as canonical import | VERIFIED | Live import confirmed: `<class 'lobster.core.runtime.data_manager.DataManagerV2'>` |
| 2 | `from lobster.core.data_manager_v2 import DataManagerV2` works via shim with DeprecationWarning | VERIFIED | Warning message confirmed: "Import from 'lobster.core.runtime.data_manager' instead of 'lobster.core.data_manager_v2'" |
| 3 | All existing test_data_manager_v2.py tests pass with updated mock.patch paths | VERIFIED | 96 passed, 5 skipped, 2 pre-existing KeyError failures unchanged (TestExportDocumentation, TestGEOMetadataStorage) |
| 4 | Shim isinstance identity holds (old path object is new path object) | VERIFIED | `OldDM is NewDM` returns `True` |
| 5 | Scaffold-generated agent code imports DataManagerV2 from canonical path | VERIFIED | Both `.j2` templates contain `from lobster.core.runtime.data_manager import DataManagerV2` |
| 6 | CI blocks PRs that add new imports from the deprecated data_manager_v2 path | VERIFIED | Step "Check for deprecated data_manager_v2 imports" present in ci-basic.yml quality-and-tests job |
| 7 | Import-linter independence contract references canonical module path | VERIFIED | `.importlinter` line 50: `lobster.core.runtime.data_manager` |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `lobster/core/runtime/data_manager.py` | DataManagerV2 class at canonical location (min 3900 lines) | VERIFIED | 3999 lines, contains full DataManagerV2 implementation |
| `lobster/core/data_manager_v2.py` | Backward-compat shim with DeprecationWarning re-exporting via wildcard | VERIFIED | Contains exact Phase 6 shim pattern: `warnings.warn` + `from lobster.core.runtime.data_manager import *` |
| `lobster/core/runtime/__init__.py` | Updated docstring mentioning data manager | VERIFIED | `"""Runtime infrastructure: workspace resolution, data manager."""` |
| `tests/unit/core/test_core_subpackage_shims.py` | Extended SHIM_PAIRS with data_manager_v2 entry (14 total) | VERIFIED | 14 tuples in SHIM_PAIRS; module docstring mentions "14 shim moves"; data_manager_v2 entry confirmed at line 34 |
| `lobster/scaffold/templates/agent.py.j2` | Scaffold template with canonical import | VERIFIED | `from lobster.core.runtime.data_manager import DataManagerV2` present |
| `lobster/scaffold/templates/shared_tools.py.j2` | Scaffold template with canonical import | VERIFIED | `from lobster.core.runtime.data_manager import DataManagerV2` present |
| `.importlinter` | Updated independence contract with canonical path | VERIFIED | Line 50: `lobster.core.runtime.data_manager` (no `data_manager_v2` in contract) |
| `.github/workflows/ci-basic.yml` | CI check step for deprecated imports | VERIFIED | Step "Check for deprecated data_manager_v2 imports" greps scaffold/ and packages/ |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `lobster/core/data_manager_v2.py` | `lobster/core/runtime/data_manager.py` | wildcard re-export with warnings.warn | WIRED | `from lobster.core.runtime.data_manager import *` confirmed in shim |
| `tests/unit/core/test_data_manager_v2.py` | `lobster/core/runtime/data_manager` | mock.patch strings targeting canonical module | WIRED | 114 mock.patch strings use `lobster.core.runtime.data_manager.X`; zero strings use old `lobster.core.data_manager_v2.X` |
| `tests/unit/agents/test_agent_registry.py` | `lobster/core/runtime/data_manager` | mock.patch strings targeting canonical module | WIRED | Zero old-path patch strings remain |
| `lobster/scaffold/templates/agent.py.j2` | `lobster/core/runtime/data_manager.py` | Jinja2 template import statement | WIRED | Canonical import confirmed in template |
| `.github/workflows/ci-basic.yml` | `lobster/core/data_manager_v2.py` | grep check that blocks new deprecated-path imports | WIRED | CI step targets `from lobster.core.data_manager_v2 import`, excludes shim file itself |
| `lobster/core/runtime/data_manager.py` (internal) | canonical core subpaths | Updated internal imports | WIRED | `lobster.core.provenance.analysis_ir`, `lobster.core.provenance.provenance`, `lobster.core.queues.queue_storage`, `lobster.core.runtime.workspace` — no shim-path internal imports |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DMGR-01 | 07-01-PLAN.md | data_manager_v2.py moved to core/runtime/data_manager.py | SATISFIED | File exists at canonical path with 3999 lines; git mv used to preserve history |
| DMGR-02 | 07-01-PLAN.md | Shim at old path re-exports everything — zero breakage for 79 source + 118 test importers | SATISFIED | Shim confirmed: wildcard re-export + DeprecationWarning; isinstance identity verified |
| DMGR-03 | 07-02-PLAN.md | Scaffold templates updated to import from new path | SATISFIED | Both agent.py.j2 and shared_tools.py.j2 use canonical path |
| DMGR-04 | 07-02-PLAN.md | New code cannot import from old path (CI/lint check) | SATISFIED | CI step present in ci-basic.yml, greps scaffold/ and packages/ directories |

No orphaned requirements — all four DMGR IDs declared in plan frontmatter are fully accounted for.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/unit/core/test_core_subpackage_shims.py` | 18 | Comment says "All 13 shim pairs" but list has 14 entries | Info | Stale inline comment; module docstring at top correctly says "14 shim moves" — no functional impact |
| `lobster/scaffold/templates/shared_tools.py.j2` | 65, 77, 115, 124, 163, 172 | `# TODO: Implement ...` markers | Info | Intentional scaffold guidance for new agent authors; expected template pattern, not a stub |

No blockers or warnings found. Both findings are informational only.

---

### Human Verification Required

None. All phase goals are mechanically verifiable and have been verified programmatically.

---

### Gaps Summary

No gaps. All seven observable truths are verified and all four DMGR requirements are satisfied.

**Key findings:**

- `lobster/core/runtime/data_manager.py` (3999 lines) exists at the canonical path and is importable
- `lobster/core/data_manager_v2.py` shim emits a correct DeprecationWarning and re-exports DataManagerV2 — isinstance identity confirmed (`OldDM is NewDM` is `True`)
- 114 mock.patch strings in `tests/unit/core/test_data_manager_v2.py` correctly target `lobster.core.runtime.data_manager.X` — zero old-path strings remain
- Internal imports inside the moved file use canonical subpaths (provenance/, queues/, runtime/) — no internal shim-chain dependencies
- Both scaffold Jinja2 templates generate code with the canonical import path
- The CI deprecated-import guard is correctly scoped to `lobster/scaffold/` and `packages/` directories
- 2 pre-existing test failures (KeyError on 'validation' key) are confirmed as pre-dating Phase 7 and documented in the SUMMARY

---

_Verified: 2026-03-04T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
