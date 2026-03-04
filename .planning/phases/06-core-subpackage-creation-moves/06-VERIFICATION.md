---
phase: 06-core-subpackage-creation-moves
verified: 2026-03-04T18:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 06: Core Subpackage Creation & Moves Verification Report

**Phase Goal:** Create 5 core/ subpackages (governance, queues, runtime, notebooks, provenance) by moving existing files to their new canonical locations with backward-compatible shims at old import paths.
**Verified:** 2026-03-04
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | governance/, queues/, runtime/ subpackages exist under core/ with __init__.py files | VERIFIED | All 3 dirs exist with docstring-only `__init__.py`. Confirmed in filesystem. |
| 2 | notebooks/ and provenance/ subpackages exist under core/ with __init__.py files | VERIFIED | Both dirs exist; `provenance/__init__.py` contains `__getattr__` lazy shim. |
| 3 | Old import paths (lobster.core.download_queue, lobster.core.license_manager, etc.) still work via shims | VERIFIED | All 12 standard shims + 1 `__getattr__` shim exist. 31/31 shim tests pass. |
| 4 | Shims emit DeprecationWarning mentioning the new canonical path and v2.0.0 removal | VERIFIED | Every shim contains `_w.warn("...Shim will be removed in v2.0.0.", DeprecationWarning, stacklevel=2)`. Test suite confirms warnings fire. |
| 5 | isinstance identity is preserved — objects from old and new paths are the same object | VERIFIED | `test_isinstance_identity` parametrized across all 13 pairs: 13/13 pass. |
| 6 | Internal imports between moved files in the same subpackage use new canonical paths | VERIFIED | `download_queue.py` and `publication_queue.py` import from `lobster.core.queues.queue_storage`. `exporter.py` imports from `lobster.core.provenance.analysis_ir` and `lobster.core.provenance.provenance`. |
| 7 | `from lobster.core.provenance import ProvenanceTracker` works via __getattr__ lazy shim (no false warnings for new-style imports) | VERIFIED | `provenance/__init__.py` uses `__getattr__` — fires only on attribute access, not on direct submodule imports. Test confirms. |
| 8 | Notebook files use shortened names (executor.py not notebook_executor.py) | VERIFIED | `lobster/core/notebooks/executor.py`, `exporter.py`, `validator.py` — all renamed on move. |
| 9 | Import-linter config updated with new subpackage paths | VERIFIED | `.importlinter` contains `lobster.core.queues.download_queue` in `core-independence` contract and a new `core-subpackage-layers` contract. |
| 10 | No import cycles exist between the 5 core subpackages | VERIFIED | `core-subpackage-layers` contract enforces `notebooks > provenance > governance > queues > runtime`. `exporter.py` imports from `provenance` (higher-to-lower, permitted). |
| 11 | All 13 shim tests pass (from Plan 01 scaffold) | VERIFIED | `pytest tests/unit/core/test_core_subpackage_shims.py` — 31 passed (13 reexport + 13 isinstance identity + 5 subpackage existence). |
| 12 | Existing test suite passes via backward-compatible shims | VERIFIED | 1,603 passed, 12 skipped across all core tests (excluding 2 pre-existing failures unrelated to phase 06). |
| 13 | All 13 files in new canonical locations with shims at old paths | VERIFIED | 13 files moved; 12 standard shims + 1 `__getattr__` lazy shim confirmed at old paths. |

**Score:** 13/13 truths verified

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `lobster/core/governance/__init__.py` | Governance subpackage | VERIFIED | `"""Governance: licensing, AQUADIF monitoring."""` (docstring-only) |
| `lobster/core/queues/__init__.py` | Queues subpackage | VERIFIED | `"""Queue infrastructure for download and publication pipelines."""` (docstring-only) |
| `lobster/core/runtime/__init__.py` | Runtime subpackage | VERIFIED | `"""Runtime infrastructure: workspace resolution."""` (docstring-only) |
| `lobster/core/governance/license_manager.py` | Moved license_manager module | VERIFIED | Substantive — canonical module content |
| `lobster/core/governance/aquadif_monitor.py` | Moved aquadif_monitor module | VERIFIED | Substantive — canonical module content |
| `lobster/core/queues/download_queue.py` | Moved download_queue module | VERIFIED | Substantive — canonical module with updated internal imports |
| `lobster/core/queues/publication_queue.py` | Moved publication_queue module | VERIFIED | Substantive — canonical module with updated internal imports |
| `lobster/core/queues/queue_storage.py` | Moved queue_storage module | VERIFIED | Substantive — canonical module content |
| `lobster/core/runtime/workspace.py` | Moved workspace module | VERIFIED | Substantive — canonical module content |
| `tests/unit/core/test_core_subpackage_shims.py` | Shim validation tests for all 13 moves | VERIFIED | 31 tests (13 reexport + 13 isinstance identity + 5 subpackage existence), all pass |

### Plan 02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `lobster/core/notebooks/__init__.py` | Notebooks subpackage | VERIFIED | `"""Notebook infrastructure: execution, export, validation."""` (docstring-only) |
| `lobster/core/provenance/__init__.py` | Provenance subpackage with `__getattr__` lazy shim | VERIFIED | `__getattr__` fires on attribute access, imports from `provenance.provenance` with DeprecationWarning |
| `lobster/core/notebooks/executor.py` | Moved notebook_executor (renamed) | VERIFIED | Canonical file exists |
| `lobster/core/notebooks/exporter.py` | Moved notebook_exporter (renamed) | VERIFIED | Canonical file with updated internal imports |
| `lobster/core/notebooks/validator.py` | Moved notebook_validator (renamed) | VERIFIED | Canonical file exists |
| `lobster/core/provenance/analysis_ir.py` | Moved analysis_ir (125 importers) | VERIFIED | Canonical file exists |
| `lobster/core/provenance/provenance.py` | Moved provenance module | VERIFIED | Canonical file exists |
| `lobster/core/provenance/lineage.py` | Moved lineage module | VERIFIED | Canonical file exists |
| `lobster/core/provenance/ir_coverage.py` | Moved ir_coverage module | VERIFIED | Canonical file exists |
| `.importlinter` | Updated import-linter config with subpackage paths + layers contract | VERIFIED | `core-independence` uses `lobster.core.queues.download_queue`; new `core-subpackage-layers` contract added |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `lobster/core/download_queue.py` (shim) | `lobster/core/queues/download_queue.py` | `from lobster.core.queues.download_queue import *` | WIRED | Exact pattern present in shim file |
| `lobster/core/queues/download_queue.py` | `lobster/core/queues/queue_storage.py` | internal import uses new canonical path | WIRED | `from lobster.core.queues.queue_storage import (` at line 17 |
| `lobster/core/queues/publication_queue.py` | `lobster/core/queues/queue_storage.py` | internal import uses new canonical path | WIRED | `from lobster.core.queues.queue_storage import (` at line 17 |
| `lobster/core/provenance/__init__.py` | `lobster/core/provenance/provenance.py` | `__getattr__` lazy import | WIRED | `importlib.import_module("lobster.core.provenance.provenance")` inside `__getattr__` |
| `lobster/core/notebooks/exporter.py` | `lobster/core/provenance/analysis_ir.py` | updated internal import | WIRED | `from lobster.core.provenance.analysis_ir import (` at line 24 |
| `lobster/core/notebooks/exporter.py` | `lobster/core/provenance/provenance.py` | updated internal import | WIRED | `from lobster.core.provenance.provenance import ProvenanceTracker` at line 30 |
| `.importlinter` | `lobster/core/queues/download_queue` | updated independence contract path | WIRED | `lobster.core.queues.download_queue` present in `core-independence` contract |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CORE-01 | 06-01, 06-02 | Core subpackages created: runtime/, queues/, notebooks/, provenance/, governance/ | SATISFIED | All 5 subpackage directories with `__init__.py` files confirmed to exist and be importable |
| CORE-02 | 06-01, 06-02 | 13 files moved to domain subpackages with backward-compatible shims at old paths | SATISFIED | 13 canonical files in new locations; 12 standard shims + 1 `__getattr__` shim at old paths |
| CORE-03 | 06-01, 06-02 | Shims emit DeprecationWarning with removal version | SATISFIED | All shims contain `DeprecationWarning` with `"Shim will be removed in v2.0.0."` message |
| CORE-04 | 06-02 | Import-linter config updated for new subpackage paths | SATISFIED | `.importlinter` updated: canonical `queues.download_queue` path + new `core-subpackage-layers` contract |
| CORE-05 | 06-01, 06-02 | No import cycles introduced | SATISFIED | `core-subpackage-layers` enforces `notebooks > provenance > governance > queues > runtime`. Exporter only imports from lower layers. |

No orphaned requirements — all 5 requirements claimed by plans are satisfied and accounted for in REQUIREMENTS.md.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `lobster/core/governance/license_manager.py` | 166 | `# placeholder for future cryptographic verification` | Info | Comment describing a deferred feature — pre-existing in original module, not introduced by phase 06 |
| `lobster/core/provenance/analysis_ir.py` | 471–487 | `TODO` markers in `create_hollow_ir()` docstring | Info | By design — `create_hollow_ir()` is explicitly documented as a fallback that produces TODO-marked output, not a stub implementation |
| `lobster/core/notebooks/exporter.py` | 744–749 | `return placeholder` + `# TODO: Manual review needed` | Info | By design — this is the exporter's intentional placeholder generation for unsupported tool calls, pre-existing in original module |

No blockers or warnings — all flagged patterns are pre-existing design decisions in the moved modules, not introduced by this phase.

---

## Pre-Existing Test Failures (Not Caused by Phase 06)

Two test failures exist in the test suite that predate phase 06 and are unrelated to the restructuring:

1. **`test_notebook_executor.py::TestNotebookExecutor::test_execute_success`** — Fails with `ValueError: No language found in notebook and no override provided.` from the `papermill` library. This is a test fixture issue (missing `kernelspec` metadata in test notebook). The test file was last modified before phase 06 began (`73025f4` — Phase 04 completion). The shim continues to work (the test imports via `lobster.core.notebook_executor` and the shim successfully re-exports from the new canonical path).

2. **`test_data_manager_v2.py`** — Pre-existing `KeyError` on metadata keys, noted in the Phase 06-01 SUMMARY.

---

## Human Verification Required

None — all observable behaviors (file existence, shim wiring, import resolution, warning emission, isinstance identity, test passage) are fully verifiable programmatically. The shim tests cover all 13 moves with both re-export and identity checks.

---

## Summary

Phase 06 fully achieves its goal. All 5 core/ subpackages (governance, queues, runtime, notebooks, provenance) have been created and populated. All 13 files are in their canonical locations with backward-compatible shims at old paths. The critical provenance module/package name collision was resolved via `__getattr__` lazy shim. Internal imports between moved files were updated to canonical paths immediately. Import-linter was updated with both the corrected module path and a new layers contract enforcing subpackage dependency ordering.

The complete shim test suite (31 tests) passes 100%. 1,603 additional core tests pass with no regressions introduced by this phase.

**Commits:** `1cf1d61` (Plan 01 Task 1), `023bd16` (Plan 01 Task 2), `1cf1f54` (Plan 02 Task 1), `47b7898` (Plan 02 Task 2).

---

_Verified: 2026-03-04_
_Verifier: Claude (gsd-verifier)_
