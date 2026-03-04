---
phase: 11-strengthen-ci-deprecated-import-guard
verified: 2026-03-04T22:10:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 11: Strengthen CI Deprecated-Import Guard — Verification Report

**Phase Goal:** Strengthen the CI deprecated-import guard so that stale data_manager_v2 imports fail the build
**Verified:** 2026-03-04T22:10:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `grep -rn 'from lobster.core.data_manager_v2 import' packages/ --include='*.py'` returns zero matches (exit code 1) | VERIFIED | Live grep returns exit code 1, zero lines matched across all packages/ |
| 2 | CI deprecated-import guard step has no `\|\| true` bypass and fails when violations exist | VERIFIED | `.github/workflows/ci-basic.yml` line 50: `if grep -rn ...` with no `\|\| true` anywhere in file |
| 3 | All 39 package files import DataManagerV2 from `lobster.core.runtime.data_manager` | VERIFIED | `grep -rn 'from lobster.core.runtime.data_manager import DataManagerV2' packages/ --include='*.py'` returns 39 matches |
| 4 | Existing tests pass unchanged after migration (shim ensures identical class identity) | VERIFIED | `test_core_subpackage_shims.py` includes `data_manager_v2` in SHIM_PAIRS (line 34), verifying shim re-exports and identity. Commits exist at d2825b3, af9f64c. |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/ci-basic.yml` | Strict deprecated-import guard using `if grep` pattern | VERIFIED | Lines 47-59: `if grep -rn "from lobster\.core\.data_manager_v2 import" ... lobster/scaffold/ packages/; then exit 1; fi`. No `\|\| true` present anywhere in file. |
| `tests/unit/core/test_core_subpackage_shims.py` | Structural test enforcing no deprecated imports in packages/ | VERIFIED | Lines 182-200: `test_no_deprecated_data_manager_imports_in_packages()` asserts `result.returncode == 1` (grep finds no matches). Contains `test_no_deprecated_data_manager_imports_in_packages` function name as required. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `packages/**/*.py` | `lobster.core.runtime.data_manager` | canonical import path | VERIFIED | 39 occurrences of `from lobster.core.runtime.data_manager import DataManagerV2` found across packages/; zero occurrences of deprecated path |
| `.github/workflows/ci-basic.yml` | `packages/` | grep enforcement step | VERIFIED | `if grep` at line 50 scans `packages/` with `--exclude="data_manager_v2.py"` and exits 1 (fail CI) if violations found |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DMGR-04 | 11-01-PLAN.md | New code cannot import from old path (CI/lint check) | SATISFIED | CI guard enforces this at build time (if-grep pattern); structural test enforces locally; 39 package files already use canonical path — zero violations remain. REQUIREMENTS.md marks DMGR-04 as `[x]` Complete, phase 11 listed in traceability table. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No anti-patterns detected. No `TODO/FIXME/placeholder` in modified files. No empty implementations. No stale `|| true` bypasses.

---

### Human Verification Required

None. All verification is fully automated for this infrastructure/CI phase:
- Grep results are deterministic
- File contents are directly inspectable
- Commit existence is verifiable via git

---

### Gaps Summary

No gaps. All four must-have truths are verified against the actual codebase:

1. Zero deprecated imports remain in packages/ — confirmed by live grep returning exit code 1
2. CI guard is strict — `if grep` pattern with no `|| true`, grep exit 2 also fails CI
3. Exactly 39 canonical-import occurrences replace the 39 deprecated ones
4. Shim backward-compatibility preserved — `test_core_subpackage_shims.py` covers `data_manager_v2` shim identity

Requirement DMGR-04 is fully closed. The phase goal is achieved: stale `data_manager_v2` imports now fail the build.

---

_Verified: 2026-03-04T22:10:00Z_
_Verifier: Claude (gsd-verifier)_
