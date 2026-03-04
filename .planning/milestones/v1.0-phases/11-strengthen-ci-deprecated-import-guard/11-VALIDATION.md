---
phase: 11
slug: strengthen-ci-deprecated-import-guard
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -v` |
| **Full suite command** | `python -m pytest tests/unit/core/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/core/test_core_subpackage_shims.py -v`
- **After every plan wave:** Run `python -m pytest tests/unit/core/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | DMGR-04 | structural | `python -m pytest tests/unit/core/test_core_subpackage_shims.py::test_no_deprecated_data_manager_imports_in_packages -v` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | DMGR-04 | structural | `grep -rn "from lobster.core.data_manager_v2 import" packages/ --include="*.py"; echo $?` (exit 1 = pass) | n/a (CI) | ⬜ pending |
| 11-01-03 | 01 | 1 | DMGR-04 | manual/CI | Verify `.github/workflows/ci-basic.yml` deprecated-import step uses `if grep; then fail; fi` pattern | n/a (CI) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_core_subpackage_shims.py` — add `test_no_deprecated_data_manager_imports_in_packages` method (file exists, method is new)

*Existing infrastructure covers test framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CI step fails on violation | DMGR-04 | Requires GitHub Actions runner | Push a branch with a deprecated import, verify CI fails |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
