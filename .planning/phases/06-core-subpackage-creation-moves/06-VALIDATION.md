---
phase: 6
slug: core-subpackage-creation-moves
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/unit/core/ -x -q` |
| **Full suite command** | `pytest tests/unit/ tests/integration/ -x` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/core/ -x -q`
- **After every plan wave:** Run `pytest tests/unit/ tests/integration/ -x && lint-imports`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 0 | CORE-01, CORE-02, CORE-03 | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | CORE-01, CORE-02 | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | CORE-01, CORE-02, CORE-03 | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -x` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 2 | CORE-04 | smoke | `lint-imports` | ✅ | ⬜ pending |
| 06-02-02 | 02 | 2 | CORE-05 | smoke | `lint-imports` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_core_subpackage_shims.py` — shim validation + isinstance identity tests for CORE-01, CORE-02, CORE-03
- [ ] Updated `.importlinter` config — covers CORE-04 (new subpackage paths in independence/layers contract)

*Existing infrastructure covers import-linter and pytest.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
