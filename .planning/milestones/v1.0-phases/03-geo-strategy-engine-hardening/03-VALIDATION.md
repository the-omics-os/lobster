---
phase: 3
slug: geo-strategy-engine-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/unit/services/data_access/test_geo_strategy.py -x --no-cov` |
| **Full suite command** | `pytest tests/unit/ -x --no-cov -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/services/data_access/test_geo_strategy.py -x --no-cov`
- **After every plan wave:** Run `pytest tests/unit/ -x --no-cov -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | GSTR-01 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestNullSanitization -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | GSTR-01 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestPipelineContext -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | GSTR-01 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestDeriveAnalysis -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | GSTR-02 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestRulesWithNulls -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | GSTR-02 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestDataAvailability -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-06 | 01 | 1 | GSTR-02 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestPipelineStepNullGuards -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-07 | 01 | 2 | GSTR-03 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestArchiveFirstRemoved -x --no-cov` | ❌ W0 | ⬜ pending |
| 03-01-08 | 01 | 2 | GSTR-03 | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestNoDeadBranches -x --no-cov` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/services/data_access/test_geo_strategy.py` — stubs for GSTR-01, GSTR-02, GSTR-03
- [ ] No framework install needed — pytest already configured

*Existing infrastructure covers framework requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
