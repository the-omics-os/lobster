---
phase: 4
slug: geo-service-decomposition
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.1 |
| **Config file** | `pytest.ini` |
| **Quick run command** | `source .venv/bin/activate && python -m pytest tests/unit/services/data_access/test_geo_*.py -x -q` |
| **Full suite command** | `source .venv/bin/activate && python -m pytest tests/unit/services/data_access/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/services/data_access/test_geo_*.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/unit/services/data_access/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 0 | GDEC-01 | structural | `pytest tests/unit/services/data_access/test_geo_decomposition.py -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 0 | GDEC-02 | unit + integration | `pytest tests/unit/services/data_access/test_geo_facade_compat.py -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 0 | GDEC-03 | unit + AST scan | `pytest tests/unit/services/data_access/test_soft_download.py -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 0 | GDEC-04 | unit | `pytest tests/unit/services/data_access/test_geo_metadata_fetch.py -x` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 0 | GDEC-04 | unit | `pytest tests/unit/services/data_access/test_geo_download_execution.py -x` | ❌ W0 | ⬜ pending |
| 04-01-06 | 01 | 0 | GDEC-04 | unit | `pytest tests/unit/services/data_access/test_geo_archive_processing.py -x` | ❌ W0 | ⬜ pending |
| 04-01-07 | 01 | 0 | GDEC-04 | unit | `pytest tests/unit/services/data_access/test_geo_matrix_parsing.py -x` | ❌ W0 | ⬜ pending |
| 04-01-08 | 01 | 0 | GDEC-04 | unit | `pytest tests/unit/services/data_access/test_geo_concatenation.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/services/data_access/test_geo_decomposition.py` — structural tests verifying 5 modules exist (GDEC-01)
- [ ] `tests/unit/services/data_access/test_geo_facade_compat.py` — facade backward compatibility (GDEC-02)
- [ ] `tests/unit/services/data_access/test_soft_download.py` — SOFT deduplication verification (GDEC-03)
- [ ] `tests/unit/services/data_access/test_geo_metadata_fetch.py` — metadata_fetch module tests (GDEC-04)
- [ ] `tests/unit/services/data_access/test_geo_download_execution.py` — download_execution module tests (GDEC-04)
- [ ] `tests/unit/services/data_access/test_geo_archive_processing.py` — archive_processing module tests (GDEC-04)
- [ ] `tests/unit/services/data_access/test_geo_matrix_parsing.py` — matrix_parsing module tests (GDEC-04)
- [ ] `tests/unit/services/data_access/test_geo_concatenation.py` — concatenation module tests (GDEC-04)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Existing integration tests pass unchanged | GDEC-02 | Full integration test suite requires network/fixtures | Run `pytest tests/integration/test_geo_*.py -x -q` on dev machine |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
