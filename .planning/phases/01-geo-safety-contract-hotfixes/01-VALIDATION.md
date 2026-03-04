---
phase: 1
slug: geo-safety-contract-hotfixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0+ with pytest-cov, pytest-timeout |
| **Config file** | `pytest.ini` (root) + `[tool.pytest.ini_options]` in pyproject.toml |
| **Quick run command** | `pytest tests/unit/ -x --no-cov -q --timeout=60` |
| **Full suite command** | `pytest tests/ --timeout=300 -q` |
| **Estimated runtime** | ~30 seconds (unit only), ~120 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/ -x --no-cov -q --timeout=60`
- **After every plan wave:** Run `pytest tests/ --timeout=300 -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | GSAF-01 | unit | `pytest tests/unit/services/data_access/test_geo_queue_preparer.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | GSAF-02 | unit | `pytest tests/unit/core/test_metadata_key_consistency.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | GSAF-03 | unit | `pytest tests/unit/core/test_store_geo_metadata.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | GSAF-04 | unit | `pytest tests/unit/core/test_archive_utils.py -x --no-cov` | ✅ (needs extension) | ⬜ pending |
| 01-01-05 | 01 | 1 | GSAF-05 | unit | `pytest tests/unit/services/data_access/test_geo_retry_types.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 01-01-06 | 01 | 1 | GSAF-06 | unit | `pytest tests/unit/core/test_download_queue.py -x --no-cov` | ✅ (needs CAS tests) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/services/data_access/test_geo_queue_preparer.py` — stubs for GSAF-01 (GDS canonicalization)
- [ ] `tests/unit/core/test_metadata_key_consistency.py` — stubs for GSAF-02 (key standardization)
- [ ] `tests/unit/core/test_store_geo_metadata.py` — stubs for GSAF-03 (write helper enforcement)
- [ ] `tests/unit/services/data_access/test_geo_retry_types.py` — stubs for GSAF-05 (typed retry results)
- [ ] Extend `tests/unit/core/test_archive_utils.py` with path-traversal rejection + symlink tests (GSAF-04)
- [ ] Extend `tests/unit/core/test_download_queue.py` with CAS update_status tests (GSAF-06)

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
