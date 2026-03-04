---
phase: 2
slug: geo-parser-data-integrity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-03
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0+ |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/unit/services/data_access/test_geo_parser_partial.py tests/unit/services/data_access/test_geo_archive_classifier.py tests/unit/services/data_access/test_geo_file_scoring.py tests/unit/services/data_access/test_geo_temp_cleanup.py -x --no-cov -q` |
| **Full suite command** | `pytest tests/unit/ -x --no-cov -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 0 | GPAR-01 | unit | `pytest tests/unit/services/data_access/test_geo_parser_partial.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | GPAR-01 | unit | `pytest tests/unit/services/data_access/test_geo_parser_partial.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | GPAR-02 | unit | `pytest tests/unit/services/data_access/test_geo_service_partial_handling.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 0 | GPAR-03 | unit | `pytest tests/unit/services/data_access/test_geo_archive_classifier.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | GPAR-03 | unit | `pytest tests/unit/services/data_access/test_geo_archive_classifier.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 1 | GPAR-04 | unit | `pytest tests/unit/services/data_access/test_geo_file_scoring.py -x --no-cov` | ❌ W0 | ⬜ pending |
| 02-02-04 | 02 | 1 | GPAR-05 | unit | `pytest tests/unit/services/data_access/test_geo_temp_cleanup.py -x --no-cov` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/services/data_access/test_geo_parser_partial.py` — stubs for GPAR-01 (ParseResult, chunk parser partial signaling)
- [ ] `tests/unit/services/data_access/test_geo_service_partial_handling.py` — stubs for GPAR-02 (call site partial handling)
- [ ] `tests/unit/services/data_access/test_geo_archive_classifier.py` — stubs for GPAR-03 (archive format detection)
- [ ] `tests/unit/services/data_access/test_geo_file_scoring.py` — stubs for GPAR-04 (expression file scoring heuristic)
- [ ] `tests/unit/services/data_access/test_geo_temp_cleanup.py` — stubs for GPAR-05 (temp file cleanup on failure)
- [ ] `tests/unit/services/data_access/conftest.py` — shared fixtures (mock GEOParser, temp dirs, sample filenames)

*Existing infrastructure:* `pytest.ini` and `tests/unit/core/test_archive_utils.py` already exist.

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
