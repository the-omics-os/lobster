---
phase: 10
slug: fix-provenance-getattr-shim
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest] |
| **Quick run command** | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q` |
| **Full suite command** | `python -m pytest tests/unit/core/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/unit/core/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | CORE-04 | unit | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q` | Partial | ⬜ pending |
| 10-01-02 | 01 | 1 | CORE-04 | unit | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 1 | CORE-04 | smoke | `python -c "from lobster.core.provenance import AnalysisStep; print('OK')"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Add AnalysisStep, ParameterSpec, LineageMetadata, IRCoverageAnalyzer test cases to `tests/unit/core/test_core_subpackage_shims.py`
- [ ] Smoke test: `python -c "from lobster.core.provenance import AnalysisStep"` passes

*Existing infrastructure covers ProvenanceTracker shim testing.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
