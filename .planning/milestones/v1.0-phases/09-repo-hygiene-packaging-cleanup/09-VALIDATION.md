---
phase: 9
slug: repo-hygiene-packaging-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/unit/ -x --no-cov -q` |
| **Full suite command** | `pytest tests/ -v --no-cov` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/ -x --no-cov -q`
- **After every plan wave:** Run `pytest tests/ -v --no-cov`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | HYGN-01 | manual-only | Visual inspection of .gitignore sections | N/A | ⬜ pending |
| 09-01-02 | 01 | 1 | HYGN-02 | smoke | `make clean && find packages -name dist -o -name '*.egg-info' -o -name '.ruff_cache' \| wc -l` | N/A | ⬜ pending |
| 09-01-03 | 01 | 1 | HYGN-03 | smoke | `find lobster tests -type d -empty -not -path '*__pycache__*' \| wc -l` | N/A | ⬜ pending |
| 09-01-04 | 01 | 1 | HYGN-04 | unit | `pytest tests/unit/tools/test_geo_downloader.py -x --no-cov` | ✅ | ⬜ pending |
| 09-01-05 | 01 | 1 | HYGN-05 | smoke | `find packages -name dist -o -name '*.egg-info' -o -name '.ruff_cache' \| wc -l` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. HYGN-04 import update verification uses the existing `tests/unit/tools/test_geo_downloader.py`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| .gitignore has ~10 logical sections with clear headers | HYGN-01 | Structural/stylistic — no programmatic assertion | Inspect .gitignore: verify grouped sections (Python, Build, IDE, Testing, Environment, Lobster, Packages, Data, OS, Negation) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
