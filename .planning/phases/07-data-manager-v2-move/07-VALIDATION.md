---
phase: 7
slug: data-manager-v2-move
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/unit/core/test_core_subpackage_shims.py tests/unit/core/test_data_manager_v2.py -x -q` |
| **Full suite command** | `pytest tests/ -x --timeout=60` |
| **Estimated runtime** | ~30 seconds (quick), ~120 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/core/test_core_subpackage_shims.py tests/unit/core/test_data_manager_v2.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x --timeout=60`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 0 | DMGR-01, DMGR-02 | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -k data_manager -x` | Partial (extend SHIM_PAIRS) | ⬜ pending |
| 07-01-02 | 01 | 0 | DMGR-02 | unit | `pytest tests/unit/core/test_data_manager_v2.py -x` | ✅ (needs mock.patch updates) | ⬜ pending |
| 07-01-03 | 01 | 1 | DMGR-01 | unit | `python -c "from lobster.core.runtime.data_manager import DataManagerV2"` | ❌ W0 | ⬜ pending |
| 07-01-04 | 01 | 1 | DMGR-02 | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -k data_manager -x` | Partial | ⬜ pending |
| 07-01-05 | 01 | 2 | DMGR-03 | unit | `grep -q "lobster.core.runtime.data_manager" lobster/scaffold/templates/agent.py.j2` | ❌ W0 | ⬜ pending |
| 07-01-06 | 01 | 2 | DMGR-04 | integration | CI lint check | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Extend `SHIM_PAIRS` in `tests/unit/core/test_core_subpackage_shims.py` with data_manager_v2 pair
- [ ] Update mock.patch strings in `tests/unit/core/test_data_manager_v2.py` (108 references)
- [ ] Update mock.patch strings in `tests/unit/core/test_agent_registry.py` (2 references)
- [ ] Add CI check step for old-path import prevention (DMGR-04)

*Existing test infrastructure covers DMGR-01 and DMGR-02 partially — extensions needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Scaffold generates new-path imports | DMGR-03 | Template output inspection | Run `lobster scaffold agent --name test_agent` and verify imports use `lobster.core.runtime.data_manager` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
