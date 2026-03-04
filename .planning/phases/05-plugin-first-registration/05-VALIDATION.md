---
phase: 5
slug: plugin-first-registration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0+ with pytest-mock |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/unit/services/data_access/test_queue_preparation_service.py tests/unit/services/data_access/test_download_services.py -x -q` |
| **Full suite command** | `pytest tests/unit && pytest tests/integration` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/services/data_access/ -x -q`
- **After every plan wave:** Run `pytest tests/unit && pytest tests/integration`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 0 | PLUG-01 | unit | `pytest tests/unit/services/data_access/test_queue_preparation_service.py -k "entry_point" -x` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 0 | PLUG-02 | unit | `pytest tests/unit/services/data_access/test_download_services.py -k "entry_point" -x` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 0 | PLUG-03/04 | contract | `pytest tests/unit/services/data_access/test_plugin_registration.py -x` | ❌ W0 | ⬜ pending |
| 5-01-04 | 01 | 0 | PLUG-06 | unit | `pytest tests/unit/services/data_access/test_queue_preparation_service.py::TestFallbackGating -x` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 1 | PLUG-03 | contract | `pytest tests/unit/services/data_access/test_plugin_registration.py::test_all_5_databases_discoverable -x` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 1 | PLUG-04 | contract | `pytest tests/unit/services/data_access/test_plugin_registration.py -x` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 2 | PLUG-06 | unit | `pytest tests/unit/services/data_access/test_queue_preparation_service.py::TestFallbackGating -x` | ❌ W0 | ⬜ pending |
| 5-03-02 | 03 | 2 | PLUG-06 | unit | `pytest tests/unit/services/data_access/test_download_services.py::TestFallbackGating -x` | ❌ W0 | ⬜ pending |
| 5-04-01 | 04 | 3 | PLUG-05 | unit | `pytest tests/unit/services/data_access/ -x -q` | ✅ (modify) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/services/data_access/test_plugin_registration.py` — new contract tests asserting entry-point declarations for PLUG-03, PLUG-04 (all 5 databases discoverable via `importlib.metadata.entry_points`)
- [ ] `tests/unit/services/data_access/test_queue_preparation_service.py` — add `TestEntryPointDiscovery` class (PLUG-01) + `TestFallbackGating` class (PLUG-06)
- [ ] `tests/unit/services/data_access/test_download_services.py` — add `TestEntryPointDiscovery` class (PLUG-02) + `TestFallbackGating` class

*Existing infrastructure covers pytest framework — no new installs required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Entry points visible after `uv pip install -e .` | PLUG-03 | Requires actual package reinstall (not easily mockable in CI) | Run: `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='lobster.queue_preparers')))"` — must show 5 entries |
| MetaboLights queue preparation works end-to-end | PLUG-04 | MetaboLights was previously missing from hardcoded fallback | Run a MetaboLights download queue preparation and confirm it succeeds |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
