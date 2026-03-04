---
phase: 8
slug: cli-decomposition
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-04
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.1 |
| **Config file** | `pytest.ini` |
| **Quick run command** | `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ -x --no-cov -q` |
| **Full suite command** | `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ tests/integration/test_session_provenance.py --no-cov -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/cli/ -x --no-cov -q`
- **After every plan wave:** Run `python -m pytest tests/unit/cli/ tests/integration/test_session_provenance.py --no-cov -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 0 | CLID-01, CLID-02 | unit | `python -m pytest tests/unit/cli/test_cli_decomposition.py -x --no-cov -q` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | CLID-01 | unit | `python -m pytest tests/unit/cli/ -x --no-cov -q` | ✅ | ⬜ pending |
| 08-02-01 | 02 | 2 | CLID-02 | unit | `python -m pytest tests/unit/cli/test_cli_decomposition.py::test_cli_is_wiring_only -x --no-cov -q` | ❌ W0 | ⬜ pending |
| 08-02-02 | 02 | 2 | CLID-03 | smoke | `lobster --help && lobster chat --help && lobster query --help && lobster init --help` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/cli/test_cli_decomposition.py` — structural tests: cli.py LOC < 500, no function bodies > 10 lines, all heavy modules importable
- [ ] Update patch paths in `tests/unit/cli/test_bug_fixes_security_logging.py` for new module locations

*Existing test infrastructure covers most phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| All CLI subcommands produce identical output | CLID-03 | Output comparison requires running actual CLI | Run `lobster --help`, `lobster chat --help`, `lobster query --help`, `lobster init --help` and verify identical to pre-decomposition output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
