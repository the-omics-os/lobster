# Project State: Session-Scoped Provenance Persistence

**Last Updated:** 2026-02-15
**Project:** Session-Scoped Provenance Persistence
**Schema Version:** 1.0

---

## Project Reference

**Core Value:** Provenance must survive process exit. If the user ran an analysis, they must be able to export it as a Jupyter notebook — from any session, after any restart.

**Current Focus:** Phase 5 — Unit Tests

**What We're Building:** Disk-persistent provenance scoped to sessions, transforming Lobster AI's reproducibility from single-process to cross-session, crash-resilient, and cloud-ready.

---

## Current Position

**Phase:** 5 of 7 — Unit Test Coverage
**Plan:** 1 of 1 (complete)
**Status:** Phase complete

**Progress:**
```
[███████████████████████████████---] 71%
Phase 1-5 complete, ready for Phase 6
```

**Completed:**
- Phase 1: Core Persistence Layer (10/10 requirements verified)
- Phase 2: DataManagerV2 Integration (Plan 1: session_dir wiring)
- Phase 3: Session JSON Schema & Continuity (Plan 1: schema 1.1, provenance restoration)
- Phase 4: CLI Session Support (Plan 1: CommandClient + --session-id option)
- Phase 5: Unit Test Coverage (Plan 1: 13 tests, 96.4% method coverage)

**Next Actions:**
1. Plan Phase 6 — Integration tests for end-to-end session persistence
2. Write 3 integration tests covering DataManagerV2 → disk workflows
3. Verify multi-session scenarios and cloud restore patterns

---

## Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Requirements Coverage | 100% | 100% | In Progress |
| Phases Defined | 7 | 7 | Complete |
| Phases Complete | 7 | 5 | In Progress |
| Unit Tests | 12 | 13 | Complete |
| Integration Tests | 3 | 0 | Pending (Phase 6) |
| Documentation Updates | 1 | 0 | Pending (Phase 7) |

---

## Accumulated Context

### Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-15 | True append (not atomic_write_jsonl) | O(1) per activity vs O(n), crash-safe with at most one corrupt trailing line |
| 2026-02-15 | JSONL with `"v": 1` per line | Enables forward-compatible schema migration across cloud ECS deploys |
| 2026-02-15 | Session dir inside workspace (`.lobster/sessions/{id}/`) | Keeps provenance co-located with workspace data, discoverable by path convention |
| 2026-02-15 | `mkdir` in ProvenanceTracker (not caller) | Cloud restore recreates files but not empty dirs — ProvenanceTracker must be self-contained |
| 2026-02-15 | `provenance_path` property (not hardcoded filename) | Cloud S3 sync needs the path without knowing internal naming |
| 2026-02-15 | Schema version bump to "1.1" in session JSON | Old sessions (1.0) don't have provenance key — loading code can distinguish |
| 2026-02-15 | IR restored as AnalysisStep on load | Not dicts - enables immediate notebook export without re-parsing |
| 2026-02-15 | fsync after each write | Crash loses at most one trailing activity |
| 2026-02-15 | session_dir as _session_dir private attribute | Prevents accidental modification, signals internal use |
| 2026-02-15 | Directory creation in AgentClient | Explicit control over when directories are created |
| 2026-02-15 | External data_manager bypasses session_dir wiring | Backward compatibility for users providing their own DataManagerV2 |
| 2026-02-15 | Provenance restoration is best-effort | Session loads even if provenance fails - logged as warning but not fatal |
| 2026-02-15 | Default session resolution: None and "latest" both resolve to most recent session | Enables immediate use without remembering session IDs |
| 2026-02-15 | Plain text error messages for CLI | Scripting compatibility (JSON output mode) |
| 2026-02-15 | Session display with timestamp | "Using session: {id} (last activity: {timestamp})" confirms selection |
| 2026-02-15 | Two test classes for unit tests | TestPersistenceMechanics (core I/O) + TestApiContracts (API surface) for logical grouping |
| 2026-02-15 | pytest tmp_path over tmpdir | Modern fixture, automatic cleanup, Path objects |
| 2026-02-15 | Synthetic IR fixture with all fields | Complete round-trip validation without service dependencies |

### Implementation Notes

**Files modified (Phase 1):**
- `lobster/core/provenance.py` — Core persistence layer complete

**Files modified (Phase 2):**
- `lobster/core/data_manager_v2.py` — session_dir parameter forwarding to ProvenanceTracker
- `lobster/core/client.py` — Session directory computation and wiring

**Files modified (Phase 3):**
- `lobster/core/provenance.py` — Added path parameter to _load_from_disk()
- `lobster/core/client.py` — Schema 1.1, provenance key in save, restoration in load

**Files modified (Phase 4):**
- `lobster/cli.py` — CommandClient session_id parameter, --session-id CLI option, provenance check in pipeline dispatch

**Files created (Phase 5):**
- `tests/unit/core/test_provenance_persistence.py` — 13 unit tests (12 required + 1 bonus)

**Files to modify (remaining):**
- `skills/lobster-use/references/cli-commands.md` — Documentation (Phase 7)

**New files to create:**
- `tests/integration/test_session_provenance.py` — 3 integration tests (Phase 6)

**Architecture patterns established:**
- Append-only JSONL with O(1) writes using `open("a")` + `fsync`
- Line-independent recovery (corrupt lines skipped with warnings)
- Schema versioning at JSONL line level (`"v": 1`)
- Session JSON schema versioning (1.0 -> 1.1)
- Path parameter pattern for cross-session file access
- Backward compatibility via `None` defaults (zero breakage)
- Self-contained directory creation (ProvenanceTracker handles mkdir)
- Session directory path: `workspace/.lobster/sessions/{session_id}/`
- Parameter wiring: AgentClient -> DataManagerV2 -> ProvenanceTracker
- Session resolution: sort by mtime, use most recent
- Provenance check: `getattr(client.data_manager, 'provenance', None)` + activities check

### Phase 1 Deliverables

- `_persist_activity()` — Appends to JSONL with fsync
- `_load_from_disk()` — Restores activities with IR reconstruction
- `_write_session_metadata()` — Companion metadata.json
- `provenance_path` property — Exposes file path for external consumers
- `get_summary()` — Compact stats for API responses

### Phase 2 Deliverables

- DataManagerV2 accepts `session_dir` parameter
- DataManagerV2 forwards `session_dir` to ProvenanceTracker
- DataManagerV2 preserves `session_dir` across `clear()` operations
- AgentClient computes session directory: `workspace/.lobster/sessions/{session_id}/`
- AgentClient creates session directory for internal data managers
- External data_manager case bypasses session_dir wiring

### Phase 3 Deliverables

- `_load_from_disk(path)` accepts optional path parameter for cross-session restoration
- Session JSON schema bumped to "1.1"
- Session JSON includes `provenance` key with session_dir and activity_count
- `load_session()` restores provenance from original session directory
- `load_session()` returns `provenance_restored` count in result dict
- Backward compatibility: schema 1.0 sessions load without error

### Phase 4 Deliverables

- CommandClient accepts `session_id` parameter for loading provenance from disk
- CLI `--session-id/-s` option for `lobster command`
- Auto-resolution to latest session by mtime when no session specified
- Pipeline export/run check provenance availability instead of hardcoded rejection
- Available sessions listed in error output

### Phase 5 Deliverables

- `test_provenance_persistence.py` — 13 unit tests for persistence layer
- 96.4% average coverage for persistence methods (_persist_activity: 100%, provenance_path: 100%, get_summary: 100%, _write_session_metadata: 95.2%, _load_from_disk: 87%)
- Test runtime: 0.20s (10x under 5s target)
- No regressions: 66 tests pass together (53 existing + 13 new)
- Fixtures: `session_dir` (tmp_path wrapper), `synthetic_ir` (complete AnalysisStep)
- Test patterns: warnings.catch_warnings for corrupt line validation, isinstance checks for IR reconstruction
- Edge cases covered: corrupt files, empty files, deleted directories, IR round-trip, metadata companion

### Active TODOs

- [x] Implement ProvenanceTracker._persist_activity()
- [x] Implement ProvenanceTracker._load_from_disk()
- [x] Implement ProvenanceTracker._write_session_metadata()
- [x] Add provenance_path property
- [x] Add get_summary() method
- [x] Wire session_dir through DataManagerV2
- [x] Wire session_dir through AgentClient
- [x] Update session JSON schema to 1.1
- [x] Implement load_session() provenance restoration
- [x] Add CommandClient session_id parameter
- [x] Add `lobster command --session-id` CLI option
- [x] Unblock pipeline export in CommandClient
- [x] Write 13 unit tests (12 required + 1 bonus)
- [ ] Write 3 integration tests
- [ ] Update CLI commands documentation

### Known Blockers

None — Phase 5 complete, ready for Phase 6.

---

## Session Continuity

**Session ID:** Not applicable (planning phase)
**Workspace:** `/Users/tyo/omics-os/lobster/.planning/`
**Config:** depth=standard, mode=yolo, parallelization=true

**Last Session:** 2026-02-15T11:32:54Z
**Stopped at:** Completed 05-01-PLAN.md
**Resume file:** None

**Quick Resume:**
```bash
# Review Phase 5 summary
cat /Users/tyo/omics-os/lobster/.planning/phases/05-unit-test-coverage/05-01-SUMMARY.md

# Plan Phase 6
/gsd:plan-phase 6
```

**Context Files:**
- PROJECT.md — What we're building, constraints, key decisions
- REQUIREMENTS.md — 38 requirements with traceability (23/38 complete)
- ROADMAP.md — 7 phases with success criteria (4/7 complete)
- config.json — Workflow settings

---

## Quality Gates

**Phase 1 (Complete):**
- [x] All Phase 1 success criteria met
- [x] 10/10 requirements verified against codebase
- [x] Backward compatibility validated (session_dir=None is no-op)
- [x] No pyproject.toml changes (stdlib only)
- [x] Crash-safe append-only writes verified

**Phase 2 (Complete):**
- [x] DataManagerV2 accepts session_dir parameter
- [x] DataManagerV2 forwards to ProvenanceTracker
- [x] clear() preserves session_dir
- [x] AgentClient computes and passes session_dir
- [x] External data_manager bypasses wiring (backward compatible)
- [x] 168 tests pass (98 DataManagerV2 + 70 client tests)

**Phase 3 (Complete):**
- [x] _load_from_disk() accepts optional path parameter
- [x] Session JSON saved with schema_version 1.1 and provenance key
- [x] load_session() restores provenance from original session directory
- [x] load_session() returns provenance_restored count
- [x] Old schema 1.0 sessions load without error
- [x] 1489 core tests pass (2 unrelated failures in test_lean_core.py)

**Phase 4 (Complete):**
- [x] CommandClient session_id parameter working
- [x] CLI --session-id option available
- [x] Pipeline export checks provenance availability
- [x] Auto-resolution to latest session works
- [x] Error messages include available sessions

**Phase 5 (Complete):**
- [x] 13 unit tests written (TEST-01 through TEST-12 + bonus)
- [x] Test suite runtime <5 seconds (0.20s actual)
- [x] Persistence method coverage >=90% (96.4% actual)
- [x] Tests validate both session_dir=None and session_dir=Path modes
- [x] Corrupt JSONL test verifies warning emission
- [x] Round-trip test verifies AnalysisStep reconstruction
- [x] No conflicts with existing test_provenance.py (66 tests pass together)

**Before releasing:**
- [ ] All 7 phases complete
- [ ] 12 unit tests + 3 integration tests passing
- [ ] Documentation updated
- [ ] `make test` passes (all existing tests unchanged)
- [ ] Manual verification: full workflow from analysis to cross-session export

---

*State initialized: 2026-02-15*
*Phase 1 complete: 2026-02-15*
*Phase 2 complete: 2026-02-15*
*Phase 3 complete: 2026-02-15*
*Phase 4 complete: 2026-02-15*
*Phase 5 complete: 2026-02-15*
