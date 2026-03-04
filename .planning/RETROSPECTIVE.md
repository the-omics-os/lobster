# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Codebase Cleanup & Restructuring

**Shipped:** 2026-03-04
**Phases:** 11 | **Plans:** 23 | **Sessions:** ~10

### What Was Built
- 6 critical GEO pipeline bug fixes (canonicalization, metadata keys, tar security, race conditions, typed retries, partial parse)
- geo_service.py decomposed into 5 domain modules (metadata_fetch, download_execution, archive_processing, matrix_parsing, concatenation)
- Plugin-first entry-point discovery for queue preparers and download services (5 databases)
- core/ restructured into 5 domain subpackages (runtime, queues, notebooks, provenance, governance) with 13 backward-compatible shims
- data_manager_v2.py (197 importers) moved to canonical path with zero breakage
- cli.py reduced from 8,049 to 1,338 LOC (83% reduction)
- Repository hygiene: normalized .gitignore, expanded make clean, removed stale artifacts

### What Worked
- **Sequential phase ordering** — fixing bugs before moving files prevented compound failures
- **Move-and-shim pattern** — proven scalable from 13 low-blast-radius files to the 197-importer data_manager
- **TDD scaffold wave** — writing RED tests in Plan 01 then making them GREEN in Plan 02/03 caught issues early (Plugin-First Registration)
- **Parallel phase execution** — Phases 2+3 ran simultaneously since both depended only on Phase 1
- **GSD workflow velocity** — 23 plans in 2.6 hours of execution time (avg 10 min/plan)
- **Milestone audit** — caught 2 real gaps (CORE-04, DMGR-04) that became Phases 10-11

### What Was Inefficient
- **ROADMAP.md progress table not updated for completed phases** — Phases 2, 3, 6, 8 still showed "In Progress" / "Planned" even after completion; had to be caught during milestone completion
- **cli.py target LOC unrealistic** — 300-400 target was too aggressive; Typer parameter declarations can't be moved, landing at 1,338 LOC
- **Nyquist validation gap** — all 11 phases have VALIDATION.md but none are fully nyquist_compliant (all draft status)

### Patterns Established
- **Move-and-shim**: Move file to canonical path → create `__getattr__` shim at old path → emit DeprecationWarning → CI guard prevents new imports via old path
- **Plugin-first registration**: Entry-point discovery first → hardcoded fallback gated behind `_ALLOW_HARDCODED_FALLBACK` flag
- **Domain module decomposition**: Extract focused modules → keep original as facade with `__getattr__` forwarding → narrow unit tests per module
- **CI deprecated-import guard**: `if grep ... then exit 1; fi` pattern (never `|| true`)
- **Multi-module `__getattr__`**: `_SUBMODULES` tuple with sequential search for module/package name collisions

### Key Lessons
1. **Fix before move** — correctness bugs must be resolved before any file restructuring; bugs at paths that change become debugging nightmares
2. **Shim blast radius scales linearly** — the same pattern that works for 1 importer works for 197; invest in the pattern early
3. **Milestone audits catch real gaps** — the audit found 2 critical issues that would have shipped broken; always audit before marking complete
4. **Typer parameter declarations are non-extractable** — cli.py will always be larger than pure wiring because parameter definitions must stay at the Typer call site
5. **Entry-point names must match runtime lookup keys** — mismatch causes silent failures; test entry-point discovery explicitly

### Cost Observations
- Model mix: ~60% sonnet (executor agents), ~25% haiku (plan-check, research), ~15% opus (planning, audit)
- Sessions: ~10 across 2 days
- Notable: Later phases (9-11) completed in 2-3 min each due to established patterns; Phase 4 (decomposition) was slowest at 51 min due to complexity

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~10 | 11 | Established GSD workflow, move-and-shim pattern, milestone audit |

### Cumulative Quality

| Milestone | Plans | Avg Duration | Requirements Satisfied |
|-----------|-------|--------------|----------------------|
| v1.0 | 23 | 10 min | 39/39 |

### Top Lessons (Verified Across Milestones)

1. Fix correctness bugs before structural refactoring
2. Milestone audits are mandatory — they catch real gaps
3. Move-and-shim pattern is the standard for backward-compatible restructuring
