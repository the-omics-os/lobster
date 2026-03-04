# Roadmap: Lobster AI Codebase Cleanup & Restructuring

## Overview

A 9-phase refactoring series (each phase = one PR) that fixes critical GEO pipeline bugs, decomposes monoliths, migrates to plugin-first registration, and restructures core/ into domain subpackages. Phases execute sequentially except Phases 2 and 3, which can run in parallel after Phase 1. Zero breaking changes throughout -- production customers use the GEO pipeline daily.

Detailed execution spec: `kevin_notes/UNIFIED_CLEANUP_PLAN.md`

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: GEO Safety & Contract Hotfixes** - Fix GDS canonicalization, metadata key consistency, tar traversal security, retry sentinels, and download race condition (PR-1) (completed 2026-03-04)
- [ ] **Phase 2: GEO Parser & Data Integrity** - Add partial parse signaling, fix archive classifier, improve file selection heuristic, add temp cleanup (PR-2) [parallel with Phase 3]
- [ ] **Phase 3: GEO Strategy Engine Hardening** - Fix null sanitization, tighten strategy derivation, resolve ARCHIVE_FIRST dead branch (PR-3) [parallel with Phase 2]
- [x] **Phase 4: GEO Service Decomposition** - Split geo_service.py into 5 domain modules with backward-compatible facade (PR-4) (completed 2026-03-04)
- [x] **Phase 5: Plugin-First Registration** - Migrate queue preparers and download services to entry-point discovery, gate hardcoded fallbacks (PR-5) (completed 2026-03-04)
- [ ] **Phase 6: Core Subpackage Creation + Moves** - Create runtime/, queues/, notebooks/, provenance/, governance/ subpackages and move 13 files with shims (PR-6)
- [x] **Phase 7: data_manager_v2 Move** - Move highest-blast-radius file to core/runtime/data_manager.py with full shim (PR-7) (completed 2026-03-04)
- [ ] **Phase 8: CLI Decomposition** - Extract command bodies from cli.py to cli_internal/commands/ (PR-8)
- [x] **Phase 9: Repo Hygiene & Packaging Cleanup** - Normalize gitignore, expand make clean, remove stale artifacts and empty dirs (PR-9) (completed 2026-03-04)
- [ ] **Phase 10: Fix Provenance `__getattr__` Shim** - Extend provenance __init__.py to cover analysis_ir/lineage/ir_coverage submodules, fixing DE analysis ImportError (PR-10) [GAP CLOSURE]
- [ ] **Phase 11: Strengthen CI Deprecated-Import Guard** - Remove || true from CI check, migrate 39 deprecated imports in packages/ to canonical paths (PR-11) [GAP CLOSURE]

## Phase Details

### Phase 1: GEO Safety & Contract Hotfixes
**Goal**: GEO queue preparation, metadata storage, archive extraction, and download orchestration are correct and safe -- no data corruption, path traversal, or duplicate workers
**Depends on**: Nothing (first phase)
**Requirements**: GSAF-01, GSAF-02, GSAF-03, GSAF-04, GSAF-05, GSAF-06
**Success Criteria** (what must be TRUE):
  1. GDS accessions submitted to the GEO pipeline emerge as canonical GSE accessions in download queue entries, with original_accession preserved
  2. Metadata written by any GEO component can be read by any other GEO component without key mismatches (validation_result standardized)
  3. Nested tar archives containing path-traversal members (../) are rejected before extraction reaches the filesystem
  4. Concurrent download workers processing the same queue entry results in exactly one succeeding and others receiving a no-op, never duplicate processing
  5. Retry logic returns typed results that all call sites handle exhaustively -- no string sentinel comparisons remain
**Plans:** 3/3 plans complete

Plans:
- [ ] 01-01-PLAN.md — Standardize metadata validation key and enforce centralized writes (GSAF-02, GSAF-03)
- [ ] 01-02-PLAN.md — Typed retry results and GDS canonicalization (GSAF-05, GSAF-01)
- [ ] 01-03-PLAN.md — Tar extraction security and CAS download queue transitions (GSAF-04, GSAF-06)

### Phase 2: GEO Parser & Data Integrity
**Goal**: GEO parser reports partial results explicitly, selects the right files from archives, and cleans up after failures
**Depends on**: Phase 1
**Requirements**: GPAR-01, GPAR-02, GPAR-03, GPAR-04, GPAR-05
**Success Criteria** (what must be TRUE):
  1. When the chunk parser hits a memory limit or truncation, the returned result includes is_partial=True, rows_read count, and truncation_reason -- never a silent partial
  2. Call sites receiving partial parse results mark the modality/metadata accordingly and surface a user-visible warning
  3. Supplementary file classifier correctly identifies .tar.gz, .tgz, and .tar.bz2 archives alongside existing formats
  4. Expression file selection uses a scoring heuristic that outperforms the keyword blacklist on known edge cases (e.g., files with "gene" in the name)
  5. Parser failures leave zero temp files on disk -- cleanup runs on every early-break path
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — ParseResult dataclass and partial parse signaling with call site propagation (GPAR-01, GPAR-02)
- [ ] 02-02-PLAN.md — Archive classifier fix, expression file scoring heuristic, and temp cleanup (GPAR-03, GPAR-04, GPAR-05)

### Phase 3: GEO Strategy Engine Hardening
**Goal**: Strategy engine makes correct decisions on null/missing metadata and has no unreachable code paths
**Depends on**: Phase 1
**Requirements**: GSTR-01, GSTR-02, GSTR-03
**Success Criteria** (what must be TRUE):
  1. Missing metadata values are stored as empty string or None -- the string "NA" never appears as a truthy value in strategy inputs
  2. Strategy derivation uses explicit null checks and allowed file type enums, producing correct strategies for datasets with missing fields
  3. The ARCHIVE_FIRST branch either has a triggering rule with test coverage or is removed entirely -- zero dead branches in strategy.py
**Plans:** 1/2 plans executed

Plans:
- [ ] 03-01-PLAN.md — Null sanitization fix, _is_null_value helper, filetype constants, consumer hardening (GSTR-01, GSTR-02)
- [ ] 03-02-PLAN.md — Remove ARCHIVE_FIRST dead branch from enum, pipeline map, and geo_service (GSTR-03)

### Phase 4: GEO Service Decomposition
**Goal**: geo_service.py (5,954 LOC) is decomposed into focused domain modules while preserving the existing API surface
**Depends on**: Phase 2, Phase 3
**Requirements**: GDEC-01, GDEC-02, GDEC-03, GDEC-04
**Success Criteria** (what must be TRUE):
  1. Five domain modules exist (metadata_fetch, download_execution, archive_processing, matrix_parsing, concatenation) each with focused responsibility
  2. GEOService class remains importable and callable with identical behavior -- existing integration tests pass unchanged
  3. SOFT-download logic exists in exactly one location (no duplication between geo_service and geo_provider)
  4. Each extracted module has its own narrow unit tests that run independently
**Plans:** 3/3 plans complete

Plans:
- [ ] 04-01-PLAN.md — Extract shared helpers and SOFT pre-download into reusable modules (GDEC-03)
- [ ] 04-02-PLAN.md — Extract 5 domain modules and convert geo_service.py to facade (GDEC-01, GDEC-02)
- [ ] 04-03-PLAN.md — SOFT dedup across codebase, narrow unit tests, facade compatibility tests (GDEC-01, GDEC-02, GDEC-03, GDEC-04)

### Phase 5: Plugin-First Registration
**Goal**: Queue preparers and download services are discovered via entry points, with hardcoded fallback available but gated
**Depends on**: Phase 4
**Requirements**: PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-05, PLUG-06
**Success Criteria** (what must be TRUE):
  1. All 5 databases (GEO, SRA, PRIDE, MassIVE, MetaboLights) are discoverable via lobster.queue_preparers and lobster.download_services entry points
  2. A new queue preparer or download service can be added by an external package declaring entry points -- zero changes to core routing code required
  3. Hardcoded fallback registration is gated behind an explicit flag and only activates when entry-point discovery yields nothing
  4. Existing tests validate the entry-point discovery path, not hardcoded defaults
**Plans:** 3/3 plans complete

Plans:
- [ ] 05-01-PLAN.md — Write failing test scaffolds: contract tests, TestEntryPointDiscovery, TestFallbackGating (PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-05, PLUG-06)
- [ ] 05-02-PLAN.md — Add queue_preparers + download_services entry-point declarations to pyproject.toml (PLUG-03, PLUG-04)
- [ ] 05-03-PLAN.md — Gate hardcoded fallback with _ALLOW_HARDCODED_FALLBACK=False, update existing tests for 5 DBs (PLUG-01, PLUG-02, PLUG-05, PLUG-06)

### Phase 6: Core Subpackage Creation + Moves
**Goal**: core/ is organized into domain subpackages with all old import paths preserved via deprecation shims
**Depends on**: Phase 5
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05
**Success Criteria** (what must be TRUE):
  1. Five subpackages exist under core/ (runtime/, queues/, notebooks/, provenance/, governance/) with 13 files in their new homes
  2. Every old import path (e.g., `from lobster.core.download_queue import ...`) works via shim and emits a DeprecationWarning with removal version
  3. Import-linter passes with updated rules that reflect new subpackage boundaries
  4. No import cycles exist between core subpackages
**Plans:** 1/2 plans executed

Plans:
- [ ] 06-01-PLAN.md — Test scaffold + move governance/, queues/, runtime/ (6 files, low blast radius) (CORE-01, CORE-02, CORE-03, CORE-05)
- [ ] 06-02-PLAN.md — Move notebooks/ + provenance/ (7 files, __getattr__ collision), update import-linter (CORE-01, CORE-02, CORE-03, CORE-04, CORE-05)

### Phase 7: data_manager_v2 Move
**Goal**: data_manager_v2.py lives at core/runtime/data_manager.py with zero breakage for its 197 importers (79 source + 118 test)
**Depends on**: Phase 6
**Requirements**: DMGR-01, DMGR-02, DMGR-03, DMGR-04
**Success Criteria** (what must be TRUE):
  1. `from lobster.core.runtime.data_manager import DataManagerV2` works as the canonical import
  2. `from lobster.core.data_manager_v2 import DataManagerV2` still works via shim with DeprecationWarning -- zero breakage
  3. `lobster scaffold agent` produces code importing from the new path (no deprecation warnings in newly generated code)
  4. A CI/lint check prevents new files from importing via the old path
**Plans**: 2 plans

Plans:
- [ ] 07-01-PLAN.md — Move data_manager_v2.py to core/runtime/data_manager.py, create shim, update mock.patch strings (DMGR-01, DMGR-02)
- [ ] 07-02-PLAN.md — Update scaffold templates, import-linter, CI deprecated-import check (DMGR-03, DMGR-04)

### Phase 8: CLI Decomposition
**Goal**: cli.py (9,323 LOC) is reduced to composition/wiring with command bodies in cli_internal/commands/
**Depends on**: Phase 7
**Requirements**: CLID-01, CLID-02, CLID-03
**Success Criteria** (what must be TRUE):
  1. Command bodies live in cli_internal/commands/ as separate modules
  2. cli.py contains only Typer app wiring and composition -- minimal control flow
  3. All CLI subcommands produce identical output and behavior (lobster --help, lobster chat, lobster query, lobster init)
**Plans:** 1/2 plans executed

Plans:
- [ ] 08-01-PLAN.md — Extract foundation modules (session_infra, animations, display_helpers) and command bodies (init, chat, query) (CLID-01)
- [ ] 08-02-PLAN.md — Extract slash commands, reduce cli.py to wiring-only, update test patch paths (CLID-01, CLID-02, CLID-03)

### Phase 9: Repo Hygiene & Packaging Cleanup
**Goal**: Repository is clean of stale artifacts, empty dirs, deprecated shims, and has comprehensive make clean targets
**Depends on**: Phase 8
**Requirements**: HYGN-01, HYGN-02, HYGN-03, HYGN-04, HYGN-05
**Success Criteria** (what must be TRUE):
  1. .gitignore is normalized with clear sections and no redundant patterns
  2. `make clean` and `make clean-all` remove all package-local build artifacts (dist/, *.egg-info/, .ruff_cache/, MagicMock/)
  3. No empty placeholder directories remain that were not populated by GEO decomposition
  4. Deprecated shim files (geo_parser.py, geo_downloader.py) are removed after verification they have zero importers
  5. No stale build artifacts exist in any package directory
**Plans:** 2/2 plans complete

Plans:
- [ ] 09-01-PLAN.md — Normalize .gitignore sections, expand Makefile clean targets, remove stale artifacts (HYGN-01, HYGN-02, HYGN-05)
- [ ] 09-02-PLAN.md — Remove empty placeholder dirs, update deprecated shim imports, remove shim files (HYGN-03, HYGN-04)

### Phase 10: Fix Provenance `__getattr__` Shim
**Goal**: `from lobster.core.provenance import AnalysisStep` (and all other provenance submodule exports) works via the backward-compat `__getattr__` shim — DE analysis provenance IR creation no longer breaks at runtime
**Depends on**: Phase 6
**Requirements**: CORE-04
**Gap Closure:** Closes gaps from audit (CORE-04 partial, integration Phase 06→lobster-transcriptomics, flow DE Analysis Provenance IR Creation)
**Success Criteria** (what must be TRUE):
  1. `from lobster.core.provenance import AnalysisStep` resolves without ImportError
  2. `from lobster.core.provenance import X` works for all public names in analysis_ir.py, lineage.py, and ir_coverage.py
  3. All 6 call sites in de_analysis_expert.py (lines 640, 1123, 2096, 2294, 2467, 2937) execute without import failures
  4. Existing provenance.py shim behavior is preserved (ProvenanceTracker still accessible)
**Plans:** 1 plan

Plans:
- [ ] 10-01-PLAN.md — Extend provenance __getattr__ shim to search all 4 submodules with TDD test coverage (CORE-04)

### Phase 11: Strengthen CI Deprecated-Import Guard
**Goal**: CI deprecated-import check blocks PRs that introduce old-path imports — existing 39 violations in packages/ migrated to canonical paths
**Depends on**: Phase 7
**Requirements**: DMGR-04
**Gap Closure:** Closes gaps from audit (integration Phase 07 CI guard → packages/)
**Success Criteria** (what must be TRUE):
  1. `.github/workflows/ci-basic.yml` deprecated-import grep step fails CI (no `|| true`) when old-path imports are found
  2. All 39 pre-existing deprecated imports in `packages/` updated to canonical paths
  3. `grep -r 'from lobster.core.data_manager_v2' packages/` returns zero matches
  4. CI passes cleanly after migration

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 + 3 (parallel) -> 4 -> 5 -> 6 -> 7 -> 8 -> 9. Phases 2 and 3 may execute in parallel (both depend only on Phase 1).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. GEO Safety & Contract Hotfixes | 3/3 | Complete   | 2026-03-04 |
| 2. GEO Parser & Data Integrity | 0/2 | Planned | - |
| 3. GEO Strategy Engine Hardening | 1/2 | In Progress|  |
| 4. GEO Service Decomposition | 3/3 | Complete   | 2026-03-04 |
| 5. Plugin-First Registration | 3/3 | Complete   | 2026-03-04 |
| 6. Core Subpackage Creation + Moves | 1/2 | In Progress|  |
| 7. data_manager_v2 Move | 2/2 | Complete   | 2026-03-04 |
| 8. CLI Decomposition | 1/2 | In Progress|  |
| 9. Repo Hygiene & Packaging Cleanup | 2/2 | Complete   | 2026-03-04 |
| 10. Fix Provenance __getattr__ Shim | 0/1 | Planned | - |
| 11. Strengthen CI Deprecated-Import Guard | 0/0 | Planned | - |
