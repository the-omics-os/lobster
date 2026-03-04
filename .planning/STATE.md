---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 11-01-PLAN.md
last_updated: "2026-03-04T21:45:20.049Z"
last_activity: 2026-03-04 -- Plan 11-01 complete (deprecated import migration + CI guard hardening)
progress:
  total_phases: 11
  completed_phases: 11
  total_plans: 23
  completed_plans: 23
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-03)

**Core value:** Preserve backward compatibility and pass all existing tests while fixing GEO bugs, decomposing monoliths, and restructuring core/
**Current focus:** All 11 phases complete. Kraken refactoring milestone finished.

## Current Position

Phase: 11 of 11 (strengthen-ci-deprecated-import-guard)
Plan: 1 of 1 in current phase (11-01 complete)
Status: All phases complete
Last activity: 2026-03-04 -- Plan 11-01 complete (deprecated import migration + CI guard hardening)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 18
- Average duration: 10 min
- Total execution time: 2.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 24 min | 8 min |
| 02 | 2 | 36 min | 18 min |
| 03 | 2 | 19 min | 10 min |
| 04 | 3 | 51 min | 17 min |
| 05 | 3 | 15 min | 5 min |
| 06 | 2 | 9 min | 5 min |
| 07 | 2 | 6 min | 3 min |
| 08 | 1 | 16 min | 16 min |
| 09 | 1 | 3 min | 3 min |
| 10 | 1 | 2 min | 2 min |
| 11 | 1 | 2 min | 2 min |

**Recent Trend:**
- Last 5 plans: 5 min, 1 min, 3 min, 2 min, 2 min
- Trend: Consistent fast execution on cleanup and hygiene tasks

*Updated after each plan completion*
| Phase 05-plugin-first-registration P03 | 15 | 2 tasks | 4 files |
| Phase 08 P02 | 21 | 2 tasks | 5 files |
| Phase 09 P02 | 2 | 2 tasks | 5 files |
| Phase 10 P01 | 2 | 2 tasks | 2 files |
| Phase 11 P01 | 2 | 2 tasks | 40 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- GEO fixes complete before folder restructuring (D1)
- GEO plan owns geo_service.py decomposition (D2)
- data_manager_v2: Fix -> Move -> Split in separate PRs (D5)
- Phases 2+3 can run in parallel (both depend only on Phase 1)
- MetadataEntry uses validation_result key (01-01)
- All GEO metadata writes via _store_geo_metadata or _enrich_geo_metadata (01-01)
- store_metadata() public method also updated for key consistency (01-01)
- Reject-all archive policy over skip-unsafe (01-03)
- CAS returns Optional[DownloadQueueEntry] for richer caller API (01-03)
- RetryOutcome enum uses lowercase string values (01-02)
- GDS resolution via lightweight Entrez eSummary in queue preparer (01-02)
- _retry_with_backoff always returns RetryResult, never bare None or strings (01-02)
- ParseResult dataclass at module level in parser.py for clean importability (02-01)
- isinstance guard on call sites for backward compat with bare DataFrame returns (02-01)
- Custom _WarningCapture handler for testing lobster loggers with propagate=False (02-01)
- [Phase 02]: Scoring heuristic replaces METADATA_KEYWORDS blacklist for file selection with contextual gene handling
- [Phase 02]: ARCHIVE_EXTENSIONS tuple constant shared across all 3 archive detection sites
- [Phase 02]: extract_dir/nested_extract_dir defined before try for cleanup access in except block
- [Phase 03]: _is_null_value() centralized helper for null detection across GEO pipeline (03-01)
- [Phase 03]: bool False and numeric 0 are valid domain values, NOT null (03-01)
- [Phase 03]: raw_data_available uses bool() AND not _is_null_value() for defense-in-depth (03-01)
- [Phase 03]: FILETYPES frozenset constants replace inline lists in rule evaluation (03-01)
- [Phase 03]: ARCHIVE_FIRST is dead code -- SUPPLEMENTARY_FIRST covers archive extraction (03-02)
- [Phase 03]: Unknown pipeline type strings fall back to FALLBACK gracefully (03-02)
- [Phase 04]: helpers.py uses lazy imports for pandas/anndata to avoid heavy deps at module level (04-01)
- [Phase 04]: _retry_with_backoff accepts console param instead of self for standalone usage (04-01)
- [Phase 04]: build_soft_url auto-detects GSE vs GSM from prefix, no id_type param needed (04-01)
- [Phase 04]: Domain modules use self.service pattern to access parent GEOService shared state (04-02)
- [Phase 04]: __getattr__ with __dict__.get() prevents recursion, lazy init handles mocked __init__ (04-02)
- [Phase 04]: Test patch paths updated to domain modules (tarfile, BulkRNASeqService) (04-02)
- [Phase 04]: pre_download_soft_file() handles GSE and GSM transparently via prefix detection (04-03)
- [Phase 04]: Domain module tests use mock_service fixture for complete isolation (04-03)
- [Phase 04]: Structural tests grep for PRE-DOWNLOAD SOFT as regression guard (04-03)
- [Phase 05]: Patch the singleton instance (lobster.core.component_registry.component_registry.list_X) not module.component_registry.list_X — routers import component_registry lazily inside method body (05-01)
- [Phase 05]: test_fallback_skipped_when_flag_false uses pytest.skip when _ALLOW_HARDCODED_FALLBACK missing — primary test already covers that RED condition (05-01)
- [Phase 05]: TestEntryPointDiscovery is GREEN (mocks substitute) while TestFallbackGating is RED (constant missing) — asymmetry is intentional per TDD wave 0 scaffold (05-01)
- [Phase 05]: Entry-point names (geo, sra, pride, massive, metabolights) match supported_databases()[0] for each class — ComponentRegistry uses these names for runtime lookup (05-02)
- [Phase 05]: pyproject.toml entry-point declarations are service discovery metadata, NOT dependency changes — CLAUDE.md Hard Rule #2 allows this edit (05-02)
- [Phase 05-plugin-first-registration]: [Phase 05]: _ALLOW_HARDCODED_FALLBACK gate uses early return before Phase 2 block — simple semantics, no complex conditional logic
- [Phase 05-plugin-first-registration]: [Phase 05]: discovered_names set tracking in Phase 1 loop enables warning log when EP discovery yields zero results and fallback is disabled
- [Phase 05-plugin-first-registration]: [Phase 05]: Phase 2 hardcoded block remains intact for emergency recovery — unreachable when flag=False, accessible when flag=True
- [Phase 06]: Docstring-only __init__.py files -- no re-exports to avoid coupling (06-01)
- [Phase 06]: Test scaffold covers all 13 shims upfront with Plan 02 tests marked xfail (06-01)
- [Phase 06]: Internal imports in moved queue files updated to canonical paths immediately (06-01)
- [Phase 06]: __getattr__ lazy shim in provenance/__init__.py for module/package name collision (06-02)
- [Phase 06]: Notebook files renamed on move (notebook_executor -> executor) for cleaner naming (06-02)
- [Phase 06]: Import-linter layers contract: notebooks > provenance > governance > queues > runtime (06-02)
- [Phase 07]: Internal imports in moved data_manager updated to canonical paths immediately (07-01)
- [Phase 07]: Test import statements left on shim path -- shim handles them transparently (07-01)
- [Phase 07]: 2 pre-existing KeyError failures on validation key accepted as out-of-scope (07-01)
- [Phase 07]: CI grep scope limited to lobster/scaffold/ and packages/ -- existing importers handled by shim (07-02)
- [Phase 08]: Classes kept in cli.py during incremental extraction -- Plan 02 converts to thin wrappers (08-01)
- [Phase 08]: Functions delegated via import+call pattern to keep cli.py working during extraction (08-01)
- [Phase 08]: init/chat/query _impl functions keep typer.Option annotations for standalone testing (08-01)
- [Phase 08]: cli.py at 1338 LOC (not 300-400) because Typer parameter declarations must remain; all function bodies are thin delegation calls
- [Phase 09]: Combined Task 1+2 commit since empty dirs not git-tracked; also removed lobster/data/ parent
- [Phase 09]: 14 gitignore sections (not 10) for better granularity -- added Linting, UI, Miscellaneous (09-01)
- [Phase 09]: Makefile clean uses find with maxdepth to scope package-local cleanup (09-01)
- [Phase 10]: Multi-module __getattr__ search order: provenance.py first, then analysis_ir, lineage, ir_coverage
- [Phase 10]: Do NOT update DE analysis call sites -- shim handles transparently, migration is Phase 11 scope
- [Phase 11]: Force-added files from gitignored premium packages to ensure deprecated import migration tracked in git

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-04T21:45:20.046Z
Stopped at: Completed 11-01-PLAN.md
Resume file: None
