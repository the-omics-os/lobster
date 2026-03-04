# Requirements: Lobster AI Codebase Cleanup & Restructuring

**Defined:** 2026-03-03
**Core Value:** Preserve backward compatibility and pass all existing tests while fixing GEO bugs, decomposing monoliths, and restructuring core/

## v1 Requirements

Requirements for this cleanup series. Each maps to roadmap phases (PRs).

### GEO Safety & Contracts

- [x] **GSAF-01**: GDS accessions canonicalized to GSE during queue preparation, with original_accession preserved (F1)
- [x] **GSAF-02**: Metadata validation key standardized to `validation_result` everywhere (F2)
- [x] **GSAF-03**: All GEO metadata_store writes use `_store_geo_metadata` helper — no malformed entries (F3)
- [x] **GSAF-04**: Nested tar extraction applies safe-member path checks — path traversal blocked (F4)
- [x] **GSAF-05**: `_retry_with_backoff` returns typed result enum, not string sentinels (F5)
- [x] **GSAF-06**: Download orchestrator status transitions are atomic with precondition check — no duplicate workers (A1)

### GEO Parser & Data Integrity

- [x] **GPAR-01**: Chunk parser returns `is_partial`, `rows_read`, `truncation_reason` flags (F6)
- [x] **GPAR-02**: All call sites handle partial parse results — mark modality/metadata and surface warning (F6)
- [x] **GPAR-03**: Supplementary file classifier handles `.tar.gz`, `.tgz`, `.tar.bz2` (F7)
- [x] **GPAR-04**: File-scoring heuristic replaces brittle keyword blacklist for expression file selection (F7)
- [x] **GPAR-05**: Partial parser failures trigger temp file cleanup and proper failure marking (A2)

### GEO Strategy Engine

- [x] **GSTR-01**: Null sanitization stores missing values as empty string/None, never truthy `"NA"` (F10)
- [x] **GSTR-02**: Strategy derivation uses explicit null checks and allowed file type enums (F10)
- [x] **GSTR-03**: `ARCHIVE_FIRST` dead branch resolved — either add triggering rule or remove (F11)

### GEO Service Decomposition

- [x] **GDEC-01**: geo_service.py split into 5 domain modules (metadata_fetch, download_execution, archive_processing, matrix_parsing, concatenation)
- [x] **GDEC-02**: GEOService class preserved as backward-compatible facade
- [x] **GDEC-03**: SOFT-download logic deduplicated between geo_service and geo_provider (F13)
- [x] **GDEC-04**: Each extracted module has narrow unit tests

### Plugin Registration

- [x] **PLUG-01**: Queue preparers discovered from `lobster.queue_preparers` entry points first-class
- [x] **PLUG-02**: Download services discovered from `lobster.download_services` entry points first-class
- [x] **PLUG-03**: Entry-point declarations added to pyproject.toml BEFORE fallback gating (A3)
- [x] **PLUG-04**: All 5 databases (GEO, SRA, PRIDE, MassIVE, MetaboLights) discoverable via entry points (A3)
- [x] **PLUG-05**: Existing tests updated for entry-point discovery assertions (A4)
- [x] **PLUG-06**: Hardcoded fallback gated with explicit flag

### Core Restructuring

- [x] **CORE-01**: Core subpackages created: runtime/, queues/, notebooks/, provenance/, governance/
- [x] **CORE-02**: 13 files moved to domain subpackages with backward-compatible shims at old paths
- [x] **CORE-03**: Shims emit DeprecationWarning with removal version
- [x] **CORE-04**: Import-linter config updated for new subpackage paths (A5)
- [x] **CORE-05**: No import cycles introduced

### Data Manager Move

- [x] **DMGR-01**: data_manager_v2.py moved to core/runtime/data_manager.py
- [x] **DMGR-02**: Shim at old path re-exports everything — zero breakage for 79 source + 118 test importers
- [x] **DMGR-03**: Scaffold templates updated to import from new path (A6)
- [x] **DMGR-04**: New code cannot import from old path (CI/lint check)

### CLI Decomposition

- [x] **CLID-01**: Command bodies moved to cli_internal/commands/
- [x] **CLID-02**: cli.py reduced to composition/wiring with minimal control flow
- [x] **CLID-03**: All CLI subcommands work identically after decomposition

### Repo Hygiene

- [ ] **HYGN-01**: .gitignore sections normalized
- [ ] **HYGN-02**: `make clean` and `make clean-all` expanded for package-local artifacts
- [x] **HYGN-03**: Empty placeholder dirs removed (those not populated by GEO decomposition)
- [x] **HYGN-04**: Deprecated shim files cleaned (geo_parser.py, geo_downloader.py) — verified unused first
- [ ] **HYGN-05**: Stale build artifacts (dist/, *.egg-info/, .ruff_cache/, MagicMock/) removed from package dirs

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Shim Retirement

- **SHIM-01**: Remove old import paths for all moved modules (after one release cycle)
- **SHIM-02**: Enforce strict target-path imports via CI
- **SHIM-03**: Remove core/vector/* backward-compat shims

### Future Decomposition

- **FDEC-01**: Provider relocation (tools/providers/ → services/data_access/providers/)
- **FDEC-02**: data_manager_v2 internal split
- **FDEC-03**: client.py decomposition (2,867 LOC)
- **FDEC-04**: content_access_service.py decomposition (2,136 LOC)
- **FDEC-05**: GEO LLM-driven planning layer (typed StrategyPlan schema)

## Out of Scope

| Feature | Reason |
|---------|--------|
| PR-10 shim retirement | Deferred — must wait one release cycle for consumers to migrate |
| Provider relocation | Clean coupling but low-urgency; reverse edge (webpage_provider → docling_service) needs resolution first |
| data_manager_v2 internal split | Must settle after move in PR-7 |
| client.py decomposition | High blast-radius (2,867 LOC); deferred until data_manager move is stable |
| GEO LLM planning layer | Architectural feature, not cleanup; requires typed StrategyPlan schema design |
| Behavior changes in extraction PRs | PR-4/6/7/8 are pure mechanical — logic fixes only in PR-1/2/3/5 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GSAF-01 | Phase 1 (PR-1) | Complete |
| GSAF-02 | Phase 1 (PR-1) | Complete |
| GSAF-03 | Phase 1 (PR-1) | Complete |
| GSAF-04 | Phase 1 (PR-1) | Complete |
| GSAF-05 | Phase 1 (PR-1) | Complete |
| GSAF-06 | Phase 1 (PR-1) | Complete |
| GPAR-01 | Phase 2 (PR-2) | Complete |
| GPAR-02 | Phase 2 (PR-2) | Complete |
| GPAR-03 | Phase 2 (PR-2) | Complete |
| GPAR-04 | Phase 2 (PR-2) | Complete |
| GPAR-05 | Phase 2 (PR-2) | Complete |
| GSTR-01 | Phase 3 (PR-3) | Complete |
| GSTR-02 | Phase 3 (PR-3) | Complete |
| GSTR-03 | Phase 3 (PR-3) | Complete |
| GDEC-01 | Phase 4 (PR-4) | Complete |
| GDEC-02 | Phase 4 (PR-4) | Complete |
| GDEC-03 | Phase 4 (PR-4) | Complete |
| GDEC-04 | Phase 4 (PR-4) | Complete |
| PLUG-01 | Phase 5 (PR-5) | Complete |
| PLUG-02 | Phase 5 (PR-5) | Complete |
| PLUG-03 | Phase 5 (PR-5) | Complete |
| PLUG-04 | Phase 5 (PR-5) | Complete |
| PLUG-05 | Phase 5 (PR-5) | Complete |
| PLUG-06 | Phase 5 (PR-5) | Complete |
| CORE-01 | Phase 6 (PR-6) | Complete |
| CORE-02 | Phase 6 (PR-6) | Complete |
| CORE-03 | Phase 6 (PR-6) | Complete |
| CORE-04 | Phase 6 (PR-6) | Complete |
| CORE-05 | Phase 6 (PR-6) | Complete |
| DMGR-01 | Phase 7 (PR-7) | Complete |
| DMGR-02 | Phase 7 (PR-7) | Complete |
| DMGR-03 | Phase 7 (PR-7) | Complete |
| DMGR-04 | Phase 7 (PR-7) | Complete |
| CLID-01 | Phase 8 (PR-8) | Complete |
| CLID-02 | Phase 8 (PR-8) | Complete |
| CLID-03 | Phase 8 (PR-8) | Complete |
| HYGN-01 | Phase 9 (PR-9) | Pending |
| HYGN-02 | Phase 9 (PR-9) | Pending |
| HYGN-03 | Phase 9 (PR-9) | Complete |
| HYGN-04 | Phase 9 (PR-9) | Complete |
| HYGN-05 | Phase 9 (PR-9) | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-03*
*Last updated: 2026-03-03 after initial definition*
