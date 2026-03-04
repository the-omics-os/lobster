# Lobster AI Codebase Cleanup & Restructuring

## What This Is

A completed refactoring series across the Lobster AI codebase that fixed critical GEO pipeline bugs, decomposed two monoliths (geo_service.py and cli.py), migrated to plugin-first registration, and restructured core/ into domain subpackages. Zero breaking changes — production customers used the GEO pipeline throughout with no disruption.

## Core Value

Every change preserves backward compatibility and passes all existing tests. Fix correctness bugs before moving files. One PR at a time.

## Requirements

### Validated

- ✓ GDS→GSE canonicalization in queue preparation — v1.0
- ✓ Metadata validation key standardized to `validation_result` — v1.0
- ✓ Centralized metadata writes via `_store_geo_metadata` helper — v1.0
- ✓ Nested tar path traversal security (reject-all policy) — v1.0
- ✓ Typed retry results replacing string sentinels — v1.0
- ✓ Atomic download queue transitions (CAS) — v1.0
- ✓ Partial parse signaling (ParseResult dataclass) — v1.0
- ✓ Archive classifier covering .tar.gz/.tgz/.tar.bz2 — v1.0
- ✓ Scoring heuristic replacing keyword blacklist for file selection — v1.0
- ✓ Temp file cleanup on parser failure — v1.0
- ✓ Null sanitization in strategy engine — v1.0
- ✓ ARCHIVE_FIRST dead branch removed — v1.0
- ✓ geo_service.py decomposed into 5 domain modules + facade — v1.0
- ✓ SOFT-download logic deduplicated — v1.0
- ✓ Plugin-first queue preparer/download service registration (5 DBs) — v1.0
- ✓ Hardcoded fallback gated behind explicit flag — v1.0
- ✓ Core subpackages: runtime/, queues/, notebooks/, provenance/, governance/ — v1.0
- ✓ 13 files moved with backward-compatible deprecation shims — v1.0
- ✓ data_manager_v2 moved to core/runtime/data_manager.py — v1.0
- ✓ Scaffold templates updated for canonical import paths — v1.0
- ✓ CI guard preventing deprecated-path imports — v1.0
- ✓ CLI decomposed: cli.py 8,049 → 1,338 LOC — v1.0
- ✓ .gitignore normalized, make clean expanded — v1.0
- ✓ Deprecated GEO shim files removed — v1.0
- ✓ Provenance __getattr__ shim covering all 4 submodules — v1.0

### Active

<!-- Fresh for next milestone -->

### Out of Scope

- Shim retirement (SHIM-01/02/03) — must wait one release cycle for consumers to migrate
- Provider relocation (tools/providers/ → services/data_access/providers/) — reverse edge needs resolution first
- data_manager_v2 internal split — must settle after move
- client.py decomposition (2,867 LOC) — deferred until data_manager move is stable
- content_access_service.py decomposition (2,136 LOC) — not targeted by v1.0
- GEO LLM-driven planning layer — architectural feature, not cleanup

## Context

Shipped v1.0 with 290,933 LOC Python across core SDK + 10 agent packages.
Tech stack: Python 3.12, LangGraph, LangChain, uv, Typer CLI, entry-point plugins.
125 commits, 682 files changed, +143,972/-65,793 net change over 2 days.

Known tech debt: 12 items (thread safety for cloud SaaS, 20 core files on deprecated import paths, orphaned geo/facade.py). All non-blocking.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GEO fixes before folder restructuring | Fix correctness bugs before moving files | ✓ Good — no regressions from moves |
| GEO plan owns geo_service.py decomposition | Domain-specific split into 5 modules | ✓ Good — clean separation achieved |
| Consolidate plugin/entry-point into PR-5 | GEO Phase 4 + Folder Phase 3 as single effort | ✓ Good — 5 DBs discoverable |
| data_manager_v2: Fix → Move → Split (3 PRs) | Highest blast-radius file, each step independently | ✓ Good — zero breakage for 197 importers |
| Reject-all tar extraction policy | Security over convenience | ✓ Good — CVE-2007-4559 mitigated |
| CAS status transitions for downloads | Prevent duplicate workers | ✓ Good — race condition eliminated |
| __getattr__ multi-module shim for provenance | Module/package name collision | ✓ Good — all 4 submodules accessible |
| Strict CI guard (no `|| true`) | Prevent regression on deprecated imports | ✓ Good — 39 violations migrated |
| cli.py keeps Typer param declarations | Parameter definitions must stay at call site | ⚠️ Revisit — 1,338 LOC vs target 300-400 |
| Defer shim retirement | One release cycle for consumers | — Pending (v1.1 scope) |

## Constraints

- Backward compatibility: Every old import path has shim for one release cycle
- Testing: `pytest tests/unit && pytest tests/integration && pytest -m contract` must pass
- Zero-downtime: GEO pipeline used daily by production customers

---
*Last updated: 2026-03-04 after v1.0 milestone*
