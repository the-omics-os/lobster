# Lobster AI Codebase Cleanup & Restructuring

## What This Is

A 9-PR refactoring series across the Lobster AI codebase that fixes critical GEO pipeline bugs, decomposes monoliths, migrates to plugin-first registration, and restructures core/ into domain subpackages. Zero breaking changes — production customers use the GEO pipeline daily.

## Core Value

Every change must preserve backward compatibility and pass all existing tests. Fix correctness bugs before moving files. One PR at a time — never start PR-N+1 until PR-N passes its gate.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ GEO pipeline operational (preparation + download + parsing) — existing
- ✓ Plugin-based agent discovery via entry points — existing
- ✓ AQUADIF tool taxonomy and contract tests — existing
- ✓ Core SDK modular package architecture (10 packages) — existing
- ✓ Backward-compatible shim pattern proven (core/vector/*) — existing

### Active

<!-- Current scope. Building toward these. -->

**Phase A — GEO Correctness (PR-1 → PR-3):**
- [ ] GDS→GSE canonicalization in queue preparation (F1)
- [ ] Metadata validation key standardization (F2)
- [ ] Malformed metadata entry prevention (F3)
- [ ] Nested tar path traversal security (F4)
- [ ] Retry sentinel replacement with typed results (F5)
- [ ] Download orchestrator race condition fix (A1)
- [ ] Parser partial status signaling (F6)
- [ ] Partial parse temp file cleanup (A2)
- [ ] Archive format classifier improvement (F7)
- [ ] Null sanitization fix for strategy engine (F10)
- [ ] ARCHIVE_FIRST dead branch resolution (F11)

**Phase B — GEO Decomposition + Plugin Migration (PR-4 → PR-5):**
- [ ] geo_service.py split into 5 domain modules + facade
- [ ] SOFT-download logic deduplication
- [ ] Plugin-first queue preparer/download service registration
- [ ] Entry-point declarations in pyproject.toml (A3)
- [ ] Test assertion updates for plugin discovery (A4)

**Phase C — Core Restructuring (PR-6 → PR-7):**
- [ ] Core subpackages created (runtime, queues, notebooks, provenance, governance)
- [ ] 13 files moved with backward-compatible shims
- [ ] Import-linter config updated (A5)
- [ ] data_manager_v2.py moved to core/runtime/data_manager.py with shim
- [ ] Scaffold templates updated to new import path (A6)

**Phase D — CLI + Hygiene (PR-8 → PR-9):**
- [ ] CLI command bodies extracted to cli_internal/commands/
- [ ] Repo hygiene (gitignore, make clean, empty dirs, stale artifacts)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- PR-10 shim retirement — deferred to after one release cycle
- Provider relocation (tools/providers/ → services/data_access/providers/) — clean coupling but low-urgency, reverse edge needs resolution first
- data_manager_v2 internal split — must settle after move
- client.py decomposition — high blast-radius (2,867 LOC), deferred until data_manager move is stable
- GEO LLM-driven planning layer — architectural feature, not cleanup
- content_access_service.py decomposition — identified in audit but not targeted by either plan

## Context

**Spec source:** Three pre-validated documents in `kevin_notes/`:
- `UNIFIED_CLEANUP_PLAN.md` — execution spec (Sections 3 and 8 are the engineer's contract)
- `GEO_cleanup.md` — findings F1-F13 with evidence locations
- `folder_cleanup.md` — findings 4.1-4.8 with structural analysis

**Validation:** Brutalist architecture critique by Codex (gpt-5.3) and Gemini (3.1 Pro). 6 amendments incorporated (A1-A6). 5 concerns reviewed and dismissed with rationale.

**Codebase scale:** 220K+ LOC across core SDK + 10 agent packages. Highest-risk files: `geo_service.py` (5,847 LOC), `cli.py` (9,226 LOC), `data_manager_v2.py` (3,962 LOC, 79 importers).

**Production dependency:** GEO pipeline used daily by customers. Zero-downtime constraint.

## Constraints

- **Backward compatibility**: Every old import path must have a shim for one release cycle
- **Execution order**: GEO correctness fixes (Phase A) MUST complete before any file moves (Phase C/D)
- **Separation of concerns**: No behavior changes mixed with file moves (PR-4/6/7/8 are pure mechanical extraction)
- **PR-5 internal sequencing**: declare entry points → verify discovery → THEN gate fallback (A3)
- **PR-7 blast radius**: 79 source + 118 test importers — shim must re-export everything
- **Testing**: Every PR must pass `pytest tests/unit && pytest tests/integration && pytest -m contract`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GEO fixes complete before folder restructuring | Fix correctness bugs before moving files — avoids fixing at paths that change | — Pending |
| GEO plan owns geo_service.py decomposition | Domain-specific split into 5 modules | — Pending |
| Consolidate plugin/entry-point into one PR (PR-5) | GEO Phase 4 + Folder Phase 3 become single effort | — Pending |
| data_manager_v2: Fix → Move → Split (3 PRs) | Highest blast-radius file, each step independently reviewable | — Pending |
| Defer provider relocation | Clean coupling but low-urgency, reverse edge needs resolution | — Pending |
| Keep empty GEO subdirectories | May be populated by decomposition, remove post-completion if still empty | — Pending |
| Single unified PR series with gates | One ordered delivery, not two parallel tracks | — Pending |

---
*Last updated: 2026-03-03 after initialization*
