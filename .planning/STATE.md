# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Agents can semantically match any biomedical term to the correct ontology concept with calibrated confidence scores, using zero configuration out of the box.
**Current focus:** Phase 2 - Service Integration

## Current Position

Phase: 2 of 6 (Service Integration)
Plan: 2 of 3 in current phase (02-02 complete)
Status: Plan 02-02 complete, ready for 02-03
Last activity: 2026-02-18 — Plan 02-02 executed (match_ontology)

Progress: [████░░░░░░] 25%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 2.5min
- Total execution time: 0.17 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 8min | 2.7min |
| 02-service-integration | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 01-01 (3min), 01-02 (2min), 01-03 (3min), 02-01 (2min)
- Trend: Stable

*Updated after each plan completion*
| Phase 02 P01 | 2min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- ChromaDB as default backend — Small data (~60K vectors), zero cost, proven in SRAgent, local+cloud symmetry
- SapBERT as default embedding — SOTA biomedical entity linking, 4M+ UMLS synonym pairs, 768d, free/local
- Cross-encoder as default reranker — Zero API cost, offline, ~1-5s for 100 docs on CPU
- Core location (not package) — Vector search is cross-agent infrastructure used by annotation, metadata, research
- Augment, don't replace — Existing annotate_cell_types tool stays; new semantic tool added alongside
- [01-01] Score rounding via model_post_init to 4 decimal places for consistent display
- [01-01] TYPE_CHECKING guard in vector/__init__.py for zero-cost imports
- [01-01] Column-oriented search() return format matching ChromaDB convention
- [01-02] CLS-token pooling for SapBERT (trained with CLS, not mean pooling)
- [01-02] batch_size=128 for SapBERT encode per model card recommendation
- [01-02] 5000-doc chunk limit for ChromaDB batch operations
- [01-02] Raw ChromaDB results from search; distance->similarity conversion in service layer
- [01-03] Dependency injection: service accepts backend/embedder for testing, falls back to config factory
- [01-03] Distance-to-similarity conversion with clamping: score = max(0.0, min(1.0, 1.0 - distance))
- [01-03] MockEmbedder/MockVectorBackend pattern for testing without heavy deps
- [01-03] __getattr__ lazy loading in __init__.py for all public classes
- [Phase 01]: Dependency injection: service accepts backend/embedder for testing, falls back to config factory
- [02-01] Import-guarded obonet inside function body, not module level
- [02-01] lru_cache(maxsize=3) for process-lifetime caching of parsed OBO graphs
- [02-01] OBO edge direction convention: child -> parent (is_a), successors = parents

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-18 (Plan 02-01 execution)
Stopped at: Completed 02-01-PLAN.md (Ontology Graph)
Resume file: .planning/phases/02-service-integration/02-01-SUMMARY.md
