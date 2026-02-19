# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Agents can semantically match any biomedical term to the correct ontology concept with calibrated confidence scores, using zero configuration out of the box.
**Current focus:** Phase 4 - Performance (reranking infrastructure)

## Current Position

Phase: 4 of 6 (Performance)
Plan: 1 of 2 in current phase (04-01 complete)
Status: Executing Phase 04
Last activity: 2026-02-19 — Plan 04-01 executed (reranker infrastructure)

Progress: [█████████░] 56%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 3.1min
- Total execution time: 0.47 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 3 | 8min | 2.7min |
| 02-service-integration | 3 | 10min | 3.3min |
| 03-agent-tooling | 2 | 10min | 5.0min |
| 04-performance | 1 | 5min | 5.0min |

**Recent Trend:**
- Last 5 plans: 02-03 (5min), 03-01 (6min), 03-02 (4min), 04-01 (5min)
- Trend: Stable

*Updated after each plan completion*
| Phase 03 P01 | 6min | 2 tasks | 2 files |
| Phase 04 P01 | 5min | 2 tasks | 7 files |

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
- [02-02] ONTOLOGY_COLLECTIONS as module-level dict constant (not class attribute) for easy import
- [02-02] 4x oversampling factor (k*4) for future reranking without over-fetching
- [02-02] Lazy import of OntologyMatch inside match_ontology() to maintain zero-pydantic-at-module-level
- [02-03] Lazy import of VectorSearchService inside __init__ body to avoid pulling deps when backend=json
- [02-03] Always build keyword index even with embeddings backend (needed for fallback and legacy APIs)
- [02-03] builtins.__import__ patching for fallback tests since VectorSearchService is dynamically imported
- [03-01] Lazy VectorSearchService closure in annotation_expert factory (nonlocal singleton for deferred init)
- [03-01] Top-3 cell types by score, top-3 markers each, max 5 genes per query for semantic matching
- [03-01] Direct modalities dict assignment for semantic tool save (not store_modality)
- [03-01] OntologyMatch field access via .term/.ontology_id/.score (canonical field names)
- [03-02] HAS_VECTOR_SEARCH guard at module level (same pattern as HAS_ONTOLOGY_SERVICE)
- [03-02] Lazy _get_vector_service() closure inside factory (nonlocal singleton for deferred init)
- [03-02] Tissue tool requires HAS_VECTOR_SEARCH; disease tool requires HAS_ONTOLOGY_SERVICE (independent conditionals)
- [03-02] Disease tool routes through DiseaseOntologyService not VectorSearchService directly (Strangler Fig)
- [03-02] AnalysisStep requires code_template, imports, parameter_schema as mandatory fields
- [04-01] Min-max normalization for cross-encoder scores (per-query relative, not cross-query comparable)
- [04-01] Resolved-flag sentinel (_reranker_resolved) to distinguish "not checked" from "checked and None"
- [04-01] Reranking only in match_ontology(), not query() (keeps lower-level API predictable)
- [04-01] COHERE_RERANK_MODEL env var override for model flexibility
- [04-01] Edge case guard: skip reranking for single-document results (no model loaded)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-19 (Phase 04 plan 01 execution)
Stopped at: Completed 04-01-PLAN.md
Resume file: .planning/phases/04-performance/04-01-SUMMARY.md
