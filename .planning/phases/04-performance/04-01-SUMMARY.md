---
phase: 04-performance
plan: 01
subsystem: vector-search
tags: [reranking, cross-encoder, cohere, sentence-transformers, vector-search]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: BaseEmbedder ABC, VectorSearchConfig factory pattern, VectorSearchService
  - phase: 02-service-integration
    provides: match_ontology() with 4x oversampling, ONTOLOGY_COLLECTIONS
provides:
  - BaseReranker ABC with rerank(query, documents, top_k) contract
  - CrossEncoderReranker (ms-marco-MiniLM-L-6-v2, lazy loading)
  - CohereReranker (rerank-v4.0-pro, graceful degradation)
  - normalize_scores() min-max helper for [0,1] score mapping
  - VectorSearchConfig.create_reranker() factory method
  - LOBSTER_RERANKER env var support
  - Reranking step in match_ontology() between search and truncation
affects: [05-unit-test-coverage, agent-tooling, annotation-expert, metadata-assistant]

# Tech tracking
tech-stack:
  added: [sentence-transformers CrossEncoder (reused), cohere SDK v5.x (optional)]
  patterns: [BaseReranker ABC, lazy reranker loading, graceful degradation, resolved-flag sentinel, min-max score normalization]

key-files:
  created:
    - lobster/core/vector/rerankers/__init__.py
    - lobster/core/vector/rerankers/base.py
    - lobster/core/vector/rerankers/cross_encoder_reranker.py
    - lobster/core/vector/rerankers/cohere_reranker.py
  modified:
    - lobster/core/vector/config.py
    - lobster/core/vector/service.py
    - lobster/core/vector/__init__.py

key-decisions:
  - "Min-max normalization for cross-encoder scores (per-query relative, not cross-query comparable)"
  - "Resolved-flag sentinel (_reranker_resolved) to distinguish 'not checked' from 'checked and None'"
  - "Reranking only in match_ontology(), not query() (keeps lower-level API predictable)"
  - "COHERE_RERANK_MODEL env var override for model flexibility"
  - "Edge case guard: skip reranking for single-document results (no model loaded)"

patterns-established:
  - "BaseReranker ABC: parallel to BaseEmbedder/BaseVectorBackend for consistent extension"
  - "Graceful degradation: log warning + return original order when optional API unavailable"
  - "Resolved-flag sentinel pattern for lazy init with None as valid cached value"

requirements-completed: [RANK-01, RANK-02, RANK-03]

# Metrics
duration: 5min
completed: 2026-02-19
---

# Phase 4 Plan 01: Reranker Infrastructure Summary

**BaseReranker ABC with CrossEncoder and Cohere implementations, config-driven factory, and match_ontology() integration using min-max score normalization**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-19T07:22:40Z
- **Completed:** 2026-02-19T07:28:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- BaseReranker ABC with rerank() contract and normalize_scores() min-max helper
- CrossEncoderReranker with lazy model loading (zero new dependencies -- reuses sentence-transformers)
- CohereReranker with graceful degradation when API key/package missing
- Config factory (create_reranker()) + LOBSTER_RERANKER env var + lazy __init__.py export
- match_ontology() reranking step wired between search and truncation
- Zero regressions: all 70 existing tests pass unchanged with default reranker=none

## Task Commits

Each task was committed atomically:

1. **Task 1: BaseReranker ABC + CrossEncoderReranker + CohereReranker** - `21fefec` (feat)
2. **Task 2: Config factory + service integration + __init__.py export** - `a93ada4` (feat)

## Files Created/Modified
- `lobster/core/vector/rerankers/__init__.py` - Empty package init (avoids circular imports per Pitfall 5)
- `lobster/core/vector/rerankers/base.py` - BaseReranker ABC + normalize_scores() helper
- `lobster/core/vector/rerankers/cross_encoder_reranker.py` - CrossEncoder reranker with lazy model loading
- `lobster/core/vector/rerankers/cohere_reranker.py` - Cohere API reranker with graceful degradation
- `lobster/core/vector/config.py` - Added reranker field, create_reranker() factory, LOBSTER_RERANKER env var
- `lobster/core/vector/service.py` - Added reranker DI, _get_reranker(), reranking step in match_ontology()
- `lobster/core/vector/__init__.py` - Added BaseReranker to __all__ and __getattr__ lazy exports

## Decisions Made
- Min-max normalization for cross-encoder scores: scores are only compared within a single query context, so per-query relative normalization is correct
- Resolved-flag sentinel pattern: _reranker_resolved bool distinguishes "not yet checked config" from "checked and reranker is None"
- Reranking in match_ontology() only, not query(): keeps the lower-level API predictable and scope tight
- COHERE_RERANK_MODEL env var allows overriding the default model (rerank-v4.0-pro)
- Single-document edge case: skip reranking entirely (no model loaded) since reranking one document adds latency for zero benefit

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Reranking is disabled by default (reranker=none). To enable:
- Cross-encoder: `export LOBSTER_RERANKER=cross_encoder` (uses existing sentence-transformers)
- Cohere: `export LOBSTER_RERANKER=cohere` and `export COHERE_API_KEY=your-key` (requires `pip install cohere`)

## Next Phase Readiness
- Reranker infrastructure complete; ready for unit test coverage (Phase 4 Plan 02)
- CrossEncoder and Cohere rerankers need mocked unit tests (TEST-03)
- All 70 existing tests pass with zero regressions

## Self-Check: PASSED

All 8 files verified present. Both commit hashes (21fefec, a93ada4) confirmed in git log.

---
*Phase: 04-performance*
*Completed: 2026-02-19*
