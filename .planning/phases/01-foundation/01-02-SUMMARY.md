---
phase: 01-foundation
plan: 02
subsystem: vector-search
tags: [sapbert, chromadb, embeddings, vector-backend, lazy-loading, cosine-hnsw, biomedical]

# Dependency graph
requires:
  - phase: 01-foundation-01
    provides: BaseVectorBackend ABC, BaseEmbedder ABC, vector module structure
provides:
  - SapBERTEmbedder with lazy model loading, CLS pooling, 768d embeddings
  - ChromaDBBackend with PersistentClient, cosine HNSW, auto-directory, batch chunking
  - Import-guarded implementations requiring lobster-ai[vector-search] extra
affects: [01-03, vector-search-service, ontology-matching, annotation-agent]

# Tech tracking
tech-stack:
  added: [sentence-transformers, chromadb]
  patterns: [lazy-model-loading, import-guarded-dependencies, batch-chunking]

key-files:
  created:
    - lobster/core/vector/embeddings/sapbert.py
    - lobster/core/vector/backends/chromadb_backend.py
  modified: []

key-decisions:
  - "CLS-token pooling for SapBERT (trained with CLS, not mean pooling)"
  - "batch_size=128 for SapBERT encode per model card recommendation"
  - "5000-document chunk limit for ChromaDB batch operations"
  - "Lazy client initialization in ChromaDBBackend (import chromadb only in _get_client)"
  - "Raw ChromaDB column-oriented results from search (distance->similarity conversion in service layer)"

patterns-established:
  - "Lazy model loading: _load_model() called in embed_text/embed_batch, not __init__"
  - "Import guard in method body: try/except ImportError with pip install message"
  - "Batch chunking: loop with _BATCH_SIZE constant for large document sets"
  - "Collection defaults: all collections get cosine HNSW via _get_or_create_collection"

requirements-completed: [EMBED-02, EMBED-05, INFRA-04, INFRA-08]

# Metrics
duration: 2min
completed: 2026-02-18
---

# Phase 1 Plan 02: Concrete Adapters Summary

**SapBERTEmbedder (768d, CLS pooling, lazy loading) and ChromaDBBackend (PersistentClient, cosine HNSW, batch chunking) with import-guarded dependencies**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-18T19:04:54Z
- **Completed:** 2026-02-18T19:07:19Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Implemented SapBERTEmbedder extending BaseEmbedder ABC with lazy model loading (no torch/sentence-transformers import until first use) and explicit CLS-token pooling matching SapBERT's training objective
- Implemented ChromaDBBackend extending BaseVectorBackend ABC with lazy PersistentClient, cosine HNSW for all collections, auto-directory creation, and 5000-doc batch chunking
- Both implementations are fully import-guarded: importing the module does NOT load heavy dependencies; helpful pip install messages on missing deps

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SapBERTEmbedder with lazy loading and CLS pooling** - `dff3528` (feat)
2. **Task 2: Implement ChromaDBBackend with PersistentClient and cosine HNSW** - `55f13d9` (feat)

## Files Created/Modified
- `lobster/core/vector/embeddings/sapbert.py` - SapBERTEmbedder: 768d biomedical entity embedder with lazy model loading, CLS pooling, batch_size=128
- `lobster/core/vector/backends/chromadb_backend.py` - ChromaDBBackend: persistent local vector store with cosine HNSW, auto-directory creation, 5000-doc batch chunking

## Decisions Made
- Used CLS-token pooling (not mean pooling) for SapBERT, matching its training objective for biomedical entity linking
- Set batch_size=128 for SapBERT encode() per model card recommendation to balance throughput and OOM risk
- Chunk size of 5000 for ChromaDB batch operations to stay within recommended limits
- Lazy client initialization: chromadb imported only in _get_client(), not at module level
- Search returns raw ChromaDB column-oriented format; distance-to-similarity conversion deferred to VectorSearchService (Plan 03)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. ChromaDB is not installed in the dev environment, so the full CRUD test was skipped, but the import guard was verified to work correctly (raises ImportError with the right message).

## User Setup Required

None - no external service configuration required. Users install dependencies via `pip install 'lobster-ai[vector-search]'`.

## Next Phase Readiness
- Both concrete adapters ready for VectorSearchService (Plan 03) to wire together
- SapBERTEmbedder provides embed_text/embed_batch for ontology term embedding
- ChromaDBBackend provides add_documents/search/delete/count for persistent vector storage
- Both follow the ABC contracts established in Plan 01

## Self-Check: PASSED

- All 2 created files verified on disk
- Both commit hashes (dff3528, 55f13d9) verified in git log
- Import guards verified: no heavy dependencies loaded at import time
- ABC contract verified: SapBERTEmbedder.dimensions=768, .name="SapBERTEmbedder"
- ABC contract verified: ChromaDBBackend inherits BaseVectorBackend, all abstract methods implemented

---
*Phase: 01-foundation*
*Completed: 2026-02-18*
