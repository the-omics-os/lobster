# Phase 1: Foundation - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Core vector search infrastructure with ChromaDB + SapBERT embeddings. A thin, general-purpose semantic matching engine that agents (metadata_assistant, annotation_expert, etc.) call programmatically. No domain logic in this layer — just embed, store, search, return ranked results. Domain-aware features (ontology ID extraction, synonym expansion, confidence calibration) belong in Phase 2 services.

</domain>

<decisions>
## Implementation Decisions

### Query API design
- Thin vector layer only — embed query, search collection, return ranked results. No domain logic.
- Return top 5 results by default
- Support both single-term `query("heart attack")` and batch `query_batch(["heart attack", "lung cancer"])` from the start
- Return shape: list of match dicts with term, ontology_id, score, metadata — flat and simple
- Callers (Phase 2 services) add domain logic on top

### Embedding behavior
- SapBERT as the embedding model from Phase 1 (not a simpler model first)
- Pluggable embedding layer from the start — BaseEmbedder interface with SapBERT as default implementation
- Lazy model loading (first query triggers load) to meet <500ms CLI startup requirement
- When SapBERT/dependencies aren't installed: raise ImportError with helpful "pip install lobster-ai[vector-search]" message. No fallback to keyword matching.
- Primary consumer driving design: metadata_assistant for harmonization and standardization

### Collection structure
- One ChromaDB collection per ontology (mondo, uberon, cell_ontology) — clean isolation, independent updates
- Minimal metadata per document: ontology_id + canonical_name only. Rich details fetched from ontology graph in Phase 2.
- Persistence location: `~/.lobster/vector_store/` (user home, shared across workspaces)
- Collection naming: versioned pattern — e.g., `mondo_v2024_01`, `uberon_v2024_01`. Allows side-by-side upgrades.

### Confidence scoring
- Return raw cosine similarity scores (0-1). No calibration or tier mapping at core layer.
- No minimum threshold filtering — return all top_k results, callers apply their own thresholds
- Include distance metric name in result output (e.g., "cosine") for transparency and debugging
- Scores comparable within a single query's results only, not across different queries

### Claude's Discretion
- Exact BaseEmbedder interface design
- ChromaDB client initialization pattern (in-process vs client-server)
- Batch query implementation strategy (parallel vs sequential embedding)
- Error handling patterns for ChromaDB connection issues
- Test fixture design for vector search unit tests

</decisions>

<specifics>
## Specific Ideas

- Foundation is driven by metadata_assistant needs — the primary consumer for harmonization and standardization
- Pluggable embedder interface should make Phase 5 (FAISS/pgvector) backend swapping straightforward
- Versioned collection naming supports Phase 6 (data pipeline) prebuilt embedding tarballs

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-02-18*
