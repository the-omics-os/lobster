# Roadmap: Vector Search for Lobster AI

## Overview

This roadmap delivers semantic vector search infrastructure for Lobster AI, replacing hardcoded keyword matching with biomedical embedding-based ontology matching. The journey begins with foundational infrastructure (ChromaDB + SapBERT embeddings), progresses through service integration and agent tooling, adds performance optimization via two-stage retrieval, enables backend flexibility with FAISS/pgvector stubs, and completes with automated data pipeline and cloud handoff. By the end, agents can semantically match any biomedical term to the correct ontology concept (diseases, tissues, cell types) with calibrated confidence scores, using zero configuration out of the box.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** - Core vector search infrastructure with ChromaDB + SapBERT embeddings
- [ ] **Phase 2: Service Integration** - SemanticSearchService with 3-tuple pattern, ontology graph, and DiseaseOntologyService migration
- [ ] **Phase 3: Agent Tooling** - Cell type, tissue, and disease standardization tools for annotation_expert and metadata_assistant
- [ ] **Phase 4: Performance** - Two-stage retrieval with cross-encoder reranking for 10-15% precision gain
- [ ] **Phase 5: Scalability** - Alternative backends (FAISS, pgvector stub) and backend factory pattern
- [ ] **Phase 6: Automation** - Data pipeline for prebuilt ontology embeddings and cloud handoff spec

## Phase Details

### Phase 1: Foundation
**Goal**: Core vector search infrastructure works locally with persistent ChromaDB and SapBERT embeddings
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-08, EMBED-01, EMBED-02, EMBED-05, SCHM-01, SCHM-02
**Success Criteria** (what must be TRUE):
  1. Developer can query "heart attack" and get ranked matches like "myocardial infarction" with confidence scores
  2. ChromaDB cache persists across process restarts (instant second query, no re-embedding)
  3. `lobster --help` executes in <500ms (no module-level model loading)
  4. Import failures show helpful "pip install lobster-ai[vector-search]" messages
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md -- Schemas, enums, and ABCs (BaseVectorBackend, BaseEmbedder)
- [ ] 01-02-PLAN.md -- SapBERTEmbedder and ChromaDBBackend implementations
- [ ] 01-03-PLAN.md -- VectorSearchConfig, VectorSearchService, and test suite (TDD)

### Phase 2: Service Integration
**Goal**: SemanticSearchService integrates with Lobster patterns and DiseaseOntologyService completes Strangler Fig migration
**Depends on**: Phase 1
**Requirements**: SRCH-01, SRCH-02, SRCH-03, SRCH-04, SCHM-03, GRPH-01, GRPH-02, GRPH-03, MIGR-01, MIGR-02, MIGR-03, MIGR-04, MIGR-05, MIGR-06, TEST-04, TEST-05, TEST-06, TEST-07
**Success Criteria** (what must be TRUE):
  1. DiseaseOntologyService.match_disease() returns semantic matches when backend="embeddings" with zero API changes
  2. All service methods return (result, stats, AnalysisStep) 3-tuple with provenance tracking
  3. Ontology graph provides parent/child/sibling relationships for MONDO, Uberon, Cell Ontology terms
  4. Keyword fallback works with explicit logger.warning when vector deps not installed
  5. Duplicate disease_ontology.json removed from lobster/config/ (canonical stays in lobster-metadata)
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md -- Ontology graph module (load_ontology_graph, get_neighbors, OBO_URLS)
- [ ] 02-02-PLAN.md -- match_ontology() on VectorSearchService with ONTOLOGY_COLLECTIONS alias resolution (TDD)
- [ ] 02-03-PLAN.md -- DiseaseOntologyService Strangler Fig migration, silent fallback fix, duplicate config deletion

### Phase 3: Agent Tooling
**Goal**: Agents can use semantic search for cell type annotation and tissue/disease standardization
**Depends on**: Phase 2
**Requirements**: AGNT-01, AGNT-02, AGNT-03, AGNT-04, AGNT-05, AGNT-06, TEST-08
**Success Criteria** (what must be TRUE):
  1. annotation_expert has annotate_cell_types_semantic tool that queries Cell Ontology with marker gene signatures
  2. metadata_assistant has standardize_tissue_term and standardize_disease_term tools
  3. Existing annotate_cell_types tool remains unchanged (new tool augments, doesn't replace)
  4. All new agent tools return ir for provenance tracking
  5. Agent can query "CD3D+/CD8A+ cluster" and get "cytotoxic T cell (CL:0000084)" as top match
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Performance
**Goal**: Two-stage retrieval with cross-encoder reranking delivers 10-15% precision improvement
**Depends on**: Phase 3
**Requirements**: RANK-01, RANK-02, RANK-03, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Search pipeline reranks top-100 candidates with cross-encoder to return top-10 results
  2. "heart attack" ranks "myocardial infarction" higher than generic "heart disease" after reranking
  3. Reranking can be disabled via config (rerank="none") without breaking search
  4. Query latency stays <2s for 60K term corpus with reranking enabled
  5. Cohere reranker gracefully degrades without API key (logs warning, returns original order)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Scalability
**Goal**: Backend factory pattern enables swapping ChromaDB/FAISS/pgvector via environment variable
**Depends on**: Phase 4
**Requirements**: INFRA-05, INFRA-06, INFRA-07, TEST-01
**Success Criteria** (what must be TRUE):
  1. Setting LOBSTER_VECTOR_BACKEND=faiss switches to FAISS backend with zero code changes
  2. FAISS backend uses L2-normalized vectors with IndexFlatL2 in-memory index
  3. pgvector backend stub raises NotImplementedError with helpful "Coming in v2.0" message
  4. Switching backends requires no changes to service layer (BaseVectorBackend abstraction works)
  5. Backend selection logic is testable with mocked backends
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Automation
**Goal**: Offline data pipeline builds prebuilt embeddings and cloud deployment handoff spec complete
**Depends on**: Phase 5
**Requirements**: EMBED-03, EMBED-04, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, CLOD-01, CLOD-02
**Success Criteria** (what must be TRUE):
  1. Build script (scripts/build_ontology_embeddings.py) parses OBO files and generates SapBERT embeddings
  2. ChromaDB tarballs for MONDO, Uberon, Cell Ontology hosted on S3 at s3://lobster-ontology-data/v1/
  3. ChromaDB backend auto-downloads tarballs to ~/.lobster/ontology_cache/ on first use with progress bar
  4. First query after fresh install completes in <60s (download time) vs 10-15min (embedding time)
  5. Cloud-hosted ChromaDB handoff spec written for vector.omics-os.com deployment (architecture, auth, API design)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/3 | Planning complete | - |
| 2. Service Integration | 0/? | Not started | - |
| 3. Agent Tooling | 0/? | Not started | - |
| 4. Performance | 0/? | Not started | - |
| 5. Scalability | 0/? | Not started | - |
| 6. Automation | 0/? | Not started | - |
