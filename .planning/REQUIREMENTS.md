# Requirements: Vector Search for Lobster AI

**Defined:** 2026-02-17
**Core Value:** Agents can semantically match any biomedical term to the correct ontology concept with calibrated confidence scores, using zero configuration out of the box.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [x] **INFRA-01**: VectorSearchService orchestrates two-stage search pipeline (embed -> search -> rerank -> return)
- [x] **INFRA-02**: VectorSearchConfig reads env vars and provides factory methods for backend, embeddings, and reranker
- [x] **INFRA-03**: BaseVectorBackend ABC defines add_documents, search, delete, count interface
- [x] **INFRA-04**: ChromaDB backend implements BaseVectorBackend with PersistentClient and auto-download from S3
- [x] **INFRA-05**: FAISS backend implements BaseVectorBackend with in-memory IndexFlatL2 and L2-normalized vectors
- [x] **INFRA-06**: pgvector backend stub raises NotImplementedError with helpful message
- [x] **INFRA-07**: Switching LOBSTER_VECTOR_BACKEND env var changes backend with zero code changes
- [x] **INFRA-08**: All optional deps (chromadb, sentence-transformers, faiss-cpu, obonet) are import-guarded with helpful install messages

### Embeddings

- [x] **EMBED-01**: BaseEmbeddingProvider ABC defines embed_text and embed_batch interface
- [x] **EMBED-02**: SapBERT provider loads cambridgeltl/SapBERT-from-PubMedBERT-fulltext (768d) with lazy singleton
- [ ] **EMBED-03**: SentenceTransformers provider loads all-MiniLM-L6-v2 (384d) as general fallback
- [ ] **EMBED-04**: OpenAI provider uses text-embedding-3-small (1536d) with lazy client init
- [x] **EMBED-05**: No model downloads at import time — all loading happens on first use

### Reranking

- [x] **RANK-01**: CrossEncoderReranker uses cross-encoder/ms-marco-MiniLM-L-6-v2 with lazy loading
- [x] **RANK-02**: CohereReranker gracefully degrades without API key (logs warning, returns original order)
- [x] **RANK-03**: Reranker is optional — search works without reranking if reranker set to "none"

### Search

- [x] **SRCH-01**: VectorSearchService.match_ontology(term, ontology, k) returns List[OntologyMatch] with confidence scores
- [x] **SRCH-02**: Two-stage pipeline: embed query -> search backend (oversample k*4) -> rerank -> return top-k
- [x] **SRCH-03**: ONTOLOGY_COLLECTIONS mapping resolves aliases ("disease" -> "mondo", "tissue" -> "uberon", "cell_type" -> "cell_ontology")
- [x] **SRCH-04**: Query results include ontology term IDs (MONDO:XXXX, UBERON:XXXX, CL:XXXX), names, and confidence scores

### Schemas

- [x] **SCHM-01**: SearchResult, OntologyMatch, LiteratureMatch, SearchResponse Pydantic models defined in lobster/core/schemas/search.py
- [x] **SCHM-02**: SearchBackend, EmbeddingProvider, RerankerType enums defined
- [x] **SCHM-03**: OntologyMatch is compatible with existing DiseaseMatch via helper converter

### Ontology Graph

- [x] **GRPH-01**: load_ontology_graph() parses OBO files via obonet into NetworkX MultiDiGraph with @lru_cache
- [x] **GRPH-02**: get_neighbors(graph, term_id, depth) returns parent/child/sibling terms
- [x] **GRPH-03**: OBO_URLS mapping covers MONDO, Uberon, and Cell Ontology

### Data Pipeline

- [ ] **DATA-01**: Build script (scripts/build_ontology_embeddings.py) parses OBO files and generates SapBERT embeddings
- [ ] **DATA-02**: Build script embeds definition + primary label per term (not synonyms separately) to avoid duplication
- [ ] **DATA-03**: Build script outputs ChromaDB collections with metadata (term_id, name, synonyms, namespace, is_obsolete)
- [ ] **DATA-04**: Build script produces tarballs for S3 upload (mondo_sapbert_768.tar.gz, uberon_sapbert_768.tar.gz, cell_ontology_sapbert_768.tar.gz)
- [ ] **DATA-05**: ChromaDB backend auto-downloads tarballs from S3 on first use to ~/.lobster/ontology_cache/
- [ ] **DATA-06**: Tarballs hosted on S3 at s3://lobster-ontology-data/v1/

### Disease Service Migration

- [x] **MIGR-01**: DiseaseOntologyService branches on config.backend field ("json" = keyword, "embeddings" = vector search)
- [x] **MIGR-02**: When backend="embeddings", match_disease() delegates to VectorSearchService.match_ontology("mondo")
- [x] **MIGR-03**: _convert_ontology_match() maps OntologyMatch to existing DiseaseMatch schema
- [x] **MIGR-04**: Keyword matching remains as fallback when vector deps not installed (with explicit logger.warning)
- [x] **MIGR-05**: DiseaseStandardizationService silent fallback fixed with logger.warning on ImportError
- [x] **MIGR-06**: Duplicate disease_ontology.json deleted from lobster/config/ (canonical stays in lobster-metadata package)

### Agent Integration

- [x] **AGNT-01**: annotation_expert gains annotate_cell_types_semantic tool that queries Cell Ontology via VectorSearchService
- [x] **AGNT-02**: Semantic annotation uses marker gene signatures as query text ("Cluster 0: high CD3D, CD3E, CD8A")
- [x] **AGNT-03**: Existing annotate_cell_types tool remains unchanged (new tool augments, does not replace)
- [x] **AGNT-04**: metadata_assistant gains standardize_tissue_term tool using VectorSearchService.match_ontology("uberon")
- [x] **AGNT-05**: metadata_assistant gains standardize_disease_term tool using DiseaseOntologyService.match_disease()
- [x] **AGNT-06**: All new tools follow 3-tuple return pattern (result, stats, AnalysisStep) with ir mandatory

### Cloud Handoff

- [ ] **CLOD-01**: Cloud-hosted ChromaDB handoff spec written for vector.omics-os.com deployment
- [ ] **CLOD-02**: Handoff spec includes architecture (ChromaDB server mode, auth, prewarmed indexes), API design, and deployment steps

### Testing

- [x] **TEST-01**: Unit tests for backends (ChromaDB, FAISS, pgvector stub) with mocked deps
- [x] **TEST-02**: Unit tests for embedding providers (SapBERT, MiniLM, OpenAI) with mocked model loading
- [x] **TEST-03**: Unit tests for rerankers (cross-encoder, Cohere) with mocked clients
- [x] **TEST-04**: Unit tests for VectorSearchService orchestration, caching, config-driven switching
- [x] **TEST-05**: Unit tests for config env var parsing and factory methods
- [x] **TEST-06**: Unit tests for DiseaseOntologyService Phase 2 backend swap branching
- [x] **TEST-07**: Integration test with small real ChromaDB (embed -> search -> rerank full pipeline)
- [x] **TEST-08**: All tests use @pytest.mark.skipif for optional deps

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Search Enhancements

- **SRCH-V2-01**: Hybrid matching — boost exact keyword matches alongside embedding similarity
- **SRCH-V2-02**: Confidence thresholds — auto-accept >=0.9, warn 0.7-0.9, manual review <0.7
- **SRCH-V2-03**: Batch review workflow — CSV export for low-confidence matches with top-3 alternatives
- **SRCH-V2-04**: Literature semantic search via FAISS (SearchResponse for publications)

### Ontology Enhancements

- **GRPH-V2-01**: Cross-ontology linking (MONDO disease -> affected Uberon tissues)
- **GRPH-V2-02**: OLS API fallback for missing/stale terms
- **GRPH-V2-03**: Automatic quarterly cache updates via CI/CD
- **GRPH-V2-04**: Cross-references (UMLS, MeSH, ICD codes) in search metadata

### Cloud

- **CLOD-V2-01**: Deploy ChromaDB server at vector.omics-os.com
- **CLOD-V2-02**: Cognito-authenticated API access for cloud users
- **CLOD-V2-03**: Prewarmed indexes for all 3 ontologies

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Custom ontology upload UI | Scope creep — most users need standard 3 ontologies. CLI script documented for power users |
| Multi-language support (XLM-RoBERTa) | Lobster is English-only. Defer to international market expansion |
| Real-time OBO parsing at startup | Anti-pattern — 10-30s latency. Pre-build embeddings offline |
| Bundled ontology data in PyPI package | Bloats package by 280MB. Use lazy download from S3 |
| pgvector full implementation | ChromaDB sufficient for 60K vectors. Implement when PostgreSQL consolidation happens |
| Editing pyproject.toml | Dependencies documented above for human review. Hard rule. |
| Replacing existing annotate_cell_types | New tool augments, old stays for backward compatibility |
| Biomedical cross-encoder fine-tuning | ms-marco sufficient for v1. Fine-tune if precision insufficient |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 5 | Complete |
| INFRA-06 | Phase 5 | Complete |
| INFRA-07 | Phase 5 | Complete |
| INFRA-08 | Phase 1 | Complete |
| EMBED-01 | Phase 1 | Complete |
| EMBED-02 | Phase 1 | Complete |
| EMBED-03 | Phase 6 | Pending |
| EMBED-04 | Phase 6 | Pending |
| EMBED-05 | Phase 1 | Complete |
| RANK-01 | Phase 4 | Complete |
| RANK-02 | Phase 4 | Complete |
| RANK-03 | Phase 4 | Complete |
| SRCH-01 | Phase 2 | Complete |
| SRCH-02 | Phase 2 | Complete |
| SRCH-03 | Phase 2 | Complete |
| SRCH-04 | Phase 2 | Complete |
| SCHM-01 | Phase 1 | Complete |
| SCHM-02 | Phase 1 | Complete |
| SCHM-03 | Phase 2 | Complete |
| GRPH-01 | Phase 2 | Complete |
| GRPH-02 | Phase 2 | Complete |
| GRPH-03 | Phase 2 | Complete |
| DATA-01 | Phase 6 | Pending |
| DATA-02 | Phase 6 | Pending |
| DATA-03 | Phase 6 | Pending |
| DATA-04 | Phase 6 | Pending |
| DATA-05 | Phase 6 | Pending |
| DATA-06 | Phase 6 | Pending |
| MIGR-01 | Phase 2 | Complete |
| MIGR-02 | Phase 2 | Complete |
| MIGR-03 | Phase 2 | Complete |
| MIGR-04 | Phase 2 | Complete |
| MIGR-05 | Phase 2 | Complete |
| MIGR-06 | Phase 2 | Complete |
| AGNT-01 | Phase 3 | Complete |
| AGNT-02 | Phase 3 | Complete |
| AGNT-03 | Phase 3 | Complete |
| AGNT-04 | Phase 3 | Complete |
| AGNT-05 | Phase 3 | Complete |
| AGNT-06 | Phase 3 | Complete |
| CLOD-01 | Phase 6 | Pending |
| CLOD-02 | Phase 6 | Pending |
| TEST-01 | Phase 5 | Complete |
| TEST-02 | Phase 4 | Complete |
| TEST-03 | Phase 4 | Complete |
| TEST-04 | Phase 2 | Complete |
| TEST-05 | Phase 2 | Complete |
| TEST-06 | Phase 2 | Complete |
| TEST-07 | Phase 2 | Complete |
| TEST-08 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 54 total
- Mapped to phases: 54
- Unmapped: 0

---
*Requirements defined: 2026-02-17*
*Last updated: 2026-02-17 after roadmap creation with 100% coverage*
