# Technology Stack

**Project:** Vector Search for Lobster AI
**Researched:** 2026-02-17

## Recommended Stack

### Core Vector Database
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **ChromaDB** | 1.5.0 | Local persistent vector storage + similarity search | Industry standard for <100K vectors, zero cost, proven in SRAgent production, PersistentClient for local persistence, simple 4-function API, Apache 2.0 license, built-in metadata filtering |

**Rationale**: ChromaDB 1.5.0 (Feb 2026) is the current stable release with production-ready persistent storage. For 60K vectors (~30K MONDO + ~30K Uberon + ~5K Cell Ontology), local persistence via `chromadb.PersistentClient()` is sufficient and eliminates API costs. SRAgent validated this pattern at scale. Client-server mode (`chroma run`) available for future cloud deployment at vector.omics-os.com.

**Confidence**: HIGH (verified via official PyPI, SRAgent reference, ChromaDB docs)

### Embedding Models
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **SapBERT** (cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | 110M params | Default biomedical entity linking | SOTA on 6 MEL benchmarks, trained on UMLS 2020AA (4M+ synonym pairs), 768d embeddings, offline/free, one-model-for-all biomedical entities |
| **sentence-transformers** | 5.2.3 | Embedding inference framework | De facto standard for embedding models, HuggingFace integration, 15K+ pretrained models, supports SapBERT via AutoModel |

**Rationale**: SapBERT is purpose-built for biomedical entity linking (Liu et al., NAACL 2021). It outperforms BioBERT, SciBERT, and PubMedBERT on entity-level tasks by learning fine-grained synonym relationships. 768d output matches BERT-base, balancing quality and storage. sentence-transformers 5.2.3 (Feb 2026) provides the inference engine. Alternative: `all-MiniLM-L6-v2` (384d, general purpose, smaller) for non-biomedical text; OpenAI `text-embedding-3-small` (cloud, API cost) if users BYOK.

**Confidence**: HIGH (SapBERT paper, HuggingFace model card, sentence-transformers official PyPI)

### Reranking (Two-Stage Retrieval)
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **cross-encoder/ms-marco-MiniLM-L6-v2** | — | Cross-encoder reranking | Best balance: NDCG@10=74.30, MRR@10=39.01, 1800 docs/sec, trained on real Bing queries, zero API cost, CPU-friendly (~1-5s for 100 candidates) |

**Rationale**: Two-stage retrieval (embedding search → cross-encoder rerank) is industry standard for accuracy. MS MARCO MiniLM-L6-v2 is explicitly recommended for semantic search by sentence-transformers docs. No domain-specific biomedical cross-encoder exists as of Feb 2026; general reranker works well for ontology matching (term similarity, not document retrieval). Alternative: `BAAI/bge-reranker-base` (newer, larger) if latency tolerant.

**Confidence**: MEDIUM (official sentence-transformers recommendation, but no biomedical-specific reranker verified)

### Ontology Parsing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **obonet** | 1.1.1 | Parse OBO files into NetworkX graphs | Industry standard for OBO parsing (MONDO, Uberon, Cell Ontology), supports v1.2 & v1.4 specs, clean NetworkX integration, lightweight |
| **NetworkX** | 3.6.1 | Graph traversal (ancestors, descendants, siblings) | Standard Python graph library, DAG verification, shortest path algorithms, well-tested for ontology navigation |

**Rationale**: obonet 1.1.1 (Mar 2025) is current stable. All three target ontologies (MONDO, Uberon, Cell Ontology) ship as OBO files. obonet converts them to `networkx.MultiDiGraph`, enabling parent/child/sibling traversal with standard graph algorithms. SRAgent validated this pattern for tissue ontology matching. Alternative: `pronto` (Python OBO parser) if OWL support needed later.

**Confidence**: HIGH (official PyPI, OBO specification compliance, SRAgent validation)

### Python Dependencies
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **transformers** | ≥4.47.0 | Load SapBERT via AutoModel | Required for SapBERT inference |
| **torch** | ≥2.5.0 | PyTorch backend for embeddings | Required for sentence-transformers and SapBERT |
| **numpy** | ≥1.23.0 | Vector operations | Already in lobster-ai core deps |
| **scikit-learn** | ≥1.3.0 | Cosine similarity fallback | Already in lobster-ai core deps |

**Confidence**: HIGH (versions match lobster-ai existing dependencies, transformer ecosystem standard)

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **Vector DB** | ChromaDB | pgvector (PostgreSQL extension) | Adds PostgreSQL dependency, overkill for 60K vectors, more ops complexity. Stub for future if cloud DB consolidation needed. |
| **Vector DB** | ChromaDB | Qdrant | Rust binary dependency, heavier for local use, no advantage at 60K scale. Consider if scaling to 10M+ vectors. |
| **Vector DB** | ChromaDB | Weaviate | Requires Docker/separate service, too heavy for local default. Consider for cloud `vector.omics-os.com`. |
| **Embeddings** | SapBERT | BiomedBERT (PubMedBERT) | General biomedical language model, not specialized for entity linking. SapBERT trained on UMLS synonyms specifically. |
| **Embeddings** | SapBERT | all-MiniLM-L6-v2 (384d) | General purpose, smaller (384d vs 768d), but loses biomedical domain knowledge. Keep as fallback for non-bio text. |
| **Embeddings** | SapBERT | OpenAI text-embedding-3-small | Requires API key, costs $0.00002/1K tokens, cloud dependency. Support as BYOK option, not default. |
| **Reranker** | ms-marco-MiniLM-L6-v2 | BAAI/bge-reranker-large | Larger model (560M params), slower inference, diminishing returns for ontology matching (not document retrieval). |
| **Reranker** | Cross-encoder | No reranking | Embedding-only search has ~10-15% worse accuracy. Two-stage retrieval is industry best practice. |
| **OBO Parser** | obonet | pronto | Heavier dependency, OWL support unneeded. Consider if RDF/OWL ontologies added later. |

## Installation

```bash
# Core vector search dependencies (optional extra)
pip install lobster-ai[vector-search]

# Expands to:
pip install \
  chromadb>=1.5.0 \
  sentence-transformers>=5.2.3 \
  transformers>=4.47.0 \
  torch>=2.5.0 \
  obonet>=1.1.1 \
  networkx>=3.6.1
```

**Import guards**: All imports wrapped with helpful error messages:
```python
try:
    import chromadb
except ImportError:
    raise ImportError(
        "ChromaDB not installed. Install with: pip install lobster-ai[vector-search]"
    )
```

**Lazy loading**: No embeddings loaded at import time. Models load on first `VectorSearchService.search()` call.

## Backend Factory Pattern

Environment variable switches backend without code changes:

```bash
# Default: ChromaDB local persistent
export LOBSTER_VECTOR_BACKEND=chromadb  # or unset

# Future: pgvector (stub, not implemented)
export LOBSTER_VECTOR_BACKEND=pgvector
export LOBSTER_POSTGRES_URI=postgresql://user:pass@host/db

# Future: cloud service
export LOBSTER_VECTOR_BACKEND=cloud
export LOBSTER_CLOUD_VECTOR_URL=https://vector.omics-os.com
```

**Factory location**: `lobster/services/search/vector_search_service.py`
```python
def _get_backend(backend: str = None) -> VectorBackend:
    backend = backend or os.getenv("LOBSTER_VECTOR_BACKEND", "chromadb")
    if backend == "chromadb":
        return ChromaDBBackend()
    elif backend == "pgvector":
        return PgvectorBackend()  # stub
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

## Ontology Data Pipeline

**Build-time** (one-time setup):
1. Download OBO files (MONDO, Uberon, Cell Ontology) from official sources
2. Parse via obonet → NetworkX graphs
3. Extract term definitions + metadata
4. Generate SapBERT embeddings (768d vectors)
5. Store in ChromaDB collections (`mondo`, `uberon`, `cell_ontology`)
6. Export as tarballs → host on S3 at `https://ontology.omics-os.com/`

**Runtime** (auto-download on first use):
1. Check `~/.lobster/ontology_cache/{ontology}.tar.gz`
2. If missing, download from S3 + extract
3. Load ChromaDB PersistentClient from extracted directory
4. Cache stays for subsequent runs

**Reference**: SRAgent `scripts/obo-embed.py` implements steps 1-5.

## Storage Estimates

| Ontology | Terms | Embeddings (768d × float32) | Metadata | ChromaDB Storage |
|----------|-------|----------------------------|----------|------------------|
| MONDO | ~30,000 | 30K × 768 × 4 bytes = ~92 MB | ~10 MB | ~105 MB |
| Uberon | ~30,000 | 30K × 768 × 4 bytes = ~92 MB | ~10 MB | ~105 MB |
| Cell Ontology | ~5,000 | 5K × 768 × 4 bytes = ~15 MB | ~2 MB | ~18 MB |
| **Total** | **~65,000** | **~199 MB** | **~22 MB** | **~228 MB** |

Add ~50 MB ChromaDB index overhead → **~280 MB total** per user at `~/.lobster/ontology_cache/`.

Tarball sizes: ~70 MB (MONDO), ~70 MB (Uberon), ~12 MB (Cell Ontology) = **~152 MB download** on first use.

## Performance Benchmarks (Estimated)

| Operation | Time (CPU) | Time (GPU) | Notes |
|-----------|------------|------------|-------|
| Load ChromaDB collection | ~500ms | N/A | One-time per session |
| Load SapBERT model | ~2s | ~1s | One-time per session |
| Embed query (1 term) | ~50ms | ~10ms | Batch for efficiency |
| Vector search (60K vectors) | ~100ms | N/A | ChromaDB HNSW index |
| Rerank top 100 candidates | ~1-5s | ~500ms | Cross-encoder bottleneck |
| **End-to-end search** | **~3-7s** | **~1-2s** | Including reranking |

Optimization: Batch embed multiple terms (10 queries → ~200ms instead of 500ms).

Reference: SRAgent reports similar timings for Uberon ontology matching.

## Cloud Deployment Plan (vector.omics-os.com)

**Not implemented in this project** — written as handoff spec for `lobster-cloud` team:

1. **Option A: ChromaDB Server Mode**
   - Run `chroma run --path /data --port 8000` in ECS Fargate
   - Client connects via `chromadb.HttpClient("https://vector.omics-os.com")`
   - Pros: Simple, same API as local
   - Cons: Immature server mode, auth story unclear

2. **Option B: FastAPI + ChromaDB PersistentClient**
   - Custom FastAPI service wrapping ChromaDB
   - `/search` endpoint with authentication
   - Pros: Full control, integrates with existing Cognito
   - Cons: More code to maintain

3. **Option C: Chroma Cloud (SaaS)**
   - Hosted ChromaDB ($5 free credits, then $0.10/GB/month)
   - Pros: Zero ops, serverless, scalable
   - Cons: Vendor lock-in, cost at scale

**Recommendation**: Start with Option B (FastAPI wrapper) for control and auth integration. Evaluate Chroma Cloud if scaling beyond 10M vectors.

## Stack Evolution Path

| Phase | Backend | Embeddings | Why |
|-------|---------|-----------|-----|
| **v1 (MVP)** | ChromaDB local | SapBERT | Proven stack, zero cost, offline-capable |
| **v2 (Cloud)** | ChromaDB server or FastAPI wrapper | SapBERT | Shared vectors across users, reduce client-side download |
| **v3 (Scale)** | Pgvector or Chroma Cloud | SapBERT + custom fine-tuned | If scaling to 10M+ entities or custom domain ontologies |

## Sources

### HIGH Confidence (Official Documentation)
- ChromaDB 1.5.0: https://pypi.org/project/chromadb/ (PyPI, Feb 2026)
- sentence-transformers 5.2.3: https://pypi.org/project/sentence-transformers/ (PyPI, Feb 2026)
- SapBERT: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext (HuggingFace model card)
- obonet 1.1.1: https://pypi.org/project/obonet/ (PyPI, Mar 2025)
- NetworkX 3.6.1: https://pypi.org/project/networkx/ (PyPI, Dec 2025)
- Cross-encoder rerankers: https://github.com/UKPLab/sentence-transformers (official docs)

### MEDIUM Confidence (Reference Implementations)
- SRAgent production patterns: /Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/SRAgent/tools/vector_db.py
- SRAgent OBO embedding script: /Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/scripts/obo-embed.py
- ChromaDB deployment modes: https://docs.trychroma.com/ (partial documentation, full deployment guide not accessed)

### Context (Project Decisions)
- Lobster AI PROJECT.md: /Users/tyo/omics-os/lobster/.planning/PROJECT.md
- SRAgent README (Uberon/MONDO agent validation): /Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/README.md

## Notes

1. **No pyproject.toml edits**: Dependencies documented here for human approval. Add `vector-search` extra after review.

2. **PEP 420 namespace**: All imports must be lazy and guarded. No module-level ChromaDB initialization.

3. **Backward compatibility**: Existing DiseaseOntologyService keyword matching stays as fallback if vector backend unavailable.

4. **SapBERT vs BiomedBERT**: BiomedBERT is a general biomedical language model. SapBERT fine-tuned specifically on UMLS synonym pairs for entity linking. SapBERT is the correct choice.

5. **Why not OpenAI embeddings**: API cost ($0.02/1M tokens) and cloud dependency. SapBERT is free, offline, and biomedical-specific. OpenAI supported as BYOK option.

6. **Reranker limitations**: No biomedical-specific cross-encoder exists as of Feb 2026. MS MARCO MiniLM-L6-v2 trained on general web search but effective for term similarity. Consider fine-tuning if accuracy insufficient.

7. **Alternative to ChromaDB**: Qdrant and Weaviate are production-ready but heavier. pgvector adds PostgreSQL dependency. ChromaDB's simplicity wins for 60K vector scale.

8. **Storage optimization**: 768d embeddings × float32 = 3KB per term. Consider float16 quantization (1.5KB/term) if storage becomes issue, but loses precision.

9. **GPU acceleration**: Optional. SapBERT inference is ~5x faster on GPU, but CPU is acceptable for interactive use (~50ms/query). Cross-encoder bottleneck regardless.

10. **Version pinning**: Minimum versions specified (`>=`). Lock files will pin exact versions. ChromaDB 1.5.x series is stable; avoid 2.0 prereleases.
