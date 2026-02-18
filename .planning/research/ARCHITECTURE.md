# Architecture Research

**Domain:** Biomedical Semantic Vector Search
**Researched:** 2026-02-17
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Agents     │  │   Services   │  │     Tools    │           │
│  │ (cell type   │  │  (disease    │  │  (metadata   │           │
│  │ annotation)  │  │  ontology)   │  │  assistants) │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                  │                   │
├─────────┴─────────────────┴──────────────────┴───────────────────┤
│                     Search Orchestration                         │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              SemanticSearchService                        │   │
│  │  • Embedding provider selection                           │   │
│  │  • Two-stage pipeline coordination                        │   │
│  │  • Backend routing (ChromaDB/FAISS/pgvector)              │   │
│  └───────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                        Core Components                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Embedding   │  │   Vector     │  │  Cross-      │           │
│  │  Provider    │  │   Backend    │  │  Encoder     │           │
│  │  (SapBERT)   │  │ (ChromaDB)   │  │  (Reranker)  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                  │                   │
├─────────┴─────────────────┴──────────────────┴───────────────────┤
│                     Knowledge Layer                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │            Ontology Graph (NetworkX)                     │    │
│  │  • OBO file parsing (obonet)                             │    │
│  │  • Hierarchical reasoning                                │    │
│  │  • Relationship traversal (is_a, part_of)                │    │
│  └──────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│                        Storage Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Vector     │  │  Ontology    │  │   Cache      │           │
│  │   Store      │  │  Files       │  │  (LRU)       │           │
│  │  (persist)   │  │  (.obo)      │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **SemanticSearchService** | Orchestrate search pipeline, backend selection, result formatting | Python class, 3-tuple return (result, stats, AnalysisStep) |
| **EmbeddingProvider** | Convert text to vectors (768d/384d/1536d) | SapBERT (HuggingFace transformers), MiniLM fallback, OpenAI API option |
| **VectorBackend** | Store/index/query embeddings, metadata filtering | ChromaDB client (default), FAISS indexes (alternative), pgvector stub |
| **CrossEncoder** | Rerank top-k candidates for precision | Sentence-transformers cross-encoder models (ms-marco, bge-reranker) |
| **OntologyGraph** | Parse OBO, provide hierarchical reasoning, traverse relationships | NetworkX MultiDiGraph via obonet, @lru_cache singleton |
| **IndexBuilder** | OBO → embeddings → ChromaDB tarballs → S3 hosting | Offline pipeline: embed with SapBERT, serialize ChromaDB, upload |
| **DataDownloader** | Auto-download prebuilt indexes on first use | HTTP GET from S3, extract to workspace, verify checksums |

## Recommended Project Structure

```
lobster/services/search/
├── semantic_search_service.py     # Main orchestration (3-tuple pattern)
├── embedding_provider.py          # Factory: SapBERT/MiniLM/OpenAI
├── backends/
│   ├── base.py                    # Abstract VectorBackend interface
│   ├── chromadb_backend.py        # ChromaDB client wrapper (default)
│   ├── faiss_backend.py           # FAISS index wrapper (alternative)
│   └── pgvector_backend.py        # PostgreSQL stub (future)
├── reranking/
│   ├── cross_encoder.py           # Two-stage reranking logic
│   └── models.py                  # Cross-encoder model registry
├── ontology/
│   ├── ontology_graph.py          # NetworkX wrapper, obonet parsing
│   ├── reasoning.py               # Hierarchical queries (ancestors, descendants)
│   └── loader.py                  # @lru_cache singleton, lazy loading
└── data/
    ├── index_builder.py           # Offline: OBO → ChromaDB pipeline
    ├── downloader.py              # Auto-download prebuilt indexes
    └── README.md                  # Index versioning, S3 URLs
```

### Structure Rationale

- **backends/:** Backend-agnostic design via factory pattern. ChromaDB default (developer-friendly, auto-embedding). FAISS for performance-critical workloads (GPU support, billions of vectors). pgvector stub for PostgreSQL integration (transactional guarantees, SQL joins).
- **reranking/:** Two-stage retrieval isolates cross-encoder logic. Cross-encoders are 10-100x slower than bi-encoders, so only run on top-k candidates (k=100 typical).
- **ontology/:** Knowledge graph layer separate from vector search. NetworkX enables graph algorithms (shortest path, subgraph extraction). OBO format standard across biomedical ontologies (Disease Ontology, Cell Ontology, Uberon).
- **data/:** Index building offline (not runtime). S3-hosted tarballs downloaded on first use (30-500MB per ontology). Checksum verification prevents corruption.

## Architectural Patterns

### Pattern 1: Two-Stage Retrieval (Bi-Encoder + Cross-Encoder)

**What:** Fast retrieval with bi-encoder (SapBERT), precise reranking with cross-encoder

**When to use:** Large ontology search (>10K terms), precision matters, latency <2s acceptable

**Trade-offs:**
- **Pros:** 10x better precision than bi-encoder alone, scalable to millions of terms, only 100 cross-encoder calls
- **Cons:** 2x latency vs. bi-encoder only, requires two models, more memory

**Example:**
```python
# Stage 1: Fast retrieval (bi-encoder) - 10ms for 100K terms
embedding = embedding_provider.embed("acute myocardial infarction")
candidates = vector_backend.search(embedding, top_k=100)  # Top 100

# Stage 2: Precise reranking (cross-encoder) - 500ms for 100 pairs
scores = cross_encoder.score_pairs(
    query="acute myocardial infarction",
    candidates=[c.text for c in candidates]
)
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
return reranked[:10]  # Final top 10
```

### Pattern 2: Backend Factory with Environment Variables

**What:** Select vector backend via env vars, lazy import to avoid hard dependencies

**When to use:** Library code (not application), optional dependencies, multi-backend support

**Trade-offs:**
- **Pros:** PEP 420 compliant (no forced deps), users choose backend, dev-friendly defaults
- **Cons:** Runtime import errors if mis-configured, testing requires all backends installed

**Example:**
```python
# backends/base.py
class VectorBackend(ABC):
    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int) -> List[SearchResult]: ...

# backends/__init__.py
def get_backend() -> VectorBackend:
    backend_type = os.getenv("LOBSTER_VECTOR_BACKEND", "chromadb")

    if backend_type == "chromadb":
        try:
            from lobster.services.search.backends.chromadb_backend import ChromaDBBackend
            return ChromaDBBackend()
        except ImportError:
            raise ImportError("pip install 'lobster-ai[chromadb]'")
    elif backend_type == "faiss":
        try:
            from lobster.services.search.backends.faiss_backend import FAISSBackend
            return FAISSBackend()
        except ImportError:
            raise ImportError("pip install 'lobster-ai[faiss]'")
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
```

### Pattern 3: Singleton Ontology Graph with LRU Cache

**What:** Load NetworkX graph once, cache parsed OBO files, lazy initialization

**When to use:** Large ontologies (>50K terms), repeated graph queries, memory is acceptable cost

**Trade-offs:**
- **Pros:** 1000x faster repeated access, avoids re-parsing OBO, thread-safe via @lru_cache
- **Cons:** ~100MB memory per ontology, stale data until process restart, not multi-process safe

**Example:**
```python
# ontology/loader.py
from functools import lru_cache
import obonet
import networkx as nx

@lru_cache(maxsize=8)  # Cache up to 8 ontologies
def load_ontology_graph(obo_path: str) -> nx.MultiDiGraph:
    """Load and cache OBO file as NetworkX graph."""
    return obonet.read_obo(obo_path)

# Usage
graph = load_ontology_graph("/path/to/doid.obo")  # First call: ~5s
graph = load_ontology_graph("/path/to/doid.obo")  # Cached: <1ms

# Hierarchical reasoning
ancestors = nx.ancestors(graph, "DOID:5844")  # myocardial infarction
descendants = nx.descendants(graph, "DOID:4")  # disease
```

### Pattern 4: Prebuilt Index Distribution via S3

**What:** Build ChromaDB indexes offline, upload tarballs to S3, auto-download on first use

**When to use:** Embedding large ontologies (>10K terms), avoid user wait times, stable data

**Trade-offs:**
- **Pros:** Users get instant search (no 10-minute embedding), consistent embeddings, version control
- **Cons:** 30-500MB downloads, S3 hosting costs, stale until index rebuilt

**Example:**
```python
# data/downloader.py
def ensure_index_available(ontology_name: str) -> Path:
    """Download prebuilt index if not present."""
    index_dir = workspace / "search_indexes" / ontology_name

    if index_dir.exists():
        return index_dir

    # Auto-download from S3
    tarball_url = f"https://search-indexes.omics-os.com/{ontology_name}_v2024.tar.gz"
    download_path = workspace / f"{ontology_name}.tar.gz"

    urllib.request.urlretrieve(tarball_url, download_path)
    verify_checksum(download_path)
    extract_tarball(download_path, index_dir)

    return index_dir

# Usage in service
index_path = ensure_index_available("disease_ontology")
backend = ChromaDBBackend(persist_directory=str(index_path))
```

## Data Flow

### Search Request Flow

```
[Agent Query: "Find diseases related to heart attack"]
    ↓
SemanticSearchService.search(query, ontology="disease_ontology")
    ↓
EmbeddingProvider.embed(query) → [768-dim vector]
    ↓
VectorBackend.search(embedding, top_k=100) → [100 candidates]
    ↓
CrossEncoder.score_pairs(query, candidates) → [100 scores]
    ↓
Sort & filter → [Top 10 results]
    ↓
OntologyGraph.enrich_results(results) → [Add ancestors, definitions]
    ↓
Return (results, stats, AnalysisStep)
```

### Index Building Flow (Offline)

```
[OBO File: doid.obo]
    ↓
OBO Parser (obonet) → NetworkX graph
    ↓
Extract terms → [(id, name, synonyms, definition)]
    ↓
Batch embed with SapBERT → [N x 768 embeddings]
    ↓
ChromaDB.add(embeddings, metadata) → Persistent collection
    ↓
Serialize ChromaDB directory → .tar.gz
    ↓
Upload to S3 → https://search-indexes.omics-os.com/doid_v2024.tar.gz
```

### Integration with Existing Services

```
[DiseaseOntologyService] (keyword-based)
    ↓ Strangler Fig Pattern
[DiseaseOntologyService] → delegate to SemanticSearchService
    ↓
Old code paths: keyword matching (deprecated)
New code paths: embedding search (default)
    ↓
Gradual cutover: feature flag, A/B testing
```

### Key Data Flows

1. **Lazy Loading:** Ontology graphs and indexes load on first query (not startup), reducing memory for unused ontologies
2. **Batch Embedding:** Group 128 terms per batch (SapBERT optimal), max_length=25 tokens, [CLS] token extraction
3. **Metadata Filtering:** ChromaDB `where` clauses filter by ontology ID prefix (e.g., `DOID:*`), relationship type, or custom fields
4. **Result Enrichment:** After vector search, query NetworkX graph for hierarchical context (parent terms, definitions, cross-references)

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **0-10K terms** | ChromaDB in-memory, no cross-encoder, single ontology, <100ms latency |
| **10K-1M terms** | ChromaDB persistent, two-stage retrieval (k=100), multiple ontologies, <2s latency, @lru_cache ontology graphs |
| **1M+ terms** | FAISS GPU indexes, pgvector for joins, distributed ChromaDB (vector.omics-os.com), pre-warming caches |

### Scaling Priorities

1. **First bottleneck:** Embedding latency for large queries → **Solution:** Batch embedding (128 terms), GPU acceleration (CUDA), cache common queries
2. **Second bottleneck:** Cross-encoder on 1000+ candidates → **Solution:** Reduce k (100 is optimal), faster cross-encoder models (distilled), skip reranking for high-confidence cases
3. **Third bottleneck:** NetworkX graph queries for 100K+ term ontologies → **Solution:** Neo4j migration (graph database), pre-compute common paths, limit traversal depth

### Memory Footprint

| Component | Memory | Mitigation |
|-----------|--------|-----------|
| SapBERT model | ~400MB | Shared across processes, quantize to int8 (200MB) |
| Cross-encoder | ~100MB | Lazy load, unload after batch |
| Ontology graph | ~100MB per ontology | @lru_cache (8 max), load on-demand |
| ChromaDB collection | 2-10x embedding size | FAISS for compression (binary quantization) |

## Anti-Patterns

### Anti-Pattern 1: Embedding Every Query Unnecessarily

**What people do:** Embed user queries even when exact match exists in ontology

**Why it's wrong:**
- SapBERT inference: 10-50ms per query (adds latency)
- Exact matches often more precise than semantic search
- Wastes GPU cycles for deterministic lookups

**Do this instead:**
```python
# Check exact match first
if query.upper() in id_to_term_map:
    return [exact_match_result]

# Fall back to semantic search
embedding = embedding_provider.embed(query)
results = vector_backend.search(embedding, top_k=10)
```

### Anti-Pattern 2: Cross-Encoder on Full Corpus

**What people do:** Score all ontology terms with cross-encoder for "maximum precision"

**Why it's wrong:**
- Cross-encoders are O(n) with corpus size (1M terms = 1M forward passes)
- 1000x slower than bi-encoder + index (FAISS/ChromaDB)
- Doesn't scale beyond 10K terms

**Do this instead:** Two-stage retrieval (bi-encoder retrieves 100, cross-encoder reranks 100)

### Anti-Pattern 3: Hardcoding Backend in Service Layer

**What people do:** `from chromadb import Client` directly in SemanticSearchService

**Why it's wrong:**
- Violates PEP 420 optional dependency principle
- Forces all users to install ChromaDB (even if using FAISS)
- Breaks Lobster's modular package architecture

**Do this instead:** Factory pattern with lazy imports
```python
# Wrong
from chromadb import Client
self.backend = Client()

# Right
from lobster.services.search.backends import get_backend
self.backend = get_backend()  # Reads LOBSTER_VECTOR_BACKEND env
```

### Anti-Pattern 4: Re-parsing OBO Files on Every Query

**What people do:** Load NetworkX graph in `__init__()` or per-query

**Why it's wrong:**
- OBO parsing: 5-30s for large ontologies (Disease Ontology: 11K terms)
- Blocks service initialization or adds query latency
- Redundant I/O and parsing

**Do this instead:** @lru_cache singleton pattern (see Pattern 3)

### Anti-Pattern 5: Storing Raw Text Instead of IDs

**What people do:** Store full term names in vector metadata, query by name

**Why it's wrong:**
- Redundant storage (name embedded in vector + metadata)
- String matching fragile (typos, casing, synonyms)
- Breaks when ontology updates change names

**Do this instead:** Store stable IDs (DOID:5844), join with ontology graph for display
```python
# Vector metadata
metadata = {"id": "DOID:5844", "ontology": "disease_ontology"}

# Enrich results with graph data
for result in results:
    term_data = graph.nodes[result.id]
    result.name = term_data.get("name")
    result.definition = term_data.get("def")
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **HuggingFace Hub** | transformers.AutoModel for SapBERT, sentence-transformers for cross-encoders | Cache models in ~/.cache/huggingface, specify revision for reproducibility |
| **S3 / CloudFront** | boto3 for index uploads, HTTPS GET for downloads | Prebuilt ChromaDB tarballs (30-500MB), checksums in index manifest |
| **OBO Foundry** | obonet.read_obo() from URLs | Daily ontology updates, version pinning recommended (http://purl.obolibrary.org/obo/doid/releases/2024-01-01/doid.obo) |
| **Ontology Lookup Service (EBI)** | REST API fallback for missing terms | Rate limited, use as supplementary source only |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Agent ↔ SemanticSearchService** | Tool calls → service method → 3-tuple return | Provenance via AnalysisStep, stats for agent feedback |
| **SemanticSearchService ↔ VectorBackend** | Abstract interface (search, add, delete) | Backend swappable via env var, no leaky abstractions |
| **SemanticSearchService ↔ OntologyGraph** | Direct NetworkX queries (ancestors, descendants) | Graph shared via @lru_cache singleton, read-only |
| **DiseaseOntologyService ↔ SemanticSearchService** | Strangler Fig: gradual delegation | Feature flag: `USE_SEMANTIC_SEARCH=true`, fallback to keyword if disabled |

### Cloud Integration (Future: vector.omics-os.com)

**Hosted ChromaDB Architecture:**
```
[lobster-cloud ECS] → [ChromaDB Cloud API] → [ChromaDB Server]
    ↓                       ↓                      ↓
API Gateway           HTTPS + JWT Auth       Persistent volumes
```

**Handoff Specification:**
- ChromaDB Cloud client with tenant API keys
- Pre-warmed indexes for all ontologies (Disease, Cell, Uberon)
- Read-only access for users, write access for index updates
- Monitoring: p95 latency <100ms, cache hit rate >80%

## Build Order Recommendations

Based on component dependencies, suggested implementation sequence:

### Phase 1: Foundation (Core Infrastructure)
**Build first:**
1. `backends/base.py` - Abstract VectorBackend interface
2. `embedding_provider.py` - SapBERT wrapper (single provider)
3. `backends/chromadb_backend.py` - ChromaDB implementation only

**Why:** Minimal viable architecture. ChromaDB handles embedding automatically (no separate provider initially). Proves core concept before complexity.

**Test with:** Single ontology (Disease Ontology), 100 terms, exact match + semantic search

### Phase 2: Search Service (Application Layer)
**Build next:**
1. `semantic_search_service.py` - 3-tuple return, single backend
2. `ontology/ontology_graph.py` - NetworkX wrapper, basic queries
3. `ontology/loader.py` - @lru_cache singleton

**Why:** Connects backend to Lobster patterns (3-tuple, provenance). Ontology graph adds hierarchical context.

**Test with:** Integration into DiseaseOntologyService (Strangler Fig), compare keyword vs. semantic results

### Phase 3: Performance (Two-Stage Retrieval)
**Build when:** Precision insufficient, >10K terms, latency acceptable
1. `reranking/cross_encoder.py` - Two-stage pipeline
2. `reranking/models.py` - ms-marco cross-encoder
3. Tune k (candidate pool size), measure precision@10

**Why:** Cross-encoder is 10x slower but 10x more precise. Only add when bi-encoder alone insufficient.

**Test with:** A/B test vs. bi-encoder only, measure nDCG@10, latency p95

### Phase 4: Scalability (Alternative Backends)
**Build when:** ChromaDB too slow (>1s queries), GPU available, >1M terms
1. `backends/faiss_backend.py` - FAISS indexes
2. `backends/pgvector_backend.py` - PostgreSQL stub
3. Backend factory with env var selection

**Why:** FAISS for GPU acceleration (100x faster), pgvector for SQL integration (joins with metadata tables)

**Test with:** Benchmark 1M term corpus, GPU vs. CPU latency

### Phase 5: Automation (Prebuilt Indexes)
**Build when:** Embedding large ontologies (>10 min), stable data, multiple users
1. `data/index_builder.py` - Offline OBO → ChromaDB pipeline
2. `data/downloader.py` - S3 auto-download
3. CI/CD: ontology updates trigger rebuild

**Why:** Users shouldn't wait 10 minutes for first query. Prebuilt indexes provide instant search.

**Test with:** Disease Ontology (11K terms), Cell Ontology (2.5K terms), Uberon (13K terms)

### Dependency Graph
```
Phase 1 (Foundation)
    ↓
Phase 2 (Search Service) ← Phase 3 (Two-Stage Retrieval)
    ↓                            ↓
Phase 4 (Alternative Backends)
    ↓
Phase 5 (Prebuilt Indexes)
```

**Critical Path:** Foundation → Search Service → Integration with DiseaseOntologyService
**Parallel Work:** Two-stage retrieval, alternative backends (independent of each other)
**Defer:** Prebuilt indexes until user feedback confirms ontology selection

## Sources

**High Confidence (Official Docs + Primary Research):**
- ChromaDB architecture: https://docs.trychroma.com/ (official docs)
- SapBERT paper: https://arxiv.org/abs/2010.11784 (Liu et al., 2020)
- SapBERT model card: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
- obonet: https://github.com/dhimmel/obonet (NetworkX OBO parser)
- Two-stage retrieval: https://www.sbert.net/examples/applications/retrieve_rerank/ (sentence-transformers docs)
- FAISS: https://github.com/facebookresearch/faiss (Facebook AI Research)
- pgvector: https://github.com/pgvector/pgvector (PostgreSQL extension)

**Medium Confidence (WebFetch, Not Verified with Multiple Sources):**
- Uberon ontology: https://github.com/obophenotype/uberon
- Disease Ontology: https://disease-ontology.org/
- Cross-encoder best practices: https://www.sbert.net/docs/pretrained_cross-encoders.html

**Gaps:**
- Cell Ontology structure (OLS page requires JavaScript, not fetched)
- Production ChromaDB scaling patterns (official docs focus on prototyping)
- Cross-encoder performance benchmarks for biomedical domains (no biomedical-specific reranker found)

---
*Architecture research for: Biomedical Semantic Vector Search*
*Researched: 2026-02-17*
*Confidence: HIGH (ChromaDB, SapBERT, two-stage retrieval verified with official sources)*
