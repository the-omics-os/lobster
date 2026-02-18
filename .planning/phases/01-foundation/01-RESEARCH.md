# Phase 1: Foundation - Research

**Researched:** 2026-02-18
**Domain:** Vector search infrastructure (ChromaDB + SapBERT embeddings)
**Confidence:** HIGH

## Summary

Phase 1 delivers a thin, general-purpose semantic matching engine: embed a query string, search a ChromaDB collection, return ranked results with cosine similarity scores. No domain logic, no reranking, no ontology graph -- just the vector plumbing that Phase 2+ services build on top of.

The stack is straightforward: ChromaDB 1.5.0 as the persistent vector store (PersistentClient with HNSW index), SapBERT via sentence-transformers for 768-dimensional biomedical entity embeddings, and Pydantic v2 for result schemas. The critical nuance is that SapBERT requires CLS-token pooling (not the default mean pooling in sentence-transformers), and ChromaDB returns cosine **distance** (1.0 - similarity), not cosine similarity directly -- the conversion must happen in our layer. All heavy dependencies (chromadb, sentence-transformers, torch) are optional and import-guarded behind a `lobster-ai[vector-search]` extra.

**Primary recommendation:** Build three clean modules -- `BaseEmbedder` ABC + `SapBERTEmbedder` implementation, `BaseVectorBackend` ABC + `ChromaDBBackend` implementation, and `VectorSearchService` orchestrator -- all in `lobster/core/vector/`. Keep the API surface minimal: `query(text, collection, top_k)` and `query_batch(texts, collection, top_k)`. Pre-compute embeddings externally and pass them to ChromaDB (bypass ChromaDB's embedding functions entirely for full control).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Query API design
- Thin vector layer only -- embed query, search collection, return ranked results. No domain logic.
- Return top 5 results by default
- Support both single-term `query("heart attack")` and batch `query_batch(["heart attack", "lung cancer"])` from the start
- Return shape: list of match dicts with term, ontology_id, score, metadata -- flat and simple
- Callers (Phase 2 services) add domain logic on top

#### Embedding behavior
- SapBERT as the embedding model from Phase 1 (not a simpler model first)
- Pluggable embedding layer from the start -- BaseEmbedder interface with SapBERT as default implementation
- Lazy model loading (first query triggers load) to meet <500ms CLI startup requirement
- When SapBERT/dependencies aren't installed: raise ImportError with helpful "pip install lobster-ai[vector-search]" message. No fallback to keyword matching.
- Primary consumer driving design: metadata_assistant for harmonization and standardization

#### Collection structure
- One ChromaDB collection per ontology (mondo, uberon, cell_ontology) -- clean isolation, independent updates
- Minimal metadata per document: ontology_id + canonical_name only. Rich details fetched from ontology graph in Phase 2.
- Persistence location: `~/.lobster/vector_store/` (user home, shared across workspaces)
- Collection naming: versioned pattern -- e.g., `mondo_v2024_01`, `uberon_v2024_01`. Allows side-by-side upgrades.

#### Confidence scoring
- Return raw cosine similarity scores (0-1). No calibration or tier mapping at core layer.
- No minimum threshold filtering -- return all top_k results, callers apply their own thresholds
- Include distance metric name in result output (e.g., "cosine") for transparency and debugging
- Scores comparable within a single query's results only, not across different queries

### Claude's Discretion
- Exact BaseEmbedder interface design
- ChromaDB client initialization pattern (in-process vs client-server)
- Batch query implementation strategy (parallel vs sequential embedding)
- Error handling patterns for ChromaDB connection issues
- Test fixture design for vector search unit tests

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | VectorSearchService orchestrates two-stage search pipeline (embed -> search -> rerank -> return) | Phase 1 scope is embed -> search -> return only (no reranking). VectorSearchService class with `query()` and `query_batch()` methods. Reranking slot left empty for Phase 4. |
| INFRA-02 | VectorSearchConfig reads env vars and provides factory methods for backend, embeddings, and reranker | Pydantic-based config class reading `LOBSTER_VECTOR_BACKEND`, `LOBSTER_EMBEDDING_PROVIDER`, persistence path. Factory methods return backend/embedder instances. |
| INFRA-03 | BaseVectorBackend ABC defines add_documents, search, delete, count interface | ABC with 4 abstract methods matching ChromaDB's core operations. Verified ChromaDB Collection API: add(), query(), delete(), count(). |
| INFRA-04 | ChromaDB backend implements BaseVectorBackend with PersistentClient and auto-download from S3 | PersistentClient(path="~/.lobster/vector_store/") with cosine HNSW space. S3 auto-download deferred to Phase 6 (DATA-05). Phase 1 creates empty collections that can be populated. |
| INFRA-08 | All optional deps (chromadb, sentence-transformers, faiss-cpu, obonet) are import-guarded with helpful install messages | Existing codebase pattern: `try: import X; except ImportError: raise ImportError("...pip install lobster-ai[vector-search]...")`. Applied to chromadb + sentence-transformers (torch). |
| EMBED-01 | BaseEmbeddingProvider ABC defines embed_text and embed_batch interface | ABC with `embed_text(str) -> list[float]` and `embed_batch(list[str]) -> list[list[float]]`. Returns raw float lists (not numpy) for ChromaDB compatibility. |
| EMBED-02 | SapBERT provider loads cambridgeltl/SapBERT-from-PubMedBERT-fulltext (768d) with lazy singleton | sentence-transformers with CLS pooling. Lazy loading via `_model` attribute initialized on first call. Model name: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`. 768 dimensions. |
| EMBED-05 | No model downloads at import time -- all loading happens on first use | Lazy singleton pattern: `__init__` stores config only, `_load_model()` called from first `embed_text()` / `embed_batch()`. Verified: sentence-transformers downloads model on `SentenceTransformer()` constructor. |
| SCHM-01 | SearchResult, OntologyMatch, LiteratureMatch, SearchResponse Pydantic models defined in lobster/core/schemas/search.py | Pydantic v2 BaseModel subclasses. OntologyMatch has: term, ontology_id, score, metadata, distance_metric. SearchResult wraps list of matches + query metadata. |
| SCHM-02 | SearchBackend, EmbeddingProvider, RerankerType enums defined | Python str enums: SearchBackend(chromadb, faiss, pgvector), EmbeddingProvider(sapbert, minilm, openai), RerankerType(cross_encoder, cohere, none). |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| chromadb | >=1.0.0,<2.0.0 | Persistent vector store with HNSW index | Official ChromaDB 1.x stable line. PersistentClient for local persistence, EphemeralClient for tests. Cosine distance HNSW out of the box. |
| sentence-transformers | >=4.0.0,<6.0.0 | Model loading and embedding generation | Standard framework for loading HuggingFace transformer models with configurable pooling. 15K+ models supported. Required for SapBERT CLS-pooling configuration. |
| pydantic | >=2.0.0 | Result schemas (SearchResult, OntologyMatch) | Already a core lobster-ai dependency. v2 BaseModel for all new schemas. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | >=2.0.0 | Tensor computation for sentence-transformers | Transitive dependency of sentence-transformers. Not imported directly. |
| transformers | >=4.34.0 | HuggingFace model architecture | Transitive dependency of sentence-transformers. Not imported directly. |
| numpy | (already in lobster) | Embedding normalization if needed | Only for cosine similarity score conversion. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sentence-transformers | Raw transformers AutoModel | More code (manual tokenization, CLS extraction, no encode() convenience). sentence-transformers wraps this cleanly with configurable Pooling module. |
| ChromaDB built-in embedding | External embedding + raw vectors | We chose external. Gives us full control over model loading, caching, and the pluggable BaseEmbedder interface. ChromaDB receives pre-computed vectors only. |
| chromadb EmbeddingFunction interface | Our BaseEmbedder ABC | ChromaDB's EmbeddingFunction ties us to their interface contract. Our own ABC is simpler, testable, and portable to FAISS/pgvector in Phase 5. |

**Installation (human must add to pyproject.toml):**
```toml
[project.optional-dependencies]
vector-search = [
    "chromadb>=1.0.0,<2.0.0",
    "sentence-transformers>=4.0.0,<6.0.0",
]
```

## Architecture Patterns

### Recommended Module Structure

```
lobster/core/vector/
    __init__.py              # Public API: VectorSearchService, VectorSearchConfig
    config.py                # VectorSearchConfig (env vars, factory methods)
    service.py               # VectorSearchService (orchestrator: embed -> search -> return)
    backends/
        __init__.py
        base.py              # BaseVectorBackend ABC
        chromadb_backend.py  # ChromaDBBackend implementation
    embeddings/
        __init__.py
        base.py              # BaseEmbedder ABC
        sapbert.py           # SapBERTEmbedder implementation

lobster/core/schemas/
    search.py                # SearchResult, OntologyMatch, SearchBackend, EmbeddingProvider enums
```

This lives in `lobster/core/` (not a separate package) because vector search is cross-agent infrastructure.

### Pattern 1: Lazy Singleton Embedding Model

**What:** Model loaded on first use, cached for subsequent calls. Meets <500ms CLI startup.
**When to use:** Always for the embedding provider.

```python
# Source: sentence-transformers official docs + SapBERT model card
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

class SapBERTEmbedder(BaseEmbedder):
    """SapBERT embedder with lazy model loading and CLS pooling."""

    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    DIMENSIONS = 768

    def __init__(self):
        self._model = None  # Lazy: loaded on first use

    def _load_model(self):
        if self._model is not None:
            return
        # SapBERT requires CLS-token pooling (not default mean pooling)
        transformer = Transformer(self.MODEL_NAME)
        pooling = Pooling(
            word_embedding_dimension=self.DIMENSIONS,
            pooling_mode="cls"  # Critical: SapBERT uses [CLS] token
        )
        self._model = SentenceTransformer(modules=[transformer, pooling])

    def embed_text(self, text: str) -> list[float]:
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=128)
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS
```

### Pattern 2: ChromaDB PersistentClient with Pre-computed Embeddings

**What:** ChromaDB stores/retrieves pre-computed embeddings. No ChromaDB embedding functions used.
**When to use:** All interactions with ChromaDB in this architecture.

```python
# Source: ChromaDB official docs (docs.trychroma.com)
import chromadb

class ChromaDBBackend(BaseVectorBackend):
    """ChromaDB backend using PersistentClient with pre-computed embeddings."""

    def __init__(self, persist_path: str):
        self._client = chromadb.PersistentClient(path=persist_path)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # Cosine distance
        )

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict:
        collection = self.get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
```

### Pattern 3: Cosine Distance to Similarity Conversion

**What:** ChromaDB returns cosine distance (1 - similarity). We convert to similarity for the user.
**When to use:** Every query result processing step.

```python
# Source: ChromaDB docs - "cosine value returns cosine distance rather than cosine similarity"
# Distance formula: d = 1.0 - cosine_similarity(A, B)
# Therefore: similarity = 1.0 - distance

def _convert_distances_to_scores(self, distances: list[float]) -> list[float]:
    """Convert ChromaDB cosine distances to similarity scores (0-1)."""
    return [max(0.0, min(1.0, 1.0 - d)) for d in distances]
```

### Pattern 4: Import Guard with Helpful Error

**What:** Optional dependencies guarded at point of use with actionable install message.
**When to use:** Every module that imports chromadb or sentence-transformers.

```python
# Source: Existing lobster pattern (e.g., vcf_adapter.py)
def _load_model(self):
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling
    except ImportError:
        raise ImportError(
            "SapBERT embeddings require sentence-transformers and PyTorch. "
            "Install with: pip install 'lobster-ai[vector-search]'"
        )
```

### Pattern 5: BaseEmbedder ABC Interface

**What:** Pluggable embedding provider interface that Phase 5+ can extend.
**When to use:** Any embedding provider implementation.

```python
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings. Returns list of embedding vectors."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings produced."""
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return self.__class__.__name__
```

### Pattern 6: BaseVectorBackend ABC Interface

**What:** Pluggable vector store interface that Phase 5 FAISS/pgvector implementations extend.
**When to use:** Any vector backend implementation.

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseVectorBackend(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents with pre-computed embeddings to a named collection."""
        ...

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Search a collection with a query embedding. Returns raw backend results."""
        ...

    @abstractmethod
    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete documents by ID from a collection."""
        ...

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        ...

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists. Default: try count(), catch exception."""
        try:
            self.count(collection_name)
            return True
        except Exception:
            return False
```

### Anti-Patterns to Avoid

- **Module-level chromadb/torch imports:** Importing at module level triggers model downloads and 2+ second startup. All imports MUST be inside functions or guarded by lazy loading. This is also a hard rule in the CLAUDE.md: "No module-level component_registry calls" -- same principle applies to heavy imports.
- **Using ChromaDB's built-in embedding functions:** Ties us to ChromaDB's embedding interface. Pre-compute embeddings externally and pass raw vectors. This ensures BaseEmbedder stays portable across backends.
- **Storing rich metadata in ChromaDB documents:** Decision is to keep metadata minimal (ontology_id + canonical_name). Rich metadata fetched from ontology graph in Phase 2. Over-storing metadata makes collection migrations harder.
- **Returning raw ChromaDB response format:** ChromaDB returns column-oriented results (`ids[q][k]`, `distances[q][k]`). Always convert to our Pydantic SearchResult/OntologyMatch models before returning to callers.
- **Normalizing scores across queries:** Cosine similarity scores are comparable within a single query's results only. Do not normalize or calibrate across different queries -- that is Phase 2+ domain logic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BERT tokenization | Custom tokenizer | sentence-transformers Transformer module | Handles max_length, padding, truncation, special tokens correctly. SapBERT uses max_length=25 for entity names. |
| CLS token extraction | Manual `output[0][:, 0, :]` | sentence-transformers Pooling(pooling_mode="cls") | Pooling module handles batching, padding mask, and output normalization. The raw transformers approach requires manual attention mask handling. |
| HNSW index management | Custom ANN index | ChromaDB's built-in HNSW | ChromaDB manages index persistence, updates, parameter tuning. Building a custom HNSW is a rabbit hole. |
| Embedding caching | Custom file cache | ChromaDB PersistentClient | ChromaDB already persists the full collection (embeddings + metadata + HNSW index) to disk. Querying a persisted collection requires no re-embedding. |
| Vector similarity math | Custom cosine implementation | ChromaDB HNSW with space="cosine" | ChromaDB's HNSW does efficient approximate nearest neighbor with cosine distance. Only convert distance->similarity on output. |
| Pydantic schema validation | Manual dict validation | Pydantic v2 BaseModel | Already a core dependency. Consistent with all other Lobster schemas. |

**Key insight:** The entire point of Phase 1 is to compose ChromaDB + sentence-transformers behind clean abstractions. The heavy lifting is in the libraries. Our value is in the interface design, lazy loading, and score conversion -- not in reimplementing vector math.

## Common Pitfalls

### Pitfall 1: SapBERT Pooling Mode Mismatch

**What goes wrong:** Using sentence-transformers default mean pooling instead of CLS pooling for SapBERT. Results in degraded matching quality -- embeddings are valid but not optimized for entity linking.
**Why it happens:** sentence-transformers defaults to mean pooling. SapBERT was trained with [CLS] token extraction. Loading via `SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")` directly may use wrong pooling if the model card doesn't specify it.
**How to avoid:** Always construct the model explicitly with `Pooling(pooling_mode="cls")` module. Never rely on auto-detection from model card.
**Warning signs:** Matching quality noticeably worse than expected. "heart attack" doesn't rank "myocardial infarction" highly.

### Pitfall 2: ChromaDB Cosine Distance vs Similarity

**What goes wrong:** Returning raw ChromaDB distance values (0 = identical, 2 = opposite) as "scores" to callers. Users expect similarity (1 = identical, 0 = unrelated).
**Why it happens:** ChromaDB documentation states "cosine value returns cosine distance rather than cosine similarity." The distance formula is `d = 1.0 - cosine_similarity(A, B)`.
**How to avoid:** Always convert in the service layer: `score = 1.0 - distance`. Clamp to [0, 1] for safety. Include `distance_metric: "cosine"` in results for transparency.
**Warning signs:** Scores near 0 for good matches, scores near 1 for bad matches.

### Pitfall 3: Module-Level Import Causing Slow CLI Startup

**What goes wrong:** `lobster --help` takes 3+ seconds because importing the vector module triggers `import torch` -> `import sentence_transformers` -> model discovery.
**Why it happens:** Python imports execute module-level code. A single `from lobster.core.vector import VectorSearchService` at the top of any module in the import chain triggers the full dependency tree.
**How to avoid:** All chromadb/sentence-transformers imports MUST be inside methods (lazy import pattern). The `__init__.py` of `lobster/core/vector/` should expose class names only via `__all__` without importing the actual classes. Use `TYPE_CHECKING` for type hints.
**Warning signs:** Running `time lobster --help` shows >500ms. Add startup profiling (`LOBSTER_PROFILE_TIMINGS=1`).

### Pitfall 4: ChromaDB Collection Name Collisions

**What goes wrong:** Two processes create collections with the same name but different HNSW configurations, or a version upgrade tries to reuse an existing collection with incompatible settings.
**Why it happens:** `get_or_create_collection()` returns the existing collection if the name matches, ignoring any metadata/configuration differences.
**How to avoid:** Use versioned collection names (e.g., `mondo_v2024_01`) as decided. When configuration changes, create a new version. Never modify existing collection configuration.
**Warning signs:** Silently using an old collection with different distance metric or embedding dimensions.

### Pitfall 5: Large Batch Embedding OOM

**What goes wrong:** Calling `embed_batch()` with 60K+ terms at once causes out-of-memory on machines with <16GB RAM. SapBERT is ~440MB in memory; 60K embeddings at 768d float32 is ~183MB; plus intermediate tensors.
**Why it happens:** sentence-transformers `encode()` with large batches can accumulate tensors. The `batch_size` parameter controls internal chunking but all results are held in memory.
**How to avoid:** Set `batch_size=128` in `encode()` (SapBERT model card recommendation). For bulk operations (building collections), process in external chunks of 5K-10K terms and add to ChromaDB incrementally.
**Warning signs:** Python process killed by OS, MemoryError exceptions during collection building.

### Pitfall 6: PersistentClient Path Permissions

**What goes wrong:** First-time user gets PermissionError because `~/.lobster/vector_store/` doesn't exist or has wrong permissions. Or on shared systems, one user's vector store is inaccessible to another.
**Why it happens:** ChromaDB PersistentClient creates the directory but may fail if parent directories don't exist or have restrictive permissions.
**How to avoid:** Create `~/.lobster/vector_store/` with `mkdir(parents=True, exist_ok=True)` before initializing PersistentClient. Follow the existing pattern in `global_config.py` which creates `~/.config/lobster/` similarly.
**Warning signs:** FileNotFoundError or PermissionError on first vector search query.

## Code Examples

Verified patterns from official sources:

### Creating a VectorSearchConfig from Environment Variables

```python
# Pattern follows existing lobster/config/settings.py conventions
import os
from enum import Enum
from pydantic import BaseModel, Field

class SearchBackend(str, Enum):
    CHROMADB = "chromadb"
    FAISS = "faiss"        # Phase 5
    PGVECTOR = "pgvector"  # Phase 5

class EmbeddingProvider(str, Enum):
    SAPBERT = "sapbert"
    MINILM = "minilm"     # Phase 6
    OPENAI = "openai"     # Phase 6

class VectorSearchConfig(BaseModel):
    """Configuration for vector search infrastructure."""
    backend: SearchBackend = Field(
        default=SearchBackend.CHROMADB,
        description="Vector store backend"
    )
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.SAPBERT,
        description="Embedding model provider"
    )
    persist_path: str = Field(
        default="",
        description="ChromaDB persistence directory"
    )
    default_top_k: int = Field(
        default=5,
        description="Default number of results to return"
    )

    @classmethod
    def from_env(cls) -> "VectorSearchConfig":
        """Create config from environment variables."""
        persist_path = os.environ.get(
            "LOBSTER_VECTOR_STORE_PATH",
            str(Path.home() / ".lobster" / "vector_store")
        )
        return cls(
            backend=os.environ.get("LOBSTER_VECTOR_BACKEND", "chromadb"),
            embedding_provider=os.environ.get("LOBSTER_EMBEDDING_PROVIDER", "sapbert"),
            persist_path=persist_path,
        )
```

### Full Query Flow (VectorSearchService)

```python
# Source: Composition of ChromaDB + sentence-transformers patterns
class VectorSearchService:
    """Orchestrates embed -> search -> format results."""

    def __init__(self, config: VectorSearchConfig | None = None):
        self._config = config or VectorSearchConfig.from_env()
        self._backend = None   # Lazy
        self._embedder = None  # Lazy

    def _get_backend(self) -> BaseVectorBackend:
        if self._backend is None:
            self._backend = self._config.create_backend()
        return self._backend

    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = self._config.create_embedder()
        return self._embedder

    def query(
        self,
        text: str,
        collection: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Single-term semantic search against a collection."""
        embedder = self._get_embedder()
        backend = self._get_backend()

        query_embedding = embedder.embed_text(text)
        raw_results = backend.search(collection, query_embedding, n_results=top_k)

        return self._format_results(raw_results, query_text=text)

    def query_batch(
        self,
        texts: list[str],
        collection: str,
        top_k: int = 5,
    ) -> list[list[dict]]:
        """Batch semantic search -- embeds all terms, then searches sequentially."""
        embedder = self._get_embedder()
        backend = self._get_backend()

        embeddings = embedder.embed_batch(texts)
        results = []
        for text, embedding in zip(texts, embeddings):
            raw = backend.search(collection, embedding, n_results=top_k)
            results.append(self._format_results(raw, query_text=text))
        return results

    def _format_results(self, raw: dict, query_text: str) -> list[dict]:
        """Convert ChromaDB column-oriented results to flat match dicts."""
        matches = []
        if not raw.get("ids") or not raw["ids"][0]:
            return matches

        ids = raw["ids"][0]
        distances = raw["distances"][0]
        metadatas = raw.get("metadatas", [[]])[0]
        documents = raw.get("documents", [[]])[0]

        for i, doc_id in enumerate(ids):
            score = max(0.0, min(1.0, 1.0 - distances[i]))
            meta = metadatas[i] if i < len(metadatas) else {}
            matches.append({
                "term": documents[i] if i < len(documents) and documents[i] else meta.get("canonical_name", ""),
                "ontology_id": meta.get("ontology_id", doc_id),
                "score": round(score, 4),
                "metadata": meta,
                "distance_metric": "cosine",
            })
        return matches
```

### Test Fixture: Ephemeral ChromaDB Backend

```python
# Source: ChromaDB docs (EphemeralClient for in-memory testing)
import pytest

@pytest.fixture
def ephemeral_backend(tmp_path):
    """ChromaDB backend using temp directory for test isolation."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    from lobster.core.vector.backends.chromadb_backend import ChromaDBBackend
    # Use tmp_path for true isolation (auto-cleaned by pytest)
    return ChromaDBBackend(persist_path=str(tmp_path / "test_vector_store"))

@pytest.fixture
def mock_embedder():
    """Deterministic mock embedder for unit tests (no torch required)."""
    from unittest.mock import MagicMock
    from lobster.core.vector.embeddings.base import BaseEmbedder

    embedder = MagicMock(spec=BaseEmbedder)
    embedder.dimensions = 768
    # Deterministic fake embeddings based on text hash
    def fake_embed(text):
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in h] * 48  # 768 dims

    embedder.embed_text.side_effect = fake_embed
    embedder.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    return embedder
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ChromaDB 0.4.x with `Settings(chroma_db_impl=...)` | ChromaDB 1.x with `PersistentClient(path=...)` | ChromaDB 1.0 (mid-2024) | Simplified client initialization. No more Settings object for basic use. |
| `collection.add(embeddings=..., metadatas=..., ids=...)` order | Same API but with optional `documents`, `images`, `uris` | ChromaDB 1.x | API stable but expanded. Our pattern (pre-computed embeddings) is unchanged. |
| sentence-transformers 2.x `SentenceTransformer(model_name)` | sentence-transformers 4-5.x same API, better defaults | sentence-transformers 4.0 (2025) | `SentenceTransformer()` constructor unchanged. Added `similarity()` method. `Pooling` module API stable. |
| Manual BERT + tokenizer + CLS extraction | `SentenceTransformer(modules=[Transformer(...), Pooling(pooling_mode="cls")])` | sentence-transformers 2.x+ | Composable module pipeline. Less boilerplate than raw transformers. |

**Deprecated/outdated:**
- `chromadb.Client()` as a named in-memory client: Still works but `chromadb.EphemeralClient()` is the explicit name (introduced in later 0.x, still available in 1.x).
- `Settings(chroma_db_impl="duckdb+parquet")`: Removed in 1.x. PersistentClient handles persistence internally.
- `collection.query(query_texts=...)` auto-embedding: Still works if collection has an embedding function, but we bypass this by providing `query_embeddings` directly.

## Open Questions

1. **ChromaDB collection configuration immutability**
   - What we know: `get_or_create_collection()` returns existing collection if name matches, ignoring configuration changes. The `hnsw:space` setting cannot be changed after creation.
   - What's unclear: Whether ChromaDB 1.x provides any mechanism to detect configuration mismatches between requested and existing collections. If we create `mondo_v2024_01` with cosine and someone recreates it with L2, it silently returns the cosine one.
   - Recommendation: Add a validation step in ChromaDBBackend that checks collection metadata against expected configuration on `get_or_create_collection()`. Log a warning if mismatched.

2. **SapBERT model download on first use**
   - What we know: `SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")` downloads ~440MB from HuggingFace Hub on first use if not cached. The HuggingFace cache is at `~/.cache/huggingface/hub/`.
   - What's unclear: Whether sentence-transformers supports progress callbacks we could hook into for user-facing download progress. Also unclear whether the model download is resumable on network failure.
   - Recommendation: On first embedding call, log a clear message: "Downloading SapBERT model (~440MB). This is a one-time operation." Accept the default HuggingFace caching behavior; don't try to manage model files ourselves.

3. **Batch query implementation: sequential vs parallel ChromaDB queries**
   - What we know: ChromaDB's `collection.query(query_embeddings=...)` supports multiple query embeddings in a single call (results indexed by `[q][k]`). This is more efficient than sequential single queries.
   - What's unclear: Whether ChromaDB's multi-query batching has any practical limits or edge cases for the collection sizes we're targeting (~60K documents).
   - Recommendation: Use ChromaDB's native multi-query support: pass all query embeddings in a single `collection.query(query_embeddings=all_embeddings, ...)` call. This avoids N round-trips and lets HNSW batch internally. Fall back to sequential if the batch exceeds ChromaDB's max_batch_size.

## Sources

### Primary (HIGH confidence)
- ChromaDB official documentation (docs.trychroma.com) -- PersistentClient, Collection API, distance functions, embedding functions, HNSW configuration. Verified: cosine distance = 1 - similarity. Version 1.5.0.
- ChromaDB Python reference (docs.trychroma.com/reference/python) -- EphemeralClient, PersistentClient, Collection methods (query, add, count, delete). Client constructor parameters.
- sentence-transformers official documentation (sbert.net) -- Pooling module with `pooling_mode="cls"`, custom model construction with `modules=[Transformer, Pooling]`. Version 5.2.3.
- SapBERT HuggingFace model card (huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) -- 768d embeddings, CLS token extraction, max_length=25, batch_size=128, 4M+ UMLS concepts. 0.1B parameters.
- PyPI (pypi.org/project/chromadb) -- ChromaDB 1.5.0 released 2026-02-09. Python >=3.9.
- PyPI (pypi.org/project/sentence-transformers) -- v5.2.3 released 2026-02-17. Python >=3.10.

### Secondary (MEDIUM confidence)
- Existing Lobster codebase patterns -- Import guard pattern from `vcf_adapter.py`, ABC pattern from `core/interfaces/backend.py`, schema patterns from `core/schemas/ontology.py`, config patterns from `config/settings.py` and `config/global_config.py`.
- ChromaDB GitHub releases (github.com/chroma-core/chroma/releases) -- Version history, 1.x release timeline.

### Tertiary (LOW confidence)
- ChromaDB 0.x -> 1.x migration details -- Could not find explicit migration guide. API surface appears backward compatible for our use case (PersistentClient, add, query with pre-computed embeddings). LOW confidence on exact breaking changes.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries verified via official docs and PyPI. Version requirements confirmed.
- Architecture: HIGH -- Module structure follows existing Lobster patterns (ABC in interfaces, schemas in core/schemas). ChromaDB and sentence-transformers APIs verified against official docs.
- Pitfalls: HIGH -- Distance-vs-similarity conversion, CLS pooling requirement, and lazy loading all verified against official documentation. OOM pitfall based on SapBERT model card batch_size recommendation.
- Test fixtures: MEDIUM -- EphemeralClient and mock embedder patterns are based on standard pytest patterns. ChromaDB 1.x EphemeralClient confirmed to exist but detailed testing API not fully documented. Using `tmp_path` + PersistentClient is a reliable alternative.

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (30 days -- stable libraries, no fast-moving changes expected)
