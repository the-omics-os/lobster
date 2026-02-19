# Phase 5: Scalability - Research

**Researched:** 2026-02-19
**Domain:** Vector database backend abstraction (FAISS, pgvector stub, factory pattern)
**Confidence:** HIGH

## Summary

Phase 5 implements two alternative vector database backends (FAISS in-memory, pgvector stub) and wires them into the existing `VectorSearchConfig.create_backend()` factory so that setting `LOBSTER_VECTOR_BACKEND=faiss` switches the entire vector search pipeline with zero code changes at the service layer. The existing `BaseVectorBackend` ABC, `SearchBackend` enum, and config factory are already designed for this -- they just need the implementations added.

The primary technical challenge is impedance mismatch between FAISS and the existing interface: FAISS uses integer IDs (not strings), returns squared L2 distances (not cosine distances), and uses numpy arrays (not Python lists). The FAISS backend must internally manage a string-to-integer ID mapping, normalize vectors with `faiss.normalize_L2`, and convert squared L2 distances to cosine distances so the service layer's `_format_results` method works identically across backends. The pgvector backend is trivially a stub that raises `NotImplementedError` on all methods.

**Primary recommendation:** Implement FAISSBackend with internal string-to-int ID mapping, per-collection indexes using IndexFlatL2 on L2-normalized vectors, and squared-L2-to-cosine distance conversion in the `search()` return. Wire both backends into `create_backend()` factory. Write comprehensive unit tests with mocked FAISS (the pattern exists in test_embedders.py and test_rerankers.py).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-05 | FAISS backend implements BaseVectorBackend with in-memory IndexFlatL2 and L2-normalized vectors | FAISSBackend class wrapping faiss.IndexFlatL2 with faiss.normalize_L2 on add/search, internal string-to-int ID mapping, and distance conversion |
| INFRA-06 | pgvector backend stub raises NotImplementedError with helpful message | PgVectorBackend class with all 4 abstract methods raising NotImplementedError("Coming in v2.0...") |
| INFRA-07 | Switching LOBSTER_VECTOR_BACKEND env var changes backend with zero code changes | VectorSearchConfig.create_backend() factory extended with faiss/pgvector branches; env var already parsed in from_env() |
| TEST-01 | Unit tests for backends (ChromaDB, FAISS, pgvector stub) with mocked deps | Test file with mocked faiss module (sys.modules patch), ChromaDB skip-if pattern, pgvector NotImplementedError assertions |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faiss-cpu | 1.13.2 | In-memory vector similarity search | Meta's production-grade ANN library, 40K+ GitHub stars, CPU-optimized with SIMD |
| numpy | >=1.23.0 (already a dep) | Array operations for FAISS vectors | Required by FAISS for all vector operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none beyond existing) | - | - | FAISS backend is self-contained with faiss-cpu + numpy |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| IndexFlatL2 | IndexIVFFlat | Only faster at >100K vectors; adds training step complexity |
| IndexFlatL2 | IndexFlatIP (inner product) | Equivalent for normalized vectors; L2 is more explicit about normalization |
| In-memory storage | IndexIDMap2 (persistent) | Not needed for 60K vectors; persistence already handled by ChromaDB default |

**Installation:**
```bash
pip install faiss-cpu
```

**Note:** `faiss-cpu` is an optional dependency -- import-guarded in the backend module. Not added to pyproject.toml (hard rule: no editing pyproject.toml).

## Architecture Patterns

### Recommended Project Structure
```
lobster/core/vector/backends/
    base.py              # BaseVectorBackend ABC (exists)
    chromadb_backend.py  # ChromaDB impl (exists)
    faiss_backend.py     # NEW: FAISS IndexFlatL2 backend
    pgvector_backend.py  # NEW: pgvector stub
    __init__.py          # Package init (exists)

lobster/core/vector/
    config.py            # MODIFIED: create_backend() gains faiss/pgvector branches
    service.py           # UNCHANGED: works via BaseVectorBackend abstraction

tests/unit/core/vector/
    test_backends.py     # NEW: Unit tests for all 3 backends
```

### Pattern 1: String-to-Integer ID Mapping for FAISS
**What:** FAISS only supports 64-bit integer IDs. The BaseVectorBackend interface uses string IDs (ontology IDs like "CL:0000084"). The FAISS backend must maintain a bidirectional mapping per collection.
**When to use:** Always, for FAISS backend.
**Example:**
```python
# Source: FAISS wiki (IndexIDMap) + BaseVectorBackend interface contract
class FAISSBackend(BaseVectorBackend):
    def __init__(self):
        self._collections: dict[str, dict] = {}
        # Per collection: {
        #   "index": faiss.IndexFlatL2(dim),
        #   "id_to_int": {"CL:0000084": 0, ...},
        #   "int_to_id": {0: "CL:0000084", ...},
        #   "documents": {0: "T cell", ...},
        #   "metadatas": {0: {"ontology_id": "CL:0000084"}, ...},
        #   "next_int_id": 1,
        # }
```

### Pattern 2: L2 Normalization for Cosine Compatibility
**What:** The service layer converts distances via `score = 1.0 - distance`. ChromaDB returns cosine distances (range 0-2, typically 0-1). FAISS IndexFlatL2 returns squared L2 distances. For L2-normalized vectors: `squared_L2 = 2 - 2*cosine_similarity = 2*cosine_distance`. So FAISS must convert: `cosine_distance = squared_L2 / 2`.
**When to use:** In FAISS backend's `search()` method, before returning results.
**Example:**
```python
# Source: FAISS wiki MetricType-and-distances
import numpy as np
import faiss

# Normalize vectors before adding
vectors = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(vectors)  # In-place L2 normalization

# On search, convert distances:
# For normalized vectors: squared_L2 = 2*(1 - cos_sim) = 2*cos_distance
# Service expects cosine distance (1 - similarity), so: cos_distance = squared_L2 / 2
distances_cosine = [d / 2.0 for d in squared_l2_distances]
```

### Pattern 3: Lazy Import Guard (Consistent with ChromaDB)
**What:** Import faiss lazily inside methods, not at module level. Show helpful error message on ImportError.
**When to use:** Every method that calls faiss.
**Example:**
```python
# Source: Existing ChromaDBBackend._get_client() pattern
def _ensure_faiss(self):
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "FAISS is required for the faiss backend. "
            "Install with: pip install faiss-cpu"
        )
```

### Pattern 4: NotImplementedError Stub (pgvector)
**What:** All 4 abstract methods raise NotImplementedError with a helpful, forward-looking message.
**When to use:** pgvector backend exclusively.
**Example:**
```python
class PgVectorBackend(BaseVectorBackend):
    def add_documents(self, collection_name, ids, embeddings, documents=None, metadatas=None):
        raise NotImplementedError(
            "pgvector backend is planned for v2.0. "
            "Use LOBSTER_VECTOR_BACKEND=chromadb (default) or "
            "LOBSTER_VECTOR_BACKEND=faiss for current backends."
        )
```

### Pattern 5: Factory Extension (create_backend)
**What:** Add `elif` branches to `VectorSearchConfig.create_backend()` for faiss and pgvector.
**When to use:** When user sets LOBSTER_VECTOR_BACKEND env var.
**Example:**
```python
def create_backend(self) -> BaseVectorBackend:
    if self.backend == SearchBackend.chromadb:
        from lobster.core.vector.backends.chromadb_backend import ChromaDBBackend
        return ChromaDBBackend(persist_path=self.persist_path)

    if self.backend == SearchBackend.faiss:
        from lobster.core.vector.backends.faiss_backend import FAISSBackend
        return FAISSBackend()

    if self.backend == SearchBackend.pgvector:
        from lobster.core.vector.backends.pgvector_backend import PgVectorBackend
        return PgVectorBackend()

    raise ValueError(f"Unsupported backend: {self.backend}. Available: chromadb, faiss, pgvector")
```

### Anti-Patterns to Avoid
- **Using IndexIDMap wrapper:** While FAISS provides IndexIDMap for custom integer IDs, it still requires integers. Since we need string IDs, maintaining our own dict mapping is simpler and gives full control over documents/metadatas storage. IndexIDMap adds complexity without solving the string-ID problem.
- **Storing metadata in FAISS:** FAISS only stores vectors and integer IDs. Documents and metadata must be stored separately in Python dicts alongside the index. Do not try to serialize metadata into FAISS.
- **Forgetting normalization on queries:** If vectors are L2-normalized when added, query vectors MUST also be L2-normalized before search. Missing this produces garbage results.
- **In-place mutation of input lists:** `faiss.normalize_L2` modifies arrays in-place. Copy the input embeddings to a numpy array before normalizing to avoid mutating caller data.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| L2 vector normalization | Manual `v / np.linalg.norm(v)` per vector | `faiss.normalize_L2(vectors)` | FAISS's C++ implementation is batch-optimized and SIMD-accelerated; handles zero vectors safely |
| Vector similarity search | Loop-based distance computation | `faiss.IndexFlatL2` | Brute-force but heavily optimized with BLAS; handles edge cases (empty index, dimension mismatch) |
| ID mapping serialization | Custom JSON persistence | In-memory dicts (sufficient for 60K) | FAISS backend is ephemeral/in-memory; persistence is ChromaDB's job |

**Key insight:** The FAISS backend is intentionally simpler than ChromaDB. It's in-memory only (no persistence), which is its advantage (speed, no disk I/O) and its constraint (data lost on restart). For 60K vectors this is fine -- rebuild takes seconds.

## Common Pitfalls

### Pitfall 1: Squared L2 vs Cosine Distance Mismatch
**What goes wrong:** The service layer converts distances via `score = max(0.0, min(1.0, 1.0 - distance))`. If FAISS returns raw squared L2 distances (which can be 0-4 for normalized vectors), the scores will be wrong -- e.g., a squared L2 of 0.4 would give `1.0 - 0.4 = 0.6` but the actual cosine distance is `0.4/2 = 0.2`, making the correct score `0.8`.
**Why it happens:** ChromaDB returns cosine distance directly; FAISS returns squared L2 distance. The math is different.
**How to avoid:** In FAISS backend's `search()`, divide squared L2 distances by 2.0 before returning. Then the service formula produces correct cosine similarity scores.
**Warning signs:** Scores systematically lower than expected; "heart attack" matching "myocardial infarction" with score ~0.5 instead of ~0.9.

### Pitfall 2: Non-Normalized Vectors at Query Time
**What goes wrong:** Vectors are normalized when added to the index, but query vectors are not normalized. This breaks the `squared_L2 = 2*(1-cos)` equivalence.
**Why it happens:** Embedder returns un-normalized vectors. ChromaDB's cosine space handles this internally; FAISS IndexFlatL2 does not.
**How to avoid:** Normalize query embedding in `search()` before calling `index.search()`. Copy to numpy array first to avoid mutating the caller's list.
**Warning signs:** Scores not in [0,1] range; same query gives different scores on FAISS vs ChromaDB.

### Pitfall 3: FAISS Requires float32 numpy Arrays
**What goes wrong:** Passing Python lists or float64 arrays to FAISS causes crashes or silent wrong results.
**Why it happens:** FAISS is a C++ library with strict type requirements.
**How to avoid:** Always convert to `np.array(data, dtype=np.float32)` before any FAISS operation.
**Warning signs:** Segfaults, `RuntimeError`, or silently incorrect results.

### Pitfall 4: Empty Collection Search
**What goes wrong:** Searching an empty FAISS index returns garbage or crashes.
**Why it happens:** IndexFlatL2 with ntotal=0 may return uninitialized memory for distances.
**How to avoid:** Guard `search()` with `if index.ntotal == 0: return empty_result`.
**Warning signs:** Random very large distances returned for empty collections.

### Pitfall 5: Collection Not Found
**What goes wrong:** Searching a collection that doesn't exist should raise ValueError (per BaseVectorBackend contract) but dict lookup would raise KeyError.
**Why it happens:** FAISS backend uses a Python dict for collections, not a database with collections.
**How to avoid:** Check `if collection_name not in self._collections:` and raise `ValueError(f"Collection '{collection_name}' does not exist")`.
**Warning signs:** KeyError instead of ValueError in test assertions.

### Pitfall 6: Modifying Existing Config Test
**What goes wrong:** The existing `test_create_backend_unsupported` test asserts that `backend=faiss` raises ValueError. After this phase, that test must change to assert it creates a FAISSBackend instead.
**Why it happens:** This test was correct before FAISS was implemented -- it verified the "not yet implemented" branch.
**How to avoid:** Update the test in `test_config.py` to assert `isinstance(backend, FAISSBackend)` and add a new test for pgvector's NotImplementedError behavior.
**Warning signs:** Existing test `test_create_backend_unsupported` starts failing after wiring the factory.

## Code Examples

Verified patterns from official sources:

### FAISSBackend.add_documents() - Core Implementation
```python
# Source: FAISS wiki Getting-started + MetricType-and-distances
import numpy as np

def add_documents(self, collection_name, ids, embeddings, documents=None, metadatas=None):
    faiss = self._ensure_faiss()

    # Convert to numpy float32
    vectors = np.array(embeddings, dtype=np.float32)

    # L2-normalize for cosine similarity equivalence
    faiss.normalize_L2(vectors)

    # Create or get collection
    if collection_name not in self._collections:
        dim = vectors.shape[1]
        self._collections[collection_name] = {
            "index": faiss.IndexFlatL2(dim),
            "id_to_int": {},
            "int_to_id": {},
            "documents": {},
            "metadatas": {},
            "next_int_id": 0,
        }

    coll = self._collections[collection_name]

    for i, str_id in enumerate(ids):
        int_id = coll["next_int_id"]
        coll["id_to_int"][str_id] = int_id
        coll["int_to_id"][int_id] = str_id
        if documents:
            coll["documents"][int_id] = documents[i]
        if metadatas:
            coll["metadatas"][int_id] = metadatas[i]
        coll["next_int_id"] += 1

    coll["index"].add(vectors)
```

### FAISSBackend.search() - With Distance Conversion
```python
# Source: FAISS wiki Getting-started + ChromaDB-compatible return format
def search(self, collection_name, query_embedding, n_results=5):
    faiss = self._ensure_faiss()

    if collection_name not in self._collections:
        raise ValueError(f"Collection '{collection_name}' does not exist")

    coll = self._collections[collection_name]

    # Handle empty index
    if coll["index"].ntotal == 0:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    # Clamp n_results to available vectors
    n_results = min(n_results, coll["index"].ntotal)

    # Prepare and normalize query
    query = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # Search
    distances, indices = coll["index"].search(query, n_results)

    # Convert squared L2 distances to cosine distances for service compatibility
    # For normalized vectors: squared_L2 = 2 * (1 - cos_sim) = 2 * cos_distance
    cosine_distances = (distances[0] / 2.0).tolist()

    # Map integer indices back to string IDs
    result_ids = []
    result_docs = []
    result_metas = []
    for idx in indices[0]:
        idx = int(idx)
        str_id = coll["int_to_id"].get(idx, str(idx))
        result_ids.append(str_id)
        result_docs.append(coll["documents"].get(idx))
        result_metas.append(coll["metadatas"].get(idx, {}))

    return {
        "ids": [result_ids],
        "distances": [cosine_distances],
        "documents": [result_docs],
        "metadatas": [result_metas],
    }
```

### PgVectorBackend Stub
```python
# Source: REQUIREMENTS.md INFRA-06 specification
class PgVectorBackend(BaseVectorBackend):
    """Placeholder for future PostgreSQL pgvector backend.

    All methods raise NotImplementedError with guidance to use
    chromadb or faiss backends in the meantime.
    """

    _MSG = (
        "pgvector backend is planned for v2.0. "
        "Use LOBSTER_VECTOR_BACKEND=chromadb (default) or "
        "LOBSTER_VECTOR_BACKEND=faiss for current backends."
    )

    def add_documents(self, collection_name, ids, embeddings, documents=None, metadatas=None):
        raise NotImplementedError(self._MSG)

    def search(self, collection_name, query_embedding, n_results=5):
        raise NotImplementedError(self._MSG)

    def delete(self, collection_name, ids):
        raise NotImplementedError(self._MSG)

    def count(self, collection_name):
        raise NotImplementedError(self._MSG)
```

### Unit Test Pattern for FAISS Backend (Mocked)
```python
# Source: Existing test_embedders.py sys.modules patching pattern
import numpy as np
from unittest.mock import MagicMock, patch

class TestFAISSBackend:
    def _make_backend(self):
        """Create FAISSBackend with mocked faiss module."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()  # No-op for tests

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from lobster.core.vector.backends.faiss_backend import FAISSBackend
            backend = FAISSBackend()

        return backend, mock_faiss, mock_index
```

### Config Factory Test (Updated)
```python
# Source: Existing test_config.py pattern + INFRA-07 requirement
def test_create_backend_faiss(self):
    """create_backend() with faiss returns FAISSBackend."""
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu not installed")

    config = VectorSearchConfig(backend=SearchBackend.faiss)
    backend = config.create_backend()

    from lobster.core.vector.backends.faiss_backend import FAISSBackend
    assert isinstance(backend, FAISSBackend)

def test_create_backend_pgvector_raises(self):
    """create_backend() with pgvector returns PgVectorBackend (stub)."""
    config = VectorSearchConfig(backend=SearchBackend.pgvector)
    backend = config.create_backend()

    from lobster.core.vector.backends.pgvector_backend import PgVectorBackend
    assert isinstance(backend, PgVectorBackend)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| faiss.IndexFlatL2 only option | faiss-cpu 1.13.2 includes IndexFlatL2, IVF, HNSW, ScaNN | Dec 2025 | We only need IndexFlatL2 for 60K vectors |
| faiss-gpu for speed | faiss-cpu only (GPU discontinued in pip since 1.7.3) | 2022+ | CPU-only is the standard pip install path |
| Manual vector normalization | `faiss.normalize_L2(x)` built-in | Stable | Single call, SIMD-optimized, in-place |

**Deprecated/outdated:**
- `faiss-gpu` pip package: Discontinued. GPU support exists only via conda. Not relevant for this use case.

## Open Questions

1. **Upsert semantics for FAISS**
   - What we know: BaseVectorBackend.add_documents says "If a document with a given ID already exists, it is overwritten (upsert semantics)." ChromaDB handles this natively.
   - What's unclear: FAISS IndexFlatL2 has no concept of "overwrite by ID" -- it just appends vectors. Supporting upsert requires checking if the string ID already exists and removing the old vector first.
   - Recommendation: Implement check-and-remove in add_documents: if `str_id in coll["id_to_int"]`, remove old vector via `index.remove_ids()`, then add new one. This is O(n) per upsert but acceptable for 60K vectors.

2. **FAISS remove_ids with sequential shifting**
   - What we know: FAISS IndexFlat uses sequential IDs internally. `remove_ids` shifts remaining IDs down, which would invalidate our int-to-string mapping.
   - What's unclear: Whether we should use IndexIDMap wrapper to avoid this issue.
   - Recommendation: Use `faiss.IndexIDMap(faiss.IndexFlatL2(dim))` which stores explicit integer IDs and avoids the shifting problem. Our `int_to_id` mapping stays valid. The cost is negligible for 60K vectors.

## Sources

### Primary (HIGH confidence)
- FAISS wiki Getting-started — `index.add()`, `index.search()`, `index.ntotal`, return format (D, I matrices)
- FAISS wiki MetricType-and-distances — L2/cosine equivalence for normalized vectors, `faiss.normalize_L2`
- FAISS wiki Special-operations-on-indexes — `remove_ids` support for IndexFlat, sequential ID shifting caveat
- FAISS wiki Pre-and-post-processing — IndexIDMap wrapper for custom integer IDs
- PyPI faiss-cpu 1.13.2 page — version, Python 3.10-3.14, platform support
- Existing codebase: `lobster/core/vector/backends/base.py` (ABC contract), `chromadb_backend.py` (reference implementation), `config.py` (factory pattern), `service.py` (distance conversion)

### Secondary (MEDIUM confidence)
- FAISS GitHub issues — IndexIDMap behavior with remove_ids (confirmed: explicit IDs survive removal without shifting)

### Tertiary (LOW confidence)
- None -- all claims verified via primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - faiss-cpu 1.13.2 verified on PyPI, API verified via official wiki
- Architecture: HIGH - Based on existing codebase patterns (ChromaDB backend, config factory, test patterns), all read from source
- Pitfalls: HIGH - Distance conversion math verified via FAISS wiki; ID mapping issue verified via wiki documentation on IndexIDMap and remove_ids
- Code examples: HIGH - Based on FAISS official wiki examples + existing codebase patterns

**Research date:** 2026-02-19
**Valid until:** 2026-04-19 (stable -- FAISS API is mature, BaseVectorBackend ABC is locked)
