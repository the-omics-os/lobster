# Domain Pitfalls

**Domain:** Biomedical semantic vector search with ontology matching
**Researched:** 2026-02-17

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Module-Level Model Loading Breaks PEP 420

**What goes wrong:** Importing module loads 420MB SapBERT model at import time, adding 2-10s to every CLI command (including `lobster --help`).

**Why it happens:** Convenient to load model once at module top:
```python
# ❌ This breaks PEP 420 namespace packages
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("cambridgeltl/SapBERT...")  # 2-10s load
```

**Consequences:**
- `lobster --help` takes 10s → users uninstall
- `import lobster` fails if optional deps (torch, sentence-transformers) not installed
- Violates Lobster Hard Rule #10: "NO module-level `component_registry` calls"
- Breaks PEP 420 namespace package pattern

**Prevention:**
1. Lazy load models on first search call (not at import)
2. Wrap all optional imports in try/except with helpful error
3. Use lazy class attributes or functions

```python
# ✅ Correct pattern
class EmbeddingProvider:
    _model = None  # Lazy class attribute

    def embed(self, text: str):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("cambridgeltl/SapBERT...")
            except ImportError:
                raise ImportError(
                    "Vector search requires optional dependencies. "
                    "Install with: pip install lobster-ai[vector-search]"
                )
        return self._model.encode(text)
```

**Detection:**
- Run `time lobster --help` → should be <500ms
- Run `python -c "import lobster"` without torch installed → should raise helpful error, not silent failure
- Check for module-level model loading: `grep -r "SentenceTransformer\|AutoModel\|chromadb" *.py` outside functions

### Pitfall 2: Hardcoded Backend Breaks Extensibility

**What goes wrong:** Directly importing `chromadb.Client()` in service layer forces all users to install ChromaDB, even if they want pgvector or FAISS.

**Why it happens:** Not anticipating future backend changes. Hardcoding simplest backend (ChromaDB) seems pragmatic.

**Consequences:**
- Users can't switch backends without editing service code
- Violates DRY: backend logic scattered across multiple services
- Impossible to test with mock backend
- Violates Lobster Hard Rule #7: "Use component_registry for premium features — NO `try/except ImportError`"

**Prevention:**
1. Define `VectorBackend` abstract interface in `lobster/services/search/backends/base.py`
2. Factory pattern with environment variable: `LOBSTER_VECTOR_BACKEND=chromadb|pgvector|faiss`
3. Lazy import backends in factory, not at module level

```python
# backends/base.py
class VectorBackend(ABC):
    @abstractmethod
    def search(self, embedding: np.ndarray, k: int) -> List[SearchResult]:
        pass

# backends/__init__.py
def get_vector_backend() -> VectorBackend:
    backend = os.getenv("LOBSTER_VECTOR_BACKEND", "chromadb")

    if backend == "chromadb":
        try:
            from .chromadb_backend import ChromaDBBackend
            return ChromaDBBackend()
        except ImportError:
            raise ImportError("pip install lobster-ai[chromadb]")
    elif backend == "pgvector":
        # Future stub
        raise NotImplementedError("pgvector backend coming soon")
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**Detection:**
- Search codebase: `grep -r "from chromadb import" --include="*.py"` should only appear in `chromadb_backend.py`
- Try: `LOBSTER_VECTOR_BACKEND=faiss lobster chat` → should work or give clear error
- Unit tests should pass with mock backend (no real ChromaDB)

### Pitfall 3: In-Memory ChromaDB Loses Data on Restart

**What goes wrong:** Using `chromadb.Client()` (in-memory mode) instead of `chromadb.PersistentClient()` means embeddings regenerate every session (2-3 minutes for 30K terms).

**Why it happens:** ChromaDB docs use in-memory client in quick-start examples. Persistent mode requires understanding file paths.

**Consequences:**
- Users wait 2-3 minutes for first query every session
- Embeddings not reproducible (random initialization, different versions)
- Disk space wasted (re-downloading OBO files every run)
- Breaks assumption that ontology cache is persistent

**Prevention:**
1. ALWAYS use `chromadb.PersistentClient(path=...)` with explicit path
2. Document cache location (`~/.lobster/ontology_cache/`) prominently
3. Add cache verification on startup (check collection exists, count terms)

```python
# ❌ Wrong: In-memory
client = chromadb.Client()  # Data lost on exit

# ✅ Correct: Persistent
cache_dir = os.path.expanduser("~/.lobster/ontology_cache/mondo")
client = chromadb.PersistentClient(path=cache_dir)
```

**Detection:**
- Run lobster twice with same query → second run should be instant (<100ms)
- Check `~/.lobster/ontology_cache/` → should contain `chroma.sqlite3` and `embeddings/` after first run
- Kill and restart Python process → cache should survive

### Pitfall 4: No Reranking Means Poor Precision

**What goes wrong:** Embedding-only search (no cross-encoder reranking) has ~10-15% worse precision. "Diabetes" matches "diabetic retinopathy" higher than "type 2 diabetes mellitus".

**Why it happens:** Bi-encoders (SapBERT) optimize for recall, not precision. Cross-encoders slow (1-5s for 100 candidates), so skipped for speed.

**Consequences:**
- Users lose trust in semantic search ("keyword matching was better")
- Wrong ontology terms propagate through analysis pipeline
- Agents make incorrect assumptions (cell type annotation failure)

**Prevention:**
1. Implement two-stage retrieval by default: bi-encoder (retrieve 100) → cross-encoder (rerank to 10)
2. Measure precision: test queries with known ground truth
3. Make reranking configurable but enabled by default: `search(rerank=True)`

```python
def search(self, query: str, k: int = 5, rerank: bool = True):
    # Stage 1: Fast retrieval
    embedding = self.embedding_provider.embed(query)
    candidates = self.vector_backend.search(embedding, k=k*20)  # oversample 20x

    if not rerank:
        return candidates[:k]

    # Stage 2: Precise reranking
    reranked = self.cross_encoder.rerank(query, candidates, k=k)
    return reranked
```

**Detection:**
- Query: "heart attack" → top 5 should include "myocardial infarction" (not just generic "heart disease")
- A/B test: embedding-only vs. two-stage → measure nDCG@10, precision@5
- User feedback: do researchers accept or reject automated matches?

### Pitfall 5: Large Package Size Slows Adoption

**What goes wrong:** Bundling 280MB ontology embeddings in PyPI package makes `pip install lobster-ai` slow (60-90s) and bloats wheel size.

**Why it happens:** Simplest distribution: include everything in package, no runtime download needed.

**Consequences:**
- Slow `pip install` → users abandon before completion
- Users who never use vector search pay 280MB storage cost
- PyPI upload limits (100MB per file, 500MB project total)
- Cloud deploys waste bandwidth (ECS image 280MB larger)

**Prevention:**
1. Ship zero ontology data in package
2. Auto-download tarballs from S3 on first use
3. Show progress bar during download (30-60s acceptable with feedback)
4. Document cache location for cleanup

```python
def ensure_ontology_cached(ontology: str) -> Path:
    cache_dir = Path.home() / ".lobster" / "ontology_cache" / ontology

    if cache_dir.exists():
        return cache_dir

    logger.info(f"Downloading {ontology} ontology cache (70MB)...")
    url = f"https://ontology.omics-os.com/{ontology}-2025-04.tar.gz"
    download_with_progress(url, cache_dir.parent / f"{ontology}.tar.gz")
    extract_tarball(cache_dir.parent / f"{ontology}.tar.gz", cache_dir)

    return cache_dir
```

**Detection:**
- `pip install lobster-ai` wheel size should be <10MB
- `lobster --version` should work instantly (no download)
- First search triggers download with progress bar
- `~/.lobster/ontology_cache/` should be 280MB after all ontologies used

## Moderate Pitfalls

### Pitfall 6: No Confidence Calibration

**What goes wrong:** Cosine similarity (0.0-1.0) not calibrated → unclear what 0.7 means. Users don't know when to trust matches.

**Prevention:**
- Validate confidence thresholds empirically: query 100 known terms, measure false positive rate at 0.6, 0.7, 0.8, 0.9
- Document thresholds: ≥0.9 (auto-accept), 0.7-0.9 (warn), <0.7 (manual review)
- Return top-K (not single match) so users see alternatives

### Pitfall 7: No Query Embedding Cache

**What goes wrong:** Embedding is expensive (50ms/term on CPU). Repeated queries (e.g., "diabetes" in cell annotation loop) waste time.

**Prevention:**
- Add LRU cache (1000 entries): `@lru_cache(maxsize=1000)` on embedding function
- Cache key: (model_name, query_text) → embedding
- Invalidate cache on model version change

### Pitfall 8: Ontology Version Staleness

**What goes wrong:** MONDO updates monthly. Local cache from January 2025 missing new diseases added in February 2025.

**Prevention:**
- Tag tarballs with version: `mondo-2025-04.tar.gz` (not `mondo-latest.tar.gz`)
- Check tarball age on startup, warn if >90 days old
- Document manual cache clear: `rm -rf ~/.lobster/ontology_cache/mondo`

### Pitfall 9: No Cross-Ontology Linking

**What goes wrong:** Can't link MONDO disease → affected Uberon tissues. Each ontology is siloed.

**Prevention:**
- Extract cross-references from OBO metadata (MONDO → Uberon mappings)
- Store in separate mapping table: `disease_to_tissue.json`
- Defer to Phase 2 (not needed for MVP)

### Pitfall 10: Embedding Dimensionality Mismatch

**What goes wrong:** SapBERT (768d) embeddings queried against MiniLM (384d) index → dimension error.

**Prevention:**
- Store embedding model name in ChromaDB metadata: `collection.metadata["model"] = "SapBERT"`
- Verify query embedding dimension matches index: `assert len(embedding) == 768`
- Rebuild index if model changed

## Minor Pitfalls

### Pitfall 11: Case Sensitivity

**What goes wrong:** "DIABETES" vs "diabetes" treated differently by some embeddings.

**Prevention:**
- Normalize queries: `query.strip().lower()` before embedding
- SapBERT is case-insensitive (trained on uncased PubMedBERT), but document this

### Pitfall 12: GPU Unavailability

**What goes wrong:** Code assumes GPU (`.cuda()`), crashes on CPU-only machines.

**Prevention:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Pitfall 13: Silent Fallback

**What goes wrong:** Strangler Fig pattern: if semantic search fails, silently fall back to keyword matching. Users don't know system degraded.

**Prevention:**
- Log warnings: `logger.warning("Vector search unavailable, using keyword fallback")`
- Return confidence=0.5 for keyword matches (vs. 0.8-1.0 for semantic)
- Surface fallback in UI: "Limited matching available (install vector-search extra)"

### Pitfall 14: No Index Verification

**What goes wrong:** Corrupted tarball download → ChromaDB load fails with cryptic error.

**Prevention:**
- Include SHA256 checksums in tarball manifest
- Verify checksum after download before extraction
- Re-download if verification fails

### Pitfall 15: Thread Safety

**What goes wrong:** Multiple threads calling `embedding_provider.embed()` → race condition loading model.

**Prevention:**
- Add threading.Lock around model initialization
- Or: singleton pattern with `@lru_cache` (thread-safe)

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **Infrastructure Setup** | Module-level imports break PEP 420 | Lazy loading + import guards |
| **Embedding Pipeline** | No reranking → poor precision | Two-stage retrieval by default |
| **Ontology Data Pipeline** | Large package size → slow pip install | Auto-download from S3, not bundled |
| **Service Integration** | Strangler Fig: breaking DiseaseOntologyService API | Backend swap only, preserve public API |
| **Cloud Deployment (vector.omics-os.com)** | ChromaDB server mode lacks auth | Use FastAPI wrapper with Cognito |
| **Testing** | Real ChromaDB required for integration tests | Mock backend for unit tests, small real index for integration |

## Sources

**HIGH confidence (verified with code/docs):**
- PEP 420 namespace packages: https://www.python.org/dev/peps/pep-0420/
- ChromaDB PersistentClient: https://docs.trychroma.com/usage-guide (official docs)
- Lobster Hard Rules: /Users/tyo/omics-os/lobster/CLAUDE.md (lines 147-159)
- SRAgent reference: /Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/ (validated patterns)

**MEDIUM confidence (inferred from best practices):**
- Two-stage retrieval precision gain: ~10-15% (sentence-transformers docs, not measured for biomedical)
- Confidence threshold 0.7: Common industry practice, needs empirical validation
- LRU cache size 1000: Heuristic, depends on query diversity

**LOW confidence (needs validation):**
- Cross-ontology linking complexity: Unclear how often users need this
- Ontology staleness impact: Unknown how quickly MONDO/Uberon updates affect users
