# Phase 4: Performance - Research

**Researched:** 2026-02-19
**Domain:** Two-stage retrieval with cross-encoder reranking for vector search
**Confidence:** HIGH

## Summary

Phase 4 adds reranking to the existing two-stage vector search pipeline. The infrastructure is well-prepared: `VectorSearchService.match_ontology()` already oversamples at 4x (requests k*4 from the backend), `RerankerType` enum already defines `cross_encoder`, `cohere`, and `none` values in `lobster/core/schemas/search.py`, and the `VectorSearchConfig` Pydantic model is ready for a `reranker` field. The work is to build a `BaseReranker` ABC, two implementations (CrossEncoderReranker + CohereReranker), wire reranking into VectorSearchService between the search and truncation steps, and add config/env var support.

The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is a 22.7M parameter model (~91MB download) that uses the same `sentence-transformers` library already installed for SapBERT embeddings. It provides a `CrossEncoder.rank()` method that takes a query and documents list and returns sorted results with scores -- directly suitable for reranking. The Cohere reranker uses a separate `cohere` Python SDK (v5.x) and requires an API key. Both must be import-guarded following the existing lazy loading patterns.

**Primary recommendation:** Create `lobster/core/vector/rerankers/` module with `base.py`, `cross_encoder_reranker.py`, and `cohere_reranker.py`. Wire into VectorSearchService via config-driven factory pattern matching existing `create_backend()`/`create_embedder()` pattern. Add `LOBSTER_RERANKER` env var. Test with mocked models following MockEmbedder/MockVectorBackend patterns.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RANK-01 | CrossEncoderReranker uses cross-encoder/ms-marco-MiniLM-L-6-v2 with lazy loading | `sentence-transformers.CrossEncoder` class supports lazy instantiation. Model is 91MB, same library as SapBERT. Use `CrossEncoder.rank()` method for reranking. Lazy loading follows SapBERTEmbedder `_load_model()` pattern. |
| RANK-02 | CohereReranker gracefully degrades without API key (logs warning, returns original order) | Cohere SDK v5.x `ClientV2.rerank()` raises auth error without key. Wrap in try/except, log warning, return input order unchanged. Check `COHERE_API_KEY` or `CO_API_KEY` env var presence. |
| RANK-03 | Reranker is optional -- search works without reranking if reranker set to "none" | Add `reranker: RerankerType` field to `VectorSearchConfig` with default `RerankerType.none`. `create_reranker()` returns `None` for `RerankerType.none`. Service skips reranking step when reranker is None. |
| TEST-02 | Unit tests for embedding providers (SapBERT, MiniLM, OpenAI) with mocked model loading | SapBERTEmbedder already tested indirectly via MockEmbedder in service tests. Need dedicated tests that mock `sentence_transformers.SentenceTransformer` and verify CLS pooling config, batch_size=128, lazy loading behavior. MiniLM/OpenAI are Phase 6 but stub tests can verify the ABC contract. |
| TEST-03 | Unit tests for rerankers (cross-encoder, Cohere) with mocked clients | Mock `sentence_transformers.CrossEncoder` for cross-encoder tests. Mock `cohere.ClientV2` for Cohere tests. Verify: score ordering, top-k truncation, graceful degradation, lazy loading, import guard messages. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sentence-transformers | 5.2.x | CrossEncoder model loading and inference | Already used for SapBERT embeddings. CrossEncoder class built-in. Same library, no new dependency. |
| cohere | 5.x | Cohere Rerank API client | Official Python SDK. MIT license. Lightweight import when guarded. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | (transitive via sentence-transformers) | Model inference runtime | Required by CrossEncoder, already present from SapBERT |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sentence-transformers CrossEncoder | Raw transformers AutoModelForSequenceClassification | More control but more boilerplate; sentence-transformers already in stack |
| ms-marco-MiniLM-L-6-v2 | ms-marco-MiniLM-L-12-v2 | L-12 has better accuracy but 2x slower; L-6 is the project decision |
| Cohere rerank-v4.0-pro | Jina Reranker | Cohere is the project decision, more established API |

**Installation:**
```bash
# sentence-transformers already installed for SapBERT
# cohere is new (optional, import-guarded)
pip install cohere  # Only if using Cohere reranker
```

## Architecture Patterns

### Recommended Module Structure
```
lobster/core/vector/
├── __init__.py                    # Add BaseReranker + lazy exports
├── config.py                      # Add reranker field + create_reranker() factory
├── service.py                     # Add _rerank() step in match_ontology() and query()
├── backends/
│   ├── base.py                    # (unchanged)
│   └── chromadb_backend.py        # (unchanged)
├── embeddings/
│   ├── base.py                    # (unchanged)
│   └── sapbert.py                 # (unchanged)
└── rerankers/                     # NEW directory
    ├── __init__.py                # Empty or re-exports
    ├── base.py                    # BaseReranker ABC
    ├── cross_encoder_reranker.py  # CrossEncoderReranker implementation
    └── cohere_reranker.py         # CohereReranker implementation
```

### Test Structure
```
tests/unit/core/vector/
├── test_vector_search_service.py  # Existing -- add reranking tests
├── test_config.py                 # Existing -- add reranker config tests
├── test_schemas.py                # Existing -- (unchanged)
├── test_ontology_graph.py         # Existing -- (unchanged)
├── test_embedders.py              # NEW -- TEST-02 (mocked embedding provider tests)
└── test_rerankers.py              # NEW -- TEST-03 (mocked reranker tests)
```

### Pattern 1: BaseReranker ABC
**What:** Abstract base class for all reranker implementations, parallel to BaseEmbedder/BaseVectorBackend.
**When to use:** Every reranker must implement this interface.
**Example:**
```python
# Source: Derived from existing BaseEmbedder pattern in lobster/core/vector/embeddings/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Abstract interface for result rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query text.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. None = all.

        Returns:
            list[dict]: Sorted results, each with keys:
                - corpus_id (int): Original index in documents list
                - score (float): Relevance score (higher = more relevant)
                - text (str): Document text
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for this reranker."""
        return self.__class__.__name__
```

### Pattern 2: CrossEncoder Lazy Loading (follows SapBERT pattern)
**What:** Lazy model loading using the same pattern as SapBERTEmbedder.
**When to use:** CrossEncoderReranker implementation.
**Example:**
```python
# Source: Pattern from lobster/core/vector/embeddings/sapbert.py + sentence-transformers CrossEncoder docs
import logging
from lobster.core.vector.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker with lazy model loading."""

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install with: pip install 'lobster-ai[vector-search]'"
            )
        self._model = CrossEncoder(MODEL_NAME)
        logger.info("Loaded cross-encoder model (%s).", MODEL_NAME)

    def rerank(self, query, documents, top_k=None):
        self._load_model()
        results = self._model.rank(
            query, documents, top_k=top_k, return_documents=True
        )
        return [
            {"corpus_id": r["corpus_id"], "score": float(r["score"]), "text": r["text"]}
            for r in results
        ]
```

### Pattern 3: Cohere Graceful Degradation
**What:** CohereReranker returns original order when API key missing.
**When to use:** RANK-02 requirement.
**Example:**
```python
# Source: Cohere Rerank API docs + graceful degradation requirement
import logging
import os
from lobster.core.vector.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """Cohere API-based reranker with graceful degradation."""

    MODEL = "rerank-v4.0-pro"

    def __init__(self) -> None:
        self._client = None
        self._available = None  # None = not checked yet

    def _init_client(self) -> bool:
        """Initialize Cohere client. Returns False if unavailable."""
        if self._available is not None:
            return self._available

        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not api_key:
            logger.warning(
                "Cohere API key not found (COHERE_API_KEY or CO_API_KEY). "
                "Reranking disabled -- returning original result order."
            )
            self._available = False
            return False

        try:
            import cohere
            self._client = cohere.ClientV2(api_key=api_key)
            self._available = True
            return True
        except ImportError:
            logger.warning(
                "cohere package not installed. "
                "Install with: pip install cohere"
            )
            self._available = False
            return False

    def rerank(self, query, documents, top_k=None):
        if not self._init_client():
            # Graceful degradation: return original order with synthetic scores
            return [
                {"corpus_id": i, "score": 1.0 - (i * 0.01), "text": doc}
                for i, doc in enumerate(documents[:top_k])
            ]

        response = self._client.rerank(
            model=self.MODEL,
            query=query,
            documents=documents,
            top_n=top_k,
        )
        return [
            {"corpus_id": r.index, "score": r.relevance_score, "text": documents[r.index]}
            for r in response.results
        ]
```

### Pattern 4: Service Reranking Integration
**What:** Wire reranking into VectorSearchService between search and return steps.
**When to use:** Modify existing `match_ontology()` and optionally `query()`.
**Example:**
```python
# Source: Existing service.py match_ontology() pattern
# In VectorSearchService.__init__:
#   self._reranker = reranker  # Injected or None
#
# In VectorSearchService._get_reranker():
#   if self._reranker is None and config has reranker != none:
#       self._reranker = self._config.create_reranker()
#   return self._reranker

def match_ontology(self, term, ontology, k=5):
    # ... existing alias resolution ...
    oversampled_k = k * 4
    raw_matches = self.query(term, collection, top_k=oversampled_k)

    # NEW: Rerank step (between search and truncation)
    reranker = self._get_reranker()
    if reranker is not None:
        documents = [m["term"] for m in raw_matches]
        reranked = reranker.rerank(term, documents, top_k=k)
        # Rebuild results in reranked order
        reranked_matches = []
        for entry in reranked:
            original = raw_matches[entry["corpus_id"]]
            # Update score from reranker
            reranked_matches.append(
                _OntologyMatch(
                    term=original["term"],
                    ontology_id=original["ontology_id"],
                    score=...,  # Normalize reranker score to 0-1
                    metadata=original["metadata"],
                    distance_metric=original["distance_metric"],
                )
            )
        return reranked_matches[:k]

    # No reranker: existing truncation
    return results[:k]
```

### Pattern 5: Config Factory Extension
**What:** Add `reranker` field and `create_reranker()` to VectorSearchConfig.
**When to use:** Config-driven reranker creation.
**Example:**
```python
# Source: Existing create_backend()/create_embedder() pattern in config.py
# Add to VectorSearchConfig:

reranker: RerankerType = Field(
    default=RerankerType.none,
    description="Reranking strategy for search results",
)

def create_reranker(self):
    """Create a configured reranker instance, or None if reranker=none."""
    if self.reranker == RerankerType.none:
        return None
    if self.reranker == RerankerType.cross_encoder:
        from lobster.core.vector.rerankers.cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker()
    if self.reranker == RerankerType.cohere:
        from lobster.core.vector.rerankers.cohere_reranker import CohereReranker
        return CohereReranker()
    raise ValueError(f"Unsupported reranker: {self.reranker}")

# In from_env(), add:
reranker_str = os.environ.get("LOBSTER_RERANKER", "none")
# ... pass RerankerType(reranker_str) to constructor
```

### Anti-Patterns to Avoid
- **Module-level model loading:** CrossEncoder and Cohere client must NEVER load at import time. Always lazy-load on first `rerank()` call.
- **Hard failure on missing Cohere key:** Must NOT raise exception. Log warning and return original order per RANK-02.
- **Importing cohere at module level:** Must be import-guarded inside method body, same as `sentence_transformers` in SapBERTEmbedder.
- **Modifying score semantics:** Cross-encoder raw scores are unbounded (can be negative). Must normalize to [0, 1] for consistency with existing OntologyMatch score field (ge=0.0, le=1.0).
- **Breaking no-reranker path:** When `reranker="none"`, the pipeline must work exactly as it does today. No regressions.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-encoder inference | Custom transformer pipeline with manual tokenization | `sentence_transformers.CrossEncoder.rank()` | Handles tokenization, batching, padding, device management. `rank()` returns sorted results directly. |
| Score normalization for cross-encoder | Manual sigmoid/softmax | `CrossEncoder.rank()` output scores | The rank() scores are already consistently ordered. For OntologyMatch compatibility, use min-max normalization across the result set. |
| Cohere API client | Raw HTTP requests to Cohere API | `cohere.ClientV2.rerank()` | Official SDK handles auth, retries, error types, pagination |
| Reranker ABC design | Ad-hoc function signatures | Follow BaseEmbedder/BaseVectorBackend ABC pattern | Consistency with existing codebase. Enables MockReranker for testing. |

**Key insight:** The `sentence-transformers` library is already in the dependency tree for SapBERT. The `CrossEncoder` class is part of the same package. This means the cross-encoder reranker adds ZERO new dependencies -- it reuses the existing sentence-transformers installation.

## Common Pitfalls

### Pitfall 1: Cross-encoder score normalization
**What goes wrong:** Cross-encoder `predict()` returns raw logit scores (e.g., 8.6, -4.3) that are unbounded. OntologyMatch requires scores in [0.0, 1.0].
**Why it happens:** Cross-encoders are trained as regression models, not classification with bounded outputs.
**How to avoid:** Use min-max normalization across the result set: `normalized = (score - min_score) / (max_score - min_score)`. If all scores are equal, set all to 1.0. The `rank()` method returns scores that are consistently ordered but still need normalization for the OntologyMatch schema.
**Warning signs:** Pydantic ValidationError on OntologyMatch creation with score > 1.0 or < 0.0.

### Pitfall 2: Reranking empty or single-result sets
**What goes wrong:** Cross-encoder `rank()` with empty documents list or single document can behave unexpectedly.
**Why it happens:** Edge cases in batch processing.
**How to avoid:** Guard: if `len(documents) <= 1`, skip reranking and return as-is. Reranking a single document adds latency for zero benefit.

### Pitfall 3: Cross-encoder latency on large candidate sets
**What goes wrong:** Reranking 100 candidates takes 1-5 seconds on CPU, which can push total query time over the 2-second target.
**Why it happens:** Cross-encoder processes each (query, document) pair through a full transformer forward pass.
**How to avoid:** The 4x oversampling factor (k=5 -> 20 candidates, k=10 -> 40 candidates, k=25 -> 100 candidates) is already well-calibrated. For k=5-10, reranking 20-40 documents takes ~200ms-1s on CPU. Only k>=25 risks hitting the 2s budget. Document this constraint.
**Warning signs:** Latency tests failing at k=25+.

### Pitfall 4: Cohere SDK version mismatch
**What goes wrong:** Cohere v1 (`cohere.Client`) vs v2 (`cohere.ClientV2`) have different APIs. v1 `rerank()` returns different response structure.
**Why it happens:** Cohere recently released v2 API.
**How to avoid:** Use `cohere.ClientV2` explicitly. The response object has `.results` with `.index` and `.relevance_score` fields. Pin to `cohere>=5.0` in the import guard error message.

### Pitfall 5: Circular import with lazy __getattr__
**What goes wrong:** Adding `BaseReranker` to `lobster/core/vector/__init__.py` lazy exports can cause circular imports if reranker modules import from the vector package.
**Why it happens:** `__getattr__` triggers imports that may cycle back.
**How to avoid:** Reranker modules should import from `lobster.core.vector.rerankers.base` directly, never from `lobster.core.vector`. Follow the same pattern as `SapBERTEmbedder` importing from `lobster.core.vector.embeddings.base`.

### Pitfall 6: MockReranker score ordering
**What goes wrong:** Tests assume reranked order matches mock return order, but the service may re-sort.
**Why it happens:** Confusion between "reranker returns sorted results" vs "service re-sorts after reranking."
**How to avoid:** MockReranker should return results in a known order. Service should trust reranker ordering and NOT re-sort. Tests should verify the reranker's order is preserved.

## Code Examples

Verified patterns from official sources:

### CrossEncoder.rank() Usage
```python
# Source: https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "heart attack"
documents = [
    "myocardial infarction",
    "heart disease",
    "acute coronary syndrome",
    "diabetes mellitus",
]

results = model.rank(query, documents, top_k=3, return_documents=True)
# Returns: [
#   {"corpus_id": 0, "score": 8.607, "text": "myocardial infarction"},
#   {"corpus_id": 2, "score": 5.123, "text": "acute coronary syndrome"},
#   {"corpus_id": 1, "score": 3.456, "text": "heart disease"},
# ]
```

### CrossEncoder.predict() for Pair Scoring
```python
# Source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([
    ("heart attack", "myocardial infarction"),
    ("heart attack", "heart disease"),
])
# scores: array([8.607, 3.456], dtype=float32)
# Higher = more relevant
```

### Cohere Rerank
```python
# Source: https://docs.cohere.com/reference/rerank
import cohere

co = cohere.ClientV2(api_key="your-key")
response = co.rerank(
    model="rerank-v4.0-pro",
    query="heart attack",
    documents=[
        "myocardial infarction",
        "heart disease",
        "diabetes mellitus",
    ],
    top_n=2,
)
for r in response.results:
    print(f"Index: {r.index}, Score: {r.relevance_score}")
# Results are sorted by relevance_score (0-1 range)
```

### MockReranker for Testing (follows MockEmbedder pattern)
```python
# Source: Derived from tests/unit/core/vector/test_vector_search_service.py MockEmbedder pattern
from lobster.core.vector.rerankers.base import BaseReranker


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit tests."""

    def __init__(self, reverse: bool = True):
        """If reverse=True, reverses document order (simulates reranking)."""
        self._reverse = reverse
        self._last_query = None
        self._last_documents = None

    def rerank(self, query, documents, top_k=None):
        self._last_query = query
        self._last_documents = documents
        indices = list(range(len(documents)))
        if self._reverse:
            indices = list(reversed(indices))
        results = [
            {"corpus_id": i, "score": 1.0 - (idx * 0.1), "text": documents[i]}
            for idx, i in enumerate(indices)
        ]
        return results[:top_k] if top_k else results
```

### Min-Max Score Normalization
```python
# For converting unbounded cross-encoder scores to [0, 1] for OntologyMatch
def normalize_scores(results: list[dict]) -> list[dict]:
    """Normalize reranker scores to [0.0, 1.0] range."""
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        # All same score -- set all to 1.0
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = round((r["score"] - min_s) / (max_s - min_s), 4)
    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| BM25 only | BM25 + cross-encoder reranking | 2021 (MS MARCO leaderboard) | 10-15% NDCG improvement standard |
| Custom transformer scoring | sentence-transformers CrossEncoder.rank() | 2023 (sbert v2.3+) | Built-in rank() method simplifies pipeline |
| Cohere v1 API (Client) | Cohere v2 API (ClientV2) | 2024 | New response format, better typing |
| ms-marco-MiniLM-L-12 | ms-marco-MiniLM-L-6-v2 | 2022 | Half the layers, minimal quality loss, 2x faster |

**Deprecated/outdated:**
- `cohere.Client` (v1): Use `cohere.ClientV2` instead. Different response structure.
- CrossEncoder without `.rank()`: Older pattern used `.predict()` + manual sorting. `.rank()` does both in one call.

## Open Questions

1. **Score normalization strategy for cross-encoder**
   - What we know: Cross-encoder scores are raw logits (unbounded). OntologyMatch requires [0, 1]. Min-max normalization works for per-query relative ranking.
   - What's unclear: Should we use sigmoid instead? Min-max makes scores relative per query (not comparable across queries). Sigmoid gives absolute interpretability but may cluster scores.
   - Recommendation: Use min-max normalization. Scores are already only compared within a single query context (one `match_ontology()` call). Cross-query comparison is not a current use case.

2. **Should `query()` also support reranking, or only `match_ontology()`?**
   - What we know: `match_ontology()` is the primary API used by agents. `query()` is lower-level.
   - What's unclear: Future callers of `query()` may want reranking.
   - Recommendation: Add reranking to `match_ontology()` only for Phase 4. `query()` stays raw. This keeps the scope tight and the lower-level API predictable.

3. **Cohere model selection**
   - What we know: Cohere offers `rerank-v4.0-pro` (latest). Previous models include `rerank-english-v3.0`.
   - What's unclear: Pricing and rate limits for the latest model vs older ones.
   - Recommendation: Default to `rerank-v4.0-pro` but make the model name configurable via an env var `COHERE_RERANK_MODEL` for flexibility.

## Sources

### Primary (HIGH confidence)
- `lobster/core/vector/service.py` -- Existing VectorSearchService with 4x oversampling and match_ontology()
- `lobster/core/vector/config.py` -- Existing VectorSearchConfig with factory pattern
- `lobster/core/vector/embeddings/sapbert.py` -- Lazy loading pattern reference
- `lobster/core/schemas/search.py` -- RerankerType enum already defined
- `tests/unit/core/vector/test_vector_search_service.py` -- MockEmbedder/MockBackend testing patterns
- HuggingFace model card: cross-encoder/ms-marco-MiniLM-L-6-v2 -- 22.7M params, 91MB, MRR@10=39.01
- sentence-transformers PyPI (v5.2.3) -- CrossEncoder class with rank() method, Python 3.10+
- Cohere Rerank API docs -- v2 endpoint, ClientV2, response format with index + relevance_score

### Secondary (MEDIUM confidence)
- sentence-transformers CrossEncoder docs (sbert.net) -- rank() method signature with return_documents parameter
- Cohere Python SDK PyPI (v5.20.6) -- ClientV2 class, MIT license

### Tertiary (LOW confidence)
- None -- all claims verified with primary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- sentence-transformers already in use, CrossEncoder is part of same package. Cohere SDK is well-documented official client.
- Architecture: HIGH -- follows existing BaseEmbedder/BaseVectorBackend ABC pattern exactly. Config factory pattern already established.
- Pitfalls: HIGH -- cross-encoder score normalization is a well-known issue. Cohere degradation pattern is straightforward. Verified against actual API responses.

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (stable -- sentence-transformers and Cohere have stable APIs)
