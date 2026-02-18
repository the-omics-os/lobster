# Phase 2: Service Integration - Research

**Researched:** 2026-02-18
**Domain:** Ontology service integration (match_ontology API, OBO graph, DiseaseOntologyService migration, 3-tuple provenance)
**Confidence:** HIGH

## Summary

Phase 2 builds on the Phase 1 vector plumbing (ChromaDB backend, SapBERT embedder, VectorSearchService) to deliver three integrated subsystems: (1) a `match_ontology()` method on VectorSearchService that resolves ontology aliases and returns typed `OntologyMatch` objects with provenance; (2) an ontology graph module that loads OBO files via obonet into cached NetworkX graphs for parent/child/sibling traversal; and (3) the Strangler Fig migration of DiseaseOntologyService to branch on `backend="embeddings"` vs `backend="json"` with zero API changes for callers.

The technical domain is well-understood. Phase 1 delivered the `query()` and `query_batch()` primitives that return flat dicts. Phase 2 wraps these with domain semantics: alias resolution (`"disease"` -> `"mondo"` collection), typed Pydantic results instead of raw dicts, provenance via AnalysisStep 3-tuples, and a converter from `OntologyMatch` to the existing `DiseaseMatch` schema. The ontology graph module is straightforward: obonet parses OBO files into NetworkX MultiDiGraph, and we wrap traversal with `get_neighbors()` behind `@lru_cache` for the graph loading. The DiseaseOntologyService migration is the most sensitive part -- it must preserve the exact API contract (`match_disease(query, k, min_confidence) -> List[DiseaseMatch]`) while swapping the internal implementation from keyword substring matching to vector search delegation.

**Primary recommendation:** Implement in three logical subsystems -- (A) ontology graph module in `lobster/core/vector/ontology_graph.py`, (B) `match_ontology()` + `ONTOLOGY_COLLECTIONS` on VectorSearchService in `lobster/core/vector/service.py`, and (C) DiseaseOntologyService backend branching + `_convert_ontology_match()` in the lobster-metadata package. The duplicate `disease_ontology.json` in `lobster/config/` gets deleted. Tests follow TDD per Phase 1 patterns.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| **SRCH-01** | VectorSearchService.match_ontology(term, ontology, k) returns List[OntologyMatch] with confidence scores | Add `match_ontology()` to existing VectorSearchService. Uses `query()` internally, resolves ontology alias via ONTOLOGY_COLLECTIONS, converts raw dicts to OntologyMatch Pydantic models, returns typed list. |
| **SRCH-02** | Two-stage pipeline: embed query -> search backend (oversample k*4) -> rerank -> return top-k | `match_ontology()` oversamples by requesting `k * 4` from backend, then truncates to `k` after sorting. Reranking slot is a no-op pass-through in Phase 2 (reranker=None), activated in Phase 4 when cross-encoder is added. |
| **SRCH-03** | ONTOLOGY_COLLECTIONS mapping resolves aliases ("disease" -> "mondo", "tissue" -> "uberon", "cell_type" -> "cell_ontology") | Dict constant `ONTOLOGY_COLLECTIONS` in service.py or a dedicated constants module. Maps user-friendly names to versioned collection names. |
| **SRCH-04** | Query results include ontology term IDs (MONDO:XXXX, UBERON:XXXX, CL:XXXX), names, and confidence scores | Already in Phase 1 schema: OntologyMatch has `ontology_id`, `term`, `score`. match_ontology() returns these typed objects instead of raw dicts. |
| **SCHM-03** | OntologyMatch is compatible with existing DiseaseMatch via helper converter | `_convert_ontology_match(match: OntologyMatch) -> DiseaseMatch` maps fields: ontology_id->disease_id, term->name, score->confidence, match_type="semantic_embedding". Converter lives in DiseaseOntologyService (lobster-metadata package). |
| **GRPH-01** | load_ontology_graph() parses OBO files via obonet into NetworkX MultiDiGraph with @lru_cache | `load_ontology_graph(ontology: str) -> nx.MultiDiGraph`. Uses obonet.read_obo() with URL from OBO_URLS mapping. @lru_cache on function for process-lifetime caching. obonet is import-guarded (optional dep). |
| **GRPH-02** | get_neighbors(graph, term_id, depth) returns parent/child/sibling terms | Three traversal functions: parents via networkx.descendants (note: edge direction in OBO is child->parent), children via networkx.ancestors, siblings as children-of-parents minus self. Depth parameter for multi-hop traversal. |
| **GRPH-03** | OBO_URLS mapping covers MONDO, Uberon, and Cell Ontology | Dict constant: `{"mondo": "https://purl.obolibrary.org/obo/mondo.obo", "uberon": "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo", "cell_ontology": "https://purl.obolibrary.org/obo/cl.obo"}`. |
| **MIGR-01** | DiseaseOntologyService branches on config.backend field ("json" = keyword, "embeddings" = vector search) | In `__init__`, check `self._config.backend`. If "json" -> current keyword path. If "embeddings" -> store VectorSearchService reference for delegation. Branching in `match_disease()` method. |
| **MIGR-02** | When backend="embeddings", match_disease() delegates to VectorSearchService.match_ontology("mondo") | `match_disease()` calls `self._vector_service.match_ontology(query, "mondo", k)`, then converts results via `_convert_ontology_match()`, applies min_confidence filter. |
| **MIGR-03** | _convert_ontology_match() maps OntologyMatch to existing DiseaseMatch schema | Field mapping: OntologyMatch.ontology_id -> DiseaseMatch.disease_id, OntologyMatch.term -> DiseaseMatch.name, OntologyMatch.score -> DiseaseMatch.confidence, match_type="semantic_embedding", matched_term=original query. |
| **MIGR-04** | Keyword matching remains as fallback when vector deps not installed (with explicit logger.warning) | When backend="embeddings" but VectorSearchService import fails -> `logger.warning("Vector search deps not installed, falling back to keyword matching")` -> use existing keyword path. |
| **MIGR-05** | DiseaseStandardizationService silent fallback fixed with logger.warning on ImportError | Current code has `try/except ImportError` with `HAS_ONTOLOGY_SERVICE = False` but no logging. Add `logger.warning("DiseaseOntologyService not available, using hardcoded fallback")`. |
| **MIGR-06** | Duplicate disease_ontology.json deleted from lobster/config/ (canonical stays in lobster-metadata) | `lobster/config/disease_ontology.json` is identical to `packages/lobster-metadata/lobster/config/disease_ontology.json` (confirmed via diff). Delete the core copy. |
| **TEST-04** | Unit tests for VectorSearchService orchestration, caching, config-driven switching | Tests for match_ontology(): alias resolution, OntologyMatch return type, oversampling k*4, empty results, invalid ontology name. Uses existing MockEmbedder/MockVectorBackend pattern. |
| **TEST-05** | Unit tests for config env var parsing and factory methods | Already 11 tests exist from Phase 1 (test_config.py). Phase 2 adds tests for ONTOLOGY_COLLECTIONS constants and any new config fields. |
| **TEST-06** | Unit tests for DiseaseOntologyService Phase 2 backend swap branching | Tests for: backend="json" uses keyword path, backend="embeddings" delegates to VectorSearchService, _convert_ontology_match() field mapping, fallback when vector deps missing with logger.warning verification. |
| **TEST-07** | Integration test with small real ChromaDB (embed -> search -> rerank full pipeline) | @pytest.mark.skipif for chromadb/sentence-transformers. Create small test collection (10 terms), embed + add, query, verify OntologyMatch results. Tests full pipeline without mocks. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| obonet | >=1.1.0 | Parse OBO ontology files into NetworkX MultiDiGraph | Standard OBO parser for Python. Returns networkx.MultiDiGraph with term metadata as node attributes. Supports reading from URLs with automatic compression detection. Lightweight (only depends on networkx). |
| networkx | >=3.0.0 (already installed: 3.6.1) | Graph traversal for ontology parent/child/sibling relationships | Already a transitive dependency via scanpy. Provides descendants(), ancestors(), and general graph traversal. Used directly by proteomics_visualization_service already. |
| pydantic | >=2.0.0 (already core dep) | OntologyMatch, DiseaseMatch schema models | Already used throughout Lobster. Phase 1 already defined OntologyMatch and SearchResult models. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| chromadb | >=1.0.0,<2.0.0 (Phase 1 dep) | Vector store backend | Already configured from Phase 1. Used by integration tests (TEST-07). |
| sentence-transformers | >=4.0.0,<6.0.0 (Phase 1 dep) | SapBERT embedding generation | Already configured from Phase 1. Used by integration tests (TEST-07). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| obonet for OBO parsing | pronto (full OWL/OBO parser) | pronto is heavier, supports OWL, but obonet is simpler and returns networkx directly. We only need OBO, not OWL. obonet is the right tool. |
| networkx graph traversal | Custom tree traversal | networkx already installed, provides well-tested graph algorithms. No reason to hand-roll. |
| @lru_cache for graph caching | Custom singleton cache | lru_cache is stdlib, simple, and sufficient for process-lifetime caching of 3 ontology graphs. No distributed caching needed. |

**Installation (human must add to pyproject.toml):**
```toml
[project.optional-dependencies]
vector-search = [
    "chromadb>=1.0.0,<2.0.0",
    "sentence-transformers>=4.0.0,<6.0.0",
    "obonet>=1.1.0",
]
```

Note: networkx is NOT added as a direct dependency since it's already transitively available via scanpy. obonet depends on networkx and will use the already-installed version.

## Architecture Patterns

### Recommended Module Structure (Phase 2 additions)

```
lobster/core/vector/
    __init__.py              # Add: match_ontology to __getattr__ lazy exports
    config.py                # (unchanged from Phase 1)
    service.py               # Add: match_ontology(), ONTOLOGY_COLLECTIONS
    ontology_graph.py        # NEW: load_ontology_graph(), get_neighbors(), OBO_URLS
    backends/                # (unchanged from Phase 1)
    embeddings/              # (unchanged from Phase 1)

lobster/core/schemas/
    search.py                # (unchanged from Phase 1 - OntologyMatch already defined)
    ontology.py              # (unchanged - DiseaseMatch, DiseaseOntologyConfig already defined)

packages/lobster-metadata/lobster/services/metadata/
    disease_ontology_service.py  # MODIFIED: Add backend branching, _convert_ontology_match()

packages/lobster-metadata/lobster/services/metadata/
    disease_standardization_service.py  # MODIFIED: Add logger.warning on fallback

lobster/config/
    disease_ontology.json    # DELETED (duplicate of lobster-metadata canonical)
```

### Pattern 1: ONTOLOGY_COLLECTIONS Alias Resolution

**What:** Maps user-friendly ontology names and common aliases to versioned ChromaDB collection names.
**When to use:** In `match_ontology()` to resolve the `ontology` parameter.

```python
# Source: SRCH-03 requirement + Phase 1 collection naming convention
ONTOLOGY_COLLECTIONS: dict[str, str] = {
    # Primary names
    "mondo": "mondo_v2024_01",
    "uberon": "uberon_v2024_01",
    "cell_ontology": "cell_ontology_v2024_01",
    # Aliases (resolve to primary)
    "disease": "mondo_v2024_01",
    "tissue": "uberon_v2024_01",
    "cell_type": "cell_ontology_v2024_01",
}
```

### Pattern 2: match_ontology() with Oversampling Slot

**What:** Domain-aware query method on VectorSearchService that returns typed OntologyMatch objects.
**When to use:** Any agent/service that wants to match a biomedical term to an ontology concept.

```python
# Source: SRCH-01, SRCH-02, SRCH-04 requirements
from lobster.core.schemas.search import OntologyMatch

def match_ontology(
    self,
    term: str,
    ontology: str,
    k: int = 5,
) -> list[OntologyMatch]:
    """
    Match a biomedical term to ontology concepts.

    Args:
        term: Query text (e.g., "heart attack", "lung tissue").
        ontology: Ontology name or alias (e.g., "disease", "mondo", "tissue").
        k: Number of results to return.

    Returns:
        list[OntologyMatch]: Typed match objects with term, ontology_id, score.

    Raises:
        ValueError: If ontology name not found in ONTOLOGY_COLLECTIONS.
    """
    collection = ONTOLOGY_COLLECTIONS.get(ontology)
    if collection is None:
        raise ValueError(
            f"Unknown ontology '{ontology}'. "
            f"Available: {list(ONTOLOGY_COLLECTIONS.keys())}"
        )

    # Oversample for future reranking (Phase 4 activates reranker here)
    oversample_k = k * 4
    raw_matches = self.query(term, collection, top_k=oversample_k)

    # Convert raw dicts to typed OntologyMatch objects
    results = []
    for match_dict in raw_matches[:k]:  # Truncate to requested k
        results.append(OntologyMatch(
            term=match_dict["term"],
            ontology_id=match_dict["ontology_id"],
            score=match_dict["score"],
            metadata=match_dict["metadata"],
            distance_metric=match_dict["distance_metric"],
        ))

    return results
```

### Pattern 3: Ontology Graph with lru_cache

**What:** Load OBO files via obonet, cache with lru_cache, provide traversal utilities.
**When to use:** When agents need parent/child/sibling context for ontology terms.

```python
# Source: GRPH-01, GRPH-02, GRPH-03 requirements + obonet docs
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

OBO_URLS: dict[str, str] = {
    "mondo": "https://purl.obolibrary.org/obo/mondo.obo",
    "uberon": "https://purl.obolibrary.org/obo/uberon/uberon-basic.obo",
    "cell_ontology": "https://purl.obolibrary.org/obo/cl.obo",
}

@lru_cache(maxsize=3)
def load_ontology_graph(ontology: str) -> "nx.MultiDiGraph":
    """
    Load an OBO ontology file into a NetworkX MultiDiGraph.

    Cached for process lifetime (3 ontologies max).
    Downloads from OBO Foundry URLs on first call.

    Args:
        ontology: Ontology name (e.g., "mondo", "uberon", "cell_ontology").

    Returns:
        networkx.MultiDiGraph with ontology terms as nodes.

    Raises:
        ValueError: If ontology name not in OBO_URLS.
        ImportError: If obonet not installed.
    """
    try:
        import obonet
    except ImportError:
        raise ImportError(
            "Ontology graph requires obonet. "
            "Install with: pip install 'lobster-ai[vector-search]'"
        )

    url = OBO_URLS.get(ontology)
    if url is None:
        raise ValueError(
            f"Unknown ontology '{ontology}'. Available: {list(OBO_URLS.keys())}"
        )

    logger.info(f"Loading {ontology} ontology graph from {url}...")
    graph = obonet.read_obo(url)
    logger.info(
        f"Loaded {ontology}: {len(graph)} terms, "
        f"{graph.number_of_edges()} edges"
    )
    return graph


def get_neighbors(
    graph: "nx.MultiDiGraph",
    term_id: str,
    depth: int = 1,
    relation: str = "all",
) -> dict[str, list[dict[str, str]]]:
    """
    Get parent, child, and sibling terms for an ontology term.

    Note on OBO edge direction: edges go FROM child TO parent (is_a).
    So networkx.descendants(graph, X) gives PARENTS of X,
    and networkx.ancestors(graph, X) gives CHILDREN of X.

    Args:
        graph: NetworkX MultiDiGraph from load_ontology_graph().
        term_id: Ontology term ID (e.g., "MONDO:0005575").
        depth: How many hops to traverse (1 = immediate neighbors).
        relation: Filter to specific relation type, or "all".

    Returns:
        dict with keys "parents", "children", "siblings", each a list of
        dicts with "term_id" and "name".
    """
    import networkx as nx

    if term_id not in graph:
        return {"parents": [], "children": [], "siblings": []}

    def _format_term(tid: str) -> dict[str, str]:
        name = graph.nodes[tid].get("name", "") if tid in graph else ""
        return {"term_id": tid, "name": name}

    # Parents: follow edges outward (is_a direction)
    if depth == 1:
        parents = [t for t in graph.successors(term_id)]
    else:
        parents = list(nx.descendants(graph, term_id))

    # Children: follow edges inward
    if depth == 1:
        children = [t for t in graph.predecessors(term_id)]
    else:
        children = list(nx.ancestors(graph, term_id))

    # Siblings: other children of my parents (depth=1 only for siblings)
    siblings = set()
    for parent in (graph.successors(term_id)):
        for sibling in graph.predecessors(parent):
            if sibling != term_id:
                siblings.add(sibling)

    return {
        "parents": [_format_term(t) for t in parents],
        "children": [_format_term(t) for t in children],
        "siblings": [_format_term(t) for t in siblings],
    }
```

### Pattern 4: DiseaseOntologyService Backend Branching (Strangler Fig)

**What:** Branch match_disease() implementation based on config.backend field.
**When to use:** In DiseaseOntologyService to swap from keyword to vector search.

```python
# Source: MIGR-01, MIGR-02, MIGR-03, MIGR-04 requirements
def __init__(self, config_path=None):
    # ... existing JSON loading code ...
    self._config = self._load_from_json(config_path)

    # Phase 2: Initialize vector backend if configured
    self._vector_service = None
    if self._config.backend == "embeddings":
        try:
            from lobster.core.vector.service import VectorSearchService
            self._vector_service = VectorSearchService()
            logger.info("DiseaseOntologyService using embeddings backend")
        except ImportError:
            logger.warning(
                "Vector search deps not installed, falling back to keyword matching. "
                "Install with: pip install 'lobster-ai[vector-search]'"
            )
            # Fall through to keyword path

    # Always build keyword index (needed for fallback)
    self._diseases = self._config.diseases
    self._keyword_index = self._build_keyword_index()

def match_disease(self, query, k=3, min_confidence=0.7):
    # Phase 2: Delegate to vector search if available
    if self._vector_service is not None:
        ontology_matches = self._vector_service.match_ontology(query, "mondo", k=k)
        disease_matches = [
            self._convert_ontology_match(m, query) for m in ontology_matches
        ]
        return [m for m in disease_matches if m.confidence >= min_confidence]

    # Phase 1 fallback: keyword matching
    # ... existing keyword code unchanged ...

def _convert_ontology_match(self, match, original_query):
    """Convert OntologyMatch -> DiseaseMatch (SCHM-03)."""
    return DiseaseMatch(
        disease_id=match.ontology_id,         # "MONDO:0005575"
        name=match.term,                       # "colorectal carcinoma"
        confidence=match.score,                # 0.0-1.0
        match_type="semantic_embedding",       # Phase 2 indicator
        matched_term=original_query,           # original user query
        metadata=match.metadata,               # extensible metadata
    )
```

### Pattern 5: Silent Fallback Fix (MIGR-05)

**What:** Add explicit logger.warning when DiseaseOntologyService import fails silently.
**When to use:** In DiseaseStandardizationService and metadata_assistant config.py.

```python
# Source: MIGR-05 requirement
# Current code (WRONG - silent):
try:
    from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService
    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False  # Silent! No log.

# Fixed code (CORRECT - explicit warning):
try:
    from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService
    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False
    logger.warning(
        "DiseaseOntologyService not available, using hardcoded disease mappings. "
        "Install lobster-metadata for centralized ontology management."
    )
```

### Anti-Patterns to Avoid

- **Module-level obonet/networkx imports in ontology_graph.py:** obonet loads and parses an OBO file, which can take 5-20 seconds for MONDO (~30K terms). All imports MUST be inside functions, guarded by import check.
- **Loading OBO graph eagerly in DiseaseOntologyService:** The ontology graph is NOT needed for the Phase 2 vector search path. It's a separate feature (GRPH-01/02/03) for enriching results with parent/child context. Don't load it during DiseaseOntologyService initialization.
- **Breaking DiseaseMatch API contract:** The existing callers (metadata_assistant config.py, DiseaseStandardizationService) use `match.disease_id`, `match.confidence`, `match.match_type`. These fields MUST remain identical. Phase 2 changes `match_type` from `"exact_keyword"` to `"semantic_embedding"` and `disease_id` from `"crc"` to `"MONDO:0005575"`, which may break downstream code that does `match.disease_id == "crc"`. The converter should handle this carefully.
- **Forgetting to delete `lobster/config/disease_ontology.json`:** The duplicate must be removed. The canonical copy lives in `packages/lobster-metadata/lobster/config/disease_ontology.json`.
- **Coupling ontology_graph to VectorSearchService:** The graph module is independent infrastructure. VectorSearchService does NOT need the graph for search. Agents will call both independently: search for matching, graph for context enrichment.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OBO file parsing | Custom OBO parser | obonet.read_obo() | OBO format has subtleties (typedefs, obsolete terms, relationship types). obonet handles all of this and returns NetworkX directly. |
| Graph traversal | Custom tree walking | networkx.successors/predecessors/descendants/ancestors | networkx is already installed, well-tested, handles cycles and multi-edges. OBO ontologies are DAGs but have multi-edge relationships. |
| Ontology graph caching | Custom singleton manager | functools.lru_cache(maxsize=3) | stdlib, simple, sufficient for 3 ontology graphs cached for process lifetime. |
| OntologyMatch -> DiseaseMatch conversion | Monkey-patching DiseaseMatch | Explicit `_convert_ontology_match()` function | Clean, testable, doesn't modify either schema. Converter is isolated and easy to update. |
| Collection name versioning | Dynamic version detection | Static ONTOLOGY_COLLECTIONS dict | Version tied to data pipeline (Phase 6). Static dict is correct until automated versioning is needed. |

**Key insight:** Phase 2 is primarily an integration layer. The heavy lifting (embedding, vector search, OBO parsing) is done by libraries. Our value is in the clean API design (`match_ontology()`), the schema conversion (`_convert_ontology_match()`), and the Strangler Fig migration (backend branching in DiseaseOntologyService).

## Common Pitfalls

### Pitfall 1: OBO Edge Direction Confusion

**What goes wrong:** Using networkx.descendants() to get children when it actually returns parents in OBO graphs, and vice versa with ancestors().
**Why it happens:** OBO format defines edges from child to parent (is_a relationship). So `node -> parent` is the edge direction. networkx.descendants() follows edges forward (to parents), networkx.ancestors() follows edges backward (to children). This is counter-intuitive.
**How to avoid:** Document the direction explicitly in get_neighbors(). Use `graph.successors(term_id)` for direct parents (depth=1) and `graph.predecessors(term_id)` for direct children (depth=1). For multi-hop, use descendants (parents) and ancestors (children) with clear comments.
**Warning signs:** "children" results are actually ancestor terms. Test with known ontology: MONDO:0005575 (colorectal cancer) should have MONDO:0004992 (intestinal cancer) as a parent.

### Pitfall 2: DiseaseMatch API Break with MONDO IDs

**What goes wrong:** Existing callers check `match.disease_id == "crc"` but Phase 2 returns `match.disease_id == "MONDO:0005575"`. Tests pass but production code breaks because the ID format changed.
**Why it happens:** Phase 1 used internal short IDs (`"crc"`, `"uc"`, `"cd"`, `"healthy"`). Phase 2 uses standard ontology IDs (`"MONDO:XXXXXXX"`). The DiseaseMatch schema allows both, but callers may not expect the new format.
**How to avoid:** The `_convert_ontology_match()` converter should be the ONLY place this mapping happens. Consider adding a `legacy_id` field to DiseaseMatch metadata for backward compatibility, or keeping a MONDO-to-short-ID lookup. Document the change clearly. Test callers (phase1_column_rescan, DiseaseStandardizationService) with MONDO-format IDs.
**Warning signs:** Any code that does `if disease_id == "crc"` or `if disease_id in ["crc", "uc", "cd", "healthy"]`.

### Pitfall 3: OBO File Download Blocking on First Use

**What goes wrong:** First call to `load_ontology_graph("mondo")` downloads a 30MB+ OBO file from OBO Foundry, which takes 10-30 seconds. If called in the middle of an agent response, the user sees a long hang.
**Why it happens:** obonet.read_obo() with a URL downloads the file synchronously. No progress indicator, no caching to disk.
**How to avoid:** The ontology graph is a separate feature from vector search. Don't couple it to DiseaseOntologyService initialization. Call it only when explicitly needed (e.g., agent requests parent/child context). Log a clear message before loading: "Loading MONDO ontology graph from OBO Foundry (~30MB)...". The @lru_cache ensures it only happens once per process.
**Warning signs:** Agent response time jumps from <2s to 30s+ on first ontology graph query.

### Pitfall 4: test_disease_ontology_service.py Tests Become Flaky

**What goes wrong:** Existing unit tests for DiseaseOntologyService assume keyword matching behavior (confidence=1.0, match_type="exact_keyword", disease_id="crc"). If backend is changed to "embeddings" in the JSON config, tests break.
**Why it happens:** Tests load the actual disease_ontology.json which has `"backend": "json"`. If someone changes this to `"embeddings"`, tests require vector deps.
**How to avoid:** Tests should explicitly set backend behavior. For keyword path tests, ensure config says `"backend": "json"`. For vector path tests, mock VectorSearchService. Don't depend on the default config file for test behavior. Use fixture-based config injection.
**Warning signs:** Tests that pass locally but fail in CI (where vector deps aren't installed).

### Pitfall 5: Circular Import Between vector/service.py and schemas/ontology.py

**What goes wrong:** VectorSearchService imports OntologyMatch from schemas/search.py. DiseaseOntologyService imports VectorSearchService. If schemas/ontology.py imports from schemas/search.py, we could create import chains.
**Why it happens:** Phase 2 introduces new cross-module dependencies. OntologyMatch is in core schemas, DiseaseMatch is in core schemas, converter lives in lobster-metadata package.
**How to avoid:** Keep imports lazy in DiseaseOntologyService (import VectorSearchService inside `__init__`, not at module level). This is already the pattern for optional deps. The converter function only needs OntologyMatch and DiseaseMatch, both from core schemas -- no circular dependency there.
**Warning signs:** ImportError at module level, or slow startup from eager imports.

## Code Examples

Verified patterns from the codebase:

### Existing DiseaseMatch Schema (ontology.py)
```python
# Source: lobster/core/schemas/ontology.py (lines 16-44)
class DiseaseMatch(BaseModel):
    disease_id: str  # "crc" (Phase 1) or "MONDO:0005575" (Phase 2)
    name: str        # "Colorectal Cancer"
    confidence: float  # 1.0 (keyword) or 0.0-1.0 (embedding)
    match_type: str  # "exact_keyword" or "semantic_embedding"
    matched_term: str  # original query string
    metadata: Dict[str, Any]  # mondo_id, umls_cui, mesh_terms
```

### Existing OntologyMatch Schema (search.py)
```python
# Source: lobster/core/schemas/search.py (lines 42-73)
class OntologyMatch(BaseModel):
    term: str          # "colorectal carcinoma"
    ontology_id: str   # "MONDO:0005575"
    score: float       # 0.0-1.0 (cosine similarity)
    metadata: dict     # extensible
    distance_metric: str  # "cosine"
```

### Existing MockEmbedder/MockVectorBackend Test Pattern
```python
# Source: tests/unit/core/vector/test_vector_search_service.py (lines 24-111)
class MockEmbedder(BaseEmbedder):
    DIMENSIONS = 768
    def embed_text(self, text): ...
    def embed_batch(self, texts): ...
    @property
    def dimensions(self): return self.DIMENSIONS

class MockVectorBackend(BaseVectorBackend):
    def set_results(self, results): ...
    def add_documents(self, ...): ...
    def search(self, ...): ...
    def delete(self, ...): ...
    def count(self, ...): ...
```

### Existing DiseaseOntologyService match_disease() Pattern
```python
# Source: packages/lobster-metadata/.../disease_ontology_service.py (lines 153-234)
def match_disease(self, query, k=3, min_confidence=0.7) -> List[DiseaseMatch]:
    query_lower = query.lower()
    matches = []
    for disease in self._diseases:
        for keyword in disease.keywords:
            if keyword.lower() in query_lower:
                matches.append(DiseaseMatch(
                    disease_id=disease.id,
                    name=disease.name,
                    confidence=1.0,
                    match_type="exact_keyword",
                    matched_term=keyword,
                    metadata={...},
                ))
    matches = [m for m in matches if m.confidence >= min_confidence]
    return matches[:k]
```

### Existing Silent Fallback in DiseaseStandardizationService
```python
# Source: packages/lobster-metadata/.../disease_standardization_service.py (lines 19-27)
try:
    from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService
    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False  # <-- Silent! MIGR-05 requires logger.warning here
```

### 3-Tuple Provenance Pattern (from DiseaseStandardizationService)
```python
# Source: packages/lobster-metadata/.../disease_standardization_service.py (lines 198-265)
def standardize_disease_terms(self, metadata, disease_column):
    # ... processing ...
    ir = AnalysisStep(
        operation="disease_standardization",
        tool_name="standardize_disease_terms",
        description="...",
        library="custom",
        imports=[],
        code_template="...",
        parameters={...},
        parameter_schema={...},
        input_entities=["metadata"],
        output_entities=["standardized_metadata"],
    )
    return result, stats, ir
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Keyword substring matching for disease terms | Embedding-based semantic search via ChromaDB + SapBERT | Phase 2 (this phase) | Matches synonyms ("colon tumor" -> "colorectal carcinoma") that keyword matching misses. Confidence scores enable threshold-based filtering. |
| Hardcoded disease_ontology.json in both core and package | Single canonical copy in lobster-metadata package | Phase 2 (this phase) | Eliminates duplicate maintenance. DiseaseOntologyService is the single source of truth. |
| Silent ImportError fallback (no logging) | Explicit logger.warning on fallback | Phase 2 (this phase) | Debugging visibility. Users know when they're running with degraded functionality. |
| No ontology graph traversal | obonet + NetworkX for parent/child/sibling relationships | Phase 2 (this phase) | Agents can enrich search results with ontology context ("colorectal cancer IS_A intestinal cancer"). |

**Deprecated/outdated:**
- Direct `disease_id="crc"` comparisons: Phase 2 introduces MONDO IDs. Callers should handle both formats during migration or use the centralized service.
- `get_extraction_keywords()` and `get_standardization_variants()`: Already marked as DEPRECATED in DiseaseOntologyService. Phase 2 doesn't remove them but they should be migrated to use `match_disease()`.

## Open Questions

1. **MONDO ID vs legacy short ID in DiseaseMatch.disease_id**
   - What we know: Phase 1 uses `"crc"`, Phase 2 will use `"MONDO:0005575"`. DiseaseMatch schema supports both. Callers like `phase1_column_rescan()` set `sample["disease"] = best_match.disease_id` which downstream code may compare against `"crc"`.
   - What's unclear: Whether changing from `"crc"` to `"MONDO:0005575"` breaks any downstream pipeline. The metadata_assistant uses the disease_id to label samples, and DiseaseStandardizationService maps it to standard categories.
   - Recommendation: In Phase 2, the converter should keep `disease_id` as the MONDO ID (correct for Phase 2 semantics). Add `metadata["legacy_id"]` with the short ID for backward compatibility. Test all callers with both formats. Flag this as a migration concern to resolve before Phase 3 agent integration.

2. **OBO file size and download time**
   - What we know: MONDO OBO is ~30MB, Uberon ~15MB, Cell Ontology ~8MB. obonet downloads synchronously.
   - What's unclear: Exact download + parse time on typical user hardware. Whether we need to cache the parsed graph to disk (pickle) or if lru_cache per-process is sufficient.
   - Recommendation: For Phase 2, lru_cache is sufficient. Graph loading happens only when agents explicitly request ontology context, not during search. If profiling shows unacceptable latency, Phase 6 can add disk caching alongside the embedding tarballs.

3. **3-tuple return for match_ontology()**
   - What we know: Phase 2 success criteria says "All service methods return (result, stats, AnalysisStep) 3-tuple." But `match_ontology()` is on VectorSearchService which doesn't currently use 3-tuples -- it returns flat lists.
   - What's unclear: Whether `match_ontology()` should return `(List[OntologyMatch], Dict, AnalysisStep)` or just `List[OntologyMatch]`. The DiseaseOntologyService callers don't expect 3-tuples from `match_disease()` either.
   - Recommendation: Keep `match_ontology()` returning `List[OntologyMatch]` for simplicity (callers like DiseaseOntologyService just want the matches). The 3-tuple success criteria applies to the DiseaseStandardizationService (which already returns 3-tuples) and any new Phase 2 service methods that wrap match_ontology(). The `match_ontology()` itself is an infrastructure method, not an analysis service.

4. **obonet as optional dependency vs always-installed**
   - What we know: obonet is lightweight (~50KB) and only depends on networkx (already installed). But it's only needed for GRPH-01/02/03, not for the core search path (SRCH-01/02/03/04).
   - What's unclear: Whether to put obonet in the `vector-search` extra or make it a separate `ontology-graph` extra.
   - Recommendation: Include obonet in the `vector-search` extra alongside chromadb and sentence-transformers. It's lightweight enough that bundling it doesn't hurt, and users who want vector search likely want ontology graph context too. Import-guard it in `ontology_graph.py` with the same helpful message pattern.

## Sources

### Primary (HIGH confidence)
- **Phase 1 codebase artifacts** -- All Phase 1 files verified on disk: `lobster/core/vector/service.py` (VectorSearchService), `lobster/core/schemas/search.py` (OntologyMatch), `lobster/core/schemas/ontology.py` (DiseaseMatch, DiseaseOntologyConfig), Phase 1 verification report (8/8 passed).
- **Existing DiseaseOntologyService** -- `packages/lobster-metadata/lobster/services/metadata/disease_ontology_service.py` read in full (303 lines). Keyword matching implementation, singleton pattern, match_disease() API with k and min_confidence params.
- **Existing DiseaseStandardizationService** -- `packages/lobster-metadata/lobster/services/metadata/disease_standardization_service.py` read in full (592 lines). Silent ImportError fallback confirmed at lines 19-27.
- **disease_ontology.json duplicate** -- Confirmed identical via `diff` between `lobster/config/disease_ontology.json` and `packages/lobster-metadata/lobster/config/disease_ontology.json`.
- **obonet PyPI page** (pypi.org/project/obonet) -- Version 1.1.1, released 2025-03-26, depends on networkx.
- **obonet source code** -- `read_obo()` returns `networkx.MultiDiGraph[str]`, supports URL loading, skips obsolete terms by default.
- **OBO Foundry** -- MONDO URL: `https://purl.obolibrary.org/obo/mondo.obo`, Uberon: `https://purl.obolibrary.org/obo/uberon/uberon-basic.obo`, CL: `https://purl.obolibrary.org/obo/cl.obo`.
- **networkx 3.6.1** -- Already installed in lobster venv (transitive via scanpy). Provides successors/predecessors/descendants/ancestors for graph traversal.

### Secondary (MEDIUM confidence)
- **obonet README** (github.com/dhimmel/obonet) -- OBO edge direction convention (child->parent), descendants returns superterms, ancestors returns subterms. Verified against read.py source.
- **Existing test patterns** -- `tests/unit/core/vector/test_vector_search_service.py` (MockEmbedder, MockVectorBackend), `packages/lobster-metadata/tests/services/metadata/test_disease_ontology_service.py` (singleton tests, keyword matching).

### Tertiary (LOW confidence)
- **OBO file sizes** -- MONDO ~30MB, Uberon ~15MB, CL ~8MB. Estimated from typical OBO Foundry file sizes. Exact sizes not verified (would require downloading). LOW confidence on download times.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- obonet verified via PyPI and source. networkx already installed. All APIs verified against docs.
- Architecture: HIGH -- All integration points verified by reading existing Phase 1 code and DiseaseOntologyService source. Schema compatibility confirmed (OntologyMatch and DiseaseMatch are already defined with compatible fields).
- Pitfalls: HIGH -- Edge direction confirmed from obonet source. DiseaseMatch API break risk identified from reading callers. Silent fallback confirmed from DiseaseStandardizationService source.
- Migration patterns: HIGH -- Strangler Fig approach documented in existing schema comments. Backend field already in DiseaseOntologyConfig. match_disease() API designed for Phase 2 compatibility.

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (30 days -- stable libraries, no fast-moving changes expected)
