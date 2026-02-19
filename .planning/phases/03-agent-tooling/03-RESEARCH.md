# Phase 3: Agent Tooling - Research

**Researched:** 2026-02-18
**Domain:** Agent tool integration (annotation_expert + metadata_assistant with VectorSearchService)
**Confidence:** HIGH

## Summary

Phase 3 adds three new semantic search tools to two existing agents: `annotate_cell_types_semantic` for annotation_expert, and `standardize_tissue_term` + `standardize_disease_term` for metadata_assistant. All infrastructure is already in place from Phases 1-2 -- the VectorSearchService with `match_ontology()`, OntologyMatch schema, ONTOLOGY_COLLECTIONS alias map, ontology graph traversal (`get_neighbors()`), and DiseaseOntologyService with backend branching. This phase is purely about wiring those existing services into agent tool closures following established patterns.

The codebase has clear, consistent patterns for both agents. annotation_expert uses a factory function with tool closures, services initialized at the top, and a `base_tools` list assembled at the bottom. metadata_assistant follows the same pattern with conditional tool registration (e.g., `if MICROBIOME_FEATURES_AVAILABLE`). The new semantic tools will use the same conditional registration approach: check VectorSearchService importability, and only add semantic tools to the tool list if deps are available.

**Primary recommendation:** Follow the existing tool closure + lazy init pattern from `kevin_notes/vector_search_implementation.md` Section 4.1/4.2. Use mock backend/embedder injection for all unit tests (same pattern as `tests/unit/core/vector/test_vector_search_service.py`). No new service classes needed -- tools call VectorSearchService and DiseaseOntologyService directly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Query construction
- annotation_expert: format marker gene signatures as text queries -- `"Cluster 0: high CD3D, CD3E, CD8A"` -- using top markers per cluster from existing `_calculate_marker_scores_from_adata()`
- metadata_assistant: free-text passthrough -- `standardize_tissue_term(term)` passes term directly to `VectorSearchService.match_ontology(term, "uberon")`
- metadata_assistant: disease routing goes through `DiseaseOntologyService.match_disease(term)` which delegates to VectorSearchService when `backend="embeddings"` -- NOT directly to vector service

#### Result shape
- All results use `OntologyMatch` schema: term_id, name, confidence (0.0-1.0), match_type, ontology_source, metadata dict
- Annotation stores results in existing obs columns: `cell_type`, `cell_type_confidence`
- Disease standardization converts `OntologyMatch` -> `DiseaseMatch` via `_convert_ontology_match()` helper (preserving existing API)
- Default k=5 for `standardize_tissue_term` and `annotate_cell_types_semantic`
- Default k=3, min_confidence=0.7 for disease (inherited from existing `DiseaseOntologyService` API)

#### Missing deps behavior
- When vector-search dependencies aren't installed, semantic tools are **absent from agent toolkits** -- not registered at all
- Agents only see their existing tools (e.g., annotation_expert sees `annotate_cell_types` but NOT `annotate_cell_types_semantic`)
- Conditional tool registration in agent factories -- check for VectorSearchService importability before adding semantic tools to tool list

#### Graph validation
- `annotate_cell_types_semantic` has `validate_graph=False` by default
- When enabled, validates results via `get_neighbors()` graph traversal for biological plausibility
- Opt-in parameter -- users/agents enable explicitly when accuracy matters more than speed

#### Workflow integration
- New `annotate_cell_types_semantic` tool added alongside existing `annotate_cell_types` (unchanged)
- Two separate metadata tools: `standardize_tissue_term` (Uberon), `standardize_disease_term` (MONDO via DiseaseOntologyService)
- Lazy VectorSearchService initialization in agent factory closures
- All tools return 3-tuple `(result, stats, AnalysisStep)` and log with `ir=ir`

### Claude's Discretion
- Exact stats dict content for each tool's return value
- Error message wording when semantic search returns no results above confidence threshold
- Whether `annotate_cell_types_semantic` annotates all clusters in one call (like existing tool) or allows per-cluster annotation
- Default min_confidence for tissue standardization

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AGNT-01 | annotation_expert gains annotate_cell_types_semantic tool that queries Cell Ontology via VectorSearchService | VectorSearchService.match_ontology(term, "cell_ontology", k=5) API verified at `lobster/core/vector/service.py:175-244`. Tool closure pattern from annotation_expert factory at line 78-1293. |
| AGNT-02 | Semantic annotation uses marker gene signatures as query text ("Cluster 0: high CD3D, CD3E, CD8A") | `_calculate_marker_scores_from_adata()` at `enhanced_singlecell_service.py:1033-1101` provides per-cluster marker scores. Top markers per cluster extracted and formatted as text. |
| AGNT-03 | Existing annotate_cell_types tool remains unchanged (new tool augments, does not replace) | Confirmed: `annotate_cell_types` tool at annotation_expert.py:134-262 stays intact. New tool added to `base_tools` list at line 1264-1278. |
| AGNT-04 | metadata_assistant gains standardize_tissue_term tool using VectorSearchService.match_ontology("uberon") | VectorSearchService.match_ontology() with "uberon" alias resolves to "uberon_v2024_01" collection. Tool follows same closure pattern as existing metadata_assistant tools. |
| AGNT-05 | metadata_assistant gains standardize_disease_term tool using DiseaseOntologyService.match_disease() | DiseaseOntologyService.match_disease() at disease_ontology_service.py:173-262 already branches on backend. Tool calls this API (Strangler Fig pattern). |
| AGNT-06 | All new tools follow 3-tuple return pattern (result, stats, AnalysisStep) with ir mandatory | AnalysisStep pattern at `lobster/core/analysis_ir.py:104-179`. data_manager.log_tool_usage() at data_manager_v2.py:1657 accepts ir= kwarg. |
| TEST-08 | All tests use @pytest.mark.skipif for optional deps | Established pattern in codebase (20+ usages found). For vector deps: check importability of VectorSearchService at test module level. |
</phase_requirements>

## Standard Stack

### Core (Already Implemented in Phase 1-2)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| VectorSearchService | `lobster/core/vector/service.py` | Semantic search orchestrator | Phase 2 complete |
| OntologyMatch | `lobster/core/schemas/search.py` | Typed result schema | Phase 1 complete |
| ONTOLOGY_COLLECTIONS | `lobster/core/vector/service.py:33-42` | Alias resolution map | Phase 2 complete |
| DiseaseOntologyService | `packages/lobster-metadata/.../disease_ontology_service.py` | Disease matching with backend branching | Phase 2 complete |
| DiseaseMatch | `lobster/core/schemas/ontology.py` | Disease result schema | Phase 1 complete |
| load_ontology_graph | `lobster/core/vector/ontology_graph.py` | OBO graph loading with lru_cache | Phase 2 complete |
| get_neighbors | `lobster/core/vector/ontology_graph.py:116-189` | Graph traversal (parents/children/siblings) | Phase 2 complete |
| AnalysisStep | `lobster/core/analysis_ir.py` | IR for provenance tracking | Pre-existing |

### Integration Targets (Existing Files to Modify)

| File | Location | What Changes |
|------|----------|-------------|
| annotation_expert.py | `packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py` | Add `annotate_cell_types_semantic` tool + conditional registration |
| metadata_assistant.py | `packages/lobster-metadata/lobster/agents/metadata_assistant/metadata_assistant.py` | Add `standardize_tissue_term` + `standardize_disease_term` tools + conditional registration |
| enhanced_singlecell_service.py | `packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py` | No changes -- `_calculate_marker_scores_from_adata()` used as-is |

### No New Libraries

This phase requires zero new dependencies. All infrastructure exists. The only imports are from within the lobster codebase.

## Architecture Patterns

### Pattern 1: Lazy VectorSearchService in Factory Closures

**What:** Initialize VectorSearchService lazily within agent factory closure using `nonlocal` pattern.
**When to use:** All three new tools.
**Why:** Avoids importing heavy deps (chromadb, torch, sentence-transformers) at agent creation time.

```python
# Source: kevin_notes/vector_search_implementation.md Section 4.1
def annotation_expert(data_manager, ...):
    # Lazy VectorSearchService initialization
    _vector_service = None

    def _get_vector_service():
        nonlocal _vector_service
        if _vector_service is None:
            from lobster.core.vector.service import VectorSearchService
            _vector_service = VectorSearchService()
        return _vector_service
```

**CRITICAL:** The import is `from lobster.core.vector.service import VectorSearchService` -- NOT `from lobster.services.search`. The actual implementation lives at `lobster/core/vector/`, not where the original spec proposed.

### Pattern 2: Conditional Tool Registration

**What:** Check importability of VectorSearchService before adding semantic tools to tool list.
**When to use:** Both annotation_expert and metadata_assistant.
**Why:** When vector-search deps aren't installed, semantic tools are absent (not broken).

```python
# At module level (lightweight check)
try:
    from lobster.core.vector.service import VectorSearchService  # noqa: F401
    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False

# In factory, after base_tools assembly:
if HAS_VECTOR_SEARCH:
    base_tools.append(annotate_cell_types_semantic)
```

**Pattern precedent:** metadata_assistant already uses this exact approach for MICROBIOME_FEATURES_AVAILABLE (lines 80-97, 3202-3203).

### Pattern 3: Marker Signature Text Query Construction

**What:** Extract top N marker genes per cluster, format as human-readable text query.
**When to use:** `annotate_cell_types_semantic` tool.
**Source:** Locked decision from CONTEXT.md.

```python
# Use existing _calculate_marker_scores_from_adata() from service
cluster_scores = singlecell_service._calculate_marker_scores_from_adata(
    adata, singlecell_service.cell_type_markers, cluster_key=cluster_key
)

# For each cluster, get top marker genes and format as query
for cluster_id, scores in cluster_scores.items():
    top_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    # Format: "Cluster 0: high CD3D, CD3E, CD8A"
    top_markers = []
    for cell_type, score in top_types:
        if score > 0:
            markers = singlecell_service.cell_type_markers.get(cell_type, [])
            top_markers.extend(markers[:3])
    query = f"Cluster {cluster_id}: high {', '.join(top_markers[:5])}"
```

### Pattern 4: Three-Tuple Return with IR

**What:** All new tools return `(result, stats_dict, AnalysisStep)` and log with `ir=ir`.
**When to use:** All three new tools.
**Source:** Hard rule from CLAUDE.md and AGNT-06.

```python
# Source: lobster/core/analysis_ir.py AnalysisStep pattern
ir = AnalysisStep(
    operation="semantic_cell_type_annotation",
    tool_name="annotate_cell_types_semantic",
    description="Semantic cell type annotation via Cell Ontology vector search",
    library="lobster.core.vector",
    code_template=code_template,  # Jinja2 template
    imports=["from lobster.core.vector import VectorSearchService"],
    parameters={"modality_name": modality_name, "cluster_key": cluster_key, "k": k},
    parameter_schema={...},
    input_entities=[modality_name],
    output_entities=[f"{modality_name}_annotated"],
)

data_manager.log_tool_usage(
    tool_name="annotate_cell_types_semantic",
    parameters={...},
    description="Semantic cell type annotation using Cell Ontology",
    ir=ir,  # MANDATORY
)
```

### Pattern 5: Disease Tool via Strangler Fig

**What:** `standardize_disease_term` calls `DiseaseOntologyService.match_disease()`, NOT VectorSearchService directly.
**When to use:** Disease standardization only.
**Why:** DiseaseOntologyService handles backend branching (json/embeddings) internally. This preserves the Strangler Fig migration pattern.

```python
# Source: kevin_notes/vector_search_implementation.md Section 4.2
@tool
def standardize_disease_term(term: str, k: int = 3, min_confidence: float = 0.7) -> str:
    """Standardize a disease term using MONDO ontology."""
    if not HAS_ONTOLOGY_SERVICE:
        return "DiseaseOntologyService not available. Install lobster-metadata."
    service = DiseaseOntologyService.get_instance()
    matches = service.match_disease(term, k=k, min_confidence=min_confidence)
    # ... format response, create IR, log ...
```

### Recommended Project Structure (No New Files/Directories Needed)

```
# Files MODIFIED (all existing):
packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py
    # + annotate_cell_types_semantic tool
    # + HAS_VECTOR_SEARCH conditional import
    # + _get_vector_service() closure
    # + conditional tool registration

packages/lobster-metadata/lobster/agents/metadata_assistant/metadata_assistant.py
    # + standardize_tissue_term tool
    # + standardize_disease_term tool
    # + HAS_VECTOR_SEARCH conditional import
    # + _get_vector_service() closure
    # + conditional tool registration

# Files CREATED (tests only):
tests/unit/agents/test_annotation_expert_semantic.py
tests/unit/agents/test_metadata_assistant_semantic.py
```

### Anti-Patterns to Avoid

- **Direct VectorSearchService call for disease:** Disease MUST route through DiseaseOntologyService (Strangler Fig). Never call `VectorSearchService.match_ontology("mondo")` directly from metadata_assistant.
- **Module-level VectorSearchService instantiation:** Would pull in chromadb/torch at import time. MUST use lazy closure pattern.
- **try/except ImportError at runtime:** For tool availability, check once at module level and conditionally register. Don't catch ImportError inside the tool body.
- **Modifying existing annotate_cell_types:** New tool augments, does NOT replace. Existing tool and its tests must remain 100% unchanged.
- **Missing ir= in log_tool_usage:** Every tool call MUST pass `ir=ir`. This is a hard rule.
- **Wrong import path:** The actual vector service is at `lobster.core.vector.service`, NOT `lobster.services.search` (the original spec's proposed path was changed during Phase 1 implementation).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Semantic matching | Custom embedding/similarity code | `VectorSearchService.match_ontology()` | Already handles embed -> search -> format pipeline with alias resolution |
| Disease matching | Direct VectorSearchService call | `DiseaseOntologyService.match_disease()` | Preserves Strangler Fig pattern, handles backend branching internally |
| Marker extraction | Custom marker gene logic | `EnhancedSingleCellService._calculate_marker_scores_from_adata()` | Already computes per-cluster mean expression for all marker sets |
| Graph validation | Custom OBO parsing | `load_ontology_graph()` + `get_neighbors()` | Already provides lru_cache'd graph loading and traversal |
| Result conversion | Custom OntologyMatch -> DiseaseMatch | `DiseaseOntologyService._convert_ontology_match()` | Already implemented and tested in Phase 2 |

**Key insight:** This phase writes zero new infrastructure. Every integration point is a call to an existing, tested API. The only new code is tool function bodies and their IR creation.

## Common Pitfalls

### Pitfall 1: Wrong VectorSearchService Import Path
**What goes wrong:** Code imports from `lobster.services.search` (the original spec path) instead of `lobster.core.vector.service` (actual implementation path).
**Why it happens:** Kevin's implementation spec was written before Phase 1 and uses proposed paths. Phase 1 placed the code at `lobster/core/vector/`.
**How to avoid:** Always import from `lobster.core.vector.service import VectorSearchService` or `from lobster.core.vector import VectorSearchService`.
**Warning signs:** `ModuleNotFoundError: No module named 'lobster.services.search'`.

### Pitfall 2: OntologyMatch Field Names Differ from DiseaseMatch
**What goes wrong:** Code assumes OntologyMatch has `confidence` field but it actually has `score`. Code assumes `term_id` but it has `ontology_id`.
**Why it happens:** Original spec defined OntologyMatch with `term_id`, `name`, `confidence`, `match_type`, `ontology_source`. Actual implementation uses `term`, `ontology_id`, `score`, `metadata`, `distance_metric`.
**How to avoid:** Reference the ACTUAL schema at `lobster/core/schemas/search.py`:
  - `OntologyMatch.term` (not `name`)
  - `OntologyMatch.ontology_id` (not `term_id`)
  - `OntologyMatch.score` (not `confidence`)
  - `OntologyMatch.metadata` (dict)
  - `OntologyMatch.distance_metric`
**Warning signs:** `AttributeError: 'OntologyMatch' object has no attribute 'confidence'`.

### Pitfall 3: DiseaseOntologyService _convert_ontology_match() Uses Actual Field Names
**What goes wrong:** The existing `_convert_ontology_match()` already maps from actual OntologyMatch fields.
**Why it matters:** If you look at `disease_ontology_service.py:264-287`, the converter uses `match.ontology_id` and `match.score` -- the ACTUAL field names, not the spec's proposed names.
**How to avoid:** Use the converter as-is. It's already correct for the actual schema.

### Pitfall 4: lobster-metadata Package Is Gitignored
**What goes wrong:** Code changes to `packages/lobster-metadata/` are not tracked by git. Commits appear empty.
**Why it happens:** `.gitignore` line 8: `packages/lobster-metadata/`. This is a private/premium package.
**How to avoid:** Acknowledge in plan that metadata_assistant changes won't be git-tracked. Test locally with `pip install -e packages/lobster-metadata`. Same for `packages/lobster-transcriptomics/` -- check if it's gitignored too.

### Pitfall 5: Annotation Tool Must Handle Missing Cluster Key
**What goes wrong:** Tool fails silently or with cryptic error when cluster_key is invalid.
**Why it happens:** User may pass wrong column name.
**How to avoid:** Follow existing `annotate_cell_types` pattern (line 154-159): validate `cluster_key in adata.obs.columns` before proceeding, return helpful error listing available columns.

### Pitfall 6: `_calculate_marker_scores_from_adata` Requires Reference Markers Dict
**What goes wrong:** The method requires a markers dict parameter. Cannot be called with zero args.
**Why it happens:** The default markers are on the service instance as `self.cell_type_markers`.
**How to avoid:** Pass `singlecell_service.cell_type_markers` explicitly, or allow user to provide custom markers.

## Code Examples

### Example 1: annotate_cell_types_semantic Tool (Complete Pattern)

```python
# Source: Verified against annotation_expert.py factory pattern + VectorSearchService API

# Module-level import guard
try:
    from lobster.core.vector.service import VectorSearchService  # noqa: F401
    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False

# Inside annotation_expert() factory:
_vector_service = None

def _get_vector_service():
    nonlocal _vector_service
    if _vector_service is None:
        from lobster.core.vector.service import VectorSearchService
        _vector_service = VectorSearchService()
    return _vector_service

@tool
def annotate_cell_types_semantic(
    modality_name: str,
    cluster_key: str = "leiden",
    k: int = 5,
    min_confidence: float = 0.5,
    validate_graph: bool = False,
    save_result: bool = True,
) -> str:
    """Annotate cell types using semantic vector search against Cell Ontology.
    ...
    """
    # 1. Validate modality
    if modality_name not in data_manager.list_modalities():
        raise ModalityNotFoundError(...)

    adata = data_manager.get_modality(modality_name)

    # 2. Get marker scores per cluster (existing infrastructure)
    cluster_scores = singlecell_service._calculate_marker_scores_from_adata(
        adata, singlecell_service.cell_type_markers, cluster_key=cluster_key
    )

    # 3. For each cluster, build text query and search Cell Ontology
    vector_service = _get_vector_service()
    cluster_annotations = {}

    for cluster_id, type_scores in cluster_scores.items():
        # Get top expressed marker types
        top_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_markers = []
        for cell_type, score in top_types:
            if score > 0:
                markers = singlecell_service.cell_type_markers.get(cell_type, [])
                top_markers.extend(markers[:3])

        # Format query: "Cluster 0: high CD3D, CD3E, CD8A"
        query = f"Cluster {cluster_id}: high {', '.join(top_markers[:5])}"
        matches = vector_service.match_ontology(query, "cell_ontology", k=k)

        # Filter by confidence
        valid = [m for m in matches if m.score >= min_confidence]
        if valid:
            cluster_annotations[cluster_id] = valid[0]
        else:
            cluster_annotations[cluster_id] = None

    # 4. Apply to adata
    adata_annotated = adata.copy()
    for cluster_id, match in cluster_annotations.items():
        mask = adata_annotated.obs[cluster_key].astype(str) == str(cluster_id)
        if match:
            adata_annotated.obs.loc[mask, "cell_type"] = match.term
            adata_annotated.obs.loc[mask, "cell_type_confidence"] = match.score
        else:
            adata_annotated.obs.loc[mask, "cell_type"] = "Unknown"
            adata_annotated.obs.loc[mask, "cell_type_confidence"] = 0.0

    # 5. Store, log with IR, return formatted string
    # ... (standard pattern)
```

### Example 2: VectorSearchService.match_ontology() API (Verified)

```python
# Source: lobster/core/vector/service.py:175-244
service = VectorSearchService()

# Supported ontology aliases (from ONTOLOGY_COLLECTIONS):
# "mondo" / "disease"      -> "mondo_v2024_01"
# "uberon" / "tissue"      -> "uberon_v2024_01"
# "cell_ontology" / "cell_type" -> "cell_ontology_v2024_01"

matches = service.match_ontology("heart attack", "disease", k=3)
# Returns: list[OntologyMatch]

for m in matches:
    print(m.term)          # "myocardial infarction"
    print(m.ontology_id)   # "MONDO:0005068"
    print(m.score)         # 0.8723 (cosine similarity, 0-1)
    print(m.metadata)      # {"ontology_id": "MONDO:...", ...}
    print(m.distance_metric)  # "cosine"
```

### Example 3: OntologyMatch Schema (Actual, Not Spec)

```python
# Source: lobster/core/schemas/search.py:42-74
# IMPORTANT: Field names differ from original spec
class OntologyMatch(BaseModel):
    term: str              # "colorectal carcinoma" (spec said "name")
    ontology_id: str       # "MONDO:0005575"        (spec said "term_id")
    score: float           # 0.87                    (spec said "confidence")
    metadata: dict         # extensible
    distance_metric: str   # "cosine"
```

### Example 4: DiseaseOntologyService.match_disease() (Verified)

```python
# Source: disease_ontology_service.py:173-262
service = DiseaseOntologyService.get_instance()
matches = service.match_disease("colon tumor", k=3, min_confidence=0.7)
# When backend="json": keyword matching, confidence=1.0
# When backend="embeddings": delegates to VectorSearchService.match_ontology("mondo")
# Returns: list[DiseaseMatch]

for m in matches:
    print(m.disease_id)    # "MONDO:0005575" (or "crc" in json mode)
    print(m.name)          # "colorectal carcinoma"
    print(m.confidence)    # 0.89
    print(m.match_type)    # "semantic_embedding" (or "exact_keyword")
    print(m.matched_term)  # original query
    print(m.metadata)      # extensible dict
```

### Example 5: Conditional Tool Registration (Existing Pattern)

```python
# Source: metadata_assistant.py:3202-3203
# Existing pattern for optional tools:
if MICROBIOME_FEATURES_AVAILABLE:
    tools.append(filter_samples_by)

# Same pattern for new semantic tools:
if HAS_VECTOR_SEARCH:
    tools.extend([standardize_tissue_term, standardize_disease_term])
```

### Example 6: Graph Validation (Optional)

```python
# Source: lobster/core/vector/ontology_graph.py:116-189
from lobster.core.vector.ontology_graph import load_ontology_graph, get_neighbors

graph = load_ontology_graph("cell_ontology")  # lru_cache'd
neighbors = get_neighbors(graph, "CL:0000084")  # cytotoxic T cell
# Returns: {"parents": [...], "children": [...], "siblings": [...]}
# Each entry: {"term_id": "CL:...", "name": "..."}
```

## State of the Art

| What | Status After Phase 2 | What Phase 3 Does |
|------|---------------------|-------------------|
| VectorSearchService | Complete, tested (72 tests) | Consumed by new tools |
| OntologyMatch schema | Complete, stable | Used as return type |
| ONTOLOGY_COLLECTIONS | Complete (6 aliases) | Resolved by match_ontology() |
| DiseaseOntologyService backend swap | Complete, tested (36 tests) | Called by standardize_disease_term |
| get_neighbors() graph traversal | Complete, tested | Used for optional validation |
| annotation_expert tools | 10 tools, no semantic search | +1 tool (annotate_cell_types_semantic) |
| metadata_assistant tools | 12 tools, no standardization | +2 tools (tissue + disease) |

**Key change from spec to implementation:**
- VectorSearchService lives at `lobster/core/vector/` (not `lobster/services/search/` as spec proposed)
- OntologyMatch uses `term`/`ontology_id`/`score` (not `name`/`term_id`/`confidence` as spec proposed)
- DiseaseOntologyService import is `from lobster.core.vector.service import VectorSearchService` (not `from lobster.services.search`)

## Open Questions

1. **All-clusters-at-once vs per-cluster annotation**
   - What we know: Existing `annotate_cell_types` does all clusters in one call. `_calculate_marker_scores_from_adata()` returns scores for ALL clusters.
   - What's unclear: Whether semantic tool should follow same pattern (all clusters) or allow single-cluster annotation.
   - Recommendation: All clusters at once (matches existing behavior, simpler agent interaction). The tool can still be called with a cluster_key that has fewer clusters if needed. This is Claude's discretion per CONTEXT.md.

2. **Default min_confidence for tissue standardization**
   - What we know: Disease uses 0.7 (inherited from DiseaseOntologyService). Kevin's spec doesn't specify tissue.
   - What's unclear: Optimal threshold for tissue matching.
   - Recommendation: Use 0.5 for tissue (more permissive than disease, since tissue terms are more ambiguous). This is Claude's discretion per CONTEXT.md.

3. **Stats dict content per tool**
   - What we know: Existing tools use domain-appropriate stats (e.g., `n_cell_types_identified`, `cell_type_counts` for annotation).
   - What's unclear: Exact fields for new tools.
   - Recommendation (Claude's discretion):
     - `annotate_cell_types_semantic`: `n_clusters_annotated`, `n_unknown`, `mean_confidence`, `cluster_annotations` (cluster -> {term, ontology_id, score})
     - `standardize_tissue_term`: `term`, `top_matches` (list of {term, ontology_id, score}), `best_match`
     - `standardize_disease_term`: `term`, `matches` (list of DiseaseMatch-like dicts), `best_match`

4. **lobster-transcriptomics gitignore status**
   - What we know: lobster-metadata is explicitly gitignored
   - What's unclear: Whether lobster-transcriptomics is also gitignored
   - Recommendation: Check `.gitignore` during planning. If gitignored, acknowledge that changes are local-only.

## Sources

### Primary (HIGH confidence)
- `lobster/core/vector/service.py` -- VectorSearchService implementation (312 lines, verified match_ontology API)
- `lobster/core/vector/__init__.py` -- Lazy public API exports
- `lobster/core/schemas/search.py` -- OntologyMatch actual schema (155 lines, verified field names)
- `lobster/core/schemas/ontology.py` -- DiseaseMatch schema (90 lines)
- `lobster/core/vector/ontology_graph.py` -- Graph loading and traversal (190 lines)
- `lobster/core/vector/config.py` -- VectorSearchConfig (128 lines)
- `lobster/core/analysis_ir.py` -- AnalysisStep IR schema (179 lines)
- `packages/lobster-metadata/lobster/services/metadata/disease_ontology_service.py` -- DiseaseOntologyService with Phase 2 backend branching (356 lines)
- `packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py` -- Annotation expert factory (1294 lines)
- `packages/lobster-metadata/lobster/agents/metadata_assistant/metadata_assistant.py` -- Metadata assistant factory (~3228 lines)
- `packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py` -- _calculate_marker_scores_from_adata (lines 1033-1101)
- `kevin_notes/vector_search_implementation.md` -- Kevin's authoritative implementation spec (Sections 4.1, 4.2)
- `tests/unit/core/vector/test_vector_search_service.py` -- MockEmbedder/MockVectorBackend pattern

### Secondary (MEDIUM confidence)
- `.planning/phases/02-service-integration/02-03-SUMMARY.md` -- Phase 2 completion status
- `.planning/STATE.md` -- Current progress (Phase 2 complete, ready for Phase 3)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All components verified by reading source code directly
- Architecture: HIGH -- Patterns extracted from existing codebase, not hypothesized
- Pitfalls: HIGH -- Field name discrepancies verified by comparing spec vs actual code
- Integration points: HIGH -- All API signatures verified from source

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable -- all APIs are internal, no external dependency changes expected)
