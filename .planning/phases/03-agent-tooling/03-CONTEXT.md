# Phase 3: Agent Tooling - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Add semantic search tools to annotation_expert (cell type annotation via Cell Ontology) and metadata_assistant (tissue/disease term standardization via Uberon/MONDO). Existing tools remain unchanged — new tools augment, don't replace. All new tools follow 3-tuple return pattern with mandatory IR for provenance.

</domain>

<decisions>
## Implementation Decisions

### Query construction
- annotation_expert: format marker gene signatures as text queries — `"Cluster 0: high CD3D, CD3E, CD8A"` — using top markers per cluster from existing `_calculate_marker_scores_from_adata()`
- metadata_assistant: free-text passthrough — `standardize_tissue_term(term)` passes term directly to `VectorSearchService.match_ontology(term, "uberon")`
- metadata_assistant: disease routing goes through `DiseaseOntologyService.match_disease(term)` which delegates to VectorSearchService when `backend="embeddings"` — NOT directly to vector service

### Result shape
- All results use `OntologyMatch` schema: term_id, name, confidence (0.0-1.0), match_type, ontology_source, metadata dict
- Annotation stores results in existing obs columns: `cell_type`, `cell_type_confidence`
- Disease standardization converts `OntologyMatch` → `DiseaseMatch` via `_convert_ontology_match()` helper (preserving existing API)
- Default k=5 for `standardize_tissue_term` and `annotate_cell_types_semantic`
- Default k=3, min_confidence=0.7 for disease (inherited from existing `DiseaseOntologyService` API)

### Missing deps behavior
- When vector-search dependencies aren't installed, semantic tools are **absent from agent toolkits** — not registered at all
- Agents only see their existing tools (e.g., annotation_expert sees `annotate_cell_types` but NOT `annotate_cell_types_semantic`)
- Conditional tool registration in agent factories — check for VectorSearchService importability before adding semantic tools to tool list

### Graph validation
- `annotate_cell_types_semantic` has `validate_graph=False` by default
- When enabled, validates results via `get_neighbors()` graph traversal for biological plausibility
- Opt-in parameter — users/agents enable explicitly when accuracy matters more than speed

### Workflow integration
- New `annotate_cell_types_semantic` tool added alongside existing `annotate_cell_types` (unchanged)
- Two separate metadata tools: `standardize_tissue_term` (Uberon), `standardize_disease_term` (MONDO via DiseaseOntologyService)
- Lazy VectorSearchService initialization in agent factory closures
- All tools return 3-tuple `(result, stats, AnalysisStep)` and log with `ir=ir`

### Claude's Discretion
- Exact stats dict content for each tool's return value
- Error message wording when semantic search returns no results above confidence threshold
- Whether `annotate_cell_types_semantic` annotates all clusters in one call (like existing tool) or allows per-cluster annotation
- Default min_confidence for tissue standardization

</decisions>

<specifics>
## Specific Ideas

- Kevin's implementation spec (`kevin_notes/vector_search_implementation.md`) is the authoritative reference — Section 4.1 (annotation) and Section 4.2 (metadata) contain pseudo-code for the exact integration pattern
- Lazy service init follows the closure pattern from the spec: `_vector_service = None` with `get_vector_service()` nonlocal accessor
- Cell type annotation query format: `"Cluster 0: high CD3D, CD3E, CD8A"` — mimics how biologists describe clusters
- Disease path deliberately routes through DiseaseOntologyService (Strangler Fig pattern) rather than bypassing to VectorSearchService directly
- `ONTOLOGY_COLLECTIONS` alias resolution already implemented in Phase 2 — tools use aliases like `"cell_ontology"`, `"uberon"`, `"mondo"`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-agent-tooling*
*Context gathered: 2026-02-18*
