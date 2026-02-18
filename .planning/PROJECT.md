# Vector Search for Lobster AI

## What This Is

Semantic vector search infrastructure for Lobster AI, replacing hardcoded keyword matching with biomedical embedding-based ontology matching. Enables agents to semantically match diseases (MONDO ~30K terms), tissues (Uberon ~30K terms), and cell types (Cell Ontology ~5K terms) using SapBERT embeddings and ChromaDB — turning "colon tumor" into a confident match for "colorectal cancer" and "CD3D+/CD8A+ cluster" into "cytotoxic T cell (CL:0000084)."

## Core Value

Agents can semantically match any biomedical term to the correct ontology concept with calibrated confidence scores, using zero configuration out of the box.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Backend-agnostic vector search service with ChromaDB default (env var switches backend)
- [ ] SapBERT biomedical embeddings (768d) as default, MiniLM and OpenAI as alternatives
- [ ] Two-stage retrieval pipeline: embedding search + cross-encoder reranking
- [ ] DiseaseOntologyService backend swap from keyword to embedding (Strangler Fig completion)
- [ ] Ontology data pipeline: OBO -> SapBERT embeddings -> ChromaDB tarballs -> S3 hosting
- [ ] Auto-download ontology data on first use to ~/.lobster/ontology_cache/
- [ ] annotation_expert gains semantic cell type annotation tool (augments existing tool, doesn't replace)
- [ ] metadata_assistant gains tissue and disease standardization tools
- [ ] NetworkX graph traversal for ontology parent/child/sibling relationships
- [ ] All optional deps import-guarded with helpful install messages
- [ ] Cloud-hosted ChromaDB service plan (vector.omics-os.com) — written as handoff spec for lobster-cloud
- [ ] Fix silent fallback in DiseaseStandardizationService (add logger.warning)
- [ ] Delete duplicate disease_ontology.json from core (canonical stays in lobster-metadata package)
- [ ] Unit tests with mocked deps + integration tests with real small ChromaDB
- [ ] Pydantic schemas for SearchResult, OntologyMatch, LiteratureMatch, SearchResponse

### Out of Scope

- pgvector backend implementation — stub only, future work
- Neptune Analytics / Neo4j graph database — future, INDRA integration path exists
- Literature semantic search — schema defined, implementation deferred
- Replacing existing `annotate_cell_types` tool — new tool augments, old stays
- Editing pyproject.toml — dependency changes documented for human review
- Cloud infrastructure deployment (lobster-cloud ECS/CDK) — handoff spec written here, executed separately
- Mobile or web UI changes — backend-only

## Context

- **Strangler Fig migration**: DiseaseOntologyService was explicitly designed for this swap. Its API (`match_disease(query, k, min_confidence)`) is migration-stable. The `backend` field in config exists but was never wired up. This project completes that planned migration.
- **PR #13 reference**: `feat/vector-search` branch has prior implementation (~51 tests, backends, embedding providers, reranker). Starting fresh on new branch but adapting patterns from PR #13.
- **SRAgent reference**: Production ChromaDB patterns at `/Users/tyo/GITHUB/omics-os/tmp_folder/SRAgent/` — auto-download tarballs, OBO embedding build script, tissue ontology matching.
- **Current state**: DiseaseOntologyService has 4 hardcoded diseases with keyword matching. Annotation expert has 10 hardcoded cell types with marker genes. Both work but can't scale.
- **Brutalist review findings**: config stub never branched on, silent fallback, duplicate config files, O(N*K) scaling collapse.

## Constraints

- **No lobster/__init__.py**: PEP 420 namespace package. Hard requirement.
- **No pyproject.toml edits**: Dependencies documented in REQUIREMENTS.md for human approval. All imports guarded.
- **Lazy loading**: No model downloads at import time. No module-level component_registry calls.
- **3-tuple return**: All service methods return `(result, stats, AnalysisStep)`.
- **IR mandatory**: All `log_tool_usage()` calls must pass `ir=ir`.
- **AGENT_CONFIG at module top**: Before heavy imports for <50ms entry point discovery.
- **Backward compatibility**: Existing tools, APIs, and DiseaseStandardizationService fallback unchanged.
- **Package location**: Core infrastructure in `lobster/services/search/` (cross-agent, not package-specific).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ChromaDB as default backend | Small data (~60K vectors), zero cost, proven in SRAgent, local+cloud symmetry | — Pending |
| SapBERT as default embedding | SOTA biomedical entity linking, 4M+ UMLS synonym pairs, 768d, free/local | — Pending |
| Cross-encoder as default reranker | Zero API cost, offline, ~1-5s for 100 docs on CPU | — Pending |
| New branch (not PR #13) | PR #13 has good patterns but needs clean rearchitecting per this spec | — Pending |
| Cloud-hosted ChromaDB at vector.omics-os.com | Separate service, plan written here, deployment handed off to lobster-cloud | — Pending |
| Core location (not package) | Vector search is cross-agent infrastructure used by annotation, metadata, research | — Pending |
| Augment, don't replace | Existing annotate_cell_types tool stays; new semantic tool added alongside | — Pending |

---
*Last updated: 2026-02-17 after initialization*
