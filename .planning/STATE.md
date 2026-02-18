# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Agents can semantically match any biomedical term to the correct ontology concept with calibrated confidence scores, using zero configuration out of the box.
**Current focus:** Phase 1 - Foundation

## Current Position

Phase: 1 of 6 (Foundation)
Plan: 2 of 3 in current phase
Status: Plan 01-02 complete, executing phase
Last activity: 2026-02-18 — Plan 01-02 executed (Concrete Adapters)

Progress: [██░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 2.5min
- Total execution time: 0.08 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 5min | 2.5min |

**Recent Trend:**
- Last 5 plans: 01-01 (3min), 01-02 (2min)
- Trend: Accelerating

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- ChromaDB as default backend — Small data (~60K vectors), zero cost, proven in SRAgent, local+cloud symmetry
- SapBERT as default embedding — SOTA biomedical entity linking, 4M+ UMLS synonym pairs, 768d, free/local
- Cross-encoder as default reranker — Zero API cost, offline, ~1-5s for 100 docs on CPU
- Core location (not package) — Vector search is cross-agent infrastructure used by annotation, metadata, research
- Augment, don't replace — Existing annotate_cell_types tool stays; new semantic tool added alongside
- [01-01] Score rounding via model_post_init to 4 decimal places for consistent display
- [01-01] TYPE_CHECKING guard in vector/__init__.py for zero-cost imports
- [01-01] Column-oriented search() return format matching ChromaDB convention
- [01-02] CLS-token pooling for SapBERT (trained with CLS, not mean pooling)
- [01-02] batch_size=128 for SapBERT encode per model card recommendation
- [01-02] 5000-doc chunk limit for ChromaDB batch operations
- [01-02] Raw ChromaDB results from search; distance->similarity conversion in service layer

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-18 (Plan 01-02 execution)
Stopped at: Completed 01-02-PLAN.md (Concrete Adapters)
Resume file: .planning/phases/01-foundation/01-02-SUMMARY.md
