---
phase: 06-automation
plan: 04
subsystem: infra
tags: [chromadb, cloud, deployment, vector-search, ecs, fargate, architecture]

# Dependency graph
requires:
  - phase: 06-automation/03
    provides: ChromaDB auto-download extension with cache management
provides:
  - Cloud-hosted ChromaDB deployment specification for vector.omics-os.com
  - Architecture diagram for ECS Fargate ChromaDB server mode
  - Token auth design for service-to-service ChromaDB access
  - Step-by-step CDK deployment instructions
  - Cost estimate (~$72-102/month)
  - 3-phase migration path
affects: [lobster-cloud CDK stacks, vector search cloud mode, Omics-OS Cloud backend]

# Tech tracking
tech-stack:
  added: []
  patterns: [cloud-handoff-spec, service-to-service-token-auth]

key-files:
  created:
    - docs/cloud-chromadb-handoff.md
  modified: []

key-decisions:
  - "ChromaDB server mode (not embedded) for multi-tenant cloud access"
  - "Single instance (not per-tenant) since ontology data is shared read-only"
  - "ECS Fargate ARM64/Graviton for 20% cost savings"
  - "Always-on (no scale-to-zero) to avoid cold start re-indexing"
  - "Token auth via Secrets Manager with X-Chroma-Token header"
  - "HttpClient switching via LOBSTER_VECTOR_CLOUD_URL env var in ChromaDBBackend"

patterns-established:
  - "Cloud handoff spec format: overview, architecture, auth, API, deployment steps, cost, migration"
  - "Service-to-service auth pattern: Secrets Manager token shared between ECS tasks"

requirements-completed: [CLOD-01, CLOD-02]

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 06 Plan 04: Cloud ChromaDB Handoff Specification Summary

**Complete deployment blueprint for vector.omics-os.com with ECS Fargate ChromaDB server, token auth, prewarmed indexes, and 3-phase migration path**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T08:44:47Z
- **Completed:** 2026-02-19T08:49:07Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Wrote comprehensive cloud-hosted ChromaDB handoff spec (625 lines) covering all deployment aspects
- Designed token-based authentication flow: Cognito JWT -> Omics-OS Cloud API -> ChromaDB server via X-Chroma-Token
- Documented HttpClient switching pattern in ChromaDBBackend via LOBSTER_VECTOR_CLOUD_URL env var
- Provided step-by-step CDK deployment instructions with cost estimate ($72-102/month)
- Defined 3-phase migration path: spec (Phase 1) -> deploy server (Phase 2) -> Cognito integration (Phase 3)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write cloud-hosted ChromaDB handoff specification** - `fc4c7f8` (docs)

**Note:** The handoff document was committed as part of a prior plan execution batch (06-01 tests commit). Content is complete and verified.

## Files Created/Modified
- `docs/cloud-chromadb-handoff.md` - Cloud-hosted ChromaDB deployment specification for vector.omics-os.com (625 lines)

## Decisions Made
- ChromaDB server mode for multi-tenant cloud access (not embedded PersistentClient)
- Single shared instance sufficient for read-only ontology data (no per-tenant isolation needed)
- ECS Fargate ARM64/Graviton for cost optimization (~20% savings)
- Always-on deployment (cold start re-downloads ~150MB, not acceptable for production latency)
- Token auth via AWS Secrets Manager, proxied through Omics-OS Cloud API (users never call ChromaDB directly)
- HttpClient mode activated by LOBSTER_VECTOR_CLOUD_URL env var -- zero changes to VectorSearchService or agent tools

## Deviations from Plan

None - plan executed exactly as written. Document was already committed in a prior batch execution; content verified to meet all plan requirements.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. This is a planning document only (Phase 1 of migration path).

## Next Phase Readiness
- Cloud-hosted ChromaDB spec is complete and self-contained
- A cloud engineer can deploy vector.omics-os.com using only this document
- Phase 2 (CLOD-V2-01) can proceed when cloud deployment is prioritized
- No blockers

## Self-Check: PASSED

- FOUND: docs/cloud-chromadb-handoff.md
- FOUND: fc4c7f8

---
*Phase: 06-automation*
*Completed: 2026-02-19*
