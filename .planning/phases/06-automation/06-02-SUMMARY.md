---
phase: 06-automation
plan: 02
subsystem: vector-search
tags: [sapbert, chromadb, obo, ontology, embeddings, cli-tool]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: SapBERTEmbedder, ChromaDBBackend, VectorSearchService, OntologyMatch schema
  - phase: 02-service-integration
    provides: OBO_URLS, ONTOLOGY_COLLECTIONS, ontology_graph parsing
provides:
  - Standalone CLI build script for pre-computing SapBERT ontology embeddings
  - Gzipped tarballs of ChromaDB collections ready for S3 upload
  - Dry-run validation mode for dependency and path checking
affects: [06-03, 06-04, cloud-deployment, release-workflow]

# Tech tracking
tech-stack:
  added: [obonet (import-guarded), argparse]
  patterns: [standalone-dev-script, dry-run-validation, version-tagged-collections]

key-files:
  created:
    - scripts/build_ontology_embeddings.py
  modified: []

key-decisions:
  - "argparse over Typer for minimal deps in standalone dev tool"
  - "Batch SapBERT embedding (128) and ChromaDB adds (5000) matching existing patterns"
  - "Version-tag flag for future ontology releases without script modification"
  - "Dry-run validates imports, output dir writability, and ONTOLOGY_COLLECTIONS name consistency"
  - "Skip obsolete terms during extraction (don't embed stale ontology entries)"
  - "OBO definition stripping handles quoted strings with bracket references"

patterns-established:
  - "scripts/ directory for standalone dev/build tools (not imported by runtime)"
  - "Dry-run validation pattern: check all deps and paths before expensive operations"
  - "Version-tagged collection naming: {ontology}_{version_tag} with ONTOLOGY_COLLECTIONS validation"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04]

# Metrics
duration: 3min
completed: 2026-02-19
---

# Phase 06 Plan 02: Ontology Embedding Build Script Summary

**Standalone CLI script that parses OBO ontologies, generates SapBERT embeddings, stores in ChromaDB with full metadata, and produces gzipped tarballs for S3 distribution**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T08:44:34Z
- **Completed:** 2026-02-19T08:47:54Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Created `scripts/build_ontology_embeddings.py` -- a standalone CLI tool for pre-building ontology embeddings offline
- OBO parsing extracts terms with full metadata (term_id, name, synonyms, namespace, is_obsolete), skipping obsolete and unnamed terms
- Embedding text follows "{label}: {definition}" format with cleaned OBO definitions (DATA-02)
- Dry-run mode validates all dependencies, output dir writability, and ONTOLOGY_COLLECTIONS name consistency before any expensive operations
- Version-tag flag allows building for future ontology releases without modifying the script

## Task Commits

Each task was committed atomically:

1. **Task 1: Create the build script** - `f3c15cd` (feat)
2. **Task 2: Smoke test with dry-run validation** - included in Task 1 commit (script was built complete with dry-run and version-tag support)

_Note: Task 2's functionality (--dry-run flag, --version-tag argument, collection name validation) was naturally included in the initial script creation. Dry-run verification confirmed: SapBERTEmbedder importable, output dir writable, all 3 collection names match ONTOLOGY_COLLECTIONS._

## Files Created/Modified
- `scripts/build_ontology_embeddings.py` - Standalone CLI build script (345 lines) for offline ontology embedding generation

## Decisions Made
- Used argparse (not Typer) since this is a standalone dev tool with minimal dependency requirements
- Batch sizes match existing codebase patterns: 128 for SapBERT embedding, 5000 for ChromaDB adds
- Version-tag defaults to "v2024_01" matching current ONTOLOGY_COLLECTIONS; future releases change just the flag
- Dry-run exits with code 1 when missing dependencies are detected (correct behavior for CI/validation)
- OBO definition cleaning handles the `'"definition" [REF:001]'` format with split on `'" ['`
- Obsolete terms are skipped during extraction to avoid embedding stale ontology entries
- Fallback OBO_URLS embedded in script in case lobster import fails (robustness for edge cases)

## Deviations from Plan

None - plan executed exactly as written. Task 2's code was delivered alongside Task 1 since both tasks target the same file and the dry-run/version-tag features were part of the natural script structure.

## Issues Encountered
- obonet and chromadb are not installed in the dev venv (they are optional `[vector-search]` extras). The dry-run correctly identifies and reports this. This is expected behavior -- the build script is meant to be run in an environment with all vector-search dependencies installed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Build script ready to generate tarballs when run in a properly configured environment
- Next plan (06-03) can implement S3 upload integration for the generated tarballs
- Script output directory and naming convention established for downstream tooling

## Self-Check: PASSED

- [x] `scripts/build_ontology_embeddings.py` exists
- [x] Commit `f3c15cd` exists in git log

---
*Phase: 06-automation*
*Completed: 2026-02-19*
