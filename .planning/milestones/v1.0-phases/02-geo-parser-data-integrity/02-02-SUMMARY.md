---
phase: 02-geo-parser-data-integrity
plan: 02
subsystem: data-access
tags: [geo, archive, scoring-heuristic, temp-cleanup, tar, tdd]

# Dependency graph
requires:
  - phase: 02-01
    provides: ParseResult dataclass, partial parse signaling
provides:
  - ARCHIVE_EXTENSIONS constant covering .tar, .tar.gz, .tgz, .tar.bz2
  - _is_archive_url helper replacing scattered .endswith(".tar") checks
  - _score_expression_file scoring heuristic replacing METADATA_KEYWORDS blacklist
  - Temp file cleanup on _process_tar_file failure via shutil.rmtree
affects: [geo-service, download-orchestrator]

# Tech tracking
tech-stack:
  added: [shutil]
  patterns: [scoring-heuristic-file-selection, archive-extension-constant, cleanup-on-failure]

key-files:
  created:
    - tests/unit/services/data_access/test_geo_archive_classifier.py
    - tests/unit/services/data_access/test_geo_file_scoring.py
    - tests/unit/services/data_access/test_geo_temp_cleanup.py
  modified:
    - lobster/services/data_access/geo_service.py

key-decisions:
  - "Scoring heuristic replaces METADATA_KEYWORDS blacklist for file selection -- contextual 'gene' handling"
  - "ARCHIVE_EXTENSIONS as module-level tuple constant for consistent archive detection"
  - "_is_archive_url as module-level function (stateless, not a method)"
  - "extract_dir and nested_extract_dir defined before try block for except-block cleanup access"

patterns-established:
  - "Scoring heuristic: expression signals boost, metadata signals penalize, ambiguous keywords contextual"
  - "Cleanup-on-failure: define cleanup targets before try, rmtree(ignore_errors=True) in except"

requirements-completed: [GPAR-03, GPAR-04, GPAR-05]

# Metrics
duration: 10min
completed: 2026-03-04
---

# Phase 02 Plan 02: Archive Classifier, Scoring Heuristic, and Temp Cleanup Summary

**ARCHIVE_EXTENSIONS constant fixes tar.gz/tgz/tar.bz2 detection, scoring heuristic replaces METADATA_KEYWORDS blacklist so gene_expression_matrix.txt.gz is no longer blocked, and _process_tar_file cleans up extraction directories on failure**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-04T03:38:28Z
- **Completed:** 2026-03-04T03:48:49Z
- **Tasks:** 2
- **Files modified:** 4 (1 source, 3 test)

## Accomplishments
- ARCHIVE_EXTENSIONS constant replaces all scattered `.endswith(".tar")` checks at 3 detection sites
- _score_expression_file scoring heuristic correctly handles "gene" ambiguity (gene_expression_matrix passes, genes.tsv rejected)
- _process_tar_file cleans up extract_dir and nested_extract_dir on exception (shutil.rmtree with ignore_errors=True)
- 40 new tests (17 archive classifier, 19 file scoring, 4 temp cleanup) all passing

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1 RED: Archive classifier + file scoring tests** - `ea54085` (test)
2. **Task 1 GREEN: Implement ARCHIVE_EXTENSIONS + scoring heuristic** - `bc4a181` (feat)
3. **Task 2 RED: Temp cleanup tests** - `4669f14` (test)
4. **Task 2 GREEN: Add cleanup in _process_tar_file** - `06d77f3` (feat)

## Files Created/Modified
- `lobster/services/data_access/geo_service.py` - ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file, cleanup in _process_tar_file except block
- `tests/unit/services/data_access/test_geo_archive_classifier.py` - 17 tests for archive format detection
- `tests/unit/services/data_access/test_geo_file_scoring.py` - 19 tests for expression file scoring heuristic
- `tests/unit/services/data_access/test_geo_temp_cleanup.py` - 4 tests for temp directory cleanup

## Decisions Made
- Scoring heuristic uses contextual "gene" handling: positive with expression context (gene_expression_matrix), negative alone (genes.tsv.gz)
- ARCHIVE_EXTENSIONS is a module-level tuple constant shared across all 3 detection sites
- _is_archive_url and _score_expression_file are module-level functions (stateless, not methods)
- extract_dir and nested_extract_dir definitions moved before try block so they are visible in the except block for cleanup
- FORMAT_BONUS scoring: .h5ad (1.0) > .h5 (0.8) > .mtx (0.5) > plain text (0)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 02 (GEO Parser & Data Integrity) now complete -- all 2 plans finished
- Ready for Phase 03 (core/ restructuring) or any subsequent phases

---
*Phase: 02-geo-parser-data-integrity*
*Completed: 2026-03-04*
