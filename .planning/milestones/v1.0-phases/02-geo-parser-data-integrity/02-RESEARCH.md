# Phase 2: GEO Parser & Data Integrity - Research

**Researched:** 2026-03-04
**Domain:** GEO data parsing, file classification, temp file lifecycle management
**Confidence:** HIGH

## Summary

Phase 2 addresses five concrete bugs/gaps in the GEO pipeline's parser and file handling layers. The chunk parser (`parse_large_file_in_chunks` in `geo/parser.py:678`) silently truncates results when hitting memory limits -- it `break`s at line 724 and concatenates partial chunks at line 753 with no truncation signal. The supplementary file classifier (`_process_supplementary_files` in `geo_service.py:3672`) only checks `.endswith(".tar")`, missing `.tar.gz`, `.tgz`, and `.tar.bz2`. The expression file selection uses a keyword blacklist (`METADATA_KEYWORDS` at `geo_service.py:3681`) that incorrectly excludes files containing "gene" (blocking legitimate files like `gene_expression_matrix.txt.gz`). Finally, there is no temp file cleanup when parsing fails mid-operation -- `_process_tar_file` creates extraction directories under `cache_dir` but never removes them on error.

All five requirements are localized to two files (`geo/parser.py` and `geo_service.py`) and one utility (`archive_utils.py`). The changes are data-structure additions and logic replacements -- no new external dependencies needed.

**Primary recommendation:** Introduce a `ParseResult` dataclass wrapping parsed data with integrity metadata (`is_partial`, `rows_read`, `truncation_reason`), fix the archive extension matching, replace the keyword blacklist with a scoring function, and add `try/finally` cleanup blocks on all extraction paths.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GPAR-01 | Chunk parser returns `is_partial`, `rows_read`, `truncation_reason` flags | ParseResult dataclass wrapping parser returns; modification to `parse_large_file_in_chunks` at line 678 |
| GPAR-02 | All call sites handle partial parse results -- mark modality/metadata and surface warning | 5 call sites identified in `geo_service.py` that call `parse_expression_file` or `parse_supplementary_file`; each needs partial-result checking |
| GPAR-03 | Supplementary file classifier handles `.tar.gz`, `.tgz`, `.tar.bz2` | Two fix sites: `_process_supplementary_files` line 3672 and `_initialize_file_type_patterns` archive patterns line 4408 |
| GPAR-04 | File-scoring heuristic replaces brittle keyword blacklist for expression file selection | Replace `METADATA_KEYWORDS` blacklist at line 3681 with scoring function similar to existing `_classify_single_file` pattern |
| GPAR-05 | Partial parser failures trigger temp file cleanup and proper failure marking | Add `try/finally` cleanup in `_process_tar_file` and `_process_supplementary_files`; extraction dirs at `cache_dir/{gse_id}_extracted` and `cache_dir/{gse_id}_nested_extracted` |
</phase_requirements>

## Standard Stack

### Core

No new external dependencies needed. All work uses existing project libraries:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | existing | DataFrame parsing, chunked reading | Already used in `parse_large_file_in_chunks` |
| psutil | existing | Memory checks during chunk parsing | Already used in parser's memory management |
| dataclasses | stdlib | `ParseResult` dataclass | Python stdlib, no dependency |
| pathlib | stdlib | File path operations | Already used throughout |
| tempfile/shutil | stdlib | Temp directory management and cleanup | Already used in `archive_utils.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| polars | existing (optional) | Fast CSV parsing | Already used as primary parser with pandas fallback |
| pytest | existing | Unit tests | Test infrastructure already configured |

### Alternatives Considered

None -- this phase modifies existing code, not adding new capabilities.

## Architecture Patterns

### Affected File Map

```
lobster/
├── services/data_access/
│   ├── geo/
│   │   └── parser.py              # GPAR-01: ParseResult, chunk parser changes
│   └── geo_service.py             # GPAR-02, GPAR-03, GPAR-04, GPAR-05: call sites, classifier, blacklist, cleanup
└── core/
    └── archive_utils.py           # Reference only (already handles .tar.gz/.tgz/.tar.bz2 correctly)
```

### Pattern 1: ParseResult Dataclass (GPAR-01)

**What:** A typed return wrapper that signals parse completeness alongside data.

**When to use:** Returned by `parse_large_file_in_chunks` and propagated through `parse_expression_file` and `parse_supplementary_file`.

**Design:**
```python
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class ParseResult:
    """Result of parsing a GEO expression file, with integrity metadata."""
    data: Optional[pd.DataFrame]
    is_partial: bool = False
    rows_read: int = 0
    truncation_reason: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.data is not None and not self.is_partial

    @property
    def is_empty(self) -> bool:
        return self.data is None or (hasattr(self.data, 'empty') and self.data.empty)
```

**Key insight:** The dataclass lives in `geo/parser.py` alongside `GEOParser`. It does NOT change the parser's public API signature -- `parse_expression_file` and `parse_supplementary_file` continue returning `Optional[pd.DataFrame]` at their top-level API. The `parse_large_file_in_chunks` method is the one that returns `ParseResult` internally, and the callers decide how to handle the `is_partial` flag.

**Alternative design considered:** Having `parse_expression_file` itself return `ParseResult` would be cleaner but would break the 5 call sites in geo_service.py and potentially other callers. Instead, the internal method returns `ParseResult` and the public methods log/flag the partial status while still returning `Optional[pd.DataFrame]`.

**Recommended approach:** Have `parse_expression_file` return `ParseResult` (breaking the return type) because all call sites are WITHIN this codebase (geo_service.py) and can be updated in the same PR. This is cleaner than hiding the partial information.

### Pattern 2: Archive Extension Matching (GPAR-03)

**What:** Comprehensive archive format detection replacing `.endswith(".tar")`.

**Current bug locations:**
1. `_process_supplementary_files` line 3672: `f.lower().endswith(".tar")`
2. `_initialize_file_type_patterns` line 4408: regex `r".*\.tar(\.gz)?$"` (misses `.tgz`, `.tar.bz2`)
3. `_try_supplementary_first` line 1880: `archive_url.lower().endswith(".tar")`

**Fix pattern:**
```python
# Helper function or constant
ARCHIVE_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".tar.bz2")

def _is_archive_url(url: str) -> bool:
    """Check if URL points to an archive file."""
    lower = url.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)
```

**Note:** `archive_utils.py` (the ArchiveExtractor) ALREADY handles `.tar.gz`, `.tgz`, `.tar.bz2` correctly (line 160-164: checks `.suffix in [".tar", ".gz", ".bz2", ".tgz"] or ".tar" in archive_path.name`). The bug is only in the supplementary file classifier in `geo_service.py`.

### Pattern 3: Expression File Scoring Heuristic (GPAR-04)

**What:** Replace the `METADATA_KEYWORDS` blacklist with a scoring function for expression file selection.

**Current bug:** The blacklist at line 3681-3688 includes "gene", which excludes legitimate expression files like `GSE12345_gene_expression_counts.txt.gz`.

**The existing scoring infrastructure:** `geo_service.py` already has a sophisticated scoring system (`_classify_single_file` at line 4474) used for sample-level supplementary file classification. The series-level selection at `_process_supplementary_files` does NOT use this system -- it uses the naive blacklist instead.

**Recommended approach:** Create a lightweight scoring function for series-level expression file selection that:
1. Gives positive score to expression-like filenames (counts, expression, matrix, tpm, fpkm)
2. Gives negative score to metadata-like filenames (barcodes-only, annotation-only, sample-info)
3. Handles the "gene" edge case: `gene_expression_matrix.txt.gz` scores high (gene + expression), `genes.tsv.gz` scores low (gene-only, likely feature list)
4. Considers file extension priority: `.h5ad` > `.h5` > `.mtx` > `.txt/.csv/.tsv`

```python
def _score_expression_file(filename: str) -> float:
    """Score a supplementary file for expression data likelihood.

    Returns positive scores for likely expression files,
    negative scores for likely metadata/annotation files.
    """
    lower = filename.lower()
    score = 0.0

    # Strong positive signals (expression data)
    EXPRESSION_SIGNALS = [
        ("count", 2.0), ("expression", 2.0), ("matrix", 1.5),
        ("tpm", 2.0), ("fpkm", 2.0), ("rpkm", 2.0),
        ("normalized", 1.5), ("processed", 1.0),
    ]

    # Strong negative signals (metadata/annotation)
    METADATA_SIGNALS = [
        ("barcode", -2.0), ("annotation", -1.5),
        ("metadata", -2.0), ("sample_info", -1.5),
        ("clinical", -1.5),
    ]

    # Ambiguous signals (context-dependent)
    # "gene" alone is metadata; "gene" + expression signal is data
    # "feature" alone is metadata; "feature" + matrix is data

    for signal, weight in EXPRESSION_SIGNALS:
        if signal in lower:
            score += weight

    for signal, weight in METADATA_SIGNALS:
        if signal in lower:
            score += weight  # weight is already negative

    # Handle ambiguous cases
    if "gene" in lower:
        if score > 0:
            score += 0.5  # "gene" in context of expression data
        else:
            score -= 1.5  # "gene" alone = likely gene list

    if "feature" in lower:
        if score > 0:
            score += 0.5
        else:
            score -= 1.5

    # Format bonus
    FORMAT_BONUS = {".h5ad": 1.0, ".h5": 0.8, ".mtx": 0.5}
    for ext, bonus in FORMAT_BONUS.items():
        if ext in lower:
            score += bonus

    return score
```

### Pattern 4: Temp File Cleanup (GPAR-05)

**What:** Ensure extraction directories are cleaned up on parser failures.

**Current gap:** `_process_tar_file` creates:
- `cache_dir / f"{gse_id}_extracted"` (line 3765)
- `cache_dir / f"{gse_id}_nested_extracted"` (line 3836)

On exception at line 3973, neither is cleaned up. The `try/except` catches everything and returns None, but the directories persist with potentially gigabytes of extracted data.

**Fix pattern:** Use `try/finally` to ensure cleanup:
```python
def _process_tar_file(self, tar_url: str, gse_id: str):
    extract_dir = self.cache_dir / f"{gse_id}_extracted"
    nested_extract_dir = self.cache_dir / f"{gse_id}_nested_extracted"
    cleanup_dirs = []

    try:
        # ... existing logic ...
        # Track dirs as they're created
        if extract_dir.exists():
            cleanup_dirs.append(extract_dir)
        if nested_extract_dir.exists():
            cleanup_dirs.append(nested_extract_dir)
        # ... processing ...
        return result  # Success: don't clean up (data is valid)
    except Exception as e:
        logger.error(f"Error processing TAR file: {e}")
        # Clean up on failure
        for d in cleanup_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.debug(f"Cleaned up failed extraction: {d}")
        return None
```

### Anti-Patterns to Avoid

- **Silent truncation:** Never return partial data without a signal. The current `break` at line 724 followed by concatenation at line 753 is the exact anti-pattern.
- **Keyword blacklists for file selection:** Simple `in` checks are too coarse. "gene" appears in both `gene_list.tsv` and `gene_expression_counts.txt.gz`.
- **Extension-only archive detection:** `.endswith(".tar")` misses compound extensions. Always check for all standard archive formats.
- **Cleanup in except-only blocks:** Use `try/finally` for cleanup that must run regardless. The current `except` at line 3973 catches but does not clean.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Archive format detection | Custom suffix checking per call site | Shared `_is_archive_url()` helper + `ARCHIVE_EXTENSIONS` constant | DRY -- currently 3 places with inconsistent checks |
| Memory-safe chunk parsing | Custom memory monitoring | Existing `psutil`-based `_check_memory_availability` + `_get_adaptive_chunk_size` | Already well-implemented; just needs to signal truncation |
| Temp dir cleanup | Manual `shutil.rmtree` at each site | Consider using `ArchiveExtractor.extract_to_temp()` from `archive_utils.py` which tracks temp dirs | Already has `cleanup()` method; but note `_process_tar_file` creates dirs in cache_dir, not temp |

**Key insight:** The `ArchiveExtractor` in `archive_utils.py` already solves temp dir tracking with `extract_to_temp()` and `cleanup()`. However, `_process_tar_file` in `geo_service.py` does its own extraction using `tarfile.open()` directly (line 3770), bypassing the shared utility. For this phase, the pragmatic fix is `try/finally` cleanup. Migrating to `ArchiveExtractor` would be a larger refactor better suited for Phase 4 (GEO Service Decomposition).

## Common Pitfalls

### Pitfall 1: Breaking parse_expression_file Return Type
**What goes wrong:** Changing `parse_expression_file` to return `ParseResult` instead of `Optional[pd.DataFrame]` breaks all call sites.
**Why it happens:** 5+ call sites in `geo_service.py` check `if matrix is not None` or access `.shape`.
**How to avoid:** Update ALL call sites in the same plan. There are exactly 5 call sites (lines 1734, 3954, 4004, 4849, 5004) plus `parse_supplementary_file` which delegates to `parse_expression_file`.
**Warning signs:** Tests fail with `AttributeError: 'ParseResult' object has no attribute 'shape'`.

### Pitfall 2: Partial Result Propagation
**What goes wrong:** Adding `is_partial` to the parser but not propagating it through `parse_supplementary_file` and up to geo_service.py callers.
**Why it happens:** `parse_supplementary_file` calls `parse_expression_file` for `.txt/.csv/.tsv` formats (line 1396). If `parse_expression_file` now returns `ParseResult`, `parse_supplementary_file` must handle and propagate it.
**How to avoid:** Design the return type chain: `parse_large_file_in_chunks` -> `parse_expression_file` -> `parse_supplementary_file` -> geo_service.py call sites. All must understand `ParseResult`.

### Pitfall 3: Archive Extension False Positives
**What goes wrong:** Adding `.tar.gz` to the archive check but not adjusting the URL parsing -- a file like `GSE12345_metadata.tar.gz` gets routed to `_process_tar_file` which tries to parse it as expression data.
**Why it happens:** The current `.endswith(".tar")` is restrictive but safe. Broadening it catches more files, some of which may not be expression data TARs.
**How to avoid:** The scoring heuristic (GPAR-04) should work in conjunction with archive detection. Archives that don't contain expression data will fail gracefully in `_process_tar_file` (it already returns None on failure).

### Pitfall 4: Cleanup Removing Valid Cache
**What goes wrong:** Cleaning up `{gse_id}_extracted` on parse failure when the extraction itself was fine -- only parsing failed.
**Why it happens:** The extraction directory doubles as cache. Removing it means the next attempt re-downloads and re-extracts.
**How to avoid:** Only clean up on PARSER failure (not extraction failure). The extraction is already cached at `{gse_id}_RAW.tar`. Removing the extracted dir is acceptable since it can be re-created from the cached tar.

### Pitfall 5: "gene" in Filename Scoring
**What goes wrong:** The new scoring heuristic still incorrectly handles "gene" -- either too aggressive (blocks gene_expression_matrix) or too permissive (allows genes.tsv as expression).
**Why it happens:** "gene" is genuinely ambiguous without context.
**How to avoid:** Score "gene" contextually: if co-occurring with expression signals (count, expression, matrix), boost; if alone or with annotation signals, penalize. Test against known edge cases: `gene_expression_matrix.txt.gz` (should pass), `genes.tsv.gz` (should fail), `gene_annotation.csv` (should fail).

## Code Examples

### Current Chunk Parser (the bug)

```python
# geo/parser.py line 716-756 (CURRENT - BUGGY)
for i, chunk in enumerate(chunk_reader):
    chunk_memory = chunk.memory_usage(deep=True).sum()
    if not self._check_memory_availability(chunk_memory, safety_factor=1.2):
        logger.warning(
            f"Memory limit reached after {i} chunks ({total_rows:,} rows). "
            f"Stopping early to prevent OOM."
        )
        break  # <-- SILENT TRUNCATION: no flag set

    chunks.append(chunk)
    total_rows += len(chunk)
    # ... memory management ...

# Combines and returns without any truncation signal
if chunks:
    df = pd.concat(chunks, axis=0)
    return df  # <-- Caller has no idea this is partial
```

### Current Blacklist (the bug)

```python
# geo_service.py line 3681-3697 (CURRENT - BUGGY)
METADATA_KEYWORDS = [
    "barcode", "feature", "gene",  # <-- "gene" blocks gene_expression_matrix.txt.gz
    "annotation", "metadata", "sample",
]

expression_files = [
    f for f in suppl_files
    if any(ext in f.lower() for ext in [".txt.gz", ".csv.gz", ".tsv.gz", ".h5", ".h5ad"])
    and not any(keyword in f.lower() for keyword in METADATA_KEYWORDS)
]
```

### Current Archive Detection (the bug)

```python
# geo_service.py line 3672 (CURRENT - BUGGY)
tar_files = [f for f in suppl_files if f.lower().endswith(".tar")]
# Misses: .tar.gz, .tgz, .tar.bz2
```

### Current Temp Handling (the bug)

```python
# geo_service.py line 3727-3975 (CURRENT - BUGGY)
def _process_tar_file(self, tar_url, gse_id):
    try:
        extract_dir = self.cache_dir / f"{gse_id}_extracted"
        # ... creates dir, extracts, processes ...
        nested_extract_dir = self.cache_dir / f"{gse_id}_nested_extracted"
        # ... creates dir, extracts nested archives ...
    except Exception as e:
        logger.error(f"Error processing TAR file: {e}")
        return None  # <-- extract_dir and nested_extract_dir left on disk
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Return `None` on parse failure | Return `ParseResult` with metadata | This phase | Callers can distinguish "no data" from "partial data" |
| `.endswith(".tar")` | `ARCHIVE_EXTENSIONS` constant | This phase | Catches all standard archive formats |
| Keyword blacklist for file selection | Scoring heuristic | This phase | Handles ambiguous filenames like `gene_expression_matrix.txt.gz` |
| No cleanup on failure | `try/finally` cleanup | This phase | Zero temp file leaks on parser failures |

## Open Questions

1. **Should `parse_expression_file` return `ParseResult` or `Optional[pd.DataFrame]`?**
   - What we know: Changing the return type is cleaner but touches 5 call sites.
   - What's unclear: Whether external consumers call `parse_expression_file` directly.
   - Recommendation: Return `ParseResult` since all known callers are in `geo_service.py` (same codebase). The deprecated shim at `tools/geo_parser.py` re-exports `GEOParser` but callers would need to handle the new return type. Given this is an internal refactor phase and `geo_parser.py` is already deprecated, this is acceptable.

2. **Should extraction directories be cleaned on success too?**
   - What we know: Extracted dirs serve as cache (avoid re-extraction on retry).
   - What's unclear: How large these can get and whether the caching is actually used.
   - Recommendation: Only clean on failure (preserve cache behavior). Defer cache eviction strategy to a future phase.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | `/Users/tyo/Omics-OS/lobster/pytest.ini` |
| Quick run command | `pytest tests/unit/services/data_access/ tests/unit/core/test_archive_utils.py -x --no-cov -q` |
| Full suite command | `pytest tests/unit/ -x --no-cov -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPAR-01 | Chunk parser returns is_partial, rows_read, truncation_reason when memory limit hit | unit | `pytest tests/unit/services/data_access/test_geo_parser_partial.py -x --no-cov` | Wave 0 |
| GPAR-01 | ParseResult dataclass properties (is_complete, is_empty) | unit | `pytest tests/unit/services/data_access/test_geo_parser_partial.py::TestParseResult -x --no-cov` | Wave 0 |
| GPAR-02 | Call sites mark modality as partial and log warning on partial parse | unit | `pytest tests/unit/services/data_access/test_geo_service_partial_handling.py -x --no-cov` | Wave 0 |
| GPAR-03 | Archive classifier identifies .tar.gz, .tgz, .tar.bz2 | unit | `pytest tests/unit/services/data_access/test_geo_archive_classifier.py -x --no-cov` | Wave 0 |
| GPAR-04 | Scoring heuristic selects gene_expression_matrix over genes.tsv | unit | `pytest tests/unit/services/data_access/test_geo_file_scoring.py -x --no-cov` | Wave 0 |
| GPAR-05 | Parser failure cleans up temp dirs (zero files left) | unit | `pytest tests/unit/services/data_access/test_geo_temp_cleanup.py -x --no-cov` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/services/data_access/test_geo_parser_partial.py tests/unit/services/data_access/test_geo_archive_classifier.py tests/unit/services/data_access/test_geo_file_scoring.py tests/unit/services/data_access/test_geo_temp_cleanup.py -x --no-cov -q`
- **Per wave merge:** `pytest tests/unit/ -x --no-cov -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/services/data_access/test_geo_parser_partial.py` -- covers GPAR-01 (ParseResult, chunk parser partial signaling)
- [ ] `tests/unit/services/data_access/test_geo_service_partial_handling.py` -- covers GPAR-02 (call site handling)
- [ ] `tests/unit/services/data_access/test_geo_archive_classifier.py` -- covers GPAR-03 (archive format detection)
- [ ] `tests/unit/services/data_access/test_geo_file_scoring.py` -- covers GPAR-04 (expression file scoring heuristic)
- [ ] `tests/unit/services/data_access/test_geo_temp_cleanup.py` -- covers GPAR-05 (temp file cleanup on failure)

## Sources

### Primary (HIGH confidence)
- `lobster/services/data_access/geo/parser.py` -- chunk parser implementation, lines 678-768
- `lobster/services/data_access/geo_service.py` -- supplementary file handling, file classification, temp dirs
- `lobster/core/archive_utils.py` -- ArchiveExtractor (correct implementation for reference)
- `kevin_notes/GEO_cleanup.md` -- original findings F6, F7, amendment A2
- `kevin_notes/UNIFIED_CLEANUP_PLAN.md` -- PR-2 work items and scope
- `tests/unit/core/test_archive_utils.py` -- existing test patterns for archive handling

### Secondary (MEDIUM confidence)
- `lobster/tools/geo_parser.py` -- deprecated shim (confirms all callers should use new location)

### Tertiary (LOW confidence)
- None -- all research based on direct codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies; all changes use existing stdlib
- Architecture: HIGH -- all affected code inspected, exact line numbers identified
- Pitfalls: HIGH -- derived from actual code paths and edge cases documented in GEO_cleanup.md

**Research date:** 2026-03-04
**Valid until:** Indefinite (internal codebase refactoring, no external dependency drift)
