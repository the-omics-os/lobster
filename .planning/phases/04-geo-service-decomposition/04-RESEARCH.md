# Phase 4: GEO Service Decomposition - Research

**Researched:** 2026-03-04
**Domain:** Python monolith decomposition / Facade pattern / GEO bioinformatics pipeline
**Confidence:** HIGH

## Summary

Phase 4 decomposes `geo_service.py` (5,954 LOC, 59 methods) into 5 focused domain modules while preserving the exact public API surface. The file is a monolith containing metadata fetching (GEOparse/Entrez), download execution (SOFT pre-download, GEOparse calls, supplementary files), archive processing (TAR extraction, nested archives, 10X data), matrix parsing/validation, and sample concatenation -- all tangled in a single class.

A `geo/` subpackage already exists with extracted modules (`constants.py`, `downloader.py`, `parser.py`, `strategy.py`, `loaders/tenx.py`) and **empty placeholder directories** (`metadata/`, `sample/`, `utils/`) that were clearly created in anticipation of this decomposition. The facade pattern is already established in `geo/facade.py` (currently just re-exporting from `geo_service.py`).

**Primary recommendation:** Extract methods into 5 domain modules under `lobster/services/data_access/geo/`, convert `GEOService` in `geo_service.py` into a thin facade that delegates to the new modules, deduplicate the SOFT pre-download logic (copied 6 times in geo_service + 1 time in geo_provider), and write narrow unit tests for each module. The facade must re-export all symbols currently imported by 30+ consumers.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GDEC-01 | geo_service.py split into 5 domain modules (metadata_fetch, download_execution, archive_processing, matrix_parsing, concatenation) | Method-to-module mapping in Architecture Patterns section; line ranges and dependency analysis complete |
| GDEC-02 | GEOService class preserved as backward-compatible facade | 30+ import sites catalogued; facade pattern already in `geo/facade.py`; all public symbols identified |
| GDEC-03 | SOFT-download logic deduplicated between geo_service and geo_provider | 7 copy-paste instances identified (6 in geo_service, 1 in geo_provider); extraction to shared helper detailed |
| GDEC-04 | Each extracted module has narrow unit tests | Test framework (pytest) verified; existing test patterns documented; requirement-to-test mapping provided |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12+ | Language runtime | Project requirement (CLAUDE.md) |
| pytest | 9.0.1 | Test framework | Already in use, 167+ GEO-related tests exist |
| unittest.mock | stdlib | Mocking for unit tests | Standard in existing GEO tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| GEOparse | installed | SOFT file parsing | Used by metadata_fetch and download_execution modules |
| anndata | installed | AnnData handling | Used by archive_processing and concatenation modules |
| pandas | installed | DataFrame handling | Used by matrix_parsing and validation |

No new dependencies needed. This is pure internal restructuring.

## Architecture Patterns

### Current geo_service.py Method-to-Module Mapping

The 59 methods in `GEOService` cluster into 5 clear domains:

#### Module 1: `metadata_fetch.py` (~750 LOC)
Methods that fetch, extract, validate, and combine metadata:
- `fetch_metadata_only()` -- main entry point (line 525)
- `_fetch_gse_metadata()` -- GEOparse SOFT download + parse (line 567)
- `_fetch_gds_metadata_and_convert()` -- GDS-to-GSE resolution (line 704)
- `_combine_gds_gse_metadata()` -- merge GDS + GSE metadata (line 807)
- `_fetch_gse_metadata_via_entrez()` -- Entrez fallback (line 876)
- `_convert_entrez_to_lobster_metadata()` -- Entrez format conversion (line 1021)
- `_extract_platform_info_from_entrez()` -- platform extraction (line 1117)
- `_extract_metadata()` -- GEOparse GSE extraction (line 2872)
- `_safely_extract_metadata_field()` -- safe field extraction (line 3002)
- `_validate_geo_metadata()` -- metadata schema validation (line 3034)
- `_check_platform_compatibility()` -- platform validation + LLM modality (line 2403)
- `_detect_sample_types()` -- sample type classification (line 2725)

**Dependencies:** data_manager (for metadata store), ssl_utils, GEOparse, DataExpertAssistant

#### Module 2: `download_execution.py` (~800 LOC)
Methods that coordinate downloads, run pipelines, and call GEOparse:
- `download_dataset()` -- main entry point (line 1160)
- `download_with_strategy()` -- pipeline coordination (line 1509)
- `_get_processing_pipeline()` -- strategy engine delegation (line 1612)
- `_try_processed_matrix_first()` -- pipeline step (line 1656)
- `_try_raw_matrix_first()` -- pipeline step (line 1716)
- `_try_h5_format_first()` -- pipeline step (line 1773)
- `_try_supplementary_first()` -- pipeline step (line 1857)
- `_try_supplementary_fallback()` -- pipeline step (line 1955)
- `_try_emergency_fallback()` -- pipeline step (line 2027)
- `_try_geoparse_download()` -- pipeline step (line 2122)

**Dependencies:** metadata_fetch (for metadata), archive_processing, matrix_parsing, concatenation, data_manager

#### Module 3: `archive_processing.py` (~600 LOC)
Methods that handle TAR archives, nested archives, and 10X file extraction:
- `_process_supplementary_files()` -- supplementary routing (line 3610)
- `_process_tar_file()` -- TAR download + extraction (line 3754)
- `_download_and_parse_file()` -- single file download + parse (line 4025)
- `_detect_kallisto_salmon_files()` -- quantification detection (line 3456)
- `_load_quantification_files()` -- quantification loading (line 3525)
- `_try_series_level_10x_trio()` -- series-level 10X detection (line 4692)
- `_download_10x_trio()` -- 10X trio download (line 4716)
- `_download_h5_file()` -- H5 file download (line 4877)

**Dependencies:** geo_downloader, geo_parser, tenx_loader, BulkRNASeqService

#### Module 4: `matrix_parsing.py` (~900 LOC)
Methods that validate, transform, and store expression matrices:
- `_get_sample_info()` -- sample info extraction (line 4074)
- `_download_sample_matrices()` -- parallel sample downloads (line 4145)
- `_download_single_sample()` -- single sample download (line 4191)
- `_extract_supplementary_files_from_metadata()` -- file classification (line 4293)
- `_initialize_file_type_patterns()` -- regex patterns (line 4363)
- `_extract_all_supplementary_urls()` -- URL extraction (line 4474)
- `_classify_single_file()` -- file scoring (line 4529)
- `_validate_10x_trio_completeness()` -- 10X validation (line 4576)
- `_download_and_combine_single_cell_files()` -- SC file combination (line 4607)
- `_detect_features_format()` -- features format detection (line 4660)
- `_load_10x_manual()` -- manual 10X loading (line 4674)
- `_validate_matrices()` -- batch matrix validation (line 5098)
- `_validate_single_matrix()` -- single matrix validation (line 5191)
- `_is_valid_expression_matrix()` -- expression validity check (line 5318)
- `_determine_transpose_biologically()` -- transpose decision (line 4923)
- `_download_single_expression_file()` -- expression file download (line 5036)
- `_determine_data_type_from_metadata()` -- data type detection (line 3265)
- `_format_metadata_summary()` -- metadata summary formatting (line 3287)

**Dependencies:** geo_downloader, geo_parser, data_manager, GEOparse

#### Module 5: `concatenation.py` (~450 LOC)
Methods that store samples and concatenate them:
- `_store_samples_as_anndata()` -- sample storage (line 5396)
- `_analyze_gene_coverage_and_decide_join()` -- join strategy (line 5550)
- `_concatenate_stored_samples()` -- ConcatenationService delegation (line 5679)
- `_store_single_sample_as_modality()` -- legacy single sample (line 5832)
- `_inject_clinical_metadata()` -- clinical data injection (line 3101)

**Dependencies:** data_manager, ConcatenationService, DataExpertAssistant

### Shared Helpers (stay in geo_service.py or move to utils/)
- `_is_data_valid()` -- data validity check (line 277)
- `_retry_with_backoff()` -- retry logic (line 308)
- `RetryOutcome` enum (line 178)
- `RetryResult` dataclass (line 186)
- `ARCHIVE_EXTENSIONS` constant (line 87)
- `_is_archive_url()` helper (line 90)
- `_score_expression_file()` helper (line 96)

### Recommended Project Structure (Post-Decomposition)

```
lobster/services/data_access/geo/
├── __init__.py              # NEW: Package exports (empty or re-exports)
├── constants.py             # EXISTING: Enums, dataclasses, platform registry
├── downloader.py            # EXISTING: GEODownloadManager
├── parser.py                # EXISTING: GEOParser, ParseResult
├── strategy.py              # EXISTING: PipelineStrategyEngine
├── facade.py                # EXISTING: Will become real facade (GDEC-02)
├── metadata_fetch.py        # NEW: Module 1 (GDEC-01)
├── download_execution.py    # NEW: Module 2 (GDEC-01)
├── archive_processing.py    # NEW: Module 3 (GDEC-01)
├── matrix_parsing.py        # NEW: Module 4 (GDEC-01)
├── concatenation.py         # NEW: Module 5 (GDEC-01)
├── soft_download.py         # NEW: Shared SOFT pre-download helper (GDEC-03)
├── helpers.py               # NEW: Shared utilities (_is_data_valid, _retry_with_backoff, etc.)
├── loaders/
│   └── tenx.py              # EXISTING: 10X Genomics loader
├── metadata/                # EXISTING (empty) -- NOT used; metadata_fetch.py is flat
├── sample/                  # EXISTING (empty) -- NOT used; matrix_parsing.py is flat
└── utils/                   # EXISTING (empty) -- NOT used; helpers.py is flat
```

### Pattern 1: Facade Delegation

The `GEOService` class in `geo_service.py` becomes a thin facade. Each method delegates to the extracted module.

```python
# geo_service.py (after decomposition) -- FACADE
from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
from lobster.services.data_access.geo.download_execution import DownloadExecutor
from lobster.services.data_access.geo.archive_processing import ArchiveProcessor
from lobster.services.data_access.geo.matrix_parsing import MatrixParser
from lobster.services.data_access.geo.concatenation import SampleConcatenator
from lobster.services.data_access.geo.helpers import RetryMixin

class GEOService:
    def __init__(self, data_manager, cache_dir=None, console=None, email=None):
        # ... same init, but instantiate sub-modules
        self._metadata = MetadataFetcher(data_manager, cache_dir, console, ...)
        self._downloader = DownloadExecutor(self, data_manager, ...)
        self._archive = ArchiveProcessor(cache_dir, geo_downloader, geo_parser, ...)
        self._matrix = MatrixParser(data_manager, geo_downloader, geo_parser, ...)
        self._concat = SampleConcatenator(data_manager, ...)

    def fetch_metadata_only(self, geo_id):
        return self._metadata.fetch_metadata_only(geo_id)

    def download_dataset(self, geo_id, adapter=None, **kwargs):
        return self._downloader.download_dataset(geo_id, adapter, **kwargs)

    # ... all existing public and private methods delegate
```

### Pattern 2: Mixin-Based Approach (Alternative)

Since many methods reference `self.data_manager`, `self.cache_dir`, `self.geo_downloader`, etc., a mixin approach avoids passing many parameters:

```python
# Each module defines a mixin class
class MetadataFetchMixin:
    """Methods for fetching and validating GEO metadata."""
    def fetch_metadata_only(self, geo_id):
        # Uses self.data_manager, self.cache_dir directly
        ...

# GEOService composes all mixins
class GEOService(MetadataFetchMixin, DownloadExecutionMixin, ...):
    def __init__(self, data_manager, ...):
        # Same init as before
        ...
```

**Recommendation: Use composition (Pattern 1), not mixins.** Mixins create tight coupling through shared `self` and make testing harder. Composition with explicit dependencies is cleaner and enables narrow unit testing per GDEC-04.

### Pattern 3: SOFT Pre-Download Extraction (GDEC-03)

The SOFT pre-download block is copied **7 times** (6 in geo_service.py, 1 in geo_provider.py):

| Location | Line | Context |
|----------|------|---------|
| `_fetch_gse_metadata()` | 580-616 | GSE metadata fetch |
| `_try_supplementary_first()` | 1864-1900 | Supplementary pipeline |
| `_try_supplementary_fallback()` | 1962-1998 | Supplementary fallback |
| `_try_emergency_fallback()` | 2036-2072 | Emergency fallback |
| `_try_geoparse_download()` | 2132-2168 | GEOparse pipeline |
| `_download_single_sample()` | 4207-4243 | GSM-level (different URL pattern!) |
| `geo_provider.py` | 760-796 | Provider metadata fetch |

Extract to `soft_download.py`:
```python
def pre_download_soft_file(
    geo_id: str,
    cache_dir: Path,
    id_type: str = "series",  # "series" or "samples"
) -> Path:
    """Pre-download SOFT file via HTTPS, bypassing GEOparse FTP.

    Returns path to downloaded file (may already exist from cache).
    """
    prefix = geo_id[:3]  # GSE or GSM
    num_str = geo_id[3:]

    if id_type == "samples":
        folder_prefix = f"GSM{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSMnnn"
        url_base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{folder_prefix}/{geo_id}"
    else:
        folder_prefix = f"GSE{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSEnnn"
        url_base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder_prefix}/{geo_id}"

    soft_file_path = cache_dir / f"{geo_id}_family.soft.gz"
    soft_url = f"{url_base}/soft/{geo_id}_family.soft.gz"

    if soft_file_path.exists():
        return soft_file_path

    # ... download with SSL handling ...
    return soft_file_path
```

### Anti-Patterns to Avoid

- **Circular imports:** The download_execution module calls methods that end up in archive_processing, matrix_parsing, and concatenation. Use forward references or pass callables, not imports between modules.
- **Breaking the facade signature:** Every existing `GEOService.method_name()` must continue to work. Tests import `GEOService` and call methods directly.
- **Moving constants out of geo_service.py without re-export:** `ARCHIVE_EXTENSIONS`, `_is_archive_url`, `_score_expression_file`, `RetryOutcome`, `RetryResult` are imported directly by tests. Must re-export from `geo_service.py`.

### Public API Surface (Must Be Preserved)

Symbols imported from `geo_service.py` across the codebase:

| Symbol | Imported By (count) | Type |
|--------|-------------------|------|
| `GEOService` | 30+ files (source + tests) | Class |
| `GEODataSource` | 2 files (actually from constants.py, re-exported) | Enum |
| `GEOResult` | 2 files (actually from constants.py, re-exported) | Dataclass |
| `RetryOutcome` | 1 test file | Enum |
| `RetryResult` | 1 test file | Dataclass |
| `ARCHIVE_EXTENSIONS` | 1 test file | Tuple |
| `_is_archive_url` | 1 test file | Function |
| `_score_expression_file` | 1 test file | Function |

**All of these must remain importable from `lobster.services.data_access.geo_service`.**

Note: `GEODataSource`, `GEOResult`, `GEOServiceError` are already defined in `geo/constants.py` and imported into `geo_service.py`. They are re-exported to consumers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Facade pattern boilerplate | Manual delegation for 59 methods | `__getattr__` forwarding for private methods | Only 2 public entry points need explicit delegation; private methods can forward via getattr |
| SOFT URL construction | Copy-paste URL building | Shared `soft_download.py` helper | 7 copies already exist; any fix must be applied 7 times currently |
| Test isolation | Full GEOService instantiation in unit tests | Direct module-level function testing with mocked dependencies | GDEC-04 requires narrow unit tests that run independently |

## Common Pitfalls

### Pitfall 1: Circular Import Between Extracted Modules
**What goes wrong:** `download_execution.py` calls `self._process_supplementary_files()` which is in `archive_processing.py`, but `archive_processing.py` may reference download helpers.
**Why it happens:** The original monolith had free cross-method calling. Extraction exposes implicit dependency cycles.
**How to avoid:** Map the call graph BEFORE extracting. download_execution calls INTO archive_processing, matrix_parsing, and concatenation (one-way). None of those call back into download_execution. This is already a DAG.
**Warning signs:** ImportError at module load time.

### Pitfall 2: Shared State Through `self`
**What goes wrong:** Methods like `_try_geoparse_download` set `self._use_intersecting_genes_only` which is read by `_concatenate_stored_samples`. After extraction, `self` refers to different objects.
**Why it happens:** The monolith uses instance variables as implicit communication channels.
**How to avoid:** Pass mutable state explicitly as parameters. The `_use_intersecting_genes_only` is set in `download_with_strategy()` and consumed in `_concatenate_stored_samples()` -- pass it through the call chain.
**Warning signs:** AttributeError for missing instance variables; silent None defaults changing behavior.

### Pitfall 3: Breaking Lazy Imports
**What goes wrong:** Several methods contain lazy imports (`from lobster.agents.data_expert.assistant import DataExpertAssistant`). Moving methods to new modules may trigger import-time side effects.
**Why it happens:** The original code used lazy imports to avoid circular dependencies and slow startup.
**How to avoid:** Preserve lazy import patterns exactly as-is in the extracted modules. Do not convert to module-level imports.
**Warning signs:** Slow startup, ImportError, circular import detection.

### Pitfall 4: Forgetting geo_provider.py Deduplication
**What goes wrong:** SOFT download is deduplicated in geo_service.py but the 7th copy in `geo_provider.py` (line 760) is forgotten.
**Why it happens:** geo_provider.py is in a different directory (`tools/providers/`), easy to miss.
**How to avoid:** GDEC-03 explicitly requires deduplication between geo_service AND geo_provider. The shared helper must be importable by both.
**Warning signs:** grep for "PRE-DOWNLOAD SOFT" finding surviving copies.

### Pitfall 5: Tests That Patch Private Methods
**What goes wrong:** Tests that mock `GEOService._fetch_gse_metadata` will break if the method moves to a different class.
**Why it happens:** The facade delegates, so `GEOService._fetch_gse_metadata` is no longer defined directly on the class.
**How to avoid:** The facade should have thin wrapper methods (not just `__getattr__`) for any method that tests mock. Alternatively, use `__getattr__` which makes patching work transparently.
**Warning signs:** `AttributeError` in test patching; `patch` targeting wrong class.

## Code Examples

### Example 1: SOFT Pre-Download Helper (Deduplication Target)
```python
# lobster/services/data_access/geo/soft_download.py
"""Shared SOFT file pre-download helper.

Replaces 7 copy-pasted blocks across geo_service.py and geo_provider.py.
"""
import urllib.request
from pathlib import Path
from typing import Optional

from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


def build_soft_url(geo_id: str) -> str:
    """Build HTTPS URL for a GEO SOFT file.

    Handles both GSE (series) and GSM (sample) accessions.
    """
    prefix = geo_id[:3]  # GSE or GSM
    num_str = geo_id[3:]

    if prefix == "GSM":
        folder_prefix = f"GSM{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSMnnn"
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{folder_prefix}/{geo_id}"
    else:
        folder_prefix = f"GSE{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSEnnn"
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder_prefix}/{geo_id}"

    return f"{base}/soft/{geo_id}_family.soft.gz"


def pre_download_soft_file(
    geo_id: str,
    cache_dir: Path,
) -> Optional[Path]:
    """Pre-download SOFT file via HTTPS, bypassing GEOparse's FTP downloader.

    Returns path to downloaded file, or None if download fails (GEOparse will retry via FTP).
    """
    soft_file_path = cache_dir / f"{geo_id}_family.soft.gz"
    if soft_file_path.exists():
        logger.debug(f"Using cached SOFT file: {soft_file_path}")
        return soft_file_path

    soft_url = build_soft_url(geo_id)
    logger.debug(f"Pre-downloading SOFT file using HTTPS: {soft_url}")

    try:
        ssl_context = create_ssl_context()
        with urllib.request.urlopen(soft_url, context=ssl_context) as response:
            with open(soft_file_path, "wb") as f:
                f.write(response.read())
        logger.debug(f"Successfully pre-downloaded SOFT file to {soft_file_path}")
        return soft_file_path
    except Exception as e:
        error_str = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
            handle_ssl_error(e, soft_url, logger)
            raise Exception(
                "SSL certificate verification failed when downloading SOFT file. "
                "See error message above for solutions."
            )
        logger.warning(f"Pre-download failed: {e}. GEOparse will attempt download.")
        return None
```

### Example 2: Facade Pattern (geo_service.py After Decomposition)
```python
# geo_service.py -- becomes thin facade
# All existing symbols remain importable from this path

from lobster.services.data_access.geo.constants import (
    GEODataSource, GEOResult, GEOServiceError, ...
)
from lobster.services.data_access.geo.helpers import (
    RetryOutcome, RetryResult, ARCHIVE_EXTENSIONS, _is_archive_url, _score_expression_file,
)
from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
from lobster.services.data_access.geo.download_execution import DownloadExecutor
# ... other imports

class GEOService:
    """Facade preserving the original API surface."""

    def __init__(self, data_manager, cache_dir=None, console=None, email=None):
        # Same initialization
        self.data_manager = data_manager
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_manager.cache_dir / "geo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console
        # ... same helper init ...

        # Compose domain modules (they share our instance attributes)
        self._metadata_fetcher = MetadataFetcher(self)
        self._download_executor = DownloadExecutor(self)
        self._archive_processor = ArchiveProcessor(self)
        self._matrix_parser = MatrixParser(self)
        self._concatenator = SampleConcatenator(self)

    # Public entry points -- explicit delegation
    def fetch_metadata_only(self, geo_id):
        return self._metadata_fetcher.fetch_metadata_only(geo_id)

    def download_dataset(self, geo_id, adapter=None, **kwargs):
        return self._download_executor.download_dataset(geo_id, adapter, **kwargs)

    def download_with_strategy(self, geo_id, manual_strategy_override=None,
                                use_intersecting_genes_only=None):
        return self._download_executor.download_with_strategy(
            geo_id, manual_strategy_override, use_intersecting_genes_only)

    # Private methods -- delegate via __getattr__ for backward compat
    def __getattr__(self, name):
        """Forward private method calls to domain modules."""
        for module in [self._metadata_fetcher, self._download_executor,
                       self._archive_processor, self._matrix_parser,
                       self._concatenator]:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"'GEOService' has no attribute '{name}'")
```

**Note on `__getattr__`:** This only fires for attributes not found on the instance directly. Since all instance attributes (data_manager, cache_dir, etc.) are set in `__init__`, they resolve normally. Only method lookups that would previously go to `GEOService` class methods now forward to domain modules. This preserves `unittest.mock.patch('...GEOService._method')` behavior transparently.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single 5,954 LOC file | 5 domain modules + facade | This phase | Testability, readability, ownership boundaries |
| 7x copy-pasted SOFT download | Single shared helper | This phase | Maintainability, single fix point |
| Empty placeholder dirs (metadata/, sample/, utils/) | Populated domain modules | This phase | Uses infrastructure already created |

**Pre-existing extractions (already done):**
- `geo/constants.py` (165 LOC) -- enums, dataclasses, platform registry
- `geo/downloader.py` (917 LOC) -- GEODownloadManager
- `geo/parser.py` (1,927 LOC) -- GEOParser, ParseResult
- `geo/strategy.py` (467 LOC) -- PipelineStrategyEngine
- `geo/loaders/tenx.py` (405 LOC) -- TenXGenomicsLoader

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.1 |
| Config file | `pytest.ini` |
| Quick run command | `source .venv/bin/activate && python -m pytest tests/unit/services/data_access/test_geo_*.py -x -q` |
| Full suite command | `source .venv/bin/activate && python -m pytest tests/unit/services/data_access/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GDEC-01 | 5 domain modules exist with focused responsibility | unit | `pytest tests/unit/services/data_access/test_geo_decomposition.py -x` | Wave 0 |
| GDEC-02 | GEOService facade preserves all imports and behavior | unit + integration | `pytest tests/unit/services/data_access/test_geo_facade_compat.py -x` | Wave 0 |
| GDEC-03 | SOFT download deduplicated, single location | unit + AST scan | `pytest tests/unit/services/data_access/test_soft_download.py -x` | Wave 0 |
| GDEC-04 | Each module has narrow unit tests | unit | `pytest tests/unit/services/data_access/test_geo_metadata_fetch.py tests/unit/services/data_access/test_geo_archive_processing.py tests/unit/services/data_access/test_geo_matrix_parsing.py tests/unit/services/data_access/test_geo_concatenation.py tests/unit/services/data_access/test_geo_download_execution.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/services/data_access/test_geo_*.py -x -q`
- **Per wave merge:** `python -m pytest tests/unit/services/data_access/ tests/integration/test_geo_*.py -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/services/data_access/test_soft_download.py` -- covers GDEC-03 (SOFT dedup)
- [ ] `tests/unit/services/data_access/test_geo_metadata_fetch.py` -- covers GDEC-04 (metadata module tests)
- [ ] `tests/unit/services/data_access/test_geo_download_execution.py` -- covers GDEC-04 (download module tests)
- [ ] `tests/unit/services/data_access/test_geo_archive_processing.py` -- covers GDEC-04 (archive module tests)
- [ ] `tests/unit/services/data_access/test_geo_matrix_parsing.py` -- covers GDEC-04 (matrix module tests)
- [ ] `tests/unit/services/data_access/test_geo_concatenation.py` -- covers GDEC-04 (concatenation module tests)
- [ ] `tests/unit/services/data_access/test_geo_facade_compat.py` -- covers GDEC-02 (facade backward compat)
- [ ] `tests/unit/services/data_access/test_geo_decomposition.py` -- covers GDEC-01 (structural verification)

## Open Questions

1. **__getattr__ vs explicit delegation for private methods**
   - What we know: Tests mock private methods like `_fetch_gse_metadata`. `__getattr__` forwarding preserves mock targets transparently.
   - What's unclear: Does `__getattr__` on a class with `__init__`-set attributes have any performance impact for frequently called methods?
   - Recommendation: Use `__getattr__` for private methods. The performance cost is negligible since Python's normal attribute lookup resolves instance attributes first. Only missing attributes trigger `__getattr__`.

2. **Empty placeholder directories (metadata/, sample/, utils/)**
   - What we know: These exist and are empty. The new module names (metadata_fetch.py, etc.) don't match them exactly.
   - What's unclear: Should we use these directories or flat files?
   - Recommendation: Use flat files in `geo/`. The modules are not large enough to warrant subdirectories. The empty dirs can be cleaned up in Phase 9 (HYGN-03).

3. **Module-level vs class-based extraction**
   - What we know: Methods heavily use `self.data_manager`, `self.cache_dir`, `self.geo_downloader`.
   - What's unclear: Should extracted modules use classes or free functions with explicit params?
   - Recommendation: Use classes that receive the parent GEOService instance (or its key attributes). This minimizes parameter threading while keeping dependencies explicit for testing.

## Sources

### Primary (HIGH confidence)
- `lobster/services/data_access/geo_service.py` -- 5,954 LOC, all 59 methods analyzed
- `lobster/services/data_access/geo/` -- existing subpackage structure verified
- `lobster/tools/providers/geo_provider.py` -- SOFT duplication confirmed (line 760)
- 30+ import sites across source and tests -- complete public API surface mapped
- `tests/unit/services/data_access/test_geo_*.py` -- 9 existing test files, 167+ test functions

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` -- Phase 4 requirements and success criteria
- `.planning/REQUIREMENTS.md` -- GDEC-01 through GDEC-04 definitions
- Previous phase decisions (STATE.md) -- patterns established in Phases 1-3

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, pure internal restructuring
- Architecture: HIGH -- method mapping verified line-by-line against 5,954 LOC source
- Pitfalls: HIGH -- based on actual code analysis (circular imports, shared state, lazy imports all verified in source)
- SOFT duplication count: HIGH -- grep verified 7 instances across 2 files

**Research date:** 2026-03-04
**Valid until:** No expiry (internal codebase analysis, not library version dependent)
