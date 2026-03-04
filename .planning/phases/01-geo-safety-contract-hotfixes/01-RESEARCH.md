# Phase 1: GEO Safety & Contract Hotfixes - Research

**Researched:** 2026-03-04
**Domain:** GEO pipeline correctness, archive security, concurrency safety
**Confidence:** HIGH

## Summary

Phase 1 addresses 6 correctness and security bugs in the GEO download pipeline. The codebase is well-structured with clear separation of concerns: `geo_queue_preparer.py` (queue preparation), `geo_service.py` (download execution), `archive_utils.py` (archive handling), `download_orchestrator.py` (routing), and `download_queue.py` (queue persistence). All 6 fixes are isolated, non-feature changes that modify existing functions in-place.

The key challenge is the metadata key mismatch (`"validation"` in `MetadataEntry` TypedDict vs `"validation_result"` in `DownloadQueueEntry` and callers). The `_store_geo_metadata()` helper uses `"validation"` as its kwargs key, but `geo_service.py` directly assigns `"validation_result"` when enriching entries post-store. This inconsistency is the root cause. The second complexity is the `_retry_with_backoff` method returning `"SOFT_FILE_MISSING"` string sentinel mixed with `None` (failure) and actual result objects -- all three must be typed.

**Primary recommendation:** Fix each requirement as a surgical edit to existing files. No new files except a test file per requirement. Order: GSAF-02 (metadata key) first since it standardizes the contract all other fixes depend on, then GSAF-01 (GDS), GSAF-03 (write helper), GSAF-04 (tar security), GSAF-05 (retry types), GSAF-06 (race condition).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Tar Security Policy:** Reject entire archive when ANY path-traversal or unsafe member is found (no partial extraction). Block symlinks pointing outside extraction directory (beyond just ../ paths -- covers CVE-2007-4559 pattern). Consolidate duplicated `_is_safe_member()` into `archive_utils.py` -- geo/downloader.py calls the shared implementation. Log offending member paths in rejection message (transparency for debugging scientific archives).
- **Duplicate Worker Behavior:** Second worker attempting to claim an IN_PROGRESS entry gets silent no-op with log ("already being processed by another worker"). Status transitions use compare-and-swap: `update_status()` takes `expected_current_status` param, returns `bool` (True = transitioned, False = precondition failed). Raise after exhausting retries (current behavior preserved).
- **Retry Type Migration:** Hard cut: remove ALL string sentinel comparisons in geo_service.py and related files -- no backward compat shim. Keep raising after exhausting max retries (current exception behavior preserved).
- **Metadata Key Standardization:** Enforce `_store_geo_metadata()` as the ONLY way to write GEO metadata -- direct `metadata_store` dict assignment is a code smell to eliminate. Validate required keys at write time: raise `ValueError` if `metadata` dict is missing, warn if validation result is `None`. Standardize the validation key name everywhere (MetadataEntry TypedDict, DownloadQueueEntry, workspace_tool, all callers).

### Claude's Discretion
- Stale entry recovery mechanism (timeout-based vs manual) -- evaluate based on usage patterns
- RetryResult type shape (minimal vs rich with retry metrics)
- Whether to extract `_retry_with_backoff` to shared `utils/retry.py` or keep in `geo/downloader.py`
- Which direction to fix the validation key mismatch (rename TypedDict key vs rename callers -- pick least-churn)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GSAF-01 | GDS accessions canonicalized to GSE during queue preparation, with original_accession preserved | GDS handling exists in `geo_service.py:_fetch_gds_metadata_and_convert()` but queue preparer passes raw accession to `prepare_queue_entry()` which stores it as `dataset_id` without canonicalization |
| GSAF-02 | Metadata validation key standardized to `validation_result` everywhere | MetadataEntry TypedDict uses `"validation"`, DownloadQueueEntry uses `"validation_result"`, geo_service.py uses both -- 2 sites use `"validation_result"` for direct dict updates bypassing the helper |
| GSAF-03 | All GEO metadata_store writes use `_store_geo_metadata` helper -- no malformed entries | Found 4 direct `metadata_store[x] = ...` assignments in `geo_service.py` (lines 2534, 2586, 5695, 5699) bypassing the helper |
| GSAF-04 | Nested tar extraction applies safe-member path checks -- path traversal blocked | `archive_utils.py` has `_is_safe_member()` but skips unsafe members silently. `geo/downloader.py` has duplicate inline `is_safe_member()`. Neither checks symlinks |
| GSAF-05 | `_retry_with_backoff` returns typed result enum, not string sentinels | Returns `"SOFT_FILE_MISSING"` string sentinel (2 sites), `None` on failure, or actual result -- compared via `== "SOFT_FILE_MISSING"` at line 495 |
| GSAF-06 | Download orchestrator status transitions are atomic with precondition check -- no duplicate workers | `download_orchestrator.py` line 294 calls `update_status(entry_id, IN_PROGRESS)` without checking current status -- two workers can both succeed |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12+ | Runtime | Project standard per CLAUDE.md |
| Pydantic | v2 (existing) | Schema validation | Already used for DownloadQueueEntry, StrategyConfig |
| pytest | 7.0+ | Testing | Project standard per pytest.ini |
| tarfile | stdlib | Archive handling | Already used, no external dependency needed |
| threading | stdlib | Lock primitives | Already used in download_queue.py |
| enum | stdlib | RetryResult type | Lightweight, pattern-match friendly |
| dataclasses | stdlib | Structured types | For enriched RetryResult if needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fcntl | stdlib | File-based locking | Already used in queue_storage.py InterProcessFileLock |
| logging | stdlib | All logging | Already used throughout |

### Alternatives Considered
None -- all fixes use existing stdlib and project dependencies. No new packages needed.

## Architecture Patterns

### Recommended Project Structure
No new directories needed. All changes are in-place edits:
```
lobster/
├── core/
│   ├── archive_utils.py          # GSAF-04: enhance _is_safe_member(), add reject-all policy
│   ├── data_manager_v2.py        # GSAF-02/03: standardize MetadataEntry, enforce write helper
│   ├── download_queue.py         # GSAF-06: add CAS update_status variant
│   └── schemas/
│       └── download_queue.py     # GSAF-02: validation_result already correct here
├── services/data_access/
│   ├── geo_service.py            # GSAF-01/02/03/05: canonicalize GDS, fix keys, typed retry
│   └── geo_queue_preparer.py     # GSAF-01: canonicalize before entry creation
├── tools/
│   ├── download_orchestrator.py  # GSAF-06: CAS on status transition
│   └── workspace_tool.py         # GSAF-02: already uses "validation_result" (correct)
└── services/data_access/geo/
    └── downloader.py             # GSAF-04: delegate to shared _is_safe_member()
```

### Pattern 1: Compare-and-Swap (CAS) Status Transition (GSAF-06)
**What:** `update_status()` gains an `expected_current_status` parameter. Returns `bool` instead of the entry. Only transitions if current status matches expectation.
**When to use:** Any time a race condition exists between workers claiming work.
**Example:**
```python
# In download_queue.py
def update_status(
    self,
    entry_id: str,
    status: DownloadStatus,
    expected_current_status: Optional[DownloadStatus] = None,  # NEW
    modality_name: Optional[str] = None,
    error: Optional[str] = None,
    downloaded_by: Optional[str] = None,
) -> bool:  # Changed from DownloadQueueEntry
    """Update entry status. Returns True if transition succeeded."""
    with self._locked():
        entries = self._load_entries()
        for entry in entries:
            if entry.entry_id == entry_id:
                # CAS check
                if expected_current_status is not None:
                    current = entry.status
                    # Handle both enum and string comparison
                    current_val = current.value if hasattr(current, 'value') else current
                    expected_val = expected_current_status.value if hasattr(expected_current_status, 'value') else expected_current_status
                    if current_val != expected_val:
                        return False  # Precondition failed
                entry.update_status(status=status, modality_name=modality_name,
                                   error=error, downloaded_by=downloaded_by)
                self._backup_queue()
                self._write_entries_atomic(entries)
                return True
        raise EntryNotFoundError(f"Entry '{entry_id}' not found")
```

### Pattern 2: Typed Retry Result (GSAF-05)
**What:** Replace string sentinel `"SOFT_FILE_MISSING"` and `None` with a typed enum+data pattern.
**When to use:** When a function has multiple distinct outcomes beyond success/failure.
**Example:**
```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class RetryOutcome(Enum):
    SUCCESS = "success"
    EXHAUSTED = "exhausted"          # All retries failed
    SOFT_FILE_MISSING = "soft_file_missing"  # Permanent: file not on server

@dataclass
class RetryResult:
    outcome: RetryOutcome
    value: Optional[Any] = None      # The actual result on SUCCESS
    retries_used: int = 0
    total_delay: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.outcome == RetryOutcome.SUCCESS

    @property
    def needs_fallback(self) -> bool:
        return self.outcome == RetryOutcome.SOFT_FILE_MISSING
```

### Pattern 3: GDS Canonicalization (GSAF-01)
**What:** Before creating a queue entry, resolve GDS to GSE and preserve `original_accession`.
**When to use:** In `IQueuePreparer.prepare_queue_entry()` for GEO specifically.
**Example:**
```python
# In geo_queue_preparer.py or IQueuePreparer base
def prepare_queue_entry(self, accession: str, priority: int = 5):
    canonical = accession
    original_accession = None

    if accession.upper().startswith("GDS"):
        canonical = self._resolve_gds_to_gse(accession)
        original_accession = accession

    # ... rest of preparation uses canonical
    queue_entry = DownloadQueueEntry(
        dataset_id=canonical,
        metadata={**metadata, "original_accession": original_accession} if original_accession else metadata,
        ...
    )
```

### Pattern 4: Reject-All Archive Security (GSAF-04)
**What:** If ANY member fails safety check, reject the ENTIRE archive. Log all offending paths.
**When to use:** When partial extraction of a malicious archive is still dangerous.
**Example:**
```python
def _validate_archive_members(self, members: list, target_dir: Path) -> list:
    """Validate ALL members. Raises if ANY are unsafe."""
    unsafe = []
    for member in members:
        if not self._is_safe_member(member, target_dir):
            unsafe.append(member.name)
    if unsafe:
        raise RuntimeError(
            f"Archive rejected: {len(unsafe)} unsafe member(s) detected. "
            f"Offending paths: {unsafe[:10]}"  # Cap log at 10
        )
    return members
```

### Anti-Patterns to Avoid
- **String sentinels for typed outcomes:** `return "SOFT_FILE_MISSING"` mixes with the actual return type. Use enum+dataclass instead.
- **Direct dict assignment bypassing helpers:** `metadata_store[x] = {...}` bypasses validation. Always use `_store_geo_metadata()`.
- **Silent skip of unsafe archive members:** Skipping bad members silently means the user doesn't know the archive was malicious. Reject entirely.
- **Optimistic status transitions:** Setting status to IN_PROGRESS without checking it's PENDING allows race conditions. Always CAS.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| File-level locking | Custom fcntl wrapper | Existing `InterProcessFileLock` in `queue_storage.py` | Already battle-tested, handles POSIX + Windows |
| Atomic JSON writes | Manual temp file + rename | Existing `atomic_write_jsonl()` in `queue_storage.py` | Already handles edge cases |
| Path traversal detection | Custom string parsing | `Path.resolve()` + `os.path.commonpath()` | Existing pattern in `_is_safe_member()`, handles edge cases |
| Thread+process locking | Manual threading.Lock | Existing `queue_file_lock()` context manager | Composes thread lock + file lock |

**Key insight:** The project already has the right primitives. The bugs are in how they're used, not in missing infrastructure.

## Common Pitfalls

### Pitfall 1: MetadataEntry "validation" vs DownloadQueueEntry "validation_result"
**What goes wrong:** Code reads `entry.get("validation_result")` but MetadataEntry stores it under `"validation"`. Silent `None` return instead of crash.
**Why it happens:** Two TypedDicts evolved independently for the same concept.
**How to avoid:** Pick ONE key name. Rename `MetadataEntry.validation` to `validation_result` (matching DownloadQueueEntry which is the public API). Update `_store_geo_metadata()` kwargs accordingly.
**Warning signs:** Any `cached.get("validation_result")` on a MetadataEntry dict returns None even when validation data exists.

### Pitfall 2: Direct metadata_store assignments after _store_geo_metadata
**What goes wrong:** `_store_geo_metadata()` creates proper structure, then code immediately does `metadata_store[geo_id] = existing_entry` with ad-hoc keys like `"validation_result"` (note: wrong key vs TypedDict).
**Why it happens:** The helper was added later; existing enrichment code wasn't migrated.
**How to avoid:** All writes must go through the helper. For enrichment (adding fields to existing entry), add `_enrich_geo_metadata()` method or extend `_store_geo_metadata()` to support update mode.
**Warning signs:** `self.data_manager.metadata_store[` followed by `] =` in geo_service.py.

### Pitfall 3: CAS update_status backward compatibility
**What goes wrong:** Changing `update_status()` return type from `DownloadQueueEntry` to `bool` breaks existing callers.
**Why it happens:** Existing callers may use the returned entry object.
**How to avoid:** Make `expected_current_status` optional (default None = no check, old behavior). Keep returning `DownloadQueueEntry` but ALSO return a boolean. Or: return `Optional[DownloadQueueEntry]` where `None` = CAS failed.
**Warning signs:** Grep for `= *.update_status(` to find callers depending on return value.

### Pitfall 4: tarfile symlink escape
**What goes wrong:** A symlink in a tar archive points to `/etc/passwd`. Current `_is_safe_member()` only checks path components, not symlink targets.
**Why it happens:** `tarfile.TarInfo.issym()` and `tarfile.TarInfo.islnk()` distinguish symlinks and hardlinks, but current code doesn't check them.
**How to avoid:** In `_is_safe_member()`, also check `member.issym()` or `member.islnk()` and resolve the link target against `target_dir`.
**Warning signs:** `member.linkname` contains `../` or absolute paths.

### Pitfall 5: _retry_with_backoff caller assumptions
**What goes wrong:** 3 call sites of `_retry_with_backoff` in geo_service.py all check `if gse == "SOFT_FILE_MISSING"` and `if gse is None`. Changing the return type breaks all 3.
**Why it happens:** String sentinels are easy to miss when searching for call sites.
**How to avoid:** Grep for ALL call sites of `_retry_with_backoff` in geo_service.py. There are 3 (lines 485, 1768, 2093). Each must be updated to use the typed result.
**Warning signs:** Any remaining `== "SOFT_FILE_MISSING"` string comparison after migration.

### Pitfall 6: GDS canonicalization losing the original accession
**What goes wrong:** User submits GDS5826, queue entry has dataset_id=GSE67835. User can't find their original request.
**Why it happens:** Canonicalization replaces the accession without preserving the original.
**How to avoid:** Store `original_accession` in metadata or as a field on the queue entry. The success criterion explicitly requires this.
**Warning signs:** User searches queue for "GDS5826" and gets no results.

## Code Examples

### GSAF-02: Metadata Key Standardization (Least-Churn Direction)

The `MetadataEntry` TypedDict uses `"validation"` but `DownloadQueueEntry` uses `"validation_result"`. The least-churn fix is to rename `MetadataEntry.validation` to `validation_result` since:
- `DownloadQueueEntry.validation_result` is the public Pydantic API (can't rename without breaking JSON)
- `workspace_tool.py` already reads `"validation_result"` (correct)
- `geo_queue_preparer.py:265` reads `cached.get("validation_result")` (correct key name, wrong for current TypedDict)

```python
# data_manager_v2.py - MetadataEntry TypedDict fix
class MetadataEntry(TypedDict, total=False):
    metadata: Dict[str, Any]
    validation_result: Dict[str, Any]  # RENAMED from "validation"
    fetch_timestamp: str
    strategy_config: Dict[str, Any]
    stored_by: str
    modality_detection: Dict[str, Any]
    concatenation_decision: Dict[str, Any]

# _store_geo_metadata kwargs must also change:
def _store_geo_metadata(self, geo_id, metadata, stored_by, **kwargs):
    entry: MetadataEntry = {
        "metadata": metadata.copy(),
        "fetch_timestamp": kwargs.get("fetch_timestamp", datetime.now().isoformat()),
        "stored_by": stored_by,
    }
    if "validation_result" in kwargs:  # RENAMED from "validation"
        entry["validation_result"] = kwargs["validation_result"]
    # ... rest unchanged
```

**Affected callers of `_store_geo_metadata(..., validation=...)`:** Search for all kwarg usage and rename to `validation_result=`.

### GSAF-03: Direct metadata_store Assignment Sites

Found 4 direct assignments in geo_service.py that bypass the helper:

```
Line 2534: self.data_manager.metadata_store[geo_id] = existing_entry   # _validate_and_enrich_metadata
Line 2586: self.data_manager.metadata_store[geo_id] = existing_entry   # _validate_and_enrich_metadata (multimodal)
Line 5695: self.data_manager.metadata_store[modality_name] = existing_entry  # _handle_multi_sample_concatenation
Line 5699: self.data_manager.metadata_store[modality_name] = {...}     # _handle_multi_sample_concatenation (fallback)
```

Also 2 sites in the exception handlers (lines 541, 852) that first call `_store_geo_metadata()` but then do `existing_entry.update({"validation_result": ...})` directly on the dict. These should use an enrichment pattern:

```python
# Option: Add _enrich_geo_metadata to DataManagerV2
def _enrich_geo_metadata(self, geo_id: str, **fields) -> Optional[MetadataEntry]:
    """Add fields to existing metadata entry. Returns updated entry or None."""
    entry = self._get_geo_metadata(geo_id)
    if entry is None:
        logger.warning(f"Cannot enrich metadata for {geo_id}: no existing entry")
        return None
    entry.update(fields)
    self.metadata_store[geo_id] = entry  # Single controlled re-assignment
    return entry
```

### GSAF-04: Consolidated Secure Extraction

```python
# archive_utils.py - enhanced _is_safe_member
def _is_safe_member(self, member: tarfile.TarInfo, target_dir: Path) -> bool:
    """Check member is safe: no path traversal, no symlinks escaping target."""
    # 1. Check symlinks
    if member.issym() or member.islnk():
        link_target = Path(member.linkname)
        if link_target.is_absolute():
            logger.warning(f"Blocked absolute symlink: {member.name} -> {member.linkname}")
            return False
        # Resolve relative symlink against member's parent directory
        member_parent = (target_dir / Path(member.name)).parent
        resolved_link = (member_parent / link_target).resolve()
        if not str(resolved_link).startswith(str(target_dir.resolve())):
            logger.warning(f"Blocked symlink escaping target: {member.name} -> {member.linkname}")
            return False

    # 2. Check path traversal (existing logic)
    member_path = Path(member.name)
    try:
        target_path = (target_dir / member_path).resolve()
        common_path = Path(os.path.commonpath([target_dir.resolve(), target_path]))
        is_safe = common_path == target_dir.resolve()
        if not is_safe:
            logger.warning(f"Blocked unsafe TAR member: {member.name}")
        return is_safe
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Path safety check failed for {member.name}: {e}")
        return False

# extract_safely - reject-all policy
def extract_safely(self, archive_path, target_dir, cleanup_on_error=True):
    with tarfile.open(archive_path, "r:*") as tar:
        members = tar.getmembers()
        unsafe = [m for m in members if not self._is_safe_member(m, target_dir)]
        if unsafe:
            unsafe_names = [m.name for m in unsafe[:10]]
            raise RuntimeError(
                f"Archive rejected: {len(unsafe)} unsafe member(s). "
                f"Offending paths: {unsafe_names}"
            )
        tar.extractall(path=target_dir, members=members)
```

### GSAF-05: RetryResult Type

```python
# geo_service.py - new types near top
from enum import Enum
from dataclasses import dataclass

class RetryOutcome(Enum):
    SUCCESS = "success"
    EXHAUSTED = "exhausted"
    SOFT_FILE_MISSING = "soft_file_missing"

@dataclass
class RetryResult:
    outcome: RetryOutcome
    value: Any = None
    retries_used: int = 0

    @property
    def succeeded(self) -> bool:
        return self.outcome == RetryOutcome.SUCCESS

    @property
    def needs_fallback(self) -> bool:
        return self.outcome == RetryOutcome.SOFT_FILE_MISSING

# Updated _retry_with_backoff return:
def _retry_with_backoff(self, operation, operation_name, ...) -> RetryResult:
    # ... on success:
    return RetryResult(RetryOutcome.SUCCESS, value=result, retries_used=retry_count)
    # ... on SOFT_FILE_MISSING:
    return RetryResult(RetryOutcome.SOFT_FILE_MISSING, retries_used=retry_count)
    # ... on exhaustion:
    return RetryResult(RetryOutcome.EXHAUSTED, retries_used=retry_count)

# Updated caller (line ~485):
result = self._retry_with_backoff(...)
if result.needs_fallback:
    return self._fetch_gse_metadata_via_entrez(gse_id)
if not result.succeeded:
    return (None, None)
gse = result.value
```

### GSAF-06: Orchestrator CAS

```python
# download_orchestrator.py - CAS claim
def execute_download(self, entry_id, strategy_override=None):
    entry = self.data_manager.download_queue.get_entry(entry_id)
    if entry is None:
        raise ValueError(f"Queue entry '{entry_id}' not found")

    # CAS: only claim if PENDING or FAILED
    claimed = self.data_manager.download_queue.update_status(
        entry_id,
        DownloadStatus.IN_PROGRESS,
        expected_current_status=entry.status,  # Must still be what we saw
    )
    if not claimed:
        logger.info(f"Entry '{entry_id}' already being processed by another worker")
        return None, {}  # No-op

    # ... rest of download logic
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tarfile.extractall()` (Python < 3.12) | `tarfile.extractall(filter='data')` (Python 3.12+) | Python 3.12 | Built-in path traversal protection via `filter` parameter |
| String sentinels for retry | Typed enum results | This phase | Type safety, exhaustive matching |

**Note on Python 3.12 tarfile filter:** Python 3.12 added a `filter` parameter to `tarfile.extractall()` that can block path traversal and symlink attacks natively. However, since the project already has custom `_is_safe_member()` logic with domain-specific logging needs, and the CONTEXT.md specifies reject-all policy with logged offending paths, the custom approach is preferred. The Python 3.12 filter could be used as defense-in-depth alongside the custom validation.

**Deprecated/outdated:**
- `tarfile.extract()` without safety checks: still works but known CVE-2007-4559 vector

## Open Questions

1. **Return type of CAS update_status**
   - What we know: Current callers use `updated_entry = queue.update_status(...)`. Changing to `bool` breaks them.
   - What's unclear: How many callers depend on the returned entry object?
   - Recommendation: Return `Optional[DownloadQueueEntry]` -- `None` means CAS failed, entry object means success. Backward compatible for callers that don't check.

2. **Where to define RetryResult types**
   - What we know: Used only in geo_service.py currently. Could live there or in a shared module.
   - What's unclear: Whether other services will need retry logic.
   - Recommendation: Keep in `geo_service.py` for now (YAGNI). Move to shared utils only when a second consumer appears. This aligns with Claude's Discretion.

3. **GDS canonicalization: where exactly to intercept**
   - What we know: `IQueuePreparer.prepare_queue_entry()` is a template method that calls `fetch_metadata()` -> `extract_download_urls()` -> `recommend_strategy()` with the raw accession.
   - What's unclear: Should canonicalization happen in the base class or the GEO-specific subclass?
   - Recommendation: Override `prepare_queue_entry()` in `GEOQueuePreparer` to canonicalize before calling `super()`. The base class shouldn't know about GEO-specific accession types.

4. **Stale entry recovery (Claude's Discretion)**
   - What we know: With CAS, an entry stuck in IN_PROGRESS (worker crashed) stays stuck forever.
   - Recommendation: Add a simple timeout-based recovery: entries IN_PROGRESS for >1 hour are reset to PENDING. Implement as a method on DownloadQueue (`recover_stale_entries(timeout_seconds=3600)`). Do NOT auto-invoke -- let callers decide when to check.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ with pytest-cov, pytest-timeout |
| Config file | `pytest.ini` (root) + `[tool.pytest.ini_options]` in pyproject.toml |
| Quick run command | `pytest tests/unit/ -x --no-cov -q` |
| Full suite command | `pytest tests/ --timeout=300` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GSAF-01 | GDS accession canonicalized to GSE in queue entry | unit | `pytest tests/unit/services/data_access/test_geo_queue_preparer.py -x --no-cov` | Likely no dedicated file |
| GSAF-02 | validation_result key consistent across MetadataEntry and callers | unit | `pytest tests/unit/core/test_metadata_key_consistency.py -x --no-cov` | No -- Wave 0 |
| GSAF-03 | All metadata writes go through _store_geo_metadata | unit | `pytest tests/unit/core/test_store_geo_metadata.py -x --no-cov` | No -- Wave 0 |
| GSAF-04 | Path traversal and symlink archives rejected entirely | unit | `pytest tests/unit/core/test_archive_utils.py -x --no-cov` | Yes (exists, needs new tests) |
| GSAF-05 | _retry_with_backoff returns RetryResult, no string sentinels | unit | `pytest tests/unit/services/data_access/test_geo_retry_types.py -x --no-cov` | No -- Wave 0 |
| GSAF-06 | CAS status transition prevents duplicate workers | unit | `pytest tests/unit/core/test_download_queue.py -x --no-cov` | Yes (exists, needs CAS tests) |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x --no-cov -q --timeout=60`
- **Per wave merge:** `pytest tests/ --timeout=300 -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/services/data_access/test_geo_queue_preparer.py` -- covers GSAF-01 (GDS canonicalization)
- [ ] `tests/unit/core/test_metadata_key_consistency.py` -- covers GSAF-02 (key standardization verification)
- [ ] `tests/unit/core/test_store_geo_metadata.py` -- covers GSAF-03 (write helper enforcement)
- [ ] `tests/unit/services/data_access/test_geo_retry_types.py` -- covers GSAF-05 (typed retry results)
- [ ] Extend `tests/unit/core/test_archive_utils.py` with path-traversal rejection + symlink tests (GSAF-04)
- [ ] Extend `tests/unit/core/test_download_queue.py` with CAS update_status tests (GSAF-06)

## Sources

### Primary (HIGH confidence)
- Direct code inspection of all files listed in CONTEXT.md Code Context section
- `lobster/core/archive_utils.py` -- current `_is_safe_member()` implementation, lacks symlink checks
- `lobster/core/data_manager_v2.py` lines 162-186 -- `MetadataEntry` TypedDict with `"validation"` key
- `lobster/core/schemas/download_queue.py` line 254 -- `DownloadQueueEntry.validation_result` field
- `lobster/services/data_access/geo_service.py` lines 184-369 -- `_retry_with_backoff` full implementation with sentinel returns
- `lobster/services/data_access/geo_service.py` lines 571-670 -- `_fetch_gds_metadata_and_convert` (GDS handling)
- `lobster/tools/download_orchestrator.py` lines 293-297 -- optimistic IN_PROGRESS transition (no CAS)
- `lobster/core/download_queue.py` lines 144-198 -- `update_status()` without CAS
- `lobster/core/queue_storage.py` lines 70-76 -- existing `queue_file_lock` (thread + process safe)
- `lobster/services/data_access/geo/downloader.py` lines 175-199 -- duplicate inline `is_safe_member()`

### Secondary (MEDIUM confidence)
- Python 3.12 tarfile filter documentation (re: built-in path traversal protection)
- CVE-2007-4559 advisory (tar path traversal in Python)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing
- Architecture: HIGH -- all patterns verified against actual code, clear locations for each fix
- Pitfalls: HIGH -- all identified from direct code inspection with line numbers
- Validation: HIGH -- test framework well-established, gaps clearly identified

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable codebase, internal refactoring)
