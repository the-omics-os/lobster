# Phase 1: GEO Safety & Contract Hotfixes - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 6 correctness/security bugs in the GEO pipeline: GDS canonicalization (GSAF-01), metadata key standardization (GSAF-02), centralized metadata writes (GSAF-03), tar path traversal security (GSAF-04), typed retry results (GSAF-05), and download orchestrator race condition (GSAF-06). No new features, no file moves, no decomposition. Pure correctness and safety fixes.

</domain>

<decisions>
## Implementation Decisions

### Tar Security Policy
- Reject entire archive when ANY path-traversal or unsafe member is found (no partial extraction)
- Block symlinks pointing outside extraction directory (beyond just ../ paths — covers CVE-2007-4559 pattern)
- Consolidate duplicated `_is_safe_member()` into `archive_utils.py` — geo/downloader.py calls the shared implementation
- Log offending member paths in rejection message (transparency for debugging scientific archives)

### Duplicate Worker Behavior
- Second worker attempting to claim an IN_PROGRESS entry gets silent no-op with log ("already being processed by another worker")
- Status transitions use compare-and-swap: `update_status()` takes `expected_current_status` param, returns `bool` (True = transitioned, False = precondition failed)
- Raise after exhausting retries (current behavior preserved)

### Retry Type Migration
- Hard cut: remove ALL string sentinel comparisons in geo_service.py and related files — no backward compat shim
- Keep raising after exhausting max retries (current exception behavior preserved)

### Metadata Key Standardization
- Enforce `_store_geo_metadata()` as the ONLY way to write GEO metadata — direct `metadata_store` dict assignment is a code smell to eliminate
- Validate required keys at write time: raise `ValueError` if `metadata` dict is missing, warn if validation result is `None`
- Standardize the validation key name everywhere (MetadataEntry TypedDict, DownloadQueueEntry, workspace_tool, all callers)

### Claude's Discretion
- Stale entry recovery mechanism (timeout-based vs manual) — evaluate based on usage patterns
- RetryResult type shape (minimal vs rich with retry metrics)
- Whether to extract `_retry_with_backoff` to shared `utils/retry.py` or keep in `geo/downloader.py`
- Which direction to fix the validation key mismatch (rename TypedDict key vs rename callers — pick least-churn)

</decisions>

<specifics>
## Specific Ideas

- Tar security: "A legitimate GEO archive should never contain ../ paths" — aggressive rejection is the right default for scientific data pipelines
- Race condition: compare-and-swap pattern preferred over file locks — simpler, no external dependencies
- Retry: preserve the raise-on-exhaustion control flow — callers already expect exceptions

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `archive_utils.py` (563 lines): Already has `ArchiveExtractor` class with `_is_safe_member()` — enhance this, don't create a new one
- `_store_geo_metadata()` in `data_manager_v2.py` (lines 2534-2594): Centralized helper already exists — enforce its exclusive use
- `_retry_with_backoff()` in `geo/downloader.py`: Already returns typed results, but call sites in `geo_service.py` may still compare strings

### Established Patterns
- MetadataEntry TypedDict (`data_manager_v2.py` lines 162-186): Uses `"validation"` key
- DownloadQueueEntry schema (`download_queue.py`): Uses `"validation_result"` key
- Download orchestrator (`download_orchestrator.py`): Explicitly states "NOT thread-safe" in comments

### Integration Points
- `geo_queue_preparer.py` line 265: `cached.get("validation_result")` — confirmed key mismatch with MetadataEntry
- `geo_queue_preparer.py` line 342: Already calls `_store_geo_metadata` — proof the pattern works
- `download_orchestrator.py` lines 294-351: Status transition flow (PENDING → IN_PROGRESS → COMPLETED) needs CAS wrapper
- `workspace_tool.py`: Uses "validation_result" — must be included in standardization

### Key Files (by requirement)
| Requirement | Primary Files |
|-------------|---------------|
| GSAF-01 (GDS canon) | `geo_queue_preparer.py`, `geo_utils.py`, `download_queue.py` |
| GSAF-02 (metadata keys) | `data_manager_v2.py`, `geo_queue_preparer.py`, `workspace_tool.py`, `download_queue.py` |
| GSAF-03 (write helper) | `data_manager_v2.py`, `geo_service.py` |
| GSAF-04 (tar security) | `archive_utils.py`, `geo/downloader.py` |
| GSAF-05 (retry types) | `geo/downloader.py`, `geo_service.py` |
| GSAF-06 (race condition) | `download_orchestrator.py`, `data_manager_v2.py` |

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-geo-safety-contract-hotfixes*
*Context gathered: 2026-03-03*
