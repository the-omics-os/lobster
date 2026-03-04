# Phase 3: GEO Strategy Engine Hardening - Research

**Researched:** 2026-03-04
**Domain:** GEO pipeline strategy engine (null handling, rule correctness, dead code)
**Confidence:** HIGH

## Summary

Phase 3 targets three bugs in the GEO strategy engine: (1) the `_sanitize_null_values` method in `DataExpertAssistant` converts missing values to the string `"NA"` instead of empty string, which is truthy and causes downstream logic to think files exist when they do not; (2) strategy derivation relies on `bool()` truthiness checks and inline string lists for file types rather than explicit null checks and typed constants; (3) the `ARCHIVE_FIRST` pipeline type is defined in the enum and has a pipeline mapping in `get_pipeline_functions` but **no rule ever returns it** -- it is dead code.

The root cause chain is clear: LLM returns JSON with null/None for missing fields --> `_sanitize_null_values` converts to `"NA"` --> `PipelineContext.has_file()` calls `bool("NA")` which is `True` --> `data_availability` returns `PARTIAL` instead of `NONE` --> wrong pipeline selected --> pipeline step tries to download `"NA".{filetype}` --> fails but wastes a pipeline step before falling through. In `_derive_analysis`, `bool(get("processed_matrix_name", ""))` with value `"NA"` also returns `True`, causing `_create_recommended_strategy` to pick `MATRIX_FIRST` when no matrix exists.

All three issues are well-scoped, self-contained, and affect only the files identified below. No external library research is needed -- this is pure internal logic repair.

**Primary recommendation:** Fix `_sanitize_null_values` to convert to empty string `""` (not `"NA"`), add a `_is_null_value()` helper for explicit null checks throughout strategy code, create an `ALLOWED_FILETYPES` constant, and remove the `ARCHIVE_FIRST` dead branch entirely.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GSTR-01 | Null sanitization stores missing values as empty string/None, never truthy "NA" | Root cause identified: `_sanitize_null_values` in `assistant.py:146-234` converts None/null to `"NA"` string. Fix: change to `""`. Impact traced through 5 downstream consumers. |
| GSTR-02 | Strategy derivation uses explicit null checks and allowed file type enums | Bug locations identified: `PipelineContext.has_file()` uses `bool()`, rules use inline `["txt", "csv", ...]` lists, `_derive_analysis` uses `bool(get(...))`. Fix: add `_is_null_value()` helper, `ALLOWED_FILETYPES` constant. |
| GSTR-03 | ARCHIVE_FIRST dead branch resolved -- either add triggering rule or remove | Confirmed dead: no rule returns `PipelineType.ARCHIVE_FIRST`. Only reachable via `manual_strategy_override` string conversion. Recommendation: remove entirely. |
</phase_requirements>

## Standard Stack

No external libraries needed. This phase modifies only internal Python logic.

### Core Files to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| `packages/lobster-research/lobster/agents/data_expert/assistant.py` | LLM response sanitization | Fix `_sanitize_null_values` to use `""` not `"NA"` |
| `lobster/services/data_access/geo/strategy.py` | Pipeline strategy engine | Add null-check helper, `ALLOWED_FILETYPES` constant, remove `ARCHIVE_FIRST` |
| `lobster/services/data_access/geo_queue_preparer.py` | Queue preparation + strategy derivation | Fix `_derive_analysis` to use explicit null checks |
| `lobster/services/data_access/geo_service.py` | Pipeline step methods | Fix `not matrix_name` guards to use null-check helper |

### Supporting Files (Read-Only Context)

| File | Purpose |
|------|---------|
| `lobster/core/schemas/download_queue.py` | StrategyConfig schema (separate from assistant's StrategyConfig) |
| `lobster/services/data_access/geo/constants.py` | GEO constants -- potential home for `ALLOWED_FILETYPES` |

## Architecture Patterns

### Current Data Flow (Bug Path)

```
LLM JSON response (null fields)
  --> extract_json_from_text()
  --> _sanitize_null_values() [BUG: converts None to "NA"]
  --> StrategyConfig(processed_matrix_name="NA", ...)
  --> .model_dump() stored in metadata_store
  --> PipelineContext(strategy_config={...})
  --> has_file("processed_matrix") calls bool("NA") == True [WRONG]
  --> data_availability returns PARTIAL instead of NONE [WRONG]
  --> ProcessedMatrixRule: name="NA", filetype="NA", "NA" not in allowed list --> None
  --> But data_availability was wrong, so SingleCellWithRawDataRule/SupplementaryFilesRule may mismatch
  --> Eventually falls through to correct pipeline, but via wrong path
```

### Fixed Data Flow

```
LLM JSON response (null fields)
  --> extract_json_from_text()
  --> _sanitize_null_values() [FIXED: converts None to ""]
  --> StrategyConfig(processed_matrix_name="", ...)
  --> .model_dump() stored in metadata_store
  --> PipelineContext(strategy_config={...})
  --> has_file("processed_matrix") calls bool("") == False [CORRECT]
  --> data_availability returns NONE [CORRECT]
  --> NoDirectFilesRule matches --> SUPPLEMENTARY_FIRST [CORRECT]
```

### Pattern: Null Value Handling

**The `_is_null_value()` helper pattern:**

```python
_NULL_STRINGS = frozenset({
    "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "nil", "NIL", ""
})

def _is_null_value(value: Any) -> bool:
    """Check if a value represents null/missing data."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in _NULL_STRINGS:
        return True
    return False
```

This helper should be used in:
- `PipelineContext.has_file()`: `return not _is_null_value(self.strategy_config.get(...))`
- `PipelineContext.get_file_info()`: return `""` for null values
- Rule `evaluate()` methods: explicit null check before filetype comparison
- `_derive_analysis()`: `has_processed_matrix = not _is_null_value(get(...))`
- Pipeline step methods in `geo_service.py`: replace `not matrix_name` with `_is_null_value(matrix_name)`

### Pattern: Allowed File Types Constant

```python
# In strategy.py (or constants.py if preferred)
MATRIX_FILETYPES = frozenset({"txt", "csv", "tsv", "h5", "h5ad"})
RAW_MATRIX_FILETYPES = frozenset({"txt", "csv", "tsv", "mtx", "h5"})
H5_FILETYPES = frozenset({"h5", "h5ad"})
```

These replace the inline lists in `ProcessedMatrixRule`, `RawMatrixRule`, and `H5FormatRule`.

### Anti-Patterns to Avoid

- **Do NOT add an ArchiveFirstRule:** The `ARCHIVE_FIRST` pipeline is redundant with `_try_supplementary_first` which already handles archives. Adding a rule would create a new pipeline path with no clear advantage over the existing archive handling in SUPPLEMENTARY_FIRST and FALLBACK pipelines. Remove the dead branch instead.
- **Do NOT change the StrategyConfig Pydantic model defaults:** The `StrategyConfig` model in `assistant.py` already defaults to `""` for string fields. The bug is in `_sanitize_null_values` overriding those defaults to `"NA"`. Fix the sanitizer, not the model.
- **Do NOT remove `_sanitize_null_values` entirely:** LLMs genuinely return varied null representations ("null", "None", "nil"). The function is needed -- it just needs to sanitize to `""` instead of `"NA"`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Null value detection | Per-site inline checks | Shared `_is_null_value()` helper | 6+ call sites need consistent behavior |
| File type validation | Inline string lists | `frozenset` constants | Single source of truth, faster membership test |

**Key insight:** The current bugs exist precisely because null handling was hand-rolled differently at each call site. A shared helper eliminates the class of bug entirely.

## Common Pitfalls

### Pitfall 1: "NA" Values Already Persisted in metadata_store
**What goes wrong:** Fixing `_sanitize_null_values` only fixes new LLM extractions. Existing cached `strategy_config` dicts in `metadata_store` may already contain `"NA"` values from prior runs.
**Why it happens:** `metadata_store` is a session dict, but `geo_queue_preparer.recommend_strategy` checks for `cached_strategy_config` before re-extracting.
**How to avoid:** The `_is_null_value()` helper at the consumer side (PipelineContext, _derive_analysis) must treat `"NA"` as null regardless of whether the producer has been fixed. This provides defense-in-depth.
**Warning signs:** Tests pass with fresh strategy configs but fail with fixtures containing `"NA"`.

### Pitfall 2: `_format_file_display` Still References "NA"
**What goes wrong:** The method `_format_file_display` in `assistant.py:572-589` checks for `"NA"` explicitly. After fixing `_sanitize_null_values`, this code becomes dead but not wrong.
**How to avoid:** Update `_format_file_display` to use the same null-check pattern. Keep backward compatibility by checking both `"NA"` and `""`.

### Pitfall 3: Removing ARCHIVE_FIRST Breaks manual_strategy_override
**What goes wrong:** Users can pass `manual_strategy_override="ARCHIVE_FIRST"` to `get_pipeline_functions` which converts strings to PipelineType enums. Removing the enum member breaks this path.
**How to avoid:** Check if `manual_strategy_override` is used anywhere with `"ARCHIVE_FIRST"`. If not, safe to remove. If yes, map it to `SUPPLEMENTARY_FIRST` instead of removing entirely.
**Warning signs:** KeyError on `PipelineType["ARCHIVE_FIRST"]` in `get_pipeline_functions`.

### Pitfall 4: `bool()` on Pydantic Model Fields vs Dict Values
**What goes wrong:** `_derive_analysis` supports both Pydantic model and dict via `isinstance(strategy_config, dict)`. The Pydantic model's `raw_data_available` defaults to `False`, but the dict version might have `"NA"` stored.
**How to avoid:** Apply `_is_null_value()` consistently regardless of whether source is Pydantic or dict.

### Pitfall 5: Two Different StrategyConfig Classes
**What goes wrong:** There are TWO classes named `StrategyConfig`:
1. `packages/lobster-research/lobster/agents/data_expert/assistant.py:StrategyConfig` -- LLM extraction result (has file names)
2. `lobster/core/schemas/download_queue.py:StrategyConfig` -- Download queue entry strategy (has strategy_name, confidence)
These are different schemas with different purposes.
**How to avoid:** Be clear which one is being modified. GSTR-01 affects assistant.py's StrategyConfig. The download_queue.py version is unaffected.

## Code Examples

### Fix 1: _sanitize_null_values (GSTR-01)

Current (buggy):
```python
# assistant.py:160-165 -- converts None to truthy "NA"
if value is None:
    if key == "raw_data_available":
        sanitized[key] = False
    else:
        sanitized[key] = "NA"  # BUG: truthy string
```

Fixed:
```python
# assistant.py:160-165 -- converts None to falsy ""
if value is None:
    if key == "raw_data_available":
        sanitized[key] = False
    else:
        sanitized[key] = ""  # FIXED: falsy empty string
```

Same pattern at lines 195-199 and 226-230 (string and catch-all branches).

### Fix 2: has_file with explicit null check (GSTR-02)

Current (buggy):
```python
# strategy.py:49-51
def has_file(self, file_type: str) -> bool:
    return bool(self.strategy_config.get(f"{file_type}_name"))
```

Fixed:
```python
# strategy.py:49-51
def has_file(self, file_type: str) -> bool:
    value = self.strategy_config.get(f"{file_type}_name", "")
    return not _is_null_value(value)
```

### Fix 3: Rule with allowed filetypes constant (GSTR-02)

Current:
```python
# strategy.py:101-104
def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
    name, filetype = context.get_file_info("processed_matrix")
    if name and filetype in ["txt", "csv", "tsv", "h5", "h5ad"]:
        ...
```

Fixed:
```python
# strategy.py:101-104
def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
    name, filetype = context.get_file_info("processed_matrix")
    if not _is_null_value(name) and filetype in MATRIX_FILETYPES:
        ...
```

### Fix 4: Remove ARCHIVE_FIRST (GSTR-03)

Remove from:
1. `PipelineType` enum (line 26)
2. `get_pipeline_functions` pipeline_map (lines 346-351)
3. `_try_archive_extraction_first` method in `geo_service.py` (lines 1954-2009) -- only if no other caller references it
4. String conversion fallback in `get_pipeline_functions` should map unknown strings to FALLBACK (already does via `.get()`)

### Fix 5: _derive_analysis explicit null checks (GSTR-02)

Current (buggy):
```python
# geo_queue_preparer.py:395-397
has_processed_matrix = bool(get("processed_matrix_name", ""))
has_raw_matrix = bool(get("raw_UMI_like_matrix_name", ""))
raw_data_available = bool(get("raw_data_available", False))
```

Fixed:
```python
# geo_queue_preparer.py:395-397
has_processed_matrix = not _is_null_value(get("processed_matrix_name", ""))
has_raw_matrix = not _is_null_value(get("raw_UMI_like_matrix_name", ""))
raw_data_available = bool(get("raw_data_available", False)) and not _is_null_value(get("raw_data_available", False))
```

## State of the Art

Not applicable -- this is internal bug fixing, not library adoption.

## Open Questions

1. **Where should `_is_null_value()` live?**
   - What we know: It's needed in `strategy.py`, `geo_queue_preparer.py`, `geo_service.py`, and `assistant.py`
   - What's unclear: Whether to put it in a shared utility module or duplicate in each file
   - Recommendation: Define in `strategy.py` since that's the primary consumer, and import from there. Alternatively, add to `geo/constants.py` alongside other GEO constants. For `assistant.py`, either import or define a local copy (to avoid circular imports with the agent package).

2. **Should `_try_archive_extraction_first` be removed along with ARCHIVE_FIRST?**
   - What we know: `_try_archive_extraction_first` is only referenced in the ARCHIVE_FIRST pipeline mapping
   - What's unclear: Whether it's also called by any other code path
   - Recommendation: Grep confirms only 2 references: the pipeline map and the method definition. Safe to remove both. But check if `_try_supplementary_first` already covers archive extraction (it does -- it calls `_process_tar_file` for archives).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.x |
| Config file | `pytest.ini` |
| Quick run command | `pytest tests/unit/services/data_access/test_geo_strategy.py -x --no-cov` |
| Full suite command | `pytest tests/unit/ -x --no-cov -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GSTR-01 | `_sanitize_null_values` converts None/"NA"/"null" to "" not "NA" | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestNullSanitization -x --no-cov` | No -- Wave 0 |
| GSTR-01 | `has_file` returns False for "NA" values | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestPipelineContext -x --no-cov` | No -- Wave 0 |
| GSTR-01 | `_derive_analysis` returns False for "NA" strategy fields | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestDeriveAnalysis -x --no-cov` | No -- Wave 0 |
| GSTR-02 | ProcessedMatrixRule uses MATRIX_FILETYPES constant | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestRulesWithNulls -x --no-cov` | No -- Wave 0 |
| GSTR-02 | data_availability returns NONE when all fields are null | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestDataAvailability -x --no-cov` | No -- Wave 0 |
| GSTR-02 | Pipeline steps reject "NA" matrix names gracefully | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestPipelineStepNullGuards -x --no-cov` | No -- Wave 0 |
| GSTR-03 | ARCHIVE_FIRST removed from PipelineType enum | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestArchiveFirstRemoved -x --no-cov` | No -- Wave 0 |
| GSTR-03 | No rule returns ARCHIVE_FIRST (regression guard) | unit | `pytest tests/unit/services/data_access/test_geo_strategy.py::TestNoDeadBranches -x --no-cov` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/services/data_access/test_geo_strategy.py -x --no-cov`
- **Per wave merge:** `pytest tests/unit/ -x --no-cov -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/services/data_access/test_geo_strategy.py` -- covers all GSTR requirements (new file)
- [ ] No framework install needed -- pytest already configured

## Sources

### Primary (HIGH confidence)
- Direct source code reading of `strategy.py`, `assistant.py`, `geo_queue_preparer.py`, `geo_service.py`
- All findings verified by tracing actual code paths and data flow

### Secondary (MEDIUM confidence)
- N/A -- no external documentation needed for internal bug fixes

### Tertiary (LOW confidence)
- N/A

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - pure internal code, no external dependencies
- Architecture: HIGH - full data flow traced from LLM response to pipeline selection
- Pitfalls: HIGH - each pitfall identified from actual code inspection

**Research date:** 2026-03-04
**Valid until:** Indefinite (internal codebase knowledge, not version-dependent)
