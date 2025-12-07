# Bug Report: fast_dataset_search Returning Invalid GDS Identifiers Instead of GSE

## Summary
`fast_dataset_search` tool is returning invalid GDS accessions (e.g., `GDS200278021`) instead of valid GSE series accessions (e.g., `GSE278021`). This breaks downstream workflows as the generated URLs are invalid.

## Root Cause Analysis

### 1. User Observation
```
Query: "10x Genomics" single cell RNA-seq
Results shown:
- GDS200278021 (INVALID - doesn't exist)
- GDS200275038 (INVALID - doesn't exist)

Expected results:
- GSE278021 (VALID - exists)
- GSE275038 (VALID - exists)
```

### 2. Investigation Flow
```
research_agent.fast_dataset_search()
  → ContentAccessService.discover_datasets()
    → GEOProvider.search_publications()
      → GEOProvider.search_geo_datasets()  [queries db=gds, returns UIDs]
      → GEOProvider.get_dataset_summaries()  [FAILS silently]
      → GEOProvider.format_geo_search_results()  [Falls back to buggy logic]
```

### 3. Technical Root Cause

The NCBI "gds" database returns internal UIDs (like `200278021`) that are NOT accession numbers. The eSummary API must be called to get the actual accession (like `GSE278021`).

**Problem Chain:**
1. **eSearch** queries `db=gds`, returns ~2,230 total results, returns first 5 UIDs + webenv for full result set
2. **eSummary** tries to use `webenv` + `query_key` to fetch summaries
3. **NCBI API Error**: "Too many UIDs in request. Maximum number of UIDs is 500 for JSON format output."
   - Reason: webenv refers to ALL 2,230 results, not just the first 5
   - eSummary tries to fetch all 2,230 summaries, exceeds limit
4. **Error Handling**: `get_dataset_summaries()` catches error, logs it, returns empty list `[]`
5. **Fallback Bug**: `format_geo_search_results()` sees empty summaries, uses fallback logic:
   ```python
   # Line 1074 in geo_provider.py
   response += f"{i}. **GDS{dataset_id}** - ..."  # WRONG: blindly prefixes UID with "GDS"
   ```

### 4. Actual API Behavior (Verified via Debug Script)

**When eSummary succeeds** (using ID list instead of webenv):
```json
{
  "uid": "200278021",
  "accession": "GSE278021",  // ← Correct accession in this field
  "gse": "278021",
  "entrytype": "GSE",        // ← Entry type is GSE, not GDS
  "title": "Distinct genomic identities...",
  ...
}
```

**Key Fields:**
- `uid`: Internal NCBI identifier (not an accession)
- `accession`: Actual accession string with correct prefix (GSE/GDS/GPL/GSM)
- `entrytype`: Type of entry (GSE, GDS, GPL, GSM)

## The Bug

### Location: `lobster/tools/providers/geo_provider.py`

**Bug 1: eSummary using webenv incorrectly (line 607-609)**
```python
# Use WebEnv if available for efficiency
if search_result.web_env and search_result.query_key:
    url_params["webenv"] = search_result.web_env
    url_params["query_key"] = search_result.query_key
```

**Issue**: No `retmax` parameter when using webenv → tries to fetch ALL results → NCBI error

**Bug 2: Invalid fallback logic (line 1074)**
```python
if not search_result.summaries:
    # Just show IDs if no summaries available
    response += "### Dataset IDs Found\n"
    for i, dataset_id in enumerate(search_result.ids[:10], 1):
        response += f"{i}. **GDS{dataset_id}** - ..."  # BUG: Wrong prefix!
```

**Issue**: Blindly prefixes UID with "GDS", creates invalid accessions like "GDS200278021"

**Bug 3: Silent error handling (line 636-638)**
```python
except (json.JSONDecodeError, KeyError) as e:
    logger.error(f"Error parsing eSummary response: {e}")
    return []  # Silent failure → triggers buggy fallback
```

**Issue**: Doesn't check for `"error"` key in NCBI response, silently returns empty list

## Proposed Fix

### Option 1: Always Use ID List for eSummary (Recommended)

**Rationale**:
- Simpler and more reliable
- ID list approach works consistently (verified in debug script)
- webenv was meant for efficiency with large result sets, but we're limiting to max_results anyway

**Change in `get_dataset_summaries()` (line 606-612):**
```python
# Always use ID list instead of webenv for reliability
# webenv can refer to thousands of results, causing "Too many UIDs" errors
url_params["id"] = ",".join(search_result.ids[:500])  # NCBI limit: 500 for JSON

# Remove webenv code:
# if search_result.web_env and search_result.query_key:
#     url_params["webenv"] = search_result.web_env
#     url_params["query_key"] = search_result.query_key
```

### Option 2: Add retmax When Using webenv (Alternative)

**Change in `get_dataset_summaries()` (line 607-612):**
```python
if search_result.web_env and search_result.query_key:
    url_params["webenv"] = search_result.web_env
    url_params["query_key"] = search_result.query_key
    url_params["retstart"] = "0"
    url_params["retmax"] = str(min(len(search_result.ids), 500))  # Limit to avoid error
else:
    url_params["id"] = ",".join(search_result.ids)
```

### Fix 2: Improve Error Handling

**Change in `get_dataset_summaries()` (line 622-638):**
```python
# Parse eSummary response
try:
    json_data = json.loads(response_data)

    # Check for API errors
    if "error" in json_data:
        logger.error(f"NCBI eSummary API error: {json_data['error']}")
        return []

    result = json_data.get("result", {})

    summaries = []
    for uid in search_result.ids:
        if uid in result:
            summary = result[uid]
            summary["uid"] = uid
            summaries.append(summary)

    if not summaries:
        logger.warning(f"eSummary returned no summaries for {len(search_result.ids)} UIDs")

    return summaries

except (json.JSONDecodeError, KeyError) as e:
    logger.error(f"Error parsing eSummary response: {e}")
    return []
```

### Fix 3: Remove Buggy Fallback Logic

**Change in `format_geo_search_results()` (line 1070-1077):**
```python
if not search_result.summaries:
    # ERROR: Cannot format results without summaries
    # UIDs alone are insufficient - they are internal IDs, not accessions
    return (
        f"## ⚠️ GEO Search Error\n\n"
        f"Query: {search_result.query or original_query}\n"
        f"Found {search_result.count:,} results, but could not retrieve summaries.\n\n"
        f"This is likely a temporary issue. Please try again or contact support.\n"
        f"Query details: {len(search_result.ids)} UIDs returned but summary fetch failed."
    )
```

**Rationale**:
- Be explicit about failure instead of showing invalid data
- Invalid accessions break downstream workflows (worse than clear error)
- Guides user to retry or report issue

## Testing Plan

### 1. Manual Test Script
```bash
python tests/manual/debug_gds_search.py
```
Expected: Should see valid GSE accessions in output

### 2. Integration Test
```python
def test_fast_dataset_search_returns_gse_accessions():
    """Test that dataset search returns valid GSE accessions, not invalid GDS prefixes."""
    # Test query from user report
    query = '"10x Genomics" single cell RNA-seq'

    # Run search
    result = research_agent.fast_dataset_search(query, data_type="geo", max_results=5)

    # Verify accessions
    assert "GSE" in result, "Should contain GSE accessions"
    assert "GDS200" not in result, "Should not contain invalid GDS prefixed UIDs"

    # Verify URLs
    assert "GDSbrowser?acc=GDS200" not in result, "Should not have invalid GDS URLs"
    assert "geo/query/acc.cgi?acc=GSE" in result, "Should have valid GSE URLs"
```

### 3. User Acceptance Test
Reproduce exact query from user report:
```
lobster query 'Search for 10x Genomics single cell RNA-seq datasets in GEO'
```

Expected output:
```
1. GSE278021 - Distinct genomic identities and transformation trajectories of thymomas
2. GSE275038 - A specific subset of Mesenchymal Stromal Cells...
...
```

## Impact Assessment

### Severity: HIGH
- Breaks core dataset discovery functionality
- Invalid accessions break all downstream tools (validate_dataset_metadata, download, etc.)
- Affects all users searching GEO database via fast_dataset_search

### Affected Code Paths:
- `research_agent.fast_dataset_search()` ✗ BROKEN
- `find_related_entries()` → likely same issue
- Any tool using `GEOProvider.search_publications()`

### User Impact:
- Cannot discover GSE datasets via keyword search
- Must manually look up accessions on NCBI website
- Workflow broken for publication-to-dataset discovery

## Recommended Action

1. **Immediate**: Implement Fix 1 (Option 1) + Fix 2 + Fix 3
2. **Testing**: Run manual debug script + integration test
3. **Validation**: User acceptance test with exact query from report
4. **Documentation**: Update wiki if GEO search behavior changes

## Files to Modify

1. `lobster/tools/providers/geo_provider.py`
   - `get_dataset_summaries()` (lines 583-638)
   - `format_geo_search_results()` (lines 1061-1117)

2. `tests/integration/test_geo_provider.py` (add new test)

3. `tests/manual/debug_gds_search.py` (keep for debugging)

---

**Report Generated**: 2025-12-07
**Severity**: HIGH
**Status**: Root cause identified, fix proposed
**Estimated Fix Time**: 30-60 minutes
