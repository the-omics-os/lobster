# Handoff Summary: Metadata Filtering Issue - 0 Samples Found

## Problem Description

The metadata_assistant agent processed 302 publication queue entries but found **0 samples** matching the filter criteria "16S human fecal". The agent successfully processed all entries (no failures), but no samples were extracted before filtering was even applied.

**Key Log Evidence:**
```
→ process_metadata_queue
  Input: {'output_key': 'aggregated_16s_human_fecal_samples',
          'status_filter': 'completed',  ← INCORRECT
          'filter_criteria': '16S human fecal'}

Result: **Entries Processed**: 302
        **Successful**: 302
        **Failed**: 0
        **Entries With Samples**: 0  ← CRITICAL
        **Samples Extracted**: 0     ← CRITICAL
        **Samples Valid**: 0
        **Samples After Filter**: 0
```

## Root Cause Analysis

### Issue 1: Wrong status_filter Parameter ⚠️

**What happened:**
- Agent passed `status_filter='completed'`
- Should have used `status_filter='handoff_ready'` (the default)

**Why this matters:**
- `status_filter` maps to `PublicationStatus` enum (line 1946 in metadata_assistant.py)
- `PublicationStatus.COMPLETED` means the publication has been fully processed
- `PublicationStatus.HANDOFF_READY` means the publication has identifiers + metadata ready for handoff to metadata_assistant
- When filtering for `status='completed'`, the agent likely found entries that research_agent already marked as "done" but which may not have the required `workspace_metadata_keys` field populated

**Code Location:**
```python
# lobster/agents/metadata_assistant.py:1945-1947
entries = queue.list_entries(
    status=PublicationStatus(status_filter.lower())
)
```

### Issue 2: Missing workspace_metadata_keys (Hypothesis)

**Expected workflow:**
1. research_agent extracts identifiers from publications
2. research_agent fetches SRA metadata and saves to `workspace/metadata/sra_PRJNA*_samples.json`
3. research_agent populates `workspace_metadata_keys` field with list of basenames (e.g., `["sra_PRJNA123_samples"]`)
4. research_agent sets `status=HANDOFF_READY`
5. metadata_assistant reads `workspace_metadata_keys`, loads sample metadata, applies filters

**What likely happened:**
- Agent queried entries with `status='completed'` instead of `status='handoff_ready'`
- These entries may have empty `workspace_metadata_keys` (research_agent already processed them)
- Result: "Entries With Samples: 0" because no metadata files to load

## Files Involved

### 1. Agent Implementation
**File:** `lobster/agents/metadata_assistant.py`
- **Line 1871-1937:** `process_metadata_queue()` tool definition
- **Line 1945-1947:** Entry filtering by status (THIS IS WHERE WRONG STATUS IS USED)
- **Line 1972-2150:** Sequential processing loop
- **Line 2006-2089:** Per-entry sample loading and validation logic
- **Line 2090-2100:** Filter application (only reached if samples extracted)

### 2. Filtering Service
**File:** `lobster/services/metadata/metadata_filtering_service.py`
- **Line 139-204:** `parse_criteria()` method - parses "16S human fecal" into structured format
- **Line 206-273:** `apply_filters()` method - applies parsed filters to samples
- **Line 304-340:** `_apply_sequencing_filter()` - validates 16S via `microbiome_service.validate_16s_amplicon()`
- **Line 342-361:** `_apply_host_filter()` - validates host via `microbiome_service.validate_host_organism()`

**CRITICAL FINDING:** Sample type filter ("fecal") is **NOT implemented**!
- `parse_criteria()` extracts `sample_types: ["fecal"]` (line 179-191)
- But `apply_filters()` never calls `_apply_sample_type_filter()` (method doesn't exist)
- Only sequencing (16S/shotgun) and host filters are applied

### 3. Publication Queue Schema
**File:** `lobster/core/schemas/publication_queue.py`
- **Line 16-27:** `PublicationStatus` enum definition
  - `HANDOFF_READY = "handoff_ready"` ← CORRECT status for metadata_assistant
  - `COMPLETED = "completed"` ← Status after full processing
- **Line 73-83:** `HandoffStatus` enum (different from PublicationStatus)
- **Line 100-102:** Documentation of workspace_metadata_keys field

### 4. Queue Management
**File:** `lobster/core/publication_queue.py`
- `list_entries(status=PublicationStatus)` method - filters by status field

## Filter Criteria Parsing

**Input:** `filter_criteria="16S human fecal"`

**Parsed to:**
```python
{
    "check_16s": True,          # "16s" in criteria
    "check_shotgun": False,
    "host_organisms": ["Human"], # "human" in criteria
    "sample_types": ["fecal"],   # "fecal" in criteria (NOT USED!)
    "standardize_disease": True  # Always enabled
}
```

**Filters Applied (in order):**
1. ✅ Disease extraction + standardization
2. ✅ Sequencing method (16S) via `microbiome_service.validate_16s_amplicon()`
3. ✅ Host organism (Human) via `microbiome_service.validate_host_organism()`
4. ❌ Sample type (fecal) - **NOT IMPLEMENTED**

## Next Steps for Investigation

### Step 1: Verify status_filter Issue (PRIORITY 1)

**Action:** Check publication queue entries to see status distribution

**Commands to run:**
```python
# Via execute_custom_code or direct Python
from lobster.core.schemas.publication_queue import PublicationStatus

queue = data_manager.publication_queue

# Count by status
status_counts = {}
for entry in queue.list_all_entries():
    status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

result = {"status_distribution": status_counts}
```

**Expected insight:**
- If there are N entries with `status='handoff_ready'`, re-run with correct filter
- If there are 0 entries with `status='handoff_ready'`, investigate why research_agent didn't set this status

### Step 2: Inspect workspace_metadata_keys for Completed Entries

**Action:** Check if 'completed' entries have workspace_metadata_keys populated

**Commands:**
```python
queue = data_manager.publication_queue
completed_entries = queue.list_entries(status=PublicationStatus("completed"))

# Check first 5 entries
sample_entries = []
for entry in completed_entries[:5]:
    sample_entries.append({
        "entry_id": entry.entry_id,
        "title": entry.title[:50],
        "has_workspace_keys": bool(entry.workspace_metadata_keys),
        "workspace_keys_count": len(entry.workspace_metadata_keys or []),
        "keys_preview": (entry.workspace_metadata_keys or [])[:3]
    })

result = {"completed_entries_analysis": sample_entries}
```

**Expected insight:**
- If `workspace_metadata_keys` is empty for 'completed' entries → explains 0 samples
- If `workspace_metadata_keys` is populated → deeper issue with file loading

### Step 3: Verify handoff_ready Entries Exist and Have Metadata

**Action:** List handoff_ready entries and check their workspace_metadata_keys

**Commands:**
```python
handoff_ready = queue.list_entries(status=PublicationStatus("handoff_ready"))

if not handoff_ready:
    result = {"error": "No handoff_ready entries found",
              "recommendation": "Run research_agent to process publications first"}
else:
    sample_entries = []
    for entry in handoff_ready[:5]:
        sample_entries.append({
            "entry_id": entry.entry_id,
            "title": entry.title[:50],
            "workspace_keys": entry.workspace_metadata_keys,
            "has_identifiers": bool(entry.extracted_identifiers)
        })

    result = {
        "handoff_ready_count": len(handoff_ready),
        "sample_entries": sample_entries
    }
```

### Step 4: Re-run with Correct status_filter

**Action:** Run process_metadata_queue with default status_filter

**Command:**
```python
# Via metadata_assistant tool
process_metadata_queue(
    status_filter="handoff_ready",  # ← CORRECT (default)
    filter_criteria="16S human fecal",
    output_key="aggregated_16s_human_fecal_samples"
)
```

### Step 5: If Still 0 Samples - Investigate Filter Matching

**Action:** Check if metadata fields match filter expectations

**Sample investigation:**
```python
# Load one workspace metadata file manually
import json
workspace_path = Path(data_manager.workspace_path) / "workspace" / "metadata"
sample_files = list(workspace_path.glob("sra_*_samples.json"))

if sample_files:
    with open(sample_files[0]) as f:
        data = json.load(f)

    # Inspect first sample
    if data.get("samples"):
        first_sample = data["samples"][0]
        result = {
            "file": sample_files[0].name,
            "sample_count": len(data["samples"]),
            "first_sample_fields": list(first_sample.keys()),
            "library_strategy": first_sample.get("library_strategy"),
            "organism": first_sample.get("organism"),
            "host_scientific_name": first_sample.get("host_scientific_name"),
            "isolation_source": first_sample.get("isolation_source"),
            "sample_type": first_sample.get("sample_type")
        }
```

**Expected insight:**
- Check if `library_strategy` contains "AMPLICON" (for 16S filter)
- Check if `organism` or `host_scientific_name` contains "Homo sapiens" (for human filter)
- Check if any field contains "fecal" (for sample type - though filter not implemented!)

## Secondary Issue: Sample Type Filter Not Implemented

**Impact:** Medium (filter criteria "fecal" is ignored)

**Evidence:**
- `parse_criteria()` extracts `sample_types` from filter string
- `apply_filters()` never uses `sample_types` (no `_apply_sample_type_filter()` method)
- Only sequencing and host filters are applied

**Recommendation:**
- Implement `_apply_sample_type_filter()` in `metadata_filtering_service.py`
- Or document that sample type filtering is not supported yet
- For now, users can post-filter results manually or use execute_custom_code

## Recommended Quick Fix

**For immediate unblocking:**

1. **Re-run with correct status:**
```python
process_metadata_queue(
    status_filter="handoff_ready",  # Use default
    filter_criteria="16S human",    # Remove "fecal" (not supported)
    output_key="aggregated_16s_human_samples"
)
```

2. **If no handoff_ready entries exist:**
   - Check research_agent workflow
   - Verify research_agent successfully extracted identifiers and fetched metadata
   - Ensure research_agent set `status=HANDOFF_READY` after metadata fetch

3. **Manual post-filtering for "fecal":**
```python
# After aggregation, filter for fecal samples
execute_custom_code(
    python_code='''
samples = metadata_store["aggregated_16s_human_samples"]["samples"]

fecal_samples = [
    s for s in samples
    if any(term in str(s.get("isolation_source", "")).lower()
           for term in ["fecal", "stool", "feces"])
]

result = {"fecal_sample_count": len(fecal_samples)}
metadata_store["aggregated_16s_human_fecal_samples"] = {
    "samples": fecal_samples,
    "filter_criteria": "16S human fecal (manual post-filter)"
}
'''
)
```

## Key Takeaways

1. **Primary Issue:** Wrong `status_filter='completed'` instead of `status_filter='handoff_ready'`
2. **Secondary Issue:** Sample type filter ("fecal") parsed but not applied
3. **Verification Needed:** Check if handoff_ready entries exist with populated workspace_metadata_keys
4. **Quick Win:** Re-run with correct default status_filter

## Agents to Involve

- **metadata_assistant:** Re-run tool with correct parameters
- **research_agent:** Verify publication queue status transitions
- **Custom code execution:** Inspect queue entries and metadata files directly
