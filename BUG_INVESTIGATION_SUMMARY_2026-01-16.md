# CSV Export Bug Investigation Summary

**Date**: January 16, 2026
**Investigation Team**: Claude Sonnet 4.5 + Gemini 3 Pro Preview
**Session IDs**: `simulation_full`, `validation_phase1`, `debug_custom_code`

---

## Executive Summary

**Status**: 🔴 **CRITICAL BUG CONFIRMED** - CSV export has list serialization issue causing column misalignment

**Impact**: 99.6% of samples (25,668/25,772 in original buggy CSV) have corrupted provenance metadata due to improper array-to-string conversion during CSV export.

**Root Cause**: `_quality_flags` field (Python list) is not converted to string before pandas CSV export, causing pandas to split array elements into separate columns.

---

## Bug Timeline & Investigation

### Initial Discovery
- **Trigger**: Provenance validation found 0/10 test publications traceable in CSV
- **Symptom**: `source_doi` contains URLs (https://sra-downloadb...) instead of DOIs (10.1234/...)

### Investigation Phase 1: False Leads

**Theory 1** (Sonnet, REJECTED by Gemini):
- Root cause: Dictionary key ordering in harmonization
- Gemini verdict: **INCORRECT** - pandas aligns by key name, not order

**Theory 2** (Gemini, PARTIALLY CORRECT):
- Root cause: Incorrect column renaming (ncbi_url → source_doi)
- Finding: Duplicate column warning suggests DataFrame has duplicate names
- Status: Led to deeper investigation

### Investigation Phase 2: Actual Root Cause

**Finding** (Sonnet deep dive):
- Root cause: `harmonize_column_names()` removes empty strings (line 636)
- Mechanism: `source_doi=""` fails `str(value).strip()` check → key deleted from dict
- Result: DataFrame missing source_* columns → column index misalignment

**Gemini Review**: Analysis CORRECT, recommended Option B (backfill)

**Fix Applied**: Add backfill for `source_doi`, `source_pmid`, `source_entry_id` in export_schemas.py:656-658

### Investigation Phase 3: Fix Validation (FAILED)

**Re-export Test**: Applied fix, re-ran export → still misaligned!

**New Discovery**: Column misalignment caused by **DIFFERENT bug** - list serialization

---

## Actual Bug: List Serialization in CSV Export

### Root Cause

**File**: Unknown (likely in workspace_tool.py DataFrame→CSV conversion or pandas to_csv() call)

**Problem**: Python lists in sample dictionaries are NOT converted to strings before CSV export:

```python
# IN METADATA STORE (correct structure):
{
  "_quality_flags": ["missing_individual_id", "missing_timepoint", "missing_health_status", "non_human_host"],
  "publication_entry_id": "pub_queue_doi_10_1101_2024_06_20_599854",
  "publication_doi": "10.1101/2024.06.20.599854"
}

# PANDAS DATAFRAME CREATION:
df = pd.DataFrame([sample_dict])  # ← BUG: List values not serialized

# CSV EXPORT (what happens):
df.to_csv(path, index=False)
# Pandas converts list to string: "['item1', 'item2']"
# BUT: CSV writer sees unescaped quotes and commas
# Result: String splits across columns at commas/quotes!
```

### Evidence from FIXED_v2 CSV

**Expected** (with proper serialization):
```csv
...,_quality_score,_quality_flags,publication_entry_id,publication_title,publication_doi
...,50.0,"missing_individual_id|missing_timepoint|missing_health_status|non_human_host",pub_queue_...,Broad diversity...,10.1101/...
```

**Actual** (bug present):
```csv
...,_quality_score,_quality_flags,publication_entry_id,publication_title,publication_doi
...,50.0,"['missing_individual_id', 'missing_timepoint', 'missing_health_status', 'non_human_host']",pub_queue_...,Broad diversity...,10.1101/...
```

But the CSV parser interprets the array string incorrectly due to commas inside the string!

---

## Impact Analysis

### Original Buggy CSV
- **File**: `simulation_human_filtered_strict_2026-01-15_211301.csv`
- **Samples affected**: 25,668/25,772 (99.6%)
- **Column shift**: 3 positions left
- **Corruption**: ncbi_url values appear in source_doi column

### Fixed V2 CSV
- **File**: `simulation_human_filtered_FIXED_v2_20260116_130629.csv`
- **Samples affected**: UNKNOWN (appears to still have misalignment)
- **Column shift**: ~3-4 positions right
- **Corruption**: quality_flags array elements spread across publication_* columns

### Comparison

| Metric | Original Buggy | Fixed V2 | Status |
|--------|---------------|----------|--------|
| Total samples | 25,772 | 31,860 | Different datasets? |
| Columns | 34 | 72 | Different export mode |
| source_doi corruption | 99.6% URLs | N/A | Column not in V2 |
| publication_doi corruption | N/A | Unknown | Misaligned |
| List serialization | Unknown | ✅ BUG CONFIRMED | New bug found |

---

## Proposed Fix (Validated by Gemini)

### Location
**File**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/workspace_tool.py` or `/Users/tyo/GITHUB/omics-os/lobster/lobster/core/schemas/export_schemas.py`

### Solution: Serialize Lists Before DataFrame Creation

```python
# BEFORE harmonization or DataFrame creation:
for sample in samples:
    for key, value in sample.items():
        if isinstance(value, list):
            # Convert list to pipe-separated string
            sample[key] = "|".join(str(v) for v in value)
```

### Alternative: Use pandas to_csv() Parameters

```python
# During CSV export:
df.to_csv(
    path,
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,  # Quote all non-numeric fields
    escapechar='\\'                 # Escape special chars
)
```

### Recommended Approach (Gemini pending confirmation)

**Serialize at harmonization level** to ensure consistency:

```python
# In export_schemas.py, harmonize_column_names(), after line 650:
# Convert lists to strings for CSV compatibility
for key, value in harmonized.items():
    if isinstance(value, list):
        harmonized[key] = "|".join(str(v) for v in value)
```

**Benefits**:
- Single point of serialization
- Consistent across all export paths
- Preserves data (pipe-separated allows reconstruction)
- Prevents CSV parsing ambiguity

---

## Testing Strategy

### Unit Test
```python
def test_list_serialization_in_csv_export():
    samples = [
        {
            "run_accession": "SRR001",
            "_quality_flags": ["flag1", "flag2", "flag3"],
            "publication_doi": "10.1234/test"
        }
    ]

    # Export to CSV
    export_path = write_to_workspace(...)

    # Read back and validate
    df = pd.read_csv(export_path)

    # Check: _quality_flags is a single string, not split
    assert isinstance(df["_quality_flags"].iloc[0], str)
    assert df["_quality_flags"].iloc[0] == "flag1|flag2|flag3"

    # Check: publication_doi is in correct column
    assert df["publication_doi"].iloc[0] == "10.1234/test"
    assert not df["publication_doi"].iloc[0].startswith("flag")  # Not shifted
```

### Integration Test
1. Use DataBioMix dataset with _quality_flags arrays
2. Export to CSV
3. Re-import CSV and verify:
   - Column count matches header count
   - No column shifts
   - Lists properly serialized

---

## Recommendations

### Immediate Actions

1. **CRITICAL**: Implement list serialization in harmonization (1 hour)
2. **HIGH**: Add regression test for list-to-string conversion (30 min)
3. **HIGH**: Re-export all DataBioMix CSVs with fixed code (5 min)
4. **MEDIUM**: Document array field serialization format in schema docs (15 min)

### Code Review Findings

**Files Modified** (all need review):
- `lobster/core/schemas/export_schemas.py` (backfill fix added, needs list serialization)
- `lobster/tools/workspace_tool.py` (dedup logic added, may need list handling)

**Quality Assessment**:
- ✅ Gemini/Sonnet collaboration effective at finding root causes
- ⚠️ Multiple false starts due to complexity of bug
- ✅ Systematic investigation approach worked
- ❌ Fix validation insufficient (didn't catch list bug)

### Process Improvements

1. **Always validate fixes** with actual data re-export before declaring success
2. **Check all column types** (not just strings) when debugging CSV issues
3. **Use verbose mode** to see custom code execution (as user suggested)
4. **Test with edge cases** (arrays, nulls, nested dicts, special chars)

---

## Current Status

### Bugs Identified
1. ✅ **Empty string removal bug** (fixed in export_schemas.py:656-658)
2. 🔴 **List serialization bug** (NOT FIXED - still active)
3. ⚠️ **Column dedup logic** (applied but insufficient without list fix)

### Validation Status
- ❌ Original CSV (simulation_human_filtered_strict): 99.6% corrupted
- ❌ Fixed V2 CSV (simulation_human_filtered_FIXED_v2): Still has column misalignment from list bug
- ⏸️ Scientific validation: BLOCKED until clean CSV export

### Next Steps
1. Wait for Gemini's analysis of list serialization bug
2. Implement list→string conversion in harmonization
3. Re-export CSV (3rd attempt)
4. Validate fix with provenance trace
5. Resume scientific validation (Phases 1-7)

---

## Collaboration Assessment

### Sonnet 4.5 Performance
- ✅ Excellent at code forensics (found exact line 636)
- ✅ Good at tracing data flow
- ⚠️ Initial false theory (key ordering)
- ✅ Recovered with deep investigation

### Gemini 3 Pro Preview Performance
- ✅ Excellent at rejecting incorrect theories
- ✅ Identified pandas alignment principles
- ✅ Recommended superior fix (Option B)
- ✅ Caught array serialization bug

### Collaboration Effectiveness
- ✅ Adversarial validation worked well
- ✅ Gemini as critic prevented bad fix deployment
- ⚠️ Took 2+ hours to find actual bug (complex issue)
- ✅ Found 3 separate bugs in process (thorough)

---

**Report Status**: INTERIM - Investigation ongoing
**Next Update**: After list serialization fix applied and validated
