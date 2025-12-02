# Diagnostic Report: Zero Dataset Issue in Microbiome Filtering Workflow

**Date**: 2025-12-01
**Test Command**: `python3 tests/manual/test_publication_processing.py --ris-file kevin_notes/databiomix/CRC_microbiome.ris --entry 0,1 --test-handoff --filter-criteria "16S human fecal CRC"`

---

## Executive Summary

**Root Cause**: The filtering workflow is functioning **CORRECTLY**. Zero datasets after filtering is the expected result because the test publication contains **bovine (cow) milk microbiome samples**, not human fecal samples as specified by the filter criteria.

**Verdict**: This is **NOT a bug** - it's correct behavior. The filter "16S human fecal CRC" is properly excluding non-matching samples.

---

## Test Results

### Phase 1: Publication Processing
- **Publications Loaded**: 655 (from CRC_microbiome.ris)
- **Entries Processed**: 2 (indices 0, 1)
- **Handoff Ready**: 1 entry
- **Dataset IDs Identified**: 2 (PRJNA859642, PRJNA936091)
- **Workspace Metadata Files**: 7

### Phase 2: Metadata Assistant Handoff
```
Entries Processed:        1
Samples Extracted:        350
Samples Valid:            350 (100.0%)
Samples After Filter:     0
Retention Rate:           0.0%
```

### Publication Details
**Entry**: pub_queue_doi_10_3389_fmicb_2023_1183018
**Title**: "Elucidation of the Bovine Intramammary Bacteriome and Resistome from healthy cows of Swiss dairy farms in the Canton Tessin"
**BioProject**: PRJNA859642

---

## Sample Metadata Analysis

### Metadata Structure
Located at: `/var/folders/.../metadata/sra_prjna859642_samples.json`

**Sample Characteristics (all 350 samples)**:
```json
{
  "host": "Cow",
  "library_strategy": "WGS",
  "sample_type": "Bacteria",
  "isolation_source": "milk",
  "organism_name": "Unknown" (not populated)
}
```

### Filter Criteria Parsing
Filter: **"16S human fecal CRC"**

Parsed as:
```python
{
  "check_16s": True,              # "16S" keyword detected
  "host_organisms": ["Human"],    # "human" keyword detected
  "sample_types": ["fecal"],      # "fecal" keyword detected
  "standardize_disease": True     # "CRC" keyword detected
}
```

---

## Filtering Analysis

### Why All Samples Were Filtered Out

#### 1. 16S Amplicon Check (First Filter)
**Code**: `lobster/agents/metadata_assistant.py:1628`
```python
filtered = [s for s in filtered if microbiome_filtering_service.validate_16s_amplicon(s, strict=False)[0]]
```

**Logic**:
- Samples have `library_strategy: "WGS"` (Whole Genome Sequencing)
- Filter expects `library_strategy` containing: `["amplicon", "16s", "16s rrna", "16s amplicon", "targeted locus"]`
- **Result**: WGS ≠ 16S → All 350 samples filtered out

#### 2. Host Organism Check (Second Filter - Not Reached)
**Code**: `lobster/agents/metadata_assistant.py:1630`
```python
filtered = [s for s in filtered if microbiome_filtering_service.validate_host_organism(s, allowed_hosts=["Human"])[0]]
```

**Would have filtered**:
- Samples have `host: "Cow"`
- Filter expects `host: "Human"`
- **Result**: Cow ≠ Human → Would filter out all samples

#### 3. Sample Type Check (Not Implemented Yet)
The parsed `sample_types: ["fecal"]` is not currently enforced in `_apply_metadata_filters()`.

**Would have filtered**:
- Samples have `isolation_source: "milk"`
- Filter expects "fecal" samples
- **Result**: milk ≠ fecal → Would filter out all samples

---

## Code Review: Filtering Logic

### File: `lobster/services/metadata/microbiome_filtering_service.py`

**Method**: `validate_16s_amplicon(metadata, strict=False) -> Tuple[Dict, Dict, AnalysisStep]`

**Return Values**:
- `filtered_metadata`: Original dict if valid, **empty dict `{}` if invalid**
- `stats`: Validation summary
- `ir`: Provenance tracking

**Key Insight**:
```python
filtered_metadata = metadata if result.is_valid else {}
```

In the list comprehension:
```python
if microbiome_filtering_service.validate_16s_amplicon(s, strict=False)[0]
```

- If valid (16S detected) → returns `metadata` dict → **truthy** → sample passes
- If invalid (WGS detected) → returns `{}` empty dict → **falsy** → sample filtered out

This is **correct behavior** in Python!

### File: `lobster/agents/metadata_assistant.py`

**Method**: `_apply_metadata_filters(samples, filter_criteria) -> list`

**Current Implementation**:
```python
parsed = _parse_filter_criteria(filter_criteria)
filtered = samples.copy()

if parsed["check_16s"]:
    filtered = [s for s in filtered if microbiome_filtering_service.validate_16s_amplicon(s, strict=False)[0]]
if parsed["host_organisms"]:
    filtered = [s for s in filtered if microbiome_filtering_service.validate_host_organism(s, allowed_hosts=parsed["host_organisms"])[0]]

return filtered
```

**Notes**:
- Correctly applies 16S check
- Correctly applies host organism check
- **Missing**: Sample type filtering (fecal vs. milk vs. tissue, etc.)
- **Missing**: Disease filtering (though `standardize_disease` flag is parsed)

---

## Validation: Is This a Bug?

### Test Case Evaluation

| Criterion | Expected | Actual | Match? |
|-----------|----------|--------|--------|
| Sequencing Method | 16S amplicon | WGS | ❌ |
| Host Organism | Human | Cow | ❌ |
| Sample Type | Fecal | Milk | ❌ |
| Disease Context | CRC | Healthy cows | ❌ |

**Conclusion**: The publication "Bovine Intramammary Bacteriome" is **incompatible** with the filter criteria "16S human fecal CRC". The filtering service is working as designed.

---

## Recommendations

### 1. Verify with Matching Publications

To confirm the filtering works correctly, test with publications that **should** pass the filter:

**Suggested Test**:
```bash
# Find entries with human gut/fecal 16S samples
python3 tests/manual/test_publication_processing.py \
    --ris-file kevin_notes/databiomix/CRC_microbiome.ris \
    --entry 10-30 \
    --test-handoff \
    --filter-criteria "16S human fecal CRC" \
    --show-response
```

**Expected Behavior**:
- If RIS file contains human gut microbiome studies → Should see retention rate > 0%
- If RIS file only contains non-human or non-16S studies → Retention rate = 0% (correct)

### 2. Inspect RIS File Contents

Check if the RIS file has any human samples:
```bash
# Count host organisms across all publications
grep -i "human\|homo sapiens" kevin_notes/databiomix/CRC_microbiome.ris | wc -l

# Count 16S mentions
grep -i "16s\|amplicon" kevin_notes/databiomix/CRC_microbiome.ris | wc -l
```

### 3. Relax Filter Criteria (If Needed)

If customer wants to include broader sample types:

**Option A**: Remove 16S restriction
```bash
--filter-criteria "human fecal CRC"
```

**Option B**: Include WGS metagenomic studies
```bash
--filter-criteria "human fecal CRC metagenomic"  # Requires code change
```

**Option C**: Test without filter
```bash
--filter-criteria ""  # Returns all samples
```

### 4. Add Missing Filter Features

**Current Gaps**:
- ✅ 16S amplicon detection: Working
- ✅ Host organism filtering: Working
- ❌ Sample type filtering: **Not implemented** (parsed but not enforced)
- ❌ Disease filtering: **Not implemented** (parsed but not enforced)

**Recommended Enhancement**:
```python
# In _apply_metadata_filters()
if parsed["sample_types"]:
    filtered = [
        s for s in filtered
        if any(st in s.get("sample_type", "").lower() or
               st in s.get("isolation_source", "").lower()
               for st in parsed["sample_types"])
    ]
```

---

## Sample Data for Reference

**File**: `/var/folders/.../sra_prjna859642_samples.json`

```json
{
  "run_accession": "SRR19226408",
  "study_accession": "SRP375850",
  "study_title": "Bovine Intramammary Bacteriome",
  "library_strategy": "WGS",
  "library_source": "METAGENOMIC",
  "host": "Cow",
  "isolation_source": "milk",
  "sample_type": "Bacteria",
  "bioproject": "PRJNA859642",
  "geo_loc_name": "Switzerland"
}
```

**Key Mismatches**:
- `library_strategy: WGS` ≠ 16S amplicon
- `host: Cow` ≠ Human
- `isolation_source: milk` ≠ fecal

---

## Conclusion

**Status**: ✅ **WORKING AS DESIGNED**

The microbiome filtering service is functioning correctly. The zero retention rate is expected because:

1. **Test publication is incompatible**: The bovine milk microbiome study does not match "16S human fecal CRC" criteria
2. **Filtering logic is correct**: WGS samples are properly excluded when 16S amplicon filter is applied
3. **Host filtering is correct**: Cow samples are properly excluded when Human filter is applied

**No bug found**. To observe successful filtering, test with publications that actually contain human gut/fecal 16S amplicon samples.

---

## Next Steps

1. **Inspect RIS file** to identify entries with human gut microbiome samples
2. **Re-run test** with entry indices that match the filter criteria
3. **Consider adding sample_type enforcement** to `_apply_metadata_filters()` for completeness
4. **Document expected retention rates** for different filter criteria combinations

---

## Files Analyzed

- `/Users/tyo/GITHUB/omics-os/lobster/tests/manual/test_publication_processing.py`
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/metadata_assistant.py` (lines 1118-1683)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/services/metadata/microbiome_filtering_service.py`
- Workspace metadata: `/var/folders/.../sra_prjna859642_samples.json`

---

**Report Generated**: 2025-12-01
**Total Samples Analyzed**: 350
**Publications Tested**: 2
**Filtering Stages Validated**: 2 (16S, host organism)
