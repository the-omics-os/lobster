# Implementation Plan: Keyword Scanning Integration into extract_disease_with_fallback

**Date**: 2026-01-10
**Author**: ultrathink (Claude Opus 4.5)
**Reviewers**: Gemini 2.0 Flash (architecture), 4x Sonnet 4.5 sub-agents (analysis)
**Target Release**: v0.5.0
**Estimated Effort**: 3-4 hours
**Priority**: HIGH

---

## 1. OBJECTIVE

Integrate deterministic keyword scanning (Phase 1 enrichment) into `extract_disease_with_fallback()` to automatically improve disease coverage from ~20-30% to ~40-60% **without LLM costs**.

**Success Criteria**:
- ✅ Keyword scanning runs automatically after column-name fallback
- ✅ Zero API costs (purely deterministic)
- ✅ Coverage improvement validated on DataBioMix dataset (655 samples)
- ✅ Backward compatible (no breaking changes)
- ✅ All tests pass (existing + new)

---

## 2. BACKGROUND & CONTEXT

### Problem Statement

**Current behavior** (`extract_disease_with_fallback`):
- Checks 10 specific column NAMES in priority order
- Misses disease data in non-standard columns like:
  - `"subject_phenotype": "colorectal cancer patient"`
  - `"notes": "UC diagnosis confirmed"`
  - `"cohort_description": "Crohn's disease cohort"`

**Result**: Only 20-30% initial coverage → triggers validation failure → requires manual `enrich_samples_with_disease` call

**Desired behavior**:
- Stage 1: Check standard column names (existing, fast)
- **Stage 2**: Scan ALL column VALUES for disease keywords (new, free)
- Result: 40-60% coverage → passes validation automatically

### Architecture Context

**Lobster Service Pattern**:
- Services are stateless
- Return 3-tuple: `(result, stats, ir)`
- W3C-PROV compliant provenance tracking

**This change**:
- Pure function enhancement (no state changes)
- Maintains service pattern
- Updates provenance tracking (`disease_source` field)

### Related Work

**Recent fixes** (2026-01-09):
1. ✅ Fixed generic "control"/"normal" false positives in `DiseaseStandardizationService`
2. ✅ Implemented Gemini's allow-list architecture for exclusion patterns
3. ✅ Split luminal/mucosal gut samples in `MicrobiomeFilteringService`

**This enhancement builds on**:
- Exclusion pattern fixes (prevents "control" keyword from matching negatives)
- Existing fallback chain architecture

---

## 3. TECHNICAL SPECIFICATION

### 3.1 Keyword Dictionary Definition

**Location**: Add to `lobster/services/metadata/metadata_filtering_service.py` after line 30

```python
# Disease keyword mappings for column value scanning
# Used as fallback when standard column names don't contain disease data
# NOTE: Generic "control" removed to prevent false positives (Issue #1 fix)
DISEASE_KEYWORDS = {
    'crc': [
        'colorectal',
        'colon_cancer',
        'colon cancer',
        'rectal_cancer',
        'rectal cancer',
        'crc',
        'colorectal carcinoma',
        'colon carcinoma',
    ],
    'uc': [
        'ulcerative',
        'ulcerative colitis',
        'ulcerative_colitis',
        'uc_',
        'colitis_ulcerosa',
        'colitis ulcerosa',
    ],
    'cd': [
        'crohn',
        'crohns',
        'crohn\'s',
        'crohns_disease',
        'crohn disease',
        'crohns disease',
        'cd_',
    ],
    'healthy': [
        'healthy',
        'non_ibd',
        'non-ibd',
        'non_diseased',
        'non-diseased',
        'disease-free',
        'disease free',
        # NOTE: 'control' and 'normal' NOT included (too generic, per Issue #1)
        # Use compound terms: 'healthy control', 'normal control' in main mappings
    ],
}
```

**Design Decisions**:
1. **No generic "control"**: Prevents false positives (see DiseaseStandardizationService.DISEASE_EXCLUSION_CONFIG)
2. **Case-insensitive**: All keywords lowercase (matching uses `.lower()`)
3. **Underscore + space variants**: Handles "crohn_disease" vs "crohn disease"
4. **Abbreviation variants**: "uc_", "cd_" catch flag-style columns

### 3.2 Function Enhancement

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Function**: `extract_disease_with_fallback` (lines 49-96)

**Current implementation**:
```python
def extract_disease_with_fallback(
    df: pd.DataFrame, study_context: Optional[Dict] = None
) -> Optional[str]:
    """Extract disease column with fallback chain across multiple source fields."""

    # Initialize target column
    if "host_disease_stat" not in df.columns:
        df["host_disease_stat"] = None

    # Apply fallback chain: fill from first available source
    for source_col in DISEASE_SOURCE_COLUMNS:
        if source_col in df.columns and source_col != "host_disease_stat":
            mask = df["host_disease_stat"].isna() & df[source_col].notna()
            if mask.any():
                df.loc[mask, "host_disease_stat"] = df.loc[mask, source_col]
                logger.debug(f"Disease fallback: populated from '{source_col}'")

    # Check if we populated any disease data
    disease_count = df["host_disease_stat"].notna().sum()
    if disease_count > 0:
        logger.info(f"Disease extraction: {disease_count}/{len(df)} samples")
        return "host_disease_stat"

    logger.debug("Disease extraction: no disease fields found")
    return None
```

**Enhanced implementation** (insert after line 84, before final check):

```python
def extract_disease_with_fallback(
    df: pd.DataFrame, study_context: Optional[Dict] = None
) -> Optional[str]:
    """
    Extract disease column with 2-stage fallback chain:
    Stage 1: Check known column names (fast path)
    Stage 2: Scan ALL columns for disease keywords (slow path)

    Args:
        df: DataFrame with sample metadata (modified in-place)
        study_context: Optional publication metadata (unused but required by interface)

    Returns:
        Column name to use for disease standardization ('host_disease_stat')
        or None if no disease data found

    Example:
        # Stage 1 finds disease in standard column
        Sample 1: has "clinical condition" → copies to "host_disease_stat"

        # Stage 2 finds disease via keyword in non-standard column
        Sample 2: missing standard columns, but "phenotype": "crohns patient"
                  → keyword "crohn" matches → sets "host_disease_stat" = "cd"
    """
    # Initialize target column
    if "host_disease_stat" not in df.columns:
        df["host_disease_stat"] = None

    # STAGE 1: Column name fallback (EXISTING - lines 73-84)
    for source_col in DISEASE_SOURCE_COLUMNS:
        if source_col in df.columns and source_col != "host_disease_stat":
            mask = df["host_disease_stat"].isna() & df[source_col].notna()
            if mask.any():
                df.loc[mask, "host_disease_stat"] = df.loc[mask, source_col]
                logger.debug(
                    f"Stage 1: populated {mask.sum()} samples from '{source_col}'"
                )

    # STAGE 2: Keyword scanning fallback (NEW)
    remaining_missing = df["host_disease_stat"].isna().sum()

    if remaining_missing > 0:
        logger.debug(
            f"Stage 2: Scanning all columns for disease keywords "
            f"({remaining_missing}/{len(df)} samples still missing disease)"
        )

        enriched_count = 0

        # Iterate over samples still missing disease
        for idx, row in df[df["host_disease_stat"].isna()].iterrows():
            matched = False

            # Scan ALL columns in this sample
            for col_name, col_value in row.items():
                # Skip empty values and target column
                if pd.isna(col_value) or col_name == "host_disease_stat":
                    continue

                col_str = str(col_value).lower()

                # Check each disease keyword set
                for disease, keywords in DISEASE_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in col_str:
                            df.at[idx, 'host_disease_stat'] = disease
                            # Add provenance tracking
                            if 'disease_source' not in df.columns:
                                df['disease_source'] = None
                            df.at[idx, 'disease_source'] = f'keyword_scan:{col_name}'

                            enriched_count += 1
                            matched = True
                            logger.debug(
                                f"  - Sample {idx}: Found '{disease}' keyword '{keyword}' "
                                f"in column '{col_name}' = '{str(col_value)[:50]}...'"
                            )
                            break
                    if matched:
                        break
                if matched:
                    break

        if enriched_count > 0:
            logger.info(
                f"Stage 2 (keyword scan): enriched {enriched_count}/{remaining_missing} samples "
                f"({enriched_count/remaining_missing*100:.1f}%)"
            )
        else:
            logger.debug("Stage 2: No additional samples enriched via keyword scanning")

    # Check if we populated any disease data
    disease_count = df["host_disease_stat"].notna().sum()
    if disease_count > 0:
        coverage_pct = disease_count / len(df) * 100
        logger.info(
            f"Disease extraction complete: {disease_count}/{len(df)} samples "
            f"({coverage_pct:.1f}%) have disease metadata"
        )
        return "host_disease_stat"

    logger.debug("Disease extraction: no disease fields found in metadata")
    return None
```

**Key Implementation Details**:

1. **Preserves priority**: Stage 1 (column names) runs first, Stage 2 only for remaining gaps
2. **Provenance tracking**: `disease_source='keyword_scan:{col_name}'` tracks which column matched
3. **Logging levels**:
   - `DEBUG`: Per-sample matches
   - `INFO`: Summary statistics
4. **Performance**: O(samples × columns × keywords) worst-case, but short-circuits on first match

### 3.3 Import Requirements

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Line**: 23

**Add**:
```python
import pandas as pd  # Already exists
import logging       # Already exists

# NEW: Import for provenance tracking
from datetime import datetime  # For enrichment_timestamp (optional)
```

---

## 4. IMPLEMENTATION STEPS

### Step 1: Add DISEASE_KEYWORDS Constant

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Location**: After line 46 (after DISEASE_SOURCE_COLUMNS)

**Action**:
1. Copy keyword dictionary from Section 3.1
2. Add comment explaining removal of "control" (references Issue #1 fix)
3. Verify alignment with `DiseaseStandardizationService.DISEASE_MAPPINGS`

**Validation**:
- Keywords should match disease categories in `disease_standardization_service.py:66-88`
- No generic terms that could cause false positives

### Step 2: Enhance extract_disease_with_fallback Function

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Location**: Lines 49-96

**Action**:
1. Update docstring to document 2-stage fallback (see Section 3.2)
2. Keep Stage 1 logic unchanged (lines 73-84)
3. Insert Stage 2 keyword scanning after line 84, before final check at line 86
4. Add `disease_source` column creation (for provenance)
5. Update logging messages to distinguish Stage 1 vs Stage 2

**Code to insert** (after line 84):
```python
    # STAGE 2: Keyword scanning fallback (NEW)
    remaining_missing = df["host_disease_stat"].isna().sum()

    if remaining_missing > 0:
        logger.debug(
            f"Stage 2: Scanning all columns for disease keywords "
            f"({remaining_missing}/{len(df)} samples still missing disease)"
        )

        enriched_count = 0

        # Iterate over samples still missing disease
        for idx, row in df[df["host_disease_stat"].isna()].iterrows():
            matched = False

            # Scan ALL columns in this sample
            for col_name, col_value in row.items():
                # Skip empty values and target column
                if pd.isna(col_value) or col_name == "host_disease_stat":
                    continue

                col_str = str(col_value).lower()

                # Check each disease keyword set
                for disease, keywords in DISEASE_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in col_str:
                            df.at[idx, 'host_disease_stat'] = disease
                            # Add provenance tracking
                            if 'disease_source' not in df.columns:
                                df['disease_source'] = None
                            df.at[idx, 'disease_source'] = f'keyword_scan:{col_name}'

                            enriched_count += 1
                            matched = True
                            logger.debug(
                                f"  - Sample {idx}: Found '{disease}' keyword '{keyword}' "
                                f"in column '{col_name}' = '{str(col_value)[:50]}...'"
                            )
                            break
                    if matched:
                        break
                if matched:
                    break

        if enriched_count > 0:
            logger.info(
                f"Stage 2 (keyword scan): enriched {enriched_count}/{remaining_missing} samples "
                f"({enriched_count/remaining_missing*100:.1f}%)"
            )
        else:
            logger.debug("Stage 2: No additional samples enriched via keyword scanning")
```

**Validation**:
- Insert exactly after line 84 (after Stage 1 loop)
- Before line 86 (disease_count check)
- Maintain 4-space indentation

### Step 3: Update Logging Messages

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Location**: Line 83 (inside Stage 1 loop)

**Change**:
```python
# Before:
logger.debug(f"Disease fallback: populated {mask.sum()} samples from '{source_col}'")

# After:
logger.debug(f"Stage 1: populated {mask.sum()} samples from '{source_col}'")
```

**Location**: Lines 88-91 (final check)

**Change**:
```python
# Before:
logger.info(
    f"Disease extraction: {disease_count}/{len(df)} samples "
    f"({disease_count/len(df)*100:.1f}%) have disease metadata"
)

# After:
logger.info(
    f"Disease extraction complete: {disease_count}/{len(df)} samples "
    f"({disease_count/len(df)*100:.1f}%) have disease metadata"
)
```

### Step 4: Add Provenance Column to DataFrame Schema

**File**: `lobster/services/metadata/metadata_filtering_service.py`
**Location**: Line 73 (after initializing host_disease_stat)

**Add**:
```python
# Initialize target column
if "host_disease_stat" not in df.columns:
    df["host_disease_stat"] = None

# Initialize provenance tracking column (NEW)
if "disease_source" not in df.columns:
    df["disease_source"] = None
```

**Purpose**: Track which stage/method found disease data for transparency

---

## 5. TESTING REQUIREMENTS

### 5.1 Unit Tests

**File to create**: `tests/unit/services/metadata/test_metadata_filtering_service.py`
**Estimated**: ~150 lines

**Required test cases**:

```python
"""
Unit tests for MetadataFilteringService disease extraction.
"""
import pandas as pd
import pytest

from lobster.services.metadata.metadata_filtering_service import (
    extract_disease_with_fallback,
    DISEASE_KEYWORDS,
    DISEASE_SOURCE_COLUMNS,
)


class TestExtractDiseaseWithFallback:
    """Test 2-stage disease extraction."""

    def test_stage1_column_name_extraction(self):
        """Test that Stage 1 extracts from standard column names."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "clinical condition": ["colorectal cancer", "ulcerative colitis", "Crohn's disease"],
            "other_field": ["some", "random", "data"],
        })

        result = extract_disease_with_fallback(df)

        assert result == "host_disease_stat"
        assert df["host_disease_stat"].tolist() == [
            "colorectal cancer", "ulcerative colitis", "Crohn's disease"
        ]
        # Stage 1 should NOT set disease_source (only Stage 2 does)
        assert "disease_source" not in df.columns or df["disease_source"].isna().all()

    def test_stage2_keyword_scanning_fallback(self):
        """Test that Stage 2 finds disease via keyword in non-standard columns."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "subject_phenotype": ["colorectal cancer patient", "ulcerative colitis", "crohns"],
            # No standard disease columns
        })

        result = extract_disease_with_fallback(df)

        assert result == "host_disease_stat"
        # Keywords should match
        assert df.loc[0, "host_disease_stat"] == "crc"
        assert df.loc[1, "host_disease_stat"] == "uc"
        assert df.loc[2, "host_disease_stat"] == "cd"

        # Provenance should track keyword scan
        assert df.loc[0, "disease_source"] == "keyword_scan:subject_phenotype"
        assert df.loc[1, "disease_source"] == "keyword_scan:subject_phenotype"
        assert df.loc[2, "disease_source"] == "keyword_scan:subject_phenotype"

    def test_stage1_takes_priority_over_stage2(self):
        """Test that column name extraction (Stage 1) takes priority over keyword scanning."""
        df = pd.DataFrame({
            "sample_id": ["S1"],
            "disease": ["colorectal cancer"],  # Standard column (Stage 1)
            "notes": ["crohn disease mentioned"],  # Would match via keyword (Stage 2)
        })

        result = extract_disease_with_fallback(df)

        # Should use Stage 1 result (standard column)
        assert df.loc[0, "host_disease_stat"] == "colorectal cancer"
        # Stage 2 should NOT run (no missing values after Stage 1)
        assert "disease_source" not in df.columns or pd.isna(df.loc[0, "disease_source"])

    def test_keyword_scan_respects_exclusion_patterns(self):
        """Test that keyword scanning does NOT match excluded terms."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "notes": [
                "negative control sample",  # Should NOT match healthy
                "normal tissue adjacent to tumor",  # Should NOT match healthy
                "healthy control subject",  # Should match healthy (compound term)
            ],
        })

        result = extract_disease_with_fallback(df)

        # NOTE: This test verifies that DISEASE_KEYWORDS doesn't include generic "control"
        # Exclusion is handled by not including the keyword in the first place
        assert pd.isna(df.loc[0, "host_disease_stat"])  # negative control → unmapped
        assert pd.isna(df.loc[1, "host_disease_stat"])  # normal tissue → unmapped
        assert df.loc[2, "host_disease_stat"] == "healthy"  # healthy control → matched

    def test_keyword_scan_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "condition": ["COLORECTAL CANCER", "Ulcerative Colitis", "crohn's disease"],
        })

        result = extract_disease_with_fallback(df)

        assert df.loc[0, "host_disease_stat"] == "crc"
        assert df.loc[1, "host_disease_stat"] == "uc"
        assert df.loc[2, "host_disease_stat"] == "cd"

    def test_keyword_scan_multiple_columns(self):
        """Test that keyword scanning checks all columns."""
        df = pd.DataFrame({
            "sample_id": ["S1"],
            "age": [45],
            "sex": ["male"],
            "notes_field_xyz": ["patient diagnosed with rectal cancer"],  # Non-standard name
        })

        result = extract_disease_with_fallback(df)

        # Should find via "rectal cancer" keyword in "notes_field_xyz"
        assert df.loc[0, "host_disease_stat"] == "crc"
        assert df.loc[0, "disease_source"] == "keyword_scan:notes_field_xyz"

    def test_keyword_scan_partial_matches(self):
        """Test substring matching within column values."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "description": [
                "patient_with_ulcerative_colitis_treatment",  # Underscore variant
                "crohns disease diagnosed in 2020",  # Partial match
            ],
        })

        result = extract_disease_with_fallback(df)

        assert df.loc[0, "host_disease_stat"] == "uc"  # "ulcerative_colitis" keyword
        assert df.loc[1, "host_disease_stat"] == "cd"  # "crohns disease" keyword

    def test_no_disease_found_returns_none(self):
        """Test that None is returned when no disease data found."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "age": [30, 45],
            "sex": ["female", "male"],
        })

        result = extract_disease_with_fallback(df)

        assert result is None
        assert "host_disease_stat" in df.columns
        assert df["host_disease_stat"].isna().all()

    def test_mixed_stage_extraction(self):
        """Test that some samples use Stage 1, others use Stage 2."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "disease": ["crc", None, None],  # S1 has standard column
            "phenotype": [None, "ulcerative colitis", None],  # S2 needs keyword scan
            "notes": [None, None, "healthy control"],  # S3 needs keyword scan
        })

        result = extract_disease_with_fallback(df)

        assert result == "host_disease_stat"
        assert df.loc[0, "host_disease_stat"] == "crc"     # Stage 1
        assert df.loc[1, "host_disease_stat"] == "uc"      # Stage 2
        assert df.loc[2, "host_disease_stat"] == "healthy" # Stage 2

        # Provenance: only Stage 2 samples have disease_source
        assert pd.isna(df.loc[0, "disease_source"]) or df.loc[0, "disease_source"] is None
        assert df.loc[1, "disease_source"] == "keyword_scan:phenotype"
        assert df.loc[2, "disease_source"] == "keyword_scan:notes"

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        result = extract_disease_with_fallback(df)

        assert result is None

    def test_all_samples_have_disease_from_stage1(self):
        """Test that Stage 2 is skipped when Stage 1 finds all diseases."""
        df = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "disease": ["crc", "uc"],  # All samples have disease
            "notes": ["mentions crohns", "mentions healthy"],  # Would match if scanned
        })

        result = extract_disease_with_fallback(df)

        # Stage 1 fills all samples → Stage 2 shouldn't run
        assert df["host_disease_stat"].tolist() == ["crc", "uc"]
        # No keyword scanning occurred
        assert "disease_source" not in df.columns or df["disease_source"].isna().all()
```

### 5.2 Integration Test

**File**: `tests/integration/test_metadata_assistant_integration.py` (existing file)
**Add new test at end of file**:

```python
def test_keyword_scanning_improves_coverage_automatically(data_manager):
    """
    Integration test: keyword scanning improves coverage without user intervention.

    Validates that process_metadata_queue uses enhanced extract_disease_with_fallback
    which automatically scans columns for keywords.
    """
    from lobster.services.metadata.metadata_filtering_service import extract_disease_with_fallback

    # Simulate samples from DataBioMix-like dataset
    # Only 2/6 samples have disease in standard columns (33% coverage)
    samples = pd.DataFrame({
        "run_accession": ["SRR001", "SRR002", "SRR003", "SRR004", "SRR005", "SRR006"],
        "disease": ["crc", "uc", None, None, None, None],  # Standard column (33% coverage)
        "phenotype": [None, None, "colorectal cancer", "crohns", None, None],
        "notes": [None, None, None, None, "healthy control", "ulcerative colitis patient"],
    })

    # Run extraction
    result_col = extract_disease_with_fallback(samples)

    # Validate coverage improved
    assert result_col == "host_disease_stat"
    coverage = samples["host_disease_stat"].notna().sum() / len(samples)
    assert coverage == 1.0  # 100% coverage (6/6 samples)

    # Validate Stage 1 vs Stage 2 attribution
    # Samples 1-2: Stage 1 (standard column)
    # Samples 3-6: Stage 2 (keyword scan)
    assert samples.loc[2, "disease_source"] == "keyword_scan:phenotype"  # "colorectal cancer"
    assert samples.loc[3, "disease_source"] == "keyword_scan:phenotype"  # "crohns"
    assert samples.loc[4, "disease_source"] == "keyword_scan:notes"      # "healthy control"
    assert samples.loc[5, "disease_source"] == "keyword_scan:notes"      # "ulcerative colitis"
```

### 5.3 Regression Test

**File**: `tests/unit/services/metadata/test_disease_standardization_service.py` (existing)
**Add to ensure keywords align with mappings**:

```python
def test_keyword_dict_aligns_with_disease_mappings():
    """
    Verify that DISEASE_KEYWORDS in metadata_filtering_service aligns with
    DISEASE_MAPPINGS in disease_standardization_service.

    This prevents mismatches where keyword scan uses different categories
    than standardization service.
    """
    from lobster.services.metadata.metadata_filtering_service import DISEASE_KEYWORDS
    from lobster.services.metadata.disease_standardization_service import (
        DiseaseStandardizationService
    )

    service = DiseaseStandardizationService()

    # All keyword categories must exist in disease mappings
    for category in DISEASE_KEYWORDS.keys():
        assert category in service.DISEASE_MAPPINGS, (
            f"Keyword category '{category}' not in DISEASE_MAPPINGS"
        )

    # Verify no generic "control" in healthy keywords (Issue #1 fix)
    assert "control" not in DISEASE_KEYWORDS.get("healthy", []), (
        "Generic 'control' should NOT be in DISEASE_KEYWORDS (causes false positives)"
    )
```

---

## 6. VALIDATION PLAN

### 6.1 Pre-Implementation Validation

**Run existing tests** to establish baseline:
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
pytest tests/unit/services/metadata/test_metadata_filtering_service.py -v
pytest tests/unit/services/metadata/test_disease_standardization_service.py -v
```

**Expected**: All tests pass before changes

### 6.2 Post-Implementation Validation

**Step 1: Run new unit tests**:
```bash
pytest tests/unit/services/metadata/test_metadata_filtering_service.py::TestExtractDiseaseWithFallback -xvs
```

**Expected**: All 11 new tests pass

**Step 2: Run existing tests** (regression check):
```bash
pytest tests/unit/services/metadata/test_disease_standardization_service.py -v
pytest tests/unit/services/metadata/test_metadata_filtering_validation.py -v
```

**Expected**: No failures (backward compatible)

**Step 3: Run integration test**:
```bash
pytest tests/integration/test_metadata_assistant_integration.py::test_keyword_scanning_improves_coverage_automatically -xvs
```

**Expected**: Coverage improves from 33% → 100%

### 6.3 Real-World Validation (DataBioMix Dataset)

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/.lobster_workspace/metadata/`
**Dataset**: DataBioMix aggregated samples (655 samples from 33 publications)

**Validation steps**:
```bash
# Backup current state
cp .lobster_workspace/metadata/databiomix_aggregated_samples.json \
   .lobster_workspace/metadata/databiomix_aggregated_samples.json.backup

# Run with enhanced extraction
lobster query "process metadata queue for DataBioMix with filter: 16S human fecal"

# Check coverage improvement
# Before: ~25% coverage (165/655 samples)
# After: ~50%+ coverage (330+/655 samples) expected
```

**Success criteria**:
- Coverage ≥ 50% (meets validation threshold)
- No "Insufficient disease data" errors
- Samples enriched via keyword scanning have `disease_source='keyword_scan:*'`

---

## 7. FILE REFERENCE MAP

### Primary Files to Modify

| File | Lines | Change Type | Estimated LOC |
|------|-------|-------------|---------------|
| `lobster/services/metadata/metadata_filtering_service.py` | 30-35 | Add constant | +30 |
| `lobster/services/metadata/metadata_filtering_service.py` | 49-96 | Enhance function | +55 |
| `lobster/services/metadata/metadata_filtering_service.py` | 73-83 | Update logging | ~5 changes |

### Test Files to Create/Modify

| File | Change Type | Estimated LOC |
|------|-------------|---------------|
| `tests/unit/services/metadata/test_metadata_filtering_service.py` | Create new | +150 |
| `tests/integration/test_metadata_assistant_integration.py` | Add test | +40 |
| `tests/unit/services/metadata/test_disease_standardization_service.py` | Add test | +20 |

### Reference Files (Read-Only)

| File | Purpose |
|------|---------|
| `lobster/services/metadata/disease_standardization_service.py` | Reference for disease categories and exclusion patterns |
| `lobster/agents/metadata_assistant/` (lines 272-322) | Source of keyword scanning logic to port |
| `lobster/services/metadata/microbiome_filtering_service.py` | Reference for sample type keywords |

---

## 8. IMPLEMENTATION CHECKLIST

### Pre-Implementation
- [ ] Read this implementation plan thoroughly
- [ ] Review Section 2 (Background & Context)
- [ ] Run baseline tests (Section 6.1)
- [ ] Read source files:
  - [ ] `metadata_filtering_service.py:49-96` (target function)
  - [ ] `metadata_assistant.py:272-322` (source logic)
  - [ ] `disease_standardization_service.py:27-60` (exclusion patterns)

### Implementation
- [ ] Task 1: Add `DISEASE_KEYWORDS` constant (Section 4, Step 1)
- [ ] Task 2: Enhance `extract_disease_with_fallback` function (Section 4, Step 2)
- [ ] Task 3: Update logging messages (Section 4, Step 3)
- [ ] Task 4: Add provenance column initialization (Section 4, Step 4)

### Testing
- [ ] Task 5: Create unit test file with 11 test cases (Section 5.1)
- [ ] Task 6: Add integration test (Section 5.2)
- [ ] Task 7: Add regression test (Section 5.3)
- [ ] Task 8: Run all tests (Section 6.2)

### Validation
- [ ] Task 9: Validate on DataBioMix dataset (Section 6.3)
- [ ] Task 10: Verify coverage improvement (≥50%)
- [ ] Task 11: Check provenance tracking works
- [ ] Task 12: Confirm backward compatibility (existing workflows unchanged)

---

## 9. EDGE CASES & CONSIDERATIONS

### Edge Case 1: Empty Column Values
**Scenario**: Column exists but has `None`, `NaN`, or empty string values
**Handling**: Skip via `if pd.isna(col_value)` check (line 103 in implementation)

### Edge Case 2: Multiple Keyword Matches
**Scenario**: Column value contains keywords for multiple diseases: `"crc and uc comorbidity"`
**Handling**: First match wins (exits inner loops via `break`), prioritized by iteration order

**Solution**: If ambiguous, use study context or leave unmapped (conservative)

### Edge Case 3: Numeric Column Values
**Scenario**: Age, counts, IDs (e.g., `45`, `1000`, `"SRR123"`)
**Handling**: Convert to string via `str(col_value).lower()`, unlikely to match disease keywords

### Edge Case 4: Special Characters
**Scenario**: `"Crohn's_disease_active"`, `"ulcerative-colitis-moderate"`
**Handling**: Keywords include underscore and space variants (`"crohns_disease"`, `"ulcerative colitis"`)

### Edge Case 5: False Positive Prevention
**Scenario**: `"colorectal surgery"` should NOT match "colorectal cancer"
**Current**: Would match "colorectal" keyword
**Mitigation**: Accept false positives for recall, rely on standardization service to validate

---

## 10. ROLLBACK PLAN

If integration causes issues:

1. **Revert changes**:
   ```bash
   git diff HEAD lobster/services/metadata/metadata_filtering_service.py
   git checkout HEAD -- lobster/services/metadata/metadata_filtering_service.py
   ```

2. **Run regression tests**:
   ```bash
   pytest tests/unit/services/metadata/ -v
   ```

3. **Identify root cause**:
   - Check logs for keyword scanning errors
   - Review failed test cases
   - Validate keyword dictionary alignment

---

## 11. FUTURE ENHANCEMENTS (Post-Sprint 1)

### Enhancement 1: Configurable Keyword Dictionary
**Current**: Hardcoded for IBD/CRC studies
**Future**: Load from JSON config or database for multi-disease support

### Enhancement 2: Confidence Scoring
**Current**: Keyword matches assumed 100% confidence
**Future**: Score based on match quality (exact vs partial vs token)

### Enhancement 3: Service Extraction (Option 1 Full Implementation)
**Current**: Enhanced function in MetadataFilteringService
**Future**: Create `DiseaseEnrichmentService` for reusability across agents

---

## 12. CONTACTS & REFERENCES

**Questions**: Contact ultrathink (implementation lead)
**Architecture decisions**: Reference Gemini collaboration output (2026-01-10)
**Related issues**: Issue #1 (control/normal false positives), DataBioMix Bug 3

**References**:
- Lobster service pattern: `CLAUDE.md` Section 4.2
- W3C-PROV provenance: `CLAUDE.md` Section 4.4
- DataBioMix validation report: `docs/customers/databiomix_validation.md` (if exists)

---

## 13. DELIVERABLES

Upon completion, provide:
1. ✅ Modified `metadata_filtering_service.py` with Stage 2 integration
2. ✅ New test file: `test_metadata_filtering_service.py` (11 test cases)
3. ✅ Integration test added to existing file
4. ✅ All tests passing (100% pass rate)
5. ✅ Coverage report showing improvement on DataBioMix dataset
6. ✅ Git commit message:
   ```
   feat: integrate keyword scanning into disease extraction fallback

   Enhances extract_disease_with_fallback() with 2-stage extraction:
   - Stage 1: Check standard column names (existing)
   - Stage 2: Scan ALL columns for disease keywords (new)

   Improves disease coverage from ~20-30% to ~40-60% automatically,
   reducing need for manual LLM-based enrichment.

   Changes:
   - Add DISEASE_KEYWORDS constant (CRC, UC, CD, healthy)
   - Implement keyword scanning loop for remaining missing values
   - Add disease_source provenance tracking
   - Add 11 unit tests + 1 integration test

   Validates on DataBioMix: 655 samples, 25% → 50%+ coverage.

   Related: Issue #1 (control/normal exclusions), Bug 3 (disease extraction)
   ```

---

**END OF IMPLEMENTATION PLAN**

This document provides complete context for implementing Option 1 (Enhanced extract_disease_with_fallback) following Lobster's service pattern architecture and scientific rigor standards.
