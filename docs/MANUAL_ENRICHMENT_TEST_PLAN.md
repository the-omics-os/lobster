# Manual Sample Enrichment Test Plan
## PRJNA642308 IBD Study - Validation of metadata_assistant Enrichment Capabilities

**Date**: 2025-12-02
**Test Entry**: PRJNA642308 (IBD Dietary Intervention)
**Entry ID**: `pub_queue_doi_10_1080_19490976_2022_2046244`
**Samples**: 409
**Current Completeness**: 87.5%
**Target**: 100% (disease 0% → 100%)

---

## Executive Summary

**Objective**: Validate that metadata_assistant can manually enrich samples using publication context via:
1. Reading full publication text (not just methods)
2. Extracting missing demographics (disease, age, sex)
3. Propagating publication-level context to samples
4. Exporting enriched dataset with provenance

**Why PRJNA642308**:
- **Best test case** (87.5% complete, only disease missing)
- Publication title explicitly states "inflammatory bowel disease"
- Large cohort (409 samples) - statistically significant
- Already processed (workspace_metadata_keys populated)

**Expected Outcome**: Prove metadata_assistant can achieve 100% metadata completeness with 1-line enrichment code

---

## Pre-Test Verification

### Step 0.1: Verify Queue Entry Exists
```bash
cd /Users/tyo/GITHUB/omics-os/lobster

# Check if entry exists in queue
grep -l "PRJNA642308" .lobster_workspace/queues/publication_queue.jsonl

# If found, extract entry_id
grep "PRJNA642308" .lobster_workspace/queues/publication_queue.jsonl | \
  python3 -c "import sys, json; entry=json.load(sys.stdin); print(entry['entry_id'])"
```

**Expected**: `pub_queue_doi_10_1080_19490976_2022_2046244`

### Step 0.2: Verify Workspace Files Exist
```bash
# Check metadata files
ls -lh .lobster_workspace/metadata/pub_queue_doi_10_1080_19490976_2022_2046244_*.json

# Expected:
# - *_metadata.json (full publication text)
# - *_methods.json (methods section)
# - *_identifiers.json (extracted SRA/GEO IDs)
# - sra_PRJNA642308_samples.json (409 samples)
```

### Step 0.3: Inspect Current Metadata Quality
```bash
# Load samples and measure completeness
python3 << 'EOF'
import json
import pandas as pd

with open(".lobster_workspace/metadata/sra_prjna642308_samples.json") as f:
    data = json.load(f)
    samples = data["data"]["samples"]

df = pd.DataFrame(samples)

critical_fields = ["organism_name", "host", "tissue", "isolation_source",
                  "disease", "age", "sex", "sample_type"]

print("BASELINE Metadata Completeness:")
print("-" * 60)
for field in critical_fields:
    if field in df.columns:
        coverage = df[field].notna().sum() / len(df) * 100
        status = "✓" if coverage >= 70 else "⚠️" if coverage >= 30 else "✗"
        print(f"{status} {field:20s}: {coverage:5.1f}%")
    else:
        print(f"✗ {field:20s}:   0.0% (column missing)")

overall = sum([df[f].notna().sum() / len(df) * 100 if f in df.columns else 0.0
               for f in critical_fields]) / len(critical_fields)
print("-" * 60)
print(f"OVERALL: {overall:.1f}%")
EOF
```

**Expected Baseline**:
```
✓ organism_name     : 100.0%
✓ host              : 100.0%
✓ tissue            : 100.0%
✓ isolation_source  : 100.0%
✗ disease           :   0.0%  ← TARGET FOR ENRICHMENT
✓ age               : 100.0%
✓ sex               : 100.0%
✗ sample_type       :   0.0%
---------------------------
OVERALL: 87.5%
```

---

## Test Plan: Manual Enrichment Workflow

### Test 1: Publication Content Accessibility (10 min)

**Goal**: Verify metadata_assistant can read full publication text

**Commands**:
```bash
# Test 1.1: Read publication metadata (should have full text)
lobster query "Get full metadata for pub_queue_doi_10_1080_19490976_2022_2046244_metadata"

# Expected: JSON with "content" field containing:
# - Title: "Dietary manipulation of gut microbiome in inflammatory bowel disease..."
# - Introduction section
# - Methods section
# - Results section
# - Discussion section
```

**Success Criteria**:
- [ ] Full publication text accessible (≥1000 characters)
- [ ] Title contains "inflammatory bowel disease"
- [ ] Methods section present
- [ ] Results section present

**Fallback Test**:
```bash
# Test 1.2: Read methods section only
lobster query "Get methods section for pub_queue_doi_10_1080_19490976_2022_2046244_methods"

# Expected: Methods text with cohort description
```

**Success Criteria**:
- [ ] Methods text ≥200 characters
- [ ] Contains cohort demographics (age/sex/disease)

---

### Test 2: Disease Extraction from Title (15 min)

**Goal**: Extract "IBD" disease from publication title and propagate to samples

**Command**:
```bash
lobster query "Read the publication metadata for entry pub_queue_doi_10_1080_19490976_2022_2046244 and enrich all 409 samples with disease='ibd' inferred from the title which mentions inflammatory bowel disease"
```

**Expected metadata_assistant behavior**:
1. Reads `pub_queue_doi_10_1080_19490976_2022_2046244_metadata` via get_content_from_workspace
2. Extracts title from metadata JSON
3. Detects "inflammatory bowel disease" → disease="ibd"
4. Uses execute_custom_code to propagate to all samples
5. Stores enriched samples in workspace

**Validation**:
```bash
# Check if enriched samples were created
ls -lh .lobster_workspace/metadata/*enriched*.json

# Measure disease coverage after enrichment
python3 << 'EOF'
import json
import pandas as pd

# Try to load enriched samples (if created)
try:
    with open(".lobster_workspace/metadata/prjna642308_enriched.json") as f:
        data = json.load(f)
        samples = data["data"]["samples"]

    df = pd.DataFrame(samples)
    disease_coverage = df["disease"].notna().sum() / len(df) * 100

    print(f"Disease coverage after enrichment: {disease_coverage:.1f}%")
    print(f"Disease values: {df['disease'].value_counts().to_dict()}")

    # Check provenance
    if "disease_source" in df.columns:
        print(f"Disease sources: {df['disease_source'].value_counts().to_dict()}")
except FileNotFoundError:
    print("❌ Enriched file not created - enrichment may have failed or used different name")
EOF
```

**Success Criteria**:
- [ ] disease coverage improves: 0% → ≥80%
- [ ] disease value = "ibd" for majority of samples
- [ ] disease_source field present (provenance tracking)
- [ ] No data corruption (all 409 samples preserved)

---

### Test 3: Sample Type Inference (10 min)

**Goal**: Infer sample_type from isolation_source field

**Current State**:
- isolation_source: 100% (all samples have this)
- sample_type: 0% (missing)
- Pattern: isolation_source contains "fecal" → sample_type="fecal"

**Command**:
```bash
lobster query "Infer sample_type field from isolation_source for PRJNA642308 samples using pattern matching (fecal → fecal, tissue → tissue, biopsy → biopsy)"
```

**Expected code execution**:
```python
samples = prjna642308_enriched["samples"]
for s in samples:
    iso_source = s.get("isolation_source", "").lower()
    if "fecal" in iso_source or "feces" in iso_source or "stool" in iso_source:
        s["sample_type"] = "fecal"
    elif "tissue" in iso_source or "biopsy" in iso_source:
        s["sample_type"] = "tissue"
    elif "blood" in iso_source or "plasma" in iso_source or "serum" in iso_source:
        s["sample_type"] = "blood"
    else:
        s["sample_type"] = "unknown"
    s["sample_type_source"] = "inferred_from_isolation_source"
```

**Validation**:
```bash
python3 << 'EOF'
import json
import pandas as pd

with open(".lobster_workspace/metadata/prjna642308_enriched.json") as f:
    data = json.load(f)
    samples = data["data"]["samples"]

df = pd.DataFrame(samples)
sample_type_coverage = df["sample_type"].notna().sum() / len(df) * 100

print(f"Sample type coverage: {sample_type_coverage:.1f}%")
print(f"Sample type distribution: {df['sample_type'].value_counts().to_dict()}")
EOF
```

**Success Criteria**:
- [ ] sample_type coverage: 0% → ≥80%
- [ ] sample_type values: fecal, tissue, blood (reasonable)
- [ ] Inference accuracy: ≥90% (spot-check 10 samples)

---

### Test 4: Final Export with Complete Metadata (5 min)

**Goal**: Export enriched dataset with 100% completeness

**Command**:
```bash
lobster query "Export PRJNA642308 enriched samples to CSV with all metadata fields"
```

**Validation**:
```bash
# Check export exists
ls -lh .lobster_workspace/metadata/exports/prjna642308_enriched*.csv

# Measure final completeness
python3 << 'EOF'
import pandas as pd

df = pd.read_csv(".lobster_workspace/metadata/exports/prjna642308_enriched_2025-12-02.csv")

critical_fields = ["organism_name", "host", "tissue", "isolation_source",
                  "disease", "age", "sex", "sample_type"]

print("FINAL Metadata Completeness:")
print("-" * 60)
for field in critical_fields:
    if field in df.columns:
        coverage = df[field].notna().sum() / len(df) * 100
        status = "✓" if coverage >= 70 else "⚠️" if coverage >= 30 else "✗"
        print(f"{status} {field:20s}: {coverage:5.1f}%")
    else:
        print(f"✗ {field:20s}:   0.0% (missing)")

overall = sum([df[f].notna().sum() / len(df) * 100 if f in df.columns else 0.0
               for f in critical_fields]) / len(critical_fields)
print("-" * 60)
print(f"OVERALL: {overall:.1f}%")
print(f"\nImprovement: 87.5% → {overall:.1f}% (+{overall - 87.5:.1f}%)")
EOF
```

**Success Criteria**:
- [ ] Overall completeness: 87.5% → ≥95%
- [ ] disease: 0% → ≥80%
- [ ] sample_type: 0% → ≥80%
- [ ] CSV file readable (no corruption)
- [ ] 409 rows present (no data loss)

---

## Test 5: Enrichment Provenance Validation (10 min)

**Goal**: Verify enrichment is traceable (audit trail)

**Validation**:
```bash
# Check enrichment source fields
python3 << 'EOF'
import pandas as pd

df = pd.read_csv(".lobster_workspace/metadata/exports/prjna642308_enriched_2025-12-02.csv")

# Check for *_source fields
source_fields = [c for c in df.columns if c.endswith("_source")]
print(f"Provenance fields: {source_fields}")

# Show distribution of sources
for field in source_fields:
    print(f"\n{field}:")
    print(df[field].value_counts())
EOF
```

**Expected source fields**:
- `disease_source`: "inferred_from_publication_title"
- `sample_type_source`: "inferred_from_isolation_source"
- (age_source, sex_source if enriched)

**Success Criteria**:
- [ ] All enriched fields have corresponding *_source field
- [ ] Source values reference specific publication or field
- [ ] Provenance enables reproducibility

---

## Test 6: Multi-Entry Batch Enrichment (Optional, 20 min)

**Goal**: Test if enrichment workflow scales to multiple entries

**Command**:
```bash
# Enrich top 3 GOLD entries
lobster query "Process and enrich PRJNA642308, PRJNA1139414, PRJNA784939 with disease from publication titles"
```

**Validation**:
- [ ] All 3 entries enriched successfully
- [ ] Each entry has unique disease context
- [ ] No cross-contamination (PRJNA642308 disease ≠ PRJNA1139414 disease)
- [ ] Batch processing time reasonable (<5 min for 3 entries)

---

## Success Criteria Summary

### Must Pass (Blockers)
- [ ] **Test 1**: Can read full publication text (not just methods)
- [ ] **Test 2**: Can extract disease and propagate to samples
- [ ] **Test 4**: Can export enriched dataset with ≥95% completeness
- [ ] **Test 5**: Enrichment is traceable (provenance fields present)

### Should Pass (Important)
- [ ] **Test 3**: Can infer sample_type from isolation_source
- [ ] Disease coverage improves by ≥50 percentage points

### Nice to Have
- [ ] **Test 6**: Batch enrichment works for 3+ entries
- [ ] Overall completeness reaches 100%

---

## Fallback Scenarios

### If Test 2 Fails (Cannot extract disease)
**Diagnosis steps**:
1. Check if publication metadata file exists
2. Verify "content" field in JSON has text
3. Check if title mentions disease terms
4. Manually inspect title and identify disease

**Workaround**:
```bash
# Manual enrichment with explicit disease code
lobster query "Enrich PRJNA642308 samples with disease='ibd' explicitly"
```

### If execute_custom_code Unavailable
**Alternative**: Use filter_samples_by which already integrates disease extraction
```bash
lobster query "Filter PRJNA642308 samples by 'IBD' to trigger disease extraction"
# Disease extraction runs automatically during filtering
```

### If Enrichment Doesn't Persist
**Check**: Workspace write permissions, file locking, storage errors
```bash
# Verify write_to_workspace works
lobster query "Test write_to_workspace by exporting current samples"
```

---

## Expected Timeline

| Test | Duration | Critical Path |
|------|----------|---------------|
| Pre-test verification | 5 min | No |
| Test 1: Content access | 10 min | Yes |
| Test 2: Disease enrichment | 15 min | Yes |
| Test 3: Sample type inference | 10 min | No |
| Test 4: Final export | 5 min | Yes |
| Test 5: Provenance validation | 10 min | Yes |
| Test 6: Batch enrichment | 20 min | No |

**Total**: 75 minutes (50 min critical path)

---

## Delegation Strategy

### Option 1: Manual Testing (User-Driven)
User executes commands via `lobster query` with natural language

**Pros**: Interactive, can adapt on the fly
**Cons**: Slower, requires user intervention

### Option 2: Automated Testing (lo-ass Agent)
Single lo-ass agent executes entire test plan programmatically

**Pros**: Faster, comprehensive, measurable
**Cons**: Less interactive

### Option 3: Hybrid (Recommended)
1. User executes Test 1-2 (critical path) to validate core capability
2. lo-ass agent executes Test 3-6 (comprehensive validation)

---

## Post-Test Deliverables

### 1. Enrichment Validation Report
```markdown
## PRJNA642308 Manual Enrichment Test Results

### Metadata Completeness Improvement
- Before: 87.5%
- After: X.X% (MEASURED)
- Improvement: +X.X%

### Field-Level Results
| Field | Before | After | Method | Status |
|-------|--------|-------|--------|--------|
| disease | 0% | X% | Title inference | PASS/FAIL |
| sample_type | 0% | X% | isolation_source | PASS/FAIL |

### Tool Capability Validation
- Full text access: PASS/FAIL
- Disease extraction: PASS/FAIL
- Sample propagation: PASS/FAIL
- Export enriched: PASS/FAIL
- Provenance tracking: PASS/FAIL

### Recommendation
- Enrichment workflow viable: YES/NO
- Ready for customer use: YES/NO
- Recommended improvements: [list]
```

### 2. Customer Demo Dataset
```
prjna642308_enriched_2025-12-02.csv
- 409 samples x 40+ columns
- 100% completeness (if successful)
- Publication context included
- Ready for DataBioMix analysis
```

### 3. Enrichment Code Template
```python
# Reusable template for future enrichments
# Parametrized by entry_id, disease_term, etc.
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Publication text not cached | LOW | HIGH | Verify workspace files exist (pre-test 0.2) |
| execute_custom_code fails | LOW | HIGH | Test with simple code first |
| Disease inference incorrect | MEDIUM | MEDIUM | Manual validation of title extraction |
| Enrichment doesn't persist | LOW | HIGH | Verify write_to_workspace works |
| Sample corruption | LOW | CRITICAL | Compare sample counts before/after |

---

## Next Steps After Testing

### If Tests PASS (≥4/5 success criteria met)
1. Document enrichment workflow in metadata_assistant system prompt
2. Create enrichment tutorial for DataBioMix
3. Package PRJNA642308 as demo dataset
4. Proceed with top 3 GOLD entries enrichment

### If Tests FAIL (<4/5 criteria met)
1. Identify specific failure mode
2. Enhance tool capabilities as needed
3. Re-test with fixes applied
4. Document limitations for customer

---

## Execution Command (Start Test)

```bash
# Single command to execute entire test plan
cd /Users/tyo/GITHUB/omics-os/lobster

# Start interactive session
lobster chat

# Then execute tests sequentially:
# > Get full metadata for pub_queue_doi_10_1080_19490976_2022_2046244_metadata
# > Enrich PRJNA642308 samples with disease from publication title
# > Infer sample_type from isolation_source for PRJNA642308
# > Export enriched PRJNA642308 to CSV
# > Show completeness statistics
```

---

**Test Owner**: ultrathink (Claude Code)
**Priority**: CRITICAL (blocks DataBioMix delivery validation)
**Timeline**: Execute within next 2 hours
