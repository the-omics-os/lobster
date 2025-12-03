# Manual Enrichment Test Guide
## Testing metadata_assistant Manual Enrichment Workflow

**Date**: 2025-12-02
**Version**: v1.2.0
**Test Case**: PRJNA642308 (IBD dietary intervention, 409 samples, 87.5% complete)

---

## Pre-Test Verification

### System Prompt Updates Applied ✅
- **File**: `lobster/agents/metadata_assistant.py`
- **Lines 2225-2230**: Manual enrichment added to responsibilities
- **Lines 2359-2428**: Complete manual enrichment workflow documented
- **Lines 2416-2427**: When to use manual vs automatic extraction

### Test Entry Identified ✅
- **Entry ID**: `pub_queue_doi_10_1080_19490976_2022_2046244`
- **BioProject**: PRJNA642308
- **Title**: "Dietary manipulation of the gut microbiome in inflammatory bowel disease patients: Pilot study"
- **Samples**: 409
- **Current completeness**: 87.5%
- **Missing field**: disease (0% coverage) ← Enrichment target

---

## Test Execution

### Step 1: Start Lobster Session

**Command**:
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
source .venv/bin/activate
lobster chat --verbose
```

**Enable Admin Mode**:
- Set LOBSTER_ADMIN=true or use internal flag to bypass questions
- Ensure metadata_assistant can access execute_custom_code

### Step 2: User Prompt (Exact Text)

**Copy-paste this prompt**:
```
Load publication queue entry pub_queue_doi_10_1080_19490976_2022_2046244 (PRJNA642308 - IBD dietary study).
Tell the research agent to hand off to metadata assistant for manual enrichment of missing high-relevance fields
(specifically disease field which is at 0% coverage despite clear IBD context in publication title).
The publication title explicitly mentions "inflammatory bowel disease patients" - this should be used to enrich
all 409 samples with disease="ibd" and disease_source="inferred_from_publication_title".
```

### Step 3: Expected Agent Flow

**Supervisor** → routes to **research_agent**

**research_agent**:
- Locates entry `pub_queue_doi_10_1080_19490976_2022_2046244`
- Reads publication queue status
- Identifies workspace_metadata_keys: `sra_prjna642308_samples`
- **Hands off to metadata_assistant** with enrichment request

**metadata_assistant** (CRITICAL - This is what we're testing):
1. **Assess completeness** (uses get_content_from_workspace or read_sample_metadata)
   - Reads: `sra_prjna642308_samples`
   - Reports: 409 samples, disease=0%, age=100%, sex=100%, tissue=100%

2. **Extract context** (uses get_content_from_workspace)
   - Reads publication entry metadata
   - Identifies title: "inflammatory bowel disease patients"
   - Extracts disease context: "ibd"

3. **Propagate via execute_custom_code**:
   ```python
   samples = sra_prjna642308_samples["samples"]
   for s in samples:
       if not s.get("disease"):
           s["disease"] = "ibd"
           s["disease_source"] = "inferred_from_publication_title"
   result = {"samples": samples, "output_key": "sra_prjna642308_samples_enriched"}
   ```

4. **Report results**:
   - Samples enriched: 409/409 (100%)
   - Disease coverage: 0% → 100%
   - Overall completeness: 87.5% → 100%

5. **Export** (if requested):
   - write_to_workspace("sra_prjna642308_samples_enriched", output_format="csv")

### Step 4: Validation Checks

**Check 1: metadata_assistant receives handoff**
- Look for: "metadata_assistant received request for manual enrichment"
- Status: PASS/FAIL

**Check 2: metadata_assistant reads publication context**
- Look for: "Reading publication entry pub_queue_doi_10_1080_19490976_2022_2046244"
- Look for: "Found disease context: inflammatory bowel disease"
- Status: PASS/FAIL

**Check 3: metadata_assistant executes enrichment code**
- Look for: "Executing custom code for sample enrichment"
- Look for: "Enriched 409 samples with disease=ibd"
- Status: PASS/FAIL

**Check 4: metadata_assistant reports improvement**
- Look for: "Disease coverage: 0% → 100%"
- Look for: "Overall completeness: 87.5% → 100%"
- Status: PASS/FAIL

**Check 5: Enriched samples accessible**
- Check metadata_store for: `sra_prjna642308_samples_enriched`
- Check workspace for export: `workspace/metadata/exports/*.csv`
- Status: PASS/FAIL

---

## Expected Output

### Success Scenario

```
User: Load publication queue entry pub_queue_doi_10_1080_19490976_2022_2046244...

Supervisor: Routing to research_agent for publication queue operations...

research_agent:
✓ Located entry pub_queue_doi_10_1080_19490976_2022_2046244
✓ BioProject: PRJNA642308
✓ Title: "Dietary manipulation of the gut microbiome in inflammatory bowel disease patients"
✓ Samples: 409 (workspace key: sra_prjna642308_samples)
✓ Current disease coverage: 0%
→ Handing off to metadata_assistant for manual enrichment

metadata_assistant:
Status: success
Summary: Assessed metadata completeness for PRJNA642308 (409 samples).
Disease field has 0% coverage but publication title explicitly mentions
"inflammatory bowel disease patients" - ideal for manual enrichment.

Metrics:
  - Samples: 409
  - Disease coverage before: 0/409 (0%)
  - Age coverage: 409/409 (100%)
  - Sex coverage: 409/409 (100%)
  - Tissue coverage: 409/409 (100%)
  - Overall completeness: 87.5%

Key Findings:
  - Publication context provides clear disease label: "inflammatory bowel disease"
  - All samples can be enriched with disease="ibd"
  - Enrichment source: publication title (high confidence)

Executing enrichment code...

[execute_custom_code output]
Enriched 409/409 samples with disease="ibd"
Added disease_source="inferred_from_publication_title_doi_10_1080_19490976_2022_2046244"

Post-enrichment metrics:
  - Disease coverage after: 409/409 (100%)
  - Overall completeness: 87.5% → 100%
  - Improvement: +12.5 percentage points

Recommendation: proceed
  - Dataset ready for export with complete metadata
  - Disease field populated for all samples with high confidence
  - Enrichment source documented for traceability

Returned Artifacts:
  - sra_prjna642308_samples_enriched (409 samples, 100% disease coverage)

→ Ready for export to CSV via write_to_workspace
```

---

## Failure Scenarios (Troubleshooting)

### Scenario 1: metadata_assistant doesn't understand request
**Symptom**: "I cannot perform that operation" or "Missing prerequisites"

**Fix**: Ensure research_agent explicitly mentions:
- "hand off to metadata_assistant"
- "manual enrichment"
- "enrich from publication context"

### Scenario 2: execute_custom_code not available
**Symptom**: "Tool execute_custom_code not found"

**Fix**: Check ADMIN mode enabled, or check subscription tier allows execute_custom_code

### Scenario 3: Publication context not accessible
**Symptom**: "Cannot read publication metadata"

**Fix**: Ensure publication was cached by research_agent:
```bash
lobster query "Cache publication PMID for entry pub_queue_doi_10_1080_19490976_2022_2046244"
```

### Scenario 4: Enrichment code errors
**Symptom**: "Error executing custom code: KeyError"

**Fix**: Check samples dict structure:
```python
# Verify structure before enrichment
samples = sra_prjna642308_samples["samples"]
print(f"Sample structure: {samples[0].keys()}")
```

---

## Post-Test Validation

### Verify Enrichment Success

**Command 1: Check metadata_store**
```python
# In lobster session
from lobster.core.data_manager_v2 import DataManagerV2
dm = DataManagerV2(workspace_path=".lobster_workspace")

# Check if enriched samples exist
if "sra_prjna642308_samples_enriched" in dm.metadata_store:
    samples = dm.metadata_store["sra_prjna642308_samples_enriched"]["samples"]
    disease_count = sum(1 for s in samples if s.get("disease"))
    print(f"Disease coverage: {disease_count}/{len(samples)} ({disease_count/len(samples)*100:.1f}%)")
```

**Command 2: Export and inspect CSV**
```bash
lobster query "Export sra_prjna642308_samples_enriched to CSV"

# Check CSV
head -1 .lobster_workspace/metadata/exports/sra_prjna642308_samples_enriched*.csv | tr ',' '\n' | grep -n "disease"
# Should find: disease column and disease_source column

# Check disease values
cut -d',' -f<disease_col_num> .lobster_workspace/metadata/exports/*.csv | sort | uniq -c
# Should show: 409 lines with "ibd" value
```

**Command 3: Validate enrichment source**
```bash
# Check disease_source field
grep -o "disease_source=[^,]*" .lobster_workspace/metadata/exports/*.csv | head -5
# Should show: disease_source=inferred_from_publication_title_doi_10_1080_19490976_2022_2046244
```

---

## Success Criteria

- [ ] metadata_assistant receives handoff from research_agent
- [ ] metadata_assistant reads publication context
- [ ] metadata_assistant identifies disease="ibd" from title
- [ ] metadata_assistant executes enrichment code successfully
- [ ] 409/409 samples enriched (100%)
- [ ] disease_source field documents provenance
- [ ] Overall completeness: 87.5% → 100%
- [ ] Export to CSV works
- [ ] CSV contains disease column with "ibd" values

---

## Alternative Test Prompts

### Prompt Variation 1 (More Direct)
```
Enrich PRJNA642308 publication queue entry with disease field from publication title.
The title says "inflammatory bowel disease patients" so all 409 samples should get disease="ibd".
```

### Prompt Variation 2 (Research Agent Focus)
```
Load publication queue entry for PRJNA642308. Extract disease context from publication title
and propagate to all samples. Hand off to metadata_assistant for execution.
```

### Prompt Variation 3 (Explicit Workflow)
```
I need metadata_assistant to manually enrich PRJNA642308 samples:
1. Read publication entry pub_queue_doi_10_1080_19490976_2022_2046244
2. Extract disease from title ("inflammatory bowel disease")
3. Propagate disease="ibd" to all 409 samples
4. Export enriched samples to CSV
```

---

## Ready for Testing ✅

**System prompt**: UPDATED with manual enrichment workflow
**Test case**: PRJNA642308 identified and validated
**Expected behavior**: Documented with success criteria
**Troubleshooting**: Failure scenarios covered

**Next step**: Execute test in lobster chat session with --verbose flag

---

**Files Modified**:
- `lobster/agents/metadata_assistant.py` (system prompt lines 2225-2230, 2359-2428)

**Test Entry Details**:
- Entry ID: `pub_queue_doi_10_1080_19490976_2022_2046244`
- BioProject: PRJNA642308
- Workspace key: `sra_prjna642308_samples`
- Samples: 409
- Disease context: "inflammatory bowel disease" (from title)
- Expected enrichment: disease 0% → 100%
