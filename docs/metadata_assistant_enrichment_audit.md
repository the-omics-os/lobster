# Metadata Assistant Manual Enrichment Capability Audit

**Date**: 2025-12-03
**Objective**: Determine if metadata_assistant has sufficient tools to MANUALLY ENRICH sample metadata with missing critical fields (host, disease, tissue, age, sex) by leveraging publication full text and metadata.

**Problem Context**: 29% disease coverage means 71% of samples LACK disease information at sample-level. Can metadata_assistant manually propagate publication-level context to individual samples?

---

## Executive Summary

**Verdict**: ✅ **SUFFICIENT** (with caveats)

metadata_assistant CAN perform manual enrichment workflows using a 3-tool combination:
1. `get_content_from_workspace(level="methods")` - Extract publication methods sections
2. `execute_custom_code()` - Parse demographics and modify samples
3. `write_to_workspace(output_format="csv")` - Export enriched samples

**Key Finding**: The `execute_custom_code` tool (lines 2044-2176 in metadata_assistant.py) is a **powerful fallback mechanism** that enables:
- Reading publication text/methods via workspace
- Extracting demographics using regex/NLP
- Modifying individual sample records
- Persisting enriched samples back to metadata_store

**Limitations**:
- NO dedicated "extract_demographics" tool (requires custom code)
- CANNOT read full PDF text (only abstracts/methods from cached publications)
- CANNOT parse PMC XML directly (would require research_agent to cache first)
- Workflow is MANUAL and study-specific (not scalable without automation)

---

## 1. Tool Inventory

### metadata_assistant Tool Capabilities

| Tool | Capability | Enrichment Relevance |
|------|-----------|---------------------|
| **get_content_from_workspace** | Read cached publications, methods sections, workspace metadata | ✅ **HIGH** - Can access publication context |
| **write_to_workspace** | Export enriched samples to CSV/JSON | ✅ **HIGH** - Can persist enriched data |
| **execute_custom_code** | Execute arbitrary Python code with workspace access | ✅ **EXCELLENT** - Flexible enrichment logic |
| **process_metadata_entry** | Process publication queue entries, extract samples | ✅ **MEDIUM** - Aggregates samples from queue |
| **process_metadata_queue** | Batch process multiple queue entries | ✅ **MEDIUM** - Batch aggregation |
| **update_metadata_status** | Update queue entry statuses | ✅ **LOW** - Status management |
| **read_sample_metadata** | Read sample-level metadata from modalities/cache | ✅ **MEDIUM** - Can inspect samples |
| **standardize_sample_metadata** | Convert to Pydantic schemas | ✅ **LOW** - Schema validation, not enrichment |
| **validate_dataset_content** | Validate dataset completeness | ✅ **LOW** - QC, not enrichment |
| **filter_samples_by** | Multi-criteria filtering (16S, host, disease) | ✅ **MEDIUM** - Can filter by disease |

### Critical Tool: `execute_custom_code`

**Location**: `metadata_assistant.py` lines 2044-2176

**Capabilities**:
- Execute arbitrary Python code for sample-level operations
- Auto-inject workspace files (CSV, JSON, publication data)
- Access to `workspace_path`, `publication_queue`, all metadata files
- Persist results back to `metadata_store` via return dict
- Support for regex, pandas, numpy for data manipulation

**Available Namespace**:
```python
{
    'workspace_path': Path('/path/to/workspace'),
    # Auto-loaded files:
    'aggregated_filtered_samples_16s_human': {...},  # From workspace/metadata/*.json
    'sra_PRJNA123_samples': {...},  # From workspace/metadata/*.json
    'publication_queue': [...],  # Publication queue entries
    # Standard libraries: pandas, numpy, re, json, etc.
}
```

**Persistence Pattern**:
```python
# Return dict with 'samples' and 'output_key' to persist to metadata_store
result = {
    'samples': enriched_samples_list,  # List of dicts
    'output_key': 'reviewed_samples',  # Key for metadata_store
    'stats': {'count': len(enriched_samples_list)}
}
# → Automatically stored in metadata_store['reviewed_samples']
# → Can then export via write_to_workspace()
```

---

## 2. Enrichment Workflow Analysis

### Scenario: Enrich PRJNA123456 with missing disease/age/sex

**Can metadata_assistant perform this workflow?** ✅ **YES** (with manual steps)

#### Step 1: Read publication metadata/methods

**Tool**: `get_content_from_workspace`

```python
# Read publication methods section
get_content_from_workspace(
    identifier="publication_PMID12345",
    workspace="literature",
    level="methods"
)
```

**Capability**: ✅ **POSSIBLE**
- Can extract methods sections from cached publications
- WorkspaceContentService supports `level="methods"` (workspace_tool.py:474-479)
- Returns formatted markdown with methods text

**Gap**:
- ❌ CANNOT read full PDF text (only if research_agent cached abstract/methods)
- ❌ CANNOT parse PMC XML directly (would need research_agent to fetch first)
- ✅ CAN access any content cached by research_agent in workspace/literature/

#### Step 2: Extract demographics from Methods

**Tool**: `execute_custom_code`

```python
execute_custom_code(
    python_code="""
import re

# Read publication methods
methods_text = get_content_from_workspace(
    identifier="publication_PMID12345",
    workspace="literature",
    level="methods"
)

# Extract age range (e.g., "Patients aged 45-65 years")
age_match = re.search(r'aged?\s+(\d+)[\s-]+(\d+)\s+years', methods_text)
age_min, age_max = int(age_match.group(1)), int(age_match.group(2)) if age_match else (None, None)
age_midpoint = (age_min + age_max) / 2 if age_min and age_max else None

# Extract sex distribution (e.g., "60% male, 40% female")
sex_match = re.search(r'(\d+)%?\s+male.*?(\d+)%?\s+female', methods_text)
male_pct = int(sex_match.group(1)) / 100 if sex_match else None

# Extract disease (e.g., "diagnosed with CRC")
disease = "crc" if "colorectal cancer" in methods_text.lower() or "crc" in methods_text.lower() else None

result = {
    'age_midpoint': age_midpoint,
    'male_pct': male_pct,
    'disease': disease
}
""",
    description="Extract demographics from publication methods"
)
```

**Capability**: ✅ **POSSIBLE**
- `execute_custom_code` supports full Python standard library (regex, pandas, numpy)
- Can call `get_content_from_workspace` from within custom code
- Can use regex, NLP, or rule-based extraction
- Returns result dict with extracted values

**Gap**:
- ❌ NO dedicated "extract_demographics" tool (must write custom regex/parsing logic per study)
- ⚠️ Requires LLM to write study-specific extraction code
- ⚠️ Not scalable across 1000s of studies (would need automation)

#### Step 3: Propagate to samples

**Tool**: `execute_custom_code`

```python
execute_custom_code(
    python_code="""
import pandas as pd
import random

# Read samples from workspace
samples = aggregated_filtered_samples_16s_human['data']['samples']

# Demographics from Step 2
age_midpoint = 55
male_pct = 0.6
disease = "crc"

# Enrich each sample
for sample in samples:
    # Propagate disease if missing
    if not sample.get("disease"):
        sample["disease"] = disease
        sample["disease_source"] = "inferred_from_publication_PMID12345"

    # Propagate age if missing
    if not sample.get("age"):
        # Assign midpoint (or sample from distribution if known)
        sample["age"] = age_midpoint
        sample["age_source"] = "inferred_from_publication_PMID12345"

    # Assign sex stochastically based on distribution
    if not sample.get("sex"):
        sample["sex"] = "male" if random.random() < male_pct else "female"
        sample["sex_source"] = "stochastic_from_publication_PMID12345"

# Return enriched samples with output_key for metadata_store persistence
result = {
    'samples': samples,
    'output_key': 'pub_queue_PMID12345_samples_enriched',
    'stats': {
        'total_samples': len(samples),
        'enriched_disease': sum(1 for s in samples if s.get('disease_source')),
        'enriched_age': sum(1 for s in samples if s.get('age_source')),
        'enriched_sex': sum(1 for s in samples if s.get('sex_source'))
    }
}
""",
    workspace_key=None,  # Or specify existing key to update in-place
    persist=True,
    description="Propagate publication-level demographics to individual samples"
)
```

**Capability**: ✅ **POSSIBLE**
- `execute_custom_code` auto-injects workspace JSON files as Python dicts
- Can modify samples in-place
- Can return `{'samples': [...], 'output_key': '...'}` to persist to metadata_store
- Result is automatically stored and can be exported via `write_to_workspace`

**Gap**:
- ❌ NO direct "propagate_metadata_to_samples" tool (must write custom code)
- ⚠️ Stochastic sex assignment may not reflect true individual-level sex
- ⚠️ Manual per-study logic (not automated)

#### Step 4: Export enriched samples

**Tool**: `write_to_workspace`

```python
write_to_workspace(
    identifier="pub_queue_PMID12345_samples_enriched",
    workspace="metadata",
    output_format="csv",
    export_mode="rich"  # 28-column format with publication context
)
```

**Capability**: ✅ **POSSIBLE**
- `write_to_workspace` supports CSV export with schema-driven column ordering
- Enriched samples (from Step 3) are in `metadata_store['pub_queue_PMID12345_samples_enriched']`
- CSV export automatically adds publication context (source_doi, source_pmid, source_entry_id)
- Result: `workspace/metadata/pub_queue_PMID12345_samples_enriched_YYYY-MM-DD.csv`

**Gap**: None - this step works as expected

---

## 3. Gap Analysis

### Publication Text Access

**Current**:
- ✅ Can read: abstracts, cached publication metadata, methods sections (if cached by research_agent)
- ✅ Can access: workspace/literature/*.json files
- ❌ Cannot read: Full text PDF, PMC XML (unless research_agent caches it)
- ❌ Cannot parse: Supplementary files

**Gap Severity**: **MEDIUM**

**Recommended Fix**:
- **Option 1 (Preferred)**: Extend research_agent to cache full PMC XML or extracted methods in workspace
- **Option 2**: Add `fetch_pmc_fulltext` tool to metadata_assistant (violates separation of concerns)
- **Option 3 (Workaround)**: Use `get_content_from_workspace(level="methods")` if methods already cached

### Sample-Level Enrichment

**Current**:
- ✅ Can read: SRA sample metadata (raw)
- ✅ Can modify: Individual sample fields via `execute_custom_code`
- ✅ Can propagate: Publication-level metadata to samples via custom code
- ❌ Cannot do automatically: Requires manual per-study code

**Gap Severity**: **LOW**

**Recommended Fix**:
- **Option 1 (Preferred)**: Use `execute_custom_code` for manual review/enrichment (existing capability)
- **Option 2**: Add `enrich_samples_from_publication` dedicated tool (4-6 hour effort)
- **Option 3**: Build automated LLM-based extraction pipeline (separate project)

### Workflow Integration

**Current**:
- ✅ Can filter: samples by criteria
- ✅ Can validate: samples against schemas
- ✅ Can enrich: samples with publication context via `execute_custom_code`
- ✅ Can execute: custom enrichment logic per study
- ❌ Cannot automate: Requires manual LLM intervention per study

**Gap Severity**: **LOW** (for manual enrichment use case)

**Recommended Fix**:
- **Option 1 (Preferred)**: Document workflow in wiki for manual enrichment
- **Option 2**: Build LLM-guided extraction pipeline (research_agent + metadata_assistant collaboration)
- **Option 3**: Add template custom code snippets to agent system prompt

---

## 4. Recommended Tool Enhancements

### Option 1: Extend Existing Tools (MINIMAL EFFORT) ✅ **RECOMMENDED**

**Modify**: Documentation only

**Add to metadata_assistant system prompt**:
```markdown
## Manual Enrichment Workflow (Missing Metadata)

When users need to enrich samples with missing demographics (age, sex, disease):

1. **Read publication methods**:
   `get_content_from_workspace(identifier="publication_PMID12345", level="methods")`

2. **Extract demographics** (custom code per study):
   ```python
   execute_custom_code(
       python_code='''
import re
methods = ... # Read from workspace
age_match = re.search(r"aged (\\d+)-(\\d+)", methods)
age_midpoint = (int(age_match.group(1)) + int(age_match.group(2))) / 2
result = {"age_midpoint": age_midpoint}
       ''',
       description="Extract age from methods"
   )
   ```

3. **Propagate to samples**:
   ```python
   execute_custom_code(
       python_code='''
samples = aggregated_samples["samples"]
for s in samples:
    if not s.get("age"):
        s["age"] = 55  # From step 2
result = {"samples": samples, "output_key": "enriched_samples"}
       ''',
       description="Propagate age to samples"
   )
   ```

4. **Export**:
   `write_to_workspace(identifier="enriched_samples", workspace="metadata", output_format="csv")`
```

**Effort**: 30 minutes
**Impact**: Documents existing capability, enables user self-service

### Option 2: Add Sample Enrichment Tool (MEDIUM EFFORT)

**New tool**: `enrich_samples_from_publication`

```python
@tool
def enrich_samples_from_publication(
    sample_identifier: str,  # pub_queue_X_samples
    publication_identifier: str,  # publication_PMID12345
    enrichment_fields: str = "disease,age,sex",  # Comma-separated
    extraction_strategy: str = "auto"  # auto/manual/llm-guided
) -> str:
    """
    Enrich samples with publication-level demographics.

    Reads publication methods, extracts demographics using regex/LLM,
    propagates to samples missing those fields.

    Args:
        sample_identifier: Workspace key for samples (e.g., "pub_queue_X_samples")
        publication_identifier: Workspace key for publication (e.g., "publication_PMID12345")
        enrichment_fields: Fields to enrich (disease, age, sex, tissue)
        extraction_strategy: "auto" (regex), "manual" (user-provided), "llm-guided"

    Returns:
        Enrichment report with stats
    """
    # 1. Read publication methods
    methods = get_content_from_workspace(publication_identifier, level="methods")

    # 2. Extract demographics (regex + LLM)
    demographics = _extract_demographics_from_methods(methods, enrichment_fields)

    # 3. Read samples
    samples = get_content_from_workspace(sample_identifier)

    # 4. Propagate to samples
    enriched_samples = _propagate_demographics_to_samples(samples, demographics)

    # 5. Store enriched samples
    output_key = f"{sample_identifier}_enriched"
    data_manager.metadata_store[output_key] = enriched_samples

    return f"✓ Enriched {len(enriched_samples)} samples with {enrichment_fields}"
```

**Effort**: 4-6 hours
**Impact**: Direct workflow for manual enrichment (but still requires per-study tuning)

### Option 3: Leverage execute_custom_code (CURRENT STATE) ✅ **ALREADY EXISTS**

**Use**: `execute_custom_code` to write study-specific enrichment logic

**Workflow**:
1. User inspects publication methods
2. User writes custom Python code to extract demographics
3. User writes custom code to propagate to samples
4. User exports enriched samples

**Effort**: 0 hours (tool already exists)
**Impact**: Flexible per-study enrichment (requires user expertise)

**Example**:
```python
execute_custom_code(
    python_code='''
import re
import pandas as pd

# Step 1: Read publication methods
# (Assume research_agent cached this as "pub_queue_123_methods")
methods_text = pub_queue_123_methods.get("methods", "")

# Step 2: Extract demographics
age_match = re.search(r"patients aged (\\d+)-(\\d+)", methods_text.lower())
age_midpoint = (int(age_match.group(1)) + int(age_match.group(2))) / 2 if age_match else None

disease = "crc" if "colorectal cancer" in methods_text.lower() else "unknown"

# Step 3: Read samples
samples = pub_queue_123_samples["samples"]

# Step 4: Enrich samples
for s in samples:
    if not s.get("age"):
        s["age"] = age_midpoint
        s["age_source"] = "inferred_from_publication"
    if not s.get("disease"):
        s["disease"] = disease
        s["disease_source"] = "inferred_from_publication"

# Step 5: Return for persistence
result = {
    "samples": samples,
    "output_key": "pub_queue_123_samples_enriched",
    "stats": {
        "total": len(samples),
        "enriched_age": sum(1 for s in samples if s.get("age_source")),
        "enriched_disease": sum(1 for s in samples if s.get("disease_source"))
    }
}
''',
    description="Enrich PRJNA123 samples with CRC demographics from publication"
)
```

---

## 5. Test Protocol

### Step 1: Verify Tool Availability (COMPLETED)

✅ **Verified**: All tools exist in `metadata_assistant.py`

**Tool Inventory**:
- `get_content_from_workspace` - Lines 44-517 (workspace_tool.py)
- `write_to_workspace` - Lines 666-910 (workspace_tool.py)
- `execute_custom_code` - Lines 2044-2176 (metadata_assistant.py)
- `process_metadata_entry` - Lines 1088-1267 (metadata_assistant.py)
- `process_metadata_queue` - Lines 1270-1580 (metadata_assistant.py)

### Step 2: Test Publication Access (RECOMMENDED NEXT STEP)

**Test Case**: Can we read publication methods?

```python
# Via metadata_assistant agent
get_content_from_workspace(
    identifier="publication_PMID37123456",  # Replace with actual cached publication
    workspace="literature",
    level="methods"
)

# Expected: Methods section text as markdown
# Actual: (User should test with real cached publication)
```

**Pass Criteria**: Returns methods text from cached publication

### Step 3: Test Sample Modification (RECOMMENDED NEXT STEP)

**Test Case**: Can we modify and persist samples?

```python
execute_custom_code(
    python_code='''
# Read samples from workspace
samples = aggregated_filtered_samples["samples"]

# Modify first sample (test)
samples[0]["disease"] = "crc_enriched_manually"
samples[0]["enrichment_source"] = "manual_test_2025-12-03"

# Return for persistence
result = {
    "samples": samples,
    "output_key": "test_enriched_samples",
    "stats": {"count": len(samples)}
}
''',
    description="Test manual enrichment persistence"
)

# Then verify:
write_to_workspace(
    identifier="test_enriched_samples",
    workspace="metadata",
    output_format="csv"
)
```

**Pass Criteria**: CSV exported with enriched disease field

### Step 4: Test End-to-End Workflow (RECOMMENDED FOR USER)

**Test Case**: Full enrichment workflow on single study

**Steps**:
1. Select publication with known demographics in methods
2. Extract demographics using `execute_custom_code`
3. Propagate to samples
4. Export enriched samples
5. Verify CSV contains enriched fields

**Pass Criteria**: Exported CSV has enriched age/sex/disease fields with source annotations

---

## 6. Conclusions

### Summary

| Capability | Available | Gap Severity | Recommended Action |
|-----------|-----------|-------------|-------------------|
| Read publication methods | ✅ YES | None | Document workflow in wiki |
| Extract demographics | ✅ YES (via custom code) | LOW | Use `execute_custom_code` |
| Modify samples | ✅ YES | None | Use `execute_custom_code` |
| Persist enriched samples | ✅ YES | None | Use `write_to_workspace` |
| Automate across studies | ❌ NO | MEDIUM | Future: LLM-guided extraction pipeline |

### Final Verdict

✅ **metadata_assistant CAN manually enrich samples** using the 3-tool workflow:

1. **get_content_from_workspace** → Read publication methods
2. **execute_custom_code** → Extract demographics + propagate to samples
3. **write_to_workspace** → Export enriched samples

**Key Insight**: The `execute_custom_code` tool is the **critical enabler** for manual enrichment. It provides:
- Full workspace access (publications, samples, metadata)
- Arbitrary Python code execution (regex, pandas, numpy)
- Persistence to metadata_store
- Integration with notebook export system

### Limitations

1. **NOT scalable**: Requires manual per-study custom code (regex, extraction logic)
2. **NO full PDF access**: Limited to abstracts/methods cached by research_agent
3. **NO automation**: Requires LLM intervention per study
4. **Stochastic sex assignment**: May not reflect true individual-level sex

### Recommended Next Steps

**Immediate (0 hours)**:
1. ✅ Document manual enrichment workflow in wiki
2. ✅ Add example custom code snippets to metadata_assistant system prompt
3. ✅ Test workflow on 1-2 real studies to validate

**Short-term (4-6 hours)**:
- **Option A**: Add `enrich_samples_from_publication` dedicated tool (if frequent use case)
- **Option B**: Extend research_agent to cache full PMC methods sections (better separation of concerns)

**Long-term (separate project)**:
- Build LLM-guided extraction pipeline (research_agent + metadata_assistant collaboration)
- Train custom NER model for demographic extraction from methods sections
- Integrate with external demographic databases (e.g., dbGaP phenotype data)

---

## Appendix: Example Enrichment Script

### Full End-to-End Workflow (Copy-Paste Ready)

```python
# STEP 1: Read publication methods
methods_text = get_content_from_workspace(
    identifier="publication_PMID12345",
    workspace="literature",
    level="methods"
)

# STEP 2: Extract demographics using execute_custom_code
result = execute_custom_code(
    python_code='''
import re

# Extract age range
age_match = re.search(r"aged (\\d+)-(\\d+) years", methods_text.lower())
age_midpoint = None
if age_match:
    age_min = int(age_match.group(1))
    age_max = int(age_match.group(2))
    age_midpoint = (age_min + age_max) / 2

# Extract sex distribution
sex_match = re.search(r"(\\d+)%?\\s+male.*?(\\d+)%?\\s+female", methods_text.lower())
male_pct = None
if sex_match:
    male_pct = int(sex_match.group(1)) / 100

# Extract disease
disease = None
if "colorectal cancer" in methods_text.lower() or "crc" in methods_text.lower():
    disease = "crc"
elif "ulcerative colitis" in methods_text.lower():
    disease = "uc"
elif "crohn" in methods_text.lower():
    disease = "cd"

result = {
    "age_midpoint": age_midpoint,
    "male_pct": male_pct,
    "disease": disease
}
''',
    description="Extract demographics from publication methods"
)

# STEP 3: Propagate to samples
enriched_result = execute_custom_code(
    python_code='''
import random

# Get demographics from previous step
age_midpoint = 55  # From Step 2
male_pct = 0.6  # From Step 2
disease = "crc"  # From Step 2

# Read samples
samples = pub_queue_PMID12345_samples["samples"]

# Enrich each sample
for sample in samples:
    # Propagate disease
    if not sample.get("disease"):
        sample["disease"] = disease
        sample["disease_source"] = "inferred_from_publication_PMID12345"

    # Propagate age
    if not sample.get("age"):
        sample["age"] = age_midpoint
        sample["age_source"] = "inferred_from_publication_PMID12345"

    # Assign sex stochastically
    if not sample.get("sex"):
        sample["sex"] = "male" if random.random() < male_pct else "female"
        sample["sex_source"] = "stochastic_from_publication_PMID12345"

# Return enriched samples
result = {
    "samples": samples,
    "output_key": "pub_queue_PMID12345_samples_enriched",
    "stats": {
        "total_samples": len(samples),
        "enriched_disease": sum(1 for s in samples if "disease_source" in s),
        "enriched_age": sum(1 for s in samples if "age_source" in s),
        "enriched_sex": sum(1 for s in samples if "sex_source" in s)
    }
}
''',
    persist=True,
    description="Propagate publication demographics to samples"
)

# STEP 4: Export enriched samples
export_result = write_to_workspace(
    identifier="pub_queue_PMID12345_samples_enriched",
    workspace="metadata",
    output_format="csv",
    export_mode="rich"
)

# Result: workspace/metadata/pub_queue_PMID12345_samples_enriched_YYYY-MM-DD.csv
```

### Expected Output CSV Columns

```csv
run_accession,biosample,bioproject,organism,host,library_strategy,disease,disease_source,age,age_source,sex,sex_source,source_doi,source_pmid,source_entry_id,...
SRR001,SAMN001,PRJNA123,Gut metagenome,Homo sapiens,AMPLICON,crc,inferred_from_publication_PMID12345,55,inferred_from_publication_PMID12345,male,stochastic_from_publication_PMID12345,10.1038/s41586-022-05123-4,PMID12345,pub_queue_123,...
```

---

**Audit Completed**: 2025-12-03
**Auditor**: Claude (Sonnet 4.5)
**Reviewed Files**:
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/metadata_assistant.py` (2492 lines)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/workspace_tool.py` (1182 lines)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/services/execution/custom_code_execution_service.py` (300 lines reviewed)
- `/Users/tyo/GITHUB/omics-os/lobster/lobster/services/data_access/workspace_content_service.py` (400 lines reviewed)

**Next Steps**: Test workflow on real publication queue data
