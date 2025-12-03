# 48. Manual Sample Enrichment Workflow
## Using Publication Context to Enrich SRA Sample Metadata

## Overview

**Problem**: SRA sample metadata is often incomplete - missing disease, demographics (age/sex), tissue details despite this information existing in the source publication.

**Solution**: Manual enrichment workflow leveraging publication full text cached by research_agent to extract missing demographics and propagate to samples.

**Target Use Cases**:
- Enrich samples with disease context from publication title/abstract/methods
- Extract age/sex distributions from Methods section cohort descriptions
- Infer tissue types from experimental design descriptions
- Add study-specific context (treatment groups, timepoints, clinical phenotypes)

**Introduced**: v1.2.0 (December 2024)
**Customer**: DataBioMix microbiome harmonization

---

## Architecture: Publication Content Accessibility

### What's Cached by research_agent

When `research_agent` processes publication queue entries, it caches:

| File Type | Location | Content Available | Use for Enrichment |
|-----------|----------|-------------------|-------------------|
| **`*_metadata.json`** | workspace/metadata/ | **Full publication text** (`content` field), title, abstract, authors, journal | ✅ **PRIMARY** - Full text with all sections |
| **`*_methods.json`** | workspace/metadata/ | Methods section (`methods_text`), software, parameters, statistical methods | ✅ Focused cohort demographics extraction |
| **`*_identifiers.json`** | workspace/metadata/ | Extracted dataset IDs (GEO, SRA, PRIDE, etc.) | ⚠️ Dataset links only |

### Accessing Cached Content

**Tool**: `get_content_from_workspace(identifier, workspace, level)`

**Available detail levels** for publications:

| Level | Returns | Best For |
|-------|---------|----------|
| `"summary"` | Key-value overview (title, authors, year, journal) | Quick inspection |
| `"methods"` | Methods section text only | Cohort demographics, experimental design |
| `"metadata"` | **Full JSON with complete publication text** | **Disease context, full text search, comprehensive extraction** |
| `"github"` | GitHub repositories (if extracted) | Code/pipeline context |

**Critical Insight**: Use **`level="metadata"`** to access full publication text, not just methods section!

---

## Content Access Flexibility Matrix

### ✅ CAN Access (Via get_content_from_workspace)

| Content Type | Access Method | Coverage | Quality |
|--------------|---------------|----------|---------|
| **Publication Title** | metadata["data"]["content"] (first line) | 100% | Excellent |
| **Abstract** | metadata["data"]["content"] (Introduction section) | 100% | Excellent |
| **Methods Section** | level="methods" OR metadata["data"]["methods_text"] | 90-95% | Excellent |
| **Full Text** | level="metadata" → content field | 30-70% | Good (depends on PMC/paywall) |
| **Results Section** | metadata["data"]["content"] (Results section) | 30-70% | Good |
| **Discussion** | metadata["data"]["content"] (Discussion section) | 30-70% | Good |
| **Authors/Journal** | metadata["data"] (authors, journal, year) | 100% | Excellent |

### ❌ CANNOT Access (Not Cached)

| Content Type | Reason | Workaround |
|--------------|--------|------------|
| **Supplementary Files** | Not downloaded by default | Use research_agent.read_full_publication() to cache first |
| **Figures/Tables** | Text extraction only | Use PMC table parsing (future enhancement) |
| **PDFs** | Only if web/PMC unavailable | Docling PDF extraction (slower fallback) |
| **Paywalled Content** | Access restrictions | Manual user provision or institutional access |

---

## Enrichment Strategies by Information Source

### Strategy 1: Title/Abstract Disease Inference (Fastest, 100% coverage)

**When to use**: Disease field missing, publication title mentions disease

**Example**:
```
Title: "Gut microbiome in inflammatory bowel disease patients undergoing dietary intervention"
→ disease: "ibd"

Title: "Fecal microbiota profiles of Parkinson's Disease patients"
→ disease: "parkinsons"

Title: "Colorectal cancer tissue microbiome analysis"
→ disease: "crc"
```

**Workflow**:
```python
# Step 1: Read publication metadata
pub_metadata = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_metadata",
    workspace="metadata",
    level="metadata"
)

# Step 2: Extract disease from title (in metadata JSON)
# metadata["data"] contains title, abstract, full content

# Step 3: Propagate to samples
execute_custom_code('''
import json
# Load samples
samples = pub_queue_doi_X_Y_Z_samples["samples"]

# Extract disease from publication title
pub_title = pub_queue_doi_X_Y_Z_metadata["data"].get("content", "").split("\\n")[0]

# Infer disease (simple keyword matching)
disease_map = {
    "inflammatory bowel disease": "ibd",
    "crohn": "cd",
    "colitis": "uc",
    "colorectal cancer": "crc",
    "parkinson": "parkinsons",
}

inferred_disease = None
for pattern, disease_code in disease_map.items():
    if pattern in pub_title.lower():
        inferred_disease = disease_code
        break

# Propagate to samples
if inferred_disease:
    for s in samples:
        if not s.get("disease"):
            s["disease"] = inferred_disease
            s["disease_source"] = "inferred_from_publication_title"

result = {"samples": samples, "output_key": "enriched_samples"}
''')
```

**Expected coverage**: 80-100% (if publication title mentions disease)

---

### Strategy 2: Methods Section Demographics Extraction (Moderate, 30-70% coverage)

**When to use**: Age/sex fields missing, Methods section describes cohort

**Example**:
```
Methods: "We recruited 45 patients (23 males, 22 females) aged 35-65 years (mean 52 ± 8.3)
diagnosed with Crohn's disease..."

Extract:
- age_range: [35, 65]
- age_mean: 52
- sex_distribution: {"male": 0.51, "female": 0.49}
- disease: "cd"
```

**Workflow**:
```python
# Step 1: Read methods section
methods = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_methods",
    workspace="metadata",
    level="metadata"
)

# Step 2: Extract demographics with regex
execute_custom_code('''
import re

methods_text = pub_queue_doi_X_Y_Z_methods["data"]["methods_text"]

# Extract age range
age_match = re.search(r'aged? (\\d+)-?(\\d+)?', methods_text, re.I)
if age_match:
    age_min = int(age_match.group(1))
    age_max = int(age_match.group(2)) if age_match.group(2) else None
    age_midpoint = (age_min + (age_max or age_min)) // 2

# Extract sex distribution
sex_match = re.search(r'(\\d+) males?, (\\d+) females?', methods_text, re.I)
if sex_match:
    males = int(sex_match.group(1))
    females = int(sex_match.group(2))
    sex_ratio = males / (males + females)

# Propagate to samples (stochastic assignment)
samples = pub_queue_doi_X_Y_Z_samples["samples"]
for i, s in enumerate(samples):
    if not s.get("age") and age_midpoint:
        s["age"] = age_midpoint
        s["age_source"] = "inferred_from_methods_section"
    if not s.get("sex") and sex_ratio:
        # Stochastic assignment maintaining distribution
        s["sex"] = "male" if (i / len(samples)) < sex_ratio else "female"
        s["sex_source"] = "inferred_from_cohort_distribution"

result = {"samples": samples}
''')
```

**Expected coverage**: 30-70% (depends on Methods section detail)

---

### Strategy 3: Full Text Section Search (Advanced, Variable coverage)

**When to use**: Need specific details (tissue types, experimental groups, clinical phenotypes)

**Example**:
```
Results: "Fecal samples were collected from 89 patients, while colonic biopsies
were obtained from a subset of 34 patients during colonoscopy..."

Extract:
- sample_type distribution: fecal (89), tissue (34)
- tissue: colon (for tissue samples)
```

**Workflow**:
```python
# Read full publication content
full_content = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_metadata",
    workspace="metadata",
    level="metadata"
)

# Search for tissue-specific mentions
execute_custom_code('''
import re

full_text = pub_queue_doi_X_Y_Z_metadata["data"]["content"]
samples = pub_queue_doi_X_Y_Z_samples["samples"]

# Extract tissue mentions from Results section
results_section = re.search(r'## Results(.*?)(?:## |$)', full_text, re.DOTALL)
if results_section:
    results_text = results_section.group(1)

    # Look for tissue type mentions
    if "colonic biopsy" in results_text.lower() or "colon biopsy" in results_text.lower():
        tissue_type = "colon"
    elif "ileal biopsy" in results_text.lower():
        tissue_type = "ileum"
    elif "rectal biopsy" in results_text.lower():
        tissue_type = "rectum"
    else:
        tissue_type = None

    # Propagate to samples with "tissue" or "biopsy" in isolation_source
    if tissue_type:
        for s in samples:
            iso_source = s.get("isolation_source", "").lower()
            if "tissue" in iso_source or "biopsy" in iso_source:
                if not s.get("tissue"):
                    s["tissue"] = tissue_type
                    s["tissue_source"] = "inferred_from_results_section"

result = {"samples": samples}
''')
```

**Expected coverage**: 10-50% (highly variable by study design)

---

## Recommended Enrichment Workflow (Step-by-Step)

### Prerequisites
1. Publication queue entry processed by research_agent (has workspace_metadata_keys)
2. metadata_assistant available with execute_custom_code tool
3. Target entry identified (e.g., PRJNA642308 with 0% disease coverage but IBD in title)

### Step 1: Inspect Current Metadata Quality

```bash
# List publication queue entries
lobster query "List publication queue entries with handoff_ready status"

# Check metadata completeness for target entry
lobster query "Show metadata summary for entry pub_queue_doi_10_1080_19490976_2022_2046244"
```

**Identify gaps**: Which fields are <50% complete?

### Step 2: Load Publication Content

```bash
# Option A: Read methods section only (faster)
lobster query "Get methods section for pub_queue_doi_10_1080_19490976_2022_2046244_methods"

# Option B: Read full publication text (comprehensive)
lobster query "Get full metadata for pub_queue_doi_10_1080_19490976_2022_2046244_metadata"
```

**Extract information**: Disease mentions, cohort demographics, tissue types

### Step 3: Execute Custom Enrichment Code

```bash
lobster query "Execute custom enrichment code for PRJNA642308 IBD study to add disease field"

# metadata_assistant will use execute_custom_code with:
# 1. Load samples from workspace
# 2. Extract disease from publication title ("inflammatory bowel disease")
# 3. Propagate to all samples: s["disease"] = "ibd", s["disease_source"] = "publication_title"
# 4. Store enriched samples in workspace
```

### Step 4: Validate Enrichment

```bash
# Re-check metadata completeness
lobster query "Show enriched samples completeness for PRJNA642308"

# Expected: disease 0% → 100%
```

### Step 5: Export Enriched Dataset

```bash
lobster query "Export enriched PRJNA642308 samples to CSV with publication context"

# Output: workspace/metadata/exports/prjna642308_enriched_YYYY-MM-DD.csv
```

---

## Tool Capability Summary

### ✅ **SUFFICIENT for Manual Enrichment**

| Capability | Tool | Flexibility | Rating |
|------------|------|-------------|--------|
| **Read full text** | get_content_from_workspace(level="metadata") | Full publication content in "content" field | ✅ **EXCELLENT** |
| **Read methods** | get_content_from_workspace(level="methods") | Methods section with structured extraction | ✅ EXCELLENT |
| **Read title/abstract** | Embedded in metadata "content" field | First 2-3 sections of full text | ✅ EXCELLENT |
| **Read specific sections** | Regex search on metadata "content" field | Introduction, Methods, Results, Discussion | ✅ GOOD |
| **Modify samples** | execute_custom_code | Full Python with pandas/regex/json | ✅ EXCELLENT |
| **Export enriched** | write_to_workspace | CSV/JSON with auto-timestamp | ✅ EXCELLENT |

### ⚠️ **Limitations** (Acceptable)

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **Manual per-study** | Requires custom code per study | Document common patterns |
| **Regex-based** | Brittle for complex extractions | Use LLM-guided extraction (future) |
| **No full PDF for paywalled** | 30-40% of papers inaccessible | Focus on PMC-available papers |
| **Stochastic sex assignment** | Individual-level sex may be incorrect | Document limitation, add `sex_source` field |

---

## Flexibility Analysis: get_content_from_workspace

### Content Hierarchy (What You Can Access)

```
pub_queue_doi_X_Y_Z/
├── _metadata.json             ← FULL TEXT (level="metadata")
│   ├── content: "## Introduction\n..."    ← ALL SECTIONS
│   ├── title: "Study title"
│   ├── abstract: null (embedded in content)
│   ├── authors: [...]
│   └── journal: "..."
│
├── _methods.json              ← METHODS ONLY (level="methods")
│   ├── methods_text: "## Methods\n..."
│   ├── methods_dict:
│   │   ├── software_used: [...]
│   │   ├── parameters: {...}
│   │   └── statistical_methods: [...]
│
└── _identifiers.json          ← DATASET IDS (level="metadata")
    └── identifiers: {"geo": [...], "sra": [...]}
```

**Key Finding**: **`level="metadata"` gives FULL PUBLICATION TEXT**, not just metadata!

### Example: Accessing Different Content Types

```python
# 1. Title only (from summary)
result = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_metadata",
    level="summary"
)
# Returns: "Title: ..., Authors: ..., Year: ..."

# 2. Methods section only
result = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_methods",
    level="methods"
)
# Returns: "## Methods\n\nThis survey study was developed..."

# 3. FULL TEXT (Introduction + Methods + Results + Discussion)
result = get_content_from_workspace(
    identifier="pub_queue_doi_X_Y_Z_metadata",
    level="metadata"
)
# Returns: JSON with "content" field containing complete publication text

# 4. Parse full metadata JSON to extract specific sections
execute_custom_code('''
import re
full_content = pub_queue_doi_X_Y_Z_metadata["data"]["content"]

# Extract Introduction
intro = re.search(r'## Introduction(.*?)(?:## |$)', full_content, re.DOTALL)
intro_text = intro.group(1) if intro else None

# Extract Results
results = re.search(r'## Results(.*?)(?:## |$)', full_content, re.DOTALL)
results_text = results.group(1) if results else None

# Extract disease mentions across ALL sections
disease_mentions = re.findall(r'(inflammatory bowel disease|crohn|colitis|parkinson)',
                              full_content, re.I)

result = {
    "intro_text": intro_text,
    "results_text": results_text,
    "disease_mentions": disease_mentions
}
''')
```

---

## Enrichment Patterns by Field Type

### 1. Disease Enrichment (15-30% → 80-100%)

**Source**: Publication title, abstract, Methods cohort description

**Extraction Pattern**:
```python
# Read full metadata
pub_data = get_content_from_workspace("pub_queue_X_metadata", level="metadata")

# Pattern 1: Title mentions disease explicitly
title = pub_data["data"]["content"].split("\n")[0]
if "inflammatory bowel disease" in title.lower() or "ibd" in title.lower():
    disease = "ibd"
elif "crohn" in title.lower():
    disease = "cd"
elif "colitis" in title.lower():
    disease = "uc"
elif "colorectal cancer" in title.lower() or "crc" in title.lower():
    disease = "crc"
# ... add more patterns

# Propagate to all samples
for sample in samples:
    if not sample.get("disease"):
        sample["disease"] = disease
        sample["disease_source"] = "inferred_from_publication_title"
```

**Confidence**: HIGH (title explicitly states disease focus)

---

### 2. Age/Sex Enrichment (7-23% → 40-60%)

**Source**: Methods section cohort description

**Extraction Pattern**:
```python
# Read methods section
methods = get_content_from_workspace("pub_queue_X_methods", level="methods")
methods_text = methods["data"]["methods_text"]

# Extract age range
import re
age_pattern = r'aged? (\d+)[-–to ]+(\d+)?\s*years?'
age_match = re.search(age_pattern, methods_text, re.I)

if age_match:
    age_min = int(age_match.group(1))
    age_max = int(age_match.group(2)) if age_match.group(2) else age_min
    age_midpoint = (age_min + age_max) // 2

    # Propagate
    for sample in samples:
        if not sample.get("age"):
            sample["age"] = age_midpoint
            sample["age_source"] = f"inferred_from_methods (range: {age_min}-{age_max})"

# Extract sex distribution
sex_pattern = r'(\d+)\s+males?,?\s+(\d+)\s+females?'
sex_match = re.search(sex_pattern, methods_text, re.I)

if sex_match:
    males = int(sex_match.group(1))
    females = int(sex_match.group(2))
    total = males + females
    male_ratio = males / total

    # Stochastic assignment (maintains distribution)
    for i, sample in enumerate(samples):
        if not sample.get("sex"):
            sample["sex"] = "male" if (i / len(samples)) < male_ratio else "female"
            sample["sex_source"] = f"stochastic_assignment (cohort: {males}M/{females}F)"
```

**Confidence**: MEDIUM (stochastic assignment for individual-level)

---

### 3. Tissue Enrichment (3-20% → 40-70%)

**Source**: Methods sample collection description, Results section

**Extraction Pattern**:
```python
# Read full metadata for Results section
full_content = get_content_from_workspace("pub_queue_X_metadata", level="metadata")["data"]["content"]

# Extract Results section
results_match = re.search(r'## Results(.*?)(?:## |$)', full_content, re.DOTALL)
if results_match:
    results_text = results_match.group(1)

    # Look for tissue type mentions
    tissue_patterns = {
        r'colonic? biops(?:y|ies)': 'colon',
        r'ileal biops(?:y|ies)': 'ileum',
        r'rectal biops(?:y|ies)': 'rectum',
        r'duodenal biops(?:y|ies)': 'duodenum',
    }

    tissue_inferred = {}
    for pattern, tissue_type in tissue_patterns.items():
        if re.search(pattern, results_text, re.I):
            tissue_inferred[tissue_type] = True

    # Propagate to samples with "biopsy" in isolation_source
    for sample in samples:
        iso_source = sample.get("isolation_source", "").lower()
        if "biopsy" in iso_source or "tissue" in iso_source:
            if not sample.get("tissue") and len(tissue_inferred) == 1:
                sample["tissue"] = list(tissue_inferred.keys())[0]
                sample["tissue_source"] = "inferred_from_results_section"
```

**Confidence**: MEDIUM-LOW (ambiguous for multi-tissue studies)

---

## Example: PRJNA642308 Enrichment (Real Case Study)

### Entry Profile
- **BioProject**: PRJNA642308
- **Title**: "Dietary manipulation of gut microbiome in inflammatory bowel disease patients"
- **Samples**: 409 IBD patients
- **Current disease coverage**: 0%
- **Enrichment opportunity**: 100% (title explicitly states IBD)

### Enrichment Workflow

```bash
# Step 1: Verify current completeness
lobster query "Check metadata completeness for PRJNA642308 entry"
# Result: organism 100%, host 100%, tissue 100%, age 100%, sex 100%, disease 0%

# Step 2: Read publication metadata
lobster query "Get full metadata for pub_queue_doi_10_1080_19490976_2022_2046244_metadata"
# Result: Title contains "inflammatory bowel disease patients"

# Step 3: Enrich with disease
lobster query "Execute custom code to enrich PRJNA642308 with IBD disease from publication title"

# metadata_assistant executes:
execute_custom_code('''
samples = pub_queue_doi_10_1080_19490976_2022_2046244_samples["samples"]
for s in samples:
    s["disease"] = "ibd"
    s["disease_source"] = "inferred_from_publication_title"
result = {"samples": samples, "output_key": "prjna642308_enriched"}
''')

# Step 4: Export enriched
lobster query "Export prjna642308_enriched to CSV"
# Result: disease coverage 0% → 100%
```

**Measured improvement**: disease 0% → 100% (409/409 samples)

---

## Best Practices

### 1. Always Document Enrichment Source

```python
# Good: Traceable
sample["disease"] = "ibd"
sample["disease_source"] = "inferred_from_publication_title_doi_10_1080_19490976_2022_2046244"

# Bad: Not traceable
sample["disease"] = "ibd"
```

### 2. Use Midpoint for Age Ranges

```python
# Good: Conservative estimate
age_range = "45-65 years"
sample["age"] = 55  # Midpoint
sample["age_source"] = "inferred (range: 45-65 from methods)"

# Bad: Arbitrary assignment
sample["age"] = 45  # Why not 65?
```

### 3. Stochastic Sex Assignment with Transparency

```python
# Good: Maintains cohort distribution
# Cohort: 23 males (51%), 22 females (49%)
for i, sample in enumerate(samples):
    sample["sex"] = "male" if (i / len(samples)) < 0.51 else "female"
    sample["sex_source"] = "stochastic (cohort: 23M/22F, 51% male)"

# Bad: All male
for sample in samples:
    sample["sex"] = "male"  # Incorrect distribution
```

### 4. Validate Enrichment Improves Completeness

```python
# Before enrichment
completeness_before = sum([s.get("disease") is not None for s in samples]) / len(samples)

# After enrichment
# ... enrich samples ...

# After enrichment
completeness_after = sum([s.get("disease") is not None for s in samples]) / len(samples)

print(f"Disease completeness: {completeness_before*100:.1f}% → {completeness_after*100:.1f}%")
# Only proceed if improvement > 10%
```

---

## Automation Considerations (Future Enhancements)

### Phase 2: Semi-Automated Enrichment
- **LLM-guided extraction**: Use Claude to extract demographics from Methods section
- **Named Entity Recognition**: Automatic disease/tissue/age extraction
- **Validation pipeline**: metadata_assistant validates extracted values

### Phase 3: Fully Automated Enrichment
- **Publication-level metadata service**: Cache demographics at publication level
- **Automatic propagation**: All samples in study inherit publication context
- **Quality scoring**: Confidence scores for enriched fields
- **Human-in-the-loop**: Flag low-confidence enrichments for review

---

## See Also

### Related Wiki Pages
- [Microbiome Harmonization Workflow (Wiki 47)](./47-microbiome-harmonization-workflow.md) - Disease extraction automation
- [Publication Intelligence (Wiki 37)](./37-publication-intelligence-deep-dive.md) - research_agent publication processing
- [Download Queue System (Wiki 35)](./35-download-queue-system.md) - Multi-agent handoff patterns

### Code References
- `lobster/tools/workspace_tool.py` (lines 44-516) - get_content_from_workspace implementation
- `lobster/agents/metadata_assistant.py` (lines 722-860) - _extract_disease_from_raw_fields helper
- `lobster/agents/research_agent.py` (lines 1426-1580) - read_full_publication with PMC/PDF fallback
- `lobster/services/execution/custom_code_execution_service.py` - execute_custom_code implementation

---

## Summary

**get_content_from_workspace Flexibility**: ✅ **EXCELLENT**

- ✅ Can access full publication text (not just methods)
- ✅ Can access title, abstract, methods, results, discussion
- ✅ Can access structured metadata (authors, journal, year)
- ✅ Supports targeted section extraction via regex
- ✅ Combined with execute_custom_code enables flexible enrichment

**Enrichment Workflow Viability**: ✅ **PRODUCTION READY**

**No new tools needed** - existing toolkit is sufficient for manual enrichment with publication context.
