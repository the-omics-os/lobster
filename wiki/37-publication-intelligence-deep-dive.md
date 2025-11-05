# Publication Content Access & Two-Tier Architecture

**Version:** 2.3.0+
**Status:** Production-ready
**Implementation:** Phase 1-3 Complete (January 2025)

## Overview

The **UnifiedContentService** provides intelligent two-tier publication content access with webpage-first extraction strategy. This architecture replaced the monolithic PublicationIntelligenceService, delivering faster abstract access, better content quality, and cleaner code organization.

### What Changed?

**Before (PublicationIntelligenceService - Deprecated):**
- ❌ Single-tier access (always tried full PDF extraction)
- ❌ No quick abstract option (forced 2-5 second wait)
- ❌ PDF-first strategy (missed directly accessible webpage content)
- ❌ 4-layer deep call chain (Agent → Service → Assistant → Resolver)
- ❌ Split caching (violated DataManager-first principle)

**After (UnifiedContentService - v2.3.0+):**
- ✅ **Two-tier access:** Quick abstract (200-500ms) vs full content (2-8s)
- ✅ **Webpage-first strategy:** Extract Nature/Science/Cell Press directly
- ✅ **Clean architecture:** Direct provider delegation, no unnecessary layers
- ✅ **DataManager-first caching:** All caching through DataManagerV2
- ✅ **Docling integration:** Structure-aware PDF/webpage parsing
- ✅ **Automatic fallback:** Webpage → PDF → PyPDF2 graceful degradation

### Performance Impact

| Metric | Before (Single-Tier) | After (Two-Tier) | Improvement |
|--------|---------------------|------------------|-------------|
| **Quick Abstract** | N/A (not available) | 200-500ms | New capability |
| **Webpage Extraction** | N/A (PDF-only) | 2-5 seconds | New capability |
| **PDF Extraction** | 3-8 seconds | 3-8 seconds | Same (optimized internally) |
| **Cache Hit Time** | 100ms | <50ms | 2x faster |
| **User Experience** | Always wait for full content | Progressive disclosure | Much better |

---

## Architecture

### Two-Tier Access Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedContentService                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tier 1: Quick Abstract (Fast Path - 200-500ms)            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ get_quick_abstract(identifier)                     │    │
│  │         ↓                                          │    │
│  │ AbstractProvider (NCBI E-utilities)                │    │
│  │         ↓                                          │    │
│  │ PublicationMetadata (title, authors, abstract)    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  Tier 2: Full Content (Deep Path - 2-8 seconds)            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ get_full_content(identifier, prefer_webpage=True)  │    │
│  │         ↓                                          │    │
│  │ 1. Check DataManager cache                         │    │
│  │         ↓                                          │    │
│  │ 2a. Detect identifier (DOI/PMID vs direct URL)    │    │
│  │         ↓                                          │    │
│  │ 2b. Auto-resolve DOI/PMID → URL (PublicationResolver) │    │
│  │         ↓                                          │    │
│  │ 3a. Webpage-first (publisher sites):              │    │
│  │      WebpageProvider → DoclingService              │    │
│  │         ↓                                          │    │
│  │ 3b. Auto-format detection (all URLs):             │    │
│  │      DoclingService (HTML/PDF auto-detect)        │    │
│  │         ↓                                          │    │
│  │ 4. Cache result in DataManager                     │    │
│  │         ↓                                          │    │
│  │ 5. Return ContentResult (markdown, tables, etc.)   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### System Design

```
User → Research Agent
           ↓
    UnifiedContentService
           ↓
    ┌──────┴──────────┐
    ↓                 ↓
AbstractProvider   DoclingService (shared foundation)
    ↓                 ↓
NCBI E-utilities   ┌──┴───────────┐
                   ↓              ↓
           WebpageProvider   PDFProvider
                   ↓              ↓
           Webpage parsing   PDF parsing
                   ↓              ↓
                   DataManagerV2 (cache)
                          ↓
                   W3C-PROV provenance
```

### Key Components

#### 1. **UnifiedContentService** (Coordination Layer)

**Location:** `lobster/tools/unified_content_service.py`

**Responsibilities:**
- Two-tier access coordination
- Provider delegation (Abstract, Webpage, Docling)
- DataManager-first caching
- Error handling and fallback strategy

**Public API:**
```python
class UnifiedContentService:
    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager
        self.resolver = PublicationResolver()
        self.abstract_provider = AbstractProvider()
        self.webpage_provider = WebpageProvider()
        self.docling_service = DoclingService(data_manager)

    def get_quick_abstract(
        self,
        identifier: str,
        force_refresh: bool = False
    ) -> PublicationMetadata:
        """Tier 1: Fast NCBI abstract (no PDF download)."""

    def get_full_content(
        self,
        identifier: str,
        prefer_webpage: bool = True
    ) -> ContentResult:
        """Tier 2: Deep extraction (webpage → PDF fallback)."""

    def extract_methods_section(
        self,
        content_result: ContentResult,
        llm: Optional[Any] = None
    ) -> MethodsExtraction:
        """Extract computational methods from retrieved content."""
```

#### 2. **AbstractProvider** (Tier 1: Fast Path)

**Location:** `lobster/tools/providers/abstract_provider.py`

**Responsibilities:**
- Quick NCBI E-utilities abstract retrieval
- No PDF download (metadata only)
- Fast response (200-500ms)

**Performance:**
- Cache hit: <50ms
- Cache miss: 200-500ms (NCBI API call)
- Use case: Quick paper overview before deciding on full extraction

#### 3. **WebpageProvider** (Tier 2: Webpage-first)

**Location:** `lobster/tools/providers/webpage_provider.py`

**Responsibilities:**
- Direct webpage extraction (Nature, Science, Cell Press, PLOS)
- Delegates to DoclingService for parsing
- Automatic fallback to PDF if webpage extraction fails

**Supported Publishers:**
- Nature family (nature.com)
- Science family (science.org)
- Cell Press (cell.com)
- PLOS (plos.org)
- BioRxiv/MedRxiv (open preprint servers)

**Performance:**
- Webpage extraction: 2-5 seconds
- Fallback to PDF: +2-3 seconds
- Cache hit: <100ms

#### 4. **DoclingService** (Shared Foundation)

**Location:** `lobster/tools/docling_service.py`

**Responsibilities:**
- Structure-aware PDF/webpage parsing using Docling (MIT License, IBM Research)
- Table extraction as pandas DataFrames
- Formula detection and LaTeX formatting
- Smart image filtering (removes base64 bloat)
- Comprehensive retry logic with PyPDF2 fallback

**Docling Benefits:**
- ✅ Intelligent Methods section detection by keywords
- ✅ Complete section extraction (no arbitrary truncation)
- ✅ Table structure preservation
- ✅ Formula detection and LaTeX formatting
- ✅ Smart image filtering (40-60% Markdown reduction)
- ✅ Comprehensive error handling
- ✅ Automatic PyPDF2 fallback for incompatible PDFs

**Performance:**
| Metric | Value |
|--------|-------|
| **First parse** | 2-5 seconds (structure analysis) |
| **Cache hit** | <100ms (30x faster) |
| **Memory usage** | ~500MB peak |
| **Methods hit rate** | >90% (vs 30% with naive PyPDF2) |
| **Table extraction** | 80%+ of tables captured |

---

## Two-Tier Access Workflow

### User Scenario 1: Quick Abstract First

```python
from lobster.tools.unified_content_service import UnifiedContentService

service = UnifiedContentService(data_manager)

# Tier 1: Fast abstract (200-500ms)
abstract = service.get_quick_abstract("PMID:38448586")

print(f"Title: {abstract.title}")
print(f"Authors: {', '.join(abstract.authors[:3])}")
print(f"Abstract: {abstract.abstract[:200]}...")

# User decides: "This looks relevant, get full content"

# Tier 2: Deep extraction (2-8 seconds)
content = service.get_full_content("PMID:38448586")

print(f"Content type: {content.content_type}")  # 'webpage' or 'pdf'
print(f"Source: {content.source}")  # 'pmc', 'biorxiv', etc.
print(f"Full text: {len(content.markdown)} characters")
print(f"Tables: {len(content.tables)}")
```

**Benefits:**
- User gets quick overview (200-500ms)
- Can decide whether to invest 2-8 seconds for full content
- Progressive disclosure improves UX

### User Scenario 2: Direct Webpage Extraction

```python
# Nature article with directly accessible content
url = "https://www.nature.com/articles/s41586-025-09686-5"

# Automatic webpage-first extraction
content = service.get_full_content(url, prefer_webpage=True)

assert content.content_type == 'webpage'
assert content.source == 'publisher'
assert len(content.markdown) > 5000

# No PDF download needed! Direct extraction from webpage.
```

### User Scenario 3: DOI/PMID Auto-Resolution (v2.3+ Enhancement)

```python
# NEW in v2.3+: Direct DOI input - automatically resolved
content = service.get_full_content("10.1101/2024.08.29.610467")
# Internally: DOI → PublicationResolver → bioRxiv PDF URL → Extraction

# PMID input - automatically resolved
content = service.get_full_content("PMID:39370688")
# Internally: PMID → PublicationResolver → PMC URL → HTML extraction

# Complex PMC URLs that serve HTML - now handled correctly
content = service.get_full_content("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12496192/pdf/")
# Internally: Docling auto-detects HTML format (not PDF!)
```

**Key v2.3+ Improvements:**
- ✅ **DOI Auto-Detection:** Bare DOIs (e.g., `"10.1038/..."`) automatically detected
- ✅ **PMID Auto-Detection:** Both `"PMID:12345"` and `"12345"` formats supported
- ✅ **Format Auto-Detection:** Docling handles HTML/PDF detection (no rigid URL classification)
- ✅ **Robust Fallback:** PMC URLs serving HTML correctly processed
- ✅ **Graceful Errors:** Paywalled papers return helpful suggestions instead of crashing

**Before v2.3 (Broken):**
```python
# This would fail with FileNotFoundError
content = service.get_full_content("10.18632/aging.204666")  # ❌ Crashed
```

**After v2.3 (Fixed):**
```python
# Same input now works - auto-resolves and handles gracefully
try:
    content = service.get_full_content("10.18632/aging.204666")  # ✅ Works
except PaywalledError as e:
    print(f"Paper is paywalled: {e.suggestions}")  # ✅ Helpful guidance
```

**Benefits:**
- Faster than PDF extraction (2-5s vs 3-8s)
- Better content quality (structured HTML vs extracted PDF)
- Respects publisher access (uses public webpage)

### User Scenario 3: PDF Fallback

```python
# bioRxiv preprint (PDF only)
url = "https://biorxiv.org/content/10.1101/2024.01.001.full.pdf"

content = service.get_full_content(url, prefer_webpage=True)

# Automatically detected PDF URL, skipped webpage extraction
assert content.content_type == 'pdf'
assert content.source == 'biorxiv'

# Still gets full Docling benefits:
print(f"Tables: {len(content.tables)}")
print(f"Formulas: {len(content.formulas)}")
print(f"Software detected: {content.software_mentioned}")
```

**Benefits:**
- Automatic PDF detection (no unnecessary webpage attempt)
- Full Docling structure-aware parsing
- PyPDF2 fallback if Docling fails

---

## API Reference

### Tier 1: Quick Abstract

#### `get_quick_abstract()`

Fast abstract retrieval via NCBI E-utilities (no PDF download).

**Signature:**
```python
def get_quick_abstract(
    self,
    identifier: str,
    force_refresh: bool = False
) -> PublicationMetadata
```

**Parameters:**
- `identifier` (str): PMID, DOI, or publication ID
- `force_refresh` (bool): Bypass cache if True (default: False)

**Returns:**
```python
PublicationMetadata(
    pmid: Optional[str],
    doi: Optional[str],
    title: str,
    authors: List[str],
    journal: Optional[str],
    published: Optional[str],
    abstract: str,
    keywords: List[str],
    source: Literal['pubmed', 'biorxiv', 'medrxiv']
)
```

**Example:**
```python
service = UnifiedContentService(data_manager)

# Quick abstract for decision-making
abstract = service.get_quick_abstract("PMID:38448586")

print(f"Title: {abstract.title}")
print(f"Authors: {', '.join(abstract.authors)}")
print(f"Abstract: {abstract.abstract}")

# Check relevance before full extraction
if "single-cell" in abstract.abstract.lower():
    # Proceed to full content
    content = service.get_full_content(abstract.pmid)
```

**Performance:**
- Cache hit: <50ms
- Cache miss: 200-500ms
- Use case: Quick screening of multiple papers

### Tier 2: Full Content

#### `get_full_content()`

Deep content extraction with webpage-first strategy.

**Signature:**
```python
def get_full_content(
    self,
    identifier: str,
    prefer_webpage: bool = True
) -> ContentResult
```

**Parameters:**
- `identifier` (str): **Auto-detects and resolves multiple formats:**
  - **Bare DOI:** `"10.1101/2024.08.29.610467"` → Auto-resolved to bioRxiv
  - **DOI with prefix:** `"DOI:10.1038/s41586-025-09686-5"` → Auto-resolved
  - **PMID:** `"PMID:39370688"` or `"39370688"` → Auto-resolved to PMC
  - **Direct URL:** `"https://www.nature.com/articles/..."` → Used directly
  - **PMC URL:** `"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC..."` → Format auto-detected
- `prefer_webpage` (bool): Try webpage before PDF (default: True)

**Resolution Strategy (v2.3+ Enhanced):**
1. Check DataManager cache
2. **Auto-detect identifier type** (DOI, PMID, or direct URL)
3. **If DOI/PMID:** Use PublicationResolver to resolve to accessible URL
   - Handles paywalled papers gracefully (returns suggestions, no crash)
4. **If prefer_webpage:** Try WebpageProvider for publisher sites
   - Nature, Science, Cell Press direct extraction
5. **Format auto-detection:** Docling determines HTML vs PDF content
   - No rigid URL classification (handles PMC HTML correctly)
6. **Graceful fallback:** HTML → PDF → PyPDF2 if needed
7. Cache result in DataManager
8. Return ContentResult

**Returns:**
```python
ContentResult(
    identifier: str,
    content_type: Literal['webpage', 'pdf', 'abstract'],
    markdown: str,  # Clean markdown from Docling
    source: Literal['pmc', 'biorxiv', 'medrxiv', 'publisher', 'webpage'],
    metadata: Optional[PublicationMetadata],
    extraction_timestamp: datetime,

    # Docling-specific enrichments
    tables: List[Dict[str, Any]],
    formulas: List[str],
    software_mentioned: List[str]
)
```

**Examples (v2.3+ Auto-Resolution):**

```python
service = UnifiedContentService(data_manager)

# Example 1: Direct DOI input (NEW - automatically resolved)
content = service.get_full_content("10.1101/2024.08.29.610467")
# System logs: "Detected identifier (DOI), resolving to URL..."
# System logs: "Resolved to: https://www.biorxiv.org/content/10.1101/2024.08.29.610467.full.pdf"
print(f"Content type: {content.content_type}")  # 'pdf' (auto-detected)
print(f"Source: {content.source}")  # 'biorxiv'

# Example 2: PMID input (NEW - automatically resolved)
content = service.get_full_content("PMID:39370688")
# System logs: "Resolved to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC.../pdf/"
# System logs: "detected formats: [<InputFormat.HTML: 'html'>]" (auto-detected!)
print(f"Content type: {content.content_type}")  # 'html' (auto-detected)
print(f"Source: {content.source}")  # 'pmc'

# Example 3: Direct URL (existing behavior maintained)
content = service.get_full_content(
    "https://www.nature.com/articles/s41586-025-09686-5",
    prefer_webpage=True
)
print(f"Content type: {content.content_type}")  # 'webpage'
print(f"Source: {content.source}")  # 'publisher'

# Example 4: Paywalled DOI (NEW - graceful handling)
try:
    content = service.get_full_content("10.18632/aging.204666")
except PaywalledError as e:
    print(f"Paper is paywalled: {e.suggestions}")
    # Suggests: try institutional access, check for preprints, etc.

# Access extracted content
print(f"Full text: {len(content.markdown)} characters")
print(f"Tables: {len(content.tables)}")
print(f"Software mentioned: {content.software_mentioned}")
```

**Performance:**
- Cache hit: <100ms
- Webpage extraction: 2-5 seconds
- PDF extraction: 3-8 seconds
- Retry overhead: +2 seconds per retry

### Methods Extraction

#### `extract_methods_section()`

Extract computational methods from retrieved content using LLM.

**Signature:**
```python
def extract_methods_section(
    self,
    content_result: ContentResult,
    llm: Optional[Any] = None,
    include_tables: bool = True
) -> MethodsExtraction
```

**Parameters:**
- `content_result` (ContentResult): Result from `get_full_content()`
- `llm` (Optional[Any]): Custom LLM instance (uses default if None)
- `include_tables` (bool): Include parameter tables in context (default: True)

**Returns:**
```python
MethodsExtraction(
    software_used: List[str],
    parameters: Dict[str, Any],
    statistical_methods: List[str],
    data_sources: List[str],
    sample_sizes: Dict[str, str],
    normalization_methods: List[str],
    quality_control: List[str],
    content_source: ContentResult,
    extraction_confidence: float
)
```

**Example:**
```python
# Get full content
content = service.get_full_content("PMID:38448586")

# Extract methods with LLM
methods = service.extract_methods_section(
    content,
    include_tables=True
)

print("Software used:", methods.software_used)
# ['Scanpy', 'Seurat', 'DESeq2']

print("Parameters:", methods.parameters)
# {'min_genes': 200, 'max_mt_pct': 5, 'resolution': 0.8}

print("Statistical methods:", methods.statistical_methods)
# ['Wilcoxon rank-sum test', 'Benjamini-Hochberg FDR']

print("Confidence:", methods.extraction_confidence)
# 0.92
```

---

## DataManager-First Caching

All caching goes through DataManagerV2 (architectural requirement).

### Cache Methods

#### `cache_publication_content()`

Cache publication content with provenance tracking.

**Signature:**
```python
def cache_publication_content(
    identifier: str,
    content: ContentResult
) -> None
```

**Behavior:**
1. Store in memory cache (`self._publication_cache`)
2. Log as tool usage for W3C-PROV provenance
3. Persist to workspace (`~/.lobster/literature_cache/{identifier}.json`)

**Example:**
```python
# UnifiedContentService automatically caches
content = service.get_full_content("PMID:38448586")

# Behind the scenes:
# data_manager.cache_publication_content("PMID:38448586", content)

# Subsequent calls are fast (<100ms)
cached = service.get_full_content("PMID:38448586")  # Cache hit
```

#### `get_cached_publication()`

Retrieve cached content.

**Signature:**
```python
def get_cached_publication(identifier: str) -> Optional[ContentResult]
```

**Behavior:**
1. Check memory cache first
2. Check persistent cache (`~/.lobster/literature_cache/`)
3. Return None if cache miss

**Cache Location:**
```
~/.lobster/literature_cache/
├── PMID:38448586.json
├── 10.1101_2024.01.001.json
└── nature_s41586-025-09686-5.json
```

**Cache Format:**
```json
{
  "identifier": "PMID:38448586",
  "content_type": "pdf",
  "markdown": "# Methods\n...",
  "source": "pmc",
  "metadata": { ... },
  "tables": [ ... ],
  "formulas": [ ... ],
  "software_mentioned": ["Scanpy", "Seurat"],
  "extraction_timestamp": "2025-01-02T10:30:00Z"
}
```

---

## Integration with Research Agent

The Research Agent provides two-tier publication access tools.

### Tool 1: `get_quick_abstract()`

**Description:** Quick abstract retrieval for paper screening (200-500ms).

**Usage:**
```python
# Research Agent tool
@tool
def get_quick_abstract(identifier: str) -> str:
    """
    Get quick abstract for paper screening (no PDF download).

    Fast path for checking if paper is relevant before full extraction.
    Returns title, authors, abstract, keywords.

    Use this when:
    - User asks to "check if paper X is relevant"
    - Screening multiple papers quickly
    - Getting basic information before full extraction
    """
    abstract = abstract_provider.get_quick_abstract(identifier)
    return formatted_abstract
```

**Example workflow:**
```
User: "Is PMID:38448586 about single-cell RNA-seq?"

Agent: [Uses get_quick_abstract]
"Yes! Title: 'Single-cell transcriptomics reveals...'
Abstract mentions single-cell RNA-seq with Scanpy.
Would you like me to extract full methods?"

User: "Yes, get full content"

Agent: [Uses get_publication_overview]
"Full content extracted. Found Methods section with..."
```

### Tool 2: `get_publication_overview()`

**Description:** Deep content extraction with webpage-first strategy (2-8 seconds).

**Usage:**
```python
@tool
def get_publication_overview(
    identifier: str,
    include_methods: bool = True
) -> str:
    """
    Get full publication content with webpage-first extraction.

    Deep path for comprehensive content analysis.
    Returns full text, tables, formulas, extracted methods.

    Use this when:
    - User needs full publication content
    - Extracting computational methods
    - Analyzing tables or formulas
    - User provides webpage URL (Nature, Science, etc.)
    """
    service = UnifiedContentService(data_manager)
    content = service.get_full_content(identifier, prefer_webpage=True)

    if include_methods:
        methods = service.extract_methods_section(content)
        return formatted_with_methods

    return formatted_content
```

**Example workflows:**

**A. DOI Auto-Resolution (v2.3+ Enhancement):**
```
User: "Extract methods from 10.1101/2024.08.29.610467"

Agent: [DOI detected → PublicationResolver → bioRxiv URL → extraction]
"✅ Resolved DOI to bioRxiv PDF (automatic detection)

**Source:** bioRxiv preprint
**Content type:** PDF (auto-detected)
**Tables found:** 2
**Software detected:** Scanpy, Seurat, CellRanger

**Methods:**
- Software: Scanpy v1.9.1, Seurat v5.0.1
- Parameters: min_genes=200, max_mt_pct=15%
- Normalization: SCTransform with default parameters
- Clustering: Leiden algorithm, resolution=0.5"
```

**B. PMID Auto-Resolution (v2.3+ Enhancement):**
```
User: "Extract methods from PMID:39370688"

Agent: [PMID detected → PublicationResolver → PMC URL → HTML extraction]
"✅ Resolved PMID to PMC article (automatic detection)

**Source:** PMC Open Access
**Content type:** HTML (auto-detected)
**Extraction time:** 1.5 seconds
**Software detected:** R, DESeq2, EdgeR

**Methods:**
- Platform: RNA-seq analysis using R 4.3
- Differential expression: DESeq2 with default parameters
- Multiple testing correction: Benjamini-Hochberg (FDR < 0.05)
- Pathway analysis: GSEA with MSigDB Hallmark gene sets"
```

**C. Direct URL (Legacy Behavior Maintained):**
```
User: "Extract methods from https://www.nature.com/articles/s41586-025-09686-5"

Agent: [Direct URL → webpage-first extraction]
"Extracted content from Nature webpage (no PDF needed).

**Content type:** webpage
**Tables found:** 3
**Software detected:** Scanpy, Seurat, CellRanger"
```

---

## Performance Benchmarks

### Tier 1: Quick Abstract

| Operation | Time | Use Case |
|-----------|------|----------|
| Cache hit | <50ms | Repeated access |
| Cache miss (NCBI) | 200-500ms | First access |
| Screening 10 papers | 2-5 seconds | Literature review |

### Tier 2: Full Content

| Content Type | First Access | Cache Hit | Notes |
|-------------|--------------|-----------|-------|
| **Webpage (Nature)** | 2-5 seconds | <100ms | Direct extraction |
| **Webpage (Science)** | 2-5 seconds | <100ms | Direct extraction |
| **PDF (bioRxiv)** | 3-8 seconds | <100ms | Docling parsing |
| **PDF (PMC)** | 3-8 seconds | <100ms | Docling parsing |

### Memory Usage

| Component | Memory |
|-----------|--------|
| UnifiedContentService | ~50MB |
| AbstractProvider | ~20MB |
| WebpageProvider | ~100MB |
| DoclingService | ~500MB (peak during parsing) |
| After gc.collect() | ~200MB |

---

## Troubleshooting

### Issue: Slow First Access

**Symptom:** First call to `get_full_content()` takes 3-8 seconds.

**Explanation:** This is expected behavior for deep content extraction:
- Docling structure-aware parsing: 2-5 seconds
- Table extraction: +1-2 seconds
- Formula detection: +0.5 seconds

**Solution:** Use two-tier pattern:
1. Start with `get_quick_abstract()` (200-500ms)
2. Only call `get_full_content()` if user needs full content
3. Subsequent calls are fast (<100ms from cache)

### Issue: Webpage Extraction Failed

**Symptom:**
```
WARNING: Webpage extraction failed, falling back to PDF
```

**Causes:**
1. Paywalled content (not openly accessible)
2. Publisher uses non-standard HTML structure
3. Network timeout or rate limiting

**Solution:** Automatic fallback to PDF extraction (no action needed)

**Check fallback:**
```python
content = service.get_full_content(url)
if content.content_type == 'pdf':
    print("Used PDF fallback (webpage extraction failed)")
```

### Issue: Methods Section Not Found

**Symptom:**
```
WARNING: No Methods section found with default keywords
```

**Solution 1:** Check if paper uses non-standard section names:
```python
# Use custom keywords via DoclingService
content = service.get_full_content(url)
# Methods extraction handles this internally
```

**Solution 2:** Inspect full content:
```python
content = service.get_full_content(url)
print(content.markdown[:1000])  # Check structure
```

### Issue: PaywalledError

**Symptom:**
```
PaywalledError: Paper PMID:12345678 is paywalled
```

**Cause:** Paper not openly accessible

**Solutions:**
1. Check if paper has preprint version (bioRxiv/medRxiv)
2. Use institutional access via VPN
3. Request paper from authors via ResearchGate
4. Use `get_quick_abstract()` for basic information

---

## Best Practices

### 1. Two-Tier Pattern

**Always start with quick abstract:**
```python
# ✅ GOOD: Two-tier approach
abstract = service.get_quick_abstract("PMID:38448586")
if "single-cell" in abstract.abstract.lower():
    content = service.get_full_content("PMID:38448586")

# ❌ BAD: Skip quick check
content = service.get_full_content("PMID:38448586")  # Wastes 2-8 seconds if irrelevant
```

### 2. Webpage-First Preference

**Use webpage-first for supported publishers:**
```python
# ✅ GOOD: Webpage-first (faster, better quality)
content = service.get_full_content(nature_url, prefer_webpage=True)

# ❌ BAD: Force PDF
content = service.get_full_content(nature_url, prefer_webpage=False)
```

### 3. Cache Awareness

**Leverage caching for repeated access:**
```python
# First call: 3-8 seconds (cache miss)
content1 = service.get_full_content("PMID:38448586")

# Second call: <100ms (cache hit)
content2 = service.get_full_content("PMID:38448586")  # Same paper, fast
```

### 4. Error Handling

**Handle PaywalledError gracefully:**
```python
from lobster.tools.unified_content_service import PaywalledError

try:
    content = service.get_full_content(identifier)
except PaywalledError as e:
    print(f"Paywalled: {e.suggestions}")
    # Fallback: Try quick abstract
    abstract = service.get_quick_abstract(identifier)
```

### 5. Batch Processing

**Use quick abstract for screening:**
```python
relevant_papers = []

for pmid in candidate_pmids:
    # Fast screening (200-500ms each)
    abstract = service.get_quick_abstract(pmid)

    if matches_criteria(abstract):
        relevant_papers.append(pmid)

# Deep extraction only for relevant papers
for pmid in relevant_papers:
    content = service.get_full_content(pmid)
    # ... analyze content ...
```

---

## Future Enhancements

### Short-term (Planned v2.4)

**Additional Provider Support:**
- arXiv LaTeX source extraction
- PubMed Central XML parsing
- Springer/Elsevier APIs (requires authentication)

### Medium-term (Planned v2.5)

**Enhanced Caching:**
- Distributed caching (Redis support)
- Cache warming for frequently accessed papers
- Automatic cache invalidation on paper updates

### Long-term (Planned v3.0)

**Multi-Modal Content:**
- Figure extraction and understanding
- Chemical structure parsing
- Supplementary data integration

---

## Version History

**v2.3.0 (January 2025):**
- ✅ Phase 1: Two-tier access architecture
- ✅ Phase 1: AbstractProvider (quick abstract)
- ✅ Phase 1: WebpageProvider (webpage-first extraction)
- ✅ Phase 2: DataManager-first caching
- ✅ Phase 2: MetadataValidationService extraction
- ✅ Phase 3: UnifiedContentService (coordination layer)
- ✅ Phase 3: PublicationIntelligenceService deletion
- Deprecated: PublicationIntelligenceService (use UnifiedContentService)
- Added: DoclingService (shared PDF/webpage foundation)
- Enhanced: Research Agent two-tier tools

**v2.2.0 (November 2024):**
- Initial: PublicationIntelligenceService with Docling
- Initial: Structure-aware PDF parsing
- Initial: Document caching system

---

## References

- **Docling Documentation:** https://docling-project.github.io/
- **Docling GitHub:** https://github.com/DS4SD/docling
- **Technical Paper:** https://arxiv.org/pdf/2408.09869
- **UnifiedContentService API:** See [16-services-api.md](16-services-api.md)
- **Research Agent:** See [15-agents-api.md](15-agents-api.md)

---

**Next Steps:**
- Review [16-services-api.md](16-services-api.md) for detailed API documentation
- See [15-agents-api.md](15-agents-api.md) for Research Agent integration
- Check [28-troubleshooting.md](28-troubleshooting.md) for common issues
- Explore [06-data-analysis-workflows.md](06-data-analysis-workflows.md) for workflow examples
