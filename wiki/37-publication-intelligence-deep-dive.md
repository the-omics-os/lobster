# Publication Content Access & Provider Architecture

**Version:** 2.4.0+ (Phase 1-6 Refactoring Complete)
**Status:** Production-ready
**Implementation:** ContentAccessService with Provider Infrastructure (January 2025)

## Overview

The **ContentAccessService** provides intelligent publication and dataset access through a capability-based provider architecture. This system replaced the legacy PublicationService and UnifiedContentService, delivering modular provider infrastructure, three-tier content cascade, and comprehensive literature mining capabilities.

### What Changed?

**Before (UnifiedContentService - Phase 3, Archived):**
- ❌ Direct provider delegation without capability routing
- ❌ Manual provider selection logic in service code
- ❌ Limited to 3 providers (Abstract, PMC, Webpage)
- ❌ No dataset discovery capabilities
- ❌ No validation or metadata extraction tools

**After (ContentAccessService - Phase 2+, Current):**
- ✅ **Provider Registry:** Capability-based routing with priority system
- ✅ **5 Specialized Providers:** Abstract, PubMed, GEO, PMC, Webpage (with Docling)
- ✅ **10 Core Methods:** Discovery (3), Metadata (2), Content (3), System (1), Validation (1)
- ✅ **Three-Tier Cascade:** PMC XML → Webpage → PDF with automatic fallback
- ✅ **Dataset Integration:** GEO/SRA/PRIDE dataset discovery and validation
- ✅ **Session Caching:** DataManager-first with W3C-PROV provenance

### Performance Impact

| Metric | UnifiedContentService | ContentAccessService | Improvement |
|--------|---------------------|---------------------|-------------|
| **Abstract Retrieval** | 200-500ms (AbstractProvider) | 200-500ms (AbstractProvider) | Same (optimized path) |
| **PMC Full-Text** | 500ms-2s (PMCProvider) | 500ms-2s (PMCProvider priority) | Same (10x faster than HTML) |
| **Dataset Discovery** | N/A (not available) | 2-5s (GEOProvider) | New capability |
| **Literature Search** | N/A (not available) | 1-3s (PubMedProvider) | New capability |
| **Provider Selection** | Manual logic | Automatic routing | Better maintainability |
| **Extensibility** | Hard-coded providers | Registry-based | Easy to add providers |

---

## Architecture

### Capability-Based Provider System

```
┌─────────────────────────────────────────────────────────────┐
│                    ContentAccessService                     │
│                   (Coordination Layer)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  10 Core Methods:                                           │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Discovery (3):                                    │     │
│  │  - search_literature                              │     │
│  │  - discover_datasets                              │     │
│  │  - find_linked_datasets                           │     │
│  │                                                    │     │
│  │ Metadata (2):                                     │     │
│  │  - extract_metadata                               │     │
│  │  - validate_metadata                              │     │
│  │                                                    │     │
│  │ Content (3):                                      │     │
│  │  - get_abstract                                   │     │
│  │  - get_full_content                               │     │
│  │  - extract_methods                                │     │
│  │                                                    │     │
│  │ System (1):                                       │     │
│  │  - query_capabilities                             │     │
│  └───────────────────────────────────────────────────┘     │
│                         ↓                                   │
│                  ProviderRegistry                           │
│              (Capability-Based Routing)                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Provider Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Provider 1: AbstractProvider (Priority: 10)               │
│  └─ Capability: GET_ABSTRACT                               │
│     Performance: 200-500ms                                  │
│                                                             │
│  Provider 2: PubMedProvider (Priority: 10)                 │
│  └─ Capabilities: SEARCH_LITERATURE, FIND_LINKED_DATASETS, │
│                   EXTRACT_METADATA                          │
│     Performance: 1-3s                                       │
│                                                             │
│  Provider 3: GEOProvider (Priority: 10)                    │
│  └─ Capabilities: DISCOVER_DATASETS, EXTRACT_METADATA,     │
│                   VALIDATE_METADATA                         │
│     Performance: 2-5s                                       │
│                                                             │
│  Provider 4: PMCProvider (Priority: 10)                    │
│  └─ Capability: GET_FULL_CONTENT (PMC XML API)            │
│     Performance: 500ms-2s (PRIORITY PATH)                  │
│                                                             │
│  Provider 5: WebpageProvider (Priority: 50)                │
│  └─ Capabilities: GET_FULL_CONTENT (Webpage + PDF)        │
│     Performance: 2-8s (FALLBACK)                           │
│     Uses: DoclingService (internal composition)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   DataManagerV2                             │
│              (Session Caching + Provenance)                 │
└─────────────────────────────────────────────────────────────┘
```

### System Design

```
User → research_agent (10 tools)
           ↓
    ContentAccessService (10 methods)
           ↓
    ProviderRegistry (capability routing)
           ↓
    ┌──────┴───────────────────┐
    ↓         ↓         ↓       ↓         ↓
Abstract  PubMed    GEO     PMC    Webpage
Provider  Provider  Provider Provider Provider
    ↓         ↓         ↓       ↓         ↓
 NCBI     PubMed   GEO API  PMC XML  Docling
E-utils    API              API      Service
                                        ↓
                                  (Webpage + PDF)
```

### Key Components

#### 1. **ContentAccessService** (Coordination Layer)

**Location:** `lobster/tools/content_access_service.py`

**Responsibilities:**
- Method routing to appropriate providers via ProviderRegistry
- Capability-based provider selection
- DataManager-first caching coordination
- Error handling and fallback orchestration
- W3C-PROV provenance tracking
- Lightweight IR (Intermediate Representation) for non-exportable research operations

**Public API (10 Methods):**

**Discovery (3 methods):**
```python
def search_literature(
    self,
    query: str,
    max_results: int = 5,
    sources: Optional[list[str]] = None,
    filters: Optional[dict[str, any]] = None
) -> Tuple[str, Dict[str, Any], AnalysisStep]:
    """Search PubMed, bioRxiv, medRxiv for literature."""

def discover_datasets(
    self,
    query: str,
    dataset_type: "DatasetType",
    max_results: int = 5,
    filters: Optional[dict[str, str]] = None
) -> Tuple[str, Dict[str, Any], AnalysisStep]:
    """Search GEO, SRA, PRIDE for omics datasets."""

def find_linked_datasets(
    self,
    identifier: str,
    dataset_types: Optional[list["DatasetType"]] = None,
    include_related: bool = True
) -> str:
    """Find datasets linked to a publication."""
```

**Metadata (2 methods):**
```python
def extract_metadata(
    self,
    identifier: str,
    source: Optional[str] = None
) -> Union["PublicationMetadata", str]:
    """Extract publication/dataset metadata."""

def validate_metadata(
    self,
    dataset_id: str,
    required_fields: Optional[List[str]] = None,
    required_values: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.8
) -> str:
    """Validate dataset metadata completeness."""
```

**Content (3 methods):**
```python
def get_abstract(
    self,
    identifier: str,
    force_refresh: bool = False
) -> dict[str, any]:
    """Tier 1: Fast abstract retrieval (200-500ms)."""

def get_full_content(
    self,
    source: str,
    prefer_webpage: bool = True,
    keywords: Optional[list[str]] = None,
    max_paragraphs: int = 100,
    max_retries: int = 2
) -> dict[str, any]:
    """Tier 2: Full content with PMC-first cascade."""

def extract_methods(
    self,
    content_result: dict[str, any],
    llm: Optional[any] = None,
    include_tables: bool = True
) -> dict[str, any]:
    """Extract structured methods from content."""
```

**System (1 method):**
```python
def query_capabilities(self) -> str:
    """Query available providers and capabilities."""
```

#### 2. **ProviderRegistry** (Routing Layer)

**Location:** `lobster/tools/providers/provider_registry.py`

**Responsibilities:**
- Provider registration and lifecycle management
- Capability-based routing to best-fit provider
- Priority-based provider ordering
- Dataset type mapping to providers
- Capability matrix generation for debugging

**Key Methods:**
```python
def register_provider(self, provider: BaseProvider) -> None:
    """Register a provider with its capabilities."""

def get_providers_for_capability(
    self,
    capability: ProviderCapability
) -> List[BaseProvider]:
    """Get all providers supporting a capability (sorted by priority)."""

def get_provider_for_dataset_type(
    self,
    dataset_type: DatasetType
) -> Optional[BaseProvider]:
    """Get provider for specific dataset type."""

def get_capability_matrix(self) -> str:
    """Generate debug matrix of providers and capabilities."""
```

#### 3. **Provider Layer** (Specialized Data Access)

**Provider Architecture:**

```python
# Base provider interface
class BaseProvider(ABC):
    name: str
    priority: int  # Lower = higher priority (10 = high, 50 = low)
    capabilities: Set[ProviderCapability]
    supported_dataset_types: Set[DatasetType]

    @abstractmethod
    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[dict] = None
    ) -> str:
        """Search for publications/datasets."""
```

**5 Registered Providers:**

| Provider | Priority | Capabilities | Performance | Coverage |
|----------|----------|--------------|-------------|----------|
| **AbstractProvider** | 10 (high) | GET_ABSTRACT | 200-500ms | All PubMed |
| **PubMedProvider** | 10 (high) | SEARCH_LITERATURE, FIND_LINKED_DATASETS, EXTRACT_METADATA | 1-3s | All PubMed indexed |
| **GEOProvider** | 10 (high) | DISCOVER_DATASETS, EXTRACT_METADATA, VALIDATE_METADATA | 2-5s | All GEO/SRA datasets |
| **PMCProvider** | 10 (high) | GET_FULL_CONTENT | 500ms-2s | 30-40% (NIH-funded + open access) |
| **WebpageProvider** | 50 (low) | GET_FULL_CONTENT | 2-8s | Major publishers + PDFs |

**Provider Details:**

**AbstractProvider** (Fast Path):
```python
# Location: lobster/tools/providers/abstract_provider.py
class AbstractProvider(BaseProvider):
    """Fast abstract retrieval via NCBI E-utilities."""

    capabilities = {ProviderCapability.GET_ABSTRACT}
    priority = 10  # High priority (fast)

    def get_abstract(self, identifier: str) -> PublicationMetadata:
        """Retrieve abstract metadata without full-text download."""
```

**PubMedProvider** (Literature & Linking):
```python
# Location: lobster/tools/providers/pubmed_provider.py
class PubMedProvider(BaseProvider):
    """PubMed literature search and dataset linking."""

    capabilities = {
        ProviderCapability.SEARCH_LITERATURE,
        ProviderCapability.FIND_LINKED_DATASETS,
        ProviderCapability.EXTRACT_METADATA,
    }
    priority = 10

    def search_publications(self, query: str, **kwargs) -> str:
        """Search PubMed with E-utilities."""

    def find_datasets_from_publication(self, identifier: str) -> str:
        """Find GEO/SRA datasets linked via PubMed."""
```

**GEOProvider** (Dataset Discovery):
```python
# Location: lobster/tools/providers/geo_provider.py
class GEOProvider(BaseProvider):
    """GEO dataset discovery and validation."""

    capabilities = {
        ProviderCapability.DISCOVER_DATASETS,
        ProviderCapability.EXTRACT_METADATA,
        ProviderCapability.VALIDATE_METADATA,
    }
    supported_dataset_types = {DatasetType.GEO}
    priority = 10

    def search_publications(self, query: str, **kwargs) -> str:
        """Search GEO datasets."""

    def search_by_accession(
        self,
        accession: str,
        include_parent_series: bool = False
    ) -> str:
        """Direct accession lookup with enhanced GSM handling."""
```

**PMCProvider** (Priority Full-Text):
```python
# Location: lobster/tools/providers/pmc_provider.py
class PMCProvider(BaseProvider):
    """PMC full-text extraction via XML API (PRIORITY PATH)."""

    capabilities = {ProviderCapability.GET_FULL_CONTENT}
    priority = 10  # High priority (10x faster than webpage)

    def extract_full_text(self, identifier: str) -> PMCFullTextResult:
        """
        Extract full-text from PMC XML with semantic tags.

        Benefits:
        - 10x faster (500ms vs 2-5s HTML scraping)
        - 95% accuracy for methods extraction
        - 100% table parsing success
        - Structured sections with <sec sec-type=\"methods\">
        - 30-40% coverage (NIH-funded + open access)
        """
```

**WebpageProvider** (Fallback Path):
```python
# Location: lobster/tools/providers/webpage_provider.py
class WebpageProvider(BaseProvider):
    """Webpage scraping and PDF extraction (FALLBACK)."""

    capabilities = {ProviderCapability.GET_FULL_CONTENT}
    priority = 50  # Low priority (slower fallback)

    def __init__(self, data_manager: DataManagerV2):
        self.docling_service = DoclingService(data_manager)  # Composition

    def extract_content(
        self,
        url: str,
        keywords: Optional[List[str]] = None,
        max_paragraphs: int = 100
    ) -> dict:
        """
        Extract content via webpage or PDF (uses DoclingService).

        Automatically detects format and routes to appropriate parser.
        """
```

**DoclingService** (Internal, Not Registered):
- Used internally by WebpageProvider via composition
- Not registered as separate provider
- Handles both webpage HTML and PDF parsing
- Structure-aware parsing with table extraction

---

## Three-Tier Content Cascade

The system implements intelligent fallback for full-text retrieval:

### Cascade Flow

```
User Request: get_full_content("PMID:35042229")
    ↓
Step 1: Check DataManager cache
    ├─ Cache hit? → Return immediately (<100ms)
    └─ Cache miss → Continue to Tier 1
    ↓
Tier 1: PMC XML API (Priority 10)
    ├─ Provider: PMCProvider
    ├─ Duration: 500ms-2s
    ├─ Coverage: 30-40% of biomedical literature
    ├─ Success? → Cache + Return ✅
    └─ PMCNotAvailableError → Continue to Tier 2
    ↓
Tier 2: Resolve to URL (if identifier)
    ├─ Use PublicationResolver
    ├─ PMID/DOI → Accessible URL
    ├─ Check accessibility
    └─ If paywalled → Return error with suggestions
    ↓
Tier 3: Webpage/PDF Extraction (Priority 50)
    ├─ Provider: WebpageProvider
    ├─ Auto-detect: Webpage HTML or PDF
    ├─ Duration: 2-8s
    ├─ Uses: DoclingService internally
    ├─ Success? → Cache + Return ✅
    └─ Failure → Return error
```

### Performance Characteristics

| Tier | Path | Duration | Success Rate | Coverage |
|------|------|----------|--------------|----------|
| **Cache** | DataManager lookup | <100ms | 100% (if cached) | Previously accessed |
| **Tier 1** | PMC XML API | 500ms-2s | 95% | 30-40% (open access) |
| **Tier 2** | URL Resolution | Variable | 70-80% | Depends on accessibility |
| **Tier 3** | Webpage/PDF | 2-8s | 70% | Major publishers + preprints |

### Code Example

```python
from lobster.tools.content_access_service import ContentAccessService

service = ContentAccessService(data_manager)

# Automatic three-tier cascade
content = service.get_full_content("PMID:35042229")

# Check which tier was used
print(f"Tier used: {content['tier_used']}")
# Possible values:
# - 'full_cached' (cache hit)
# - 'full_pmc_xml' (Tier 1: PMC)
# - 'full_webpage' (Tier 3: webpage HTML)
# - 'full_pdf' (Tier 3: PDF via Docling)

print(f"Source type: {content['source_type']}")
print(f"Extraction time: {content['extraction_time']:.2f}s")
print(f"Content length: {len(content['content'])} characters")
```

---

## Method Categories & Usage

### Discovery Methods (3)

#### search_literature()

Search PubMed, bioRxiv, medRxiv for publications.

**Example:**
```python
results, stats, ir = service.search_literature(
    query="BRCA1 breast cancer",
    max_results=10,
    sources=["pubmed"],  # Optional: filter to specific sources
    filters={"publication_year": "2023"}  # Optional: date filters
)

print(f"Found {stats['results_count']} papers")
print(f"Provider: {stats['provider_used']}")  # PubMedProvider
print(f"Time: {stats['execution_time_ms']}ms")
```

#### discover_datasets()

Search for omics datasets with automatic accession detection.

**Example:**
```python
# Direct accession (auto-detected)
results, stats, ir = service.discover_datasets(
    query="GSM6204600",  # GEO sample ID
    dataset_type=DatasetType.GEO
)

# Text search
results, stats, ir = service.discover_datasets(
    query="single-cell RNA-seq breast cancer",
    dataset_type=DatasetType.GEO,
    max_results=5
)

print(f"Found {stats['results_count']} datasets")
print(f"Accession detected: {stats.get('accession_detected', False)}")
```

#### find_linked_datasets()

Find datasets associated with a publication.

**Example:**
```python
results = service.find_linked_datasets(
    identifier="PMID:35042229",
    dataset_types=[DatasetType.GEO, DatasetType.SRA]
)

print(results)  # Formatted list of linked datasets
```

### Metadata Methods (2)

#### extract_metadata()

Extract publication or dataset metadata.

**Example:**
```python
# Publication metadata
metadata = service.extract_metadata("PMID:35042229")

print(f"Title: {metadata.title}")
print(f"Authors: {metadata.authors}")
print(f"Abstract: {metadata.abstract[:200]}...")

# Dataset metadata
metadata = service.extract_metadata("GSE180759", source="geo")
```

#### validate_metadata()

Validate dataset metadata completeness before download.

**Example:**
```python
report = service.validate_metadata(
    dataset_id="GSE180759",
    required_fields=["smoking_status", "treatment_response"],
    threshold=0.8  # 80% of samples must have fields
)

print(report)
# Formatted validation report with:
# - Completeness scores
# - Missing fields
# - Sample coverage
# - Recommendations (PROCEED/COHORT/SKIP)
```

### Content Methods (3)

#### get_abstract()

Fast abstract retrieval (Tier 1: 200-500ms).

**Example:**
```python
abstract = service.get_abstract("PMID:35042229")

print(f"Title: {abstract['title']}")
print(f"Authors: {abstract['authors']}")
print(f"Abstract: {abstract['abstract']}")
print(f"Keywords: {abstract['keywords']}")
```

#### get_full_content()

Full-text extraction with three-tier cascade.

**Example:**
```python
# Automatic cascade: PMC → Webpage → PDF
content = service.get_full_content("PMID:35042229")

print(f"Tier used: {content['tier_used']}")
print(f"Methods section: {content.get('methods_text', 'N/A')[:200]}...")
print(f"Tables: {content['metadata']['tables']}")
print(f"Software detected: {content['metadata']['software']}")
```

#### extract_methods()

Extract structured methods from full content.

**Example:**
```python
# Get full content first
content = service.get_full_content("PMID:35042229")

# Extract methods
methods = service.extract_methods(content, include_tables=True)

print(f"Software: {methods['software_used']}")
print(f"GitHub repos: {methods['github_repos']}")
```

### System Methods (1)

#### query_capabilities()

Query available providers and their capabilities.

**Example:**
```python
capabilities = service.query_capabilities()

print(capabilities)
# Returns formatted matrix showing:
# - Available operations
# - Registered providers
# - Supported dataset types
# - Performance tiers
# - Cascade logic
```

---

## Integration with Research Agent

The research_agent uses ContentAccessService through 10 tools:

### Tool Mapping

| Agent Tool | ContentAccessService Method | Category |
|-----------|----------------------------|----------|
| `search_literature` | `search_literature()` | Discovery |
| `fast_dataset_search` | `discover_datasets()` | Discovery |
| `find_related_entries` | `find_linked_datasets()` | Discovery |
| `get_dataset_metadata` | `extract_metadata()` | Metadata |
| `fast_abstract_search` | `get_abstract()` | Content |
| `read_full_publication` | `get_full_content()` | Content |
| `extract_methods` | `extract_methods()` | Content |
| `validate_dataset_metadata` | `validate_metadata()` | Metadata |

### Example Agent Workflow

```python
# User: "Find breast cancer datasets with smoking status"

# Step 1: Literature search (PubMedProvider)
results, stats, ir = service.search_literature("breast cancer smoking")

# Step 2: Discover datasets (GEOProvider)
datasets, stats, ir = service.discover_datasets(
    "breast cancer",
    DatasetType.GEO,
    filters={"organism": "human"}
)

# Step 3: Validate metadata (GEOProvider)
report = service.validate_metadata(
    "GSE180759",
    required_fields=["smoking_status"]
)

# Step 4: Get full publication (PMC → Webpage → PDF cascade)
content = service.get_full_content("PMID:35042229")

# All operations tracked in W3C-PROV provenance
```

---

## Performance Benchmarks

**Benchmark Metadata:**
- **Date Measured:** 2025-01-15
- **Lobster Version:** v0.2.0
- **Network:** Residential broadband (100 Mbps)
- **Sample Size:** 100 operations per provider
- **Test Conditions:** Mixed cache hit/miss scenarios

### Provider Performance

| Provider | Operation | Mean Duration | P95 | P99 | Success Rate |
|----------|-----------|---------------|-----|-----|--------------|
| **AbstractProvider** | `get_abstract()` | 350ms | 450ms | 500ms | 95%+ |
| **PubMedProvider** | `search_literature()` | 2.1s | 3.5s | 5s | 99%+ |
| **GEOProvider** | `discover_datasets()` | 3.2s | 4.8s | 6s | 95%+ |
| **PMCProvider** | `get_full_content()` | 1.2s | 2s | 2.5s | 95% (of eligible) |
| **WebpageProvider** | `get_full_content()` | 4.5s | 7s | 10s | 70-80% |

**Note:** Performance varies with network conditions and external API load. P95/P99 represent 95th and 99th percentile latencies.

### Cascade Performance

| Scenario | Tier Used | Duration | Frequency |
|----------|-----------|----------|-----------|
| **Cache hit** | Cache | <100ms | High (repeated access) |
| **PMC available** | Tier 1 | 500ms-2s | 30-40% of requests |
| **PMC unavailable** | Tier 3 | 2-8s | 60-70% of requests |
| **Paywalled** | Error | Variable | 10-15% of requests |

### Optimization Strategies

1. **DataManager-first caching** - All operations check cache before API calls
2. **Capability-based routing** - Optimal provider selected automatically
3. **Priority ordering** - Fast providers tried first (Priority 10 before 50)
4. **Graceful degradation** - Automatic fallback on provider failures
5. **Session persistence** - Workspace caching for handoffs

---

## DataManager-First Caching

All caching goes through DataManagerV2 (architectural requirement).

### Cache Flow

```
Service Method Call
    ↓
1. Check DataManager cache
    ├─ Cache hit? → Return immediately
    └─ Cache miss → Continue
    ↓
2. Execute provider operation
    ├─ Success? → Store in DataManager + Return
    └─ Error? → Return error (no cache)
    ↓
3. DataManager stores:
    ├─ In-memory cache (session-scoped)
    ├─ Workspace filesystem (persistent)
    └─ W3C-PROV provenance log
```

### Cache Methods

```python
# ContentAccessService automatically caches all operations

# Cache publication content
data_manager.cache_publication_content(
    identifier="PMID:38448586",
    content=content_result,
    format="json"
)

# Retrieve cached content
cached = data_manager.get_cached_publication("PMID:38448586")

# Cache location
# ~/.lobster/literature_cache/{identifier}.json
```

---

## Troubleshooting

### Issue: "No providers available for capability"

**Symptom:**
```
ERROR: No available providers for literature search.
```

**Cause:** Provider not registered or capability not declared.

**Solution:**
```python
# Check capability matrix
capabilities = service.query_capabilities()
print(capabilities)

# Verify provider registration
providers = service.registry.get_all_providers()
print(f"Registered providers: {len(providers)}")
```

### Issue: PMC Full-Text Not Available

**Symptom:**
```
INFO: PMC full text not available for PMID:12345, falling back...
```

**Cause:** Paper not in PMC open access collection (70% of papers).

**Expected:** Automatic fallback to Tier 3 (Webpage/PDF).

**Verification:**
```python
content = service.get_full_content("PMID:12345")
print(f"Tier used: {content['tier_used']}")  # Should be 'full_webpage' or 'full_pdf'
```

### Issue: Dataset Validation Failed

**Symptom:**
```
WARNING: Dataset GSE12345 missing required metadata
```

**Solution:**
```python
# Check validation report
report = service.validate_metadata(
    "GSE12345",
    required_fields=["condition", "sample_id"]
)
print(report)

# Review recommendations:
# - PROCEED: Full integration possible
# - COHORT: Cohort-level only
# - SKIP: Insufficient metadata
```

---

## Best Practices

### 1. Use Capability-Based Routing

**✅ GOOD: Let the registry route**
```python
# System automatically selects PubMedProvider
results, stats, ir = service.search_literature("BRCA1")
```

**❌ BAD: Manual provider selection**
```python
# Don't access providers directly
provider = service.registry.get_provider_for_capability(...)
```

### 2. Leverage Three-Tier Cascade

**✅ GOOD: Trust the cascade**
```python
# Automatically tries PMC → Webpage → PDF
content = service.get_full_content("PMID:35042229")
```

**❌ BAD: Force specific tier**
```python
# Don't try to manually control cascade
```

### 3. Validate Before Download

**✅ GOOD: Pre-download validation**
```python
# Check metadata first
report = service.validate_metadata("GSE180759", required_fields=["condition"])

if "PROCEED" in report:
    # Then download dataset
    pass
```

### 4. Check Capabilities

**✅ GOOD: Query capabilities first**
```python
# Check what's available
capabilities = service.query_capabilities()
print(capabilities)
```

---

## Version History

**v0.2.0 (January 2025) - Phase 1-6 Complete:**
- ✅ Phase 1: Provider infrastructure (5 providers)
- ✅ Phase 2: ContentAccessService consolidation (10 methods)
- ✅ Phase 3: metadata_assistant agent (4 tools)
- ✅ Phase 4: research_agent enhancements (10 tools)
- ✅ Phase 5: Multi-agent handoff patterns (3 workflows)
- ✅ Phase 6: Integration testing (127 tests, 3988 lines)
- Added: ProviderRegistry with capability-based routing
- Added: GEOProvider for dataset discovery
- Added: Validation and metadata standardization
- Enhanced: Three-tier cascade with PMC priority
- Deprecated: UnifiedContentService (archived)
- Deprecated: PublicationService (replaced)

**v0.2.0 (January 2025) - Phase 3:**
- ✅ UnifiedContentService (coordination layer)
- ✅ PMC-first access strategy
- ✅ DoclingService integration
- ✅ PublicationIntelligenceService deletion

**v0.2.0 (November 2024):**
- Initial: PublicationIntelligenceService with Docling

---

## References

- **ContentAccessService API:** See [16-services-api.md](16-services-api.md)
- **Provider Architecture:** Source code in `lobster/tools/providers/`
- **Research Agent:** See [15-agents-api.md](15-agents-api.md)
- **Metadata Assistant:** Phase 3 documentation in code
- **Integration Tests:** `tests/integration/test_*_real_api.py` (127 tests)

---

**Next Steps:**
- Review [16-services-api.md](16-services-api.md) for detailed API documentation
- See [15-agents-api.md](15-agents-api.md) for Research Agent integration
- Check [28-troubleshooting.md](28-troubleshooting.md) for common issues
- Explore Phase 7 test suite for usage examples
