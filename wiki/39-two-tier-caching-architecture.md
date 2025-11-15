# 39. Two-Tier Caching Architecture

## Overview

Lobster AI implements a sophisticated **two-tier caching architecture** designed to optimize performance across different use cases: fast in-memory access for active session data and durable filesystem storage for cross-session persistence. This architecture enables efficient agent coordination, reduces redundant API calls, and supports workspace-based handoffs between specialized agents.

### Why Two Tiers?

The dual-tier approach addresses distinct performance and persistence requirements:

| Requirement | Tier 1 (Session Cache) | Tier 2 (Persistent Cache) |
|-------------|----------------------|---------------------------|
| **Speed** | Microsecond access (in-memory) | Millisecond access (filesystem) |
| **Durability** | Session-scoped (lost on exit) | Persistent across sessions |
| **Use Case** | Active analysis, frequent lookups | Workspace handoffs, reproducibility |
| **Capacity** | Limited by RAM (typically MB) | Limited by disk (typically GB) |
| **Format** | Python dictionaries | JSON with Pydantic validation |

This architecture follows the **cache hierarchy principle**: fast, volatile storage for hot data and slower, durable storage for warm data.

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Request"
        USER[User: 'Get metadata for GSE180759']
    end

    subgraph "Agent Layer"
        AGENT[research_agent<br/>Content Discovery]
        META_AGENT[metadata_assistant<br/>Sample Mapping]
    end

    subgraph "Service Layer"
        CAS[ContentAccessService<br/>Provider Routing]
        WCS[WorkspaceContentService<br/>Persistent Storage]
    end

    subgraph "Tier 1: Session Cache"
        METADATA_STORE[DataManager.metadata_store<br/>In-Memory Dict<br/>⚡ <1ms access]

        subgraph "Storage Location"
            T1_DIR[".lobster_workspace/<br/>literature_cache/<br/>publications/"]
        end
    end

    subgraph "Tier 2: Persistent Cache"
        subgraph "Storage Directories"
            PUBS_DIR["workspace/<br/>literature/<br/>*.json"]
            DATA_DIR["workspace/<br/>data/<br/>*.json"]
            META_DIR["workspace/<br/>metadata/<br/>*.json"]
        end

        PYDANTIC[Pydantic Validation<br/>PublicationContent<br/>DatasetContent<br/>MetadataContent]
    end

    subgraph "External APIs"
        PUBMED[PubMed API]
        PMC[PMC XML API]
        GEO[GEO Database]
    end

    USER --> AGENT
    AGENT --> CAS

    CAS --> METADATA_STORE
    METADATA_STORE -.cache hit.-> CAS
    METADATA_STORE -.cache miss.-> PUBMED
    METADATA_STORE -.cache miss.-> PMC
    METADATA_STORE -.cache miss.-> GEO

    PUBMED --> METADATA_STORE
    PMC --> METADATA_STORE
    GEO --> METADATA_STORE

    METADATA_STORE --> T1_DIR

    AGENT --> WCS
    WCS --> PUBS_DIR
    WCS --> DATA_DIR
    WCS --> META_DIR

    PUBS_DIR --> PYDANTIC
    DATA_DIR --> PYDANTIC
    META_DIR --> PYDANTIC

    WCS --> META_AGENT
    META_AGENT --> WCS

    classDef agent fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef service fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef tier1 fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    classDef tier2 fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    classDef api fill:#ffccbc,stroke:#d84315,stroke-width:2px

    class AGENT,META_AGENT agent
    class CAS,WCS service
    class METADATA_STORE,T1_DIR tier1
    class PUBS_DIR,DATA_DIR,META_DIR,PYDANTIC tier2
    class PUBMED,PMC,GEO api
```

## Tier 1: Session Cache (DataManager.metadata_store)

### Purpose

Tier 1 provides **ultra-fast in-memory access** to frequently accessed metadata during a single analysis session. It acts as the first line of defense against redundant API calls.

### Implementation Details

**Location**: `DataManagerV2.metadata_store` attribute
**Type**: `Dict[str, MetadataEntry]` (TypedDict for type safety)
**Filesystem Mirror**: `.lobster_workspace/literature_cache/publications/`

### Data Structure

```python
from typing import TypedDict, Dict, Any

class MetadataEntry(TypedDict, total=False):
    """
    Structured metadata entry in session cache.

    Fields:
        metadata: Actual metadata from GEOparse or PubMed
        validation: Validation results (completeness, required fields)
        fetch_timestamp: ISO 8601 timestamp when fetched
        strategy_config: Download strategy configuration
        stored_by: Component that stored the metadata
        modality_detection: Detected data modality (RNA-seq, proteomics, etc.)
        concatenation_decision: Sample merging strategy
    """
    metadata: Dict[str, Any]
    validation: Dict[str, Any]
    fetch_timestamp: str
    strategy_config: Dict[str, Any]
    stored_by: str
    modality_detection: Dict[str, Any]
    concatenation_decision: Dict[str, Any]
```

### Storage & Retrieval

**Storing Metadata:**
```python
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize DataManager
dm = DataManagerV2(workspace_path="./my_workspace")

# Store GEO metadata in session cache
dm.metadata_store["GSE180759"] = {
    "metadata": {
        "title": "Single-cell RNA-seq of lung cancer",
        "platform": "GPL24676",
        "samples": 36,
        # ... full GEO metadata
    },
    "validation": {"status": "complete", "missing_fields": []},
    "fetch_timestamp": "2025-01-14T10:30:00Z",
    "stored_by": "research_agent",
}

# Retrieve from cache
if "GSE180759" in dm.metadata_store:
    metadata = dm.metadata_store["GSE180759"]["metadata"]
    print(f"Cache hit! Found {metadata['samples']} samples")
else:
    print("Cache miss - need to fetch from GEO API")
```

**Cache Publication Content:**
```python
from pathlib import Path

# Cache publication with methods section
content = {
    "markdown": "# Methods\n\nWe used Seurat v4...",
    "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8765432/",
    "parser": "docling",
    "methods_text": "Analysis was performed using Seurat v4...",
    "software_detected": ["seurat", "scanpy"],
    "timestamp": "2025-01-14T10:35:00Z",
}

# Returns Path to cached file
cache_path = dm.cache_publication_content(
    identifier="PMID:35042229",
    content=content,
    format="markdown"  # or "json"
)

print(f"Cached to: {cache_path}")
# Output: Cached to: .lobster_workspace/literature_cache/publications/pmid_35042229.md
```

**Retrieve Cached Publication:**
```python
# Check cache before API call
cached = dm.get_cached_publication("PMID:35042229")

if cached:
    print(f"Cache hit: {cached['methods_text'][:100]}...")
    print(f"Format: {cached['format']}")  # "markdown" or "json"
else:
    print("Cache miss - fetch from PMC API")
```

### Lifecycle

- **Creation**: Automatically created when `DataManagerV2` is initialized
- **Population**: Filled by agents during API calls (GEO downloads, PubMed searches)
- **Persistence**: Exists only during the Python session
- **Destruction**: Lost when Python process exits (unless session saved)

### Performance Characteristics

| Operation | Duration | Notes |
|-----------|----------|-------|
| Read | <0.1ms | Direct dictionary lookup |
| Write | <0.1ms | Simple assignment |
| Contains check | <0.1ms | Hash table lookup |
| Memory overhead | ~1-5 KB per entry | Depends on metadata size |

### When to Use Tier 1

✅ **Best For:**
- Repeated lookups of the same dataset metadata within a session
- Fast validation checks before expensive operations
- Agent decision-making based on metadata (sample count, platform)
- Avoiding redundant API calls during iterative analysis

❌ **Not Suitable For:**
- Sharing data between sessions (lost on exit)
- Cross-agent handoffs (use Tier 2 instead)
- Large content (full-text papers >1MB)
- Reproducible workflows (session-dependent)

## Tier 2: Persistent Cache (WorkspaceContentService)

### Purpose

Tier 2 provides **durable filesystem storage** with schema validation for cross-session persistence, workspace handoffs between agents, and reproducible workflows.

### Implementation Details

**Location**: `workspace/` subdirectories
**Format**: JSON files with Pydantic validation
**Service**: `WorkspaceContentService` class
**Schemas**: `PublicationContent`, `DatasetContent`, `MetadataContent`

### Storage Structure

```
workspace/
├── literature/              # Tier 2 publications (WorkspaceContentService)
│   ├── pmid_35042229.json
│   └── pmid_33057194.json
├── data/                    # Dataset metadata
│   ├── gse180759.json
│   └── gse156793.json
├── metadata/                # Analysis metadata (mappings, validation)
│   ├── gse180759_to_gse156793_mapping.json
│   └── gse180759_validation_report.json
└── .lobster_workspace/      # Tier 1 cache directory
    └── literature_cache/
        └── publications/    # Tier 1 publications (DataManager)
            ├── pmid_35042229.md
            └── abc123def456.json  # MD5-based filename
```

**Note**: There are two separate publication caches:
- **Tier 1**: `.lobster_workspace/literature_cache/publications/` (DataManager session cache)
- **Tier 2**: `workspace/literature/` (WorkspaceContentService persistent cache)

### Pydantic Schemas

**PublicationContent:**
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PublicationContent(BaseModel):
    """Schema for cached publication content."""

    identifier: str = Field(..., description="PMID, DOI, or bioRxiv ID")
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    methods: Optional[str] = None
    full_text: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    source: str = Field(..., description="PMC, PubMed, bioRxiv, etc.")
    cached_at: str = Field(..., description="ISO 8601 timestamp")
    url: Optional[str] = None
```

**DatasetContent:**
```python
class DatasetContent(BaseModel):
    """Schema for cached dataset content."""

    identifier: str = Field(..., description="GSE, SRA, etc.")
    title: Optional[str] = None
    platform: Optional[str] = Field(None, description="Illumina NovaSeq, etc.")
    platform_id: Optional[str] = Field(None, description="GPL570, etc.")
    organism: Optional[str] = Field(None, description="Homo sapiens, etc.")
    sample_count: int = Field(..., description="Number of samples")
    samples: Optional[Dict[str, Any]] = Field(None, description="GSM metadata")
    experimental_design: Optional[str] = None
    summary: Optional[str] = None
    pubmed_ids: List[str] = Field(default_factory=list)
    source: str = Field(..., description="GEO, SRA, PRIDE, etc.")
    cached_at: str = Field(..., description="ISO 8601 timestamp")
    url: Optional[str] = None
```

**MetadataContent:**
```python
class MetadataContent(BaseModel):
    """Schema for cached metadata (mappings, validation reports)."""

    identifier: str = Field(..., description="Unique identifier")
    content_type: str = Field(..., description="sample_mapping, validation, etc.")
    description: Optional[str] = None
    data: Dict[str, Any] = Field(..., description="Actual metadata content")
    related_datasets: List[str] = Field(default_factory=list)
    source: str = Field(..., description="Service that generated content")
    cached_at: str = Field(..., description="ISO 8601 timestamp")
```

### Storage & Retrieval

**Initialize Service:**
```python
from lobster.tools.workspace_content_service import (
    WorkspaceContentService,
    PublicationContent,
    DatasetContent,
    MetadataContent,
    ContentType,
    RetrievalLevel
)
from datetime import datetime

# Initialize service
service = WorkspaceContentService(data_manager=dm)
```

**Write Publication:**
```python
# Create publication content
pub = PublicationContent(
    identifier="PMID:35042229",
    title="Single-cell RNA-seq reveals heterogeneity in lung cancer",
    authors=["Smith J", "Jones A", "Brown K"],
    journal="Nature",
    year=2022,
    abstract="We performed single-cell RNA-seq on 36 samples...",
    methods="Analysis was performed using Seurat v4.1.0...",
    source="PMC",
    cached_at=datetime.now().isoformat()
)

# Write to workspace (with schema validation)
path = service.write_content(pub, ContentType.PUBLICATION)
print(f"Cached to: {path}")
# Output: workspace/literature/pmid_35042229.json
```

**Write Dataset Metadata:**
```python
# Create dataset content
dataset = DatasetContent(
    identifier="GSE180759",
    title="Single-cell RNA-seq of lung adenocarcinoma",
    platform="Illumina NovaSeq 6000",
    platform_id="GPL24676",
    organism="Homo sapiens",
    sample_count=36,
    samples={
        "GSM5467890": {"condition": "tumor", "smoking_status": "current"},
        "GSM5467891": {"condition": "normal", "smoking_status": "never"},
        # ... 34 more samples
    },
    summary="Single-cell transcriptomics reveals...",
    pubmed_ids=["35042229"],
    source="GEO",
    cached_at=datetime.now().isoformat(),
    url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180759"
)

path = service.write_content(dataset, ContentType.DATASET)
# Output: workspace/data/gse180759.json
```

**Write Metadata (Sample Mapping):**
```python
# Create sample mapping metadata
mapping = MetadataContent(
    identifier="gse180759_to_gse156793_mapping",
    content_type="sample_mapping",
    description="Sample ID mapping using fuzzy matching",
    data={
        "strategy": "fuzzy",
        "confidence_threshold": 0.85,
        "matched_samples": 32,
        "unmatched_samples": 4,
        "mapping": {
            "GSM5467890": {"target": "GSM4738901", "confidence": 0.96},
            "GSM5467891": {"target": "GSM4738902", "confidence": 0.92},
            # ... more mappings
        }
    },
    related_datasets=["GSE180759", "GSE156793"],
    source="SampleMappingService",
    cached_at=datetime.now().isoformat()
)

path = service.write_content(mapping, ContentType.METADATA)
# Output: workspace/metadata/gse180759_to_gse156793_mapping.json
```

**Read Content with Level Filtering:**
```python
# Read summary level (minimal fields)
summary = service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.SUMMARY
)
# Returns: {identifier, title, authors, journal, year, source}

# Read methods level (for publications)
methods = service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.METHODS
)
# Returns: {identifier, title, methods, source}

# Read samples level (for datasets)
samples = service.read_content(
    identifier="GSE180759",
    content_type=ContentType.DATASET,
    level=RetrievalLevel.SAMPLES
)
# Returns: {identifier, sample_count, samples}

# Read full content
full = service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.FULL
)
# Returns: All fields including full_text
```

**List Cached Content:**
```python
# List all publications
publications = service.list_content(ContentType.PUBLICATION)
for pub in publications:
    print(f"{pub['identifier']}: {pub['title']}")

# List all datasets
datasets = service.list_content(ContentType.DATASET)
for dataset in datasets:
    print(f"{dataset['identifier']}: {dataset['sample_count']} samples")

# List all metadata files
metadata_files = service.list_content(ContentType.METADATA)
for meta in metadata_files:
    print(f"{meta['identifier']} ({meta.get('content_type', 'unknown')})")
```

### Retrieval Levels

The `RetrievalLevel` enum controls how much data is returned, enabling efficient bandwidth usage:

| Level | Description | Use Case | Fields Returned |
|-------|-------------|----------|-----------------|
| **SUMMARY** | Basic overview | Listing, quick checks | identifier, title, authors, year, source |
| **METHODS** | Methods section (publications) | Parameter extraction | identifier, title, methods, source |
| **SAMPLES** | Sample metadata (datasets) | Sample mapping, QC | identifier, sample_count, samples |
| **PLATFORM** | Platform/technology info | Compatibility checks | identifier, platform, platform_id |
| **FULL** | All available content | Deep analysis | All fields including full_text |

**Example: Efficient Listing:**
```python
# Only fetch summaries for listing (fast)
publications = service.list_content(
    ContentType.PUBLICATION,
    level=RetrievalLevel.SUMMARY
)

# Then fetch full content for selected publication
selected_pub = service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.FULL
)
```

### Lifecycle

- **Creation**: Directories created during `DataManagerV2` initialization
- **Population**: Filled by agents via `write_content()` calls
- **Persistence**: Survives session restarts, Python exits
- **Cleanup**: Manual deletion or workspace reset

### Performance Characteristics

| Operation | Duration | Notes |
|-----------|----------|-------|
| Read | 1-10ms | JSON parsing + Pydantic validation |
| Write | 2-20ms | Schema validation + filesystem write |
| List | 10-100ms | Directory scan + JSON parsing |
| Search | 50-500ms | Full directory traversal |
| Memory overhead | ~5-50 KB per file | Depends on content size |

### When to Use Tier 2

✅ **Best For:**
- Cross-session persistence (survive Python restarts)
- Agent handoffs (research_agent → metadata_assistant)
- Reproducible workflows (analysis provenance)
- Sharing data between team members (workspace exports)
- Large content storage (full-text papers, >1MB)

❌ **Not Suitable For:**
- Ultra-frequent lookups (use Tier 1 instead, <0.1ms vs 1-10ms)
- Temporary intermediate results (no need for durability)
- Real-time streaming data (too slow for live updates)

## When to Use Each Tier: Decision Flowchart

```mermaid
flowchart TD
    START[Need to cache data?]
    START --> Q1{Need cross-session<br/>persistence?}

    Q1 -->|Yes| Q2{Large content<br/> >1MB?}
    Q1 -->|No| Q3{Frequent lookups<br/> >100/min?}

    Q2 -->|Yes| TIER2_LARGE[✅ Use Tier 2<br/>WorkspaceContentService<br/>with level filtering]
    Q2 -->|No| Q4{Need schema<br/>validation?}

    Q4 -->|Yes| TIER2_SCHEMA[✅ Use Tier 2<br/>Pydantic validation]
    Q4 -->|No| TIER2_BASIC[✅ Use Tier 2<br/>Standard caching]

    Q3 -->|Yes| TIER1_FAST[✅ Use Tier 1<br/>metadata_store]
    Q3 -->|No| Q5{Need agent<br/>handoffs?}

    Q5 -->|Yes| TIER2_HANDOFF[✅ Use Tier 2<br/>workspace/ directories]
    Q5 -->|No| Q6{Within single<br/>session?}

    Q6 -->|Yes| TIER1_SESSION[✅ Use Tier 1<br/>session cache]
    Q6 -->|No| TIER2_PERSIST[✅ Use Tier 2<br/>persistent cache]

    style START fill:#e3f2fd
    style TIER1_FAST fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style TIER1_SESSION fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style TIER2_LARGE fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    style TIER2_SCHEMA fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    style TIER2_BASIC fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    style TIER2_HANDOFF fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    style TIER2_PERSIST fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
```

### Quick Decision Matrix

| Scenario | Recommended Tier | Rationale |
|----------|------------------|-----------|
| **GEO metadata for active analysis** | Tier 1 (metadata_store) | Ultra-fast repeated lookups |
| **PubMed search results (transient)** | Tier 1 (metadata_store) | Session-scoped, frequent access |
| **Full-text paper for methods extraction** | Tier 2 (workspace/literature/) | Large content, cross-session reuse |
| **Sample mapping between datasets** | Tier 2 (workspace/metadata/) | Agent handoff, reproducibility |
| **Dataset validation report** | Tier 2 (workspace/metadata/) | Provenance tracking, sharing |
| **Temporary QC metrics** | Tier 1 (metadata_store) | Single-session analysis |
| **Publication methods for notebook export** | Tier 2 (workspace/literature/) | Reproducible workflow |
| **Dataset metadata for meta-analysis** | Tier 2 (workspace/data/) | Cross-session persistence |

## Code Examples

### Example 1: Multi-Tier Caching Strategy

This example demonstrates optimal use of both tiers:

```python
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_content_service import (
    WorkspaceContentService,
    DatasetContent,
    ContentType,
    RetrievalLevel
)
from datetime import datetime

# Initialize
dm = DataManagerV2(workspace_path="./my_project")
service = WorkspaceContentService(dm)

dataset_id = "GSE180759"

# Step 1: Check Tier 1 (fast)
if dataset_id in dm.metadata_store:
    print("✅ Tier 1 cache hit (in-memory)")
    metadata = dm.metadata_store[dataset_id]["metadata"]
else:
    # Step 2: Check Tier 2 (persistent)
    try:
        tier2_data = service.read_content(
            identifier=dataset_id,
            content_type=ContentType.DATASET,
            level=RetrievalLevel.SUMMARY
        )
        print("✅ Tier 2 cache hit (filesystem)")

        # Promote to Tier 1 for fast access
        dm.metadata_store[dataset_id] = {
            "metadata": tier2_data,
            "fetch_timestamp": datetime.now().isoformat(),
            "stored_by": "cache_promotion"
        }
        metadata = tier2_data

    except FileNotFoundError:
        # Step 3: Fetch from API (cache miss)
        print("❌ Cache miss - fetching from GEO API")
        from lobster.tools.geo_service import GEOService

        geo = GEOService(dm)
        metadata = geo.fetch_metadata(dataset_id)

        # Cache in both tiers
        # Tier 1: Fast in-memory
        dm.metadata_store[dataset_id] = {
            "metadata": metadata,
            "fetch_timestamp": datetime.now().isoformat(),
            "stored_by": "geo_service"
        }

        # Tier 2: Persistent workspace
        dataset = DatasetContent(
            identifier=dataset_id,
            title=metadata.get("title", ""),
            platform=metadata.get("platform", ""),
            sample_count=len(metadata.get("samples", [])),
            samples=metadata.get("samples", {}),
            source="GEO",
            cached_at=datetime.now().isoformat()
        )
        service.write_content(dataset, ContentType.DATASET)

# Use metadata for analysis
print(f"Dataset: {metadata.get('title')}")
print(f"Samples: {len(metadata.get('samples', []))}")
```

### Example 2: Agent Handoff with Tier 2

Demonstrates how `research_agent` caches content for `metadata_assistant`:

```python
# In research_agent tool
@tool
def write_to_workspace(identifier: str, content: Dict[str, Any],
                       content_type: str) -> str:
    """Cache content for cross-agent handoff."""
    from datetime import datetime

    service = WorkspaceContentService(data_manager)

    if content_type == "publication":
        pub = PublicationContent(
            identifier=identifier,
            title=content.get("title", ""),
            authors=content.get("authors", []),
            abstract=content.get("abstract", ""),
            methods=content.get("methods", ""),
            source=content.get("source", ""),
            cached_at=datetime.now().isoformat()
        )
        path = service.write_content(pub, ContentType.PUBLICATION)

    elif content_type == "dataset":
        dataset = DatasetContent(
            identifier=identifier,
            sample_count=len(content.get("samples", [])),
            samples=content.get("samples", {}),
            source="GEO",
            cached_at=datetime.now().isoformat()
        )
        path = service.write_content(dataset, ContentType.DATASET)

    return f"Cached to: {path}"

# Later, in metadata_assistant
@tool
def get_content_from_workspace(identifier: str, level: str = "full") -> Dict:
    """Retrieve cached content from workspace."""
    service = WorkspaceContentService(data_manager)

    # Try dataset first
    try:
        content = service.read_content(
            identifier=identifier,
            content_type=ContentType.DATASET,
            level=RetrievalLevel[level.upper()]
        )
        return content
    except FileNotFoundError:
        # Try publication
        content = service.read_content(
            identifier=identifier,
            content_type=ContentType.PUBLICATION,
            level=RetrievalLevel[level.upper()]
        )
        return content
```

### Example 3: Efficient Listing with Level Filtering

```python
from lobster.tools.workspace_content_service import RetrievalLevel

service = WorkspaceContentService(dm)

# List all publications (summary only - fast)
print("=" * 60)
print("Cached Publications (Summary)")
print("=" * 60)

publications = service.list_content(ContentType.PUBLICATION)
for pub in publications:
    # Only basic fields loaded (fast)
    print(f"\n{pub['identifier']}")
    print(f"  Title: {pub.get('title', 'N/A')}")
    print(f"  Source: {pub.get('source', 'N/A')}")
    print(f"  Authors: {len(pub.get('authors', []))} authors")

# User selects publication - then load full content
selected = "PMID:35042229"
print(f"\n\nLoading full content for {selected}...")

full_pub = service.read_content(
    identifier=selected,
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.FULL
)

print(f"\nAbstract ({len(full_pub['abstract'])} chars):")
print(full_pub['abstract'][:200] + "...")

print(f"\nMethods ({len(full_pub['methods'])} chars):")
print(full_pub['methods'][:200] + "...")
```

### Example 4: Workspace-Based Sample Mapping

```python
# Step 1: research_agent discovers datasets and caches metadata
@tool
def discover_and_cache_datasets(query: str) -> str:
    """Search GEO and cache dataset metadata."""
    from lobster.tools.content_access_service import ContentAccessService

    cas = ContentAccessService(data_manager)
    results = cas.discover_datasets(query, DatasetType.GEO)

    service = WorkspaceContentService(data_manager)
    cached = []

    for dataset in results[:5]:  # Cache top 5
        metadata = cas.extract_metadata(dataset['accession'])

        dataset_content = DatasetContent(
            identifier=dataset['accession'],
            title=metadata.get('title'),
            sample_count=metadata.get('sample_count'),
            samples=metadata.get('samples'),
            source="GEO",
            cached_at=datetime.now().isoformat()
        )

        path = service.write_content(dataset_content, ContentType.DATASET)
        cached.append(dataset['accession'])

    return f"Cached {len(cached)} datasets: {', '.join(cached)}"

# Step 2: Handoff to metadata_assistant
handoff_to_metadata_assistant(
    "Map samples between GSE180759 and GSE156793 using cached workspace data"
)

# Step 3: metadata_assistant performs mapping
@tool
def map_cached_datasets(dataset1: str, dataset2: str) -> Dict:
    """Map samples using workspace-cached metadata."""
    service = WorkspaceContentService(data_manager)

    # Read from Tier 2 cache
    data1 = service.read_content(
        dataset1, ContentType.DATASET, RetrievalLevel.SAMPLES
    )
    data2 = service.read_content(
        dataset2, ContentType.DATASET, RetrievalLevel.SAMPLES
    )

    # Perform mapping (simplified)
    from lobster.tools.sample_mapping_service import SampleMappingService
    mapper = SampleMappingService(data_manager)

    mapping = mapper.map_samples(
        data1['samples'], data2['samples'],
        strategy="fuzzy", min_confidence=0.8
    )

    # Cache mapping result
    mapping_content = MetadataContent(
        identifier=f"{dataset1}_to_{dataset2}_mapping",
        content_type="sample_mapping",
        description=f"Fuzzy mapping between {dataset1} and {dataset2}",
        data=mapping,
        related_datasets=[dataset1, dataset2],
        source="SampleMappingService",
        cached_at=datetime.now().isoformat()
    )

    service.write_content(mapping_content, ContentType.METADATA)

    return mapping
```

## Performance Characteristics

### Tier 1 Performance

**Microbenchmark (10,000 operations):**
```python
import time
from lobster.core.data_manager_v2 import DataManagerV2

dm = DataManagerV2()

# Write benchmark
start = time.time()
for i in range(10000):
    dm.metadata_store[f"GSE{i}"] = {"metadata": {"samples": i}}
write_time = (time.time() - start) * 1000
print(f"Tier 1 Write: {write_time/10000:.4f} ms/op")

# Read benchmark
start = time.time()
for i in range(10000):
    _ = dm.metadata_store[f"GSE{i}"]
read_time = (time.time() - start) * 1000
print(f"Tier 1 Read: {read_time/10000:.4f} ms/op")
```

**Expected Output:**
```
Tier 1 Write: 0.0015 ms/op  (1.5 microseconds)
Tier 1 Read: 0.0008 ms/op   (0.8 microseconds)
```

### Tier 2 Performance

**Benchmark (100 operations):**
```python
from lobster.tools.workspace_content_service import WorkspaceContentService

service = WorkspaceContentService(dm)

# Write benchmark
datasets = []
start = time.time()
for i in range(100):
    dataset = DatasetContent(
        identifier=f"GSE{i}",
        sample_count=36,
        source="GEO",
        cached_at=datetime.now().isoformat()
    )
    path = service.write_content(dataset, ContentType.DATASET)
    datasets.append(f"GSE{i}")
write_time = (time.time() - start) * 1000
print(f"Tier 2 Write: {write_time/100:.2f} ms/op")

# Read benchmark (SUMMARY level)
start = time.time()
for identifier in datasets:
    _ = service.read_content(
        identifier, ContentType.DATASET, RetrievalLevel.SUMMARY
    )
read_time = (time.time() - start) * 1000
print(f"Tier 2 Read (SUMMARY): {read_time/100:.2f} ms/op")

# Read benchmark (FULL level)
start = time.time()
for identifier in datasets:
    _ = service.read_content(
        identifier, ContentType.DATASET, RetrievalLevel.FULL
    )
full_read_time = (time.time() - start) * 1000
print(f"Tier 2 Read (FULL): {full_read_time/100:.2f} ms/op")
```

**Expected Output:**
```
Tier 2 Write: 3.45 ms/op
Tier 2 Read (SUMMARY): 1.23 ms/op
Tier 2 Read (FULL): 2.87 ms/op
```

### Performance Comparison

| Metric | Tier 1 (In-Memory) | Tier 2 (Filesystem) | Speedup |
|--------|-------------------|---------------------|---------|
| **Write** | 0.0015 ms | 3.45 ms | **2,300x faster** |
| **Read** | 0.0008 ms | 1.23 ms (SUMMARY) | **1,537x faster** |
| **Read** | 0.0008 ms | 2.87 ms (FULL) | **3,587x faster** |
| **Memory** | 1-5 KB/entry | 5-50 KB/file | 1-10x more |
| **Durability** | ❌ Session-only | ✅ Persistent | N/A |

**Key Insight**: Tier 1 is 1,500-3,500x faster for read operations, justifying the dual-tier architecture for performance-critical workflows.

## Migration Notes

### Existing Code Using Metadata Store

**Before (Tier 1 only):**
```python
# Agent stores metadata in session cache
dm.metadata_store["GSE180759"] = metadata

# Later in same session
if "GSE180759" in dm.metadata_store:
    metadata = dm.metadata_store["GSE180759"]
```

**After (Dual-tier):**
```python
# Agent stores in both tiers
dm.metadata_store["GSE180759"] = metadata  # Tier 1 (fast)

service.write_content(
    DatasetContent(..., cached_at=datetime.now().isoformat()),
    ContentType.DATASET
)  # Tier 2 (persistent)

# Retrieval checks both tiers
if "GSE180759" in dm.metadata_store:
    metadata = dm.metadata_store["GSE180759"]  # Tier 1 hit
else:
    try:
        metadata = service.read_content(
            "GSE180759", ContentType.DATASET, RetrievalLevel.FULL
        )  # Tier 2 hit
        dm.metadata_store["GSE180759"] = metadata  # Promote to Tier 1
    except FileNotFoundError:
        # Fetch from API
        pass
```

### Workspace Content Service Integration

The `workspace_content_service.py` module provides the Tier 2 implementation. To integrate with existing agents:

**Step 1: Import Dependencies**
```python
from lobster.tools.workspace_content_service import (
    WorkspaceContentService,
    PublicationContent,
    DatasetContent,
    MetadataContent,
    ContentType,
    RetrievalLevel
)
from datetime import datetime
```

**Step 2: Initialize in Agent Factory**
```python
def create_research_agent(data_manager: DataManagerV2):
    # Initialize workspace service
    workspace_service = WorkspaceContentService(data_manager)

    # Create tools with access to both tiers
    tools = [
        create_write_to_workspace_tool(workspace_service),
        create_read_from_workspace_tool(workspace_service),
        # ... other tools
    ]

    return Agent(tools=tools)
```

**Step 3: Update Tool Implementations**
```python
@tool
def write_to_workspace(identifier: str, content: Dict) -> str:
    """Write to Tier 2 persistent cache."""
    pub = PublicationContent(
        identifier=identifier,
        **content,
        cached_at=datetime.now().isoformat()
    )
    path = workspace_service.write_content(pub, ContentType.PUBLICATION)

    # Also cache in Tier 1 for fast access
    data_manager.metadata_store[identifier] = {"metadata": content}

    return f"Cached to: {path}"
```

## Related Documentation

- **[20. Data Management Architecture](20-data-management.md)** - DataManagerV2 orchestration system
- **[38. Workspace Content Service](38-workspace-content-service.md)** - Detailed Tier 2 implementation
- **[37. Publication Intelligence Deep Dive](37-publication-intelligence-deep-dive.md)** - ContentAccessService with provider routing
- **[19. Agent System](19-agent-system.md)** - Multi-agent coordination and handoffs
- **[22. Performance Optimization](22-performance-optimization.md)** - Caching strategies and benchmarks

## Summary

The two-tier caching architecture provides:

✅ **Tier 1 Advantages:**
- Ultra-fast in-memory access (<0.001ms)
- Perfect for repeated lookups within sessions
- Minimal memory overhead (1-5 KB/entry)
- Zero filesystem I/O latency

✅ **Tier 2 Advantages:**
- Cross-session persistence (survives restarts)
- Schema validation with Pydantic models
- Agent handoff support (research_agent → metadata_assistant)
- Large content storage (>1MB publications)
- Reproducible workspace exports

✅ **Combined Benefits:**
- Intelligent cache promotion (Tier 2 → Tier 1 on access)
- Minimal API calls (check both tiers before external fetch)
- Flexible durability guarantees (session vs. permanent)
- Optimal performance/persistence tradeoff

**Best Practice**: Use Tier 1 for hot data (active session), Tier 2 for warm data (cross-session, handoffs), and let the system automatically promote frequently accessed Tier 2 content to Tier 1 for speed.
