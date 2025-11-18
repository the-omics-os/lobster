# Workspace Content Service

## Overview

The **WorkspaceContentService** provides structured, type-safe caching of research content (publications, datasets, metadata) in the DataManagerV2 workspace. Introduced in Lobster v0.2+, it replaces manual JSON file operations with a centralized service using Pydantic schemas for validation and enum-based type safety.

**Key Benefits:**
- **Type Safety**: Pydantic models validate all cached content
- **Enum-Based Validation**: ContentType and RetrievalLevel enums prevent string typos
- **Automatic File Management**: Professional naming conventions and directory organization
- **Level-Based Retrieval**: Flexible detail levels (summary/methods/samples/platform/full)
- **Workspace Integration**: Seamless integration with DataManagerV2 and research_agent tools

**Two-Tier Architecture:**
```
research_agent tools (write_to_workspace, get_content_from_workspace)
                    ↓
       WorkspaceContentService (validation, file I/O)
                    ↓
          DataManagerV2 workspace directory
                    ↓
         literature/  data/  metadata/  (JSON files)
```

## Architecture

### Content Types (Enum)

```python
from lobster.tools.workspace_content_service import ContentType

class ContentType(str, Enum):
    PUBLICATION = "publication"  # Research papers (PubMed, PMC, bioRxiv)
    DATASET = "dataset"          # GEO, SRA, PRIDE datasets
    METADATA = "metadata"        # Sample mappings, validation results, QC reports
```

**Workspace Directory Mapping:**
- `ContentType.PUBLICATION` → `workspace/literature/*.json`
- `ContentType.DATASET` → `workspace/data/*.json`
- `ContentType.METADATA` → `workspace/metadata/*.json`

### Retrieval Levels (Enum)

```python
from lobster.tools.workspace_content_service import RetrievalLevel

class RetrievalLevel(str, Enum):
    SUMMARY = "summary"      # Key-value overview (title, authors, sample count)
    METHODS = "methods"      # Methods section (publications only)
    SAMPLES = "samples"      # Sample IDs and metadata (datasets only)
    PLATFORM = "platform"    # Platform/technology info (datasets only)
    FULL = "full"            # All available content
```

**Level-Specific Fields:**

| Content Type | Summary | Methods | Samples | Platform | Full |
|-------------|---------|---------|---------|----------|------|
| **Publication** | identifier, title, authors, journal, year, keywords | identifier, title, methods | N/A | N/A | All fields |
| **Dataset** | identifier, title, sample_count, organism | N/A | identifier, sample_count, samples | identifier, platform, platform_id | All fields |
| **Metadata** | identifier, content_type, description, related_datasets | N/A | N/A | N/A | All fields |

### Pydantic Content Schemas

#### PublicationContent

```python
from lobster.tools.workspace_content_service import PublicationContent

pub = PublicationContent(
    identifier="PMID:35042229",
    title="Single-cell RNA-seq reveals...",
    authors=["Smith J", "Jones A"],
    journal="Nature",
    year=2022,
    abstract="We performed single-cell RNA-seq...",
    methods="Cells were processed using 10X Chromium...",
    full_text="...",  # Complete paper text
    keywords=["single-cell", "RNA-seq", "cancer"],
    source="PMC",  # PMC, PubMed, bioRxiv
    cached_at="2025-01-12T10:30:00",  # ISO 8601 timestamp
    url="https://pubmed.ncbi.nlm.nih.gov/35042229/"
)
```

**Fields:**
- `identifier` (required): PMID, DOI, or bioRxiv ID
- `title`, `authors`, `journal`, `year`: Bibliographic metadata
- `abstract`, `methods`, `full_text`: Content sections
- `keywords`: Publication keywords (MeSH terms, author keywords)
- `source` (required): Provider (PMC, PubMed, bioRxiv, medRxiv)
- `cached_at` (required): ISO 8601 timestamp
- `url`: Publication URL

#### DatasetContent

```python
from lobster.tools.workspace_content_service import DatasetContent

dataset = DatasetContent(
    identifier="GSE123456",
    title="Single-cell RNA-seq of aging brain",
    platform="Illumina NovaSeq 6000",
    platform_id="GPL24676",
    organism="Homo sapiens",
    sample_count=12,
    samples={
        "GSM1": {"age": 25, "tissue": "brain"},
        "GSM2": {"age": 65, "tissue": "brain"}
    },
    experimental_design="Age comparison: young (n=6) vs old (n=6)",
    summary="Dataset comparing transcriptional changes...",
    pubmed_ids=["35042229"],
    source="GEO",  # GEO, SRA, PRIDE
    cached_at="2025-01-12T10:30:00",
    url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456"
)
```

**Fields:**
- `identifier` (required): GSE, SRA, PRIDE accession
- `title`, `summary`: Dataset descriptions
- `platform`, `platform_id`: Technology information
- `organism`: Species (e.g., Homo sapiens, Mus musculus)
- `sample_count` (required): Number of samples (≥0)
- `samples`: Dictionary mapping sample IDs to metadata
- `experimental_design`: Study design description
- `pubmed_ids`: Associated publications
- `source` (required): Repository (GEO, SRA, PRIDE)
- `cached_at` (required): ISO 8601 timestamp
- `url`: Dataset URL

#### MetadataContent

```python
from lobster.tools.workspace_content_service import MetadataContent

metadata = MetadataContent(
    identifier="gse12345_to_gse67890_mapping",
    content_type="sample_mapping",
    description="Sample ID mapping between two datasets",
    data={
        "exact_matches": 10,
        "fuzzy_matches": 5,
        "unmapped": 2,
        "mapping_rate": 0.88
    },
    related_datasets=["GSE12345", "GSE67890"],
    source="SampleMappingService",
    cached_at="2025-01-12T10:30:00"
)
```

**Fields:**
- `identifier` (required): Unique metadata identifier
- `content_type` (required): Type descriptor (sample_mapping, validation, qc_report, etc.)
- `description`: Human-readable description
- `data` (required): Arbitrary JSON-serializable content
- `related_datasets`: Related dataset accessions
- `source` (required): Tool or service name
- `cached_at` (required): ISO 8601 timestamp

## Service API

### Initialization

```python
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_content_service import WorkspaceContentService

data_manager = DataManagerV2(workspace_path="~/.lobster_workspace")
workspace_service = WorkspaceContentService(data_manager=data_manager)
```

**Directory Structure Created:**
```
workspace_path/
├── literature/    # Publications (PublicationContent)
├── data/          # Datasets (DatasetContent)
└── metadata/      # Metadata (MetadataContent)
```

### Writing Content

```python
from lobster.tools.workspace_content_service import (
    PublicationContent,
    ContentType,
    WorkspaceContentService
)
from datetime import datetime

# Create content model
pub_content = PublicationContent(
    identifier="PMID:35042229",
    title="Single-cell analysis of aging",
    authors=["Smith J", "Jones A"],
    journal="Nature",
    year=2022,
    abstract="Abstract text...",
    methods="Methods text...",
    source="PMC",
    cached_at=datetime.now().isoformat()
)

# Write to workspace
cache_path = workspace_service.write_content(
    content=pub_content,
    content_type=ContentType.PUBLICATION
)
# Returns: "/workspace/literature/pmid_35042229.json"
```

**Naming Convention:**
- Identifier sanitized: lowercase, special characters → underscores
- `PMID:35042229` → `pmid_35042229.json`
- `GSE123456` → `gse123456.json`
- `DOI:10.1038/s41586-021-12345-6` → `doi_10_1038_s41586_021_12345_6.json`

### Reading Content

#### Basic Retrieval

```python
from lobster.tools.workspace_content_service import ContentType, RetrievalLevel

# Read full content
full_content = workspace_service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.FULL
)
# Returns: Dict with all fields

# Read summary only
summary = workspace_service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.SUMMARY
)
# Returns: Dict with identifier, title, authors, journal, year, keywords

# Read methods section
methods = workspace_service.read_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION,
    level=RetrievalLevel.METHODS
)
# Returns: Dict with identifier, title, methods
```

#### Dataset Retrieval Examples

```python
# Get dataset summary
summary = workspace_service.read_content(
    identifier="GSE123456",
    content_type=ContentType.DATASET,
    level=RetrievalLevel.SUMMARY
)
# Returns: identifier, title, sample_count, organism

# Get sample metadata
samples = workspace_service.read_content(
    identifier="GSE123456",
    content_type=ContentType.DATASET,
    level=RetrievalLevel.SAMPLES
)
# Returns: identifier, sample_count, samples, experimental_design

# Get platform information
platform = workspace_service.read_content(
    identifier="GSE123456",
    content_type=ContentType.DATASET,
    level=RetrievalLevel.PLATFORM
)
# Returns: identifier, platform, platform_id, organism
```

### Listing Content

```python
# List all cached content
all_content = workspace_service.list_content()
# Returns: List[Dict] with all publications, datasets, metadata

# List only publications
publications = workspace_service.list_content(
    content_type=ContentType.PUBLICATION
)
# Returns: List[Dict] with publication metadata

# List only datasets
datasets = workspace_service.list_content(
    content_type=ContentType.DATASET
)
# Returns: List[Dict] with dataset metadata
```

**List Result Format:**
```python
[
    {
        "identifier": "PMID:35042229",
        "title": "Single-cell analysis...",
        "authors": ["Smith J", "Jones A"],
        "cached_at": "2025-01-12T10:30:00",
        "_content_type": "publication",  # Added by service
        "_file_path": "/workspace/literature/pmid_35042229.json"  # Added by service
    },
    # ... more items
]
```

### Deleting Content

```python
# Delete cached publication
deleted = workspace_service.delete_content(
    identifier="PMID:35042229",
    content_type=ContentType.PUBLICATION
)
# Returns: True if deleted, False if not found
```

### Workspace Statistics

```python
stats = workspace_service.get_workspace_stats()
# Returns:
# {
#     "total_items": 42,
#     "publications": 15,
#     "datasets": 20,
#     "metadata": 7,
#     "total_size_mb": 12.5,
#     "cache_dir": "/workspace/cache/content"
# }
```

## Integration with research_agent Tools

The research_agent provides two tools that use WorkspaceContentService under the hood:

### write_to_workspace Tool

**Purpose**: Cache research content for persistent access and specialist handoff.

**Usage Pattern:**
```python
# In research_agent tool
from lobster.tools.workspace_content_service import (
    ContentType,
    PublicationContent,
    WorkspaceContentService
)

@tool
def write_to_workspace(identifier: str, workspace: str, content_type: str = None) -> str:
    # 1. Initialize service
    workspace_service = WorkspaceContentService(data_manager=data_manager)

    # 2. Map workspace categories to ContentType enum
    workspace_to_content_type = {
        "literature": ContentType.PUBLICATION,
        "data": ContentType.DATASET,
        "metadata": ContentType.METADATA,
    }

    # 3. Validate workspace category
    if workspace not in workspace_to_content_type:
        return f"Error: Invalid workspace '{workspace}'"

    # 4. Retrieve content from data_manager
    if identifier in data_manager.metadata_store:
        content_data = data_manager.metadata_store[identifier]
    elif identifier in data_manager.list_modalities():
        adata = data_manager.get_modality(identifier)
        content_data = {...}  # Extract metadata
    else:
        return f"Error: Identifier '{identifier}' not found"

    # 5. Create Pydantic model
    content_model = PublicationContent(
        identifier=identifier,
        # ... populate fields
        cached_at=datetime.now().isoformat()
    )

    # 6. Write using service
    cache_path = workspace_service.write_content(
        content=content_model,
        content_type=workspace_to_content_type[workspace]
    )

    return f"Cached to {cache_path}"
```

**Naming Conventions:**
- Publications: `publication_PMID12345` or `publication_DOI...`
- Datasets: `dataset_GSE12345`
- Metadata: `metadata_GSE12345_samples`

**Example:**
```bash
# Cache publication after reading
> "I just read PMID:35042229. Please cache it for later."
→ write_to_workspace("publication_PMID35042229", workspace="literature", content_type="publication")

# Cache dataset metadata
> "Cache GSE123456 metadata for validation."
→ write_to_workspace("dataset_GSE123456", workspace="data", content_type="dataset")
```

### get_content_from_workspace Tool

**Purpose**: Retrieve cached research content with flexible detail levels.

**Usage Pattern:**
```python
@tool
def get_content_from_workspace(
    identifier: str = None,
    workspace: str = None,
    level: str = "summary"
) -> str:
    # 1. Initialize service
    workspace_service = WorkspaceContentService(data_manager=data_manager)

    # 2. Map strings to enums
    workspace_to_content_type = {
        "literature": ContentType.PUBLICATION,
        "data": ContentType.DATASET,
        "metadata": ContentType.METADATA,
    }

    level_to_retrieval = {
        "summary": RetrievalLevel.SUMMARY,
        "methods": RetrievalLevel.METHODS,
        "samples": RetrievalLevel.SAMPLES,
        "platform": RetrievalLevel.PLATFORM,
        "metadata": RetrievalLevel.FULL,
    }

    # 3. List mode (no identifier)
    if identifier is None:
        content_type_filter = workspace_to_content_type[workspace] if workspace else None
        all_cached = workspace_service.list_content(content_type=content_type_filter)
        return format_list_response(all_cached)

    # 4. Retrieve mode (with identifier)
    retrieval_level = level_to_retrieval[level]

    # Try each content type if workspace not specified
    content_types_to_try = (
        [workspace_to_content_type[workspace]] if workspace
        else list(ContentType)
    )

    for content_type in content_types_to_try:
        try:
            cached_content = workspace_service.read_content(
                identifier=identifier,
                content_type=content_type,
                level=retrieval_level
            )
            return format_response(cached_content, level)
        except FileNotFoundError:
            continue

    return f"Error: Identifier '{identifier}' not found"
```

**Examples:**

```bash
# List all cached content
> "What content do I have cached?"
→ get_content_from_workspace()

# List publications only
> "Show me cached publications."
→ get_content_from_workspace(workspace="literature")

# Get publication methods section
> "Show methods from PMID:35042229."
→ get_content_from_workspace(
    identifier="publication_PMID35042229",
    workspace="literature",
    level="methods"
)

# Get dataset samples
> "Show sample IDs for GSE123456."
→ get_content_from_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    level="samples"
)

# Get full metadata
> "Show full metadata for my sample mapping."
→ get_content_from_workspace(
    identifier="metadata_gse12345_to_gse67890_mapping",
    workspace="metadata",
    level="metadata"
)
```

## Common Workflows

### Workflow 1: Cache Publication for Later Analysis

```python
# 1. Search literature
search_literature("BRCA1 breast cancer", max_results=5)

# 2. Read full publication
read_full_publication("PMID:35042229")
# → Content automatically cached in metadata_store

# 3. Cache to workspace
write_to_workspace(
    identifier="publication_PMID35042229",
    workspace="literature",
    content_type="publication"
)

# 4. Later: Retrieve methods section
get_content_from_workspace(
    identifier="publication_PMID35042229",
    workspace="literature",
    level="methods"
)
```

### Workflow 2: Cache Dataset Before Handoff to Specialist

```python
# 1. Discover dataset
find_related_entries("PMID:35042229", entry_type="dataset")
# → Found: GSE123456

# 2. Get dataset metadata
get_dataset_metadata("GSE123456")
# → Metadata stored in metadata_store

# 3. Cache to workspace before handoff
write_to_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    content_type="dataset"
)

# 4. Hand off to metadata_assistant
handoff_to_metadata_assistant(
    instructions="Validate GSE123456 for treatment_response field. "
                "Dataset cached in data workspace."
)
```

### Workflow 3: Multiple Detail Levels

```python
# Start with summary
get_content_from_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    level="summary"
)
# → Returns: title, sample_count, organism

# Need more details? Get samples
get_content_from_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    level="samples"
)
# → Returns: sample IDs and metadata

# Need platform info?
get_content_from_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    level="platform"
)
# → Returns: platform, platform_id, organism

# Need everything?
get_content_from_workspace(
    identifier="dataset_GSE123456",
    workspace="data",
    level="metadata"
)
# → Returns: all fields
```

## Best Practices

### Naming Conventions

**Follow Professional Naming:**
- Lowercase identifiers
- Underscores for separators
- Descriptive prefixes

```python
# ✅ Good
"publication_PMID35042229"
"dataset_GSE123456"
"metadata_gse12345_to_gse67890_mapping"

# ❌ Bad
"PMID:35042229"  # Contains colon
"GSE 123456"      # Contains space
"mapping-12345"   # Ambiguous prefix
```

### Content Validation

**Always Use Pydantic Models:**
```python
# ✅ Good - Validation enforced
pub_content = PublicationContent(
    identifier="PMID:35042229",
    source="PMC",
    cached_at=datetime.now().isoformat()
)
workspace_service.write_content(pub_content, ContentType.PUBLICATION)

# ❌ Bad - No validation
raw_dict = {"identifier": "PMID:35042229"}  # Missing required fields
# Will fail validation
```

### Error Handling

**Handle FileNotFoundError:**
```python
from lobster.tools.workspace_content_service import ContentType, RetrievalLevel

try:
    content = workspace_service.read_content(
        identifier="publication_PMID12345",
        content_type=ContentType.PUBLICATION,
        level=RetrievalLevel.SUMMARY
    )
except FileNotFoundError as e:
    logger.warning(f"Content not found: {e}")
    # List available content
    available = workspace_service.list_content(ContentType.PUBLICATION)
    logger.info(f"Available publications: {[c['identifier'] for c in available]}")
```

### Level Selection

**Choose Appropriate Detail Level:**

| Use Case | Recommended Level | Why |
|----------|------------------|-----|
| Quick overview | `SUMMARY` | Fast, minimal data transfer |
| Replication protocol | `METHODS` | Focused on procedures |
| Sample alignment | `SAMPLES` | Just sample metadata |
| Platform validation | `PLATFORM` | Technology compatibility check |
| Full export | `FULL` | Complete content for archival |

### Workspace Organization

**Categorize Content by Type:**
```python
# Literature review project
workspace_service.write_content(pub1, ContentType.PUBLICATION)  # → literature/
workspace_service.write_content(pub2, ContentType.PUBLICATION)  # → literature/

# Dataset analysis project
workspace_service.write_content(dataset1, ContentType.DATASET)  # → data/
workspace_service.write_content(dataset2, ContentType.DATASET)  # → data/

# Metadata operations
workspace_service.write_content(mapping, ContentType.METADATA)  # → metadata/
```

### Backward Compatibility

**Maintain Tool Signatures:**
- Both tools (`write_to_workspace`, `get_content_from_workspace`) maintain original signatures
- String-based parameters at tool level
- Enum conversion happens internally
- Same response formats as before refactoring

## Performance Considerations

### Caching Strategy

**When to Cache:**
- ✅ After expensive operations (PDF parsing, full-text extraction)
- ✅ Before handing off to other agents (context preservation)
- ✅ When content will be reused (literature reviews, multi-step workflows)

**When NOT to Cache:**
- ❌ Temporary scratch data
- ❌ Duplicates of in-memory modalities
- ❌ Large binary files (use modalities storage instead)

### File Size Management

**Monitor Workspace Size:**
```python
stats = workspace_service.get_workspace_stats()
if stats["total_size_mb"] > 100:
    logger.warning("Workspace size exceeding 100MB")
    # Consider cleaning old cached content
```

**Delete Old Content:**
```python
# Remove cached content no longer needed
workspace_service.delete_content(
    identifier="old_publication_PMID12345",
    content_type=ContentType.PUBLICATION
)
```

## Troubleshooting

### Common Issues

**Issue: "File has been modified since read"**
- **Cause**: Auto-formatter/linter running between Read and Edit
- **Solution**: Read larger context window (400+ lines) before editing

**Issue: "Invalid workspace 'xyz'"**
- **Cause**: Typo in workspace parameter
- **Solution**: Use enum mapping: `"literature"`, `"data"`, or `"metadata"`

**Issue: "Invalid detail level 'abc'"**
- **Cause**: Unsupported level string
- **Solution**: Use valid levels: `"summary"`, `"methods"`, `"samples"`, `"platform"`, `"metadata"`

**Issue: "ValidationError: Field required"**
- **Cause**: Missing required Pydantic fields
- **Solution**: Check schema requirements (identifier, source, cached_at)

### Debugging

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Service operations will log:
# - File paths created
# - Content validated
# - Errors encountered
```

**Inspect Workspace Contents:**
```bash
# Check cached files
ls -lh ~/.lobster_workspace/literature/
ls -lh ~/.lobster_workspace/data/
ls -lh ~/.lobster_workspace/metadata/

# View JSON content
cat ~/.lobster_workspace/literature/pmid_35042229.json | jq .
```

## Migration from Manual JSON Handling

### Before (Manual Implementation)

```python
# Old approach - manual file operations
import json
from pathlib import Path

cache_dir = Path(workspace_path) / "literature"
cache_file = cache_dir / f"{identifier.lower()}.json"

# Write
with open(cache_file, "w") as f:
    json.dump({"identifier": identifier, ...}, f)

# Read
with open(cache_file, "r") as f:
    content = json.load(f)

# List
cached_files = list(cache_dir.glob("*.json"))
```

### After (WorkspaceContentService)

```python
# New approach - service-based
from lobster.tools.workspace_content_service import (
    WorkspaceContentService,
    PublicationContent,
    ContentType,
    RetrievalLevel
)

workspace_service = WorkspaceContentService(data_manager=data_manager)

# Write
pub_content = PublicationContent(identifier=identifier, ...)
workspace_service.write_content(pub_content, ContentType.PUBLICATION)

# Read
content = workspace_service.read_content(
    identifier, ContentType.PUBLICATION, RetrievalLevel.SUMMARY
)

# List
cached_list = workspace_service.list_content(ContentType.PUBLICATION)
```

**Benefits:**
- ✅ Pydantic validation (catch errors early)
- ✅ Enum type safety (no string typos)
- ✅ Automatic directory management
- ✅ Level-based filtering (no manual if/elif chains)
- ✅ Professional naming (automatic sanitization)

## Version History

| Version | Changes |
|---------|---------|
| **v0.2+** | Initial implementation with Pydantic schemas, enum-based validation, two-tier architecture |

## Related Documentation

- [Data Management (DataManagerV2)](20-data-management.md) - Multi-modal data orchestration
- [Services API Reference](16-services-api.md) - Service design patterns
- [Creating Services](10-creating-services.md) - Service development guidelines
- [Agent API Reference](15-agents-api.md) - research_agent tool integration

## See Also

- **WorkspaceContentService Source**: `lobster/tools/workspace_content_service.py` (714 lines)
- **Pydantic Schemas**: PublicationContent, DatasetContent, MetadataContent
- **Integration**: research_agent tools (write_to_workspace, get_content_from_workspace)
- **Testing**: `tests/integration/test_workspace_content_service.py`
