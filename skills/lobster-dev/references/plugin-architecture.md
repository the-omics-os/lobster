# Omics Plugin Architecture

How to extend Lobster AI with new omics types, database providers, and data adapters — without modifying core code.

**Full documentation:** [docs.omics-os.com/docs/architecture/overview](https://docs.omics-os.com/docs/architecture/overview) (Omics Plugin Architecture section)
**Entry point guide:** [docs.omics-os.com/docs/extending/entry-points](https://docs.omics-os.com/docs/extending/entry-points)
**Plugin contract:** [docs.omics-os.com/docs/extending/plugin-contract](https://docs.omics-os.com/docs/extending/plugin-contract)

## Quick Reference

**Two registries:**
- `OmicsTypeRegistry` (`core/omics_registry.py`) — type metadata: detection config, preferred databases, QC thresholds
- `ComponentRegistry` (`core/component_registry.py`) — component instances: adapters, providers, download services, queue preparers

**7 entry point groups:**

| Group | Discovers | Factory Contract |
|-------|-----------|-----------------|
| `lobster.agents` | Agent configs | `AGENT_CONFIG` object |
| `lobster.services` | Service classes | Class reference |
| `lobster.agent_configs` | Custom LLM configs | Config object |
| `lobster.adapters` | Adapter factories | **Callable returning configured instance** |
| `lobster.providers` | Provider classes | Class (instantiated with `data_manager`) |
| `lobster.download_services` | Download service classes | Class (instantiated with `data_manager`) |
| `lobster.queue_preparers` | Queue preparer classes | Class (instantiated with `data_manager`) |

**Built-in omics types (5):** transcriptomics, proteomics, genomics, metabolomics, metagenomics

## Adding a New Omics Type (6 Steps)

1. Define `OmicsTypeConfig` with detection keywords, preferred databases, QC thresholds
2. Create adapter factory → `lobster.adapters` entry point
3. Create provider (if new database) → `lobster.providers` entry point
4. Create download service → `lobster.download_services` entry point
5. Create queue preparer → `lobster.queue_preparers` entry point
6. Register config → `lobster.omics_types` entry point

**Full walkthrough with code:** [docs.omics-os.com/docs/extending/entry-points#adding-a-new-omics-type-via-plugins](https://docs.omics-os.com/docs/extending/entry-points)

### Minimal pyproject.toml

```toml
[project.entry-points."lobster.omics_types"]
epigenomics = "my_package.config:EPIGENOMICS_CONFIG"

[project.entry-points."lobster.adapters"]
epigenomics_chip_seq = "my_package.adapter:create_chip_seq_adapter"

[project.entry-points."lobster.providers"]
encode = "my_package.provider:ENCODEProvider"

[project.entry-points."lobster.download_services"]
encode = "my_package.download_service:ENCODEDownloadService"

[project.entry-points."lobster.queue_preparers"]
encode = "my_package.queue_preparer:ENCODEQueuePreparer"
```

### OmicsTypeConfig Template

```python
from lobster.core.omics_registry import OmicsTypeConfig, OmicsDetectionConfig

MY_CONFIG = OmicsTypeConfig(
    name="epigenomics",
    display_name="Epigenomics",
    schema_class="my_package.schema:EpigenomicsSchema",
    adapter_names=["epigenomics_chip_seq"],
    preferred_databases=["geo", "encode"],
    data_type_aliases=["chip_seq", "atac_seq"],
    detection=OmicsDetectionConfig(
        keywords=["ChIP-seq", "ATAC-seq", "histone", "methylation", "chromatin"],
        feature_count_range=(10_000, 500_000),
        weight=10,
    ),
    qc_thresholds={"min_reads": 10_000_000},
)
```

### Adapter Factory Pattern

```python
from lobster.core.adapters.base import BaseAdapter

class EpigenomicsAdapter(BaseAdapter):
    def __init__(self, data_type="chip_seq"):
        super().__init__(name="EpigenomicsAdapter")
        self.data_type = data_type
    # Implement: from_source, validate, get_schema, preprocess_data

def create_chip_seq_adapter():
    return EpigenomicsAdapter(data_type="chip_seq")
```

## DataTypeDetector

Unified detection — replaces scattered `_is_single_cell_dataset()`, `_is_proteomics_dataset()`, etc.

```python
from lobster.core.omics_registry import DataTypeDetector

detector = DataTypeDetector()
results = detector.detect_from_metadata({"title": "ChIP-seq of histone marks"})
# => [("epigenomics", 0.85), ("genomics", 0.12), ...]

results = detector.detect_from_data(adata)        # Feature count + column signatures
results = detector.detect(metadata=meta, adata=adata)  # Combined (60% metadata, 40% data)
```

## Registration Flow

All 4 registration sites (adapters, providers, download services, queue preparers):

```
Phase 1: Entry-point discovery → ComponentRegistry
Phase 2: Hardcoded fallback (only if name not already registered)
```

Faulty plugins get logged and skipped — never crash the system.

## Reference Implementation

Study the metabolomics stack for canonical patterns:

| Component | File |
|-----------|------|
| OmicsTypeConfig | `core/omics_registry.py` (metabolomics section) |
| MetabolomicsAdapter | `core/adapters/metabolomics_adapter.py` |
| MetaboLightsProvider | `tools/providers/metabolights_provider.py` |
| MetaboLightsDownloadService | `services/data_access/metabolights_download_service.py` |
| MetaboLightsQueuePreparer | `services/data_access/metabolights_queue_preparer.py` |

## Key Files

| File | Purpose |
|------|---------|
| `core/omics_registry.py` | `OmicsTypeConfig`, `OMICS_TYPE_REGISTRY`, `DataTypeDetector` |
| `core/component_registry.py` | All 7 entry point groups + get/list/has APIs |
| `core/adapters/base.py` | `BaseAdapter` (inherit for new adapters) |
| `core/interfaces/download_service.py` | `IDownloadService` interface |
| `core/interfaces/queue_preparer.py` | `IQueuePreparer` interface |
| `tools/providers/base_provider.py` | `BasePublicationProvider` + enums |
