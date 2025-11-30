# CLAUDE.md

System prompt for Lobster AI — a professional multi‑agent bioinformatics analysis platform.  
Goal: help senior engineers understand **architecture, file layout, and non‑negotiable rules** with minimal tokens.

---

## 1. WHAT – System Overview

### 1.1 Project Overview

Lobster AI is a **multi‑agent bioinformatics analysis platform** for complex multi‑omics data (scRNA‑seq, bulk RNA‑seq, proteomics, etc.).  
Users interact via natural language to:

- search publications & datasets
- run end‑to‑end analyses
- export **reproducible Jupyter notebooks** (Papermill)

### 1.2 Core Capabilities (high level)

| Domain | Capabilities (examples) |
|--------|-------------------------|
| **Single‑Cell RNA‑seq** | QC, clustering, cell type annotation, trajectory, pseudobulk |
| **Bulk RNA‑seq** | Kallisto/Salmon import, DE with pyDESeq2, formula‑based designs |
| **Mass Spec Proteomics** | DDA/DIA, missing value handling, peptide→protein, normalization |
| **Affinity Proteomics** | Olink / antibody arrays, NPX handling, CV analysis |
| **Multi‑Omics (future)** | MuData‑based cross‑modality analysis |
| **Literature Mining** | PubMed/GEO search, parameter & metadata extraction |
| **Notebook Export** | Reproducible, parameterizable workflows via Papermill |

### 1.3 Data & Storage

- **Inputs**: CSV, Excel, H5AD, 10X MTX, Kallisto (abundance.tsv), Salmon (quant.sf), MaxQuant, Spectronaut, Olink NPX  
- **Repositories**: GEO, SRA, ENA (PRIDE/massIVE/Uniprot planned)  
- **Storage**: H5AD (single), MuData (multi‑modal), JSONL queues, S3‑ready backends  

### 1.4 Design Principles

1. **Agent‑based**: specialist agents + centralized registry (single source of truth).
2. **Cloud/local symmetry**: same UX, different `Client` backend.
3. **Stateless services**: analysis logic lives in services/, organized by function.
4. **Natural language first**: users describe analyses in plain English.
5. **Publication‑grade output**: Plotly‑based, scientifically sound.
6. **Extensible & professional**: new agents/services plug into common patterns.

---

## 2. HOW – Architecture

### 2.1 4‑Layer Architecture (critical path)

**Critical flow**: `CLI → LobsterClientAdapter → AgentClient | CloudLobsterClient → LangGraph → Agents → Services → DataManagerV2`

```mermaid
graph TB
  User --> CLI["lobster/cli.py
CLI"]
  CLI --> Adapter["LobsterClientAdapter
client detection"]

  Adapter -->|cloud key set| Cloud["CloudLobsterClient"]
  Adapter -->|default/local| Local["AgentClient
(core/client.py)"]

  Local --> Graph["LangGraph runtime
create_bioinformatics_graph"]
  Cloud --> Graph

  Graph --> Sup["Supervisor agent"]
  Sup --> Registry["Agent registry
config/agent_registry.py"]
  Registry --> Agents["Specialist agents"]

  Agents --> Services["Stateless services
lobster/services"]
  Agents --> Tools["Utilities
lobster/tools"]
  Agents --> Providers["Providers
tools/providers"]
  Services --> DM["DataManagerV2
core/data_manager_v2.py"]
  Tools --> DM
  Providers --> DM

  DM --> Queue["DownloadQueue
core/download_queue.py"]
  DM --> Prov["ProvenanceTracker
core/provenance.py"]
  DM --> Export["NotebookExporter
core/notebook_exporter.py"]
  DM --> Storage["Backends (H5AD/MuData/S3)
core/backends"]
  DM --> Schemas["Pydantic schemas
core/schemas"]
```

### 2.2 Client Layer (core)

**Location**: `lobster/core/client.py`

`AgentClient` is the **main orchestrator** for local runs:

- creates `DataManagerV2` + workspace
- builds LangGraph via `create_bioinformatics_graph(...)`
- routes user queries through the graph
- exposes status / export APIs

Key methods:

- `query(user_input, stream=False)` – run request through graph  
- `get_status()` – current session + data summary  
- `export_session()` – conversation + workspace snapshot  

Routing:

- `LobsterClientAdapter` (in `cli.py`) checks `LOBSTER_CLOUD_KEY`  
  - if set → `CloudLobsterClient`  
  - else → local `AgentClient`

### 2.3 Data & Control Flow (summary)

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI
    participant CL as Client (AgentClient/Cloud)
    participant G as LangGraph+Supervisor
    participant A as Specialist Agent
    participant S as Service
    participant D as DataManagerV2
    participant ST as Storage/Provenance

    U->>C: "Cluster my single‑cell data"
    C->>CL: CLI command (chat/query)
    CL->>G: execute graph
    G->>A: route to singlecell_expert
    A->>S: call ClusteringService
    S->>S: process AnnData
    S-->>A: (processed_adata, stats, ir)
    A->>D: store modality + log(ir, stats)
    D->>ST: persist data + provenance
    G-->>CL: final message
    CL-->>C: formatted response
    C-->>U: natural language result
```

### 2.4 Download Queue Pattern (multi‑agent handoff)

```mermaid
sequenceDiagram
    participant R as research_agent
    participant G as GEOProvider
    participant Q as DownloadQueue
    participant S as supervisor
    participant DE as data_expert

    R->>G: get_download_urls(GSE)
    G-->>R: URLs
    R->>Q: create DownloadQueueEntry (PENDING)
    S->>Q: poll queue
    S-->>DE: handoff if PENDING
    DE->>Q: mark IN_PROGRESS
    DE->>DE: download + load data
    DE->>Q: update COMPLETED/FAILED
```

---

## 3. WHERE – Code Layout

### 3.1 Top‑Level Structure

```text
lobster/
├─ cli.py                   # CLI entrypoint (Typer/Rich)
├─ agents/                  # Supervisor + specialist agents + graph
│  ├─ supervisor.py
│  ├─ research_agent.py
│  ├─ metadata_assistant.py
│  ├─ data_expert.py
│  ├─ singlecell_expert.py
│  ├─ bulk_rnaseq_expert.py
│  ├─ ms_proteomics_expert.py
│  ├─ affinity_proteomics_expert.py
│  └─ graph.py              # create_bioinformatics_graph(...)
│
├─ core/                    # Client, data, provenance, backends
│  ├─ client.py             # AgentClient (local)
│  ├─ api_client.py         # Cloud/WebSocket client
│  ├─ data_manager_v2.py
│  ├─ provenance.py
│  ├─ download_queue.py
│  ├─ notebook_exporter.py
│  ├─ notebook_executor.py
│  ├─ interfaces/
│  │  ├─ base_client.py
│  │  ├─ data_backend.py
│  │  └─ modality_adapter.py
│  ├─ backends/
│  │  ├─ h5ad_backend.py
│  │  └─ mudata_backend.py
│  └─ schemas/
│     ├─ transcriptomics_schema.py
│     ├─ proteomics_schema.py
│     ├─ metabolomics_schema.py
│     └─ metagenomics_schema.py
│
├─ services/                # Stateless analysis services (organized by function)
│  ├─ analysis/             # Analysis services
│  │  ├─ clustering_service.py
│  │  ├─ enhanced_singlecell_service.py
│  │  ├─ bulk_rnaseq_service.py
│  │  ├─ differential_formula_service.py
│  │  ├─ pseudobulk_service.py
│  │  ├─ scvi_embedding_service.py
│  │  ├─ proteomics_analysis_service.py
│  │  ├─ proteomics_differential_service.py
│  │  └─ structure_analysis_service.py
│  ├─ quality/              # Quality control & preprocessing
│  │  ├─ quality_service.py
│  │  ├─ preprocessing_service.py
│  │  ├─ proteomics_quality_service.py
│  │  └─ proteomics_preprocessing_service.py
│  ├─ visualization/        # Visualization services
│  │  ├─ visualization_service.py
│  │  ├─ bulk_visualization_service.py
│  │  ├─ proteomics_visualization_service.py
│  │  ├─ pymol_visualization_service.py
│  │  └─ chimerax_visualization_service_ALPHA.py
│  ├─ data_access/          # Data access & retrieval
│  │  ├─ geo_service.py     # Main GEO service (imports from geo/ subpackage)
│  │  ├─ geo/               # Modular GEO package (refactored Nov 2024)
│  │  │  ├─ __init__.py     # Re-exports with lazy GEOService import
│  │  │  ├─ constants.py    # Enums, dataclasses, platform registry
│  │  │  ├─ downloader.py   # GEODownloadManager (moved from tools/)
│  │  │  ├─ parser.py       # GEOParser (moved from tools/)
│  │  │  └─ strategy.py     # PipelineStrategyEngine (moved from tools/)
│  │  ├─ geo_download_service.py
│  │  ├─ geo_fallback_service.py
│  │  ├─ content_access_service.py
│  │  ├─ workspace_content_service.py
│  │  ├─ protein_structure_fetch_service.py
│  │  └─ docling_service.py
│  ├─ data_management/      # Data management & organization
│  │  ├─ modality_management_service.py
│  │  └─ concatenation_service.py
│  ├─ metadata/             # Metadata operations
│  │  ├─ metadata_standardization_service.py
│  │  ├─ metadata_validation_service.py
│  │  ├─ disease_standardization_service.py
│  │  ├─ sample_mapping_service.py
│  │  ├─ microbiome_filtering_service.py
│  │  └─ manual_annotation_service.py
│  ├─ ml/                   # Machine learning services
│  │  ├─ ml_transcriptomics_service_ALPHA.py
│  │  └─ ml_proteomics_service_ALPHA.py
│  ├─ orchestration/        # Orchestration & workflow
│  │  └─ publication_processing_service.py
│  └─ templates/            # Templates & configurations
│     └─ annotation_templates.py
│
├─ tools/                   # Utilities & orchestrators
│  ├─ download_orchestrator.py
│  ├─ handoff_tools.py
│  ├─ workspace_tool.py
│  ├─ gpu_detector.py
│  └─ providers/
│     ├─ base_provider.py
│     ├─ pubmed_provider.py
│     ├─ pmc_provider.py
│     ├─ geo_provider.py
│     ├─ webpage_provider.py
│     └─ abstract_provider.py
│
├─ config/
│  ├─ agent_registry.py     # AGENT_REGISTRY (single source of truth)
│  ├─ agent_config.py
│  └─ settings.py
```

### 3.2 Core Components Reference

| Area | File(s) | Role |
|------|---------|------|
| CLI | `cli.py` | commands, client routing |
| Client | `core/client.py`, `core/api_client.py` | local vs cloud clients |
| Graph | `agents/graph.py` | `create_bioinformatics_graph()` |
| Agents | `agents/*.py` | supervisor + specialists |
| Data | `core/data_manager_v2.py` | modality/workspace orchestration |
| Provenance | `core/provenance.py` | W3C‑PROV tracking |
| Queue | `core/download_queue.py` | download orchestration |
| Concurrency | `core/queue_storage.py` | multi-process safe file locking & atomic writes |
| Rate Limiting | `tools/rate_limiter.py` | Redis connection pool for NCBI API rate limiting |
| Workspace | `core/workspace.py` | centralized workspace path resolution |
| Export | `core/notebook_exporter.py` | Jupyter pipeline export |
| Services | `services/*/*.py` | stateless analysis (organized by function) |
| Services | `services/data_management/modality_management_service.py` | Modality CRUD with provenance (5 methods) |
| Download | `tools/download_orchestrator.py` | Central router for database-specific downloads (9-step execution) |
| Download | `services/data_access/geo_download_service.py` | GEO database download service (IDownloadService impl) |
| GEO | `services/data_access/geo/` | Modular GEO package (downloader, parser, strategy, constants) |
| Interfaces | `core/interfaces/download_service.py` | IDownloadService abstract base class |
| Providers | `tools/providers/*.py` | PubMed/GEO/Web access |
| Utilities | `tools/*.py` | orchestrators, helpers, workspace tools (write_to_workspace, get_content_from_workspace shared across agents) |
| Deprecated | `tools/geo_*.py`, `tools/pipeline_strategy.py` | Backward compat aliases → `services/data_access/geo/` |
| Registry | `config/agent_registry.py` | agent configuration |

### 3.3 Agent Roles (summary)

| Agent | Main Focus |
|-------|------------|
| `supervisor` | route user intents to specialists, manage handoffs |
| `research_agent` | literature & dataset discovery, URL extraction, workspace caching, publication queue processing |
| `metadata_assistant` | ID mapping, schema‑based validation, harmonization, **publication queue filtering** (3 tools: process_metadata_entry, process_metadata_queue, update_metadata_status) |
| `data_expert` | **ZERO ONLINE ACCESS**: execute downloads from pre-validated queue entries (research_agent creates), load local files via adapter system, manage modalities with ModalityManagementService (5 CRUD tools), retry failed downloads with strategy overrides, queue monitoring & troubleshooting |
| `singlecell_expert` | scRNA‑seq: QC, clustering, pseudobulk, trajectories, markers |
| `bulk_rnaseq_expert` | bulk RNA‑seq import + DE (pyDESeq2, formula designs) |
| `ms_proteomics_expert` | DDA/DIA workflows, missing values, normalization |
| `affinity_proteomics_expert` | Olink/antibody arrays, NPX, CV, panel harmonization |

### 3.4 Deployment & Infrastructure

| File | Visibility | Purpose |
|------|------------|---------|
| `Dockerfile` | **PUBLIC** | CLI container (synced to lobster-local) |
| `Dockerfile.server` | **PRIVATE** | FastAPI server (uses `ARG CLI_BASE_IMAGE` for local builds) |
| `docker-compose.yml` | **PRIVATE** | Multi-service orchestration (server + Redis) |
| `Makefile` | PUBLIC | Build/test automation |
| `.github/workflows/docker.yml` | PUBLIC | CI builds **CLI only** (no server) |
| `.github/workflows/sync-to-public.yml` | PRIVATE | Auto-syncs code to lobster-local/main on push |
| `.github/workflows/sync-wikis.yml` | PRIVATE | Auto-syncs wiki to both wikis |
| `scripts/sync_to_public.py` | PRIVATE | Code sync script (supports manual dev syncs) |
| `scripts/sync_wikis.py` | PRIVATE | Wiki sync script |
| `scripts/public_allowlist.txt` | PRIVATE | Code sync allowlist (gitignore-style patterns) |
| `scripts/wiki_public_allowlist.txt` | PRIVATE | Wiki sync allowlist (filename matching) |
| `pyproject.toml` | PUBLIC | Dependencies (do NOT edit – see 4.1) |

**Build strategy**:
- `Dockerfile` → public CLI image → published to Docker Hub as `omicsos/lobster:latest`
- `Dockerfile.server` → private server image → builds from local CLI (not Docker Hub)
- CI/CD tests CLI only; server builds are local-only via `make docker-build`

**Sync strategy**:
- Automated: pushes to `main` → sync to lobster-local/main + both wikis (filtered for public)
- Manual: `python scripts/sync_to_public.py --repo <url> --branch <branch>` (e.g., dev)
- Exclusions: `scripts/public_allowlist.txt` ensures server code, premium features stay private

---

## 4. RULES – Development Guidelines

### 4.1 Hard Rules (non‑negotiable)

1. **Do NOT edit `pyproject.toml`** – all dependency changes go through humans.  
2. **Prefer editing existing files** over adding new ones.  
3. **Use `config/agent_registry.py`** for agents – do not hand‑edit `graph.py` with new agents.  
4. **Keep services stateless**: pure functions on `AnnData` / data + 3‑tuple return.  
5. **Use professional modality naming** (see 4.6).  
6. **Ensure both local and cloud clients work** (CLI must behave identically).  
7. **Preserve CLI backward compatibility** where reasonable.  
8. **Maintain scientific correctness** – no “quick hacks” that break analysis rigor.

### 4.2 Service Pattern (3‑tuple)

All analysis services must follow:

```python
def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
    ...
    return processed_adata, stats_dict, ir
```

- `processed_adata`: modified AnnData (results in `.obs`, `.var`, `.obsm`, `.uns`, etc.)  
- `stats_dict`: concise, human‑readable summary  
- `ir`: `AnalysisStep` used by provenance + notebook export (see 4.4)

### 4.3 Tool Pattern (agent tools)

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    if modality_name not in data_manager.list_modalities():
        raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

    adata = data_manager.get_modality(modality_name)
    result, stats, ir = service.analyze(adata, **params)

    new_name = f"{modality_name}_analyzed"
    data_manager.modalities[new_name] = result

    data_manager.log_tool_usage(
        "analyze_modality", params, stats, ir=ir  # IR is mandatory
    )
    return f"Analysis complete: {stats}"
```

Key points:

- validate modality existence  
- delegate to **stateless service**  
- store result with descriptive suffix  
- **always pass `ir`** into `log_tool_usage(...)`

### 4.4 Provenance & `AnalysisStep` IR (W3C‑PROV)

**Every logged analysis step must emit IR. No IR → not reproducible.**

Minimal required ideas (see existing services for full detail):

- Service returns `(adata, stats, ir: AnalysisStep)`  
- `AnalysisStep` captures:
  - `operation` (e.g. `"scanpy.pp.calculate_qc_metrics"`)  
  - `tool_name` (service method)  
  - `description` (human explanation)  
  - `library` (e.g. `"scanpy"`, `"pyDESeq2"`)  
  - `code_template` (Jinja2 snippet with `{{ params }}`) using **only standard libraries**  
  - `imports` (pure Python imports)  
  - `parameters` + `parameter_schema` (types, defaults, validation rules)  
  - `input_entities` / `output_entities`  
  - optional `execution_context`, validation flags

Pattern:

```python
class QualityService:
    def assess_quality(self, adata: AnnData, min_genes: int = 200,
                       max_genes: int = 8000) -> Tuple[AnnData, Dict, AnalysisStep]:
        processed = adata.copy()
        # ... processing ...
        stats = {...}

        ir = self._create_ir(min_genes=min_genes, max_genes=max_genes)
        return processed, stats, ir
```

Checklist for new services:

- [ ] Returns `Tuple[AnnData, Dict[str, Any], AnalysisStep]`
- [ ] Helper `_create_ir(**params)` builds IR + parameter schema
- [ ] Jinja2 `code_template` uses `{{ param }}` only, standard libs only
- [ ] Agent tools call `log_tool_usage(..., ir=ir)`
- [ ] Works with `/pipeline export` + notebook execution

**Recent Enhancement (v2.4+)**: All ModalityManagementService methods and DownloadOrchestrator operations emit AnalysisStep IR, ensuring complete provenance tracking for data loading, downloads, and modality operations.

### 4.5 Patterns & Abstractions

- **Queue pattern**: use `DownloadQueue` for multi‑step downloads (see 2.4).
- **Concurrency pattern** (`core/queue_storage.py`): multi-process safe file access for shared JSON/JSONL files.
  - `InterProcessFileLock` – file-based lock using `fcntl.flock` (POSIX) / `msvcrt.locking` (Windows)
  - `queue_file_lock(thread_lock, lock_path)` – combines threading.Lock + file lock
  - `atomic_write_json(path, data)` – temp file + fsync + `os.replace` for crash-safe writes
  - `atomic_write_jsonl(path, entries, serializer)` – same for JSONL files
  - **Protected files**: download_queue.jsonl, publication_queue.jsonl, .session.json, cache_metadata.json
  - **Rule**: Future features persisting shared state should use these utilities
- **Redis rate limiting pattern** (`tools/rate_limiter.py`): thread-safe connection pool for NCBI API rate limiting
  - Uses `redis.ConnectionPool` with `health_check_interval=30` for auto-recovery from stale connections
  - Double-checked locking for thread-safe lazy initialization
  - Works across all usage scenarios: interactive (`lobster chat`), non-interactive (`lobster query`), programmatic
  - Each process gets its own pool; cross-process coordination is handled by Redis keys with TTL
  - `reset_redis_pool()` for test isolation
  - **Rule**: When creating providers that use rate limiting, reuse provider instances via lazy properties (see `PublicationProcessingService.pubmed_provider`)
- **Workspace resolution pattern** (`core/workspace.py`): centralized workspace path resolution
  - `resolve_workspace(explicit_path, create)` – single entry point for workspace path resolution
  - Resolution order: explicit path > `LOBSTER_WORKSPACE` env var > `Path.cwd() / ".lobster_workspace"`
  - Used by: CLI, AgentClient, DataManagerV2
  - Always returns absolute path
  - **Rule**: All code that needs workspace paths MUST use `resolve_workspace()` instead of hardcoding paths
- **Error hierarchy** – prefer specific exceptions:
  - `ModalityNotFoundError` – missing dataset in `DataManagerV2`  
  - `ServiceError` – analysis failures  
  - `ValidationError` – schema/data issues  
  - `ProviderError` – external API failures  
- **Registry pattern** (`config/agent_registry.py`):

```python
@dataclass
class AgentConfig:
    name: str
    display_name: str
    description: str
    factory_function: str
    handoff_tool_name: Optional[str]
    handoff_tool_description: Optional[str]

AGENT_REGISTRY = {
    "new_agent": AgentConfig(
        name="new_agent",
        display_name="New Agent",
        description="Agent purpose",
        factory_function="lobster.agents.new_agent.new_agent",
        handoff_tool_name="handoff_to_new_agent",
        handoff_tool_description="When to handoff"
    )
}
```

Adding a new agent should be **registry‑only** wherever possible.

- **Adapter pattern**:
  - `IModalityAdapter` – format‑specific loading (10x, H5AD, etc.)
  - `IDataBackend` – H5AD/MuData/S3 backends
  - `BaseClient` – local vs cloud client abstraction
- **Delegation tool pattern** (`agents/graph.py`):
  - `_create_delegation_tool(agent_name, agent, description)` wraps sub-agents as `@tool` functions
  - Agent factories accept `delegation_tools` parameter (list of pre-wrapped tools)
  - Parent agents in registry specify `child_agents` → graph.py auto-creates delegation tools
  - **Critical**: inner function must have a proper docstring (f-strings do NOT work as docstrings)

### 4.6 Download Architecture (Queue-Based Pattern)

**Problem**: data_expert had online access, could fetch metadata/URLs directly, breaking single-responsibility principle.

**Solution**: Established ZERO online access boundary with queue-based coordination:

1. **research_agent** (online): Validates metadata, extracts URLs, creates DownloadQueueEntry (status: PENDING)
2. **supervisor**: Extracts entry_id from research_agent response, delegates to data_expert
3. **data_expert** (offline): Executes download via execute_download_from_queue(entry_id), updates status

**Key Components**:

- **IDownloadService** (`core/interfaces/download_service.py`): Abstract base class for database-specific services
  - `supports_database(database: str) -> bool`
  - `download_dataset(queue_entry, strategy_override) -> (adata, stats, ir)`
  - `validate_strategy_params(params) -> (bool, Optional[str])`
  - `get_supported_strategies() -> List[str]`

- **DownloadOrchestrator** (`tools/download_orchestrator.py`): Central router with 9-step execution logic
  - Service registration: `register_service(service: IDownloadService)`
  - Execution: `execute_download(entry_id, strategy_override) -> (modality_name, stats)`
  - Automatic service detection by database type
  - Comprehensive error handling with queue status updates

- **GEODownloadService** (`services/data_access/geo_download_service.py`): Adapter wrapping GEOService
  - Composition pattern (uses GEOService internally)
  - Adapts string return to (AnnData, stats, ir) tuple
  - Retrieves stored modality from DataManagerV2

**Usage Pattern**:
```python
# research_agent creates queue entry
entry_id = research_agent.validate_and_queue("GSE12345")

# data_expert executes
orchestrator = DownloadOrchestrator(data_manager)
orchestrator.register_service(GEODownloadService(data_manager))
modality_name, stats = orchestrator.execute_download(entry_id)
```

**Benefits**:
- Clear separation of concerns (online vs offline operations)
- Extensible to new databases (SRA, PRIDE, etc.) via IDownloadService
- Provenance tracking at every step
- Retry mechanism with strategy overrides
- Comprehensive error handling and status tracking

### 4.7 Publication Queue → Metadata Filtering Workflow (v2.5+)

**Problem**: Customers need to process large publication collections (.ris files), extract dataset identifiers, fetch sample metadata, and filter by criteria (e.g., "16S human fecal CRC").

**Solution**: Multi-agent workflow with automatic handoff readiness detection and batch metadata processing.

**Workflow**:
1. **research_agent**: Processes publication queue, extracts identifiers via NCBI E-Link, fetches SRA metadata
2. **Auto-status**: Entries with identifiers + datasets + metadata → `HANDOFF_READY` status
3. **metadata_assistant**: Batch processes HANDOFF_READY entries, applies filter criteria, aggregates samples
4. **Export**: Unified CSV with publication context + sample metadata + download URLs

**Key Files**:
- `services/orchestration/publication_processing_service.py` - Queue processing with auto-status
- `agents/metadata_assistant.py` - 3 new tools (process_metadata_entry, process_metadata_queue, update_metadata_status)
- `tools/workspace_tool.py` - Shared write_to_workspace factory (CSV export)
- `core/schemas/publication_queue.py` - PublicationQueueEntry with handoff fields

**Auto-Status Logic**:
```python
is_ready_for_handoff = (
    bool(extracted_identifiers) and  # Has identifiers
    bool(dataset_ids) and            # Has dataset IDs
    bool(workspace_metadata_keys)     # Has workspace metadata files
)
# → Status transitions to HANDOFF_READY
```

### 4.8 ModalityManagementService Pattern

**Problem**: Modality CRUD operations scattered across data_expert tools, inconsistent provenance tracking.

**Solution**: Centralized service with 5 standardized methods, all returning (result, stats, ir) tuples.

**Location**: `lobster/services/data_management/modality_management_service.py`

**Methods**:
```python
class ModalityManagementService:
    def list_modalities(filter_pattern: Optional[str]) -> (List[Dict], Dict, AnalysisStep):
        """List modalities with optional glob filtering."""

    def get_modality_info(modality_name: str) -> (Dict, Dict, AnalysisStep):
        """Get detailed info (shape, layers, obsm/varm/uns keys, quality metrics)."""

    def remove_modality(modality_name: str) -> (bool, Dict, AnalysisStep):
        """Remove modality from DataManagerV2."""

    def validate_compatibility(modality_names: List[str]) -> (Dict, Dict, AnalysisStep):
        """Validate obs/var overlap, batch effects, recommend integration strategy."""

    def load_modality(
        modality_name: str,
        file_path: str,
        adapter: str,
        dataset_type: str = "custom",
        validate: bool = True
    ) -> (AnnData, Dict, AnalysisStep):
        """Load data file via adapter system with schema validation."""
```

**Integration**: data_expert agent creates service instance and exposes tools wrapping each method.

**Benefits**:
- Consistent 3-tuple pattern (result, stats, ir)
- W3C-PROV compliance via AnalysisStep IR
- Centralized error handling
- Reusable across multiple agents

### 4.9 Naming & Data Quality

**Naming convention (example)**:

```text
geo_gse12345
├─ geo_gse12345_quality_assessed
├─ geo_gse12345_filtered_normalized
├─ geo_gse12345_doublets_detected
├─ geo_gse12345_clustered
├─ geo_gse12345_markers
└─ geo_gse12345_annotated
```

Data standards:

- W3C‑PROV compliant logging
- Pydantic schema validation for all modalities
- Good QC metrics at each step
- Proper missing‑value handling (esp. proteomics)
- Support batch effect detection/correction where relevant

### 4.10 Security Considerations

#### CustomCodeExecutionService Security Model

**Location**: `lobster/services/execution/custom_code_execution_service.py`

**Purpose**: Enables data_expert to execute arbitrary Python code for edge cases not covered by specialized tools.

**Security Model**: **Subprocess isolation with Phase 1 hardening (NOT full sandboxing)**

```python
# What the current implementation provides (Phase 1 complete):
✅ Process crash isolation (user code crashes don't kill Lobster)
✅ Timeout enforcement (300s default)
✅ Output capture (stdout/stderr isolated)
✅ Environment variable filtering (API keys, AWS credentials NOT passed to subprocess)
✅ AST validation blocks eval/exec/compile/__import__ calls
✅ Module shadowing prevented (sys.path.append, not insert)
✅ Expanded import blocking (subprocess, multiprocessing, pickle, ctypes, etc.)

# What still requires Phase 2 (Docker isolation):
❌ Network isolation (can make HTTP requests)
❌ File access restriction (has full user permissions)
❌ Resource limits (only timeout, no memory/CPU/disk limits)
```

#### Known Vulnerabilities (Comprehensive Testing Results)

**Testing Date**: 2025-11-30
**Methodology**: 8-agent parallel adversarial testing (201+ attack vectors)
**Results**: ~90% of attacks succeeded, confirming multiple critical vulnerabilities

**Detailed documentation**: `tests/manual/custom_code_execution/`

| Category | Severity | Status (Post-Phase 1) | Details |
|----------|----------|----------------------|---------|
| **Data Exfiltration** | 🟠 HIGH | PARTIAL | Env vars filtered; filesystem unrestricted; network open (Phase 2 needed) |
| **Resource Exhaustion** | 🟠 HIGH | PARTIAL | Only 300s timeout, no memory/CPU/disk limits (Phase 2 needed) |
| **Privilege Escalation** | 🟢 LOW | BLOCKED | Import blocks prevent multiprocessing, signal blocked via FORBIDDEN_FROM_IMPORTS |
| **Supply Chain** | 🟢 LOW | BLOCKED | sys.path.append() prevents module shadowing |
| **AST Bypass** | 🟢 LOW | BLOCKED | exec/eval/compile/__import__ blocked at AST level |
| **Timing Attacks** | 🟡 MEDIUM | DOCUMENTED | No timing normalization (acceptable for local CLI) |
| **Workspace Pollution** | 🟠 HIGH | PARTIAL | File access unrestricted (Phase 2 Docker needed for isolation) |
| **Integration Attacks** | 🟡 MEDIUM | PARTIAL | Import blocking reduces attack surface; file persistence still possible |

#### Production Deployment Guidelines

**Local CLI (current target):**
- ✅ **PRODUCTION READY** (Phase 1 complete as of 2025-11-30)
- Acceptable risk profile: trusted users, local machine, documented limitations
- Remaining gap: Network access still allowed (acceptable for local CLI)

**Cloud SaaS (future):**
- ❌ NOT ACCEPTABLE without Docker isolation
- Requires: Phase 2 mitigations (Docker with --network=none, memory limits)

**Enterprise (mixed trust):**
- ⚠️ CONDITIONAL GO - Phase 1 complete, Phase 2 recommended
- Recommended: Add Phase 3 enhancements for defense-in-depth

#### Mitigation Roadmap

**Phase 1: Critical Fixes - ✅ COMPLETE (2025-11-30)**

All Phase 1 fixes have been implemented:
- ✅ Environment variable filtering (`env=safe_env` in subprocess.run)
- ✅ sys.path.append() instead of insert() (prevents module shadowing)
- ✅ Expanded FORBIDDEN_MODULES (subprocess, multiprocessing, pickle, ctypes, etc.)
- ✅ Expanded FORBIDDEN_FROM_IMPORTS (os.kill, os.fork, signal.*, etc.)
- ✅ AST validation blocks exec/eval/compile/__import__ with CodeValidationError
- ✅ Security documentation updated (tool docstring, CLAUDE.md)

Note: Runtime builtin removal was attempted but reverted because Python's import
system requires exec() internally. AST-level blocking is sufficient for user code.

**Phase 2: Production Hardening (6 days) - CLOUD REQUIRED**

```bash
# Docker isolation with resource limits
docker run --rm \
    --network=none \
    --memory=2g \
    --cpus=2 \
    --pids-limit=100 \
    --read-only \
    --volume workspace:/workspace:rw \
    python:3.11 python script.py
```

**Phase 3: Long-Term (10 weeks) - ENTERPRISE RECOMMENDED**
- gVisor sandbox (user-space kernel)
- Firecracker MicroVMs (full VM isolation)
- RestrictedPython (runtime policy enforcement)
- Cryptographic provenance (sign IR templates)
- Third-party security audit

#### Security Testing

**Manual test suite**: `tests/manual/custom_code_execution/`
- 30+ test files, 201+ attack vectors
- 8 comprehensive security reports
- Run with: `pytest tests/manual/custom_code_execution/ -v`

**Reports:**
- `COMPREHENSIVE_SECURITY_ASSESSMENT.md` - Consolidated findings
- `README.md` - Testing methodology
- Individual category reports (data exfiltration, resource exhaustion, etc.)

#### Developer Guidelines

When working on CustomCodeExecutionService or related code:

1. **Never weaken existing protections** without security review
2. **Test all changes** against adversarial test suite
3. **Document new limitations** in tool docstring and CLAUDE.md
4. **Assume malicious input** when designing validation logic
5. **Follow principle of least privilege** for all operations

**Critical files to protect:**
- `.session.json` - Contains API keys, credentials
- `download_queue.jsonl`, `publication_queue.jsonl` - Orchestration state
- `.lobster/provenance/` - W3C-PROV logs (reproducibility)
- `data/*.h5ad` - Modality data files

**Red flags** (require security review):
- Adding new imports to context setup code
- Modifying sys.path manipulation
- Changing subprocess execution parameters
- Adding filesystem or network operations
- Modifying AST validation logic

#### User-Facing Documentation

**Tool docstring must include**:
```python
⚠️ SECURITY NOTICE:
- Code runs with YOUR user permissions
- Network access is ALLOWED (can make HTTP requests)
- Environment variables are FILTERED (API keys protected)
- Resource limits: 300s timeout, no memory/CPU limits
- Use ONLY for trusted code on trusted data
- For untrusted code, use cloud deployment with Docker isolation

This feature is designed for scientific flexibility, not security.
```

#### References

- Comprehensive testing results: `tests/manual/custom_code_execution/COMPREHENSIVE_SECURITY_ASSESSMENT.md`
- Testing methodology: `tests/manual/custom_code_execution/README.md`
- Original plan: `.claude/plans/idempotent-stargazing-stroustrup.md`
- Baseline tests: `REAL_WORLD_TEST_RESULTS.md`

---

## 5. Tooling, Commands & Environment

### 5.1 Technology Stack

| Area | Tech |
|------|------|
| Agent framework | LangGraph |
| Models | AWS Bedrock (Claude) |
| Language | Python 3.11+ (typing, async/await) |
| Data structures | AnnData, MuData |
| Bioinformatics | Scanpy, PyDESeq2 |
| CLI | Typer, Rich, prompt_toolkit |
| Visualization | Plotly |
| Storage | H5AD, HDF5, JSONL, S3 backends |

### 5.2 Environment Setup

```bash
make dev-install     # full dev setup
make install         # minimal install
make clean-install   # fresh env
source .venv/bin/activate
```

### 5.3 Testing

```bash
make test            # all tests
make test-fast       # parallel subset
make format          # black + isort
make lint            # flake8/pylint/bandit
make type-check      # mypy

pytest tests/unit/
pytest tests/integration/
pytest tests/integration/ -m real_api
pytest tests/integration/ -m "real_api and slow"
```

Markers / keys:

- `@pytest.mark.real_api` – requires network + keys  
- `@pytest.mark.slow` – >30s tests  
- `@pytest.mark.integration` – multi‑component

Env vars:

| Variable | Required | Purpose |
|----------|----------|---------|
| `AWS_BEDROCK_ACCESS_KEY` | Yes | AWS Bedrock access key |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | Yes | AWS Bedrock secret key |
| `NCBI_API_KEY` | No | PubMed (higher rate limit) |
| `LOBSTER_CLOUD_KEY` | No | enables cloud client mode |

### 5.4 Running the App

```bash
lobster chat                    # interactive, multi‑turn
lobster query "your request"    # single‑turn automation
lobster --help                  # CLI help
```

Chat vs query:

- `chat`: can ask follow‑ups, clarify, exploratory work
- `query`: single‑shot, script/CI‑friendly, no follow‑ups

**Workspace configuration** (applies to both `chat` and `query`):

```bash
# Use custom workspace directory
lobster chat --workspace /path/to/workspace
lobster query "analyze data" -w ~/my_workspace

# Or set via environment variable
export LOBSTER_WORKSPACE=/shared/workspace
lobster chat  # Uses /shared/workspace

# Resolution order: --workspace flag > LOBSTER_WORKSPACE env > cwd/.lobster_workspace
```

Useful CLI commands:

| Category | Commands |
|----------|----------|
| Help | `/help`, `/status`, `/modes` |
| Data | `/data`, `/files`, `/read <file>` |
| Workspace | `/workspace`, `/workspace list`, `/workspace load <name>` |
| Plots | `/plots` |
| Pipelines | `/pipeline export`, `/pipeline list`, `/pipeline run <nb> <modality>` |

### 5.5 Package Publishing (PyPI)

**Package Name**: `lobster-ai` (PyPI) / `lobster` (import name)
**Version Source**: `lobster/version.py` (single source of truth)
**Publishing**: Automated via GitHub Actions on git tags (`v*.*.*`)

**Critical Security Rule**: Only `lobster-local` (public repo) is published to PyPI. The private `lobster/` repo contains premium features that must NOT be published.

**Release Process**:
```bash
# 1. Update version in lobster/version.py
# 2. Commit and push to main
# 3. Create and push tag
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
# 4. GitHub Actions workflow handles the rest
```

**Documentation**:
- **Setup Guide**: `docs/PYPI_SETUP_GUIDE.md` (first-time setup, authentication methods)
- **Release Summary**: `docs/PYPI_RELEASE_SUMMARY.md` (quick reference, workflow details)
- **Workflow**: `.github/workflows/publish-pypi.yml` (7-stage automated pipeline)

**Authentication**: Supports Trusted Publishing (OIDC) and API tokens. See setup guide for details.

---

## 6. Troubleshooting (quick scan)

- **Install issues**
  - Python 3.11+ required
  - try `make clean-install`
- **CLI quirks**
  - check `PROMPT_TOOLKIT_AVAILABLE`
  - verify `LobsterClientAdapter` picks correct client type
- **Profiling**
  - Global timings via `--profile-timings` or `LOBSTER_PROFILE_TIMINGS=1` (CLI). Toggles `DataManagerV2.enable_timing`/`get_latest_timings` and `PublicationProcessingService.enable_timing`; CLI prints tables via `_maybe_print_timings` (see `cli.py`, `data_manager_v2.py`, `publication_processing_service.py`).
- **Cloud mode**
  - ensure `LOBSTER_CLOUD_KEY` set
  - check network + timeouts (cloud vs local caches)

---

## 8. Who You Are – ultrathink

You are **ultrathink** – an agent that blends scientist, engineer, and designer mindsets.

Principles:

1. **Think different** – challenge defaults; search for the cleanest architecture, not the first working hack.  
2. **Obsess over patterns** – understand codebase philosophy, reuse existing abstractions, extend registries instead of ad‑hoc wiring.  
3. **Plan first** – sketch flows (often as Mermaid) before editing code. Explain the plan clearly before implementation.  
4. **Craft, don’t just code** – choose precise names, clean APIs, and robust tests. Design for future contributors.  
5. **Iterate** – propose v1, refine with feedback, compare options.  
6. **Simplify ruthlessly** – remove unnecessary complexity when it doesn’t reduce power. Favor smaller, composable pieces over cleverness.

Git history and docs (especially `CLAUDE.md`) are your **source of truth**.  
Every change should make Lobster AI more **reproducible, elegant, and scientifically trustworthy**.
