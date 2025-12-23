# CLAUDE.md

System prompt for Lobster AI — a professional multi‑agent bioinformatics analysis platform.
Goal: help senior engineers understand **architecture, file layout, and non‑negotiable rules** with minimal tokens.

---

## 📍 Documentation Navigation

**This document**: Lobster AI engine development guide (agents, services, core architecture)
**Parent**: `../CLAUDE.md` (Monorepo overview)
**Related**:
  - `../lobster-cloud/CLAUDE.md` (Cloud infrastructure & AWS backend)
  - `../landing/CLAUDE.md` (Marketing website & React frontend)
  - `../lobster-local/` (Public open-core distribution)

**Detailed documentation**:
  - `wiki/` (58 pages: user guides, tutorials, API reference, architecture)
  - `docs/` (17 files: technical specs, PyPI publishing, custom packages)
  - `../../docs/` (Root: business docs, premium licensing, customer proposals)

**Key external links**:
  - Premium Licensing: `../../docs/PREMIUM_LICENSING.md`
  - Testing Guide: `../../docs/PREMIUM_TESTING_CHECKLIST.md`
  - Custom Packages: `../../docs/CUSTOM_PACKAGES_ACTIVATION_FLOW.md`

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

### 1.5 LLM Provider System

Lobster supports **3 LLM providers** via modular, testable architecture:

| Provider | Type | Setup | Use Case |
|----------|------|-------|----------|
| **Ollama** | Local | `ollama pull gpt-oss:20b` | Privacy, zero cost, offline |
| **Anthropic Direct** | Cloud | API key | Fastest, best quality |
| **AWS Bedrock** | Cloud | AWS credentials | Enterprise, compliance |

**Architecture** (separation of concerns):
```
CLI (cli.py)                 → Logic (provider_setup.py)      → Factory (llm_factory.py)
├─ User interaction           ├─ Ollama detection              ├─ Model instantiation
├─ Rich console output        ├─ Provider validation           ├─ Provider enum
└─ lobster init wizard        ├─ Config creation               └─ Auto-detection
                              └─ Pure functions (testable)
```

**Key files**:
- `lobster/config/provider_setup.py`: **NEW** - Pure provider logic (detection, validation, config creation)
- `lobster/config/llm_factory.py`: Model instantiation + provider enum
- `lobster/config/settings.py`: Environment configuration
- `lobster/cli.py`: User-facing commands (`lobster init`, `lobster config-show`)

**Auto-detection priority**: Explicit `LOBSTER_LLM_PROVIDER` → Ollama (if running) → Anthropic → Bedrock

**User configuration**: Via `lobster init` wizard (recommended) or manual `.env` editing.

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
├─ claude-skill/            # Claude Code integration
│  ├─ SKILL.md             # Skill definition (installed to ~/.claude/skills/)
│  ├─ install.sh           # Automated installation script
│  └─ README.md            # Integration overview
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
│  ├─ utils/
│  │  └─ h5ad_utils.py      # H5AD validation, compression (v3.4.2+)
│  ├─ interfaces/
│  │  ├─ base_client.py
│  │  ├─ data_backend.py
│  │  └─ modality_adapter.py
│  ├─ backends/
│  │  ├─ h5ad_backend.py
│  │  └─ mudata_backend.py
│  ├─ identifiers/
│  │  ├─ __init__.py           # Module exports
│  │  └─ accession_resolver.py # Centralized identifier validation (29 patterns)
│  └─ schemas/
│     ├─ transcriptomics_schema.py
│     ├─ proteomics_schema.py
│     ├─ metabolomics_schema.py
│     ├─ metagenomics_schema.py
│     └─ database_mappings.py  # DATABASE_ACCESSION_REGISTRY (single source of truth)
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
| CLI | `cli.py` | User-facing commands, Rich output (presentation layer) |
| LLM Config | `config/provider_setup.py` | **NEW**: Provider detection, validation, config creation (pure logic, testable) |
| LLM Factory | `config/llm_factory.py` | Model instantiation, provider enum |
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
| H5AD Utils | `core/utils/h5ad_utils.py` | H5AD file validation, compression, optimization (v3.4.2+) |
| Services | `services/*/*.py` | stateless analysis (organized by function) |
| Services | `services/data_management/modality_management_service.py` | Modality CRUD with provenance (5 methods) |
| Download | `tools/download_orchestrator.py` | Central router for database-specific downloads (9-step execution) |
| Download | `services/data_access/geo_download_service.py` | GEO database download service (IDownloadService impl) |
| GEO | `services/data_access/geo/` | Modular GEO package (downloader, parser, strategy, constants) |
| Interfaces | `core/interfaces/download_service.py` | IDownloadService abstract base class |
| Identifiers | `core/identifiers/accession_resolver.py` | Centralized accession validation (29 patterns, thread-safe singleton) |
| Providers | `tools/providers/*.py` | PubMed/GEO/Web access (delegate to AccessionResolver for validation) |
| GEO Provider | `tools/providers/geo_provider.py` | GEO search/metadata with GDS/GSE disambiguation (bug fixes v3.4.2+) |
| PMC Provider | `tools/providers/pmc_provider.py` | PubMed Central full-text retrieval (parser fixes v3.4.2+) |
| Utilities | `tools/*.py` | orchestrators, helpers |
| Workspace Tools | `tools/workspace_tool.py` | **Unified workspace tool (v2.6+)**: write_to_workspace, get_content_from_workspace (adapter pattern for 5 workspace types) |
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
| `scripts/generate_allowlist.py` | PRIVATE | Generates allowlist from subscription_tiers.py |
| `scripts/public_allowlist.txt` | PRIVATE | Code sync allowlist (AUTO-GENERATED from subscription_tiers.py) |
| `scripts/wiki_public_allowlist.txt` | PRIVATE | Wiki sync allowlist (manual, filename matching) |
| `pyproject.toml` | PUBLIC | Dependencies (do NOT edit – see 4.1) |

**Build strategy**:
- `Dockerfile` → public CLI image → published to Docker Hub as `omicsos/lobster:latest`
- `Dockerfile.server` → private server image → builds from local CLI (not Docker Hub)
- CI/CD tests CLI only; server builds are local-only via `make docker-build`
- **Publishing optimization**: Uses sparse-checkout for `constraints-cpu.txt` (single source of truth, no artifact extraction)

**Sync strategy** (Single Source of Truth Architecture):

```
subscription_tiers.py (SOURCE OF TRUTH)
         ↓
  generate_allowlist.py --validate (CI check)
         ↓
public_allowlist.txt (DERIVED - DO NOT EDIT MANUALLY)
         ↓
   sync_to_public.py
         ↓
lobster-local (PUBLIC PACKAGE)
```

- Automated: pushes to `main` → sync to lobster-local/main + both wikis (filtered for public)
- Manual: `python scripts/sync_to_public.py --repo <url> --branch <branch>` (e.g., dev)
- **Single Source of Truth**: `subscription_tiers.py` defines FREE/PREMIUM features
- **Generated allowlist**: `python scripts/generate_allowlist.py --write` regenerates `public_allowlist.txt`
- **CI validation**: Both `sync-to-public.yml` and `publish-pypi.yml` validate allowlist before syncing
- **NEVER edit `public_allowlist.txt` directly** – always modify `subscription_tiers.py` and regenerate

---

## 4. RULES – Development Guidelines

### 4.1 Hard Rules (non‑negotiable)

1. **Do NOT edit `pyproject.toml`** – all dependency changes go through humans. **Exception**: ML extras (`torch`, `scvi-tools`) are now optional (PREMIUM tier). Install with `pip install lobster-ai[ml]`.
2. **Prefer editing existing files** over adding new ones.
3. **Use `config/agent_registry.py`** for agents – do not hand‑edit `graph.py` with new agents.
4. **Keep services stateless**: pure functions on `AnnData` / data + 3‑tuple return.
5. **Use professional modality naming** (see 4.6).
6. **Ensure both local and cloud clients work** (CLI must behave identically).
7. **Preserve CLI backward compatibility** where reasonable.
8. **Maintain scientific correctness** – no "quick hacks" that break analysis rigor.

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

Adding a new agent should be **registry‑only** wherever possible. For premium-only agents, use `PREMIUM_REGISTRY` in `lobster-premium` package instead.

- **Adapter pattern**:
  - `IModalityAdapter` – format‑specific loading (10x, H5AD, etc.)
  - `IDataBackend` – H5AD/MuData/S3 backends
  - `BaseClient` – local vs cloud client abstraction
- **Delegation tool pattern** (`agents/graph.py`):
  - `_create_delegation_tool(agent_name, agent, description)` wraps sub-agents as `@tool` functions
  - Agent factories accept `delegation_tools` parameter (list of pre-wrapped tools)
  - Parent agents in registry specify `child_agents` → graph.py auto-creates delegation tools
  - **Critical**: inner function must have a proper docstring (f-strings do NOT work as docstrings)
- **AccessionResolver pattern** (`core/identifiers/accession_resolver.py`): centralized identifier validation
  - Thread-safe singleton via `get_accession_resolver()`
  - 29 pre-compiled patterns in `DATABASE_ACCESSION_REGISTRY` (`core/schemas/database_mappings.py`)
  - Case-insensitive matching for better UX (`gse12345` = `GSE12345`)
  - Key methods: `detect_database()`, `extract_accessions_by_type()`, `validate()`, `get_url()`
  - Helper methods: `is_geo_identifier()`, `is_sra_identifier()`, `is_proteomics_identifier()`
  - **Rule**: All providers MUST use AccessionResolver instead of hardcoded regex patterns
  - Supported databases: GEO (GSE/GSM/GPL/GDS), SRA (SRP/SRX/SRR/SRS), ENA, DDBJ, BioProject, BioSample, PRIDE, MassIVE, MetaboLights, ArrayExpress, DOI
- **Unified Workspace Tool pattern** (`tools/workspace_tool.py`): adapter-based workspace access (v2.6+)
  - **Architecture**: Dispatcher → Adapter → WorkspaceItem TypedDict → Unified Formatter
  - **5 Workspace Types**: literature, data, metadata, download_queue, publication_queue
  - **3 Adapters**: `_adapt_general_content()`, `_adapt_download_queue()`, `_adapt_publication_queue()`
  - **WorkspaceItem**: Unified structure (identifier, workspace, type, status, title, cached_at, details)
  - **Benefits**: Consistent API across all workspaces, defensive against missing fields, extensible (new workspace = 1 adapter)
  - **Tools**: `get_content_from_workspace(identifier, workspace, level, status_filter)`, `write_to_workspace(identifier, workspace, ...)`
  - **Rule**: All agents use same mental model for workspace operations (list, filter, retrieve)
  - **Doc**: See `wiki/38-workspace-content-service.md` for usage examples

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
- `agents/metadata_assistant.py` - 3 tools (process_metadata_entry, process_metadata_queue, update_metadata_status)
- `tools/workspace_tool.py` - Unified workspace tools (v2.6+: adapter pattern for all workspace types)
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

**CustomCodeExecutionService** (`services/execution/custom_code_execution_service.py`):
- **Purpose**: Execute arbitrary Python for edge cases
- **Model**: Subprocess isolation (Phase 1 hardening, NOT full sandboxing)
- **Status**: ✅ Production-ready for local CLI (trusted users)
- **Gaps**: Network access, file permissions (acceptable for local, NOT for cloud SaaS)
- **Testing**: `tests/manual/custom_code_execution/` (30+ files, 201+ attack vectors)

**Deployment**:
- Local CLI: ✅ Production-ready
- Cloud SaaS: ❌ Requires Docker isolation (Phase 2)
- Enterprise: ⚠️ Conditional (Phase 2 recommended)

### 4.11 Feature Tiering & Conditional Activation

Lobster supports FREE, PREMIUM, and ENTERPRISE tiers. Features/agents can be gated by subscription level or customer-specific packages. Design new features for conditional activation.

**Key Files**:
- `lobster/config/subscription_tiers.py` – Tier definitions (agents, handoff restrictions, features)
- `lobster/core/plugin_loader.py` – Plugin discovery for premium/custom packages
- `lobster/core/license_manager.py` – Entitlement validation & AWS license service integration

**Rules**:

1. **Agent factories**: Accept `subscription_tier: str = "free"`. Use `get_restricted_handoffs()` to conditionally exclude tools.
2. **New premium agents**: Add to `PREMIUM_REGISTRY` in `lobster-premium`, not `AGENT_REGISTRY`.
3. **Feature checks**: Use `is_agent_available(agent, tier)` from `subscription_tiers.py`, not hardcoded strings.
4. **Handoff restrictions**: Define in `subscription_tiers.py` under `restricted_handoffs`, not in agent code.
5. **Graceful degradation**: Return helpful messages when tier-restricted, don't crash.

**Tier Reference**:
| Tier | Agents | Notes |
|------|--------|-------|
| FREE | supervisor, research_agent, data_expert, transcriptomics_expert, visualization_expert, machine_learning_expert, protein_structure_visualization_expert | 7 agents, no metadata_assistant handoff |
| PREMIUM | + metadata_assistant, proteomics_expert | 9 agents, full handoffs, premium services |
| ENTERPRISE | + lobster-custom-* packages | Per-customer agents via plugin_loader |

#### License Manager & AWS Integration

**Location**: `lobster/core/license_manager.py`
**Purpose**: Entitlement validation via AWS license service (RSA signature verification)
**File**: `~/.lobster/license.json` (signed entitlement with tier, features, expiration)

**Key Functions**:
- `activate_license()` - Exchange cloud_key for signed entitlement
- `check_revocation_status()` - Periodic validation (24h interval)
- `_verify_signature()` - JWKS/RSA verification

**Resolution**: `LOBSTER_SUBSCRIPTION_TIER` env → `LOBSTER_CLOUD_KEY` → `license.json` → free tier default

**Commands**: `lobster activate <key>`, `lobster status` (shows tier)

**Full spec**: See root repository `../../docs/PREMIUM_LICENSING.md` (comprehensive implementation guide)

### 4.12 Custom Package Development

**Purpose**: Customer-specific premium packages (`lobster-custom-{customer}`) that extend `lobster-ai` with enterprise features.

**Template Location**: `/Users/tyo/GITHUB/omics-os/lobster-custom-template/` (copy-and-customize approach)

**Quick Workflow**:
1. Copy template → replace `TEMPLATE` with customer name
2. Add premium files from `lobster/` (check `scripts/public_allowlist.txt` for excluded files)
3. Update imports: `lobster.*` → `lobster_custom_{customer}.*`
4. Build & upload to S3: `s3://lobster-license-packages-649207544517/{customer}/`

**Key Files**:
- `lobster-custom-template/README.md` - Comprehensive guide (versioning, testing, deployment, troubleshooting)
- `lobster/config/subscription_tiers.py` - Premium feature definitions (single source of truth)
- `lobster/scripts/public_allowlist.txt` - Files excluded from public PyPI package (prefix `!`)
- `lobster-custom-databiomix/` - Reference implementation (v2.0.5)

**Critical Rules**:
1. **Import paths**: ALL `lobster.*` imports MUST become `lobster_custom_{customer}.*` (except public files)
2. **Dependencies**: Copy ALL transitive dependencies (check imports recursively)
3. **Versioning**: Track lobster-ai version (e.g., `0.3.4.1` for lobster-ai 0.3.4.x)
4. **Testing**: Test in clean environment with fresh `lobster-ai` install
5. **Plugin registration**: Entry point in `pyproject.toml` + `register_agents()` in `__init__.py`

**Common Pattern** (premium core module):
```python
# Custom package copies lobster/core/publication_queue.py
# Update internal imports:
from lobster.core.schemas.publication_queue import ...  # ❌ WRONG
from lobster_custom_{customer}.core.schemas.publication_queue import ...  # ✅ CORRECT
```

**See**: `lobster-custom-template/README.md` for complete workflow, pitfalls, and deployment checklist.

---

## 5. Tooling, Commands & Environment

### 5.1 Technology Stack

| Area | Tech |
|------|------|
| Agent framework | LangGraph |
| Models | AWS Bedrock (Claude), Anthropic Direct, **Ollama (local)** |
| Language | Python 3.11+ (typing, async/await) |
| Data structures | AnnData, MuData |
| Bioinformatics | Scanpy, PyDESeq2 |
| ML (optional) | **PyTorch, scVI-tools** (install with `pip install lobster-ai[ml]`) |
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
| `AWS_BEDROCK_ACCESS_KEY` | Conditional | AWS Bedrock access key (cloud) |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | Conditional | AWS Bedrock secret key (cloud) |
| `ANTHROPIC_API_KEY` | Conditional | Anthropic Direct API key (cloud) |
| `LOBSTER_LLM_PROVIDER` | No | Explicit provider: `ollama`, `bedrock`, `anthropic` |
| `OLLAMA_BASE_URL` | No | Ollama server (default: `http://localhost:11434`) |
| `OLLAMA_DEFAULT_MODEL` | No | Model name (default: `gpt-oss:20b`) |
| `NCBI_API_KEY` | No | PubMed (higher rate limit) |
| `LOBSTER_CLOUD_KEY` | No | enables cloud client mode |

**Note:** At least one LLM provider is required (Anthropic OR Bedrock OR Ollama).

### 5.4 Running the App

```bash
lobster chat                    # interactive, multi‑turn
lobster query "your request"    # single‑turn automation
lobster --help                  # CLI help
```

Chat vs query:

- `chat`: can ask follow‑ups, clarify, exploratory work
- `query`: single‑shot, script/CI‑friendly, no follow‑ups

**Session continuity** (multi-turn conversations):

```bash
# Start session with custom ID
lobster query --session-id "project_1" "Search PubMed for CRISPR papers"

# Continue same session
lobster query --session-id "project_1" "Download the first dataset"

# Use 'latest' to auto-continue most recent session
lobster query --session-id latest "Cluster that dataset"
```

- Session files stored in workspace as `session_*.json`
- Enables follow-up questions that reference previous context
- Critical for Claude Code integration (see 5.6)

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

**Package**: `lobster-ai` (PyPI) / `lobster` (import)
**Version**: `lobster/version.py` (single source of truth)
**Publishing**: Automated via GitHub Actions on git tags (`v*.*.*`)

**Critical Rule**: Only `lobster-local` (public) publishes to PyPI. Private repo contains premium features.

**Quick Release**:
```bash
# 1. Update lobster/version.py
__version__ = "0.2.0"

# 2. Tag and push
git add lobster/version.py
git commit -m "chore: bump version to 0.2.0"
git push origin main
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0  # Triggers 7-stage pipeline
```

**Pipeline**: Sync → Build → TestPyPI → Manual Approval → PyPI → Release → Summary

**Security**: Allowlist validation (`subscription_tiers.py` → `public_allowlist.txt`), private code detection, import tests.

**Full details**: `docs/PYPI_SETUP_GUIDE.md`, `docs/PYPI_RELEASE_SUMMARY.md`, `.github/workflows/publish-pypi.yml`

### 5.6 Claude Code Integration

Lobster integrates with [Claude Code](https://claude.ai/code) as a custom skill, enabling IDE-native bioinformatics workflows.

**Installation**:
```bash
# Automated install
curl -fsSL https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/install.sh | bash

# Manual install
mkdir -p ~/.claude/skills/
curl -o ~/.claude/skills/lobster-bioinformatics.md \
  https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/SKILL.md
```

**Key Files**:
- `claude-skill/SKILL.md` – Skill definition with trigger keywords, usage patterns, troubleshooting
- `claude-skill/install.sh` – Automated installation script
- `wiki/50-claude-code-integration.md` – Complete integration documentation

**Usage Pattern**:
```
User (in IDE) → Claude Code: "Download GSE109564 and cluster cells"
                      ↓
Claude Code detects bioinformatics task → Invokes Lobster skill
                      ↓
Executes: lobster query --session-id latest "Download GSE109564 and cluster cells"
                      ↓
Lobster orchestrates: research_agent → data_expert → singlecell_expert
                      ↓
Returns: Natural language summary + file paths (.lobster_workspace/)
                      ↓
Claude Code → User: Results + next step suggestions
```

**Benefits**:
- Seamless workflow: Stay in IDE while running complex analyses
- Intelligent routing: Claude Code auto-detects bioinformatics tasks
- Multi-turn conversations: Session continuity via `--session-id`
- Direct file access: Results immediately usable in code

**Trigger Keywords**: H5AD, CSV, GEO/SRA accessions, QC, clustering, differential expression, PubMed, single-cell, bulk RNA-seq, proteomics

**Version**: Compatible with Lobster v0.3.1.4+ and Claude Code v1.0+

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
