# Code Layout Reference

Detailed file structure and component reference. Load on-demand when navigating codebase.

---

## Top-Level Structure

**PEP 420 Namespace Package**: No `lobster/__init__.py` – enables namespace merging for `lobster-custom-*` packages.

```text
lobster/
├─ cli.py                   # CLI entrypoint (Typer/Rich)
├─ claude-skill/            # Claude Code integration
│  ├─ SKILL.md             # Skill definition
│  ├─ install.sh           # Automated installation
│  └─ README.md            # Integration overview
├─ agents/                  # Supervisor + specialist agents + graph
│  ├─ supervisor.py
│  ├─ research/             # Modular structure
│  │  ├─ __init__.py, state.py, config.py, prompts.py
│  │  └─ research_agent.py
│  ├─ metadata_assistant/   # Modular structure
│  │  ├─ __init__.py, config.py, prompts.py
│  │  └─ metadata_assistant.py
│  ├─ data_expert/          # Modular structure
│  │  ├─ __init__.py, state.py, config.py, prompts.py
│  │  ├─ data_expert.py
│  │  └─ assistant.py
│  ├─ transcriptomics/      # Modular structure
│  │  ├─ __init__.py, state.py, config.py, prompts.py
│  │  ├─ shared_tools.py
│  │  ├─ transcriptomics_expert.py
│  │  ├─ annotation_expert.py
│  │  └─ de_analysis_expert.py
│  ├─ proteomics/           # Modular structure
│  │  ├─ __init__.py, state.py, config.py, prompts.py
│  │  └─ proteomics_expert.py
│  ├─ genomics/             # Modular structure
│  │  ├─ __init__.py, prompts.py
│  │  └─ genomics_expert.py
│  ├─ machine_learning_expert.py
│  ├─ protein_structure_visualization_expert.py
│  ├─ visualization_expert.py
│  ├─ custom_feature_agent.py
│  └─ graph.py              # create_bioinformatics_graph(...)
│
├─ core/                    # Client, data, provenance, backends
│  ├─ client.py             # AgentClient (local)
│  ├─ api_client.py         # Cloud/WebSocket client
│  ├─ component_registry.py # PEP 420: service + agent discovery
│  ├─ plugin_loader.py      # Delegates to component_registry
│  ├─ data_manager_v2.py
│  ├─ provenance.py
│  ├─ download_queue.py
│  ├─ notebook_exporter.py
│  ├─ notebook_executor.py
│  ├─ utils/
│  │  └─ h5ad_utils.py      # H5AD validation, compression
│  ├─ interfaces/
│  │  ├─ base_client.py
│  │  ├─ data_backend.py
│  │  └─ modality_adapter.py
│  ├─ backends/
│  │  ├─ h5ad_backend.py
│  │  └─ mudata_backend.py
│  ├─ identifiers/
│  │  ├─ __init__.py
│  │  └─ accession_resolver.py  # 29 patterns
│  └─ schemas/
│     ├─ transcriptomics_schema.py
│     ├─ proteomics_schema.py
│     ├─ metabolomics_schema.py
│     ├─ metagenomics_schema.py
│     └─ database_mappings.py   # DATABASE_ACCESSION_REGISTRY
│
├─ services/                # Stateless analysis services
│  ├─ analysis/             # clustering, DE, GWAS, etc.
│  ├─ quality/              # QC & preprocessing
│  ├─ visualization/        # Plotly-based visualizations
│  ├─ data_access/          # GEO, SRA, PRIDE, etc.
│  │  └─ geo/               # Modular GEO package
│  ├─ data_management/      # Modality CRUD
│  ├─ metadata/             # Standardization, validation
│  ├─ ml/                   # Machine learning (ALPHA)
│  ├─ orchestration/        # Publication processing
│  └─ templates/            # Annotation templates
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

---

## Core Components Reference

| Area | File(s) | Role |
|------|---------|------|
| CLI | `cli.py` | User-facing commands, Rich output |
| LLM Config | `config/provider_setup.py` | Provider detection, validation |
| LLM Factory | `config/llm_factory.py` | Model instantiation |
| Client | `core/client.py`, `core/api_client.py` | local vs cloud clients |
| Graph | `agents/graph.py` | `create_bioinformatics_graph()` |
| Agents | `agents/*/` | supervisor + specialists |
| Data | `core/data_manager_v2.py` | modality/workspace orchestration |
| Provenance | `core/provenance.py` | W3C-PROV tracking |
| Queue | `core/download_queue.py` | download orchestration |
| Concurrency | `core/queue_storage.py` | multi-process safe file locking |
| Export | `core/notebook_exporter.py` | Jupyter pipeline export |
| Services | `services/*/*.py` | stateless analysis |
| Download | `tools/download_orchestrator.py` | Central router (9-step execution) |
| Identifiers | `core/identifiers/accession_resolver.py` | 29 patterns, thread-safe singleton |
| Registry | `config/agent_registry.py` | agent configuration |
| Component Registry | `core/component_registry.py` | PEP 420 service + agent discovery |

---

## Agent Roles

| Agent | Main Focus | Structure |
|-------|------------|-----------|
| `supervisor` | route user intents, manage handoffs | Single file |
| `research_agent` | literature & dataset discovery, URL extraction | Modular folder |
| `metadata_assistant` | ID mapping, validation, publication queue filtering | Modular folder |
| `data_expert` | **ZERO ONLINE ACCESS**: execute downloads, load files, manage modalities | Modular folder |
| `transcriptomics_expert` | scRNA-seq: QC, clustering, pseudobulk, markers | Modular folder |
| `annotation_expert` | Cell type annotation (sub-agent) | Modular folder |
| `de_analysis_expert` | Differential expression (sub-agent) | Modular folder |
| `proteomics_expert` | DDA/DIA workflows, normalization | Modular folder |
| `genomics_expert` | VCF/PLINK, GWAS, PCA, variant annotation | Modular folder |
| `machine_learning_expert` | ML predictions (PREMIUM) | Single file |
| `protein_structure_visualization_expert` | Structure viz (PREMIUM) | Single file |
| `visualization_expert` | General-purpose plotting | Single file |

---

## Deployment & Infrastructure

| File | Visibility | Purpose |
|------|------------|---------|
| `Dockerfile` | **PUBLIC** | CLI container (synced to lobster-local) |
| `Dockerfile.server` | **PRIVATE** | FastAPI server |
| `docker-compose.yml` | **PRIVATE** | Multi-service orchestration |
| `Makefile` | PUBLIC | Build/test automation |
| `.github/workflows/docker.yml` | PUBLIC | CI builds CLI only |
| `.github/workflows/sync-to-public.yml` | PRIVATE | Auto-syncs to lobster-local |
| `pyproject.toml` | PUBLIC | Dependencies (do NOT edit) |

**Sync strategy** (Single Source of Truth):
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
