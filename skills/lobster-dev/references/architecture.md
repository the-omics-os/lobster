# Architecture Reference

This document explains Lobster AI's architecture. For implementation details, read the source code directly.

## High-Level Flow

```
User Query
    ↓
CLI (lobster/cli.py)
    ↓
LobsterClientAdapter
    ↓
AgentClient (local) or CloudLobsterClient (cloud)
    ↓
LangGraph Graph (lobster/agents/graph.py)
    ↓
Supervisor → Specialist Agents
    ↓
Tools → Services → DataManagerV2
    ↓
Results + Provenance (W3C-PROV)
```

## Core Components

### ComponentRegistry (`lobster/core/component_registry.py`)

Single source of truth for component discovery. Uses Python entry points (PEP 420).
Discovers 7 entry point groups: agents, services, agent_configs, adapters, providers, download_services, queue_preparers.

```python
from lobster.core.component_registry import component_registry

agents = component_registry.list_agents()
agent_config = component_registry.get_agent("transcriptomics_expert")
adapter = component_registry.get_adapter("metabolomics_lc_ms")  # Returns factory callable
provider = component_registry.get_provider("metabolights")
```

**Key methods** (same pattern for all groups — get/has/list):
- `list_agents()` / `get_agent(name)` / `has_agent(name)`
- `list_adapters()` / `get_adapter(name)` / `has_adapter(name)`
- `list_providers()` / `get_provider(name)` / `has_provider(name)`
- `list_download_services()` / `list_queue_preparers()`

**Full docs:** [docs.omics-os.com/docs/core/component-registry](https://docs.omics-os.com/docs/core/component-registry)

### DataManagerV2 (`lobster/core/data_manager_v2.py`)

Manages data, workspaces, modalities, and provenance persistence.

```python
from pathlib import Path
from lobster.core.data_manager_v2 import DataManagerV2

dm = DataManagerV2(
    workspace_path=Path("./workspace"),  # Required
    session_dir=Path("./workspace/.lobster/sessions/my_session"),  # Provenance persistence
    default_backend="local",              # "local" or "s3"
    enable_provenance=True,               # Enable W3C-PROV tracking
    auto_scan=True,                       # Auto-scan for modalities
    console=None                          # Optional Rich console
)

# Load data
adata = dm.get_modality("transcriptomics")

# Store processed data
dm.store_modality("transcriptomics_processed", processed_adata)

# Log operations for provenance (persisted to session_dir/provenance.jsonl)
dm.log_tool_usage("analyze", params, stats, ir=ir)

# List available modalities
modalities = dm.list_modalities()
```

**Session directory flow:**
```
CLI (init_client) or AgentClient
  → computes session_dir = workspace/.lobster/sessions/{session_id}/
  → passes to DataManagerV2(session_dir=...)
    → passes to ProvenanceTracker(session_dir=...)
      → writes provenance.jsonl (append-only, one JSON line per activity)
      → writes metadata.json (session timestamps, activity count)
```

When `session_dir` is provided, provenance survives process exit and can be restored by a new client using the same session_id.

### Provenance (`lobster/core/provenance.py`)

W3C-PROV compliant tracking for reproducibility. Supports optional disk persistence.

```python
from lobster.core.provenance import AnalysisStep, ProvenanceTracker

# Create analysis step
ir = AnalysisStep(
    activity_type="differential_expression",
    inputs={"n_samples": 100},
    outputs={"n_degs": 500},
    params={"fdr": 0.05}
)

# Track in provenance (with disk persistence)
tracker = ProvenanceTracker(session_dir=Path("./sessions/my_session"))
tracker.create_activity("de_analysis", ir)
# → Appends to ./sessions/my_session/provenance.jsonl

# Without disk persistence (backward compatible)
tracker = ProvenanceTracker()  # session_dir=None → memory only
tracker.create_activity("de_analysis", ir)
```

### Graph Builder (`lobster/agents/graph.py`)

Creates the LangGraph workflow.

```python
from lobster.agents.graph import create_bioinformatics_graph

graph = create_bioinformatics_graph(
    agent_configs=registry.get_available_agents(),
    data_manager=dm,
    llm=llm
)
```

## Agent Architecture

### Supervisor (`lobster/agents/supervisor.py`)

Routes user intents to specialist agents. Uses lazy delegation tools.

### Specialist Agents

Each domain has specialist agents in separate packages:

| Package | Agents |
|---------|--------|
| `lobster-transcriptomics` | transcriptomics_expert, annotation_expert, de_analysis_expert |
| `lobster-research` | research_agent, data_expert_agent |
| `lobster-visualization` | visualization_expert_agent |
| `lobster-proteomics` | proteomics_expert [alpha] |
| `lobster-genomics` | genomics_expert [alpha] |
| `lobster-ml` | machine_learning_expert [alpha] |

## Service Architecture

Services are stateless analysis functions returning 3-tuples:

```python
class MyService:
    def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict, AnalysisStep]:
        # Analysis logic
        result = process(adata)
        stats = {"status": "complete", "n_results": len(result)}
        ir = AnalysisStep(...)
        return result, stats, ir
```

Services live in:
- `lobster/services/` — Core services (stay in core)
- `packages/lobster-{domain}/lobster/services/{domain}/` — Domain services

## Data Flow

### Download Flow (9-step orchestrator)

1. URL detection and validation
2. Source identification (GEO, SRA, etc.)
3. Metadata extraction
4. Download planning
5. File download
6. Format detection
7. Loading into AnnData
8. Quality validation
9. Workspace registration

See `lobster/tools/download_orchestrator.py`.

### Analysis Flow

1. User query → Supervisor
2. Supervisor routes to specialist agent
3. Agent selects appropriate tool
4. Tool calls service
5. Service processes data, returns 3-tuple
6. Tool logs operation with provenance
7. Result returned to user

## Modality System

Lobster supports multiple data types (modalities) via the **Omics Plugin Architecture**:

- `transcriptomics` — scRNA-seq, bulk RNA-seq (GEO, SRA)
- `proteomics` — DDA/DIA mass spec, affinity (PRIDE, MassIVE, GEO)
- `genomics` — VCF, PLINK, GWAS (GEO, SRA, dbGaP)
- `metabolomics` — LC-MS, GC-MS, NMR (MetaboLights, Metabolomics Workbench, GEO)
- `metagenomics` — 16S, ITS, shotgun (SRA, GEO)

Each modality has:
- **OmicsTypeConfig** in `OmicsTypeRegistry` — detection keywords, preferred databases, QC thresholds
- **Adapters** for format conversion (discovered via `lobster.adapters` entry points)
- **Providers** for database search (discovered via `lobster.providers` entry points)
- **Download services + queue preparers** (discovered via entry points)
- Specialized agents

New omics types can be added via entry points — zero core changes needed.

**Plugin guide:** [references/plugin-architecture.md](plugin-architecture.md)
**Full docs:** [docs.omics-os.com/docs/architecture/overview](https://docs.omics-os.com/docs/architecture/overview)

## For More Details

- Architecture decisions: `.planning/PROJECT.md`
- Plugin architecture: [references/plugin-architecture.md](plugin-architecture.md)
- Online docs: [docs.omics-os.com/docs/architecture](https://docs.omics-os.com/docs/architecture/overview)
