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

Single source of truth for agent discovery. Uses Python entry points (PEP 420).

```python
from lobster.core.component_registry import ComponentRegistry

registry = ComponentRegistry()
agents = registry.get_available_agents()
agent_config = registry.get_agent("transcriptomics_expert")
```

**Key methods:**
- `get_available_agents()` — Returns all discovered agents
- `get_agent(name)` — Get specific agent config
- `is_agent_available(name, tier)` — Check if agent is available at tier (all official agents are free)

### DataManagerV2 (`lobster/core/data_manager_v2.py`)

Manages data, workspaces, and modalities.

```python
from pathlib import Path
from lobster.core.data_manager_v2 import DataManagerV2

dm = DataManagerV2(
    workspace_path=Path("./workspace"),  # Required
    default_backend="local",              # "local" or "s3"
    enable_provenance=True,               # Enable W3C-PROV tracking
    auto_scan=True,                       # Auto-scan for modalities
    console=None                          # Optional Rich console
)

# Load data
adata = dm.get_modality("transcriptomics")

# Store processed data
dm.store_modality("transcriptomics_processed", processed_adata)

# Log operations for provenance
dm.log_tool_usage("analyze", params, stats, ir=ir)

# List available modalities
modalities = dm.list_modalities()
```

### Provenance (`lobster/core/provenance.py`)

W3C-PROV compliant tracking for reproducibility.

```python
from lobster.core.provenance import AnalysisStep, ProvenanceTracker

# Create analysis step
ir = AnalysisStep(
    activity_type="differential_expression",
    inputs={"n_samples": 100},
    outputs={"n_degs": 500},
    params={"fdr": 0.05}
)

# Track in provenance
tracker = ProvenanceTracker()
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

Lobster supports multiple data types (modalities):

- `transcriptomics` — scRNA-seq, bulk RNA-seq
- `proteomics` — DDA/DIA mass spec
- `genomics` — VCF, PLINK, GWAS
- `metabolomics` — (planned)

Each modality has:
- Adapters for format conversion
- Services for domain-specific analysis
- Specialized agents

## For More Details

Read the source code:
- Architecture decisions: `.planning/PROJECT.md`
- Implementation: `lobster/core/`, `lobster/agents/`
- Online docs: docs.omics-os.com/docs/architecture
