# Development Rules & Patterns

Comprehensive development guidelines. Load on-demand when implementing new features.

---

## Service Pattern (3-tuple)

All analysis services MUST return:

```python
def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
    ...
    return processed_adata, stats_dict, ir
```

- `processed_adata`: modified AnnData
- `stats_dict`: concise, human-readable summary
- `ir`: `AnalysisStep` for provenance + notebook export

---

## Tool Pattern (agent tools)

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

---

## Provenance & AnalysisStep IR (W3C-PROV)

**Every logged step must emit IR. No IR → not reproducible.**

`AnalysisStep` captures:
- `operation` (e.g. `"scanpy.pp.calculate_qc_metrics"`)
- `tool_name` (service method)
- `description` (human explanation)
- `library` (e.g. `"scanpy"`, `"pyDESeq2"`)
- `code_template` (Jinja2 snippet with `{{ params }}`)
- `imports` (pure Python imports)
- `parameters` + `parameter_schema`
- `input_entities` / `output_entities`

**Checklist for new services**:
- [ ] Returns `Tuple[AnnData, Dict[str, Any], AnalysisStep]`
- [ ] Helper `_create_ir(**params)` builds IR + parameter schema
- [ ] Jinja2 `code_template` uses `{{ param }}` only, standard libs only
- [ ] Agent tools call `log_tool_usage(..., ir=ir)`
- [ ] Works with `/pipeline export` + notebook execution

---

## Patterns & Abstractions

### Queue Pattern
Use `DownloadQueue` for multi-step downloads. See `core/download_queue.py`.

### Concurrency Pattern (`core/queue_storage.py`)
Multi-process safe file access:
- `InterProcessFileLock` – file-based lock
- `queue_file_lock(thread_lock, lock_path)` – combines threading.Lock + file lock
- `atomic_write_json(path, data)` – crash-safe writes
- **Protected files**: download_queue.jsonl, publication_queue.jsonl, .session.json

### Redis Rate Limiting (`tools/rate_limiter.py`)
- Thread-safe connection pool for NCBI API
- `redis.ConnectionPool` with `health_check_interval=30`
- `reset_redis_pool()` for test isolation

### Workspace Resolution (`core/workspace.py`)
- `resolve_workspace(explicit_path, create)` – single entry point
- Resolution order: explicit path > `LOBSTER_WORKSPACE` env > `cwd/.lobster_workspace`
- **Rule**: All code MUST use `resolve_workspace()` instead of hardcoding

### Error Hierarchy
- `ModalityNotFoundError` – missing dataset
- `ServiceError` – analysis failures
- `ValidationError` – schema/data issues
- `ProviderError` – external API failures

### Registry Pattern (`config/agent_registry.py`)
```python
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

### Component Registry Pattern (`core/component_registry.py`)
PEP 420 namespace packages for premium features:
- **No `lobster/__init__.py`** – enables namespace merging
- Entry point groups: `lobster.services`, `lobster.agents`
- `component_registry.get_service('name')` → class or None
- **Rule**: ALL premium features MUST use component_registry, NOT `try/except ImportError`

```toml
[project.entry-points."lobster.services"]
my_service = "package.module:ServiceClass"

[project.entry-points."lobster.agents"]
my_agent = "package.module:AGENT_CONFIG"
```

### AccessionResolver Pattern (`core/identifiers/accession_resolver.py`)
- Thread-safe singleton via `get_accession_resolver()`
- 29 pre-compiled patterns in `DATABASE_ACCESSION_REGISTRY`
- Case-insensitive matching
- **Rule**: All providers MUST use AccessionResolver, not hardcoded regex

### Unified Workspace Tool (`tools/workspace_tool.py`)
- 5 Workspace Types: literature, data, metadata, download_queue, publication_queue
- `get_content_from_workspace(identifier, workspace, level, status_filter)`
- `write_to_workspace(identifier, workspace, ...)`

### Custom Code Tool Factory (`tools/custom_code_tool.py`)
```python
from lobster.tools.custom_code_tool import create_execute_custom_code_tool
execute_custom_code = create_execute_custom_code_tool(
    data_manager, custom_code_service, "agent_name", post_processor=None
)
```
**Rule**: Do NOT define inline `execute_custom_code` in agents; always use factory

### Modular Agent Structure (`agents/*/`)
Each agent: `__init__.py`, `config.py`, `prompts.py`, `state.py` (optional), main agent file
**Template**: `agents/unified_agent_creation_template.md`
**Rule**: New agents MUST follow this pattern

---

## ModalityManagementService Pattern

**Location**: `lobster/services/data_management/modality_management_service.py`

5 standardized methods, all returning (result, stats, ir):
- `list_modalities(filter_pattern)` – List with optional glob filtering
- `get_modality_info(modality_name)` – Detailed info
- `remove_modality(modality_name)` – Remove from DataManagerV2
- `validate_compatibility(modality_names)` – Validate obs/var overlap
- `load_modality(modality_name, file_path, adapter, ...)` – Load via adapter system

---

## Naming & Data Quality

**Naming convention**:
```text
geo_gse12345
├─ geo_gse12345_quality_assessed
├─ geo_gse12345_filtered_normalized
├─ geo_gse12345_doublets_detected
├─ geo_gse12345_clustered
├─ geo_gse12345_markers
└─ geo_gse12345_annotated
```

**Data standards**:
- W3C-PROV compliant logging
- Pydantic schema validation for all modalities
- Good QC metrics at each step
- Proper missing-value handling (esp. proteomics)

---

## Security Considerations

**CustomCodeExecutionService** (`services/execution/custom_code_execution_service.py`):
- Subprocess isolation (Phase 1 hardening)
- ✅ Production-ready for local CLI
- ❌ Cloud SaaS requires Docker isolation (Phase 2)

---

## Feature Tiering & Conditional Activation

**Key Files**:
- `lobster/config/subscription_tiers.py` – Tier definitions
- `lobster/core/component_registry.py` – Service + agent discovery
- `lobster/core/license_manager.py` – Entitlement validation

**Rules**:
1. Agent factories: Accept `subscription_tier: str = "free"`
2. New premium agents: Register via entry points
3. Premium services: Use `component_registry.get_service('name')`
4. Feature checks: Use `is_agent_available(agent, tier)`
5. Handoff restrictions: Define in `subscription_tiers.py`
6. Graceful degradation: Return helpful messages when tier-restricted

**Tier Reference**:
| Tier | Agents |
|------|--------|
| FREE | research_agent, data_expert, transcriptomics_expert, visualization_expert, annotation_expert, de_analysis_expert |
| PREMIUM | + metadata_assistant, proteomics_expert, machine_learning_expert, protein_structure_visualization_expert |
| ENTERPRISE | + lobster-custom-* packages |

---

## Custom Package Development

**Template**: `/Users/tyo/GITHUB/omics-os/lobster-custom-template/`

**Critical Rules**:
1. **Import paths**: ALL `lobster.*` → `lobster_custom_{customer}.*`
2. **Dependencies**: Copy ALL transitive dependencies
3. **Versioning**: Track lobster-ai version
4. **Entry points**: Register in `pyproject.toml`
5. **AGENT_CONFIG**: Define at module top (before heavy imports)
6. **Package structure**: Add `namespaces = true`

```python
# In my_agent.py - DEFINE FIRST (top of file)
AGENT_CONFIG = AgentRegistryConfig(
    name="my_agent",
    display_name="My Agent",
    ...
)

# Heavy imports below (after AGENT_CONFIG)
from lobster.core.data_manager_v2 import DataManagerV2
```
