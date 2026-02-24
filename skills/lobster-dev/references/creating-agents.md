# Creating Agents

Complete guide for building Lobster AI agents — from scoping through PR.

## Scoping: Before You Write Code

### Tool Inventory

Enumerate candidate tools by workflow stage:

| Stage | Example Tools |
|-------|--------------|
| Import/Load | `import_data`, `load_from_file` |
| QC | `assess_quality`, `calculate_qc_metrics` |
| Filtering | `filter_samples`, `filter_features` |
| Preprocessing | `normalize`, `batch_correct`, `impute` |
| Analysis | `run_analysis`, `run_statistics` |
| Annotation | `annotate_features`, `classify_subtypes` |
| Export | `export_results` |

**Tool count target: 8–15 per agent.** <5 feels incomplete. >20 degrades LLM tool selection.

### Parent vs Child Decision

**Create a child agent when ALL are true:**
1. Distinct user skill required (e.g., clinical genomicist vs population geneticist)
2. Different knowledge bases needed (e.g., ClinVar vs GWAS catalog)
3. >5 tools with a different focus area
4. The child workflow is a natural "next step" after the parent

**Keep on the parent when ANY are true:**
1. Downstream analysis is identical across sub-platforms
2. Auto-detection handles platform differences
3. Total tool count <15
4. Workflow is linear without branching

**Real examples:**
- Created `variant_analysis_expert` as child of `genomics_expert` — clinical interpretation is a distinct skill
- Did NOT create `affinity_proteomics_expert` — affinity and MS share identical DE and biomarker tools
- `metabolomics_expert` has no children — 10 tools, linear workflow

### Package Boundary Decision

**New package when:** New omics domain, >1 agent, domain-specific services.
**Extend existing package when:** Adding a child, adding tools, adding services for existing domain.

---

## Package Structure

### Directory Layout

```
packages/lobster-DOMAIN/
├── pyproject.toml
├── lobster/
│   ├── agents/
│   │   └── DOMAIN/
│   │       ├── __init__.py          # Tier-aware exports
│   │       ├── DOMAIN_expert.py     # Agent factory + AGENT_CONFIG
│   │       ├── config.py            # Constants, thresholds, PlatformConfig
│   │       ├── prompts.py           # System prompt builder function
│   │       ├── state.py             # LangGraph state class
│   │       └── shared_tools.py      # Tool factory (8-15 tools)
│   └── services/
│       ├── quality/
│       │   └── DOMAIN_quality_service.py
│       └── analysis/
│           └── DOMAIN_analysis_service.py
└── tests/
    └── unit/
        └── agents/
            └── test_DOMAIN_expert.py
```

**With children:**
```
lobster/agents/DOMAIN/
├── DOMAIN_expert.py          # Parent: child_agents=["child_expert"]
├── child_expert.py           # Child: supervisor_accessible=False
├── config.py                 # Shared config
├── prompts.py                # Prompt builders for ALL agents in domain
├── state.py                  # State classes for ALL agents
└── shared_tools.py           # Parent tools (children get their own)
```

**No `lobster/__init__.py` anywhere** — PEP 420 namespace package.

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "lobster-DOMAIN"
version = "1.0.0"
description = "DOMAIN agent for Lobster AI"
license = "AGPL-3.0-or-later"
authors = [{name = "Omics-OS", email = "info@omics-os.com"}]
requires-python = ">=3.12,<3.14"
dependencies = [
    "lobster-ai~=1.0.0",
    # Domain-specific deps here
]

[project.entry-points."lobster.agents"]
domain_expert = "lobster.agents.DOMAIN.domain_expert:AGENT_CONFIG"
# If children exist:
child_expert = "lobster.agents.DOMAIN.child_expert:AGENT_CONFIG"

[project.entry-points."lobster.states"]
DomainExpertState = "lobster.agents.DOMAIN.state:DomainExpertState"

# CRITICAL: Enable PEP 420 namespace merging
[tool.setuptools]
packages.find = {where = ["."], include = ["lobster*"], namespaces = true}
```

### __init__.py (Tier-Aware)

```python
from lobster.agents.DOMAIN.state import DomainExpertState

try:
    from lobster.agents.DOMAIN.domain_expert import domain_expert
    DOMAIN_EXPERT_AVAILABLE = True
except ImportError:
    DOMAIN_EXPERT_AVAILABLE = False
    domain_expert = None

__all__ = ["DOMAIN_EXPERT_AVAILABLE", "domain_expert", "DomainExpertState"]
```

---

## AGENT_CONFIG

**Must be at the TOP of the agent file**, before any heavy imports. Enables <50ms entry point discovery.

```python
"""Domain Expert agent for Lobster AI."""

from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="domain_expert",
    display_name="Domain Expert",
    description="One-sentence with key capabilities",
    factory_function="lobster.agents.DOMAIN.domain_expert.domain_expert",
    handoff_tool_name="handoff_to_domain_expert",
    handoff_tool_description="When to route here: capability A, capability B, capability C",
    supervisor_accessible=True,       # False for child agents
    tier_requirement="free",          # All official agents are free
    child_agents=["child_expert"],    # or None
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional
from langgraph.prebuilt import create_react_agent
from lobster.core.data_manager_v2 import DataManagerV2
```

### Field Reference

| Field | Purpose |
|-------|---------|
| `name` | Internal ID. Must match entry point name. |
| `display_name` | Human-readable name for UI/logs. |
| `description` | Capabilities summary for component registry. |
| `factory_function` | Dotted path: `"module.path:function_name"` |
| `handoff_tool_name` | `"handoff_to_{name}"` — tool the supervisor calls. |
| `handoff_tool_description` | **Critical** — supervisor LLM reads this to decide routing. Be specific. |
| `supervisor_accessible` | `True` = supervisor routes directly. `False` = only via parent. |
| `tier_requirement` | `"free"` for official. `"enterprise"` for custom packages. |
| `child_agents` | List of child names. `None` = no children. |

**The `handoff_tool_description` is the single most important field for routing.**

Good: `"Assign genomics tasks: WGS/SNP QC, GWAS, LD pruning, kinship, clumping, variant annotation."`
Bad: `"Handle genomics tasks"`

---

## Factory Function

```python
def domain_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "domain_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    **kwargs,
) -> CompiledGraph:
    # 1. Lazy prompt import (allows AGENT_CONFIG discovery before prompts.py exists)
    from lobster.agents.DOMAIN.prompts import create_domain_expert_prompt

    # 2. LLM creation
    settings = get_settings()
    model_params = settings.get_agent_llm_params("domain_expert")
    llm = create_llm(
        "domain_expert", model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # 3. Callback wiring
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # 4. Validate data manager
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("Requires DataManagerV2 for modular analysis")

    # 5. Initialize stateless services
    quality_service = DomainQualityService()
    analysis_service = DomainAnalysisService()

    # 6. Create tools via shared_tools factory
    tools = create_shared_tools(data_manager, quality_service, analysis_service)

    # 7. Add delegation tools for children
    if delegation_tools:
        tools = tools + delegation_tools

    # 8. Build prompt and return agent
    system_prompt = create_domain_expert_prompt()
    return create_react_agent(
        model=llm, tools=tools, prompt=system_prompt,
        name=agent_name, state_schema=DomainExpertState,
    )
```

**Key points:**
- Lazy prompt import (step 1) — enables incremental development
- Services instantiated inside factory, not at module level
- `delegation_tools` appended, not prepended — domain tools first in LLM's tool list

---

## Tool Design (shared_tools.py)

Every tool follows this exact structure:

```python
def create_shared_tools(data_manager, quality_service, analysis_service) -> list:
    """Create domain tools. Returns list of 8-15 tool functions."""

    @tool
    def assess_quality(modality_name: str) -> str:
        """Assess data quality. Args: modality_name - dataset to assess."""
        adata = data_manager.get_modality(modality_name)
        if adata is None:
            return f"Error: Modality '{modality_name}' not found. Use list_modalities."

        result, stats, ir = quality_service.assess(adata)
        new_name = f"{modality_name}_quality_assessed"
        data_manager.store_modality(new_name, result)
        data_manager.log_tool_usage("assess_quality", {"modality_name": modality_name}, stats, ir=ir)
        return f"QC complete. Stored as '{new_name}'. {stats}"

    @tool
    def normalize_data(modality_name: str, method: str = None) -> str:
        """Normalize data. Args: modality_name - dataset, method - normalization method."""
        adata = data_manager.get_modality(modality_name)
        if adata is None:
            return f"Error: Modality '{modality_name}' not found."

        result, stats, ir = analysis_service.normalize(adata, method=method)
        new_name = f"{modality_name}_normalized"
        data_manager.store_modality(new_name, result)
        data_manager.log_tool_usage("normalize_data", {"modality_name": modality_name, "method": method}, stats, ir=ir)
        return f"Normalization ({method}) complete. Stored as '{new_name}'. {stats}"

    return [assess_quality, normalize_data]
```

### Tool Rules

1. `modality_name` is always the first parameter
2. Tool name in `log_tool_usage()` must match the function name
3. `ir=ir` is mandatory — never skip
4. Return a string — tells the LLM what happened and what to do next
5. Use `return f"Error: ..."` not exceptions for user-facing errors

### Modality Naming

```
{source}_{id}                        # raw import
{source}_{id}_quality_assessed       # after QC
{source}_{id}_filtered               # after filtering
{source}_{id}_normalized             # after normalization
{source}_{id}_clustered              # after clustering
{source}_{id}_annotated              # after annotation
{source}_{id}_de_results             # differential expression
```

### Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Missing `ir=ir` in log_tool_usage | Always pass IR |
| `len(param)` without None guard | `len(param) if param else 'all'` |
| Inline tool definitions in agent file | Use `shared_tools.py` factory |
| Module-level `component_registry` calls | Use lazy functions inside factory |
| Silent platform default | Log warning when auto-detecting |

---

## Platform-Aware Design

Use when one agent handles multiple data sub-types (e.g., LC-MS vs GC-MS vs NMR).

### PlatformConfig

```python
# config.py
@dataclass
class PlatformConfig:
    platform_type: str
    display_name: str
    default_normalization: str
    default_imputation: str
    log_transform: bool
    max_missing_per_sample: float
    max_missing_per_feature: float

PLATFORM_CONFIGS: Dict[str, PlatformConfig] = {
    "lc_ms": PlatformConfig(platform_type="lc_ms", display_name="LC-MS", ...),
    "gc_ms": PlatformConfig(...),
    "nmr": PlatformConfig(...),
}
```

### Auto-Detection + Usage in Tools

```python
def create_shared_tools(data_manager, ..., force_platform_type=None):
    _forced = force_platform_type

    def _get_platform(modality_name):
        if _forced:
            return get_platform_config(_forced)
        adata = data_manager.get_modality(modality_name)
        return get_platform_config(detect_platform_type(adata))

    @tool
    def normalize(modality_name: str, method: str = None) -> str:
        platform = _get_platform(modality_name)
        method = method or platform.default_normalization  # Auto-default
        # ... rest of tool
```

---

## System Prompts (prompts.py)

```python
def create_domain_expert_prompt() -> str:
    return f"""<Identity_And_Role>
You are the Domain Expert in Lobster AI's multi-agent architecture.

<Core_Mission>
- [Key workflow 1]
- [Key workflow 2]
</Core_Mission>
</Identity_And_Role>

<Your_Environment>
You are a specialist agent in 'lobster-ai' by Omics-OS.
You operate in a LangGraph supervisor-multi-agent architecture.
You report to the supervisor — never interact with end users directly.
</Your_Environment>

<Your_Responsibilities>
- [Responsibility 1]
- [Responsibility 2]
</Your_Responsibilities>

<Your_Not_Responsibilities>
- Literature search (research_agent)
- Downloading datasets (data_expert)
- Direct user communication (supervisor only)
</Your_Not_Responsibilities>

<Your_Tools>
**Import & Loading:**
- `import_data(...)` — Load raw data files

**Quality Control:**
- `assess_quality(...)` — Calculate QC metrics

**Analysis:**
- `run_analysis(...)` — Main analysis method
</Your_Tools>

<Decision_Tree>
**Handle directly:** [domain]-specific QC, preprocessing, analysis
**Delegate to child:** [child domain] tasks → handoff_to_child_expert
**Return to supervisor:** Tasks outside scope, analysis complete
</Decision_Tree>

<Important_Rules>
1. Always store results as new modalities (never overwrite source)
2. Use naming: {{source}}_{{id}}_{{operation}}
3. Report clear metrics to supervisor
4. Today's date: {date.today().isoformat()}
</Important_Rules>
"""
```

### Required Sections

| Section | Required |
|---------|----------|
| `<Identity_And_Role>` | Yes |
| `<Your_Environment>` | Yes |
| `<Your_Responsibilities>` | Yes |
| `<Your_Not_Responsibilities>` | Yes |
| `<Your_Tools>` (grouped by stage) | Yes |
| `<Decision_Tree>` | Yes |
| `<Important_Rules>` | Yes |
| `<Data_Types>` | If multi-platform |
| `<Quality_Control_Workflow>` | If QC-heavy |

---

## Parent-Child Hierarchy

### Topology

```
Supervisor
├── transcriptomics_expert (parent)
│   ├── annotation_expert (child)
│   └── de_analysis_expert (child)
├── genomics_expert (parent)
│   └── variant_analysis_expert (child)
├── proteomics_expert (parent)
│   ├── proteomics_de_analysis_expert (child)
│   └── biomarker_discovery_expert (child)
├── machine_learning_expert (parent)
│   ├── feature_selection_expert (child)
│   └── survival_analysis_expert (child)
└── metabolomics_expert (no children)
```

**Rules:** Supervisor → parents only. Parents → children via `delegation_tools`. No child-to-child.

**Parent AGENT_CONFIG:** `supervisor_accessible=True, child_agents=["child_name"]`
**Child AGENT_CONFIG:** `supervisor_accessible=False, child_agents=None`

The graph builder uses lazy delegation tools — children are resolved at invocation time, not creation time.

---

## Testing

### Contract Tests (Required)

```python
from lobster.testing import AgentContractTestMixin

class TestDomainExpertAgent(AgentContractTestMixin):
    agent_module = "lobster.agents.DOMAIN.domain_expert"
    factory_name = "domain_expert"
```

### Tool Unit Tests

```python
def test_assess_quality(mock_data_manager):
    tools = create_shared_tools(mock_data_manager, quality_service)
    result = tools[0].invoke({"modality_name": "test_dataset"})
    assert "complete" in result.lower()
    mock_data_manager.store_modality.assert_called_once()
    # Verify IR was passed
    call_kwargs = mock_data_manager.log_tool_usage.call_args
    assert "ir" in call_kwargs.kwargs or len(call_kwargs.args) >= 4
```

### Semantic Prompt Tests

```python
def test_prompt_has_required_sections():
    prompt = create_domain_expert_prompt()
    for section in ["<Identity_And_Role>", "<Your_Tools>", "<Decision_Tree>", "<Important_Rules>"]:
        assert section in prompt

def test_prompt_documents_all_tools():
    prompt = create_domain_expert_prompt()
    tools = create_shared_tools(mock_dm)
    for t in tools:
        assert t.name in prompt, f"Tool '{t.name}' not documented in prompt"
```

---

## Checklist: New Agent from Zero to PR

```
RESEARCH
[ ] 1. Audit domain — data formats, workflows, key libraries
[ ] 2. Inventory candidate tools by workflow stage
[ ] 3. Decide: new package vs extend existing
[ ] 4. Decide: parent-only vs parent+children
[ ] 5. Decide: platform-aware defaults needed?

SCAFFOLD
[ ] 6. Create package directory structure
[ ] 7. Write pyproject.toml with entry points
[ ] 8. Create state.py with state class
[ ] 9. Create config.py with constants + PlatformConfig if needed

IMPLEMENT
[ ] 10. Write services following 3-tuple contract
[ ] 11. Write shared_tools.py with tool factory (8-15 tools)
[ ] 12. Write AGENT_CONFIG at top of agent file
[ ] 13. Write factory function
[ ] 14. Write prompts.py with all required sections
[ ] 15. Write __init__.py with tier-aware exports

TEST
[ ] 16. Contract tests pass
[ ] 17. Tool unit tests pass
[ ] 18. Semantic prompt tests pass
[ ] 19. Agent creates successfully with mock data_manager

INTEGRATE
[ ] 20. Entry points registered and discoverable
[ ] 21. Graph wires agent correctly
[ ] 22. Supervisor routes for relevant queries
[ ] 23. Child handoffs work (if applicable)

VERIFY
[ ] 24. Tool count in 8-15 range
[ ] 25. All tools log with ir=ir
[ ] 26. Modality naming follows conventions
[ ] 27. Prompt documents all tools by name
[ ] 28. No module-level component_registry calls
```
