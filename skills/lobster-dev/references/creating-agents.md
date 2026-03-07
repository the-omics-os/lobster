# Creating Agents

Complete guide for building Lobster AI agents — from scoping through PR.

## Scoping: Before You Write Code

### Tool Inventory via AQUADIF Categories

Enumerate candidate tools by AQUADIF category to ensure comprehensive domain coverage.
Read [references/aquadif-contract.md](aquadif-contract.md) first to internalize the taxonomy — AQUADIF is the design basis for all Lobster tools.

| AQUADIF Category | Purpose | Provenance |
|------------------|---------|------------|
| IMPORT | Load external data formats | Required |
| QUALITY | QC metrics, artifact detection | Required |
| FILTER | Subset data (remove samples/features) | Required |
| PREPROCESS | Transform representation (normalize, batch correct) | Required |
| ANALYZE | Extract patterns (PCA, clustering, statistics) | Required |
| ANNOTATE | Add biological meaning | Required |
| DELEGATE | Handoff to specialist child agent | Not required |
| SYNTHESIZE | Combine results across analyses | Required |
| UTILITY | Workspace management, status, export | Not required |
| CODE_EXEC | Custom code execution (escape hatch) | Conditional |

**Tool count target: 8–15 per agent.** <5 feels incomplete. >20 degrades LLM tool selection.

**AQUADIF guides completeness:** Every parent agent needs at minimum IMPORT + QUALITY + one of ANALYZE or DELEGATE. If a category is missing, verify it's intentional — the agent either handles it differently or delegates to a child.

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

### Package Boundary Decision

**New package when:** New omics domain, >1 agent, domain-specific services.
**Extend existing package when:** Adding a child, adding tools, adding services for existing domain.

### Data Loading Boundary: data_expert vs. Domain IMPORT Tools

Lobster has a **two-layer data loading architecture**. Understanding this boundary prevents your agent from duplicating functionality that already exists in core.

**Layer 1 — `data_expert` (core, in `lobster-research` package):**
- Downloads from public databases via `execute_download_from_queue` → `DownloadOrchestrator` → database-specific services (GEO, SRA, PRIDE, MetaboLights, etc.)
- Loads local files via `load_modality(adapter="...")` → generic adapter system
- Manages modalities (list, remove, concatenate, validate)
- Has **ZERO** domain-specific parsing logic — it dispatches to adapters

**Layer 2 — Domain agent IMPORT tools (your package):**
- Parse **vendor-specific or domain-specific file formats** that require domain expertise
- Apply scientific defaults during import (contaminant filtering, localization thresholds, orientation detection)
- Examples: MaxQuant `proteinGroups.txt` (proteomics), Salmon `quant.sf` directories (transcriptomics), VCF files via cyvcf2 (genomics)

**Decision rule for your IMPORT tools:**

| Your domain has... | Action |
|---------------------|--------|
| Vendor-specific formats requiring specialized parsing (e.g., narrowPeak, IDAT, vendor software output) | Create domain IMPORT tools in your agent |
| Only generic CSV/TSV/H5AD files | Register an **adapter** via `lobster.adapters` entry point — `data_expert.load_modality()` handles loading |
| Both specialized formats AND generic files | Create IMPORT tools for specialized formats; let `data_expert` handle generic loading |

**Do NOT** duplicate generic CSV/H5AD loading in your IMPORT tools — `data_expert` already handles that. Your IMPORT tools should accept domain-specific formats and apply domain knowledge that the generic adapter cannot.

**Real examples from existing agents:**
- `proteomics_expert.import_proteomics_data` — auto-detects MaxQuant/DIA-NN/Spectronaut, applies contaminant filtering, selects intensity types (LFQ vs raw)
- `transcriptomics_expert.import_bulk_counts` — auto-detects Salmon/kallisto/featureCounts, handles `#` comment headers, auto-transposes gene-vs-sample orientation
- `genomics_expert.load_vcf` — uses cyvcf2 for VCF parsing, supports region filtering, PASS-only selection
- `metabolomics_expert` — has NO import tools; metabolomics data loads through `data_expert.load_modality()` via adapters

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
    "lobster-ai>=1.0.13",
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

### __init__.py (PEP 420 Namespace)

PEP 420 namespace packages do not need `__init__.py` at the agent level. Agent discovery happens through entry points in `pyproject.toml`, not eager imports. If you need an `__init__.py` for state exports, keep it minimal:

```python
# __init__.py — PEP 420 namespace, minimal exports only
# Agent discovery happens through entry points in pyproject.toml
# Do NOT use try/except ImportError — it contradicts PEP 420 and Hard Rule #8

from lobster.agents.DOMAIN.state import DomainExpertState

__all__ = ["DomainExpertState"]
```

**WARNING:** Do NOT import agent functions or use `try/except ImportError` here. Entry points handle agent availability — see `pyproject.toml` entry points above.

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
    child_agents=["child_expert"],    # or None
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional
from langgraph.prebuilt import create_react_agent
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.runtime.data_manager import DataManagerV2
```

### Field Reference

| Field | Purpose |
|-------|---------|
| `name` | Internal ID. Must match entry point name. |
| `display_name` | Human-readable name for UI/logs. |
| `description` | Capabilities summary for component registry. |
| `factory_function` | Dotted path: `"lobster.agents.DOMAIN.domain_expert.domain_expert"` |
| `handoff_tool_name` | `"handoff_to_{name}"` — tool the supervisor calls. |
| `handoff_tool_description` | **Critical** — supervisor LLM reads this to decide routing. Be specific. |
| `supervisor_accessible` | `True` = supervisor routes directly. `False` = only via parent. |
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

### Critical Import Paths

**Do NOT guess these paths.** They are the only correct imports for agent infrastructure:

| Symbol | Correct Import | Common Hallucination |
|--------|---------------|----------------------|
| `create_llm` | `from lobster.config.llm_factory import create_llm` | ~~`lobster.core.llm_factory`~~ |
| `get_settings` | `from lobster.config.settings import get_settings` | ~~`lobster.core.settings`~~ |
| `AgentRegistryConfig` | `from lobster.config.agent_registry import AgentRegistryConfig` | — |
| `DataManagerV2` | `from lobster.core.runtime.data_manager import DataManagerV2` | — |
| `create_react_agent` | `from langgraph.prebuilt import create_react_agent` | — |
| `AgentState` | `from langgraph.prebuilt.chat_agent_executor import AgentState` | ~~`MessagesState`~~ |
| `get_logger` | `from lobster.utils.logger import get_logger` | ~~`lobster.core.logger`~~ |

The `lobster.config` package holds LLM/settings/registry. The `lobster.core` package holds data infrastructure. Do not mix them.

**State base class:** Your state class MUST extend `AgentState`, NOT `MessagesState`. `AgentState` adds `remaining_steps` which `create_react_agent` requires at runtime. Using `MessagesState` will pass all structural checks but fail when the graph tries to load the agent.

---

## Tool Design (shared_tools.py)

**Contributor only.** Adding tools to an existing agent requires write access to that agent's package source (e.g., `packages/lobster-visualization/`). There is no entry point mechanism to inject tools into another agent's factory from an external package. If you're a plugin author without repo access, your options are:
1. Contribute the tool upstream (fork → PR)
2. Create a new agent in your own package that complements the existing one

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

**Cardinal rule: NEVER describe a tool in both the system prompt AND the tool's docstring.** LangGraph serializes every `@tool` docstring into the JSON tool schema sent on every API call. If you also list tools in a `<Your_Tools>` section, the LLM sees the same info twice — wasting tokens and creating split-brain maintenance.

**Budget targets:** Prompt text should be 800-1200 tokens. Combined prompt + tool schemas should stay under 50% of the smallest target model's context window when added to supervisor overhead (~4,300 tokens).

**What goes WHERE:**

| Information | Goes in... | NOT in... |
|------------|-----------|-----------|
| Tool parameters, types | `@tool` docstring | System prompt |
| When to use tool A vs B | `<Decision_Trees>` | Tool docstring |
| Hard boundaries | `<Constraints>` | Scattered across sections |
| Error recovery guidance | Tool return values (just-in-time) | System prompt |
| Domain workflows from GPTomics | `<Decision_Trees>` branches | Duplicate `<Your_Tools>` listing |

```python
def create_domain_expert_prompt() -> str:
    return f"""You are the Domain Expert in Lobster AI. You handle [domain]-specific
analysis under the supervisor. You never interact with end users directly.

<Constraints>
DO: [2-3 bullet points — what this agent handles]
DO NOT:
- Literature search (research_agent), downloads (data_expert), direct user communication
- [domain-specific boundary, e.g., "generic CSV loading — data_expert handles via adapters"]
</Constraints>

<Operational_Rules>
1. [Sequencing — what must be sequential vs parallel]
2. [Error handling — fail-fast, retry limits]
3. [Naming — modality naming pattern: {{source}}_{{operation}}]
4. [Domain-specific — e.g., "always normalize before DE", from GPTomics research]
5. Provenance: every data-modifying tool must pass ir=ir to log_tool_usage
</Operational_Rules>

<Decision_Trees>

TASK_TYPE_1 (most common):
  tool_a → tool_b →
  ├─ Good result → tool_c
  └─ Error → investigate, retry once

TASK_TYPE_2:
  tool_d(params) →
  ├─ Condition A → handoff_to_child_expert
  └─ Condition B → tool_e

OUTSIDE SCOPE:
  → Report to supervisor, recommend correct agent

</Decision_Trees>

Today's date is {{date.today().isoformat()}}.
"""
```

**Note:** No `<Your_Tools>`, no `<Your_Environment>`, no `<Communication_Style>`. Tool schemas carry the "how"; decision trees carry the "when"; constraints carry the "what not". If you discovered domain workflows from GPTomics, encode them as decision tree branches — NOT as a duplicate tool listing.

### Required Sections

| Section | Purpose | Target Size | Required |
|---------|---------|-------------|----------|
| Identity (1-2 sentences) | Who, role, reports to supervisor | ~50 tok | Yes |
| `<Constraints>` | DO/DO NOT boundaries | ~100-200 tok | Yes |
| `<Operational_Rules>` | Numbered rules: sequencing, errors, naming | ~200-400 tok | Yes |
| `<Decision_Trees>` | WHEN to use which tool, tree-structured routing | ~200-500 tok | Yes |
| Domain-specific (1 section max) | Adapters, platform routing, validation levels | ~100-200 tok | If needed |
| Date footer | `Today's date is {date.today().isoformat()}.` | ~10 tok | Yes |

**NEVER include** (they duplicate tool schemas or waste tokens):

| Removed Section | Why | Where info lives instead |
|----------------|-----|--------------------------|
| `<Your_Tools>` | Duplicates tool docstrings | Tool schemas (auto-serialized) |
| `<Your_Environment>` | Unnecessary context | Agent knows it's an LLM |
| `<Example_Workflows>` | Duplicates decision trees | `<Decision_Trees>` |
| `<Communication_Style>` | Generic formatting | Models already format well |
| `<Core_Capabilities>` | Duplicates constraints | `<Constraints>` DO list |

---

## Parent-Child Hierarchy

### Topology

```
Supervisor
├── research_agent (parent)
│   └── metadata_assistant (child, shared)
├── data_expert_agent (parent)
│   └── metadata_assistant (child, shared)
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
├── drug_discovery_expert (parent)
│   ├── cheminformatics_expert (child)
│   ├── clinical_dev_expert (child)
│   └── pharmacogenomics_expert (child)
├── metabolomics_expert (no children)
├── visualization_expert_agent (no children)
└── protein_structure_visualization_expert (no children)
```

**Rules:** Supervisor → parents only. Parents → children via `delegation_tools`. No child-to-child.

**Parent AGENT_CONFIG:** `supervisor_accessible=True, child_agents=["child_name"]`
**Child AGENT_CONFIG:** `supervisor_accessible=False, child_agents=None`

The graph builder uses lazy delegation tools — children are resolved at invocation time, not creation time.

---

## Testing

See [testing.md](testing.md) for full testing patterns, fixtures, and CI setup.

**Required for every new agent:**
- Contract tests via `AgentContractTestMixin` (validates AQUADIF compliance, factory signature, entry points)
- Tool unit tests (mock `data_manager`, verify `ir=ir` in `log_tool_usage`)
- Semantic prompt tests (all required XML sections present, all tools documented by name)

---

## Checklist: New Agent from Zero to PR

```
RESEARCH
[ ] 1. Audit domain — data formats, workflows, key libraries
[ ] 2. Inventory candidate tools by workflow stage
[ ] 3. Decide: new package vs extend existing
[ ] 4. Decide: parent-only vs parent+children
[ ] 5. Decide: platform-aware defaults needed?

AQUADIF
[ ] 5a. Map candidate tools to AQUADIF categories (see aquadif-contract.md)
[ ] 5b. Verify: parent has IMPORT + QUALITY + ANALYZE or DELEGATE
[ ] 5c. Set .metadata and .tags on every tool in create_shared_tools()

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
[ ] 15. Write __init__.py — PEP 420 namespace, NO try/except ImportError (use entry points for discovery)

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
[ ] 29. No try/except ImportError in __init__.py (use entry points, not eager imports)
```
