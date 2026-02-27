# Architecture Research: AQUADIF Tool Metadata Integration

**Domain:** Multi-agent bioinformatics system with LangChain/LangGraph callback infrastructure
**Researched:** 2026-02-27
**Confidence:** HIGH

## Executive Summary

The AQUADIF taxonomy integrates with Lobster AI's existing LangChain/LangGraph architecture through **native metadata propagation**. LangChain's `BaseTool` provides `metadata: dict[str, Any]` and `tags: list[str]` fields that automatically propagate to callback handlers via kwargs at every tool invocation. This means AQUADIF categories can be assigned post-`@tool` creation and consumed in callbacks without modifying core LangGraph execution flow.

**Key architectural insight:** Metadata flows FROM tool definition → THROUGH LangGraph → TO callback handlers as first-class citizens of the execution context. This enables runtime monitoring, provenance enforcement, and prompt generation without bypassing or extending LangGraph's callback system.

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION LAYER                           │
│  CLI / Cloud API / Jupyter Notebook → LobsterClientAdapter              │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                          LANGGRAPH ORCHESTRATION                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ create_bioinformatics_graph() — Agent Discovery & Tool Assembly   │ │
│  │  - ComponentRegistry.list_agents() (entry point discovery)        │ │
│  │  - Agent factory invocation → tools list                          │ │
│  │  - Delegation tool creation (_create_lazy_delegation_tool)        │ │
│  │  - METADATA ASSIGNMENT happens HERE (post-@tool, pre-agent build) │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Supervisor (create_react_agent) with delegation tools             │ │
│  └──────┬─────────────────────────────────────────────────────────────┘ │
│         │ invoke_agent_lazy() → child agent invocation                  │
│  ┌──────▼─────────────────────────────────────────────────────────────┐ │
│  │ Worker Agents (18 total across 9 packages)                         │ │
│  │  - transcriptomics_expert, genomics_expert, proteomics_expert...  │ │
│  │  - Each with ~10-20 domain-specific tools                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                       CALLBACK HANDLER LAYER                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ BaseCallbackHandler.on_tool_start(serialized, input_str, **kwargs)│ │
│  │   kwargs['metadata'] = {'categories': [...], 'provenance': bool}  │ │
│  │   kwargs['tags'] = ['CATEGORY1', 'CATEGORY2', ...]                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ AquadifCallbackHandler (NEW) — Runtime Monitoring                  │ │
│  │  - Logs category with every tool invocation                        │ │
│  │  - Tracks category distribution per session                        │ │
│  │  - Flags CODE_EXEC usage                                           │ │
│  │  - Checks provenance compliance (IR expected but not logged?)      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ TextualCallbackHandler (EXISTING) — UI updates                     │ │
│  │ TerminalCallbackHandler (EXISTING) — CLI display                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│                       DATA & PROVENANCE LAYER                             │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ DataManagerV2 — Modality orchestration                             │ │
│  │  - log_tool_usage(tool_name, params, description, ir=AnalysisStep)│ │
│  │  - IR parameter MANDATORY for provenance-required categories       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ ProvenanceTracker — W3C-PROV tracking                              │ │
│  │  - Stores AnalysisStep IR for reproducibility                      │ │
│  │  - Exports to Jupyter notebooks via Papermill                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Current Implementation | AQUADIF Changes |
|-----------|---------------|----------------------|-----------------|
| **ComponentRegistry** | Agent discovery via entry points (`lobster.agents`) | `lobster/core/component_registry.py` | None — metadata is per-tool, not per-agent |
| **Agent Factory** | Creates tools + returns agent instance | `transcriptomics_expert.py:75-2350` (example) | Add 2 lines per tool: `tool.metadata = {...}`, `tool.tags = [...]` |
| **graph.py** | Assembles all agents + creates delegation tools | `lobster/agents/graph.py:280-600` | Add metadata to delegation tools in `_create_lazy_delegation_tool()` |
| **BaseTool** | LangChain tool abstraction with native metadata/tags | `langchain_core.tools.BaseTool` (external) | None — already supports what we need |
| **BaseCallbackHandler** | LangGraph execution event observer | `langchain_core.callbacks.BaseCallbackHandler` | Subclass for AQUADIF monitoring |
| **AquadifCallbackHandler** | NEW — AQUADIF-aware callback handler | NEW: `lobster/core/aquadif_callback.py` | Logs categories, checks provenance compliance |
| **DataManagerV2** | Modality management + provenance logging | `lobster/core/data_manager_v2.py:1728` | None — `log_tool_usage(ir=...)` already enforces provenance |
| **ProvenanceTracker** | W3C-PROV activity tracking | `lobster/core/provenance.py` | None — consumes AnalysisStep IR unchanged |
| **Contract Tests** | Agent compliance validation | `lobster/testing/contract_mixins.py` | Add 5 test methods for AQUADIF validation |

## Data Flow

### Metadata Assignment Flow

```
1. Agent Factory Function Invoked
   └─> graph.py: agent_factory = import_agent_factory(agent_config)
       └─> agent_factory(data_manager, callback_handler, delegation_tools)

2. Tools Created Inside Factory Closure
   └─> @tool decorator creates StructuredTool instances
       └─> Tools captured in closure with data_manager, services

3. METADATA ASSIGNMENT (Post-@tool, Inside Factory)
   └─> assess_quality.metadata = {"categories": ["QUALITY"], "provenance": True}
   └─> assess_quality.tags = ["QUALITY"]
   └─> ✓ Safe: @tool schema extraction already complete
   └─> ✓ Co-located: metadata lives WITH tool, no drift

4. Factory Returns Tools List + Agent
   └─> return create_react_agent(llm, tools=[...])

5. Graph Assembly
   └─> created_agents[agent_name] = agent
   └─> Delegation tools created with DELEGATE metadata
```

**Critical constraint:** Metadata assignment MUST happen **after** `@tool` creation but **before** agent creation. Modifying `BaseTool.metadata` or `.tags` after schema extraction is safe because these fields are not part of the tool's args schema.

### Runtime Execution Flow (Tool Invocation)

```
User Query
    ↓
Supervisor Agent (with delegation tools)
    ↓
handoff_to_transcriptomics_expert(task_description)
    ↓ [LangGraph invokes tool]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LangGraph Tool Invocation Pipeline                               │
│                                                                   │
│  1. BaseCallbackHandler.on_tool_start() FIRES                    │
│     ├─> serialized = {"name": "...", "description": "..."}       │
│     └─> kwargs = {                                               │
│           "metadata": {"categories": ["DELEGATE"], "prov...},    │
│           "tags": ["DELEGATE"],                                  │
│           "run_id": UUID, "parent_run_id": UUID, ...             │
│         }                                                         │
│                                                                   │
│  2. ALL Callback Handlers Receive kwargs['metadata'] & ['tags']  │
│     ├─> AquadifCallbackHandler.on_tool_start()                   │
│     │    └─> Log category: "Tool: handoff_to_... [DELEGATE]"     │
│     │    └─> Update session stats: DELEGATE_count += 1           │
│     │    └─> Check: provenance=False → no IR validation needed   │
│     │                                                              │
│     ├─> TextualCallbackHandler.on_tool_start()                   │
│     │    └─> Update UI activity log (existing behavior)          │
│     │                                                              │
│     └─> TerminalCallbackHandler.on_tool_start()                  │
│          └─> Print "🔧 Tool: handoff_to_..." (existing behavior) │
│                                                                   │
│  3. Tool Function Executes                                        │
│     └─> transcriptomics_expert agent invoked                      │
│                                                                   │
│  4. BaseCallbackHandler.on_tool_end() FIRES                      │
│     └─> AquadifCallbackHandler: Log completion with duration     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
    ↓
Worker Agent Tool Invoked (e.g., assess_quality)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Provenance-Required Tool (categories: ["QUALITY"])               │
│                                                                   │
│  1. on_tool_start() — AquadifCallbackHandler                     │
│     └─> metadata["provenance"] = True                            │
│     └─> Register expectation: IR must be logged                  │
│                                                                   │
│  2. Tool Function Executes                                        │
│     ├─> service.assess_quality(adata, ...) → (adata, stats, ir) │
│     └─> data_manager.log_tool_usage(tool_name, params, desc,    │
│                                      ir=AnalysisStep)             │
│         └─> ProvenanceTracker.record_activity(ir)                │
│         └─> AquadifCallbackHandler: Mark IR expectation fulfilled│
│                                                                   │
│  3. on_tool_end() — AquadifCallbackHandler                       │
│     └─> If provenance=True AND no IR logged → WARN (or ERROR)   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key mechanism:** LangChain's `RunnableConfig` propagates `metadata` and `tags` from tool definitions to callback handlers at every invocation. No custom propagation logic needed.

### Provenance Compliance Check Flow

```
Tool with provenance=True invoked
    ↓
AquadifCallbackHandler.on_tool_start()
    └─> Extract primary_category from metadata["categories"][0]
    └─> Check if primary_category in PROVENANCE_REQUIRED
    └─> If yes: Register expectation { run_id: True }

Tool function executes
    ↓
data_manager.log_tool_usage(tool_name, params, desc, ir=AnalysisStep)
    ↓
ProvenanceTracker.record_activity(ir)
    ↓
AquadifCallbackHandler.on_provenance_logged(run_id)  # Custom event
    └─> Mark expectation fulfilled: { run_id: False }

AquadifCallbackHandler.on_tool_end()
    └─> Check if run_id in expectations AND still True
    └─> If yes: logger.error(f"{tool_name} requires provenance but none logged")
    └─> Emit structured event for Omics-OS Cloud analytics
```

**Enforcement point:** `AquadifCallbackHandler` maintains a `pending_provenance_checks: dict[UUID, bool]` mapping `run_id` → expectation. Tools with `provenance=True` register expectations; `data_manager.log_tool_usage()` clears them. Mismatches are logged as compliance violations.

## Component Boundaries

### 1. AQUADIF Configuration (`lobster/config/aquadif.py`)

**NEW file** — category enum + provenance rules

```python
from enum import Enum

class AquadifCategory(str, Enum):
    IMPORT = "IMPORT"
    QUALITY = "QUALITY"
    FILTER = "FILTER"
    PREPROCESS = "PREPROCESS"
    ANALYZE = "ANALYZE"
    ANNOTATE = "ANNOTATE"
    DELEGATE = "DELEGATE"
    SYNTHESIZE = "SYNTHESIZE"
    UTILITY = "UTILITY"
    CODE_EXEC = "CODE_EXEC"

PROVENANCE_REQUIRED = {
    AquadifCategory.IMPORT,
    AquadifCategory.QUALITY,
    AquadifCategory.FILTER,
    AquadifCategory.PREPROCESS,
    AquadifCategory.ANALYZE,
    AquadifCategory.ANNOTATE,
    AquadifCategory.SYNTHESIZE,
}
```

**Boundary:** Pure configuration, no runtime logic. Imported by agent factories, tests, callbacks.

### 2. Agent Factory Metadata Assignment (Per-Package)

**MODIFIED files** — agent factory functions (e.g., `transcriptomics_expert.py`)

**Pattern:**
```python
@tool
def assess_quality(modality_name: str) -> str:
    """Assess data quality for a modality."""
    # ... tool implementation ...

# AQUADIF metadata assignment (after @tool, before agent creation)
assess_quality.metadata = {"categories": ["QUALITY"], "provenance": True}
assess_quality.tags = ["QUALITY"]
```

**Boundary:** Tool factories are responsible for self-documenting their capabilities. No central registry of tools needed — metadata is co-located.

### 3. Delegation Tool Metadata (`lobster/agents/graph.py`)

**MODIFIED function** — `_create_lazy_delegation_tool()`

**Pattern:**
```python
@tool(f"handoff_to_{_name}", description=f"Delegate task to {_name}. {_desc}")
def invoke_agent_lazy(task_description: str) -> str:
    # ... delegation logic ...

# AQUADIF metadata for delegation tools
invoke_agent_lazy.metadata = {"categories": ["DELEGATE"], "provenance": False}
invoke_agent_lazy.tags = ["DELEGATE"]

return invoke_agent_lazy
```

**Boundary:** Graph creation owns delegation tool metadata. Dynamic tools get metadata at creation time.

### 4. AQUADIF Callback Handler (`lobster/core/aquadif_callback.py`)

**NEW file** — runtime monitoring callback handler

**Responsibilities:**
- Log category with every tool invocation
- Track category distribution per session (`session_stats: dict[str, int]`)
- Check provenance compliance (register expectations, validate fulfillment)
- Emit structured events for Cloud analytics
- Flag CODE_EXEC usage with details

**Methods:**
```python
class AquadifCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        # Extract metadata from kwargs
        # Log category
        # Register provenance expectation if needed

    def on_tool_end(self, output, **kwargs):
        # Check provenance compliance
        # Log completion with category

    def on_provenance_logged(self, run_id: UUID):
        # Custom event fired by data_manager.log_tool_usage()
        # Mark expectation fulfilled

    def get_session_stats(self) -> dict[str, int]:
        # Return category distribution
```

**Boundary:** Observes tool execution via LangChain callbacks. Does NOT modify tool behavior or execution flow. Purely monitoring + compliance checking.

### 5. Contract Test Mixins (`lobster/testing/contract_mixins.py`)

**MODIFIED class** — `AgentContractTestMixin`

**New test methods:**
```python
def test_tools_have_metadata(self):
    # Every tool has .metadata set

def test_categories_are_valid(self):
    # All categories from AQUADIF set

def test_categories_capped_at_three(self):
    # Max 3 categories per tool

def test_provenance_tools_have_flag(self):
    # Provenance-required categories declare provenance=True

def test_minimum_viable_parent(self):
    # Parent agents have IMPORT + QUALITY + (ANALYZE or DELEGATE)
```

**Boundary:** Test infrastructure validates AQUADIF compliance during CI. Runs on all 18 agents.

### 6. Prompt Auto-Generation (`lobster/agents/prompts.py` per package)

**MODIFIED files** — prompt generation utilities

**Function:**
```python
def generate_tool_prompt_section(tools: List[BaseTool]) -> str:
    """Generate <Your_Tools> section from tool metadata.

    Groups tools by primary category for structured presentation.
    """
    from collections import defaultdict

    by_category = defaultdict(list)
    for tool in tools:
        primary = tool.metadata.get("categories", ["UTILITY"])[0]
        by_category[primary].append(tool)

    sections = []
    for category in sorted(by_category.keys()):
        sections.append(f"## {category} Tools")
        for tool in by_category[category]:
            sections.append(f"- {tool.name}: {tool.description}")

    return "\n".join(sections)
```

**Boundary:** Consumes tool metadata to generate prompt text. Replaces hardcoded tool lists in agent prompts. Phase 6 work (requires all tools to have metadata first).

## Integration Points

### LangChain/LangGraph Integration (External)

| LangChain Component | AQUADIF Usage | Notes |
|---------------------|--------------|-------|
| `BaseTool.metadata` | Category + provenance storage | Native Pydantic field, dict type |
| `BaseTool.tags` | Category propagation | Native Pydantic field, list type |
| `BaseCallbackHandler.on_tool_start()` | Metadata consumption | `kwargs['metadata']` and `kwargs['tags']` |
| `RunnableConfig` | Metadata propagation | LangGraph passes tool metadata to callbacks |
| `create_react_agent()` | Agent creation | Tools with metadata → agent with metadata-aware tools |

**Compatibility:** LangChain 0.3+ (current Lobster dependency). `metadata` and `tags` fields existed since LangChain 0.1 but propagation to callbacks improved in 0.3.

### Lobster Internal Boundaries

| Lobster Module | Interface with AQUADIF | Direction |
|----------------|----------------------|-----------|
| `ComponentRegistry` | None — metadata is per-tool, not per-agent | N/A |
| `DataManagerV2` | `log_tool_usage(ir=...)` validates provenance | Callback → DataManager |
| `ProvenanceTracker` | Consumes AnalysisStep IR unchanged | DataManager → Provenance |
| Agent factories | Assign metadata to tools | Factory → Tool |
| `graph.py` | Assign metadata to delegation tools | Graph → Tool |
| Contract tests | Validate metadata compliance | Test → Agent |

## Build Order & Dependencies

### Phase 1: Create AQUADIF Skill (lobster-dev update)
**Dependencies:** None
**Deliverable:** Updated skill files in `skills/lobster-dev/`

This MUST be first because if the approach isn't teachable, we need to know before touching 180 tools.

### Phase 2: Contract Test Specification
**Dependencies:** Phase 1 (skill defines what tests should validate)
**Deliverable:**
- `lobster/config/aquadif.py` (enum + provenance set)
- Updated `lobster/testing/contract_mixins.py` (5 new test methods)

Tests define "correct" before implementing. No validation = no way to know if metadata is right.

### Phase 3: Reference Implementation (Transcriptomics)
**Dependencies:** Phase 2 (tests exist to validate)
**Deliverable:** All 24 tools in `transcriptomics_expert.py` have metadata

Validates pattern on most mature agent. Serves as reference for Phase 4.

### Phase 4: Roll Out to All Agents
**Dependencies:** Phase 3 (reference pattern proven)
**Deliverable:** ~156 additional tools across 16 remaining agents

Systematic application of reference pattern. Contract tests validate each agent.

### Phase 5: Callback Handler (Parallel with Phase 6)
**Dependencies:** Phase 4 (all tools have metadata)
**Deliverable:** `lobster/core/aquadif_callback.py`

Monitoring infrastructure. Can be built in parallel with Phase 6.

### Phase 6: Prompt Auto-Generation (Parallel with Phase 5)
**Dependencies:** Phase 4 (all tools have metadata)
**Deliverable:** `generate_tool_prompt_section()` utility + updated prompts

Replaces hardcoded tool lists. Can be built in parallel with Phase 5.

### Phase 7: Extension Case Study
**Dependencies:** Phases 1-6 complete
**Deliverable:** Epigenomics package built by coding agent following AQUADIF skill

Publication evidence. Requires everything working end-to-end.

## Architectural Patterns

### Pattern 1: Post-Decorator Metadata Assignment

**What:** Assign `metadata` and `tags` AFTER `@tool` creation, inside factory closure.

**When to use:** Every tool in every agent factory function.

**Trade-offs:**
- ✓ Safe: Schema extraction already complete, no conflicts
- ✓ Co-located: Metadata lives with tool definition
- ✓ No dependencies: Works with existing `@tool` decorator
- ✗ Manual: Each tool needs 2 lines added (but this is what makes it teachable)

**Example:**
```python
def transcriptomics_expert(data_manager, callback_handler, delegation_tools):
    # ... services initialization ...

    @tool
    def assess_quality(modality_name: str) -> str:
        """Assess data quality."""
        result, stats, ir = quality_service.assess(adata)
        data_manager.log_tool_usage("assess_quality", params, stats, ir=ir)
        return format_result(stats)

    # AQUADIF metadata (after @tool, before agent creation)
    assess_quality.metadata = {"categories": ["QUALITY"], "provenance": True}
    assess_quality.tags = ["QUALITY"]

    # ... more tools ...

    tools = [assess_quality, ...]
    return create_react_agent(llm, tools)
```

### Pattern 2: Callback Handler Provenance Compliance

**What:** Track provenance expectations in `on_tool_start()`, validate in `on_tool_end()`.

**When to use:** Tools with `provenance=True` in categories that require IR.

**Trade-offs:**
- ✓ Runtime enforcement: Catches missing IR at execution time
- ✓ Non-blocking: Logs warnings, doesn't break execution
- ✓ Observable: Emits structured events for analytics
- ✗ Stateful: Callback handler maintains `pending_provenance_checks` dict

**Example:**
```python
class AquadifCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.pending_provenance_checks: dict[UUID, str] = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        metadata = kwargs.get("metadata", {})
        if metadata.get("provenance"):
            run_id = kwargs.get("run_id")
            tool_name = serialized.get("name")
            self.pending_provenance_checks[run_id] = tool_name

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id in self.pending_provenance_checks:
            tool_name = self.pending_provenance_checks.pop(run_id)
            logger.warning(
                f"Tool '{tool_name}' requires provenance but none logged"
            )
```

### Pattern 3: Category-Based Prompt Generation

**What:** Group tools by primary category in agent prompts.

**When to use:** Phase 6 — after all tools have metadata.

**Trade-offs:**
- ✓ Auto-updates: Add tool → prompt updates automatically
- ✓ Structured: LLM sees tools grouped by capability
- ✓ No drift: Prompt always matches actual tool set
- ✗ Delayed: Can't use until all tools have metadata

**Example:**
```python
def create_transcriptomics_expert_prompt(tools):
    tool_section = generate_tool_prompt_section(tools)

    return f"""You are a transcriptomics analysis expert.

<Your_Tools>
{tool_section}
</Your_Tools>

Use IMPORT tools to load data, QUALITY tools to validate...
"""
```

## Anti-Patterns

### Anti-Pattern 1: Module-Level ComponentRegistry Calls

**What people might do:** Call `component_registry.get_service()` at module top-level to access metadata.

**Why it's wrong:** Triggers loading ALL agents at import time (Hard Rule #11). Causes slow startup and unwanted side effects.

**Do this instead:** Use lazy functions that call `component_registry` inside function bodies, not at module level.

### Anti-Pattern 2: Custom Metadata Propagation

**What people might do:** Create custom middleware to propagate metadata from tools to callbacks.

**Why it's wrong:** LangChain already does this natively via `RunnableConfig`. Custom propagation would duplicate LangChain's built-in mechanism.

**Do this instead:** Use `BaseTool.metadata` and `BaseTool.tags` — they propagate automatically.

### Anti-Pattern 3: Centralized Metadata Registry

**What people might do:** Create a dict mapping `tool_name → metadata` in `AGENT_CONFIG`.

**Why it's wrong:** Metadata drifts from tool definitions. No enforcement that tools exist for registered metadata.

**Do this instead:** Co-locate metadata with tool definitions via post-`@tool` assignment.

### Anti-Pattern 4: Pre-Decorator Metadata Assignment

**What people might do:** Try to pass metadata as argument to `@tool` decorator.

**Why it's wrong:** `@tool` decorator doesn't accept `metadata` parameter. Would require custom decorator, adding complexity.

**Do this instead:** Assign after `@tool` creation — it's safe and works with LangChain's native fields.

## Scalability Considerations

| Scale | AQUADIF Impact | Mitigation |
|-------|---------------|------------|
| 180 tools | Manual metadata assignment for all tools | Phase 4 systematic rollout, contract tests validate |
| 18 agents across 9 packages | Entry-point discovery already handles scale | Metadata is per-tool, no agent-level changes |
| Callback overhead | `on_tool_start()` fires for every tool invocation | Metadata extraction is O(1) dict lookup, negligible |
| Provenance compliance checks | Stateful tracking per run_id | Use WeakValueDictionary to prevent memory leaks |
| Prompt generation | Runs once per agent creation | No runtime cost, only initialization |

**Performance validation:** Add `AquadifCallbackHandler` overhead measurement in Phase 5. Target: <1ms per tool invocation.

## Sources

### HIGH Confidence

- **LangChain BaseTool source code** (`langchain_core/tools/base.py`) — metadata/tags fields verified via Python introspection
- **Existing Lobster codebase** (`lobster/agents/graph.py`, `lobster/core/data_manager_v2.py`, `lobster/utils/callbacks.py`) — callback propagation patterns confirmed
- **Test validation** (Python test script) — metadata propagation to `on_tool_start(**kwargs)` verified empirically

### MEDIUM Confidence

- **LangChain callback documentation** (redirected, not directly accessible) — relied on source code inspection instead
- **Provenance tracking integration** — inferred from existing `log_tool_usage()` calls, not explicitly documented

### LOW Confidence

- None — all architectural claims are backed by code inspection or empirical testing.

---
*Architecture research for: AQUADIF Tool Metadata Integration in Lobster AI*
*Researched: 2026-02-27*
