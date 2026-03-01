# Scaffold Reference — `lobster scaffold`

The `lobster scaffold` command generates structurally correct, AQUADIF-compliant plugin packages. Use it as the **first step** when creating any new Lobster AI agent.

## Quick Start

```bash
# Generate a new agent package
lobster scaffold agent \
  --name epigenomics_expert \
  --display-name "Epigenomics Expert" \
  --description "Epigenomics analysis: bisulfite-seq, ATAC-seq, ChIP-seq" \
  --tier free

# Install the generated package
cd lobster-epigenomics
uv pip install -e '.[dev]'

# Run contract tests (should pass out of the box)
python -m pytest tests/ -v -m contract

# Validate plugin structure
lobster validate-plugin .
```

## CLI Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--name` | Yes | — | Agent name in snake_case (e.g., `epigenomics_expert`) |
| `--display-name` | Yes | — | Human-readable name (e.g., `"Epigenomics Expert"`) |
| `--description` | Yes | — | Agent capabilities description |
| `--tier` | No | `free` | Subscription tier: `free`, `premium`, `enterprise` |
| `--children` | No | — | Comma-separated child agent names |
| `--output-dir`, `-o` | No | `.` | Output directory |
| `--author-name` | No | `Lobster AI Community` | Package author |
| `--author-email` | No | `community@lobsterbio.com` | Author email |

## Generated Structure

```
lobster-{domain}/
├── pyproject.toml                          # Entry points + namespace config
├── README.md                               # Installation + usage
├── lobster/                                # PEP 420 namespace (NO __init__.py)
│   └── agents/                             # PEP 420 namespace (NO __init__.py)
│       └── {domain}/
│           ├── __init__.py                 # Graceful imports + availability flag
│           ├── {agent_name}.py             # AGENT_CONFIG + factory function
│           ├── shared_tools.py             # AQUADIF-categorized tools
│           ├── state.py                    # LangGraph AgentState subclass
│           ├── config.py                   # Platform/domain configuration
│           └── prompts.py                  # System prompt factory
└── tests/
    ├── __init__.py
    ├── conftest.py                         # Test fixtures
    └── test_contract.py                    # Contract compliance tests
```

**PEP 420 critical**: There must be NO `__init__.py` at the `lobster/` or `lobster/agents/` level. Only the domain directory (`lobster/agents/{domain}/`) has `__init__.py`.

## What To Fill In After Scaffolding

The scaffold generates working code with TODO markers. Fill in these sections:

### 1. shared_tools.py — Add Domain-Specific Tools

The scaffold generates 4 example tools (IMPORT, QUALITY, ANALYZE, UTILITY). Modify these and add more:

```python
# Each tool follows this pattern:
@tool
def your_tool_name(modality_name: str, ...) -> str:
    """Tool docstring (shown to LLM)."""
    adata = data_manager.get_modality(modality_name)

    # Call your stateless service
    result_adata, stats, ir = your_service.method(adata, ...)

    # Log provenance (REQUIRED for IMPORT/QUALITY/FILTER/PREPROCESS/ANALYZE/ANNOTATE/SYNTHESIZE)
    data_manager.log_tool_usage(
        tool_name="your_tool_name",
        parameters={...},
        description="What happened",
        ir=ir,
    )
    return f"Result: {stats}"

# AQUADIF metadata (MUST be assigned after @tool)
your_tool_name.metadata = {
    "categories": [AquadifCategory.ANALYZE.value],  # Primary category FIRST
    "provenance": True,
}
your_tool_name.tags = [AquadifCategory.ANALYZE.value]
```

**Category ordering rule**: Provenance-required categories (IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE) MUST be listed FIRST in the categories list. The contract test `test_provenance_categories_not_buried()` enforces this.

### 2. config.py — Add Platform Configuration

Define platform-specific settings as dataclasses:

```python
@dataclass
class EpigenomicsPlatformConfig:
    platform_type: str
    display_name: str
    # Add your fields...
    default_normalization: str
    min_coverage: int

PLATFORM_CONFIGS = {
    "bisulfite_seq": EpigenomicsPlatformConfig(...),
    "atac_seq": EpigenomicsPlatformConfig(...),
}
```

### 3. prompts.py — Customize System Prompt

Edit the XML sections to describe your agent's actual capabilities and tools.

### 4. {agent_name}.py — Wire Services

Initialize your stateless services in the factory function:

```python
# Inside the factory function:
quality_service = YourQualityService()
analysis_service = YourAnalysisService()

shared_tools = create_shared_tools(
    data_manager,
    quality_service,
    analysis_service,
)
```

### 5. state.py — Add Domain State Fields

Add fields your agent needs to track:

```python
class EpigenomicsExpertState(AgentState):
    next: str = ""  # Required
    methylation_type: str = ""  # bisulfite, RRBS, etc.
    peak_results: Dict[str, Any] = {}
```

## Validation

After modifying the scaffolded code, validate it:

```bash
# Run contract tests
python -m pytest tests/ -v -m contract

# Run structural validation (7 checks)
lobster validate-plugin .
```

The 7 validation checks:
1. **PEP 420 compliance** — No `__init__.py` at namespace boundaries
2. **Entry points** — `lobster.agents` group in pyproject.toml
3. **AGENT_CONFIG position** — Before heavy imports
4. **Factory signature** — Standard parameters (data_manager, callback_handler, etc.)
5. **AQUADIF metadata** — Tools have .metadata with categories and provenance
6. **Provenance calls** — Tools with provenance=True call log_tool_usage(ir=ir)
7. **Import boundaries** — No cross-agent imports

## Installing and Testing

```bash
# Install in development mode
cd lobster-{domain}
uv pip install -e '.[dev]'

# Verify agent discovery
python -c "from lobster.core.component_registry import component_registry; component_registry.reset(); print(component_registry.list_agents())"

# Run all tests
python -m pytest tests/ -v

# Run contract tests only
python -m pytest tests/ -v -m contract
```

## With Child Agents

For agents that delegate to specialists:

```bash
lobster scaffold agent \
  --name epigenomics_expert \
  --display-name "Epigenomics Expert" \
  --description "Epigenomics analysis" \
  --children methylation_expert,chromatin_expert
```

This generates additional `{child_name}.py` files and wires entry points for all agents. The parent's `child_agents` field in AGENT_CONFIG is set automatically.
