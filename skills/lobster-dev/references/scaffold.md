# Scaffold Reference — `lobster scaffold`

The `lobster scaffold` command generates structurally correct, AQUADIF-compliant plugin packages. Use it as the **first step** when creating any new Lobster AI agent.

## Prerequisites

Before scaffolding, confirm lobster is installed:

```bash
which lobster          # Should print a path
lobster --version      # Should print version ≥1.0.0
```

If not installed: `uv pip install lobster-ai` in the appropriate venv

## Quick Start

```bash
# Generate a new agent package
lobster scaffold agent \
  --name <domain>_expert \
  --display-name "<Domain> Expert" \
  --description "<what this agent analyzes>"

# Validate the generated structure (should pass immediately)
lobster validate-plugin ./lobster-<domain>/
```

The scaffold generates a working package with placeholder tools. Your job is to **enhance** the generated files with real domain logic — see "What To Fill In" below.

## CLI Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--name` | Yes | — | Agent name in snake_case (e.g., `<domain>_expert`) |
| `--display-name` | Yes | — | Human-readable name (e.g., `"<Domain> Expert"`) |
| `--description` | Yes | — | Agent capabilities description |
| `--children` | No | — | Comma-separated child agent names |
| `--output-dir`, `-o` | No | `.` (current directory) | Output directory |
| `--author-name` | No | `Lobster AI Community` | Package author |
| `--author-email` | No | `community@lobsterbio.com` | Author email |

## Generated Structure

```
./lobster-<domain>/
├── pyproject.toml                          # Entry points + namespace config
├── README.md                               # Installation + usage
├── lobster/                                # PEP 420 namespace (NO __init__.py)
│   └── agents/                             # PEP 420 namespace (NO __init__.py)
│       └── <domain>/
│           ├── __init__.py                 # Minimal state exports only
│           ├── <domain>_expert.py          # AGENT_CONFIG + factory function
│           ├── shared_tools.py             # AQUADIF-categorized tools
│           ├── state.py                    # LangGraph AgentState subclass
│           ├── config.py                   # Platform/domain configuration
│           └── prompts.py                  # System prompt factory
└── tests/
    ├── __init__.py
    ├── conftest.py                         # Test fixtures
    └── test_contract.py                    # Contract compliance tests
```

**PEP 420 critical**: There must be NO `__init__.py` at the `lobster/` or `lobster/agents/` level. Only the domain directory (`lobster/agents/<domain>/`) has `__init__.py`.

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
# Use string literals for categories — do NOT import AquadifCategory enum
your_tool_name.metadata = {
    "categories": ["ANALYZE"],  # Primary category FIRST
    "provenance": True,
}
your_tool_name.tags = ["ANALYZE"]
```

**Category ordering rule**: Provenance-required categories (IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE) MUST be listed FIRST in the categories list. The contract test `test_provenance_categories_not_buried()` enforces this.

### 2. config.py — Add Platform Configuration

Define platform-specific settings as dataclasses:

```python
@dataclass
class DomainPlatformConfig:
    platform_type: str
    display_name: str
    # Add your fields...
    default_normalization: str
    min_coverage: int

PLATFORM_CONFIGS = {
    "<platform_a>": DomainPlatformConfig(...),
    "<platform_b>": DomainPlatformConfig(...),
}
```

### 3. prompts.py — Customize System Prompt

Edit the XML sections to describe your agent's actual capabilities and tools.

### 4. <domain>_expert.py — Wire Services

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
class DomainExpertState(AgentState):
    next: str = ""  # Required
    # Add domain-specific fields
    platform_type: str = ""
    analysis_results: Dict[str, Any] = {}
```

## Validation

After enhancing the scaffolded code, validate it:

```bash
lobster validate-plugin ./lobster-<domain>/
```

The 8 validation checks:
1. **PEP 420 compliance** — No `__init__.py` at namespace boundaries
2. **Entry points** — `lobster.agents` group in pyproject.toml
3. **AGENT_CONFIG position** — Before heavy imports
4. **Factory signature** — Standard parameters (data_manager, callback_handler, etc.)
5. **AQUADIF metadata** — Tools have .metadata with categories and provenance
6. **Provenance calls** — Tools with provenance=True call log_tool_usage(ir=ir)
7. **Import boundaries** — No cross-agent imports
8. **No ImportError guards** — Domain `__init__.py` must not use try/except ImportError

## Installing Your Scaffolded Package

The scaffold output is a **standalone PEP 420 namespace package**. It works identically whether you're in the lobster repo or building externally. You must install it so `ComponentRegistry` discovers your agent.

**Contributor (in the lobster repo):**
```bash
# Scaffold into the packages directory
lobster scaffold agent --name <domain>_expert ... -o packages/

# Reinstall all packages in dev mode
make dev-install
```

**Plugin author (standalone):**
```bash
# Scaffold anywhere
lobster scaffold agent --name <domain>_expert ...

# Install alongside your existing lobster installation
cd lobster-<domain>/
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
```

**Verify discovery (both modes):**
```bash
python -c "from lobster.core.component_registry import component_registry; component_registry.reset(); print(component_registry.has_agent('<domain>_expert'))"
# Should print: True
```

If it prints `False`, check that:
1. The package is actually installed: `uv pip list | grep lobster-<domain>`
2. Entry points are declared in `pyproject.toml` under `[project.entry-points."lobster.agents"]`
3. There is NO `__init__.py` at `lobster/` or `lobster/agents/` level (PEP 420 violation)

## With Child Agents

For agents that delegate to specialists:

```bash
lobster scaffold agent \
  --name <domain>_expert \
  --display-name "<Domain> Expert" \
  --description "<domain> analysis" \
  --children <sub>_expert,<sub2>_expert
```

This generates additional `<child_name>.py` files and wires entry points for all agents. The parent's `child_agents` field in AGENT_CONFIG is set automatically.
