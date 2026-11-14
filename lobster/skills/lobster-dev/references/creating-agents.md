# Creating Agents — Lobster AI 1.2

Start with [planning-workflow.md](planning-workflow.md) for a new biological
capability. For an existing agent's tool, read its actual factory and service
implementations before changing the contract.

## Scope the Capability

Inventory the inputs, scientific decisions, outputs and failure cases. Map each
operation to AQUADIF before choosing a package/agent structure:

| Category | Main operation | IR required |
|---|---|---|
| IMPORT | External data into workspace representation | Yes |
| QUALITY | Quality metrics and technical assessments | Yes |
| FILTER | Remove observations/features | Yes |
| PREPROCESS | Transform representation | Yes |
| ANALYZE | Statistics, embeddings, patterns | Yes |
| ANNOTATE | Biological meaning/labels | Yes |
| SYNTHESIZE | Combine scientific results | Yes |
| DELEGATE | Specialist handoff | No |
| UTILITY | Administrative/session support | No |
| CODE_EXEC | Custom code escape hatch | Exempt in the category gate |

Do not make arbitrary code the primary scientific workflow. Add a child agent
when the decision space and expertise are distinct; do not split agents merely
because vendor formats differ. Use entry-point discovery rather than counting
or hardcoding a fixed agent catalog.

### Data Loading Boundary

The Data Expert dispatches generic loading through adapters and downloads through
queue/database services. Domain IMPORT tools handle formats requiring domain
knowledge, such as vendor outputs or count-quantification directories. Do not
reimplement generic CSV/H5AD loading in every domain.

For GEO bulk count workflows,
`execute_download_from_queue(..., data_requirement="gene_counts")` requires a
suitable series-level gene-count import; empty requirement retains ordinary
imports. Refusals are scientific constraints, not a reason to substitute normalized
expression. DESeq2 count validation prefers valid `layers['counts']`, then `X`
only when the counts layer is absent. Do not round normalized matrices to pass.

## Generate the Package

```bash
lobster scaffold agent --name domain_expert --display-name "Domain Expert" \
  --description "Analyze the selected biological domain" -o packages/
lobster validate-plugin ./packages/lobster-domain/
```

External plugin authors omit `-o packages/` or choose their own approved directory.
Do not create package scaffolding by hand. The generated namespace, entry points,
configuration and tests are the starting point; placeholder services/fixtures must
be replaced with domain implementations.

The 1.2 scaffold generates `lobster-ai~=1.2`. Validate the plugin against the engine version that will execute it.

Keep engine/plugin compatibility aligned. Preserve shared PEP 420 boundaries:
no plugin `lobster/__init__.py` or `lobster/agents/__init__.py`. A domain's optional
`__init__.py` should expose only cheap package-specific state, not eager agent
factories or ImportError-based discovery.

## AGENT_CONFIG Before Heavy Imports

```python
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="domain_expert",
    display_name="Domain Expert",
    description="Specific supported inputs and analyses",
    factory_function="lobster.agents.domain.domain_expert.domain_expert",
    handoff_tool_name="handoff_to_domain_expert",
    handoff_tool_description="Route the selected domain's supported analyses here",
    supervisor_accessible=True,
    child_agents=None,
)
```

Entry-point name must match the config name. `factory_function` is the actual
module-and-function path. `supervisor_accessible=False` is appropriate for a
child routed through its parent. Provide concrete handoff descriptions; vague
"handle domain tasks" text is a poor routing signal.

Use the scaffold's factory rather than copying an old complete factory example.
The standard parameters include `data_manager`, `callback_handler`, `agent_name`,
`delegation_tools` and `workspace_path`; source signatures determine supported
optional inference overrides. Use the existing inference-construction path, not
an independently created LLM that loses the session's authorized context.

### Infrastructure Import Paths

| Symbol | Module |
|---|---|
| AgentRegistryConfig | `lobster.config.agent_registry` |
| DataManagerV2 | `lobster.core.runtime.data_manager` |
| AnalysisStep | `lobster.core.provenance.analysis_ir` |
| create_llm | `lobster.config.llm_factory` |
| get_settings | `lobster.config.settings` |
| create_react_agent | `langgraph.prebuilt` |
| AgentState | `langgraph.prebuilt.chat_agent_executor` |

Generated state inherits the appropriate AgentState rather than replacing it with
a messages-only state that lacks required graph fields. Heavy scientific imports
belong inside factories/services; no module-level `component_registry` calls.

## Tool Construction

Create factory-local tools using real stateless services:

```python
from langchain_core.tools import tool


@tool
def assess_quality(modality_name: str) -> str:
    """Assess the quality of the supplied modality."""
    adata = data_manager.get_modality(modality_name)
    output, stats, ir = quality_service.assess(adata)
    data_manager.store_modality(f"{modality_name}_quality_assessed", output)
    data_manager.log_tool_usage(
        "assess_quality",
        {"modality_name": modality_name},
        description="Assessed modality quality",
        ir=ir,
    )
    return str(stats)


assess_quality.metadata = {
    "categories": ["QUALITY"],
    "provenance": True,
    "suggests_templates": ["sc.qc"],  # only when that installed template matches the output
}
assess_quality.tags = ["QUALITY"]
```

This is an illustrative wrapper; `quality_service.assess` must be implemented
and produce the data expected by the suggested template. Avoid unsupported
suggestions: contract tests resolve declared names against the installed catalog.

- Metadata/tags are assigned after `@tool`, with the primary category first.
- Use parameterized collection annotations such as `list[str]`, not bare `list`.
- Use a fresh metadata dictionary per tool and matching category tags.
- Log a real IR describing the same operation, inputs, output and parameters.
- Do not assume every tool's first argument is a modality; import/search tools
  have different contracts. Follow the actual exposed schema.
- A returned error string is not successful science just because invocation
  returned normally. Tests must reject it.

## Scientific Results and Figures

An ordinary scientific tool produces data/results and may suggest a template.
It must not silently stage/save a figure as a side effect. Templates draw from
retained results without refitting or mutating the input. QC drawing requires
`assess_data_quality` first, then `create_qc_plots` on the full QC-assessed modality;
drawing does not compute missing metrics or accept DE/small-gene subsets.

For a supported figure producer, use the shared staging/saving path. Preview is
not a stored revision; report save outcomes, not merely Python object creation.
Custom-code figures require `submit_figure` explicitly. An edit loads a literal
saved revision ref and resubmits with the same parent; it does not replace the
original. Source data and Plotly config must be handled explicitly.

Add the executed figure contract alongside agent/AQUADIF tests. Ordinary
provenance-category tools need representative valid cases or explicit skip
reasons; transitional producers cannot skip and need saving cases. See
[testing.md](testing.md), [plugin-architecture.md](plugin-architecture.md) and
[aquadif-contract.md](aquadif-contract.md).

Factories offering custom code must use
`lobster.tools.code_tool_switch.attach_code_tool(tools, data_manager, AGENT_CONFIG.name,
 default=AGENT_CONFIG.code_tool_default)` rather than constructing and appending
the tool unconditionally. The helper returns a new tool list; use that result.
When disabled, it does not construct the code execution service. Declare the
plugin's default explicitly if code access is intended; the registry default is
`False`. The supervisor and selected built-ins enable it by default, but
`research_agent` has no code-tool wiring. See [cli.md](cli.md#code-tool-availability)
for canonical env names, workspace settings, precedence and limitations.
Custom code runs as your own OS user, not within a confinement boundary; run only code you trust.

## Prompt Design

The tool schema already carries names, parameters and docstrings. Do not duplicate
that catalog in a system prompt. The prompt should explain decision order,
scientific assumptions, boundaries and when to delegate.

Useful sections:

- Identity and domain responsibility.
- Constraints: unsupported tasks, representation requirements and output boundaries.
- Operational rules: sequential dependencies, replicate/design checks, naming,
  provenance, refusals and bounded retry behavior.
- Decision trees: when to choose an operation, ask for metadata or delegate.

Do not write "always normalize before DE". Analysis representation is method-
specific: count-model DE must preserve raw counts even when a normalized copy is
used for exploratory plots. Do not claim an analysis completed without inspecting
its actual outputs.

## Error Recovery and Privacy

`InferenceRefused` is terminal for the current authorized operation. Generic
recovery handlers must propagate it before logging exceptions or returning a
fallback:

```python
from lobster.core.exceptions import reraise_inference_refusal

try:
    result = run_operation()
except Exception as error:
    reraise_inference_refusal(error)
    # Handle only the remaining domain errors with a bounded, safe response.
    raise
```

Do not expose raw SDK request bodies or secrets, switch credentials silently, or
interpret a budget/policy failure as permission to run alternative science.
Handle custom-code execution as powerful code access, not a confinement claim.

## Completion Checklist

- Need and overlap assessment completed for new capabilities.
- Package generated with scaffold; entry points/config/factory paths agree.
- Real stateless services and typed tool schemas implemented.
- Metadata/tags, real IR and template suggestions match runtime behavior.
- Focused scientific/refusal tests and executed figure cases use real fixtures.
- Placeholder `tools_required=False` changed once working tools exist.
- Local discovery, approved CLI behavior and output paths checked.
- Prompt decision trees do not duplicate the schema catalog.
- Documentation describes only the delivered behavior; publication needs approval.
