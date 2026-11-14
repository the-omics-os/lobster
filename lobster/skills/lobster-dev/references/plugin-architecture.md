# Plugin Architecture — Lobster AI 1.2

Plugins extend domain capabilities without editing a central catalog. Inspect
current source and use scaffold for new agent packages; metadata declarations
alone are not evidence that a plugin executes correctly.

## Discovery Boundaries

| Mechanism | Discovers |
|---|---|
| `ComponentRegistry`: `lobster.agents` | Agent configuration objects |
| `ComponentRegistry`: `lobster.services` | Service classes |
| `ComponentRegistry`: `lobster.adapters` | Adapter factories/classes |
| `ComponentRegistry`: `lobster.providers` | Database providers |
| `ComponentRegistry`: `lobster.download_services` | Download services |
| `ComponentRegistry`: `lobster.queue_preparers` | Queue preparers |
| Separate `lobster.omics_types` discovery | OmicsTypeConfig |
| `lobster.figure_template_providers` namespace scanning | Cheap TEMPLATE_MANIFEST descriptors and drawing callables |

Do not invent a ComponentRegistry `lobster.agent_configs` group. State entry
points in agent packages and template namespace discovery are distinct from this
registry's component groups. Preserve PEP 420 namespaces so independently
installed packages can contribute modules.

`ComponentRegistry` exposes get/list/has APIs such as `get_agent`, `list_agents`,
`has_agent`, `get_adapter`, `list_adapters` and `list_providers`. Do not assume
configs are dictionaries or use an old `get_available_agents` example.

## Agent and Omics Registration

```toml
[project.entry-points."lobster.agents"]
domain_expert = "lobster.agents.domain.domain_expert:AGENT_CONFIG"

[project.entry-points."lobster.omics_types"]
domain = "my_package.config:DOMAIN_CONFIG"

[project.entry-points."lobster.adapters"]
domain_matrix = "my_package.adapter:create_domain_adapter"

[project.entry-points."lobster.providers"]
domain_database = "my_package.provider:DomainProvider"

[project.entry-points."lobster.download_services"]
domain_database = "my_package.download_service:DomainDownloadService"

[project.entry-points."lobster.queue_preparers"]
domain_database = "my_package.queue_preparer:DomainQueuePreparer"
```

The named modules/classes must exist and satisfy the actual interfaces. An adapter
factory returns a configured adapter; providers/download services/queue preparers
must honor their data-manager construction contract. Study the existing source
rather than copying a class stub with missing abstract methods.

`OmicsTypeConfig` in `lobster/core/omics_registry.py` describes name/display label,
schema class, adapter names, preferred databases, aliases, detection configuration
and QC thresholds. `DataTypeDetector` uses metadata/data heuristics; a type score
is not scientific validation of the dataset.

Keep raw-count values and exact sample identity intact during import when the
workflow requests `gene_counts`. A generic adapter's normalization or casting
must not silently turn an accepted count candidate into a different matrix.

## Figure Template Providers

Ship a uniquely named module beneath `lobster.figure_template_providers`; do not
add `__init__.py` at that shared namespace boundary. Keep heavy imports inside the
drawing callable so discovery does not load a scientific stack.

```python
from lobster.figure_templates import figure_template

TEMPLATE_MANIFEST = [
    {
        "name": "domain.scatter",
        "version": "1",
        "purpose": "Draw retained measurements without refitting",
        "callable": "draw_scatter",
    }
]


@figure_template("domain.scatter", version="1")
def draw_scatter(data, *, x, y, title="Domain measurements"):
    import pandas as pd
    import plotly.express as px

    rows = pd.DataFrame(data).copy(deep=True)
    return px.scatter(rows, x=x, y=y, title=title)
```

This is an illustrative provider, not a scientifically qualified chart. Implement
stable entity/trace identities where the chart supports individual selections;
do not manufacture IDs from row position or claim single-entity binding for
aggregate/matrix marks.

The decorator records declared template name/version and resolved data-free
arguments. Additional input arrays must be declared through `data_args=(...)`
rather than accidentally recorded as settings. Settings must be finite JSON and
fit the template recipe limit; data belongs in explicit source payloads.

Use `list_templates()`, `get_template(name)` and `show_template_source(name)` to
discover/edit drawings. Templates return editable Plotly figures and do not save,
mutate modalities or rerun statistical analyses. Custom-code callers explicitly
submit a figure; supplied source data is needed for retained regeneration input.
Template recipe metadata is a declaration, not proof of execution.

Scientific tools may declare `suggests_templates` with installed names. The
figure contract checks those names and actual tool behavior; a suggestion does
not save a plot or qualify the science.

## Compatibility and Tests

The 1.2 scaffold generates `lobster-ai~=1.2`. Keep the plugin's dependency range aligned with the engine version used for validation and execution.

Use the installed engine's canonical validator, not the superseded bundled script:

```bash
lobster validate-plugin ./lobster-domain/
uv run pytest ./lobster-domain/tests/ -m contract -v
```

Agent contracts check source/schema/metadata and real IR. Figure contracts execute
representative valid tool cases against a real DataManagerV2. Templates also need
fixture tests for input immutability, Plotly validity, explicit transformations,
style and identities where applicable. Skips and placeholders are not execution
coverage. See [scaffold.md](scaffold.md), [testing.md](testing.md) and
[aquadif-contract.md](aquadif-contract.md).

**Source:** `lobster/core/component_registry.py`, `lobster/core/omics_registry.py`,
`lobster/core/interfaces/`, `lobster/figure_templates/_api.py` and
`lobster/testing/contract_mixins.py`.
