# Phase 4: Agent Rollout - Research

**Researched:** 2026-02-28
**Domain:** Python tool metadata tagging, pytest contract tests, LangGraph agent factories
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Rollout batching:** Size-based waves grouping packages by tool count:
  - **Wave 1 (small + graph.py):** structural-viz (5 tools), metabolomics (10 tools), metadata (11 tools), graph.py delegation tools (separate plan)
  - **Wave 2 (medium):** genomics (17 tools), visualization (11 tools), ml (18 tools)
  - **Wave 3 (large):** proteomics (34 tools), research (21 tools), drug-discovery (35 tools)
- Parallel execution within each wave (packages are independent — no cross-package dependencies)
- Sequential across waves: Wave 1 → verify → Wave 2 → verify → Wave 3 → verify
- Contract tests gate each wave — all contract tests + existing tests must pass before proceeding
- One atomic commit per package (e.g., "feat(04): add AQUADIF metadata to genomics package")

- **Category mapping authority:** Trust Claude fully for all ~155 tool categorizations — no per-tool user review. Contract tests enforce correctness.

- **Category philosophy:** "Single category preferred" applies uniformly. 80% rule. Multi-category ratio (<40%) enforced globally across all ~190 tools, not per-package.

- **Drug discovery coverage:** Include lobster-drug-discovery (35 tools) in Phase 4 in Wave 3. Add ROLL-10 requirement to REQUIREMENTS.md.

- **Graph.py delegation tools:** Separate plan in Wave 1. Add `.metadata` and `.tags` inside `_create_lazy_delegation_tool` factory function — single source of truth, all delegation tools auto-tagged.

### Claude's Discretion

- Whether to document rationale for ambiguous categorizations in plan summaries (per-package mapping tables optional)
- Delegation tool provenance flag (True/False) — determine based on AQUADIF contract definition for DELEGATE
- Delegation tool testing approach — dedicated contract test vs smoke test in graph.py tests
- Technical implementation details within each package (helper patterns, code organization)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ROLL-01 | `genomics_expert` (11 tools) and `variant_analysis_expert` (6 tools) have metadata and pass contract tests | Wave 2 plan; 17 actual tools verified in codebase |
| ROLL-02 | `proteomics_expert` (3 tools), `proteomics_de_analysis_expert` (7 tools), and `biomarker_discovery_expert` (7 tools) have metadata and pass contract tests; shared_tools.py (17 tools) also tagged | Wave 3 plan; 34 actual tools verified |
| ROLL-03 | `metabolomics_expert` (10 tools) has metadata and passes contract tests | Wave 1 plan; 10 actual tools in shared_tools.py |
| ROLL-04 | `annotation_expert` (12 tools) and `de_analysis_expert` (15 tools) in transcriptomics package have metadata and pass contract tests | Wave 1 plan (same package as Phase 3 reference); 27 actual tools verified |
| ROLL-05 | `machine_learning_expert` (7 tools), `feature_selection_expert` (1 tool), and `survival_analysis_expert` (3 tools) have metadata; shared_tools.py (7 tools) also tagged | Wave 2 plan; 18 actual tools verified |
| ROLL-06 | `research_agent` (11 tools) and `data_expert` (9 tools) have metadata and pass contract tests | Wave 3 plan; 20 actual tools verified |
| ROLL-07 | `visualization_expert` (11 tools), `metadata_assistant` (11 tools), and `protein_structure_visualization_expert` (5 tools) have metadata and pass contract tests | Wave 1 plan for structural-viz/metadata; Wave 2 for visualization |
| ROLL-08 | Dynamic DELEGATE tools in `graph.py` (`_create_lazy_delegation_tool`) have DELEGATE metadata | Wave 1 separate plan; add before `return invoke_agent_lazy` on line 277 |
| ROLL-09 | All ~200+ tools across all agents pass contract tests; multi-category usage is <40% | Phase gate check after Wave 3 |
| ROLL-10 (new) | `drug_discovery_expert` (10 tools), `cheminformatics_expert` (9 tools), `clinical_dev_expert` (8 tools), and `pharmacogenomics_expert` (8 tools) have metadata and pass contract tests | Wave 3 plan; 35 actual tools verified |
</phase_requirements>

---

## Summary

Phase 4 is a systematic metadata tagging rollout across 9 remaining agent packages plus `graph.py`. The infrastructure is fully built (AquadifCategory enum, PROVENANCE_REQUIRED set, AgentContractTestMixin with 14 test methods, migration guide at `skills/lobster-dev/references/aquadif-migration.md`). The work is mechanical but precise: for each tool, apply a 2-line pattern after the `@tool` closure, set `.metadata` and `.tags` to matching category lists, then verify with contract tests.

The primary research finding is that the codebase has **188 domain tools + 12 delegation tools = 200 tools** remaining in Phase 4 (the REQUIREMENTS.md tool count estimates are slightly off from actuals). The wave batching in CONTEXT.md is sound — ROLL-04 (annotation/de_analysis) should join Wave 1 since they live in the already-familiar `lobster-transcriptomics` package. ROLL-07 splits across Wave 1 (structural-viz, metadata) and Wave 2 (visualization).

The key technical risk is the AST provenance validation in `test_provenance_ast_validation` — every tool declaring `provenance: True` must contain a `log_tool_usage(ir=ir)` call in its body. The transcriptomics pattern proves this works, but agents like `metadata_assistant`, `research_agent`, and `data_expert` may have tools that use different data manager patterns. Category ambiguity is low-risk because the Category Decision Quick Reference table plus Phase 3 decision precedents cover most cases.

**Primary recommendation:** Follow the 7-step migration checklist from `aquadif-migration.md` for every package. The reference implementation (transcriptomics) and the migration guide eliminate all ambiguity about the pattern. Spend most planning effort on producing accurate per-package categorization tables — these are the inputs to the executor.

---

## Standard Stack

### Core (already present — no installation needed)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `AquadifCategory` enum | `lobster/config/aquadif.py` | 10 categories + string comparison via `(str, Enum)` | Verified in codebase |
| `PROVENANCE_REQUIRED` | `lobster/config/aquadif.py` | 7-category set requiring `log_tool_usage(ir=ir)` | Verified in codebase |
| `has_provenance_call()` | `lobster/config/aquadif.py` | Standalone AST helper for provenance validation | Verified in codebase |
| `AgentContractTestMixin` | `lobster/testing/contract_mixins.py` | 14 test methods, LLM mock, PregelNode traversal, cached | Verified in codebase |
| Migration guide | `skills/lobster-dev/references/aquadif-migration.md` | 7-step checklist, patterns, worked example | Verified 302-line doc |

### Supporting Tools

| Tool | Version | Purpose |
|------|---------|---------|
| `pytest -m contract` | pytest 7+ | Runs `@pytest.mark.contract` tests per package |
| `ast.parse()` + `ast.walk()` | stdlib | AST-based provenance call validation |
| `unittest.mock.patch` | stdlib | LLM factory mocking in contract tests |

### No New Dependencies Required

Phase 4 adds no new libraries. All tools, test infrastructure, and patterns are already in place.

---

## Architecture Patterns

### Recommended Package Structure for Contract Tests

Every package needs a `tests/agents/` directory with one test file:

```
packages/lobster-<domain>/
├── lobster/
│   └── agents/<domain>/
│       ├── <domain>_expert.py      # @tool + .metadata + .tags
│       ├── shared_tools.py         # @tool + .metadata + .tags (if exists)
│       └── <child>_expert.py       # @tool + .metadata + .tags
└── tests/
    ├── agents/
    │   ├── __init__.py             # Empty file (must create)
    │   └── test_aquadif_<domain>.py  # One class per agent
    └── services/                   # Existing tests (do not touch)
```

Packages that already have `tests/agents/`:
- `lobster-transcriptomics` (has contract tests — Phase 3 complete)
- `lobster-proteomics` (has `test_proteomics_integration.py` — NOT aquadif yet)
- `lobster-metadata` (has tests — NOT aquadif yet)
- `lobster-structural-viz` (has tests — NOT aquadif yet)

Packages that need `tests/agents/` created:
- `lobster-genomics`, `lobster-metabolomics`, `lobster-ml`, `lobster-research`, `lobster-visualization`, `lobster-drug-discovery`

### Pattern 1: Metadata Assignment (post-decorator inline)

The established pattern from Phase 3 — always after the `@tool` closure, before the next `@tool`:

```python
@tool
def load_vcf(modality_name: str, vcf_path: str) -> str:
    """Load VCF file into genomics workspace."""
    adata, stats, ir = vcf_service.load(vcf_path)
    data_manager.log_tool_usage("load_vcf", {"vcf_path": vcf_path}, stats, ir=ir)
    return f"Loaded {stats['n_variants']} variants"

load_vcf.metadata = {"categories": ["IMPORT"], "provenance": True}
load_vcf.tags = ["IMPORT"]


@tool
def assess_quality(modality_name: str) -> str:
    ...
```

**Rules (HIGH confidence, verified against Phase 3 implementation):**
- Use string literals (`"IMPORT"`) not enum (`AquadifCategory.IMPORT`) — no import needed in tool files
- Both `.metadata` and `.tags` MUST be set to identical category lists
- `.metadata` dict is unique per tool (contract enforces this — no shared dicts)
- If `provenance: True`, the tool body MUST call `data_manager.log_tool_usage(..., ir=ir)`

### Pattern 2: Contract Test Class

One class per agent, inheriting `AgentContractTestMixin`:

```python
# packages/lobster-genomics/tests/agents/test_aquadif_genomics.py
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifGenomicsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for genomics_expert."""

    agent_module = "lobster.agents.genomics.genomics_expert"
    factory_name = "genomics_expert"
    is_parent_agent = True  # Has child: variant_analysis_expert


@pytest.mark.contract
class TestAquadifVariantAnalysisExpert(AgentContractTestMixin):
    """AQUADIF contract tests for variant_analysis_expert."""

    agent_module = "lobster.agents.genomics.variant_analysis_expert"
    factory_name = "variant_analysis_expert"
    is_parent_agent = False  # Child agent — no IMPORT/QUALITY requirement
```

The mixin provides all 14 test methods automatically. Key parameters:
- `agent_module`: module containing `AGENT_CONFIG` (entry point module)
- `factory_name`: factory function name
- `is_parent_agent`: `True` if agent has `child_agents` in its `AGENT_CONFIG`
- `factory_module`: only needed if factory is in a different module than `AGENT_CONFIG`

### Pattern 3: Graph.py Delegation Tool Tagging

The `_create_lazy_delegation_tool` function in `lobster/agents/graph.py` (line 198) creates all DELEGATE tools. Add metadata before the return:

```python
# In _create_lazy_delegation_tool(), before line 277 (return invoke_agent_lazy)
invoke_agent_lazy.metadata = {"categories": ["DELEGATE"], "provenance": False}
invoke_agent_lazy.tags = ["DELEGATE"]

return invoke_agent_lazy
```

This tags all 12 delegation tools at creation time automatically — single source of truth.

**Provenance flag for DELEGATE:** `False`. DELEGATE tools hand off work to child agents; they do not transform data or produce scientific results themselves. The child agent is responsible for its own provenance tracking.

### Pattern 4: Shared Tools Files

For packages that use `create_shared_tools()` / `create_<domain>_tools()` factory functions:

```python
# shared_tools.py (e.g., metabolomics, proteomics)
def create_shared_tools(data_manager, ...) -> List:

    @tool
    def assess_metabolomics_quality(modality_name: str) -> str:
        ...
        data_manager.log_tool_usage("assess_metabolomics_quality", ..., ir=ir)
        return result

    assess_metabolomics_quality.metadata = {"categories": ["QUALITY"], "provenance": True}
    assess_metabolomics_quality.tags = ["QUALITY"]

    return [assess_metabolomics_quality, ...]
```

The metadata dict uniqueness is guaranteed because each `@tool` creates a new closure — no shared objects.

### Anti-Patterns to Avoid

- **Metadata before @tool:** The `@tool` decorator wraps the function and creates a new object — metadata set before decoration is lost.
- **Missing .tags:** LangChain callbacks receive `.tags` but not `.metadata`. Both must be set.
- **Provenance mismatch:** If `categories[0]` is in `PROVENANCE_REQUIRED` but `provenance: False`, `test_provenance_tools_have_flag` fails.
- **AST violation:** If `provenance: True` but no `log_tool_usage(ir=ir)` call exists, `test_provenance_ast_validation` fails.
- **Importing AquadifCategory in tool files:** Unnecessary coupling. Use string literals.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Validating category strings | Custom regex/list check | `AquadifCategory(cat)` raises `ValueError` on invalid | Already implemented |
| AST provenance check | Custom source scan | `has_provenance_call()` in `aquadif.py` | Already handles `ast.walk`, method attributes |
| Contract test logic | Custom pytest fixtures | `AgentContractTestMixin` | 14 tests, LLM mock, PregelNode traversal, caching all included |
| LLM mocking in tests | Custom fixtures | Mixin's dual-site `patch("lobster.config.llm_factory.create_llm")` | Handles both import patterns |
| Tool extraction | Manual graph inspection | Mixin's `_get_tools_from_factory()` | Handles PregelNode wrapping |

---

## Common Pitfalls

### Pitfall 1: Provenance flag vs. AST mismatch

**What goes wrong:** A tool is categorized as IMPORT/ANALYZE/etc. (provenance required) but the implementation doesn't call `log_tool_usage(ir=ir)`. Contract tests fail on `test_provenance_ast_validation`.

**Why it happens:** Some tools in packages like `metadata_assistant`, `research_agent`, and `data_expert` may use different data manager patterns or may be UTILITY-category tools (status/listing) that don't produce scientific results.

**How to avoid:** Before assigning a provenance-required category, scan the tool body for `log_tool_usage(ir=ir)`. If it's absent, reconsider: is this actually a UTILITY tool (read-only, status)?

**Warning signs:** Tool functions with names like `check_*`, `get_*`, `list_*`, `show_*` are likely UTILITY (no provenance). Functions with names like `load_*`, `import_*`, `run_*`, `analyze_*`, `filter_*` likely require provenance.

### Pitfall 2: is_parent_agent flag misconfiguration

**What goes wrong:** Setting `is_parent_agent = False` for an agent that has `child_agents` in its config, causing `test_minimum_viable_parent` to skip. Or setting `True` for a child agent, causing spurious MVP check failures.

**Why it happens:** There are 7 parent agents in this codebase; it's easy to miss one.

**How to avoid:** Cross-reference `child_agents` in each agent's `AGENT_CONFIG`:
- `genomics_expert`: parent (child: `variant_analysis_expert`)
- `proteomics_expert`: parent (children: `proteomics_de_analysis_expert`, `biomarker_discovery_expert`)
- `transcriptomics_expert`: parent — already done in Phase 3
- `machine_learning_expert`: parent (children: `feature_selection_expert`, `survival_analysis_expert`)
- `research_agent`: parent (child: `metadata_assistant`)
- `data_expert`: parent (child: `metadata_assistant`)
- `drug_discovery_expert`: parent (children: `cheminformatics_expert`, `clinical_dev_expert`, `pharmacogenomics_expert`)
- All other agents: child/leaf (is_parent_agent = False)

### Pitfall 3: Missing tests/agents/ directory

**What goes wrong:** Creating the test file without the directory and `__init__.py`. pytest can't discover the tests.

**Why it happens:** 6 of 9 packages don't have `tests/agents/` yet.

**How to avoid:** Create both `tests/agents/` directory and `tests/agents/__init__.py` (empty) before writing the test file. See the transcriptomics reference at `packages/lobster-transcriptomics/tests/agents/__init__.py`.

### Pitfall 4: Module path for agents with non-standard structures

**What goes wrong:** Using wrong `agent_module` in test class leads to `AGENT_CONFIG` not found or wrong factory.

**Why it happens:** Some packages have flat layout (visualization, structural-viz, metadata) while others have nested layout.

**How to avoid:** Use the entry point module paths from each package's `pyproject.toml`. They're authoritative:

| Agent | agent_module | factory_name |
|-------|-------------|--------------|
| genomics_expert | `lobster.agents.genomics.genomics_expert` | `genomics_expert` |
| variant_analysis_expert | `lobster.agents.genomics.variant_analysis_expert` | `variant_analysis_expert` |
| proteomics_expert | `lobster.agents.proteomics.proteomics_expert` | `proteomics_expert` |
| proteomics_de_analysis_expert | `lobster.agents.proteomics.de_analysis_expert` | `de_analysis_expert` |
| biomarker_discovery_expert | `lobster.agents.proteomics.biomarker_discovery_expert` | `biomarker_discovery_expert` |
| metabolomics_expert | `lobster.agents.metabolomics.metabolomics_expert` | `metabolomics_expert` |
| annotation_expert | `lobster.agents.transcriptomics.annotation_expert` | `annotation_expert` |
| de_analysis_expert (transcriptomics) | `lobster.agents.transcriptomics.de_analysis_expert` | `de_analysis_expert` |
| machine_learning_expert | `lobster.agents.machine_learning.machine_learning_expert` | `machine_learning_expert` |
| feature_selection_expert | `lobster.agents.machine_learning.feature_selection_expert` | `feature_selection_expert` |
| survival_analysis_expert | `lobster.agents.machine_learning.survival_analysis_expert` | `survival_analysis_expert` |
| research_agent | `lobster.agents.research.research_agent` | `research_agent` |
| data_expert | `lobster.agents.data_expert.data_expert` | `data_expert` |
| visualization_expert | `lobster.agents.visualization_expert` | `visualization_expert` |
| metadata_assistant | `lobster.agents.metadata_assistant.metadata_assistant` | `metadata_assistant` |
| protein_structure_visualization_expert | `lobster.agents.protein_structure_visualization_expert` | `protein_structure_visualization_expert` |
| drug_discovery_expert | `lobster.agents.drug_discovery.drug_discovery_expert` | `drug_discovery_expert` |
| cheminformatics_expert | `lobster.agents.drug_discovery.cheminformatics_expert` | `cheminformatics_expert` |
| clinical_dev_expert | `lobster.agents.drug_discovery.clinical_dev_expert` | `clinical_dev_expert` |
| pharmacogenomics_expert | `lobster.agents.drug_discovery.pharmacogenomics_expert` | `pharmacogenomics_expert` |

### Pitfall 5: Metadata assistant module path mismatch

**What goes wrong:** Entry point says `lobster.agents.metadata_assistant` (no sub-module) but the actual AGENT_CONFIG is in `lobster.agents.metadata_assistant.metadata_assistant.py`.

**Why it happens:** The metadata package uses a non-standard layout with a `metadata_assistant/` subdirectory but the entry point references the package-level `lobster.agents.metadata_assistant` (the `__init__.py`).

**How to avoid:** The pyproject.toml entry point says `metadata_assistant = "lobster.agents.metadata_assistant:AGENT_CONFIG"`. This means AGENT_CONFIG is in the `__init__.py` of `lobster/agents/metadata_assistant/`. The factory, however, may be in `metadata_assistant.py`. Use `agent_module = "lobster.agents.metadata_assistant.metadata_assistant"` for both (factory is in that file).

### Pitfall 6: Proteomics shared_tools.py belongs to all 3 agents

**What goes wrong:** Tagging shared_tools.py tools but then the contract tests for `proteomics_de_analysis_expert` and `biomarker_discovery_expert` fail because they don't receive the shared tools.

**Why it happens:** `shared_tools.py` tools are injected into all 3 proteomics agents via `create_shared_tools()`. Each agent factory calls this function. The contract test for each agent will see the shared tools.

**How to avoid:** Tag the tools in `shared_tools.py` once. All 3 contract test classes will pick them up automatically via the factory. This is the same pattern as transcriptomics `shared_tools.py`.

### Pitfall 7: ROLL-04 placement in wave plan

**What goes wrong:** ROLL-04 (annotation_expert + de_analysis_expert) is in the transcriptomics package but wasn't done in Phase 3. The CONTEXT.md wave plan doesn't explicitly list ROLL-04 — it needs to join Wave 1 since the transcriptomics package is already the reference.

**How to avoid:** Plan ROLL-04 as part of Wave 1 alongside the other small/reference packages.

---

## Code Examples

### Complete Working Reference: Transcriptomics (Phase 3)

Source: `packages/lobster-transcriptomics/lobster/agents/transcriptomics/transcriptomics_expert.py`

```python
@tool
def cluster_cells(
    modality_name: str,
    resolution: float = 0.5,
    method: str = "leiden",
) -> str:
    """Cluster cells using graph-based community detection."""
    # ... implementation ...
    data_manager.log_tool_usage("cluster_cells", params, stats, ir=ir)
    return result

cluster_cells.metadata = {"categories": ["ANALYZE"], "provenance": True}
cluster_cells.tags = ["ANALYZE"]


@tool
def check_data_status() -> str:
    """Show loaded datasets and current analysis state."""
    return data_manager.get_status_summary()

check_data_status.metadata = {"categories": ["UTILITY"], "provenance": False}
check_data_status.tags = ["UTILITY"]
```

Source: `packages/lobster-transcriptomics/tests/agents/test_aquadif_transcriptomics.py`

```python
@pytest.mark.contract
class TestTranscriptomicsExpertAquadif(AgentContractTestMixin):
    agent_module = "lobster.agents.transcriptomics.transcriptomics_expert"
    factory_name = "transcriptomics_expert"
    is_parent_agent = True


@pytest.mark.contract
class TestAnnotationExpertAquadif(AgentContractTestMixin):
    agent_module = "lobster.agents.transcriptomics.annotation_expert"
    factory_name = "annotation_expert"
    is_parent_agent = False
```

### Graph.py Delegation Tool (to implement)

Source: `lobster/agents/graph.py` line 222–277

```python
@tool(f"handoff_to_{_name}", description=f"Delegate task to {_name}. {_desc}")
def invoke_agent_lazy(task_description: str) -> str:
    # ... invocation logic ...
    return content

# ADD THESE TWO LINES before return:
invoke_agent_lazy.metadata = {"categories": ["DELEGATE"], "provenance": False}
invoke_agent_lazy.tags = ["DELEGATE"]

return invoke_agent_lazy
```

### Running Contract Tests Per Package

```bash
# From package directory
cd packages/lobster-<domain>
pytest -m contract tests/agents/test_aquadif_<domain>.py -v

# Quick all-contract from repo root (only finds root tests/)
python -m pytest -m contract --no-cov -v

# Run per-package after tagging
cd packages/lobster-genomics
pytest -m contract tests/agents/test_aquadif_genomics.py -v
pytest tests/ -v  # existing tests — verify no regressions
```

---

## Accurate Tool Count Inventory

Verified from codebase on 2026-02-28. REQUIREMENTS.md estimates are slightly off.

### Phase 4 Scope: 200 Tools Total

| Wave | Package | Agents | Tool Files | Actual Tools | ROLL req |
|------|---------|--------|------------|--------------|----------|
| Wave 1 | lobster-transcriptomics (children) | annotation_expert, de_analysis_expert | annotation_expert.py (12), de_analysis_expert.py (15) | **27** | ROLL-04 |
| Wave 1 | lobster-structural-viz | protein_structure_visualization_expert | protein_structure_visualization_expert.py (5) | **5** | ROLL-07 |
| Wave 1 | lobster-metabolomics | metabolomics_expert | shared_tools.py (10) | **10** | ROLL-03 |
| Wave 1 | lobster-metadata | metadata_assistant | metadata_assistant.py (11) | **11** | ROLL-07 |
| Wave 1 | lobster-ai (core) | graph.py delegation | graph.py factory | **12** | ROLL-08 |
| Wave 2 | lobster-genomics | genomics_expert, variant_analysis_expert | genomics_expert.py (11), variant_analysis_expert.py (6) | **17** | ROLL-01 |
| Wave 2 | lobster-visualization | visualization_expert | visualization_expert.py (11) | **11** | ROLL-07 |
| Wave 2 | lobster-ml | machine_learning_expert, feature_selection_expert, survival_analysis_expert | machine_learning_expert.py (7), shared_tools.py (7), feature_selection_expert.py (1), survival_analysis_expert.py (3) | **18** | ROLL-05 |
| Wave 3 | lobster-proteomics | proteomics_expert, proteomics_de_analysis_expert, biomarker_discovery_expert | shared_tools.py (17), proteomics_expert.py (3), de_analysis_expert.py (7), biomarker_discovery_expert.py (7) | **34** | ROLL-02 |
| Wave 3 | lobster-research | research_agent, data_expert | research_agent.py (11), data_expert.py (9) | **20** | ROLL-06 |
| Wave 3 | lobster-drug-discovery | drug_discovery_expert, cheminformatics_expert, clinical_dev_expert, pharmacogenomics_expert | shared_tools.py (10), cheminformatics_tools.py (9), clinical_tools.py (8), pharmacogenomics_tools.py (8) | **35** | ROLL-10 |
| **TOTAL** | | | | **200** | |

Phase 3 already completed 22 tools (transcriptomics_expert + shared_tools). Grand total after Phase 4: **222 tools**.

### Delegation Tools by Parent-Child Relationship

There are 12 delegation tools created by `_create_lazy_delegation_tool`:

| Parent | Child(ren) | Delegation Count |
|--------|-----------|-----------------|
| genomics_expert | variant_analysis_expert | 1 |
| proteomics_expert | proteomics_de_analysis_expert, biomarker_discovery_expert | 2 |
| transcriptomics_expert | annotation_expert, de_analysis_expert | 2 |
| machine_learning_expert | feature_selection_expert, survival_analysis_expert | 2 |
| research_agent | metadata_assistant | 1 |
| data_expert | metadata_assistant | 1 |
| drug_discovery_expert | cheminformatics_expert, clinical_dev_expert, pharmacogenomics_expert | 3 |
| **Total** | | **12** |

---

## Category Guidance by Domain

Pre-mapped categorization guidance for each package. Use the 80% rule and Category Decision Quick Reference from `aquadif-migration.md` as definitive authority.

### Genomics

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `load_vcf`, `load_plink` | IMPORT | True |
| `assess_quality` | QUALITY | True |
| `filter_samples`, `filter_variants`, `ld_prune` | FILTER | True |
| `run_gwas`, `calculate_pca` | ANALYZE | True |
| `annotate_variants` | ANNOTATE | True |
| `compute_kinship`, `clump_results` | ANALYZE | True |
| `normalize_variants` | PREPROCESS | True |
| `predict_consequences`, `query_population_frequencies`, `query_clinical_databases`, `prioritize_variants`, `lookup_variant` | ANNOTATE | True |

### Proteomics

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `import_proteomics_data`, `import_ptm_sites`, `import_affinity_data` | IMPORT | True |
| `check_proteomics_status`, `assess_proteomics_quality`, `assess_lod_quality`, `add_peptide_mapping`, `validate_antibody_specificity` | QUALITY/UTILITY | True/False |
| `filter_proteomics_data`, `select_variable_proteins` | FILTER | True |
| `normalize_proteomics_data`, `impute_missing_values`, `correct_batch_effects`, `summarize_peptide_to_protein`, `normalize_ptm_to_protein`, `normalize_bridge_samples`, `correct_plate_effects` | PREPROCESS | True |
| `analyze_proteomics_patterns`, `assess_cross_platform_concordance` | ANALYZE | True |
| `create_proteomics_summary` | UTILITY | False |
| `find_differential_proteins`, `run_time_course_analysis`, `run_correlation_analysis` | ANALYZE | True |
| `run_pathway_enrichment`, `run_differential_ptm_analysis`, `run_kinase_enrichment`, `run_string_network_analysis` | ANALYZE | True |
| `identify_coexpression_modules`, `correlate_modules_with_traits`, `perform_survival_analysis`, `find_survival_biomarkers`, `select_biomarker_panel`, `evaluate_biomarker_panel`, `extract_hub_proteins` | ANALYZE | True |

### Metabolomics

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `assess_metabolomics_quality` | QUALITY | True |
| `filter_metabolomics_features` | FILTER | True |
| `handle_missing_values`, `normalize_metabolomics`, `correct_batch_effects` | PREPROCESS | True |
| `run_metabolomics_statistics`, `run_multivariate_analysis`, `analyze_lipid_classes` | ANALYZE | True |
| `annotate_metabolites` | ANNOTATE | True |
| `run_pathway_enrichment` | ANALYZE | True |

### Machine Learning

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `check_ml_ready_modalities` | QUALITY | True |
| `prepare_ml_features`, `create_ml_splits` | PREPROCESS | True |
| `export_for_ml_framework` | UTILITY | False (exports, not analysis) |
| `create_ml_analysis_summary` | UTILITY | False |
| `check_scvi_availability` | UTILITY | False |
| `train_scvi_embedding` | ANALYZE | True |
| `run_stability_selection`, `run_lasso_selection` | ANALYZE | True (feature selection extracts patterns) |
| `run_variance_filter` | FILTER | True |
| `enrich_pathways_for_selected_features` | ANALYZE | True |
| `train_cox_model`, `optimize_risk_threshold`, `run_kaplan_meier` | ANALYZE | True |
| `get_feature_selection_results`, `get_hazard_ratios` | UTILITY | False (read-only retrieval) |
| `check_survival_data`, `check_survival_availability` | QUALITY/UTILITY | True/False |

### Research / Data Expert

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `search_literature`, `find_related_entries`, `fast_dataset_search`, `fast_abstract_search` | UTILITY | False (online search, no data modification) |
| `get_dataset_metadata`, `validate_dataset_metadata` | QUALITY | True |
| `prepare_dataset_download` | UTILITY | False (prepares queue, doesn't load data) |
| `extract_methods`, `read_full_publication` | UTILITY | False (content retrieval) |
| `process_publication_entry`, `process_publication_queue` | PREPROCESS | True |
| `execute_download_from_queue` | IMPORT | True (loads data into workspace) |
| `get_modality_details`, `get_adapter_info`, `get_queue_status` | UTILITY | False (status/listing) |
| `remove_modality` | UTILITY | False (workspace management) |
| `validate_modality_compatibility` | QUALITY | True |
| `load_modality` | IMPORT | True |
| `create_mudata_from_modalities`, `concatenate_samples` | PREPROCESS | True |

### Visualization

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `check_visualization_readiness` | QUALITY | True |
| `create_umap_plot`, `create_qc_plots`, `create_violin_plot`, `create_feature_plot`, `create_dot_plot`, `create_heatmap`, `create_elbow_plot`, `create_cluster_composition_plot` | ANALYZE | True (produces analytical output) |
| `get_visualization_history` | UTILITY | False |
| `report_visualization_complete` | UTILITY | False |

### Metadata

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| Most metadata tools involve ANNOTATE (ID mapping, metadata validation, queue filtering) | ANNOTATE/UTILITY | True/False |
| Validation tools | QUALITY | True |
| Queue management tools | UTILITY | False |

### Structural Viz

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| Structure fetching/loading | IMPORT | True |
| Structure visualization | ANALYZE | True |
| Structure comparison | ANALYZE | True |

### Drug Discovery

| Tool Pattern | Category | Provenance |
|-------------|----------|-----------|
| `search_drug_targets`, `search_compounds` | UTILITY | False (external search) |
| `score_drug_target`, `rank_targets`, `get_compound_bioactivity`, `get_compound_properties` | ANALYZE | True |
| `calculate_descriptors`, `lipinski_check`, `fingerprint_similarity`, `predict_admet` | ANALYZE | True |
| `predict_mutation_effect`, `extract_protein_embedding`, `compare_variant_sequences` | ANALYZE | True |
| `get_drug_indications`, `get_variant_drug_interactions`, `get_pharmacogenomic_evidence` | ANNOTATE | True |
| `check_drug_discovery_status`, `list_available_databases` | UTILITY | False |

**Note:** Category guidance above is HIGH confidence for obvious cases. Boundary cases (e.g., `search_*` functions that could be UTILITY or ANNOTATE) should be resolved using the 80% rule during execution. The Phase 3 precedents (especially `merge_sample_metadata → ANNOTATE`, `convert_gene_identifiers → ANNOTATE`) are the model for ambiguous cases.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7+ |
| Marker | `@pytest.mark.contract` |
| Contract marker defined | `pyproject.toml` + `pytest.ini` (both registered) |
| Per-package run command | `cd packages/lobster-<domain> && pytest -m contract tests/agents/test_aquadif_<domain>.py -v` |
| Root run (core tests only) | `python -m pytest -m contract --no-cov -v` (finds only root `tests/`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| ROLL-01 | genomics tools have metadata + contract passes | contract | `cd packages/lobster-genomics && pytest -m contract tests/agents/test_aquadif_genomics.py -v` |
| ROLL-02 | proteomics tools have metadata + contract passes | contract | `cd packages/lobster-proteomics && pytest -m contract tests/agents/test_aquadif_proteomics.py -v` |
| ROLL-03 | metabolomics tools have metadata + contract passes | contract | `cd packages/lobster-metabolomics && pytest -m contract tests/agents/test_aquadif_metabolomics.py -v` |
| ROLL-04 | transcriptomics children have metadata + contract passes | contract | `cd packages/lobster-transcriptomics && pytest -m contract tests/agents/test_aquadif_transcriptomics.py -v` |
| ROLL-05 | ml tools have metadata + contract passes | contract | `cd packages/lobster-ml && pytest -m contract tests/agents/test_aquadif_ml.py -v` |
| ROLL-06 | research tools have metadata + contract passes | contract | `cd packages/lobster-research && pytest -m contract tests/agents/test_aquadif_research.py -v` |
| ROLL-07 | viz/meta/structural tools have metadata + contract passes | contract | Per-package in respective directories |
| ROLL-08 | delegation tools have DELEGATE metadata | contract (smoke) | Verified via parent agent contract tests (delegation tools appear in parent's tool list) |
| ROLL-09 | All tools pass globally; multi-cat <40% | contract + manual | `pytest -m contract` from all package dirs + count multi-cat ratio |
| ROLL-10 | drug-discovery tools have metadata + contract passes | contract | `cd packages/lobster-drug-discovery && pytest -m contract tests/agents/test_aquadif_drug_discovery.py -v` |

### Backward Compatibility Gate

Every wave must also run existing tests:

```bash
# After tagging each package
cd packages/lobster-<domain>
pytest tests/ -v  # All existing tests must pass
```

### Wave Gate Commands

```bash
# Wave 1 gate (run from each Wave 1 package dir)
cd packages/lobster-transcriptomics && pytest tests/ -v
cd packages/lobster-structural-viz && pytest tests/ -v
cd packages/lobster-metabolomics && pytest tests/ -v
cd packages/lobster-metadata && pytest tests/ -v
# graph.py: run root tests
cd /path/to/lobster && pytest tests/unit/agents/ -v

# Wave 2 gate
cd packages/lobster-genomics && pytest tests/ -v
cd packages/lobster-visualization && pytest tests/ -v
cd packages/lobster-ml && pytest tests/ -v

# Wave 3 gate
cd packages/lobster-proteomics && pytest tests/ -v
cd packages/lobster-research && pytest tests/ -v
cd packages/lobster-drug-discovery && pytest tests/ -v
```

### Wave 0 Gaps (Files Needing Creation)

The following `tests/agents/` directories and files must be created before the contract tests can run:

- [ ] `packages/lobster-genomics/tests/agents/__init__.py` + `test_aquadif_genomics.py` — covers ROLL-01
- [ ] `packages/lobster-metabolomics/tests/agents/__init__.py` + `test_aquadif_metabolomics.py` — covers ROLL-03
- [ ] `packages/lobster-ml/tests/agents/__init__.py` + `test_aquadif_ml.py` — covers ROLL-05
- [ ] `packages/lobster-research/tests/agents/__init__.py` + `test_aquadif_research.py` — covers ROLL-06
- [ ] `packages/lobster-visualization/tests/agents/__init__.py` + `test_aquadif_visualization.py` — covers ROLL-07 (viz)
- [ ] `packages/lobster-drug-discovery/tests/agents/__init__.py` + `test_aquadif_drug_discovery.py` — covers ROLL-10

Already have `tests/agents/` but need AQUADIF contract test files:

- [ ] `packages/lobster-proteomics/tests/agents/test_aquadif_proteomics.py` — covers ROLL-02 (directory exists)
- [ ] `packages/lobster-metadata/tests/agents/test_aquadif_metadata.py` — covers ROLL-07 (directory exists)
- [ ] `packages/lobster-structural-viz/tests/agents/test_aquadif_structural_viz.py` — covers ROLL-07 (directory exists)
- [ ] `packages/lobster-transcriptomics/tests/agents/test_aquadif_transcriptomics.py` — needs 2 new classes for annotation_expert and de_analysis_expert added to existing file (covers ROLL-04)

No new framework install needed — pytest is already configured.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded tool lists in agent prompts | AQUADIF `.metadata` + `.tags` on each tool | Phase 3 (2026-03-01) | Introspectable, enforceable, teachable |
| Manual provenance tracking validation | AST-based `has_provenance_call()` in contract tests | Phase 2 (2026-02-28) | Catches metadata-runtime disconnect at test time |
| No category validation | `test_categories_are_valid` in mixin | Phase 2 (2026-02-28) | Invalid category strings fail immediately |
| Two-pass agent creation (for delegation tools) | `_create_lazy_delegation_tool` (lazy resolution via dict reference) | Prior to Phase 4 | Enables single-pass creation; delegation tools resolved at invocation |

---

## Open Questions

1. **ROLL-04 wave assignment**
   - What we know: CONTEXT.md Wave 1 lists structural-viz, metabolomics, metadata, and graph.py. ROLL-04 (annotation/de_analysis) is not explicitly mentioned in the wave plan.
   - What's unclear: Should ROLL-04 be added to Wave 1 (same package as Phase 3 reference) or treated as a separate small plan?
   - Recommendation: Include ROLL-04 in Wave 1. It's in the transcriptomics package which is already established, has the same tooling, and both agents are child agents (simpler — no `is_parent_agent = True` check). The existing test file just needs 2 new classes.

2. **Metadata assistant module path ambiguity**
   - What we know: Entry point is `lobster.agents.metadata_assistant:AGENT_CONFIG` (not `lobster.agents.metadata_assistant.metadata_assistant`). The `metadata_assistant/` subdirectory exists with `metadata_assistant.py` inside.
   - What's unclear: Does `AGENT_CONFIG` live in `__init__.py` or in `metadata_assistant.py`?
   - Recommendation: Check `lobster/agents/metadata_assistant/__init__.py` during execution to confirm. Use the entry point module path (`lobster.agents.metadata_assistant`) as `agent_module`.

3. **Drug discovery tests directory**
   - What we know: `packages/lobster-drug-discovery/` has no `tests/` directory at all.
   - What's unclear: Does this package have any existing tests to protect?
   - Recommendation: Create `tests/`, `tests/__init__.py`, `tests/agents/`, `tests/agents/__init__.py`, and `test_aquadif_drug_discovery.py`. No backward compatibility risk since there are no existing tests.

4. **Delegation tool testing approach (Claude's discretion)**
   - What we know: 12 delegation tools created by `_create_lazy_delegation_tool`. Tagging is straightforward (2 lines before `return`).
   - Recommendation: A dedicated graph test is unnecessary. The delegation tools appear in each parent agent's tool list (via `delegation_tools` parameter), so the parent agent contract tests will automatically validate delegation tool metadata via `test_tools_have_aquadif_metadata`. This means ROLL-08 is validated indirectly by ROLL-01, ROLL-02, ROLL-04, ROLL-05, ROLL-06, and ROLL-10 parent tests. No separate test needed.

---

## Sources

### Primary (HIGH confidence)

- Codebase: `lobster/config/aquadif.py` — verified AquadifCategory enum, PROVENANCE_REQUIRED, has_provenance_call()
- Codebase: `lobster/testing/contract_mixins.py` — verified 14 test methods, LLM mock pattern, PregelNode traversal
- Codebase: `skills/lobster-dev/references/aquadif-migration.md` — verified 7-step checklist and patterns
- Codebase: `packages/lobster-transcriptomics/` — verified 22 tools with metadata, 14/14 contract tests
- Codebase: All 9 package `pyproject.toml` files — verified entry point module paths
- Codebase: All agent files — verified actual tool counts via `@tool` grep
- Codebase: `lobster/agents/graph.py` lines 198-277 — verified `_create_lazy_delegation_tool` structure

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — Phase 3 completion details and decision log
- `.planning/phases/03-reference-implementation/03-01-SUMMARY.md` — categorization mapping table
- `.planning/phases/04-agent-rollout/04-CONTEXT.md` — locked decisions and wave plan

### Tertiary (LOW confidence)

- None. All findings verified directly from codebase.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all infrastructure verified in codebase
- Architecture patterns: HIGH — Phase 3 reference implementation validated
- Tool counts: HIGH — counted directly from @tool grep on each package
- Category guidance: MEDIUM — derived from function names/domains; executor must verify against actual tool bodies using 80% rule
- Test infrastructure: HIGH — verified from transcriptomics reference and contract_mixins.py

**Research date:** 2026-02-28
**Valid until:** Stable — patterns frozen in Phase 3; no library upgrades expected
