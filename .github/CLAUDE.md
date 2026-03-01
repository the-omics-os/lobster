# Lobster AI — Contributor & Bot Guidelines

Guidelines for contributing to **Lobster AI**, the open-source multi-agent bioinformatics engine.
These rules apply to all code changes, whether from human contributors or automated agents.

---

## Project Overview

Lobster AI is a multi-agent bioinformatics platform for multi-omics data analysis
(scRNA-seq, bulk RNA-seq, proteomics, genomics, metabolomics). Users interact via
natural language; Lobster routes requests to specialized AI agents backed by stateless services.

**Key flow**: `CLI → Client → LangGraph → Supervisor → Specialist Agents → Services → DataManagerV2`

---

## Package Architecture

Lobster uses **PEP 420 namespace packages**. Agents are separate PyPI packages:

```
lobster-ai (core SDK)
├── lobster-transcriptomics  # 3 agents (scRNA-seq, annotation, DE)
├── lobster-research         # 2 agents (literature, data loading)
├── lobster-visualization    # 1 agent (plotting)
├── lobster-metadata         # 1 agent (ID mapping, filtering)
├── lobster-structural-viz   # 1 agent (protein structure)
├── lobster-genomics         # 2 agents (genomics, variant analysis)
├── lobster-proteomics       # 3 agents (proteomics, proteomics DE, biomarker discovery)
├── lobster-metabolomics     # 1 agent (LC-MS/GC-MS/NMR)
├── lobster-ml               # 3 agents (ML, feature selection, survival)
└── lobster-drug-discovery   # 4 agents (drug discovery, cheminformatics, clinical dev, pharmacogenomics)
```

Agents register via **entry points** — `ComponentRegistry` is the single source of truth.

---

## Hard Rules

1. **Do NOT edit `pyproject.toml`** — dependency changes require maintainer approval
2. **Prefer editing existing files** over creating new ones
3. **No `lobster/__init__.py`** — PEP 420 namespace package (hard requirement)
4. **ComponentRegistry is the source of truth** — agents discovered via entry points, NOT hardcoded registries
5. **AGENT_CONFIG at module top** — define before heavy imports for fast entry point discovery
6. **Keep services stateless** — pure functions returning `(AnnData, Dict, AnalysisStep)`
7. **Always pass `ir` into `log_tool_usage()`** — no IR = not reproducible
8. **Agent packages in `packages/`** — new agents go in separate packages, NOT in core
9. **No module-level `component_registry` calls** — causes slow startup and import side effects
10. **Run `make format` before committing** — black + isort
11. **All tools MUST have AQUADIF metadata** — `.metadata` and `.tags` assigned after `@tool` decorator. Use string literals (`"ANALYZE"`), not enum imports. See `skills/lobster-dev/references/aquadif-contract.md`
12. **PRs adding new tools MUST include contract tests** — run `pytest -m contract` before submitting. CI validates this automatically on every PR

---

## Service Pattern (3-tuple)

All analysis services MUST return a 3-tuple:

```python
def analyze(self, adata: AnnData, **params) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
    return processed_adata, stats_dict, ir  # ir = provenance/reproducibility
```

Tools wrap services and log provenance:

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    adata = data_manager.get_modality(modality_name)
    result, stats, ir = service.analyze(adata, **params)
    data_manager.log_tool_usage("analyze", params, stats, ir=ir)  # IR mandatory
    return f"Complete: {stats}"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `agents/graph.py` | `create_bioinformatics_graph()` — config-driven graph builder |
| `agents/supervisor.py` | Supervisor agent — routes user intents |
| `core/component_registry.py` | Agent + plugin discovery via entry points |
| `core/data_manager_v2.py` | Modality/workspace orchestration |
| `core/provenance.py` | W3C-PROV tracking |
| `tools/download_orchestrator.py` | Download execution (9-step pipeline) |
| `core/omics_registry.py` | Omics type metadata and detection |
| `config/aquadif.py` | AQUADIF taxonomy — AquadifCategory enum, provenance rules |
| `testing/contract_mixins.py` | AgentContractTestMixin — 14 contract test methods |
| `core/aquadif_monitor.py` | Runtime AQUADIF monitoring (category + provenance tracking) |

---

## Agent Roles

| Agent | Package | Focus |
|-------|---------|-------|
| `supervisor` | core | Route user intents, manage handoffs |
| `research_agent` | lobster-research | Literature discovery, URL extraction (online) |
| `data_expert` | lobster-research | Execute downloads, load files (ZERO online access) |
| `transcriptomics_expert` | lobster-transcriptomics | scRNA-seq + bulk RNA-seq: QC, clustering, normalization |
| `annotation_expert` | lobster-transcriptomics | Cell type annotation (sub-agent) |
| `de_analysis_expert` | lobster-transcriptomics | Differential expression, GSEA, pathway enrichment (sub-agent) |
| `genomics_expert` | lobster-genomics | VCF/PLINK, LD pruning, kinship, GWAS, PCA |
| `variant_analysis_expert` | lobster-genomics | Variant annotation, ClinVar, gnomAD (sub-agent) |
| `proteomics_expert` | lobster-proteomics | MS import (MaxQuant/DIA-NN/Spectronaut), affinity proteomics, batch correction |
| `proteomics_de_analysis_expert` | lobster-proteomics | Proteomics DE, pathway enrichment, KSEA, STRING PPI (sub-agent) |
| `biomarker_discovery_expert` | lobster-proteomics | Panel selection, nested CV, hub protein extraction (sub-agent) |
| `metabolomics_expert` | lobster-metabolomics | LC-MS/GC-MS/NMR QC, normalization, PCA/PLS-DA, m/z annotation |
| `machine_learning_expert` | lobster-ml | ML parent, routes to sub-agents |
| `feature_selection_expert` | lobster-ml | Stability selection, LASSO, variance filter (sub-agent) |
| `survival_analysis_expert` | lobster-ml | Cox models, Kaplan-Meier, risk stratification (sub-agent) |
| `visualization_expert` | lobster-visualization | General-purpose Plotly plotting |
| `metadata_assistant` | lobster-metadata | ID mapping, filtering, validation |
| `protein_structure_visualization_expert` | lobster-structural-viz | PDB fetch, protein structure visualization |
| `drug_discovery_expert` | lobster-drug-discovery | Drug discovery workflows, routes to sub-agents |
| `cheminformatics_expert` | lobster-drug-discovery | Molecular descriptors, fingerprints, similarity search (sub-agent) |
| `clinical_dev_expert` | lobster-drug-discovery | Clinical trial design, endpoint analysis, safety signals (sub-agent) |
| `pharmacogenomics_expert` | lobster-drug-discovery | PGx variants, drug-gene interactions, metabolizer phenotyping (sub-agent) |

---

## Plugin Architecture

New omics types self-register via entry points — zero core changes required.

**Entry point groups** (7):
- `lobster.agents` — Agent configs
- `lobster.services` — Service classes
- `lobster.agent_configs` — Custom LLM configs
- `lobster.adapters` — Adapter factories
- `lobster.providers` — Provider classes
- `lobster.download_services` — Download service classes
- `lobster.queue_preparers` — Queue preparer classes

---

## Testing

```bash
make dev-install    # Full dev setup
make test           # All tests with coverage
make format         # black + isort (run before committing)
make lint           # flake8/pylint/bandit
```

Tests use `pytest`. Markers: `@pytest.mark.real_api` (needs network), `@pytest.mark.slow` (>30s).

---

## Code Style

- **Python 3.12+**, type hints encouraged
- **black + isort** formatting (enforced)
- Naming: `snake_case` for functions/variables, `PascalCase` for classes
- Modality naming: `geo_gse12345` → `geo_gse12345_filtered_normalized` → `geo_gse12345_clustered`
- Keep changes minimal and focused — don't refactor surrounding code unless asked
- Don't add features, comments, or type annotations beyond what's needed for the change

---

## PR Guidelines

- Keep PRs focused on a single concern
- Include test coverage for new services and tools
- Services must follow the 3-tuple pattern with provenance
- Don't break backward compatibility with existing `lobster-custom-*` packages
- New agents belong in `packages/`, not in core `lobster/agents/`
