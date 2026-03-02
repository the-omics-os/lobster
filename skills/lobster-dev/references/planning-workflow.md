# Planning Workflow for New Capabilities

**STOP. Before creating any new agent, service, or package, follow these six phases.**

This workflow prevents wasted effort by ensuring you understand the need, know what
already exists, and gather domain knowledge before writing code.

**Skip this workflow if:** you're fixing a bug, adding a tool to an existing agent,
or working on core infrastructure. This is for NEW capabilities only.

**Prerequisite:** Read [aquadif-contract.md](aquadif-contract.md) before designing any tools.

---

## Phase 1: Understand the Need

Ask these 6 questions in order. Do not proceed until you have clear answers.

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | What biological domain or data type? | Maps to existing agents + external domain knowledge |
| 2 | What is the end-to-end workflow? (raw data to final result) | Reveals which steps Lobster already handles |
| 3 | What input data formats? | May already have parsers/adapters |
| 4 | What key analysis tools or libraries? | Determines service scope and dependencies |
| 5 | What outputs does the user expect? | Plots, tables, annotated data, reports |
| 6 | Is there an example dataset available? | Essential for testing during build |

### Clarifying Follow-ups

Your main purpose is to take of thinking load from the user when it comes to implementation. You only want to understand the input and output requirements of the user. Your job is to reason about how to implement using the skill.

### Output: Structured Need Summary

```
Domain:        [e.g., Shotgun metagenomics]
Workflow:      [e.g., FASTQ -> QC -> Classification -> Abundance -> Diversity -> Functional]
Input formats: [e.g., FASTQ, Kraken2 reports, BIOM tables]
Key libraries: [e.g., Kraken2, Bracken, MetaPhlAn, HUMAnN3]
Outputs:       [e.g., Taxonomic profiles, diversity metrics, pathway tables]
Example data:  [e.g., SRA accession SRR12345678 or local path]
```

---

## Phase 2: Check What Exists

**Goal:** Compare the user's domain against what Lobster already handles. You already discovered installed agents and your development mode in Step 0 — use those results here.

### Step 1: Source Discovery (reading reference implementations)

You need to read existing agent and service code to understand patterns. How you access source depends on your development mode.

**Contributor (in repo):**
```bash
# Browse agent packages directly
ls packages/lobster-*/pyproject.toml
cat packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py

# Browse core services
ls lobster/services/
```

**Plugin author (installed package only):**

Agent packages are PEP 420 namespace packages — they install as separate distributions, NOT inside the core `lobster.__path__`. Use Python's import system to find them:

```bash
# Find a specific agent package's source (each is its own distribution)
python -c "import lobster.agents.transcriptomics as m; print(m.__path__[0])"
# e.g. → .venv/lib/python3.12/site-packages/lobster/agents/transcriptomics

# Browse that agent's tools (read-only — do NOT edit installed files)
AGENT_PATH=$(python -c "import lobster.agents.transcriptomics as m; print(m.__path__[0])")
ls "$AGENT_PATH"
cat "$AGENT_PATH/shared_tools.py"

# Core services ARE in the core SDK path
python -c "import lobster.services as s; print(s.__path__[0])"
```

**If installed source is insufficient** (e.g., only `.pyc` files or missing packages), clone the repo for reference:
```bash
git clone --depth 1 https://github.com/the-omics-os/lobster /tmp/lobster-ref
# Use /tmp/lobster-ref/packages/ to read reference implementations
# Do NOT develop inside this clone — scaffold your package separately
```

### Step 2: Ecosystem Reference

These are ALL published Lobster AI packages. Not all may be installed locally. Check for overlap with the user's domain:

| Package | PyPI | Agents | Domain Coverage |
|---|---|---|---|
| `lobster-ai` | Public | supervisor | Core SDK, orchestration, shared services |
| `lobster-research` | Public | research_agent, data_expert_agent | Literature search (PubMed/GEO/SRA), dataset download, file loading |
| `lobster-transcriptomics` | Public | transcriptomics_expert, annotation_expert, de_analysis_expert | scRNA-seq, bulk RNA-seq, QC, clustering, DE, GSEA, cell annotation |
| `lobster-proteomics` | Public | proteomics_expert, proteomics_de_analysis_expert, biomarker_discovery_expert | MS (MaxQuant/DIA-NN/Spectronaut), affinity (Olink/SomaScan), DE, biomarkers |
| `lobster-genomics` | Public | genomics_expert, variant_analysis_expert | VCF/PLINK, GWAS, LD pruning, variant annotation (VEP/ClinVar) |
| `lobster-metabolomics` | Public | metabolomics_expert | LC-MS, GC-MS, NMR, QC, normalization, PCA/PLS-DA, m/z annotation |
| `lobster-visualization` | Public | visualization_expert_agent | Plotly-based plots for any tabular/matrix data |
| `lobster-drug-discovery` | Public | drug_discovery_expert, cheminformatics_expert, clinical_dev_expert, pharmacogenomics_expert | Target ID, compound profiling, clinical translation, PGx |
| `lobster-metadata` | in-development | metadata_assistant | ID mapping, validation, publication queue filtering |
| `lobster-ml` | in-development | machine_learning_expert, feature_selection_expert, survival_analysis_expert | Feature selection, survival, cross-validation, SHAP |
| `lobster-structural-viz` | in-development | protein_structure_visualization_expert | Protein structure rendering |

### Step 3: Cross-Domain Capabilities

These capabilities are domain-agnostic and available to ALL new agents via the existing ecosystem. Do NOT rebuild them if you could just create a plugin for the available packagaes:

| Capability | Handled By | What It Does |
|---|---|---|
| Literature & dataset search | `research_agent` | PubMed, GEO, SRA, PRIDE, MetaboLights search for ANY domain |
| File loading & download | `data_expert_agent` | Generic CSV/H5AD/10X loading, queue-based download orchestration |
| Visualization | `visualization_expert_agent` | Plotly charts for any tabular or matrix data |
| Machine learning | `machine_learning_expert` | Feature selection, survival analysis, CV for any feature matrix |
| Metadata management | `metadata_assistant` | ID mapping, validation, filtering for any domain |

### Output: Overlap Assessment

```
Installed agents:
- [list from Step 0 discovery in SKILL.md]

Existing coverage for this domain:
- [what Lobster already handles — name agents/services, not file paths]

Gaps (what needs building):
- [specific capabilities not covered by any existing agent]

Reusable (can leverage as-is):
- [cross-domain agents/services that apply to this workflow]
```

---

## Phase 3: Find Domain Knowledge

Use the GPTomics bio-skills library to discover external domain expertise.

> **Load [bioskills-bridge.md](bioskills-bridge.md)** and follow the dynamic discovery
> process to find relevant GPTomics skills for the domain.
>
> Use discovered skills as a **requirements specification** for Lobster service
> design — parameters, workflows, QC criteria, best practices.

If GPTomics skills don't cover the domain, fall back to:
1. Official documentation for the key tools/libraries
2. Published workflow papers (Nature Methods, Bioinformatics)
3. PyPI packages that wrap the tools
4. Developer-provided reference code or scripts

### Map Domain Knowledge to AQUADIF Categories

After discovering domain workflows, map each workflow step to an AQUADIF category:

| Domain Step | AQUADIF Category | Rationale |
|---|---|---|
| [workflow step] | [category] | [why this fits] |

This mapping becomes your tool inventory for agent design. See [aquadif-contract.md](aquadif-contract.md) for category definitions and the multi-category decision flowchart.

---

## Phase 4: Present Findings

Before recommending an approach, present what you found:

```markdown
## Capability Assessment

### Already Handled by Lobster
- [capability] via [agent_name] (installed: yes/no)

### Needs Building
- [capability] — requires new [service/tool/agent]

### Domain Knowledge Found
- [source] — covers [specific knowledge]

### Reusable Infrastructure
- [existing agent/service] can be used for [purpose]
```

---

## Phase 5: Recommend Approach

Based on findings, recommend **one** of these options:

### Option A: Extend Existing Agent

**When:** The domain is closely related to an existing agent's scope.

**Examples:** Adding methylation calling to `lobster-genomics`. Adding trajectory inference to `lobster-transcriptomics`. Adding a new plot type to `lobster-visualization`.

**Next:** Read the existing agent's source code, then add services + tools. See [creating-agents.md](creating-agents.md) §Tool Design.

### Option B: New Agent Package

**When:** The domain is distinct and needs its own specialist agent.

**Examples:** Metagenomics (taxonomy, diversity, functional profiling). Spatial transcriptomics (coordinate-aware analysis). Flow cytometry (gating, compensation, population analysis).

**Next:** Run `lobster scaffold agent` then follow [creating-agents.md](creating-agents.md) for the full implementation guide and 29-step checklist.

### Option C: Service Only (No New Agent)

**When:** The capability is a utility that existing agents can call.

**Examples:** New file format parser (e.g., BIOM reader). New QC metric applicable across domains. New statistical test usable by multiple agents.

**Next:** Implement per [creating-services.md](creating-services.md) (3-tuple pattern), wrap as a tool in the relevant agent.

### Option D: Not Appropriate for Lobster

**When:** The capability doesn't fit the multi-agent workflow model.

**Examples:** you have to decide

**Next:** Suggest GPTomics skills directly or a standalone package. If the developer disagrees, revisit — they may see integration potential you missed.

### Recommendation Template

```
Recommended: [A / B / C / D]
Rationale:   [why this option fits best]
Scope:
  - [N] new service(s): [names and brief descriptions]
  - [N] new tool(s): [names and brief descriptions]
  - [N] existing to reuse: [names and how they'll be leveraged]
Domain knowledge: [what was discovered, from where]
```

---

## Phase 6: Build

Once the developer approves the recommendation, execute:

| Approach | What To Read |
|---|---|
| A (Extend) | Existing agent source → [creating-agents.md](creating-agents.md) §Tool Design |
| B (New package) | [scaffold.md](scaffold.md) → [creating-agents.md](creating-agents.md) → 29-step checklist |
| C (Service only) | [creating-services.md](creating-services.md) |
| D (Not Lobster) | Guide developer to appropriate approach outside Lobster |

### Development Loop by Mode

**Contributor (in repo):**
```bash
# 1. Scaffold inside the repo
lobster scaffold agent --name <domain>_expert ... -o packages/

# 2. Install in dev mode (picks up all packages/)
make dev-install

# 3. Verify agent is discovered
python -c "from lobster.core.component_registry import component_registry; component_registry.reset(); print(component_registry.has_agent('<domain>_expert'))"

# 4. Run tests
pytest packages/lobster-<domain>/tests/ -v
make test  # full suite, check for regressions
```

**Plugin author (standalone):**
```bash
# 1. Scaffold anywhere on disk
lobster scaffold agent --name <domain>_expert ...

# 2. Install alongside lobster (editable for development)
cd lobster-<domain>/
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"

# 3. Verify agent is discovered by the existing lobster installation
python -c "from lobster.core.component_registry import component_registry; component_registry.reset(); print(component_registry.has_agent('<domain>_expert'))"

# 4. Run tests
pytest tests/ -v

# 5. Test with lobster CLI
lobster chat   # your agent should now appear in routing
```

Verification steps for each approach are in the respective reference files.
