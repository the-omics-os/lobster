# Planning Workflow for New Capabilities

**STOP. Before creating any new agent, service, or package, follow these six phases.**

This workflow prevents wasted effort by ensuring you understand the need, know what
already exists, and gather domain knowledge before writing code.

**Skip this workflow if:** you're fixing a bug, adding a tool to an existing agent,
or working on core infrastructure. This is for NEW capabilities only.

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

Use these when initial answers are ambiguous:

- **Domain is broad** (e.g., "genomics") -- "Which specific assay? WGS, WES, targeted panel, amplicon?"
- **Workflow is vague** (e.g., "analyze my data") -- "Walk me through each step from raw files to the result you want."
- **Overlaps with existing** -- "How is this different from what lobster-transcriptomics (or lobster-X) already does?"
- **Tools unclear** -- "Which specific tools or libraries does your lab currently use for this?"

### Output: Structured Need Summary

After gathering answers, produce this summary:

```
Domain:        [e.g., Shotgun metagenomics]
Workflow:      [e.g., FASTQ -> QC -> Classification -> Abundance -> Diversity -> Functional]
Input formats: [e.g., FASTQ, Kraken2 reports, BIOM tables]
Key libraries: [e.g., Kraken2, Bracken, MetaPhlAn, HUMAnN3]
Outputs:       [e.g., Taxonomic profiles, diversity metrics, pathway tables]
Example data:  [e.g., SRA accession SRR12345678 or local path]
```

---

## Phase 2: Check What Exists in Lobster

**ALL DISCOVERY IS DYNAMIC.** Do not rely on memorized package lists -- scan the filesystem.

### Step 2a: Scan Agent Packages

```bash
# List all agent packages
ls packages/lobster-*/pyproject.toml

# For each package, read description and entry points:
# Look for [project] description = "..."
# Look for [project.entry-points."lobster.agents"] -> agent names
```

For each package found, assess: does it overlap with the developer's stated domain or workflow?

### Step 2b: Scan Core Services

```bash
# List service categories
ls lobster/services/

# Key directories to check:
# analysis/      - clustering, DE, GWAS, etc.
# data_access/   - GEO, SRA, PRIDE downloaders
# data_management/ - modality CRUD, concatenation
# visualization/ - plot generation
# metadata/      - standardization, validation
# ml/            - feature selection, survival, CV, SHAP
# quality/       - QC and preprocessing
# orchestration/ - publication processing
```

For each service directory, assess: can any existing services be reused for the developer's workflow?

### Step 2c: Check Partial Overlap

Lobster commonly handles PART of a new domain's needs. Check these:

| Existing Capability | Covers | Agent/Service |
|---------------------|--------|---------------|
| Data discovery | Searching PubMed/GEO/SRA for any domain | `research_agent` |
| File loading | Standard formats (H5AD, CSV, 10X) | `data_expert` |
| Visualization | Plotly charts for any tabular/matrix data | `visualization_expert` |
| Machine learning | Feature selection, survival, CV for any features | `machine_learning_expert` |
| Metadata | ID mapping, filtering for any domain | `metadata_assistant` |
| Download orchestration | Queue-based download of any dataset | `tools/download_orchestrator.py` |

### Output: Overlap Assessment

```
Existing coverage:
- [what Lobster already handles for this domain]

Gaps:
- [what needs building -- specific capabilities]

Reusable:
- [existing services/agents that can be leveraged as-is]
```

---

## Phase 3: Find Domain Knowledge

Use the GPTomics bio-skills library to discover external domain expertise.

> **Load [references/bioskills-bridge.md](bioskills-bridge.md)** and follow the
> dynamic discovery process to find relevant GPTomics skills for the domain.
>
> Use discovered skills as a **requirements specification** for Lobster
> service design -- parameters, workflows, QC criteria, best practices.

If GPTomics skills don't cover the domain, fall back to:
1. Official documentation for the key tools/libraries
2. Published workflow papers (Nature Methods, Bioinformatics)
3. PyPI packages that wrap the tools
4. Developer-provided reference code or scripts

---

## Phase 4: Present Findings

Before recommending an approach, present what you found. Use this template:

```markdown
## Capability Assessment

### Already Handled by Lobster
- [capability] via [agent/service] (found at packages/lobster-X/ or lobster/services/Y/)

### Needs Building
- [capability] -- requires new service in [location]
- [capability] -- requires new agent tool

### Domain Knowledge Found
- [source/category] -- covers [specific knowledge] (discovered via [method])

### No Domain Knowledge Available
- [what requires manual research or developer input]

### Reusable Infrastructure
- [existing component] can be used for [purpose]
```

This gives the developer a clear picture before any code is written.

---

## Phase 5: Recommend Approach

Based on findings, recommend **one** of these options:

### Option A: Extend Existing Agent

**When:** The domain is closely related to an existing agent's scope.

**Examples:**
- Adding methylation calling to `lobster-genomics`
- Adding trajectory inference to `lobster-transcriptomics`
- Adding a new plot type to `lobster-visualization`

**Action:**
1. Add new services to the existing package
2. Add new tools to the existing agent
3. Update the agent's system prompt to cover new capabilities
4. Add tests for new services

### Option B: New Agent Package

**When:** The domain is distinct and needs its own specialist agent.

**Examples:**
- Metagenomics (taxonomy, diversity, functional profiling)
- Spatial transcriptomics (coordinate-aware analysis)
- Flow cytometry (gating, compensation, population analysis)

**Action:**
1. Follow the full package structure in [creating-agents.md](creating-agents.md)
2. Implement agent, services, tools, prompts, config, state
3. Register entry points in `pyproject.toml`
4. Pass all contract tests
5. Complete the 28-step checklist

### Option C: Service Only (No New Agent)

**When:** The capability is a utility that existing agents can call.

**Examples:**
- New file format parser (e.g., BIOM table reader)
- New QC metric applicable across domains
- New plot type for the visualization agent
- New statistical test usable by multiple agents

**Action:**
1. Implement service per [creating-services.md](creating-services.md) (3-tuple pattern)
2. Register in the appropriate agent package or core
3. Wrap as a tool in the relevant agent
4. Add unit tests

### Option D: Not Appropriate for Lobster

**When:** The capability is a standalone tool, not a multi-agent workflow step.

**Examples:**
- Snakemake/Nextflow pipeline runner
- Raw CLI wrapper without data transformation
- One-off file conversion utility
- Tool that doesn't produce or consume AnnData-compatible data

**Action:**
- Suggest using GPTomics skills directly or building a standalone package
- If the developer disagrees, revisit -- they may see integration potential you missed

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

## Phase 6: Build and Test

Once the developer approves the recommendation, execute:

| Approach | Implementation Steps |
|----------|---------------------|
| A (Extend) | Read existing agent code -> add services -> add tools -> update prompts -> test |
| B (New package) | Follow [creating-agents.md](creating-agents.md) scaffold -> implement -> test |
| C (Service only) | Implement per [creating-services.md](creating-services.md) -> register as tool -> test |
| D (Not Lobster) | Guide developer to appropriate approach outside Lobster |

### Verification Checklist

Before declaring the work complete:

- [ ] Contract tests pass (`AgentContractTestMixin` for new agents)
- [ ] Service tests pass (3-tuple return validated for each method)
- [ ] `make test` passes (no regressions)
- [ ] Manual test with example dataset from Phase 1
- [ ] Entry point discoverable via `ComponentRegistry`
- [ ] AGENT_CONFIG defined at module top (before heavy imports)
- [ ] All `log_tool_usage()` calls include `ir=ir`
- [ ] No `lobster/__init__.py` created (PEP 420 namespace preserved)
