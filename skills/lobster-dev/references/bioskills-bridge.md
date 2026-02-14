# Domain Knowledge Bridge: bioSkills

bioSkills is an external repository of bioinformatics skills for coding agents.
It provides deep domain expertise -- tool parameters, workflow steps, QC criteria,
common pitfalls -- that Lobster's development skill does not cover.

**Purpose:** Lobster-dev teaches you HOW to build agents and services. bioSkills
teaches you WHAT a domain's tools do and how they work. Combine both to build
domain-specific Lobster capabilities grounded in real bioinformatics knowledge.

**Not installed?** Clone it:

```bash
git clone https://github.com/[org]/bioSkills ~/GITHUB/bioSkills

# Or install specific categories for your coding agent:
cd ~/GITHUB/bioSkills && ./install-claude.sh --categories "metagenomics,microbiome"
```

---

## How to Find Relevant bioSkills

**All discovery is dynamic.** Never memorize or hardcode category lists, skill
names, or file paths. The repository evolves -- new categories appear, skills
are renamed, workflows are added. Always scan at runtime.

### Step 1: Locate the Repository

Check these paths in order. Use the first one that exists:

```bash
ls ~/GITHUB/bioSkills/          # Development clone
ls ~/.claude/skills/            # Installed via install-claude.sh
ls ~/.agents/skills/            # Installed for other agents
```

If none exist, ask the developer for the path or suggest cloning.

### Step 2: List Available Categories

Scan top-level directories. Each directory is a domain category:

```bash
ls ~/GITHUB/bioSkills/
# Example output: single-cell/  metagenomics/  variant-calling/  workflows/  ...
```

Do not assume which categories exist. Read the filesystem.

### Step 3: Identify Candidate Categories

Match the developer's stated domain (from Planning Workflow Phase 1) against
the directory names found in Step 2. Use keyword matching:

- **Direct match:** developer said "metagenomics" and you see `metagenomics/`
- **Related domains:** also check adjacent categories (e.g., metagenomics -> also check `microbiome/`, `amplicon/`)
- **Workflows:** always check `workflows/` for end-to-end pipeline skills

### Step 4: Read SKILL.md Frontmatter for Each Candidate

For each candidate category, scan the skills within it:

```bash
ls ~/GITHUB/bioSkills/{category}/
# Returns: skill-a/  skill-b/  skill-c/  ...
```

Read the first 10-15 lines of each SKILL.md file to get the frontmatter:

```yaml
---
name: bio-{category}-{skill}
description: |
  Use when [trigger phrases]...
tool_type: [analysis|qc|visualization|...]
primary_tool: [tool name]
---
```

The `description` field contains "Use when..." triggers that confirm relevance
to the developer's stated need.

### Step 5: Read Full SKILL.md for Confirmed Matches

For skills whose description matches the developer's need, read the full
SKILL.md content. Extract:

| What to Extract | Where to Find It |
|-----------------|-----------------|
| Tool parameters and CLI flags | Parameter tables or code blocks |
| Input/output file formats | Format description sections |
| Workflow steps and dependencies | Step-by-step sections |
| QC checkpoints and thresholds | Quality control sections |
| Common pitfalls and best practices | Tips, warnings, or gotcha sections |
| Required dependencies and versions | Prerequisites or installation sections |

### Step 6: Check Workflows

Also scan the `workflows/` directory for end-to-end pipeline skills:

```bash
ls ~/GITHUB/bioSkills/workflows/
# Look for: *-pipeline skills matching the domain
```

Workflow skills are particularly valuable -- they show the complete analysis
flow that a Lobster agent should orchestrate, including step ordering,
decision points, and expected intermediate outputs.

---

## Translating bioSkills Knowledge into Lobster Services

bioSkills describes HOW external tools work. You need to translate that into
Lobster's service pattern. Here is the mapping:

| Extract from bioSkills SKILL.md | Use in Lobster Service |
|--------------------------------|----------------------|
| Tool CLI flags / function parameters | Service method signature (`def classify(self, adata, confidence=0.0, threads=4)`) |
| CLI command structure | `AnalysisStep.code_template` (Jinja2 template for notebook export) |
| Input file format description | Input validation in service method |
| Output file format description | AnnData mapping -- decide what goes in `.X`, `.obs`, `.var`, `.uns` |
| QC checkpoints / thresholds | Stats dict keys and validation logic |
| Common pitfalls / "gotchas" | Input validation guards and error messages |
| Tool dependencies / versions | `pyproject.toml` dependencies for the agent package |
| Workflow step ordering | Agent tool execution order and system prompt guidance |

### AnnData Mapping Convention

When a bioSkill describes output that isn't naturally gene expression data,
map it to AnnData following these conventions:

| AnnData Slot | What Goes There |
|-------------|-----------------|
| `.X` | Primary data matrix (abundance, counts, distances, scores) |
| `.obs` | Sample/row metadata (group, timepoint, condition, sample ID) |
| `.var` | Feature/column metadata (taxon lineage, gene ID, pathway, annotation) |
| `.obsm` | Embeddings or coordinates (PCA, PCoA, UMAP, t-SNE) |
| `.uns` | Run metadata (tool versions, parameters used, QC summary statistics) |
| `.layers` | Alternative representations (raw counts alongside normalized) |

### Example Translation

Suppose a bioSkill describes Kraken2 classification:
- Takes `--confidence 0.0`, `--threads 4`, `--db /path/to/db`
- Outputs a report with columns: pct, reads_rooted, reads_direct, rank, taxid, name

The Lobster service translation:

```python
class TaxonomicClassificationService:
    def classify(self, reads_path: str, db_path: str,
                 confidence: float = 0.0, threads: int = 4,
                 **kwargs) -> Tuple[AnnData, Dict, AnalysisStep]:
        # 1. Run Kraken2 (or call Python wrapper)
        # 2. Parse report into structured data
        # 3. Map to AnnData:
        #    X = abundance matrix (samples x taxa)
        #    var = taxonomy info (taxid, rank, lineage)
        #    uns = {"tool": "kraken2", "version": "2.1.3",
        #           "confidence": confidence, "db": db_path}
        # 4. Build AnalysisStep with code_template for notebook export
        return adata, stats, ir
```

### Service Design Checklist (Using bioSkills Knowledge)

When translating bioSkills knowledge into a Lobster service:

- [ ] Method signature mirrors the tool's key parameters (from bioSkills)
- [ ] Input validation catches the pitfalls described in bioSkills
- [ ] Stats dict includes the QC metrics bioSkills recommends checking
- [ ] AnnData mapping preserves all meaningful output fields
- [ ] `code_template` in AnalysisStep reproduces the tool invocation
- [ ] Dependencies from bioSkills are listed in `pyproject.toml`
- [ ] Workflow ordering from bioSkills informs agent tool sequence

---

## When bioSkills Has No Coverage

If bioSkills doesn't cover the domain, or the repository is not installed:

1. **Official documentation** -- check the tool's docs for parameters, formats, workflows
2. **Published papers** -- look for workflow papers in Nature Methods, Bioinformatics, Genome Biology
3. **PyPI packages** -- search for existing Python wrappers (`pip search` or pypi.org)
4. **Developer knowledge** -- ask the developer for reference code, papers, or example scripts
5. **Web search** -- search for `"{tool name} tutorial"` or `"{tool name} best practices"`

Document what you find in the same structured format as Phase 4 of the
planning workflow so the developer can review before you start building.

---

## Key Principles

1. **Never hardcode** -- bioSkills evolves. Always discover dynamically.
2. **Read, don't copy** -- use bioSkills as a requirements spec, not source code.
3. **Translate, don't wrap** -- bioSkills describes CLI tools; Lobster needs Python services returning 3-tuples.
4. **Preserve provenance** -- every step must produce an `AnalysisStep` for notebook export.
5. **Map to AnnData** -- all results must fit Lobster's data model, even non-expression data.
