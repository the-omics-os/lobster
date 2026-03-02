# Domain Knowledge Bridge: GPTomics Bio-Skills

GPTomics (`https://github.com/GPTomics/bioSkills`) is an open-source library of 425 bioinformatics skills across 62 categories. Each skill contains tool parameters, workflow steps, QC criteria, code patterns, and common pitfalls for a specific analysis task.

**Purpose:** lobster-dev teaches you HOW to build Lobster agents and services. GPTomics bio-skills teach you WHAT a domain's tools do and how they work. Combine both to build Lobster capabilities grounded in real bioinformatics knowledge.

---

## Repository Structure

```
GPTomics/bioSkills/
├── {category}/                    # 62 domain categories
│   ├── README.md                  # Category overview + skill list table
│   └── {skill}/                   # Individual skill
│       ├── SKILL.md               # Code patterns, parameters, best practices
│       ├── usage-guide.md         # Overview, prerequisites, example prompts
│       └── examples/              # Runnable scripts
└── workflows/                     # 40 end-to-end pipeline skills
    └── {domain}-pipeline/         # Complete analysis workflow
        ├── SKILL.md               # Pipeline steps, orchestration, dependencies
        └── ...
```

**Access content via raw GitHub URLs:**
```
# Category overview (lists all skills in that category)
https://raw.githubusercontent.com/GPTomics/bioSkills/main/{category}/README.md

# Individual skill (code patterns, parameters, QC criteria)
https://raw.githubusercontent.com/GPTomics/bioSkills/main/{category}/{skill}/SKILL.md

# End-to-end pipeline
https://raw.githubusercontent.com/GPTomics/bioSkills/main/workflows/{domain}-pipeline/SKILL.md
```

---

## Category Catalog

Use this catalog to identify which categories are relevant to the domain you're building for. Then fetch the category README from GitHub to discover specific skills.

| Area | Categories |
|------|-----------|
| **Sequencing & QC** | read-qc, read-alignment, alignment, alignment-files, sequence-io, sequence-manipulation, expression-matrix, rna-quantification |
| **Transcriptomics** | single-cell, differential-expression, alternative-splicing, rna-structure, ribo-seq, small-rna-seq, epitranscriptomics |
| **Genomics** | variant-calling, population-genetics, genome-assembly, genome-annotation, genome-intervals, copy-number, phasing-imputation, comparative-genomics, causal-genomics |
| **Epigenomics** | chip-seq, atac-seq, methylation-analysis, hi-c-analysis |
| **Proteomics & Metabolomics** | proteomics, metabolomics, chemoinformatics |
| **Immunology & Cytometry** | flow-cytometry, imaging-mass-cytometry, tcr-bcr-analysis, immunoinformatics |
| **Microbiome & Ecology** | metagenomics, microbiome, ecological-genomics |
| **Spatial** | spatial-transcriptomics |
| **Clinical** | clinical-databases, liquid-biopsy, genome-engineering, crispr-screens |
| **Systems Biology** | pathway-analysis, gene-regulatory-networks, systems-biology, multi-omics-integration |
| **ML & Reporting** | machine-learning, experimental-design, data-visualization, reporting |
| **Infrastructure** | workflow-management, database-access, primer-design, restriction-analysis |
| **Other Specialized** | long-read-sequencing, temporal-genomics, phylogenetics, structural-biology, epidemiological-genomics, clip-seq |
| **Pipelines** | `workflows/` — 40 end-to-end pipelines (scrnaseq, rnaseq-to-de, proteomics, metabolomics, cytometry, chipseq, atacseq, metagenomics, spatial, hic, gwas, multi-omics, and more) |

---

## Discovery Process

### Step 1: Identify Relevant Categories

From your Planning Workflow Phase 1 summary, match the biological domain and key tools against the catalog above. Most domains map to 1-3 categories. For example:

| Domain | Primary Categories | Pipeline Skill |
|--------|--------------------|---------------|
| Flow cytometry | flow-cytometry | `workflows/cytometry-pipeline` |
| Shotgun metagenomics | metagenomics, microbiome | `workflows/metagenomics-pipeline` |
| Epigenomics (ChIP-seq) | chip-seq, methylation-analysis | `workflows/chipseq-pipeline`, `workflows/methylation-pipeline` |
| Spatial transcriptomics | spatial-transcriptomics, single-cell | `workflows/spatial-pipeline` |

### Step 2: Fetch and Read

1. **Start with the pipeline skill** — fetch `workflows/{domain}-pipeline/SKILL.md` first. Pipeline skills show the end-to-end analysis flow and list their dependencies on individual skills. This is the highest-value content for designing a Lobster agent.

2. **Fetch the category README** — `{category}/README.md` lists all skills in that category with descriptions. Scan this to identify which specific skills are relevant.

3. **Fetch individual SKILL.md files** — for the specific analysis steps you need to implement. Each SKILL.md contains parameters, defaults, QC thresholds, and code patterns.

### Step 3: Extract What You Need

From each skill, extract:

| What to Extract | Use in Lobster |
|-----------------|----------------|
| Tool parameters and defaults | Service method signature |
| Input/output file formats | Input validation, AnnData mapping |
| Workflow steps and order | Agent tool sequence, system prompt |
| QC checkpoints and thresholds | Stats dict keys, validation logic |
| Common pitfalls | Error handling, input guards |
| Dependencies and versions | `pyproject.toml` dependencies |

---

## Translating Skills into Lobster Services

GPTomics skills describe HOW external tools work. Translate that into Lobster's service pattern:

| From GPTomics Skill | Lobster Service |
|---------------------|----------------|
| Tool CLI flags / function parameters | Service method signature |
| CLI command structure | `AnalysisStep.code_template` (Jinja2 for notebook export) |
| Input file format | Input validation in service method |
| Output file format | AnnData mapping (see below) |
| QC checkpoints | Stats dict keys and validation logic |
| Pitfalls / "gotchas" | Input validation guards and error messages |
| Dependencies | `pyproject.toml` dependencies |
| Workflow ordering | Agent tool execution order and system prompt |

### AnnData Mapping Convention

When a skill describes output that isn't naturally gene expression data, map it to AnnData:

| AnnData Slot | What Goes There |
|-------------|-----------------|
| `.X` | Primary data matrix (abundance, counts, scores) |
| `.obs` | Sample/row metadata (group, condition, sample ID) |
| `.var` | Feature/column metadata (taxon lineage, gene ID, pathway) |
| `.obsm` | Embeddings or coordinates (PCA, PCoA, UMAP) |
| `.uns` | Run metadata (tool versions, parameters, QC summary) |
| `.layers` | Alternative representations (raw counts + normalized) |

### Service Design Checklist

When translating GPTomics knowledge into a Lobster service:

- [ ] Method signature mirrors the tool's key parameters
- [ ] Input validation catches pitfalls described in the skill
- [ ] Stats dict includes QC metrics the skill recommends
- [ ] AnnData mapping preserves all meaningful output fields
- [ ] `code_template` in AnalysisStep reproduces the tool invocation
- [ ] Dependencies are listed in `pyproject.toml`
- [ ] Workflow ordering informs agent tool sequence

---

## When No Skills Cover the Domain

If GPTomics has no relevant skills for your domain:

1. **Official documentation** — check the tool's docs for parameters, formats, workflows
2. **Published papers** — workflow papers in Nature Methods, Bioinformatics, Genome Biology
3. **PyPI packages** — search for existing Python wrappers
4. **Developer knowledge** — ask for reference code, papers, or example scripts
5. **Web search** — `"{tool name} tutorial"` or `"{tool name} best practices"`

Document findings in the same structured format as Phase 4 of the planning workflow so the developer can review before building.

---

## Key Principles

1. **Read, don't copy** — use skills as a requirements spec, not source code
2. **Translate, don't wrap** — skills describe CLI tools; Lobster needs Python services returning 3-tuples
3. **Preserve provenance** — every step must produce an `AnalysisStep` for notebook export
4. **Map to AnnData** — all results must fit Lobster's data model
5. **Combine both skills** — GPTomics for domain knowledge + lobster-dev for implementation patterns
