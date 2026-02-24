# Domain Knowledge Bridge: GPTomics Bio-Skills

GPTomics is a library of bioinformatics skills for coding agents (Claude Code, Codex,
Gemini CLI, OpenClaw). It provides deep domain expertise — tool parameters, workflow
steps, QC criteria, common pitfalls — that Lobster's development skill does not cover.

**Purpose:** lobster-dev teaches you HOW to build agents and services. GPTomics
bio-skills teach you WHAT a domain's tools do and how they work. Combine both to
build Lobster capabilities grounded in real bioinformatics knowledge.

---

## How GPTomics Skills Work

GPTomics skills are installed as coding agent skills. They follow the naming pattern:

```
bio-{category}-{skill-name}
```

**Examples:**
- `bio-single-cell-preprocessing` — QC, filtering, normalization for scRNA-seq
- `bio-chip-seq-peak-calling` — MACS3 peak calling for ChIP-seq
- `bio-metagenomics-kraken-classification` — Kraken2 taxonomic classification
- `bio-workflows-rnaseq-to-de` — End-to-end RNA-seq pipeline
- `bio-metabolomics-xcms-preprocessing` — XCMS3 LC-MS preprocessing

Skills prefixed with `bio-workflows-` describe end-to-end pipelines and are
particularly valuable — they show the complete analysis flow a Lobster agent
should orchestrate.

---

## How to Find Relevant Skills

### Step 1: Identify the Domain

From the Planning Workflow Phase 1 summary, extract the biological domain
and key analysis tools. For example:

```
Domain: Shotgun metagenomics
Key libraries: Kraken2, Bracken, MetaPhlAn, HUMAnN3
```

### Step 2: Search by Domain Keywords

Search your available skills for domain-related terms. The coding agent
can discover skills by searching its skill list for keywords:

```
Search patterns to try:
- bio-{domain}*           → bio-metagenomics-*
- bio-*-{tool}*           → bio-*-kraken-*
- bio-workflows-{domain}* → bio-workflows-metagenomics-*
```

### Step 3: Invoke Relevant Skills

When you find a matching skill, invoke it to load its full content.
The skill body contains:

| What to Extract | Use in Lobster |
|-----------------|----------------|
| Tool parameters and defaults | Service method signature |
| Input/output file formats | Input validation, AnnData mapping |
| Workflow steps and order | Agent tool sequence, system prompt |
| QC checkpoints and thresholds | Stats dict keys, validation logic |
| Common pitfalls | Error handling, input guards |
| Dependencies and versions | `pyproject.toml` dependencies |

### Step 4: Check for Workflow Skills

Always look for `bio-workflows-*` skills matching the domain. These describe
complete analysis pipelines and map directly to what a Lobster agent orchestrates:

- `bio-workflows-scrnaseq-pipeline` → transcriptomics expert workflow
- `bio-workflows-chipseq-pipeline` → potential genomics child agent
- `bio-workflows-metabolomics-pipeline` → metabolomics expert workflow
- `bio-workflows-proteomics-pipeline` → proteomics expert workflow

---

## Translating Skills into Lobster Services

GPTomics skills describe HOW external tools work. Translate that into Lobster's
service pattern:

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

When a skill describes output that isn't naturally gene expression data,
map it to AnnData:

| AnnData Slot | What Goes There |
|-------------|-----------------|
| `.X` | Primary data matrix (abundance, counts, scores) |
| `.obs` | Sample/row metadata (group, condition, sample ID) |
| `.var` | Feature/column metadata (taxon lineage, gene ID, pathway) |
| `.obsm` | Embeddings or coordinates (PCA, PCoA, UMAP) |
| `.uns` | Run metadata (tool versions, parameters, QC summary) |
| `.layers` | Alternative representations (raw counts + normalized) |

### Example Translation

Suppose a `bio-metagenomics-kraken-classification` skill describes:
- Parameters: `--confidence 0.0`, `--threads 4`, `--db /path/to/db`
- Output: report with columns (pct, reads_rooted, reads_direct, rank, taxid, name)

The Lobster service translation:

```python
class TaxonomicClassificationService:
    def classify(self, reads_path: str, db_path: str,
                 confidence: float = 0.0, threads: int = 4,
                 **kwargs) -> Tuple[AnnData, Dict, AnalysisStep]:
        # 1. Run Kraken2
        # 2. Parse report
        # 3. Map to AnnData:
        #    X = abundance matrix (samples x taxa)
        #    var = taxonomy info (taxid, rank, lineage)
        #    uns = {"tool": "kraken2", "confidence": confidence}
        # 4. Build AnalysisStep with code_template
        return adata, stats, ir
```

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

If GPTomics has no relevant skills:

1. **Official documentation** — check the tool's docs for parameters, formats, workflows
2. **Published papers** — workflow papers in Nature Methods, Bioinformatics, Genome Biology
3. **PyPI packages** — search for existing Python wrappers
4. **Developer knowledge** — ask for reference code, papers, or example scripts
5. **Web search** — `"{tool name} tutorial"` or `"{tool name} best practices"`

Document findings in the same structured format as Phase 4 of the
planning workflow so the developer can review before building.

---

## Key Principles

1. **Read, don't copy** — use skills as a requirements spec, not source code
2. **Translate, don't wrap** — skills describe CLI tools; Lobster needs Python services returning 3-tuples
3. **Preserve provenance** — every step must produce an `AnalysisStep` for notebook export
4. **Map to AnnData** — all results must fit Lobster's data model
5. **Combine both skills** — GPTomics for domain knowledge + lobster-dev for implementation patterns
