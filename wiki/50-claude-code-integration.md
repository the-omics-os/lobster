# Claude Code Integration

> **Use Lobster AI as a skill within Claude Code for IDE-native bioinformatics workflows**

## Overview

Lobster AI integrates with [Claude Code](https://claude.ai/code) as a custom skill, enabling bioinformatics analyses directly from your IDE through natural language. Claude Code automatically detects bioinformatics tasks and delegates to Lobster's specialized agents.

**Key Benefits:**
- 🎯 **Seamless workflow**: Stay in your IDE while running complex analyses
- 🤖 **Intelligent routing**: Claude Code detects when to use Lobster automatically
- 💬 **Multi-turn conversations**: Session continuity enables follow-up questions
- 📊 **Direct file access**: Results saved in your workspace, immediately usable
- 🔄 **Version control**: Track analysis workflows alongside code

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your IDE/Terminal                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Claude Code Agent                       │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │    User: "Analyze GSE109564 single-cell data"  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                        ↓                              │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Claude detects: bioinformatics task            │  │  │
│  │  │ Triggers: lobster skill (via SKILL.md)         │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    Lobster CLI (Skill)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Execution: lobster query --session-id latest          │  │
│  │            "Analyze GSE109564 single-cell data"       │  │
│  └──────────────────────────────────────────────────────┘  │
│                        ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Multi-Agent Orchestration                     │  │
│  │  • Supervisor → research_agent (download)             │  │
│  │  • data_expert (load dataset)                         │  │
│  │  • singlecell_expert (QC, clustering)                 │  │
│  │  • visualization_expert (UMAP plots)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                        ↓                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Output: .lobster_workspace/                           │  │
│  │  ├── geo_gse109564_clustered.h5ad                     │  │
│  │  ├── umap_clusters.html                               │  │
│  │  ├── qc_metrics.png                                   │  │
│  │  └── session_latest.json (for continuity)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                 Results back to Claude Code                  │
│  • Natural language summary of analysis                      │
│  • File paths to generated outputs                           │
│  • Next step suggestions                                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. **Claude Code installed** (version 1.0+)
2. **Lobster AI installed**:
   ```bash
   uv pip install lobster-ai
   lobster init  # Configure LLM provider
   ```

### Skill Installation

**Option 1: Automated install script (recommended)**

```bash
curl -fsSL https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/install.sh | bash
```

**Option 2: Manual installation**

```bash
# Create skills directory if it doesn't exist
mkdir -p ~/.claude/skills/

# Download the skill definition
curl -o ~/.claude/skills/lobster-bioinformatics.md \
  https://raw.githubusercontent.com/the-omics-os/lobster-local/main/claude-skill/SKILL.md

# Verify installation
ls ~/.claude/skills/
```

**Option 3: Clone and symlink (for development)**

```bash
# Clone repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# Symlink skill file
mkdir -p ~/.claude/skills/
ln -s "$(pwd)/claude-skill/SKILL.md" ~/.claude/skills/lobster-bioinformatics.md
```

### Verification

Test the installation:

```bash
# 1. Verify Lobster is configured
lobster config-test --json

# 2. Test skill in Claude Code
# In your IDE terminal, ask Claude Code:
# "Use Lobster to search PubMed for CRISPR papers"
```

Expected output from `config-test`:
```json
{
  "valid": true,
  "checks": {
    "llm_provider": {"status": "pass", "provider": "bedrock"},
    "ncbi_api": {"status": "pass", "has_key": true},
    "workspace": {"status": "pass", "path": "/path/to/.lobster_workspace"}
  }
}
```

## Usage Patterns

### 1. Basic Analysis Request

**User input to Claude Code:**
```
"Download GSE109564 and perform single-cell QC analysis"
```

**What happens:**
1. Claude Code detects bioinformatics task (keywords: GEO dataset, single-cell, QC)
2. Invokes Lobster skill: `lobster query "Download GSE109564 and perform single-cell QC analysis"`
3. Lobster orchestrates:
   - `research_agent` validates GSE109564
   - `data_expert` downloads dataset
   - `singlecell_expert` runs QC pipeline
4. Returns summary + file paths to Claude Code
5. Claude Code presents results in natural language

### 2. Multi-Turn Conversation (Session Continuity)

**Turn 1:**
```
"Search PubMed for CRISPR screens in cancer and extract GEO datasets"
```
**Lobster execution:** `lobster query --session-id "crispr_project" "..."`

**Turn 2:**
```
"Download the first dataset from that search"
```
**Lobster execution:** `lobster query --session-id "crispr_project" "Download the first dataset"`

**Turn 3:**
```
"Cluster it and show me UMAP"
```
**Lobster execution:** `lobster query --session-id "crispr_project" "Cluster and show UMAP"`

Session files stored in: `.lobster_workspace/session_crispr_project.json`

### 3. Workspace-Based Project Organization

**Project 1: Cancer research**
```
# Claude Code request:
"With workspace ~/cancer-project, download GSE163913 and cluster"
```
**Lobster:** `lobster query --workspace ~/cancer-project --session-id latest "..."`

**Project 2: Immunology (separate context)**
```
# Claude Code request:
"With workspace ~/immuno-project, analyze T cell datasets"
```
**Lobster:** `lobster query --workspace ~/immuno-project --session-id latest "..."`

### 4. Code Integration

Claude Code can read Lobster outputs and generate downstream code:

**User:**
```
"Download GSE109564, cluster cells, then write Python code to load
the clustered data and perform custom analysis"
```

**Claude Code workflow:**
1. Delegates to Lobster: Downloads + clusters dataset
2. Reads output: `.lobster_workspace/geo_gse109564_clustered.h5ad`
3. Generates Python code:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad(".lobster_workspace/geo_gse109564_clustered.h5ad")
   # Your custom analysis here
   ```

## Configuration

### Skill Metadata (SKILL.md)

The skill is defined by `claude-skill/SKILL.md` in the repository:

```yaml
---
name: lobster-bioinformatics
description: Run bioinformatics analyses using Lobster AI - single-cell RNA-seq,
  bulk RNA-seq, literature mining, dataset discovery, quality control, and
  visualization. Use when analyzing genomics data, searching for papers/datasets,
  or working with H5AD, CSV, GEO/SRA accessions, or biological data.
---
```

### Trigger Keywords

Claude Code automatically uses Lobster when detecting:
- **Data types**: H5AD, CSV, Excel, 10X MTX, GEO/SRA accessions
- **Operations**: QC, clustering, differential expression, cell type annotation
- **Domains**: Single-cell, bulk RNA-seq, proteomics, literature mining
- **Databases**: PubMed, PMC, GEO, SRA, ENA, PRIDE, MASSive

### Environment Variables

Lobster inherits environment from parent shell:

```bash
# Required (one of):
export ANTHROPIC_API_KEY=sk-ant-...
export AWS_BEDROCK_ACCESS_KEY=...
export OLLAMA_BASE_URL=http://localhost:11434

# Optional:
export NCBI_API_KEY=...              # Higher rate limits for PubMed
export LOBSTER_WORKSPACE=/shared/ws  # Custom workspace location
export LOBSTER_LLM_PROVIDER=ollama   # Force specific provider
```

## Pre-Flight Check

**Before using Lobster via Claude Code, verify configuration:**

```bash
lobster config-test --json
```

**Common issues:**

| Error | Solution |
|-------|----------|
| No LLM provider configured | Run `lobster init` or set API keys |
| Ollama server not accessible | Start Ollama: `ollama serve` |
| Ollama: No models installed | Install: `ollama pull gpt-oss:20b` |
| Anthropic/Bedrock API error | Verify API key validity in `.env` |
| NCBI API not configured | Add `NCBI_API_KEY` to `.env` (optional) |
| Workspace not writable | Check directory permissions |

## Advanced Features

### 1. Reasoning Mode for Complex Tasks

Claude Code can request detailed reasoning:

```bash
# Claude internally calls:
lobster query --reasoning "Perform complex multi-step analysis"
```

Shows agent decision-making process for debugging.

### 2. Reproducible Notebook Export

```bash
# User request:
"Export this analysis pipeline as a Jupyter notebook"

# Lobster creates:
.lobster_workspace/pipeline_20241208.ipynb  # Papermill-compatible
```

### 3. Provider Switching

If multiple LLM providers configured:

```bash
# Claude Code can request:
"Use Ollama for this expensive analysis"

# Translates to:
lobster query --provider ollama "..."
```

### 4. Parallel Project Management

```bash
# Project A: Cancer genomics
lobster query --workspace ~/projectA --session-id "cancer_1" "..."

# Project B: Immunology
lobster query --workspace ~/projectB --session-id "immuno_1" "..."

# Projects maintain separate:
# - Session history (conversation context)
# - Workspaces (data + outputs)
# - Provenance tracking
```

## Use Cases

### 1. Exploratory Data Analysis

**Workflow:**
```
User → Claude Code: "Download GSE163913, run QC, cluster, and show me top marker genes"

Claude → Lobster: Multi-agent pipeline
├─ research_agent: Validate GSE163913
├─ data_expert: Download dataset
├─ singlecell_expert: QC → clustering → markers
└─ visualization_expert: UMAP + marker plots

Lobster → Claude: Summary + file paths
Claude → User: "Analysis complete! Found 12 clusters with 500 marker genes.
                 See: .lobster_workspace/markers.csv"
```

### 2. Literature Mining → Data Analysis Pipeline

**Workflow:**
```
Step 1: "Search PubMed for CRISPR screens in melanoma"
Step 2: "Extract all GEO dataset IDs from those papers"
Step 3: "Check which datasets have cell_type and treatment metadata"
Step 4: "Download the best one and analyze"
```

All coordinated by Claude Code + Lobster with session continuity.

### 3. Reproducible Research

**Scenario:** Publish analysis pipeline with paper

```
User: "Analyze GSE123456 with standard pipeline, then export as notebook"

Results:
├─ geo_gse123456_analyzed.h5ad      # Data
├─ analysis_report.html              # Interactive results
├─ pipeline_20241208.ipynb           # Reproducible notebook
└─ provenance.json                   # W3C-PROV compliant tracking
```

Notebook can be shared for peer review/reproduction.

### 4. Code Generation + Analysis Integration

**User:**
```
"Download GSE109564, cluster cells, then write a Python script to
identify marker genes for cluster 3 and export to CSV"
```

**Claude Code workflow:**
1. Delegates download + clustering to Lobster
2. Reads output file path from Lobster response
3. Generates Python script:
   ```python
   import scanpy as sc
   import pandas as pd

   adata = sc.read_h5ad(".lobster_workspace/geo_gse109564_clustered.h5ad")

   # Filter cluster 3
   cluster_3 = adata[adata.obs['cluster'] == '3']

   # Find markers
   sc.tl.rank_genes_groups(cluster_3, groupby='cluster', method='wilcoxon')
   markers = sc.get.rank_genes_groups_df(cluster_3, group='3')

   # Export
   markers.to_csv('cluster_3_markers.csv', index=False)
   ```

## Troubleshooting

### Skill Not Detected

**Symptom:** Claude Code doesn't invoke Lobster for bioinformatics tasks

**Solutions:**
1. Verify skill file exists: `ls ~/.claude/skills/lobster-bioinformatics.md`
2. Check file format (must have YAML frontmatter)
3. Restart Claude Code
4. Test with explicit request: "Use the Lobster skill to analyze data"

### Command Not Found

**Symptom:** Claude Code reports "lobster: command not found"

**Solutions:**
1. Verify installation: `which lobster`
2. Install: `uv pip install lobster-ai`
3. Ensure Lobster is in PATH for Claude Code's shell
4. Add to `~/.bashrc` or `~/.zshrc`:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

### Session Continuity Issues

**Symptom:** Follow-up questions lose context

**Solution:** Use consistent workspace + session ID:
```bash
# Initial query
lobster query --workspace ~/project --session-id "project_1" "..."

# Follow-ups must use same session
lobster query --workspace ~/project --session-id "project_1" "..."
```

### Rate Limit Errors

**Symptom:** "Rate limit exceeded" from LLM provider

**Solutions:**
1. Switch to local Ollama: `export LOBSTER_LLM_PROVIDER=ollama`
2. Install Ollama model: `ollama pull llama3:8b-instruct-q8_0`
3. Wait 60 seconds and retry
4. Use AWS Bedrock for higher limits

## Best Practices

### 1. Use Descriptive Session IDs

❌ **Bad:**
```
--session-id "session1"
```

✅ **Good:**
```
--session-id "gse109564_tcell_analysis"
```

### 2. Organize by Workspace

Structure projects with dedicated workspaces:
```
~/bioinformatics/
├── cancer_project/
│   └── .lobster_workspace/
├── immunology_project/
│   └── .lobster_workspace/
└── methods_paper_reproduction/
    └── .lobster_workspace/
```

### 3. Chain Operations Efficiently

❌ **Bad** (separate queries):
```
"Download GSE109564"
"Run QC on it"
"Cluster the cells"
"Generate UMAP"
```

✅ **Good** (single query):
```
"Download GSE109564, run QC, cluster cells, and generate UMAP visualization"
```

### 4. Verify Outputs

Always check generated files:
```bash
ls -lh .lobster_workspace/
cat .lobster_workspace/analysis_summary.json
```

### 5. Use Reasoning Mode for Complex Tasks

For multi-step or ambiguous requests:
```bash
lobster query --reasoning "Design and execute a complete single-cell analysis pipeline"
```

## Limitations

- **LLM dependency**: Requires active LLM provider (Ollama/Anthropic/Bedrock)
- **Large datasets**: Performance depends on system resources (>100K cells may be slow)
- **Premium features**: Some agents require paid subscription (proteomics, metadata assistant)
- **Rate limits**: Cloud LLM providers have API rate limits
- **No GUI**: CLI-only interface (outputs are files, not interactive)

## Related Documentation

- [Installation Guide](02-installation.md)
- [Configuration](03-configuration.md)
- [CLI Commands](05-cli-commands.md)
- [Examples Cookbook](27-examples-cookbook.md)
- [Agent System Overview](19-agent-system.md)
- [Data Management](20-data-management.md)
- [Troubleshooting](28-troubleshooting.md)

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Lobster AI | 0.3.1.4 | Latest |
| Claude Code | 1.0 | Latest |
| Python | 3.11 | 3.12+ |
| Ollama (if using) | 0.1.0 | Latest |

## Community & Support

- **GitHub Issues**: https://github.com/the-omics-os/lobster-local/issues
- **Wiki**: https://github.com/the-omics-os/lobster-local/wiki
- **Examples**: https://github.com/the-omics-os/lobster-local/wiki/27-examples-cookbook

## Contributing

Skill improvements welcome! See the [Claude Code skill file](https://github.com/the-omics-os/lobster-local/blob/main/claude-skill/SKILL.md) on GitHub.
