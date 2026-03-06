<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; background: transparent;">
    <tr>
      <td valign="middle" style="border: none; background: transparent;">
        <img alt="Lobster AI Logo" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/lobster-logo.png" width="80">
      </td>
      <td valign="middle" align="left" style="border: none; background: transparent; padding-left: 18px;">
        <b>Open-source multi-agent bioinformatics engine.<br/>Describe your analysis in natural language.</b>
      </td>
    </tr>
  </table>
</div>

<br/>

<div align="center">
  <table border="0" cellspacing="6" cellpadding="0" style="border: none; background: transparent;">
    <tr>
      <td style="border: none; background: transparent;"><a href="https://docs.omics-os.com"><img src="https://img.shields.io/badge/docs-omics--os.com-black?style=for-the-badge&logo=readthedocs" alt="Docs"></a></td>
      <td style="border: none; background: transparent;"><a href="https://app.omics-os.com"><img src="https://img.shields.io/badge/cloud-Omics--OS-blue?style=for-the-badge&logo=googlecloud" alt="Cloud"></a></td>
      <td style="border: none; background: transparent;"><a href="https://pypi.org/project/lobster-ai/"><img src="https://img.shields.io/badge/PyPI-lobster--ai-black?style=for-the-badge&logo=pypi" alt="PyPI"></a></td>
    </tr>
  </table>
</div>

<br/>

<div align="center">
  <table border="0" cellspacing="0" cellpadding="8" style="border: none; background: transparent;">
    <tr>
      <td style="border: none; background: transparent;"><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-1.svg" width="260" alt="Testimonial 1"></td>
      <td style="border: none; background: transparent;"><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-2.svg" width="260" alt="Testimonial 2"></td>
      <td style="border: none; background: transparent;"><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-3.svg" width="260" alt="Testimonial 3"></td>
    </tr>
  </table>
</div>

<br/>

---

# Quickstart

**1. Install Lobster AI (macOS/Linux):**
```bash
curl -fsSL https://install.lobsterbio.com | bash
```
*(Windows users: `irm https://install.lobsterbio.com/windows | iex`)*

**2. Configure your LLM (Anthropic, Gemini, local Ollama, etc.):**
```bash
lobster init
```

<details>
<summary><b>Watch: installation & init walkthrough</b></summary>
<br/>
<div align="center">
  <img alt="Installation and Init" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/recordings/installation_and_init.gif" width="720">
</div>
</details>

**3. Start an interactive session:**
```bash
lobster chat
```
Then describe your analysis in plain language:
```
> Search PubMed for single-cell CRISPR screens in T cells from 2023–2024,
  download the most cited dataset, run QC, integrate batches with Harmony,
  cluster the cells, annotate cell types, and export a reproducible notebook.
```

<br/>

# 🤖 For AI Coding Agents

Install skills that give Claude Code, Cursor, or Gemini CLI deep knowledge of the Lobster architecture:
```bash
curl -fsSL https://skills.lobsterbio.com | bash
```
This installs `lobster-use` (analysis workflows) and `lobster-dev` (agent development). With these loaded, your coding agent understands the full 10-package structure, tool patterns, entry point registration, and AQUADIF contract — without needing to read source code manually.

**Scaffold a new agent package from the command line:**
```bash
lobster scaffold agent \
  --name epigenomics_expert \
  --display-name "Epigenomics Expert" \
  --description "ATAC-seq, ChIP-seq, and DNA methylation analysis" \
  --tier free
```
Generates a complete, contract-compliant package: `pyproject.toml`, entry point wiring, tool stubs with AQUADIF metadata, and contract tests. Then point your coding agent at the generated scaffolding and ask it to implement the domain logic.

<br/>

# Use Cases

End-to-end walkthroughs across omics domains:

<table width="100%" style="border: none; background: transparent;">
  <thead>
    <tr>
      <th align="left">Domain</th>
      <th align="left">Case Study</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Single-Cell Transcriptomics</td><td><a href="https://docs.omics-os.com/docs/case-studies/transcriptomics/">Cell clustering, annotation &amp; trajectory inference</a></td></tr>
    <tr><td>CML Drug Resistance</td><td><a href="https://docs.omics-os.com/docs/case-studies/cml-resistance/">Resistance mechanism discovery from scRNA-seq</a></td></tr>
    <tr><td>Drug Discovery</td><td><a href="https://docs.omics-os.com/docs/case-studies/drug-discovery/">Target identification &amp; compound prioritization</a></td></tr>
    <tr><td>Clinical Genomics</td><td><a href="https://docs.omics-os.com/docs/case-studies/genomics/">Variant annotation &amp; GWAS analysis</a></td></tr>
    <tr><td>Mass Spec Proteomics</td><td><a href="https://docs.omics-os.com/docs/case-studies/proteomics/">Biomarker panel selection from DIA-NN data</a></td></tr>
    <tr><td>Literature Mining</td><td><a href="https://docs.omics-os.com/docs/case-studies/research/">Automated dataset discovery from PubMed</a></td></tr>
    <tr><td>Multi-Omics ML</td><td><a href="https://docs.omics-os.com/docs/case-studies/machine-learning/">Feature selection &amp; survival analysis</a></td></tr>
  </tbody>
</table>

<br/>

# 🧠 Architecture

Lobster AI is a multi-agent system: **22 specialist agents across 10 installable packages**, orchestrated by a LangGraph supervisor. Each agent owns a specific omics domain and calls validated scientific libraries directly — no code generation, no hallucinated results.

* **Local execution:** All analysis runs on your machine. Patient data never leaves your hardware.
* **Scientific libraries:** Agents call Scanpy, PyDESeq2, Harmony, and others via tool functions — not by generating scripts.
* **W3C-PROV provenance:** Every analysis step is tracked and exportable as a reproducible Jupyter notebook.

<div align="center">
  <img alt="Ecosystem Topology" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/architecture-topology.svg" width="88%">
  <br/><br/>
  <img alt="Core Architecture" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/architecture-core.svg" width="88%">
</div>

<br/>

# 🛠️ Build Your Own Agent

New agents are standalone packages that plug into Lobster via Python entry points. The `lobster-dev` skill loads the full architecture reference into your coding agent (Claude Code, Gemini CLI, Cursor) — package layout, tool patterns, AQUADIF contract, and test fixtures. Use `lobster scaffold` to generate the package skeleton, then let your coding agent implement the domain logic.

<div align="center">
  <table border="0" cellspacing="0" cellpadding="12">
    <tr>
      <td valign="top" align="center">
        <b>1. The Request</b><br/><br/>
        <img alt="Claude Terminal" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/terminal-preview.svg" width="380">
      </td>
      <td valign="top" align="center">
        <b>2. The Result</b><br/><br/>
        <img alt="Hackability Preview" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/hackability-preview.svg" width="380">
      </td>
    </tr>
  </table>
</div>

<br/>

# FAQ

<details>
<summary><b>What omics domains are supported?</b></summary>

| Domain | Input Formats | Key Capabilities |
|--------|--------------|-----------------|
| **Single-Cell RNA-seq** | AnnData, 10x, h5ad | QC, doublet detection (Scrublet), batch integration (Harmony/scVI), clustering, cell type annotation, trajectory inference (DPT/PAGA) |
| **Bulk RNA-seq** | Salmon, kallisto, featureCounts | Sample QC, normalization (DESeq2/VST/CPM), differential expression (PyDESeq2), GSEA, publication-ready export |
| **Genomics** | VCF, PLINK | GWAS, LD pruning, kinship estimation, association testing, result clumping |
| **Clinical Genomics** | VCF, ClinVar, gnomAD | Variant annotation (VEP), pathogenicity scoring, clinical variant prioritization |
| **Mass Spec Proteomics** | MaxQuant, DIA-NN, Spectronaut | PTM analysis (phospho/acetyl/ubiquitin), peptide-to-protein rollup, batch correction |
| **Affinity Proteomics** | Olink NPX, SomaScan ADAT, Luminex MFI | LOD quality filtering, bridge normalization, cross-platform concordance |
| **Proteomics Downstream** | Any loaded proteomics modality | GO/Reactome/KEGG enrichment, kinase enrichment (KSEA), STRING PPI, biomarker panel selection (LASSO/Boruta) |
| **Metabolomics** | LC-MS, GC-MS, NMR | QC (RSD/TIC), imputation, normalization (PQN/TIC/IS), PCA, PLS-DA, OPLS-DA, m/z annotation (HMDB/KEGG), lipid class analysis |
| **Machine Learning** | Any modality | Feature selection (stability/LASSO/variance), survival analysis (Cox/KM), cross-validation, SHAP, multi-omics integration (MOFA) |
| **Research & Data Access** | — | PubMed/GEO/PRIDE/MetaboLights search, dataset download orchestration, metadata harmonization |
</details>

<details>
<summary><b>Which LLMs can I use?</b></summary>

Configure via `lobster init` or environment variables. All providers use the same agent interface.

| Provider | Type | Setup | Notes |
|----------|------|-------|-------|
| **Anthropic** | Cloud | API key | Claude models — recommended default |
| **Ollama** | Local | `ollama pull <model>` | Fully offline, no data leaves the machine |
| **OpenRouter** | Cloud | API key | Access 200+ models via a single endpoint |
| **Google Gemini** | Cloud | Google API key | Long context window |
| **AWS Bedrock** | Cloud | AWS credentials | Enterprise compliance, IAM-based auth |
| **Azure AI** | Cloud | Endpoint + credential | Azure-hosted deployments |
</details>

<details>
<summary><b>Pipeline export and slash commands</b></summary>

```bash
lobster chat
> /pipeline export         # Export reproducible Jupyter notebook
> /pipeline list           # List exported pipelines
> /pipeline run analysis.ipynb geo_gse109564
> /data                    # Show loaded datasets
> /status                  # Session info
> /help                    # All commands
```
</details>

<details>
<summary><b>Advanced installation (Windows, pip)</b></summary>

**Windows** (PowerShell):
```powershell
irm https://install.lobsterbio.com/windows | iex
```

**uv** (recommended manual install):
```bash
uv tool install 'lobster-ai[full,anthropic]'
lobster init
```

**pip**:
```bash
pip install 'lobster-ai[full]'
lobster init
```

**Upgrade**:
```bash
uv tool upgrade lobster-ai    # uv
pip install -U lobster-ai      # pip
```
</details>

<details>
<summary><b>How do I build my own agent?</b></summary>

Agents are standalone Python packages that register via PEP 517 entry points. No changes to core required — Lobster discovers them automatically at startup.

**1. Scaffold the package:**
```bash
lobster scaffold agent \
  --name my_domain_expert \
  --display-name "My Domain Expert" \
  --description "Analysis for [your domain]" \
  --tier free
```

**2. Implement your tools** in the generated `tools/` directory. Each tool must declare AQUADIF metadata:
```python
@tool
def run_analysis(modality_name: str) -> str:
    """Run domain-specific analysis on a loaded modality."""
    ...

run_analysis.metadata = {"categories": ["ANALYZE"], "provenance": True}
run_analysis.tags = ["ANALYZE"]
```

**3. Validate the package structure** before wiring:
```bash
lobster validate-plugin ./my-domain-package/
```

**4. Install and test:**
```bash
uv pip install -e ./my-domain-package/
pytest -m contract  # runs all AQUADIF contract checks
```

Install the `lobster-dev` skill to give your coding agent the complete reference — package layout, `AGENT_CONFIG` pattern, factory function signature, tool design rules, and the full validation checklist:
```bash
curl -fsSL https://skills.lobsterbio.com | bash
```
</details>

<br/>

<div align="center">
  <b>Multi-omics data infrastructure for foundation models &amp; biotech.</b><br/><br/>
  <a href="https://omics-os.com">Omics-OS</a> &nbsp;·&nbsp; <a href="https://lobsterbio.com">Lobster AI</a> &nbsp;·&nbsp; <a href="https://docs.omics-os.com">Docs</a>
</div>
