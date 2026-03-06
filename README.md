<div align="center">
  <img alt="Lobster AI Logo" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/lobster-logo.png" width="200">
</div>

<br/>

<div align="center">
  <b>Open-source multi-agent bioinformatics engine. Describe your analysis in natural language.</b>
</div>

<br/>

<div align="center">
  <table border="0" cellspacing="6" cellpadding="0">
    <tr>
      <td><a href="https://docs.omics-os.com"><img src="https://img.shields.io/badge/docs-omics--os.com-black?style=for-the-badge&logo=readthedocs" alt="Docs"></a></td>
      <td><a href="https://app.omics-os.com"><img src="https://img.shields.io/badge/cloud-Omics--OS-blue?style=for-the-badge&logo=googlecloud" alt="Cloud"></a></td>
      <td><a href="https://pypi.org/project/lobster-ai/"><img src="https://img.shields.io/badge/PyPI-lobster--ai-black?style=for-the-badge&logo=pypi" alt="PyPI"></a></td>
      <td><a href="https://discord.gg/omics-os"><img src="https://img.shields.io/badge/Discord-Join_Community-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a></td>
    </tr>
  </table>
</div>

<br/>

<div align="center">
  <table border="0" cellspacing="0" cellpadding="8">
    <tr>
      <td><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-1.svg" width="260" alt="Testimonial 1"></td>
      <td><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-2.svg" width="260" alt="Testimonial 2"></td>
      <td><img src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/testimonial-3.svg" width="260" alt="Testimonial 3"></td>
    </tr>
  </table>
</div>

<br/>

<div align="center">
  🌤️ <b>Want to skip the setup?</b> Run massive multi-omics pipelines instantly on <b><a href="https://app.omics-os.com">Omics-OS Cloud</a></b>!
</div>

---

# 🧑‍🔬 Human Quickstart

**1. Install Lobster AI (macOS/Linux):**
```bash
curl -fsSL https://install.lobsterbio.com | bash
```
*(Windows users: `irm https://install.lobsterbio.com/windows | iex`)*

**2. Configure your LLM (Anthropic, Gemini, local Ollama, etc.):**
```bash
lobster init
```

**3. Run your first analysis:**
```bash
lobster query "Download GSE109564 and cluster the cells"
# Or run interactively: lobster chat
```

<br/>

# 🤖 For AI Coding Agents

Teach your coding agent (Claude Code, Cursor, Gemini) to use and extend Lobster AI instantly:
```bash
curl -fsSL https://skills.lobsterbio.com | bash
```
*Installs the `lobster-use` and `lobster-dev` skills so your AI knows our entire 10-package architecture.*

<br/>

# 🎬 Watch it Work

### 🧬 Single-Cell Transcriptomics
**Task:** *"Download GSE109564, run QC, and cluster cells."*
```bash
$ lobster query "Download GSE109564, run QC, and cluster cells"
[Supervisor] Routing to Transcriptomics Expert...
[Transcriptomics] Delegating download of 'GSE109564' to Data Expert...
[Data Expert] Download complete. Modality 'GSE109564' loaded.
[Transcriptomics] Running QC (calculating MT fraction, filtering cells/genes)...
[Transcriptomics] Normalizing and finding highly variable genes...
[Transcriptomics] Running PCA, Neighborhood graph, and UMAP...
[Transcriptomics] Clustering via Leiden (resolution=1.0)...
✅ Analysis complete. UMAP plot generated and saved to workspace.
```

### 🔬 Mass Spec Proteomics
**Task:** *"Import MaxQuant data, perform batch correction, and select biomarker panels."*
```bash
$ lobster query "Import MaxQuant data, perform batch correction, and select biomarker panels"
[Supervisor] Routing to Proteomics Expert...
[Proteomics] Parsing proteinGroups.txt and experimental design...
[Proteomics] Filtering contaminants and reverse hits...
[Proteomics] Performing median normalization and ComBat batch correction...
[Proteomics] Delegating to Biomarker Discovery Expert...
[Biomarker Expert] Running LASSO stability selection with nested cross-validation...
✅ Complete. 12 robust biomarker candidates identified (Stability score > 0.8).
```

### 📚 Automated Literature Discovery
**Task:** *"Search PubMed for CRISPR studies in 2024 and download the top 3 datasets."*
```bash
$ lobster query "Search PubMed for CRISPR studies in 2024 and download the top 3 datasets."
[Supervisor] Routing to Research Agent...
[Research] Searching PubMed: "(CRISPR[Title/Abstract]) AND 2024[Date - Publication]"
[Research] Extracting GEO accessions from top 10 relevant papers...
[Research] Found GSE251842, GSE252910, GSE260124.
[Research] Delegating batch download to Data Expert...
✅ Datasets downloaded and harmonized into AnnData objects.
```

<br/>

# 🧠 The Architecture

Lobster isn't just a chatbot; it's a modular ecosystem of **22 specialist agents across 10 packages**.
* **Your machine, your data:** Patient data never leaves your hardware.
* **Tool calls, not token dreams:** Agents execute real, validated Python packages (Scanpy, PyDESeq2).
* **100% Reproducible:** W3C-PROV tracking and automatic Jupyter notebook exports.

<div align="center">
  <img alt="Ecosystem Topology" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/architecture-topology.svg" width="88%">
  <br/><br/>
  <img alt="Core Architecture" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/architecture-core.svg" width="88%">
</div>

<br/>

# 🛠️ Build Your Own Agent

The `lobster-dev` skill gives your coding assistant (Claude Code, Gemini CLI, Cursor) deep knowledge of how Lobster agents are structured. Describe the biological domain you need — it scaffolds the package, wires the tools, writes the tests, and registers the agent.

<div align="center">
  <table border="0" cellspacing="0" cellpadding="12">
    <tr>
      <td width="50%" valign="top" align="center">
        <b>1. The Request</b><br/><br/>
        <img alt="Claude Terminal" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/terminal-preview.svg" width="100%">
      </td>
      <td width="50%" valign="top" align="center">
        <b>2. The Result</b><br/><br/>
        <img alt="Hackability Preview" src="https://raw.githubusercontent.com/the-omics-os/lobster/main/docs/assets/hackability-preview.svg" width="100%">
      </td>
    </tr>
  </table>
</div>

<br/>

# ❓ Deep Dives & FAQ

<details>
<summary><b>What omics domains are supported?</b></summary>

**Transcriptomics**
- Single-cell RNA-seq: QC, doublet detection (Scrublet), batch integration (Harmony/scVI), clustering, cell type annotation, trajectory inference (DPT/PAGA)
- Bulk RNA-seq: Salmon/kallisto/featureCounts import, sample QC, batch detection, normalization (DESeq2/VST/CPM), DE with PyDESeq2, GSEA, publication-ready export

**Genomics**
- GWAS: VCF/PLINK import, LD pruning, kinship, association testing, result clumping
- Clinical: variant annotation (VEP), gnomAD frequencies, ClinVar pathogenicity, variant prioritization

**Proteomics**
- Mass spec: MaxQuant/DIA-NN/Spectronaut import, PTM analysis, peptide-to-protein rollup, batch correction
- Affinity: Olink NPX/SomaScan ADAT/Luminex MFI import, LOD quality, bridge normalization
- Downstream: GO/Reactome/KEGG enrichment, kinase enrichment (KSEA), STRING PPI, biomarker panel selection

**Metabolomics**
- LC-MS, GC-MS, NMR with auto-detection
- QC (RSD, TIC), filtering, imputation, normalization (PQN/TIC/IS)
- PCA, PLS-DA, OPLS-DA, m/z annotation (HMDB/KEGG), lipid class analysis

**Machine Learning**
- Feature selection (stability selection, LASSO, variance filter)
- Survival analysis (Cox models, Kaplan-Meier, risk stratification)
- Cross-validation, SHAP interpretability, multi-omics integration (MOFA)

**Research & Metadata**
- Literature discovery (PubMed, PMC, GEO, PRIDE, MetaboLights)
- Dataset download orchestration, metadata harmonization, sample filtering
</details>

<details>
<summary><b>Which LLMs can I use?</b></summary>

Lobster supports 5 LLM providers. Configure via `lobster init` or environment variables.

| Provider | Type | Setup | Use Case |
|----------|------|-------|----------|
| **Ollama** | Local | `ollama pull gpt-oss:20b` | Privacy, zero cost, offline |
| **Anthropic** | Cloud | API key | Fastest, best quality |
| **AWS Bedrock** | Cloud | AWS credentials | Enterprise, compliance |
| **Google Gemini** | Cloud | Google API key | Multimodal, long context |
| **Azure AI** | Cloud | Endpoint + credential | Enterprise Azure |
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

Create custom agents for any domain. Agents plug in via Python entry points — discovered automatically, no core changes needed.

Install the **lobster-dev** skill to teach your coding agent the full architecture:

```bash
curl -fsSL https://skills.lobsterbio.com | bash
```

Then ask your coding agent: *"Create a Lobster agent for [your domain]"* — it knows the package structure, AGENT_CONFIG pattern, factory function, tool design, testing, and the 28-step checklist.
</details>

<br/>

<div align="center">
  <b>Built to accelerate multi-omics research.</b><br/><br/>
  <a href="https://omics-os.com">Omics-OS</a> &nbsp;·&nbsp; <a href="https://lobsterbio.com">Lobster AI</a> &nbsp;·&nbsp; <a href="https://docs.omics-os.com">Docs</a>
</div>
