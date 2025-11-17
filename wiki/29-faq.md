# Frequently Asked Questions (FAQ)

This comprehensive FAQ addresses common questions about Lobster AI, covering everything from basic usage to advanced features and troubleshooting.

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation-setup)
3. [Data & File Formats](#data-file-formats)
4. [Analysis Capabilities](#analysis-capabilities)
5. [Performance & Scalability](#performance-scalability)
6. [Visualization & Output](#visualization-output)
7. [Cloud Integration](#cloud-integration)
8. [Comparison with Other Tools](#comparison-with-other-tools)
9. [Technical Details](#technical-details)
10. [Troubleshooting](#troubleshooting)

---

## General Questions

### Q: What is Lobster AI?

**A:** Lobster AI is a professional multi-agent bioinformatics analysis platform that combines specialized AI agents with proven scientific tools to analyze complex multi-omics data. Users interact through natural language to perform RNA-seq, proteomics, and multi-omics analysis without needing to write code or manage complex command-line tools.

### Q: Who is Lobster AI designed for?

**A:** Lobster AI is designed for:
- **Bioinformatics researchers** analyzing RNA-seq, proteomics, and multi-omics data
- **Computational biologists** seeking intelligent analysis workflows
- **Life science teams** requiring reproducible, publication-ready results
- **Students & educators** learning modern bioinformatics approaches
- **Data scientists** working with genomic data who want to focus on insights rather than tool management

### Q: What makes Lobster AI different from other bioinformatics tools?

**A:** Key differentiators include:
- **Natural language interface** - Describe analyses in plain English
- **Specialized AI agents** - Expert agents for different omics platforms
- **Professional CLI** - Rich interface with orange branding and real-time monitoring
- **Publication-ready output** - High-quality visualizations and statistical reports
- **Cloud/local hybrid** - Seamless switching between local and cloud processing
- **Complete provenance** - W3C-PROV compliant analysis history tracking

### Q: Is Lobster AI free to use?

**A:** Yes, Lobster AI is open source under the Apache 2.0 License. You can use it freely for academic and commercial projects. Cloud services may require a subscription for enhanced computing resources.

### Q: Can I use Lobster AI for commercial research?

**A:** Yes, the Apache 2.0 License allows commercial use. For enterprise deployments, custom integrations, or dedicated support, contact the enterprise team.

---

## Installation & Setup

### Q: What are the system requirements?

**A:** Minimum requirements:
- **Python**: 3.12 or higher
- **Memory**: 4GB RAM (8GB+ recommended for larger datasets)
- **Storage**: 2GB for installation, additional space for data and results
- **OS**: Linux, macOS, or Windows with WSL2
- **Terminal**: Modern terminal with Unicode support for enhanced CLI features

### Q: How do I install Lobster AI?

**A:** Installation options:

```bash
# Option 1: Quick install from GitHub
git clone https://github.com/the-omics-os/lobster.git
cd lobster && make install

# Option 2: Development install
make dev-install

# Option 3: Clean installation (removes existing environment)
make clean-install

# Start using Lobster
lobster chat
```

### Q: What API keys do I need?

**A:** Required API keys:
- **Anthropic API key** - OR
- **AWS Bedrock credentials** - For Claude models (required)

Optional API keys:
- **NCBI API key** - For faster GEO downloads (recommended)
- **Lobster Cloud API key** - For cloud processing (optional)

Set these in your `.env` file:
```bash
OPENAI_API_KEY=your-openai-key #TODO future support
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key
```

### Q: Can I use Lobster AI without cloud services?

**A:** Yes! Lobster AI works completely offline in local mode. You only need API keys for the LLM models (OpenAI/AWS), not for cloud processing. All analysis runs on your local machine.

### Q: How do I update Lobster AI?

**A:** Update methods:
```bash
# For git installations
git pull origin main
make install

# For pip installations (when available)
pip install --upgrade lobster-ai

# Verify update
lobster --version
```

---

## Data & File Formats

### Q: What data formats does Lobster AI support?

**A:** Supported formats:

**Input formats:**
- **Single-cell**: H5AD, 10X MTX, CSV, Excel
- **Bulk RNA-seq**: CSV, TSV, Excel, raw counts
- **Proteomics**: MaxQuant output, Spectronaut, Olink NPX, generic CSV/Excel
- **Spatial**: Visium H5, H5AD with spatial coordinates

**Output formats:**
- **Data**: H5AD, CSV, Excel, HDF5
- **Visualizations**: HTML (interactive), PNG, SVG, PDF
- **Reports**: TXT, JSON, CSV

### Q: How do I load data from GEO?

**A:** Simple GEO loading:
```bash
🦞 You: "Download GSE109564 from GEO"
🦞 You: "Search GEO for single-cell datasets related to cancer"
🦞 You: "Download all datasets from series GSE123456"
```

Lobster automatically handles GEO metadata, sample information, and data formatting.

### Q: Can I analyze my own local files?

**A:** Yes! Load local files easily:
```bash
🦞 You: "Load the H5AD file from /path/to/my_data.h5ad"
🦞 You: "Load CSV count matrix from ./counts.csv with genes as rows"
🦞 You: "Load 10X data from the ./cellranger_output/ directory"
```

### Q: What's the largest dataset I can analyze?

**A:** Local analysis limits depend on your system:
- **Single-cell**: Up to 100K cells (with 16GB RAM)
- **Bulk RNA-seq**: Virtually unlimited (efficient matrix operations)
- **Proteomics**: Up to 10K samples × 10K proteins

For larger datasets:
- Use cloud processing with `LOBSTER_CLOUD_KEY`
- Apply subsampling for initial exploration
- Use chunked processing strategies

### Q: How do I handle missing values in proteomics data?

**A:** Lobster automatically handles missing values:
```bash
🦞 You: "Analyze missing value patterns in my proteomics data"
🦞 You: "Apply MNAR imputation for low-abundance proteins"
🦞 You: "Use different imputation strategies for MS vs affinity proteomics"
```

Missing value handling is platform-specific and follows best practices.

---

## Analysis Capabilities

### Q: What types of analyses can Lobster AI perform?

**A:** Analysis capabilities by data type:

**Single-cell RNA-seq:**
- Quality control and filtering
- Normalization and batch correction
- Clustering and cell type annotation
- Trajectory analysis and pseudotime
- Marker gene identification
- Pseudobulk conversion

**Bulk RNA-seq:**
- Differential expression with pyDESeq2
- Complex experimental designs with interactions
- Batch effect correction
- Pathway enrichment analysis
- Time series analysis

**Proteomics:**
- MS proteomics (DDA/DIA workflows)
- Affinity proteomics (Olink panels)
- Missing value analysis and imputation
- Differential protein expression
- Pathway and network analysis

**Multi-omics:**
- Integration across omics platforms
- Correlation analysis between RNA and protein
- Multi-omics pathway analysis
- Integrated visualizations

### Q: Can I perform custom statistical analyses?

**A:** Yes! Lobster supports:
```bash
🦞 You: "Design custom contrast matrix for my complex experimental design"
🦞 You: "Apply custom gene sets for pathway analysis"
🦞 You: "Use specific statistical methods like MAST for single-cell DE"
🦞 You: "Implement time series analysis with spline regression"
```

### Q: How accurate are the AI-generated analyses?

**A:** Lobster AI uses:
- **Professional-grade algorithms** - Same methods used in leading bioinformatics tools
- **Statistical best practices** - Appropriate multiple testing correction, normalization methods
- **Publication standards** - Following established protocols and guidelines
- **Quality validation** - 60% compliant with data quality standards, targeting 95%+

All analyses use proven scientific methods; AI provides the interface, not the statistics.

### Q: Can I reproduce analyses performed by Lobster AI?

**A:** Yes! Lobster provides:
- **Complete provenance tracking** - W3C-PROV compliant analysis history
- **Parameter logging** - All analysis settings recorded
- **Export capabilities** - Full analysis scripts and data
- **Session export** - Reproducible analysis workflows

```bash
🦞 You: "/export session"  # Export complete analysis history
🦞 You: "Generate methods description for publication"
```

### Q: How do I validate Lobster's results?

**A:** Validation strategies:
- **Cross-method validation** - Compare results using different algorithms
- **External validation** - Verify with independent datasets
- **Manual inspection** - Export data for custom analysis
- **Literature comparison** - Check against published results

```bash
🦞 You: "Compare DESeq2 results with edgeR and limma"
🦞 You: "Validate cell type annotations with reference datasets"
```

---

## Performance & Scalability

### Q: How can I speed up my analyses?

**A:** Performance optimization:
- **Use cloud processing** - Set `LOBSTER_CLOUD_KEY` for large analyses
- **Parallel processing** - Enable multi-core processing
- **Data optimization** - Filter unnecessary genes/cells early
- **Chunked processing** - Process large datasets in smaller pieces

```bash
🦞 You: "Enable parallel processing for this analysis"
🦞 You: "Use approximate methods for faster clustering"
🦞 You: "Process this dataset in cloud for better performance"
```

### Q: What should I do if analysis takes too long?

**A:** Troubleshooting slow analyses:
1. **Check progress**: Use `/progress` and `/dashboard` commands
2. **Optimize parameters**: Reduce complexity for initial exploration
3. **Use subsampling**: Test with smaller datasets first
4. **Switch to cloud**: Use cloud resources for intensive analyses

### Q: How much memory do I need for different dataset sizes?

**A:** Memory guidelines:
- **1K cells**: 1-2GB RAM
- **10K cells**: 4-8GB RAM
- **50K cells**: 16-32GB RAM
- **100K+ cells**: Cloud processing recommended

Proteomics data typically requires less memory than single-cell RNA-seq.

### Q: Can I run multiple analyses simultaneously?

**A:** Currently, Lobster processes analyses sequentially for resource management. However, you can:
- Run separate Lobster instances in different directories
- Use cloud processing for parallel workflows
- Queue multiple commands within a single session

---

## Visualization & Output

### Q: What types of visualizations does Lobster create?

**A:** Comprehensive visualization suite:

**Single-cell:**
- UMAP/tSNE plots with annotations
- Violin plots for gene expression
- Heatmaps of marker genes
- Quality control dashboards

**Bulk RNA-seq:**
- Volcano plots and MA plots
- PCA plots with experimental factors
- Expression heatmaps
- Pathway enrichment plots

**Proteomics:**
- Missing value heatmaps
- Volcano plots for differential proteins
- Protein correlation networks
- Quality control dashboards

All plots are interactive (Plotly) and publication-ready.

### Q: Can I customize the visualizations?

**A:** Yes! Customization options:
```bash
🦞 You: "Create UMAP plot with custom colors and larger point sizes"
🦞 You: "Generate high-resolution figure (300 DPI) for publication"
🦞 You: "Use specific color palette for cell type annotations"
🦞 You: "Export plot data for custom visualization in R/Python"
```

### Q: How do I export results for publication?

**A:** Publication export:
```bash
🦞 You: "/export results"  # Comprehensive export
🦞 You: "Generate publication-ready figures in SVG format"
🦞 You: "Export statistical tables with significance indicators"
🦞 You: "Create methods description for manuscript"
```

Exports include:
- High-resolution figures (SVG, PNG)
- Statistical result tables (CSV)
- Analysis parameters (JSON)
- Methods descriptions (TXT)

### Q: Can I access the raw data behind visualizations?

**A:** Absolutely! Access options:
```bash
🦞 You: "Show me the data used to create this plot"
🦞 You: "Export plot coordinates and metadata"
🦞 You: "/read plot_data.csv"  # Access underlying data
```

---

## Cloud Integration

### Q: How does cloud processing work?

**A:** Cloud integration:
- **Automatic detection** - Set `LOBSTER_CLOUD_KEY` and Lobster automatically uses cloud
- **Seamless switching** - Falls back to local if cloud unavailable
- **Same interface** - Identical user experience local vs cloud
- **Scalable resources** - Handle larger datasets without local hardware limits

### Q: Is my data secure in the cloud?

**A:** Security measures:
- **Enterprise-grade security** - Professional cloud infrastructure
- **No permanent storage** - Data processed and removed after analysis
- **Encrypted transmission** - All data transfer encrypted
- **Privacy controls** - You control data sharing and retention

### Q: When should I use cloud vs local processing?

**A:** Use cloud for:
- Large datasets (>50K cells, >10GB files)
- Memory-intensive analyses
- Long-running computations
- Collaborative projects

Use local for:
- Small to medium datasets
- Sensitive data requiring local control
- Quick exploratory analyses
- Offline environments

### Q: How do I get cloud access?

**A:** Cloud access:
1. **Request API key** - Contact info@omics-os.com
2. **Set environment variable** - `export LOBSTER_CLOUD_KEY="your-key"`
3. **Start Lobster** - Automatically detects and uses cloud mode

```bash
# Check if cloud is active
🦞 You: "/status"  # Shows "Cloud mode active" if configured
```

---

## Comparison with Other Tools

### Q: How does Lobster compare to Seurat?

**A:** Comparison with Seurat:

| Feature | Lobster AI | Seurat |
|---------|------------|--------|
| **Interface** | Natural language | R programming |
| **Learning curve** | Minimal | Steep (R knowledge required) |
| **Automation** | High (AI-guided) | Manual scripting |
| **Reproducibility** | Built-in provenance | Manual documentation |
| **Visualization** | Interactive dashboards | Static plots (customizable) |
| **Multi-omics** | Native support | Limited integration |
| **Cloud processing** | Seamless | Manual setup |

### Q: How does it compare to scanpy?

**A:** Comparison with scanpy:

| Feature | Lobster AI | scanpy |
|---------|------------|--------|
| **Language** | Natural language | Python programming |
| **Ease of use** | Beginner-friendly | Programming required |
| **Analysis breadth** | Multi-omics platform | Single-cell focused |
| **Customization** | AI-guided + exportable | Full programmatic control |
| **Performance** | Optimized + cloud | Local Python performance |
| **Documentation** | Interactive help | Manual/tutorial based |

### Q: Can I export results to use with other tools?

**A:** Yes! Export compatibility:
```bash
🦞 You: "Export data in Seurat format for R analysis"
🦞 You: "Save results as AnnData for scanpy compatibility"
🦞 You: "Export gene lists for external pathway analysis"
🦞 You: "Generate analysis scripts for manual reproduction"
```

### Q: Can I import analyses from other tools?

**A:** Import capabilities:
```bash
🦞 You: "Load Seurat object from RDS file"
🦞 You: "Import scanpy analysis from H5AD file"
🦞 You: "Load clustering results from CSV file"
```

---

## Technical Details

### Q: What AI models does Lobster use?

**A:** AI model architecture:
- **Language Models** - OpenAI GPT-4 and AWS Claude for natural language understanding
- **Agent Framework** - LangGraph for multi-agent coordination
- **Scientific Computing** - Traditional bioinformatics algorithms (no AI in statistics)
- **Quality Assurance** - AI provides interface, proven algorithms provide results

### Q: How does the multi-agent system work?

**A:** Agent architecture:
- **Supervisor Agent** - Routes queries to appropriate specialists
- **Data Expert** - Handles data loading and management
- **Single-cell Expert** - Specialized in scRNA-seq analysis
- **Bulk RNA-seq Expert** - Handles bulk transcriptomics
- **Proteomics Experts** - MS and affinity proteomics specialists
- **Research Agent** - Literature mining and dataset discovery

### Q: What programming languages does Lobster use?

**A:** Technology stack:
- **Core Platform** - Python 3.12+
- **Statistical Computing** - NumPy, SciPy, pandas
- **Bioinformatics** - scanpy, squidpy, pyDESeq2
- **Visualization** - Plotly, matplotlib, seaborn
- **AI Framework** - LangChain, LangGraph
- **Data Storage** - HDF5, AnnData, MuData

### Q: How extensible is Lobster AI?

**A:** Extensibility features:
- **Custom agents** - Add specialized analysis agents
- **Plugin system** - Extend functionality with custom tools
- **Service architecture** - Modular, stateless analysis services
- **API access** - Programmatic access to core functions

See the [Custom Agent Tutorial](26-tutorial-custom-agent.md) for details.

### Q: Does Lobster work offline?

**A:** Offline capabilities:
- **Local processing** - All analysis runs locally
- **Cached data** - GEO datasets cached for offline use
- **API requirement** - Only needs internet for LLM API calls
- **Offline mode** - Can work with cached responses (limited functionality)

---

## Troubleshooting

### Q: Lobster says "No data loaded" but I uploaded files

**A:** Common solutions:
1. **Check file format** - Verify Lobster recognizes your file type
2. **Specify structure** - Describe your file format explicitly
3. **Check file path** - Ensure file paths are correct and accessible
4. **Review error messages** - Use `/status` to check for loading errors

```bash
🦞 You: "/files"  # Check what files are visible
🦞 You: "Load CSV file with genes as rows, samples as columns"
```

### Q: Analysis fails with "insufficient memory" error

**A:** Memory solutions:
1. **Reduce data size** - Filter genes/cells before analysis
2. **Use sparse matrices** - Convert to memory-efficient format
3. **Chunked processing** - Process data in smaller pieces
4. **Cloud processing** - Use cloud resources for large datasets

```bash
🦞 You: "Convert to sparse matrix to save memory"
🦞 You: "Process this dataset using cloud resources"
```

### Q: Visualizations are not displaying

**A:** Visualization troubleshooting:
1. **Check plot generation** - Use `/plots` to list available plots
2. **Browser compatibility** - Open HTML plots in modern browser
3. **Regenerate plots** - Try creating plots with different parameters
4. **Export alternative formats** - Use PNG instead of interactive HTML

### Q: How do I get help with specific issues?

**A:** Support resources:
1. **Built-in help** - Use `/help` command for guidance
2. **Documentation** - Check [tutorials](23-tutorial-single-cell.md) and guides
3. **Troubleshooting guide** - See [detailed troubleshooting](28-troubleshooting.md)
4. **Community support** - GitHub issues, Discord community
5. **Diagnostic report** - Use built-in diagnostics for complex issues

```bash
🦞 You: "Generate diagnostic report for troubleshooting"
🦞 You: "/help analysis"  # Context-specific help
```

### Q: Can I recover from a crashed analysis?

**A:** Recovery options:
1. **Session recovery** - Lobster auto-saves progress
2. **Workspace backup** - Regular workspace snapshots
3. **Analysis history** - Complete provenance tracking
4. **Manual recovery** - Access intermediate results directly

```bash
🦞 You: "/workspace"  # Check workspace status
🦞 You: "Show analysis history for recovery"
```

---

## General Questions (v2.4+)

### Q: What's the difference between local and cloud mode?

**A:** Mode comparison:

| Aspect | Local Mode | Cloud Mode |
|--------|-----------|------------|
| **Execution** | On your machine | Remote servers |
| **API Key** | Not required (except LLM) | Requires LOBSTER_CLOUD_KEY |
| **Data Storage** | Local filesystem | Cloud storage (S3) |
| **Performance** | Depends on hardware | Scalable resources |
| **Cost** | Free (own hardware) | Usage-based pricing |
| **Privacy** | Full local control | Data transmitted to cloud |
| **Use Case** | Small-medium datasets, sensitive data | Large datasets, collaboration |

```bash
# Check current mode
> /status

# Switch to cloud mode
export LOBSTER_CLOUD_KEY="your-key"
lobster chat

# Switch back to local
unset LOBSTER_CLOUD_KEY
lobster chat
```

### Q: How do I switch between Bedrock and OpenAI models?

**A:** Model switching:

```bash
# Use AWS Bedrock (recommended for production)
export AWS_BEDROCK_ACCESS_KEY=your_access_key
export AWS_BEDROCK_SECRET_ACCESS_KEY=your_secret_key
# Remove: ANTHROPIC_API_KEY

# Use Claude API directly
export ANTHROPIC_API_KEY=sk-ant-xxx
# Remove: AWS_BEDROCK_*

# Use OpenAI (if configured) #TODO future support
export OPENAI_API_KEY=sk-xxx
# Remove: ANTHROPIC_API_KEY

# Verify active model
> /status
# Shows: "Model: AWS Bedrock (Claude)" or "Model: Anthropic API"
```

**Recommendation:** Use AWS Bedrock for production workloads (higher rate limits, better reliability).

### Q: Can I use Lobster offline?

**A:** Offline capabilities and limitations:

**What works offline:**
- Loading local data files
- Analysis on previously loaded datasets
- Accessing cached publications/datasets
- Export and visualization generation
- Pipeline execution (if cached)

**What requires internet:**
- LLM API calls (Anthropic, OpenAI, AWS Bedrock)
- Downloading new GEO datasets
- PubMed/literature searches
- Fetching protein structures from PDB
- ContentAccessService operations

**Partial offline mode:**
```bash
# Work with cached content
> "Show me cached publications"
> "List available datasets in workspace"

# Use local files only
> "Load H5AD file from local directory"
> "Analyze data without fetching external resources"
```

---

## Data Management Questions (v2.4+)

### Q: How do I access files from my workspace?

**A:** Workspace file access (v2.4+ WorkspaceContentService):

```bash
# List all workspace content
> /workspace
> "What content do I have cached?"

# List by type
> "Show me cached publications"
> "List cached datasets"
> "Show metadata files"

# Access specific content
> "Show methods from PMID:35042229"
> "Get sample metadata for GSE180759"
> "Retrieve validation report for GSE12345"

# Workspace structure
~/.lobster_workspace/
├── literature/     # Publications (PMIDs, DOIs)
├── data/           # Dataset metadata (GSE, SRA)
└── metadata/       # Custom metadata, mappings
```

### Q: What's the difference between modalities and workspace files?

**A:** Key differences:

| Aspect | Modalities | Workspace Files |
|--------|-----------|-----------------|
| **Type** | Analysis datasets (AnnData) | Research content (JSON) |
| **Storage** | In-memory + H5AD/MuData | Filesystem (workspace dirs) |
| **Purpose** | Active analysis data | Cached research content |
| **Access** | `get_modality()` | `WorkspaceContentService` |
| **Examples** | rna_seq_normalized, proteomics_filtered | publication_PMID12345, dataset_GSE180759 |
| **Lifespan** | Session | Persistent across sessions |

**Use modalities for:**
- Active bioinformatics analysis
- QC, normalization, clustering
- Statistical testing, DE analysis

**Use workspace files for:**
- Caching publications for later reference
- Storing dataset metadata before download
- Persistent research context across sessions

```bash
# Modalities (analysis data)
> "List all loaded datasets"  # /data command

# Workspace files (research content)
> "Show cached publications"  # WorkspaceContentService
```

### Q: How do I export my analysis results?

**A:** Comprehensive export options:

**Session Export:**
```bash
# Export entire session
> /export session

# Includes:
# - Conversation history
# - Loaded modalities
# - Workspace content
# - Generated plots
# - Analysis provenance
```

**Modality Export:**
```bash
# Export specific dataset
> "Export rna_seq_normalized as H5AD"
> "Save proteomics_data to CSV"

# Export with metadata
> "Export dataset with full provenance and plots"
```

**Pipeline Export:**
```bash
# Export reproducible notebook
> /pipeline export my_analysis.ipynb

# Generated notebook includes:
# - All analysis steps
# - Parameter configurations
# - Code snippets (Papermill-ready)
# - Provenance metadata
```

**Plot Export:**
```bash
# Export visualizations
> "Export all plots as high-resolution PNG"
> "Save UMAP plot as SVG for publication"

# Plots saved to workspace/plots/
```

---

## Services & Features Questions (v2.4+)

### Q: When should I use ContentAccessService vs providers?

**A:** Service vs provider usage:

**Use ContentAccessService (recommended):**
- Automatic provider selection
- Three-tier cascade (PMC → Webpage → PDF)
- Built-in caching and fallback
- Single API for all content types
- W3C-PROV provenance tracking

```bash
# ContentAccessService (high-level)
> "Read full publication PMID:35042229"
> "Search literature for BRCA1"
> "Discover datasets about breast cancer"
```

**Direct provider use (advanced):**
- Debugging specific provider issues
- Custom provider configuration
- Testing provider capabilities

```bash
# Direct provider (low-level, for developers)
from lobster.tools.providers.pmc_provider import PMCProvider
provider = PMCProvider()
# ... manual provider calls
```

**Recommendation:** Always use ContentAccessService unless you have specific advanced needs.

### Q: How do I visualize protein structures?

**A:** Protein visualization with PyMOL (v2.4+):

**Step 1: Install PyMOL**
```bash
# Automated installation
make install-pymol

# Or manual
brew install brewsci/bio/pymol  # macOS
sudo apt-get install pymol       # Linux
```

**Step 2: Fetch and Visualize**
```bash
# Basic workflow
> "Fetch protein structure 1AKE"
> "Visualize 1AKE with PyMOL"

# Interactive mode (opens GUI)
> "Visualize 1AKE mode=interactive style=cartoon"

# Batch mode (generates PNG)
> "Visualize 1AKE mode=batch style=surface color_by=bfactor"
```

**Step 3: Link to Expression Data**
```bash
# Connect structures to your omics data
> "Link protein structures to my RNA-seq data"

# Result: adata.var gets columns:
# - pdb_structures: comma-separated PDB IDs
# - has_structure: boolean flag
```

**Highlight specific residues:**
```bash
# Single group
> "Visualize 1AKE highlight_residues=15,42,89 highlight_color=red"

# Multiple groups (binding site + active site)
> "Visualize 1AKE highlight_groups='15,42|red|sticks;100-120|blue|surface'"
```

See [Protein Structure Visualization Guide](40-protein-structure-visualization.md) for details.

### Q: Can I create custom agents?

**A:** Yes! Custom agent development:

**Requirements:**
- Agent configuration in `config/agent_registry.py`
- Factory function that returns agent
- Tools for agent capabilities
- Optional: handoff tool for supervisor

**Basic structure:**
```python
from lobster.config.agent_registry import AgentConfig, AGENT_REGISTRY

AGENT_REGISTRY["my_custom_agent"] = AgentConfig(
    name="my_custom_agent",
    display_name="My Custom Agent",
    description="Specialized analysis for X",
    factory_function="lobster.agents.my_agent.my_agent",
    handoff_tool_name="handoff_to_my_custom_agent",
    handoff_tool_description="When to use this agent"
)
```

See [Creating Agents Guide](09-creating-agents.md) and [Custom Agent Tutorial](26-tutorial-custom-agent.md).

---

## Performance Questions (v2.4+)

### Q: How do I speed up large dataset analysis?

**A:** Performance optimization strategies:

**1. Use Cloud Processing:**
```bash
# For datasets >50K cells or >10GB
export LOBSTER_CLOUD_KEY="your-key"
lobster chat
> "Process this large dataset on cloud infrastructure"
```

**2. Enable Caching:**
```bash
# ContentAccessService caches automatically
# DataManagerV2 caches modalities

# Check cache usage
> /workspace
# Shows cached content size
```

**3. Sparse Matrix Conversion:**
```bash
# Convert dense to sparse (saves memory)
> "Convert to sparse matrix format"
# Can reduce memory by 10-100x for scRNA-seq
```

**4. Subsample for Testing:**
```bash
# Test parameters on subset
> "Subsample 1000 cells for parameter testing"
> "Once optimized, run on full dataset"
```

**5. Parallel Processing:**
```bash
# Use multiple cores (automatic in most services)
export LOBSTER_N_CORES=8
lobster chat
```

### Q: What's the recommended hardware for Lobster?

**A:** Hardware recommendations by use case:

**Minimum (Small Datasets <10K cells):**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free
- Network: Stable internet for API calls

**Recommended (Medium Datasets 10K-100K cells):**
- CPU: 8+ cores
- RAM: 16-32GB
- Storage: 50GB+ SSD
- Network: High-speed internet

**Advanced (Large Datasets >100K cells):**
- CPU: 16+ cores or use cloud
- RAM: 64GB+ or use cloud
- Storage: 100GB+ NVMe SSD
- Network: Gigabit ethernet
- **Alternative:** Use cloud mode with LOBSTER_CLOUD_KEY

**Cloud Mode (Any Size):**
- Local: Just need API access
- Cloud handles: All compute and storage
- Recommended for: >100K cells, >10GB datasets

### Q: How do I reduce memory usage?

**A:** Memory reduction techniques:

**1. Use Sparse Matrices:**
```bash
> "Convert data to sparse format"
# Saves 10-100x memory for sparse data (scRNA-seq)
```

**2. Filter Early:**
```bash
# Remove low-quality cells/genes before downstream analysis
> "Filter cells with <200 genes"
> "Keep only highly variable genes"
```

**3. Chunked Processing:**
```bash
# Process in batches
> "Process this dataset in chunks of 10,000 cells"
```

**4. Delete Unused Modalities:**
```bash
# Remove intermediate datasets
> "Delete modality rna_seq_raw"  # Keep only final version
```

**5. Monitor Usage:**
```bash
# Check memory consumption
> /dashboard

# Shows:
# - RAM usage
# - Disk usage
# - Active modalities
# - Cache sizes
```

**6. Use Cloud Mode:**
```bash
# Offload to cloud resources
export LOBSTER_CLOUD_KEY="your-key"
> "Process this memory-intensive analysis on cloud"
```

---

## Troubleshooting Questions (v2.4+)

### Q: Why is my analysis taking so long?

**A:** Performance debugging:

**Check 1: Dataset Size**
```bash
> "Show dataset statistics"
# Look for: number of cells, genes, total size

# If very large (>100K cells):
> "Use cloud processing for faster analysis"
```

**Check 2: Operation Type**
```bash
# Some operations are inherently slow:
# - Slow: UMAP (minutes), clustering (minutes)
# - Fast: QC metrics (seconds), filtering (seconds)
```

**Check 3: System Resources**
```bash
> /dashboard
# Check: CPU usage, RAM usage, disk I/O

# If maxed out:
> "Reduce dataset size or use cloud mode"
```

**Check 4: Provider Performance**
```bash
# For ContentAccessService operations
> "Query capabilities"
# Shows provider performance tiers

# PMC XML: 500ms-2s (fast)
# Webpage/PDF: 2-8s (slower)
```

### Q: How do I debug failed analyses?

**A:** Debugging workflow:

**Step 1: Check Status**
```bash
> /status
# Shows: system health, loaded data, errors
```

**Step 2: Enable Debug Mode**
```bash
# Start with verbose output
lobster chat --debug --verbose

# Or during session
> "Enable detailed error reporting"
```

**Step 3: Review Error Messages**
```bash
# Lobster provides helpful error context
# Look for:
# - Specific error type
# - Suggested solutions
# - Related documentation links
```

**Step 4: Check Provenance**
```bash
# Review what was executed
> "Show analysis history and provenance"
> /pipeline list

# Identify where failure occurred
```

**Step 5: Test Smaller Scope**
```bash
# Isolate the problem
> "Test this analysis on a small subset"
> "Run just the preprocessing step"
```

**Step 6: Generate Diagnostic Report**
```bash
> "Generate diagnostic report for troubleshooting"

# Includes:
# - System info
# - Error logs
# - Workspace state
# - Recent operations
```

### Q: What do I do if I run out of disk space?

**A:** Disk space management:

**Check Usage:**
```bash
# Terminal
df -h ~
du -sh ~/.lobster_workspace/

# In Lobster
> /workspace
# Shows workspace size
```

**Clean Workspace:**
```bash
# Remove old cached content (>30 days)
find ~/.lobster_workspace/ -mtime +30 -delete

# Or in Lobster
> "Clear old cached publications from workspace"
```

**Archive Old Workspaces:**
```bash
# Backup and compress
tar -czf lobster_backup_$(date +%Y%m%d).tar.gz ~/.lobster_workspace/

# Delete original
rm -rf ~/.lobster_workspace/

# Lobster will recreate on next run
```

**Move to Larger Disk:**
```bash
# Move workspace to external drive
mv ~/.lobster_workspace /mnt/external/lobster_workspace

# Create symlink
ln -s /mnt/external/lobster_workspace ~/.lobster_workspace
```

**Use S3 Backend:**
```bash
# Store data in cloud (v2.4+)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
> "Use S3 backend for data storage"
```

**Set Size Limits:**
```bash
# Prevent workspace from growing too large
export LOBSTER_MAX_WORKSPACE_SIZE_MB=500
lobster chat
```

### Q: ContentAccessService says "not available" - how do I fix it?

**A:** ContentAccessService troubleshooting:

**Check 1: Service Initialization**
```bash
> "Query available capabilities"

# Should show all 5 providers:
# - AbstractProvider
# - PubMedProvider
# - GEOProvider
# - PMCProvider
# - WebpageProvider

# If missing providers, service not properly initialized
```

**Check 2: Dependencies**
```bash
# Install optional dependencies
pip install lobster[docling]

# Verify
python -c "import docling; print('OK')"
```

**Check 3: Restart**
```bash
# Clean restart
rm -rf ~/.lobster_workspace/
lobster chat
```

**Check 4: Check Logs**
```bash
# Enable debug logging
lobster chat --debug

# Look for initialization errors
```

**If still failing:**
```bash
# Use alternative methods
> "Get abstract for PMID:12345"  # Simpler, always works
> "Search PubMed directly"  # Bypass ContentAccessService
```

See [Troubleshooting Guide](28-troubleshooting.md) for detailed solutions.

### Q: How do I resolve "permission denied" errors in workspace?

**A:** Workspace permission fixes:

**Fix Ownership:**
```bash
# Take ownership of workspace
chown -R $USER:$USER ~/.lobster_workspace/
```

**Fix Permissions:**
```bash
# Make readable/writable
chmod -R u+rw ~/.lobster_workspace/

# Directories need execute permission
chmod -R u+rwx ~/.lobster_workspace/*/
```

**Check SELinux (Linux):**
```bash
# If SELinux is enforcing
getenforce

# Temporarily disable for testing
sudo setenforce 0

# Don't disable permanently in production!
```

**Fresh Workspace:**
```bash
# Nuclear option: delete and recreate
rm -rf ~/.lobster_workspace/
lobster chat  # Recreates with correct permissions
```

**Alternative Location:**
```bash
# Use different path with proper permissions
lobster chat --workspace /path/with/write/access
```

---

## Additional Resources

### Q: Where can I find more examples and tutorials?

**A:** Learning resources:
- **[Single-cell Tutorial](23-tutorial-single-cell.md)** - Complete scRNA-seq workflow
- **[Bulk RNA-seq Tutorial](24-tutorial-bulk-rnaseq.md)** - Differential expression analysis
- **[Proteomics Tutorial](25-tutorial-proteomics.md)** - MS and affinity proteomics
- **[Custom Agent Tutorial](26-tutorial-custom-agent.md)** - Extend Lobster functionality
- **[Examples Cookbook](27-examples-cookbook.md)** - Common analysis patterns

### Q: How can I contribute to Lobster AI?

**A:** Contribution opportunities:
- **Report bugs** - GitHub issues with detailed reports
- **Suggest features** - Community feedback and feature requests
- **Contribute code** - Pull requests for improvements
- **Create content** - Tutorials, examples, documentation
- **Community support** - Help other users in Discord/forums

### Q: Is there a community forum or chat?

**A:** Community resources:
- **Discord Community** - Real-time help and discussion
- **GitHub Discussions** - Feature requests and general discussion
- **GitHub Issues** - Bug reports and technical questions
- **Email Support** - Direct contact for complex issues

---

This FAQ covers the most common questions about Lobster AI. For additional information, consult the comprehensive [documentation](README.md) or reach out to the community through the support channels listed above.