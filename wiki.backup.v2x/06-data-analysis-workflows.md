# Data Analysis Workflows

## Overview

This guide provides step-by-step workflows for analyzing different types of biological data using Lobster AI. Each workflow combines natural language interaction with specialized AI agents to perform publication-quality analysis.

## Single-Cell RNA-seq Analysis Workflow

### Workflow Overview

**Goal**: Analyze single-cell RNA-seq data to identify cell types, find marker genes, and understand cellular heterogeneity.

**Agent**: Single-Cell Expert handles all aspects of scRNA-seq analysis.

**Time**: 15-30 minutes for typical dataset (10K-50K cells)

### Step 1: Data Loading and Initial Assessment

```bash
# Load your single-cell data
/read my_singlecell_data.h5ad

# Alternative: Load from multiple formats
/read counts_matrix.csv
/read filtered_feature_bc_matrix/  # 10X format
/read *.h5                        # Multiple files
```

**Natural Language Alternative**:
```
"Load my single-cell RNA-seq data from the h5ad file"
```

**Expected Output**:
- Data shape (cells × genes)
- File format confirmation
- Initial data structure summary

### Step 2: Data Quality Assessment

```bash
# Check data overview
/data

# Request quality control analysis
"Perform quality control analysis on this single-cell data"
```

**Quality Control Includes**:
- **Mitochondrial Gene Percentage**: Cell viability indicator
- **Ribosomal Gene Percentage**: Translation activity
- **Total Gene Counts**: Library complexity
- **Total UMI Counts**: Sequencing depth
- **Doublet Detection**: Multi-cell artifacts

**Expected Results**:
- Quality control metrics for each cell
- Distribution plots for QC metrics
- Recommendations for filtering thresholds

### Step 3: Data Filtering and Preprocessing

```
"Filter low-quality cells and normalize the data using standard parameters"
```

**Or specify custom parameters**:
```
"Filter cells with less than 200 genes and more than 20% mitochondrial content, then normalize using log1p transformation"
```

**Processing Steps**:
1. **Cell Filtering**: Remove low-quality cells
2. **Gene Filtering**: Remove rarely expressed genes
3. **Normalization**: Library size normalization + log1p
4. **Highly Variable Genes**: Identify most informative features

**Expected Output**:
- Filtered dataset dimensions
- Normalization parameters used
- Quality metrics after filtering

### Step 4: Dimensionality Reduction and Clustering

```
"Perform PCA, compute neighbors, and cluster the cells using Leiden algorithm"
```

**Or request comprehensive analysis**:
```
"Run the complete single-cell workflow: PCA, UMAP, clustering, and find marker genes"
```

**Analysis Steps**:
1. **Principal Component Analysis (PCA)**: Reduce dimensionality
2. **Neighborhood Graph**: Build cell-cell similarity network
3. **Leiden Clustering**: Identify cell communities
4. **UMAP Embedding**: 2D visualization

**Expected Results**:
- UMAP plot with colored clusters
- Cluster statistics and cell counts
- Quality assessment of clustering

### Step 5: Cell Type Annotation

```
"Identify the cell types in each cluster using marker genes"
```

**For specific tissue**:
```
"Annotate cell types in this liver single-cell data using known liver cell markers"
```

**Annotation Methods**:
1. **Marker Gene Analysis**: Find top genes per cluster
2. **Reference Mapping**: Compare to cell atlases
3. **Manual Annotation**: User-guided cell type assignment
4. **Automated Annotation**: ML-based cell type prediction

**Expected Results**:
- Marker genes table for each cluster
- Cell type annotations
- UMAP plot with cell type labels
- Confidence scores for annotations

### Step 6: Differential Expression Analysis

```
"Find differentially expressed genes between cell types"
```

**For specific comparison**:
```
"Compare hepatocytes and stellate cells to find differentially expressed genes"
```

**Or condition-based analysis**:
```
"Find genes differentially expressed between control and treatment conditions in each cell type"
```

**Analysis Features**:
- **Statistical Testing**: Wilcoxon rank-sum test
- **Multiple Testing Correction**: Benjamini-Hochberg FDR
- **Effect Size Filtering**: Log fold change thresholds
- **Visualization**: Volcano plots and heatmaps

### Step 7: Advanced Analysis (Optional)

#### Trajectory Analysis
```
"Perform trajectory analysis to identify developmental paths"
```

#### Pseudobulk Analysis
```
"Aggregate cells by type and perform bulk RNA-seq differential expression"
```

#### Gene Set Enrichment
```
"Perform pathway enrichment analysis on the differentially expressed genes"
```

### Complete Workflow Example

```bash
# 1. Load data
/read liver_scrnaseq.h5ad

# 2. Comprehensive analysis request
"Analyze this liver single-cell RNA-seq data: perform quality control,
filter low-quality cells, normalize, cluster cells, identify cell types,
and find marker genes for each cluster"

# 3. Specific follow-up
"Compare hepatocytes between control and fibrotic conditions"

# 4. Visualization
/plots  # View all generated plots

# 5. Save results
/save
```

## Bulk RNA-seq Analysis Workflow

### Workflow Overview

**Goal**: Analyze bulk RNA-seq data to identify differentially expressed genes between conditions.

**Agent**: Bulk RNA-seq Expert specializes in count-based differential expression analysis.

**Time**: 10-20 minutes for typical experiment

### Step 1: Data Preparation

#### Option A: Load Kallisto/Salmon Quantification Files (Recommended)

**⚠️ NEW in v2.3+**: Use CLI `/read` command directly for quantification files.

```bash
# Load Kallisto quantification files
/read /path/to/kallisto_output

# Or load Salmon quantification files
/read /path/to/salmon_output
```

**Expected Directory Structure**:
```
quantification_output/
├── sample1/
│   └── abundance.tsv  (Kallisto) or quant.sf (Salmon)
├── sample2/
│   └── abundance.tsv  (Kallisto) or quant.sf (Salmon)
└── sample3/
    └── abundance.tsv  (Kallisto) or quant.sf (Salmon)
```

**Features**:
- **Direct CLI Loading**: Use `/read` command - no agent interaction needed
- **Automatic Tool Detection**: CLI detects Kallisto vs Salmon from file patterns
- **Per-Sample Merging**: Merges quantification from all sample subdirectories
- **Correct Orientation**: Transposes to samples × genes (bulk RNA-seq standard)
- **Sample Names**: Extracted from subdirectory names
- **Quality Validation**: Verifies file integrity and consistency

#### Option B: Load Count Matrix (Traditional)

```bash
# Load count matrix
/read counts_matrix.csv

# Load with metadata
/read counts.csv
"Load the sample metadata file to define experimental conditions"
```

**Expected Data Format**:
- Rows: Genes/transcripts
- Columns: Samples
- Raw or normalized counts

### Step 2: Experimental Design Setup

```
"Set up differential expression analysis comparing treatment vs control groups"
```

**For complex designs**:
```
"Analyze differential expression using the formula: ~condition + batch + gender"
```

**Features**:
- **R-style Formulas**: Support complex experimental designs
- **Batch Effect Handling**: Automatic detection and correction
- **Multiple Factors**: Age, gender, batch, treatment interactions
- **Contrasts**: Flexible comparison specifications

### Step 3: Quality Control

```
"Generate quality control plots and assess data distribution"
```

**QC Analysis Includes**:
- **Count Distribution**: Library size assessment
- **PCA Plots**: Sample clustering and batch effects
- **Correlation Heatmaps**: Sample relationships
- **Dispersion Plots**: Model fitting quality

### Step 4: Differential Expression with pyDESeq2

```
"Perform differential expression analysis using DESeq2"
```

**Analysis Features**:
- **Normalization**: Size factor estimation
- **Dispersion Modeling**: Gene-wise and fitted dispersions
- **Statistical Testing**: Wald test or likelihood ratio test
- **Shrinkage**: Effect size shrinkage for better estimates

**Results Include**:
- Log2 fold changes with confidence intervals
- P-values and adjusted P-values (FDR)
- Base means and dispersion estimates
- Convergence diagnostics

### Step 5: Results Visualization

```
"Create volcano plots and heatmaps for the differential expression results"
```

**Visualization Options**:
- **Volcano Plots**: Effect size vs significance
- **MA Plots**: Mean expression vs fold change
- **Heatmaps**: Top differentially expressed genes
- **PCA Plots**: Sample relationships

### Step 6: Downstream Analysis

```
"Perform pathway enrichment analysis on the upregulated genes"
```

**Advanced Analysis**:
- Gene set enrichment analysis (GSEA)
- Pathway over-representation analysis
- Gene ontology analysis
- KEGG pathway mapping

### Complete Workflow Example

```bash
# 1. Load data
/read rnaseq_counts.csv

# 2. Define experimental setup
"Analyze differential expression between high-fat diet and control mice,
accounting for batch effects and gender differences"

# 3. Request comprehensive analysis
"Perform complete bulk RNA-seq analysis: quality control, normalization,
differential expression testing, and generate volcano plots"

# 4. Follow-up analysis
"Show me the top 20 upregulated genes and their functions"

# 5. Export results
/export
```

## Mass Spectrometry Proteomics Workflow

### Workflow Overview

**Goal**: Analyze label-free quantitative proteomics data to identify differentially abundant proteins.

**Agent**: MS Proteomics Expert handles mass spectrometry data analysis.

**Time**: 20-40 minutes depending on dataset complexity

### Step 1: Data Loading

```bash
# Load MaxQuant output
/read proteinGroups.txt

# Load Spectronaut results
/read spectronaut_results.csv

# Load generic proteomics data
/read protein_intensities.csv
```

### Step 2: Data Assessment

```
"Assess the quality of this proteomics data and show missing value patterns"
```

**Quality Assessment**:
- **Missing Value Analysis**: MNAR vs MCAR patterns
- **Coefficient of Variation**: Technical and biological CV
- **Intensity Distributions**: Dynamic range assessment
- **Batch Effect Detection**: Systematic biases

### Step 3: Data Preprocessing

```
"Filter proteins with excessive missing values and normalize intensities"
```

**Preprocessing Steps**:
1. **Protein Filtering**: Remove contaminants and reverse sequences
2. **Missing Value Handling**: Imputation strategies (MNAR/MCAR)
3. **Intensity Normalization**: TMM, quantile, or VSN normalization
4. **Log Transformation**: Variance stabilization

### Step 4: Statistical Analysis

```
"Perform differential protein abundance analysis between treatment groups"
```

**Statistical Methods**:
- **Linear Models**: limma-based analysis
- **Empirical Bayes**: Moderated t-statistics
- **Multiple Testing**: FDR control
- **Effect Size Estimation**: Protein fold changes

### Step 5: Results Interpretation

```
"Identify significantly changed proteins and perform pathway analysis"
```

**Results Analysis**:
- Volcano plots for differential proteins
- Protein interaction networks
- Pathway enrichment analysis
- GO term analysis

### Complete Workflow Example

```bash
# Load MaxQuant data
/read proteinGroups.txt

# Comprehensive analysis
"Analyze this label-free proteomics data: assess data quality,
handle missing values, normalize intensities, and identify proteins
differentially abundant between control and treatment groups"

# Pathway analysis
"Perform pathway enrichment analysis on the significantly changed proteins"
```

## Affinity Proteomics Workflow

### Workflow Overview

**Goal**: Analyze targeted proteomics data from Olink panels or antibody arrays.

**Agent**: Affinity Proteomics Expert specializes in targeted protein analysis.

**Time**: 15-25 minutes for typical panel

### Step 1: Data Loading

```bash
# Load Olink NPX data
/read olink_npx_data.csv

# Load antibody array data
/read antibody_intensities.csv
```

### Step 2: Quality Assessment

```
"Assess the quality of this Olink panel data and check for batch effects"
```

**Quality Metrics**:
- **Coefficient of Variation**: Within and between batch CV
- **Detection Rates**: Protein detectability across samples
- **Control Performance**: Internal control assessment
- **Batch Effects**: Systematic biases between runs

### Step 3: Statistical Analysis

```
"Compare protein levels between disease and healthy control groups"
```

**Analysis Features**:
- **Linear Models**: Account for covariates
- **Batch Correction**: ComBat or similar methods
- **Multiple Testing**: FDR correction
- **Effect Size**: Clinical significance assessment

### Complete Workflow Example

```bash
# Load Olink data
/read olink_cardiovascular_panel.csv

# Comprehensive analysis
"Analyze this Olink cardiovascular panel data: assess quality,
check for batch effects, and identify proteins associated with
cardiovascular disease status"
```

## Multi-Omics Integration Workflow

### Workflow Overview

**Goal**: Integrate multiple data modalities for comprehensive biological insights.

**Agents**: Multiple agents coordinate for multi-modal analysis.

**Time**: 30-60 minutes depending on complexity

### Step 1: Load Multiple Datasets

```bash
# Load different modalities
/read transcriptomics_data.h5ad
/read proteomics_data.csv
/read metabolomics_data.xlsx
```

### Step 2: Data Integration

```
"Integrate the transcriptomics and proteomics data to identify
coordinated changes across molecular layers"
```

**Integration Methods**:
- **Sample Matching**: Align samples across modalities
- **Feature Integration**: Multi-omics factor analysis
- **Pathway Integration**: Combine evidence across layers
- **Network Analysis**: Multi-layer biological networks

### Step 3: Coordinated Analysis

```
"Find genes and proteins that change together in response to treatment"
```

**Results**:
- Correlation analysis across omics layers
- Pathway-level integration
- Multi-omics visualizations
- Integrated statistical models

## Literature Integration Workflow

### Workflow Overview

**Goal**: Integrate literature knowledge with experimental data analysis.

**Agent**: Research Agent with **automatic PMID/DOI → PDF resolution** (v2.2+) and **structure-aware Docling parsing** (v2.3+).

**Key Capabilities**:
- **v2.2+**: Automatic resolution of PMIDs and DOIs to accessible PDFs (70-80% success rate) using tiered waterfall strategy: PMC → bioRxiv/medRxiv → Publisher → Alternative suggestions
- **v2.3+**: Structure-aware PDF parsing with Docling for intelligent Methods section detection (>90% hit rate vs ~30% previously), complete section extraction, table and formula preservation, and document caching

### Step 1: Literature Search

```
"Find papers about single-cell RNA-seq analysis of liver fibrosis"
```

### Step 2: Method Extraction (Enhanced with v2.3+ DOI Resolution)

**Enhanced (v2.2+)**: Directly provide PMIDs or DOIs - automatic resolution to PDFs happens internally.

**Enhanced (v2.3+)**: Robust DOI/PMID auto-detection and resolution with Docling format auto-detection.

**All these formats now work seamlessly:**
```bash
# Bare DOI (NEW - auto-detected and resolved)
"Extract methods from 10.1101/2024.08.29.610467"

# DOI with prefix
"Extract methods from DOI:10.1038/s41586-025-09686-5"

# PMID with or without prefix
"Extract methods from PMID:39370688"
"Extract methods from 39370688"

# Direct URLs (existing behavior maintained)
"Extract methods from https://www.nature.com/articles/s41586-025-09686-5"

# PMC URLs (now correctly handled as HTML, not PDF)
"Extract methods from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12496192/pdf/"
```

**Batch processing for competitive analysis:**
```bash
"Extract methods from these papers: 10.1101/2024.01.001, PMID:12345678, DOI:10.1038/s41586-021-12345-6"
```

**Automatic handling**:
- ✅ Accessible papers → Methods extracted immediately using Docling structure-aware parsing
- ✅ Complete Methods sections extracted (no arbitrary truncation)
- ✅ Parameter tables and formulas preserved
- ✅ Results cached for fast repeat access
- ❌ Paywalled papers → 5 alternative access strategies provided (PMC accepted manuscripts, preprints, institutional access, author contact, Unpaywall)

**Quality Improvement (v2.3+)**:
- Methods section detection: >90% success rate (vs ~30% with naive truncation)
- Complete section extraction (no 10K character limit)
- Table extraction: 80%+ of parameter tables detected
- Smart image filtering: 40-60% context size reduction
- Document caching: 30-50x faster on repeat access

### v2.3+ Enhancement: Robust DOI Resolution

**What Changed:**
The v2.3+ release fixed critical DOI/PMID resolution bugs and enhanced format detection:

**✅ Fixed Issues:**
- DOIs and PMIDs are now automatically detected and resolved
- No more "URL not found" errors for valid DOIs (e.g., `10.18632/aging.204666`)
- PMC URLs serving HTML content correctly handled (not misclassified as PDF)
- Eliminated duplicate code paths in research agent

**✅ New Capabilities:**
- **Bare DOI input:** `"Extract methods from 10.1101/2024.01.001"` (no URL wrapper needed)
- **Numeric PMID input:** `"Extract methods from 38448586"` (no "PMID:" prefix needed)
- **Format auto-detection:** Docling determines HTML vs PDF automatically
- **Graceful error handling:** Paywalled papers return helpful suggestions

**Examples that now work reliably:**
```bash
# These previously failed with FileNotFoundError, now work:
"Extract methods from 10.1101/2024.01.001"          # bioRxiv DOI
"Extract methods from 38448586"                      # Numeric PMID
"Extract methods from 10.18632/aging.204666"        # Paywalled (graceful handling)

# These work better with enhanced format detection:
"Extract methods from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC..."  # HTML auto-detected
```

**See also**: [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md) for comprehensive Docling integration details.

### Step 3: Check Accessibility (Optional)

For competitive analysis, check accessibility before extraction:

```
"Check if PMID:12345678 is accessible"
```

### Step 4: Method Application

```
"Apply the methods from PMID:12345678 to analyze my data using their parameters"
```

## GEO Database Integration Workflow

### Workflow Overview

**Goal**: Download and analyze public datasets from GEO database.

**Agent**: Data Expert handles GEO integration.

### Step 1: Dataset Discovery

```
"Find GEO datasets related to liver single-cell RNA-seq"
```

**Research Agent** will search GEO database and return relevant datasets with accession numbers.

### Step 2: Pre-Download Metadata Validation (Recommended)

**Before downloading large datasets**, validate that they contain the required metadata fields:

```
"Validate GSE200997 for required fields: cell_type, tissue"
```

**Or with specific value requirements**:
```
"Check if GSE179994 has treatment_response field with responder and non-responder values"
```

**What This Does**:
- Fetches only metadata (no expression data download)
- Analyzes sample characteristics from all samples
- Checks field presence and coverage (% of samples)
- Provides recommendation: proceed/skip/manual_check
- Returns confidence score (0-1)

**Example Validation Report**:
```
## Metadata Validation Report for GSE200997

**Recommendation:** ✅ **PROCEED**
**Confidence Score:** 1.00/1.00
**Total Samples:** 23

### Field Analysis:
- **cell_type**: ✅ 100.0% coverage (values: 'Colon,Right,Cecum', 'Colon,Left,Sigmoid', ...)
- **tissue**: ✅ 100.0% coverage (values: 'Colorectal cancer')

### 💡 Recommendation Rationale:
All required fields are present with sufficient coverage. Dataset is suitable for analysis.
```

**Why Validate First?**:
- ⏱️ **Save time**: 2-5 seconds vs 5-30 minutes full download
- 💾 **Save storage**: Avoid downloading datasets missing critical metadata
- 🎯 **Better selection**: Compare metadata across multiple candidates
- 📊 **Field coverage**: See actual sample-level completeness

**Common Use Cases**:
- Drug discovery: Validate treatment response fields
- Biomarker studies: Check clinical outcome metadata
- Multi-dataset analysis: Filter by metadata completeness
- Time series: Verify timepoint field exists

### Step 3: Data Download

Once validation confirms the dataset is suitable:

```
"Download GSE200997 and prepare it for analysis"
```

**Data Expert** will download expression data and create analysis-ready dataset.

### Step 4: Comparative Analysis

```
"Compare my results to the downloaded GEO dataset GSE200997"
```

## Session Continuation and Workspace Management

### Overview

Lobster AI v2.2+ includes powerful workspace management capabilities that allow you to save your analysis progress and seamlessly continue work across sessions. This is particularly useful for long-running analyses or when working with multiple datasets.

### Workspace Restoration Workflow

#### Step 1: Check Current Workspace State

Before starting any analysis session, check what data is currently loaded and what's available in your workspace:

```bash
# Check currently loaded data
/data

# List available datasets in workspace
/workspace list

# Show comprehensive workspace information
/workspace
```

**Natural Language Alternative**:
```
"What data do I have available in my workspace?"
"Show me my current analysis session status"
```

#### Step 2: Restore Previous Session

Use the `/restore` command to load datasets from previous sessions:

```bash
# Restore most recent datasets (recommended for session continuation)
/restore

# Restore specific dataset by name
/restore geo_gse123456_processed

# Restore all datasets matching a pattern
/restore geo_*                    # All GEO datasets
/restore *single_cell*           # All single-cell datasets
/restore experiment_batch_2*     # Specific experiment datasets

# Restore all available datasets (use with caution for memory)
/restore all
```

**Natural Language Alternative**:
```
"Continue my analysis from yesterday's session"
"Load the GSE123456 dataset I was working on"
"Restore all my single-cell datasets for comparison"
```

#### Step 3: Verify Restored Data

After restoration, verify that your datasets are properly loaded:

```bash
# Check loaded modalities
/modalities

# Get detailed data summary
/data

# List available plots from previous session
/plots
```

### Complete Session Continuation Example

#### Scenario: Continuing Single-Cell Analysis

```bash
# Day 1: Initial Analysis
"Download and analyze GSE123456 single-cell data"
# ... perform quality control, clustering, etc.
/save  # Save progress

# Day 2: Continue Analysis
/restore recent
# System loads: geo_gse123456, geo_gse123456_filtered, geo_gse123456_clustered

"Continue the differential expression analysis on the clustered data"
# Agent automatically uses geo_gse123456_clustered for analysis
```

#### Scenario: Comparative Analysis Across Multiple Datasets

```bash
# Load multiple related datasets for comparison
/restore geo_gse123*             # Loads multiple GSE datasets
"Compare these datasets and identify common cell types"

# Work with specific experiment batches
/restore experiment_*
"Perform batch correction across these experiment datasets"
```

#### Scenario: Project-Based Workflow

```bash
# Organize by project patterns
/restore liver_*                 # All liver-related datasets
/restore *cancer_study*          # All cancer study datasets
/restore proteomics_*            # All proteomics datasets

"Integrate these liver datasets for multi-omics analysis"
```

### Advanced Workspace Management

#### Pattern Matching Best Practices

| Use Case | Pattern | Example |
|----------|---------|---------|
| Continue recent work | `recent` | `/restore recent` |
| Load specific dataset | `exact_name` | `/restore geo_gse123456_processed` |
| Load by data type | `*type*` | `/restore *single_cell*` |
| Load by experiment | `prefix*` | `/restore batch_2*` |
| Load by source | `source_*` | `/restore geo_*` |

#### Memory Management

```bash
# Check memory usage before loading
/modalities                      # See current memory usage

# Load incrementally for large datasets
/restore experiment_1*           # Load first batch
# Perform analysis
/restore experiment_2*           # Load second batch when needed
```

#### Data Organization Tips

**Recommended Naming Conventions**:
```
geo_gse123456                    # Raw GEO data
geo_gse123456_filtered          # After quality control
geo_gse123456_clustered         # After clustering
geo_gse123456_annotated         # With cell type annotations
custom_liver_study_raw          # Custom dataset
custom_liver_study_processed    # After processing
```

### Integration with Analysis Workflows

#### Single-Cell Workflow Continuation

```bash
# Session 1: Initial processing
"Download GSE123456 and perform quality control"
/save

# Session 2: Clustering analysis
/restore recent
"Perform clustering and find marker genes"
/save

# Session 3: Cell type annotation
/restore recent
"Annotate cell types based on marker genes"
```

#### Multi-Dataset Comparison Workflow

```bash
# Load multiple datasets for comparison
/restore geo_gse123456 geo_gse789012 custom_study
"Compare these three datasets and identify batch effects"

# Load by pattern for systematic comparison
/restore *liver*
"Perform integrated analysis of all liver datasets"
```

#### Cross-Session Plot Management

```bash
# Restore data and plots from previous session
/restore recent
/plots                          # List available plots

"Generate additional plots comparing the clustered results"
# New plots are automatically saved to workspace
```

### Natural Language Workspace Commands

The data expert agent understands various natural language requests for workspace management:

```
"Load my recent datasets"
"Continue my analysis from yesterday"
"Load all the GEO datasets I downloaded"
"Restore the liver study data for comparison"
"What datasets do I have available?"
"Load the processed single-cell data"
"Continue working on the GSE123456 dataset"
"Restore all my proteomics experiments"
```

### Troubleshooting Workspace Issues

#### Common Problems and Solutions

**Dataset Not Found**:
```bash
Problem: "Dataset 'my_dataset' not found"
Solution: Check available datasets with /workspace list
         Verify spelling and use Tab completion
```

**Memory Issues**:
```bash
Problem: System runs out of memory
Solution: Use more specific patterns instead of /restore all
         Load datasets incrementally
         Check current usage with /modalities
```

**Outdated Workspace**:
```bash
Problem: Restored data seems outdated
Solution: Check workspace location with /workspace
         Verify you're in the correct project directory
         Use /workspace list to see available datasets
```

### Best Practices for Session Management

1. **Regular Saves**: Use `/save` after major analysis steps
2. **Descriptive Names**: Use clear dataset names for easy pattern matching
3. **Incremental Loading**: Load datasets as needed to manage memory
4. **Verify Restoration**: Always check `/data` after restoration
5. **Organize by Project**: Use consistent naming patterns for related analyses
6. **Document Progress**: Keep track of analysis steps and parameters

## Advanced Workspace Management

> **Version**: v2.2+
> **Prerequisites**: Basic workspace usage (see [Session Continuation and Workspace Management](#session-continuation-and-workspace-management))

While the basic workspace restoration features enable session continuation, advanced workspace management provides enterprise-grade capabilities for backup, migration, templating, analytics, cleanup, and multi-workspace orchestration. These features are critical for:

- **Reproducibility**: Archive complete analysis environments
- **Collaboration**: Share workspaces between team members
- **Automation**: Template-based workflows for standardized pipelines
- **Resource Management**: Monitor and optimize workspace storage
- **Project Organization**: Manage multiple concurrent analyses

### 1. Workspace Backup and Restore

#### Complete Workspace Backup

Create a complete snapshot of your workspace including all datasets, provenance, and configurations.

**Basic Backup:**
```bash
# Backup current workspace to archive
/workspace backup --name my_analysis_v1 --destination ./backups/

# With compression and metadata
/workspace backup --name liver_study_final \
                  --destination ./backups/ \
                  --compress \
                  --include-metadata
```

**Natural Language Alternative:**
```
"Create a backup of my current workspace named liver_study_final"
"Archive this workspace with all datasets and analysis history"
```

**What Gets Backed Up:**
- ✅ All H5AD/MuData files in workspace
- ✅ Provenance tracking history (W3C-PROV format)
- ✅ Download queue state (JSONL)
- ✅ Cached plots and visualizations
- ✅ Workspace configuration and metadata
- ✅ Analysis pipeline exports (Jupyter notebooks)
- ❌ Large external files (can be optionally included)

**Backup Structure:**
```
backups/
└── liver_study_final_20250116/
    ├── workspace.tar.gz              # Compressed workspace data
    ├── manifest.json                 # File inventory
    ├── provenance_graph.json         # Complete W3C-PROV graph
    ├── metadata.json                 # Workspace info
    └── checksum.sha256               # Integrity verification
```

#### Incremental Backup

For large workspaces, use incremental backups to save only changes since the last backup.

```bash
# Initial full backup
/workspace backup --name project_v1 --destination ./backups/

# Incremental backup (only changes)
/workspace backup --name project_v2 \
                  --destination ./backups/ \
                  --incremental \
                  --base project_v1
```

**Incremental Backup Benefits:**
- 80-95% faster than full backups
- 70-90% smaller backup size
- Maintains complete restore capability
- Delta compression using rsync-like algorithm

#### Workspace Restore from Backup

**Complete Restore:**
```bash
# Restore from backup archive
/workspace restore --source ./backups/liver_study_final_20250116/

# Restore to specific location
/workspace restore --source ./backups/project_v2/ \
                   --destination ./new_workspace/ \
                   --verify-checksums
```

**Selective Restore:**
```bash
# Restore only specific datasets
/workspace restore --source ./backups/liver_study_final/ \
                   --datasets geo_gse123456,custom_liver_study

# Restore datasets matching pattern
/workspace restore --source ./backups/proteomics_study/ \
                   --pattern "*single_cell*"

# Restore provenance only (for audit)
/workspace restore --source ./backups/project_v1/ \
                   --provenance-only
```

**Verification After Restore:**
```bash
# Verify backup integrity
/workspace verify --source ./backups/liver_study_final/

# Compare restored workspace to original
/workspace compare --workspace1 ./original/ \
                   --workspace2 ./restored/
```

#### Automated Backup Strategies

**Scheduled Backups:**
```python
# In automation script or config
from lobster.core.workspace_manager import WorkspaceBackupScheduler

scheduler = WorkspaceBackupScheduler(
    workspace_path="./my_workspace",
    backup_dir="./backups",
    schedule="daily",       # Options: hourly, daily, weekly
    retention_days=30,      # Delete backups older than 30 days
    incremental=True,       # Use incremental backups
    compress=True
)

scheduler.start()
```

**Event-Triggered Backups:**
```python
# Backup after major analysis steps
from lobster.core.workspace_manager import WorkspaceManager

wm = WorkspaceManager(workspace_path="./my_workspace")

# Register backup trigger
wm.register_backup_trigger(
    event="analysis_complete",
    backup_name_pattern="auto_{timestamp}",
    retention_count=10  # Keep last 10 backups
)
```

**Backup Best Practices:**

| Scenario | Backup Frequency | Retention Period | Strategy |
|----------|------------------|------------------|----------|
| Active development | Hourly | 7 days | Incremental |
| Production analysis | Daily | 30 days | Full + incremental |
| Long-term archival | On completion | Indefinite | Full + compression |
| Collaboration | Before handoff | Per project | Full + metadata |

### 2. Workspace Migration

#### Local to Cloud Migration

Migrate workspaces from local development to cloud infrastructure.

**Migration Command:**
```bash
# Migrate to S3-backed workspace
/workspace migrate --source ./local_workspace/ \
                   --destination s3://my-bucket/workspaces/project_1/ \
                   --backend s3 \
                   --verify \
                   --dry-run  # Test first

# Execute migration
/workspace migrate --source ./local_workspace/ \
                   --destination s3://my-bucket/workspaces/project_1/ \
                   --backend s3 \
                   --verify
```

**Natural Language Alternative:**
```
"Migrate my workspace to S3 storage for cloud analysis"
"Move this workspace to cloud infrastructure"
```

**Migration Process:**
1. **Pre-migration Check**: Verify source workspace integrity
2. **Format Conversion**: Convert H5AD to cloud-optimized format if needed
3. **Data Transfer**: Upload with resumable transfers and checksums
4. **Provenance Migration**: Transfer W3C-PROV graph to cloud storage
5. **Configuration Update**: Update workspace config for cloud backend
6. **Verification**: Verify all data accessible in target location
7. **Cleanup** (optional): Remove local copies after verification

#### Cross-Platform Migration

Migrate between different operating systems or environments.

**macOS → Linux Migration:**
```bash
# Export workspace for Linux
/workspace export --platform linux \
                  --destination ./linux_compatible_workspace.tar.gz

# On Linux machine
/workspace import --source ./linux_compatible_workspace.tar.gz \
                  --verify-platform
```

**Path Translation:**
```python
# Automatic path translation during migration
from lobster.core.workspace_migrator import WorkspaceMigrator

migrator = WorkspaceMigrator()

# Migrate with automatic path adjustment
migrator.migrate(
    source_path="./workspace",
    target_path="/mnt/analysis/workspace",
    translate_paths=True,        # Adjust absolute paths
    platform="linux",             # Target platform
    preserve_symlinks=False       # Convert symlinks to copies
)
```

#### Multi-User Environment Migration

Migrate workspaces between users or teams with permission management.

**Export for Sharing:**
```bash
# Export with anonymization (remove personal paths)
/workspace export --anonymize \
                  --include-data \
                  --format tar.gz \
                  --output shared_workspace.tar.gz

# Export with access control metadata
/workspace export --access-control \
                  --allowed-users user1,user2 \
                  --expiration-date 2025-12-31
```

**Import with Permission Setup:**
```bash
# Import to shared location
/workspace import --source shared_workspace.tar.gz \
                  --destination /shared/workspaces/project_1/ \
                  --permissions group-rw \
                  --owner analysis_team
```

### 3. Workspace Templates

#### Creating Workspace Templates

Templates enable standardized analysis pipelines and reproducible project structures.

**Template Creation:**
```bash
# Create template from existing workspace
/workspace create-template --source ./my_workflow/ \
                           --name single_cell_qc_template \
                           --description "Standard single-cell QC pipeline"

# Create template with parameterization
/workspace create-template --source ./bulk_rnaseq_workflow/ \
                           --name bulk_rnaseq_template \
                           --parameters design_formula,contrast,fdr_threshold
```

**Template Structure:**
```
templates/
└── single_cell_qc_template/
    ├── template.json              # Template metadata
    ├── workspace_structure.yaml   # Directory layout
    ├── analysis_pipeline.py       # Analysis script template
    ├── config_schema.json         # Configurable parameters
    └── example_config.yaml        # Example configuration
```

**Template Definition (template.json):**
```json
{
  "name": "single_cell_qc_template",
  "version": "1.0.0",
  "description": "Standard single-cell QC pipeline",
  "author": "Bioinformatics Team",
  "parameters": {
    "min_genes": {
      "type": "integer",
      "default": 200,
      "description": "Minimum genes per cell"
    },
    "max_mito_pct": {
      "type": "float",
      "default": 20.0,
      "description": "Maximum mitochondrial percentage"
    },
    "resolution": {
      "type": "float",
      "default": 0.5,
      "description": "Clustering resolution"
    }
  },
  "expected_inputs": ["raw_counts.h5ad"],
  "expected_outputs": ["filtered.h5ad", "clustered.h5ad", "markers.csv"]
}
```

#### Using Templates

**Instantiate New Workspace from Template:**
```bash
# Create workspace from template
/workspace new --template single_cell_qc_template \
               --name liver_study_2025 \
               --parameters config.yaml

# Create with inline parameters
/workspace new --template bulk_rnaseq_template \
               --name drug_treatment_study \
               --param design_formula="~treatment+batch" \
               --param contrast="treatment,drug,control" \
               --param fdr_threshold=0.05
```

**Configuration File (config.yaml):**
```yaml
# Parameters for single_cell_qc_template
min_genes: 250
max_mito_pct: 15.0
resolution: 0.4
tissue_type: "liver"
organism: "human"
```

**Natural Language Template Usage:**
```
"Create a new workspace using the single-cell QC template for my liver study"
"Set up a bulk RNA-seq analysis workspace using the standard template"
```

#### Template Library Management

**List Available Templates:**
```bash
# List all templates
/workspace templates list

# Search templates by tag
/workspace templates search --tag single_cell
/workspace templates search --tag proteomics
```

**Install Templates from Repository:**
```bash
# Install from GitHub
/workspace templates install \
    --source https://github.com/omics-os/analysis-templates \
    --name community_single_cell_v1

# Install from local file
/workspace templates install --source ./custom_template.tar.gz
```

**Share Templates:**
```bash
# Export template for sharing
/workspace templates export \
    --name my_custom_template \
    --output ./my_template.tar.gz \
    --include-examples

# Publish to registry (future feature)
/workspace templates publish \
    --name my_custom_template \
    --registry omics-os-registry \
    --visibility public
```

### 4. Workspace Analytics

#### Workspace Health Monitoring

Monitor workspace health, identify issues, and optimize performance.

**Health Check:**
```bash
# Comprehensive health check
/workspace health-check

# Detailed report with recommendations
/workspace health-check --detailed --output health_report.json
```

**Health Check Report:**
```
=== Workspace Health Report ===

Overall Status: 🟡 WARNING
Workspace: /Users/tyo/analysis/liver_study
Last Updated: 2025-01-16 14:30:00

📊 Storage Usage:
  Total Size: 15.2 GB
  Datasets: 12.8 GB (84%)
  Plots: 1.8 GB (12%)
  Provenance: 0.6 GB (4%)
  Warning: Approaching 80% of 20GB quota

📁 Dataset Health:
  Total Datasets: 24
  ✅ Healthy: 22 (92%)
  ⚠️  Warnings: 2 (8%)
    - geo_gse123456_old: Not accessed in 60 days
    - temp_analysis: Missing provenance metadata

🔍 Provenance Integrity:
  ✅ Complete: 20 datasets
  ⚠️  Partial: 2 datasets
  ❌ Missing: 2 datasets

🚀 Performance Metrics:
  Average Load Time: 2.3s (Good)
  Cache Hit Rate: 76% (Good)
  Slow Queries: 3 identified

💡 Recommendations:
  1. Archive or delete unused datasets (geo_gse123456_old)
  2. Clean up temporary files (temp_analysis)
  3. Run provenance repair on partial datasets
  4. Consider upgrading to S3 backend for better performance
```

#### Storage Analytics

**Storage Breakdown:**
```bash
# Analyze storage usage by type
/workspace storage-usage

# Detailed analysis with visualization
/workspace storage-usage --visualize --output storage_report.html
```

**Storage Usage Output:**
```
=== Storage Usage Analysis ===

By Data Type:
┌─────────────────┬──────────┬────────┬─────────┐
│ Type            │ Size     │ Count  │ % Total │
├─────────────────┼──────────┼────────┼─────────┤
│ H5AD            │ 10.5 GB  │   18   │  69%    │
│ MuData          │  2.3 GB  │    4   │  15%    │
│ Plots (HTML)    │  1.8 GB  │  156   │  12%    │
│ Provenance      │  0.6 GB  │   24   │   4%    │
└─────────────────┴──────────┴────────┴─────────┘

Top 10 Largest Datasets:
1. geo_gse200997_integrated (2.8 GB)
2. custom_liver_cohort_raw (1.9 GB)
3. geo_gse156793_processed (1.5 GB)
...

Growth Trend (Last 30 Days):
📈 +3.2 GB total (+26% growth rate)
Average: +107 MB/day

Projection:
At current growth, workspace will reach 80% quota in 42 days.
```

#### Dataset Usage Analytics

**Access Patterns:**
```bash
# Analyze dataset access patterns
/workspace analytics access-patterns --days 30

# Identify unused datasets
/workspace analytics find-unused --threshold-days 60
```

**Access Pattern Report:**
```
=== Dataset Access Patterns (Last 30 Days) ===

Most Accessed Datasets:
1. geo_gse123456_clustered (48 accesses, last: 1 hour ago)
2. custom_liver_study (32 accesses, last: 3 hours ago)
3. proteomics_batch_2 (21 accesses, last: 1 day ago)

Least Accessed Datasets:
1. geo_gse987654_old (0 accesses, last: 87 days ago) ⚠️
2. temp_analysis_v1 (0 accesses, last: 65 days ago) ⚠️
3. exploratory_test (1 access, last: 45 days ago)

💡 Cleanup Candidates:
  - 3 datasets not accessed in >60 days (5.4 GB reclaimable)
  - 7 temporary datasets with "temp_" prefix (2.1 GB reclaimable)
  - Total potential savings: 7.5 GB (49% of current usage)
```

#### Provenance Analytics

**Analyze Analysis Lineage:**
```bash
# Visualize provenance graph
/workspace analytics provenance-graph \
    --dataset geo_gse123456_final \
    --output lineage.html

# Find dataset dependencies
/workspace analytics dependencies \
    --dataset geo_gse123456_final
```

**Dependency Graph Output:**
```
=== Dataset Dependency Analysis ===

Dataset: geo_gse123456_final

Direct Dependencies (3):
├─ geo_gse123456_clustered (parent)
│  └─ geo_gse123456_filtered (parent)
│     └─ geo_gse123456 (root)

Processing Steps (5):
1. download_geo → geo_gse123456
2. assess_quality → geo_gse123456_qc
3. filter_normalize → geo_gse123456_filtered
4. cluster_leiden → geo_gse123456_clustered
5. annotate_cell_types → geo_gse123456_final

Tools Used: (6 unique)
  - GEOService
  - QualityService
  - PreprocessingService
  - ClusteringService
  - AnnotationService
  - VisualizationService
```

### 5. Cleanup Strategies

#### Manual Cleanup

**Identify Cleanup Candidates:**
```bash
# Find datasets to clean up
/workspace cleanup --dry-run \
                   --threshold-days 60 \
                   --min-size 500MB

# Show what would be deleted
/workspace cleanup --preview \
                   --unused-days 90 \
                   --temp-files
```

**Selective Cleanup:**
```bash
# Delete specific datasets
/workspace delete geo_gse987654_old temp_analysis_v1

# Delete by pattern
/workspace delete "temp_*"

# Delete old plots
/workspace cleanup-plots --older-than 30d
```

**Safe Deletion with Backup:**
```bash
# Archive before deletion
/workspace delete geo_gse123456_old \
                  --archive ./archive/ \
                  --verify

# Delete with confirmation
/workspace delete "exploratory_*" \
                  --interactive  # Prompt for each file
```

#### Automated Cleanup Policies

**Define Cleanup Policy:**
```yaml
# cleanup_policy.yaml
policies:
  - name: delete_old_temp
    description: "Delete temporary files older than 7 days"
    conditions:
      pattern: "temp_*"
      age_days: 7
    action: delete

  - name: archive_unused
    description: "Archive datasets unused for 60 days"
    conditions:
      unused_days: 60
      min_size_mb: 100
    action: archive
    destination: ./archive/

  - name: compress_old_plots
    description: "Compress plots older than 30 days"
    conditions:
      type: plot
      age_days: 30
    action: compress

schedule: daily  # Run daily at midnight
retention:
  deleted_log: 90  # Keep deletion log for 90 days
```

**Apply Policy:**
```bash
# Apply cleanup policy
/workspace apply-policy cleanup_policy.yaml --dry-run
/workspace apply-policy cleanup_policy.yaml

# Run specific policy
/workspace apply-policy cleanup_policy.yaml --policy delete_old_temp
```

#### Quota Management

**Set Storage Quotas:**
```bash
# Set workspace quota
/workspace set-quota --size 20GB --warn-at 80%

# Set quota by dataset type
/workspace set-quota --type h5ad --size 15GB \
                     --type plots --size 3GB \
                     --type provenance --size 2GB
```

**Quota Enforcement:**
```python
# Automatic quota enforcement
from lobster.core.workspace_manager import WorkspaceManager

wm = WorkspaceManager(workspace_path="./my_workspace")

# Enable quota enforcement
wm.set_quota(
    total_size_gb=20,
    warn_threshold_pct=80,
    block_threshold_pct=95,
    auto_cleanup=True,          # Auto-delete old temp files
    cleanup_policy="cleanup_policy.yaml"
)

# Quota will automatically trigger cleanup when 80% reached
```

### 6. Multi-Workspace Workflows

#### Managing Multiple Workspaces

**Workspace Registry:**
```bash
# List all workspaces
/workspace list-all

# Register new workspace
/workspace register --path ./project_1/ --name liver_study
/workspace register --path ./project_2/ --name cancer_analysis

# Switch between workspaces
/workspace switch liver_study
/workspace switch cancer_analysis

# Show active workspace
/workspace current
```

**Workspace Registry Output:**
```
=== Registered Workspaces ===

┌───────────────────┬─────────────────────────┬──────────┬──────────┐
│ Name              │ Path                    │ Size     │ Status   │
├───────────────────┼─────────────────────────┼──────────┼──────────┤
│ liver_study ●     │ ./project_1/            │ 15.2 GB  │ Active   │
│ cancer_analysis   │ ./project_2/            │  8.7 GB  │ Inactive │
│ proteomics_cohort │ ./project_3/            │ 12.1 GB  │ Inactive │
└───────────────────┴─────────────────────────┴──────────┴──────────┘

Total: 3 workspaces, 36.0 GB used
```

#### Cross-Workspace Data Sharing

**Link Datasets Between Workspaces:**
```bash
# Link dataset from another workspace (read-only)
/workspace link --source liver_study:geo_gse123456 \
                --target current \
                --mode readonly

# Copy dataset to current workspace
/workspace copy --source cancer_analysis:processed_cohort \
                --target current
```

**Natural Language Alternative:**
```
"Link the GSE123456 dataset from my liver_study workspace"
"Copy the processed cohort data from cancer_analysis workspace"
```

#### Workspace Comparison

**Compare Workspaces:**
```bash
# Compare two workspaces
/workspace compare liver_study cancer_analysis

# Compare datasets
/workspace compare-datasets \
    --workspace1 liver_study:geo_gse123456_final \
    --workspace2 cancer_analysis:geo_gse987654_final
```

**Comparison Report:**
```
=== Workspace Comparison ===

Workspace 1: liver_study
Workspace 2: cancer_analysis

Datasets:
  Unique to liver_study: 12
  Unique to cancer_analysis: 8
  Shared (by name): 4
    - geo_gse111111
    - custom_controls
    - reference_atlas
    - quality_standards

Storage:
  liver_study: 15.2 GB
  cancer_analysis: 8.7 GB
  Difference: +6.5 GB (75% larger)

Analysis Pipelines:
  Common tools used: 8
  Unique to liver_study: 3 (trajectory analysis, pseudobulk, enrichment)
  Unique to cancer_analysis: 2 (survival analysis, CNV detection)
```

#### Workspace Synchronization

**Sync Workspaces Across Machines:**
```bash
# Push workspace to remote
/workspace sync --push \
                --destination s3://backup/workspaces/liver_study/

# Pull workspace updates from remote
/workspace sync --pull \
                --source s3://backup/workspaces/liver_study/ \
                --strategy merge  # or 'overwrite'

# Bidirectional sync
/workspace sync --bidirectional \
                --remote s3://backup/workspaces/liver_study/
```

**Sync Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `merge` | Combine changes from both sides | Collaborative work |
| `overwrite` | Replace local with remote | Reset to known state |
| `mirror` | Exact copy (delete removed files) | Backup/disaster recovery |
| `incremental` | Only transfer changes | Bandwidth optimization |

#### Multi-Workspace Batch Operations

**Batch Commands Across Workspaces:**
```bash
# Run cleanup on all workspaces
/workspace foreach --command cleanup --args "--dry-run --unused-days 60"

# Backup all workspaces
/workspace foreach --command backup --args "--destination ./backups/"

# Health check all workspaces
/workspace foreach --command health-check --output health_summary.json
```

**Aggregate Reporting:**
```bash
# Generate report across all workspaces
/workspace aggregate-report --output workspace_summary.html

# Monitor all workspaces
/workspace monitor --refresh-interval 60s  # Live dashboard
```

### Best Practices for Advanced Workspace Management

#### Backup Strategy

1. **3-2-1 Rule**: 3 copies, 2 different media types, 1 offsite
   ```bash
   # Local backup
   /workspace backup --name daily_backup --destination ./local_backup/

   # Remote backup (different medium)
   /workspace backup --name daily_backup --destination s3://backup/

   # Archive important milestones (offsite)
   /workspace backup --name milestone_v1 --destination gs://archive/
   ```

2. **Incremental Backups for Active Projects**: Save time and space
3. **Full Backups for Milestones**: Before publication, major releases
4. **Automated Schedules**: Daily incrementals, weekly fulls

#### Migration Planning

1. **Test Migrations**: Always use `--dry-run` first
2. **Verify Integrity**: Use checksums and validation
3. **Document Paths**: Record absolute paths for reproducibility
4. **Maintain Provenance**: Ensure provenance transfers correctly

#### Template Design

1. **Parameterize Everything**: Max flexibility for reuse
2. **Include Examples**: Provide sample configurations
3. **Version Templates**: Track template evolution
4. **Document Assumptions**: Specify expected input formats

#### Monitoring and Analytics

1. **Regular Health Checks**: Weekly for active projects
2. **Set Quotas Early**: Prevent runaway storage growth
3. **Track Access Patterns**: Identify unused data
4. **Review Provenance**: Ensure analysis lineage is complete

#### Cleanup Guidelines

1. **Archive Before Delete**: Preserve data you might need later
2. **Use Policies**: Automated cleanup reduces manual work
3. **Interactive Mode**: For important deletions, use `--interactive`
4. **Log Deletions**: Maintain audit trail of cleaned data

#### Multi-Workspace Organization

1. **Clear Naming**: Use descriptive workspace names
2. **Logical Separation**: One workspace per project or dataset
3. **Shared Standards**: Use templates for consistency
4. **Regular Sync**: Keep remote backups synchronized

## Workflow Best Practices

### General Principles

1. **Start with Data Quality**: Always assess data quality before analysis
2. **Iterative Approach**: Build analysis step-by-step
3. **Parameter Documentation**: Keep track of analysis parameters
4. **Validation**: Cross-validate results with multiple methods
5. **Visualization**: Generate plots at each major step

### Quality Control Guidelines

1. **Check Data Distribution**: Ensure appropriate data characteristics
2. **Assess Missing Values**: Handle missing data appropriately
3. **Batch Effect Detection**: Look for systematic biases
4. **Outlier Identification**: Handle outliers appropriately
5. **Normalization Validation**: Verify normalization effectiveness

### Statistical Considerations

1. **Multiple Testing Correction**: Always apply appropriate corrections
2. **Effect Size Reporting**: Report both significance and effect size
3. **Confidence Intervals**: Provide uncertainty estimates
4. **Sample Size Assessment**: Ensure adequate statistical power
5. **Assumption Validation**: Check statistical model assumptions

### Reproducibility Guidelines

1. **Parameter Recording**: Document all analysis parameters
2. **Version Control**: Track software and data versions
3. **Random Seeds**: Set seeds for reproducible results
4. **Session Export**: Save complete analysis sessions
5. **Method Documentation**: Record rationale for method choices

## Troubleshooting Common Issues

### Data Loading Problems

**Issue**: File format not recognized
```
# Solution: Check file format and convert if necessary
"Convert this Excel file to a format suitable for analysis"
```

**Issue**: Large file loading slowly
```
# Solution: Use streaming or chunked loading
"Load this large dataset efficiently in chunks"
```

### Analysis Issues

**Issue**: Poor clustering results
```
# Solution: Adjust parameters or try different methods
"The clusters look over-fragmented, can you try different resolution parameters?"
```

**Issue**: No significant results
```
# Solution: Check power and adjust thresholds
"I'm not getting significant results, can you assess the statistical power and suggest improvements?"
```

### Interpretation Challenges

**Issue**: Unexpected biological results
```
# Solution: Literature validation and quality assessment
"These results seem unexpected, can you check the literature and validate the analysis?"
```

**Issue**: Complex statistical output
```
# Solution: Request explanation and visualization
"Can you explain these statistics in simpler terms and create visualizations?"
```

This comprehensive workflow guide covers the major analysis types supported by Lobster AI. Each workflow can be customized based on specific research questions and data characteristics.