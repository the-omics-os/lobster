# Data Formats Guide

## Overview

Lobster AI supports a wide range of biological data formats for different omics types. This guide provides detailed specifications for supported input and output formats, including format conversion capabilities and best practices.

## Supported Input Formats

### Single-Cell RNA-seq Formats

#### H5AD (AnnData HDF5)
**Description**: Standard format for single-cell data, used by scanpy and other Python tools.

**File Extension**: `.h5ad`

**Structure**:
```
AnnData object with:
- X: Expression matrix (cells × genes)
- obs: Cell metadata (cell barcodes, QC metrics, clusters)
- var: Gene metadata (gene symbols, chromosome, biotype)
- obsm: Multi-dimensional cell annotations (PCA, UMAP coordinates)
- varm: Multi-dimensional gene annotations
- layers: Additional expression matrices (raw counts, normalized)
- uns: Unstructured annotations (parameters, plots)
```

**Example Loading**:
```bash
/read single_cell_data.h5ad
```

**Advantages**:
- Efficient storage with compression
- Preserves all analysis metadata
- Native format for scanpy workflows
- Supports both sparse and dense matrices

#### 10X Genomics Formats

**10X HDF5 Format**
- **File Extension**: `.h5`
- **Structure**: HDF5 file with matrix, features, and barcodes
- **Loading**: `/read filtered_feature_bc_matrix.h5`

**10X MTX Format**
- **Files**: `matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`
- **Structure**: Market Matrix format with separate metadata files
- **Loading**: `/read /path/to/filtered_feature_bc_matrix/`

**10X CSV Format**
- **Files**: CSV/TSV files with gene expression matrix
- **Structure**: Genes as rows, cells as columns (or transposed)

#### CSV/TSV Formats
**Structure Options**:
1. **Genes as rows, cells as columns**:
   ```
   gene_id,cell_1,cell_2,cell_3,...
   ENSG00000001,10,5,0,...
   ENSG00000002,0,15,3,...
   ```

2. **Cells as rows, genes as columns**:
   ```
   cell_id,ENSG00000001,ENSG00000002,...
   cell_1,10,0,...
   cell_2,5,15,...
   ```

**Loading**:
```bash
/read expression_matrix.csv
/read expression_matrix.tsv
```

**Auto-detection**: Lobster AI automatically detects orientation and format.

#### Excel Formats
**File Extensions**: `.xlsx`, `.xls`

**Structure**: Expression matrix with optional metadata sheets

**Example Loading**:
```bash
/read single_cell_data.xlsx
```

### Bulk RNA-seq Formats

#### Quantification File Formats (Kallisto/Salmon)

**Kallisto abundance.tsv Format**
```
target_id	length	eff_length	est_counts	tpm
ENST00000456328	1657	1497	0	0
ENST00000450305	632	472	10.5	3.2
ENST00000488147	1351	1191	125.8	15.4
```

**Directory Structure**:
```
quantification_directory/
├── sample1/
│   └── abundance.tsv
├── sample2/
│   └── abundance.tsv
└── sample3/
    └── abundance.tsv
```

**Loading Kallisto Data**:

**⚠️ NEW in v0.2+**: Use CLI `/read` command directly for quantification files.

```bash
/read /path/to/kallisto_output
```

**Salmon quant.sf Format**
```
Name	Length	EffectiveLength	TPM	NumReads
ENST00000456328	1657	1497.000	0.000	0.000
ENST00000450305	632	472.000	3.215	10.500
ENST00000488147	1351	1191.000	15.432	125.800
```

**Loading Salmon Data**:

**⚠️ NEW in v0.2+**: Use CLI `/read` command directly for quantification files.

```bash
/read /path/to/salmon_output
```

**Key Features**:
- **Automatic Tool Detection**: System detects Kallisto vs Salmon from file patterns
- **Per-Sample Merging**: Automatically merges quantification from multiple samples
- **Correct Orientation**: Transposes to samples × genes (bulk RNA-seq standard)
- **Metadata Preservation**: Extracts sample names from directory structure
- **Quality Validation**: Verifies quantification file integrity and consistency

**Supported File Names**:
- **Kallisto**: `abundance.tsv`, `abundance.h5`, `abundance.txt`
- **Salmon**: `quant.sf`, `quant.genes.sf`

#### Count Matrices

**CSV/TSV Count Matrix**
```
gene_id,sample_1,sample_2,sample_3,sample_4
ENSG00000001,150,200,175,220
ENSG00000002,0,5,2,8
ENSG00000003,1200,1500,1300,1800
```

**Requirements**:
- Raw or normalized counts
- Gene identifiers (Ensembl, Symbol, etc.)
- Sample identifiers as column headers

#### DESeq2 Format
**Structure**: Compatible with DESeq2 input requirements
- Integer count values (for raw counts)
- Gene metadata optional
- Sample metadata in separate file

#### Metadata Files
**Sample Metadata**:
```
sample_id,condition,batch,replicate
sample_1,control,batch1,1
sample_2,control,batch1,2
sample_3,treatment,batch2,1
sample_4,treatment,batch2,2
```

**Gene Metadata**:
```
gene_id,gene_symbol,biotype,chromosome
ENSG00000001,DDX11L1,processed_transcript,chr1
ENSG00000002,WASH7P,unprocessed_pseudogene,chr1
```

### Mass Spectrometry Proteomics Formats

#### MaxQuant Output

**proteinGroups.txt**
- **Description**: Main MaxQuant output file with protein quantification
- **Key Columns**:
  - `Protein IDs`: UniProt identifiers
  - `Gene names`: Gene symbols
  - `Intensity <sample>`: Raw protein intensities
  - `LFQ intensity <sample>`: Label-free quantified intensities
  - `Razor + unique peptides`: Peptide counts

**Loading**:
```bash
/read proteinGroups.txt
```

**peptides.txt**
- **Description**: Peptide-level quantification
- **Usage**: For peptide-level analysis or filtering

#### Spectronaut Output (Biognosys)

**Description**: Spectronaut is a leading commercial software for DIA-MS (Data-Independent Acquisition) proteomics analysis. Lobster AI provides comprehensive support for both Spectronaut export formats with automatic format detection.

**Supported Formats**: `.tsv`, `.txt`, `.csv`, `.xls`, `.xlsx`

**Format 1: Long Format (Recommended)**
```
R.FileName	PG.ProteinGroups	PG.Genes	PG.Quantity	PG.Qvalue
Sample1.raw	P12345	EGFR	1234567.8	0.001
Sample1.raw	P67890;P67891	KRAS	987654.3	0.002
Sample2.raw	P12345	EGFR	1456789.2	0.001
Sample2.raw	P67890;P67891	KRAS	1098765.4	0.003
```

**Key Columns**:
- `R.FileName` / `Run` / `File.Name`: Sample identifier (with .raw, .d, .wiff extensions)
- `PG.ProteinGroups`: Semicolon-separated protein IDs (protein groups)
- `PG.Genes`: Gene symbols (semicolon-separated for protein groups)
- `PG.Quantity`: Linear-scale protein abundance (NOT log-transformed)
- `PG.Qvalue`: FDR-corrected Q-value at protein group level
- `PG.Normalised` / `PG.NormalizedQuantity`: Alternative quantification columns

**Format 2: Matrix Format**
```
PG.ProteinGroups	PG.Genes	Sample1	Sample2	Sample3
P12345	EGFR	1234567.8	1456789.2	1678901.3
P67890	KRAS	987654.3	1098765.4	1209876.5
```

**Parser Features**:
- **Automatic Format Detection**: Distinguishes between long and matrix formats
- **Q-value Filtering**: Configurable FDR threshold (default: 0.01 = 1% FDR)
- **Log2 Transformation**: Optional with configurable pseudocount (default: 1.0)
- **Sample Name Cleaning**: Removes file extensions (.raw, .d, .wiff, .wiff2, .mzML)
- **Protein Group Handling**: Extracts representative protein ID from semicolon-separated groups
- **Gene Symbol Indexing**: Uses gene symbols as row names for biologist-friendly output
- **Contaminant Filtering**: Detects CON__, KERATIN, TRYP_ patterns
- **Reverse Hit Filtering**: Identifies decoy database hits (REV__, _rev patterns)
- **Missing Value Handling**: Zeros converted to NaN (consistent across formats)
- **Aggregation Methods**: sum, mean, median, max for precursor-to-protein rollup

**Loading**:
```bash
# Basic loading with defaults
/read spectronaut_report.tsv

# Advanced: Custom Q-value threshold and log transformation
"Load spectronaut_report.tsv with Q-value threshold of 0.05 and apply log2 transformation"

# Matrix format
/read spectronaut_matrix.csv
```

**Parser Parameters** (when requesting through natural language):
- `qvalue_threshold`: Maximum Q-value for filtering (default: 0.01)
- `quantity_column`: Which quantification column to use ("auto", "PG.Quantity", "PG.Normalised")
- `log_transform`: Apply log2 transformation (default: True)
- `pseudocount`: Value added before log transformation (default: 1.0)
- `use_genes_as_index`: Use gene symbols as var index (default: True)
- `filter_contaminants`: Remove contaminant proteins (default: True)
- `filter_reverse`: Remove reverse/decoy hits (default: True)
- `aggregation_method`: Precursor aggregation ("sum", "mean", "median", "max")

**Output Structure** (AnnData):
- **X**: Intensity matrix (samples × proteins), float32, NaN for missing
- **obs**: Sample metadata (sample_name)
- **var**: Protein metadata
  - `protein_groups`: Full protein group string
  - `protein_id`: Representative protein ID
  - `gene_symbols`: Gene symbol string
  - `n_precursors`: Precursor count (long format only)
  - `mean_q_value`: Mean Q-value
  - `is_contaminant`: Contaminant flag
  - `is_reverse`: Reverse/decoy flag
- **uns**: Parsing metadata (format_type, qvalue_threshold, etc.)

**Common Use Cases**:

```bash
# Biognosys pilot workflow - strict filtering
"Load biognosys_spectronaut.tsv with Q-value threshold 0.01, log2 transform, and remove contaminants"

# Exploratory analysis - relaxed filtering
"Load spectronaut_results.tsv with Q-value 0.05, no contaminant filtering"

# Matrix format with custom normalization
"Load spectronaut_matrix.xlsx using PG.Normalised column without log transformation"
```

**Quality Control Notes**:
- **Q-values**: Values >1.0 are automatically filtered (invalid)
- **Missing Values**: 0-50% typical for DIA-MS, higher indicates issues
- **Sample Correlations**: Should be >0.75 for good technical reproducibility
- **Dynamic Range**: Typically 4-6 orders of magnitude after log2 transformation

#### Generic Proteomics Format

**Intensity Matrix**:
```
protein_id,sample_1,sample_2,sample_3,sample_4
P12345,1200.5,1500.2,1300.8,1800.1
Q67890,800.3,950.7,750.2,1100.4
```

**Requirements**:
- Protein identifiers (UniProt, gene symbols)
- Quantitative values (intensities, ratios)
- Missing values as NA, NaN, or empty

### Affinity Proteomics Formats

#### Olink NPX Data

**CSV Format**:
```
SampleID,UniProt,Assay,NPX,Panel
Sample_1,P12345,IL6,5.2,Inflammation
Sample_1,Q67890,TNF,4.8,Inflammation
Sample_2,P12345,IL6,5.5,Inflammation
```

**Structure**:
- **NPX Values**: Normalized protein expression
- **Panel Information**: Olink panel designation
- **UniProt IDs**: Protein identifiers
- **Assay Names**: Protein assay identifiers

#### Antibody Array Data

**Intensity Matrix**:
```
sample_id,protein_1,protein_2,protein_3,...
control_1,1500,2200,800,...
control_2,1600,2100,750,...
treatment_1,2200,3500,1200,...
```

**Metadata Requirements**:
- Antibody validation information
- Protein identifiers
- Sample annotations

### Multi-Omics Formats

#### MuData (Multi-modal AnnData)
**File Extension**: `.h5mu`

**Description**: Stores multiple omics modalities in single file

**Structure**:
```
MuData object with:
- mod['rna']: Transcriptomics AnnData
- mod['protein']: Proteomics AnnData
- mod['atac']: Chromatin accessibility AnnData
- obs: Shared sample metadata
- var: Combined feature metadata
```

**Loading**:
```bash
/read multiomics_data.h5mu
```

#### Integrated CSV Formats
**Separate Files for Each Modality**:
- `transcriptomics.csv`
- `proteomics.csv`
- `metadata.csv`

**Sample Matching**: Common sample identifiers across files

### Metadata Formats

#### Sample Metadata

**Standard Format**:
```
sample_id,condition,batch,age,gender,replicate
sample_1,control,batch1,25,female,1
sample_2,control,batch1,27,male,2
sample_3,treatment,batch2,24,female,1
```

**Required Columns**:
- `sample_id`: Unique sample identifier
- Additional columns as needed for experimental design

**Supported Data Types**:
- Categorical: condition, batch, gender
- Numerical: age, dose, time
- Date/time: collection_date, processing_time

#### Feature Metadata

**Gene Metadata**:
```
gene_id,gene_symbol,biotype,chromosome,start,end
ENSG00000001,DDX11L1,processed_transcript,chr1,11869,14409
```

**Protein Metadata**:
```
protein_id,gene_symbol,protein_name,molecular_weight
P12345,IL6,Interleukin-6,23.7
```

### GEO Database Integration

#### Automatic GEO Download
**Usage**:
```bash
"Download GSE12345 from GEO database"
```

**Supported GEO Formats**:
- Series Matrix Files (`GSE*_series_matrix.txt.gz`)
- Supplementary Files (various formats)
- Platform Annotations (`GPL*`)

**Processing**:
- Automatic format detection
- Metadata extraction
- Sample annotation processing
- Expression matrix reconstruction

#### Manual GEO Files
**Loading Downloaded Files**:
```bash
/read GSE12345_series_matrix.txt.gz
/archive GSE12345_RAW.tar  # Extract and process archived samples
```

### Archive Formats

#### Supported Archive Types

**TAR Archives**
- **File Extensions**: `.tar`, `.tar.gz`, `.tar.bz2`
- **Compression**: Supports gzip and bzip2 compression
- **Usage**: Common for GEO RAW files and multi-sample datasets

**ZIP Archives**
- **File Extensions**: `.zip`
- **Compression**: Standard ZIP compression
- **Usage**: Alternative archive format for data distribution

#### Archive Content Detection

Lobster AI automatically detects and processes multiple bioinformatics formats within archives:

**10X Genomics Archives**
- **V3 Chemistry**: `matrix.mtx`, `features.tsv`, `barcodes.tsv`
- **V2 Chemistry**: `matrix.mtx`, `genes.tsv`, `barcodes.tsv`
- **Compression**: Handles both `.gz` compressed and uncompressed files
- **Structure**: Automatically detects nested sample directories

**Example Structure**:
```
GSE155698_RAW.tar
├── GSM4701116_PDAC_PBMC_01/
│   ├── matrix.mtx.gz
│   ├── features.tsv.gz        # V3 chemistry
│   └── barcodes.tsv.gz
├── GSM4701131_PDAC_PBMC_16/
│   ├── matrix.mtx.gz
│   ├── genes.tsv.gz           # V2 chemistry
│   └── barcodes.tsv.gz
└── ... (additional samples)
```

**Kallisto/Salmon Quantification Archives**
- **Kallisto Files**: `abundance.tsv`, `abundance.h5`, `abundance.txt`
- **Salmon Files**: `quant.sf`, `quant.genes.sf`
- **Requirements**: Multiple sample subdirectories (≥2 samples)
- **Auto-Merge**: Automatically combines all samples into unified dataset

**Example Structure**:
```
kallisto_results.tar.gz
├── sample_1/
│   └── abundance.tsv
├── sample_2/
│   └── abundance.tsv
└── sample_3/
    └── abundance.tsv
```

**GEO RAW Expression Files**
- **Pattern**: `GSM<digits>_*.txt`, `GSM<digits>_*.txt.gz`
- **Format**: Expression matrices or quantification files
- **Metadata**: Automatically extracts sample information from filenames

#### Loading Archives

**Basic Usage**:
```bash
/archive /path/to/archive.tar
/archive /path/to/archive.tar.gz
/archive /path/to/archive.zip
```

**Features**:
- **Smart Content Detection**: Identifies data format without full extraction
- **Memory Efficiency**: Streaming extraction for large archives
- **Multi-Sample Processing**: Automatic sample concatenation
- **Format Mixing**: Handles archives with V2 and V3 chemistry mixed
- **Progress Tracking**: Real-time status updates during processing

**Example Workflow**:
```bash
# Load GEO archive with multiple 10X samples
/archive GSE155698_RAW.tar

# System automatically:
# 1. Inspects archive contents (no extraction yet)
# 2. Detects 17 10X Genomics samples (V2 and V3 mixed)
# 3. Extracts each sample efficiently
# 4. Loads and concatenates all samples
# 5. Result: 94,371 cells × 32,738 genes

# Load Kallisto quantification archive
/archive kallisto_batch_results.tar.gz

# System automatically:
# 1. Detects multiple abundance.tsv files
# 2. Identifies Kallisto format
# 3. Merges samples with proper orientation
# 4. Result: samples × genes count matrix
```

#### Archive Processing Pipeline

**Step 1: Manifest Inspection**
- Fast archive contents scan
- File pattern matching
- Format identification

**Step 2: Content Type Detection**
- 10X Genomics (V2/V3 detection)
- Kallisto/Salmon quantification
- GEO RAW expression files
- Generic expression matrices

**Step 3: Extraction Strategy**
- Memory-efficient streaming
- Selective extraction (only needed files)
- Nested archive handling

**Step 4: Data Loading**
- Format-specific loaders
- Sample concatenation
- Metadata preservation
- Quality validation

#### Archive vs Individual Files

**Use `/archive` for**:
- TAR/ZIP compressed archives
- Multi-sample datasets (10X, Kallisto, Salmon)
- GEO RAW downloads
- Nested directory structures

**Use `/read` for**:
- Individual H5AD files
- Single CSV/Excel files
- Uncompressed directories
- Pre-extracted data

#### Archive Format Validation

**Quality Checks**:
- File completeness (all required files present)
- Format consistency (matching structures across samples)
- Compression integrity
- Data type validation

**Error Handling**:
- Missing required files (e.g., missing `barcodes.tsv`)
- Corrupted archives
- Unsupported nested structures
- Mixed incompatible formats

## Output Formats

### Analysis Results

#### H5AD Output
**Generated Data**:
- Processed expression matrices
- Quality control metrics
- Clustering results
- Dimensionality reduction coordinates
- Differential expression results

**Professional Naming Convention**:
```
geo_gse12345_quality_assessed.h5ad
geo_gse12345_filtered_normalized.h5ad
geo_gse12345_clustered.h5ad
geo_gse12345_annotated.h5ad
```

#### CSV Export
**Differential Expression Results**:
```
gene_id,gene_symbol,log2FoldChange,pvalue,padj,baseMean
ENSG00000001,DDX11L1,2.5,0.001,0.05,150.2
ENSG00000002,WASH7P,-1.8,0.002,0.06,89.7
```

**Cluster Annotations**:
```
cell_id,cluster,cell_type,confidence
cell_1,0,Hepatocyte,0.95
cell_2,1,Stellate_Cell,0.87
```

### Visualization Outputs

#### Interactive HTML Plots
**Format**: Plotly HTML files
**Features**:
- Zoom, pan, hover information
- Publication-quality rendering
- Embedded metadata

**Example Files**:
- `plot_1_UMAP_clusters.html`
- `plot_2_volcano_plot.html`
- `plot_3_quality_metrics.html`

#### Static Image Exports
**Formats**: PNG, PDF, SVG
**Usage**: Publications and presentations
**Resolution**: High-resolution (300+ DPI)

### Session Exports

#### Complete Data Package
**Format**: ZIP archive
**Contents**:
- All processed data files (H5AD format)
- Generated plots (HTML and PNG)
- Analysis metadata and parameters
- Technical summary report
- Provenance information

**Structure**:
```
lobster_analysis_package_20240115_143022.zip
├── modalities/
│   ├── dataset_processed.h5ad
│   ├── dataset_processed.csv
│   └── dataset_metadata.json
├── plots/
│   ├── plot_1_clusters.html
│   ├── plot_1_clusters.png
│   └── index.json
├── technical_summary.md
├── workspace_status.json
└── provenance.json
```

#### Session State
**Format**: JSON metadata
**Content**: Analysis parameters, tool usage history, session information

## Format Conversion Capabilities

### Automatic Conversion

Lobster AI automatically handles format conversion during loading:

#### Single-Cell Conversions
- **CSV/Excel → AnnData**: Matrix orientation detection and conversion
- **10X → AnnData**: MTX format to AnnData with metadata
- **H5 → AnnData**: 10X HDF5 to AnnData format

#### Bulk RNA-seq Conversions
- **CSV → Count Matrix**: Proper gene/sample orientation
- **Excel → Multiple Sheets**: Extract expression and metadata

#### Proteomics Conversions
- **MaxQuant → Standard Matrix**: Extract relevant columns
- **Wide → Long Format**: Reshape for analysis tools
- **Missing Value Handling**: Consistent NA representation

### Manual Conversion Requests

```bash
# Convert Excel to CSV
"Convert this Excel file to CSV format for analysis"

# Reshape data matrix
"Transpose this matrix so genes are rows and samples are columns"

# Extract specific columns
"Extract only the LFQ intensity columns from this MaxQuant file"

# Merge files
"Combine the expression data with the sample metadata file"
```

## Data Validation and Quality Checks

### Automatic Validation

#### Structure Validation
- **Matrix Dimensions**: Consistent row/column counts
- **Data Types**: Numeric values in expression matrices
- **Identifiers**: Valid gene/protein/sample IDs
- **Missing Values**: Appropriate handling of NA values

#### Content Validation
- **Expression Ranges**: Biologically reasonable values
- **Count Data**: Non-negative values for count matrices
- **Metadata Consistency**: Matching sample identifiers
- **Format Compliance**: Standard field requirements

### Quality Assessments

#### Single-Cell Data
- **Gene Detection**: Minimum genes per cell
- **Cell Quality**: Mitochondrial content, doublet detection
- **Library Complexity**: UMI and gene count distributions

#### Bulk RNA-seq Data
- **Library Sizes**: Total count distributions
- **Gene Detection**: Expressed genes per sample
- **Batch Effects**: PCA-based assessment

#### Proteomics Data
- **Missing Value Patterns**: MNAR vs MCAR assessment
- **Coefficient of Variation**: Technical reproducibility
- **Dynamic Range**: Protein intensity distributions

## Best Practices

### Data Preparation

#### File Organization
```
project/
├── raw_data/
│   ├── expression_matrix.csv
│   ├── sample_metadata.csv
│   └── gene_annotations.csv
├── processed_data/
└── results/
```

#### Naming Conventions
- **Descriptive Names**: Include data type, condition, date
- **No Spaces**: Use underscores instead of spaces
- **Version Control**: Include version numbers for iterations

#### Metadata Standards
- **Complete Annotations**: All relevant experimental factors
- **Consistent Identifiers**: Use standard gene/protein IDs
- **Missing Data**: Explicit NA values, never empty strings

### Format Selection

#### Choose Based on Analysis Type
- **H5AD**: Single-cell analysis workflows
- **CSV**: Simple bulk RNA-seq experiments
- **Excel**: Small datasets with multiple annotation sheets
- **HDF5**: Large datasets requiring compression

#### Consider Downstream Tools
- **scanpy**: H5AD format preferred
- **DESeq2**: Count matrices with integer values
- **Custom Analysis**: CSV for maximum compatibility

### Performance Considerations

#### Large Datasets
- **Compression**: Use compressed formats (H5AD, HDF5)
- **Sparse Matrices**: Appropriate for single-cell data
- **Chunked Loading**: For very large files
- **Memory Management**: Monitor memory usage during loading

#### Network Transfer
- **Compressed Files**: Reduce transfer time
- **Batch Loading**: Multiple small files vs. single large file
- **Cloud Storage**: Consider cloud-native formats

## Troubleshooting Common Issues

### Loading Problems

#### "File format not recognized"
**Cause**: Unsupported or malformed file format
**Solution**:
```bash
# Check file structure
"What format is this file and how can I load it?"

# Manual format specification
"Load this file treating it as a CSV with genes as rows"
```

#### "Inconsistent dimensions"
**Cause**: Matrix dimensions don't match metadata
**Solution**:
```bash
# Validate data structure
"Check if my expression matrix matches the sample metadata"

# Fix dimension mismatch
"Transpose this matrix to match the metadata"
```

### Data Quality Issues

#### "High percentage of missing values"
**Cause**: Poor data quality or incorrect format interpretation
**Solution**:
```bash
# Assess missing value patterns
"Analyze the missing value patterns in this proteomics data"

# Apply appropriate handling
"Handle missing values using MNAR imputation for this MS data"
```

#### "No variance in expression data"
**Cause**: Data may be pre-normalized or log-transformed
**Solution**:
```bash
# Check data distribution
"Examine the distribution of expression values"

# Apply appropriate preprocessing
"Skip normalization since this data appears pre-normalized"
```

### Format Compatibility

#### "Cannot convert between formats"
**Cause**: Incompatible data structures or missing information
**Solution**:
```bash
# Identify conversion requirements
"What information do I need to convert this data to AnnData format?"

# Provide missing metadata
"Use default gene symbols for missing gene annotations"
```

This comprehensive data formats guide covers all major biological data formats supported by Lobster AI, providing detailed specifications and best practices for effective data analysis.