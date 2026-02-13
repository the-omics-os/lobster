# Research & Dataset Discovery

How to search literature and discover datasets using Lobster AI.

## Overview

**Agent**: Research Agent (literature) + Data Expert (loading)
**Capabilities**: PubMed, GEO, SRA, paper full-text extraction

## Literature Search

### Basic PubMed Search
```
"Search PubMed for CRISPR screens in cancer"
```

### With Filters
```
"Find papers about CAR-T therapy from 2023-2024"
```

### Specific Journals
```
"Search for single-cell papers in Nature and Cell from the last year"
```

### Author Search
```
"Find recent papers by Aviv Regev on single-cell"
```

## Dataset Discovery

### GEO Search
```
"Search GEO for single-cell pancreatic beta cell datasets"
```

### With Criteria
```
"Find GEO datasets with more than 10,000 cells and cell type annotations"
```

### Specific Organism
```
"Search for mouse liver single-cell datasets in GEO"
```

### By Technology
```
"Find 10X Genomics datasets for human PBMC"
```

## Paper → Dataset Extraction

### Extract GEO IDs from Papers
```
"Find papers about tumor microenvironment and extract their GEO datasets"
```

### Get Methods Section
```
"Get the methods and data availability from PMID:12345678"
```

### Full-Text Analysis
```
"Search for papers with available single-cell data and extract accession numbers"
```

## Dataset Validation

### Check Metadata
```
"Show metadata for GSE109564"
```

### Validate Fields
```
"Check if GSE200997 has cell_type and treatment metadata"
```

### Compare Datasets
```
"Compare the top 5 GEO results by sample count and metadata quality"
```

## Downloading Data

### Single Dataset
```
"Download GSE109564"
```

### With Processing
```
"Download GSE109564 and run initial quality control"
```

### Specific Samples
```
"Download only the treatment samples from GSE109564"
```

### From SRA
```
"Download SRP123456 from SRA"
```

## Complete Research Workflow

```bash
lobster chat --workspace ./literature_review

# Step 1: Find relevant papers
> "Search PubMed for single-cell liver fibrosis papers from 2022-2024"

# Step 2: Extract datasets
> "Extract GEO accession numbers from those papers"

# Step 3: Evaluate datasets
> "Show me metadata summary for the top 5 datasets"

# Step 4: Select best match
> "Which dataset has the most samples and best metadata?"

# Step 5: Download
> "Download that dataset"

# Step 6: Begin analysis
> "Run quality control on the loaded data"
```

## Advanced Queries

### Systematic Review Style
```
"Find all papers on scRNA-seq of human liver disease, extract methods, datasets, and key findings"
```

### Meta-analysis Prep
```
"Find comparable single-cell datasets for human PBMC that can be integrated"
```

### Replication Studies
```
"Find datasets similar to GSE109564 for validation"
```

## Data Sources

| Source | What's Available |
|--------|------------------|
| **GEO** | Expression datasets, metadata, processed files |
| **SRA** | Raw sequencing data (FASTQ) |
| **PubMed** | Paper abstracts, metadata, PMIDs |
| **PMC** | Full-text papers (when open access) |
| **ENA** | European sequencing archive |

## Tips

1. **Start broad, then narrow**: Begin with general terms, add filters
2. **Check metadata quality**: Good metadata = easier analysis
3. **Verify dataset contents**: Sample counts, cell types, conditions
4. **Use session continuity**: Build on previous search results
5. **Extract systematically**: Paper → accession → validation → download
