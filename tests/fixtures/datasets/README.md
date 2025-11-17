# Test Dataset Repository

## Overview
This directory contains **12 real biological datasets** from GEO for comprehensive testing.
All dataset metadata is stored in `datasets.yml` for easy maintenance.

## Configuration Structure
```yaml
omics_type:           # single_cell, bulk_rnaseq, edge_cases
  analysis_type:      # clustering, differential_expression, platform_rejection
    platform:         # 10x_chromium, standard_counts, microarray
      - id: GSE...    # Dataset identifier
        ...metadata...
```

## Usage

### Download datasets
```bash
# Download all datasets (~5-10 GB)
python download_datasets.py --all

# Download specific omics type
python download_datasets.py --omics-type single_cell

# Download specific analysis type
python download_datasets.py --omics-type bulk_rnaseq --analysis-type differential_expression

# Download single dataset
python download_datasets.py --dataset GSE132044
```

### Access in tests
```python
from tests.fixtures.datasets.dataset_manager import get_dataset_manager

dm = get_dataset_manager()

# Get dataset path
path = dm.get('single_cell/clustering/10x_chromium/GSE132044')

# Get metadata
metadata = dm.get_metadata('single_cell/clustering/10x_chromium/GSE132044')

# Find datasets by tag
clustering_datasets = dm.list_by_tag('clustering')

# List all datasets for a type
bulk_datasets = dm.list_by_type('bulk_rnaseq')
```

## Adding New Datasets
1. Edit `datasets.yml` and add entry under appropriate omics_type/analysis_type/platform
2. Run `python download_datasets.py --dataset YOUR_GEO_ID`
3. Tests automatically have access via DatasetManager

## Directory Structure (after download)
```
datasets/
├── datasets.yml              # Configuration (single source of truth)
├── dataset_manager.py        # Helper class for test access
├── download_datasets.py      # Automated downloader
├── README.md                 # This file
├── single_cell/
│   ├── clustering/
│   │   ├── 10x_chromium/
│   │   │   └── GSE132044/
│   │   └── matrix_market/
│   │       └── GSE144735/
│   ├── quality_control/
│   │   └── small_datasets/
│   │       └── GSE117089/
│   └── performance/
│       └── large_datasets/
│           └── GSE154763/
├── bulk_rnaseq/
│   ├── differential_expression/
│   │   ├── standard_counts/
│   │   │   ├── GSE147507/
│   │   │   └── GSE130036/
│   │   └── complex_design/
│   │       └── GSE137710/
│   └── quantification/
│       └── kallisto/
│           └── GSE114762/
└── edge_cases/
    ├── platform_rejection/
    │   ├── microarray/
    │   │   ├── GSE42057/
    │   │   └── GSE100618/
    ├── format_specific/
    │   └── soft/
    │       └── GSE48968/
    └── ambiguous/
        └── generated/
            ├── ambiguous_100x100.csv
            ├── ambiguous_50x80.csv
            └── ambiguous_with_headers.csv
```

## Dataset Details

### Single-Cell RNA-seq (4 datasets)
- **GSE132044**: 10X Chromium format, melanoma, ~15,000 cells
- **GSE144735**: Matrix Market format, COVID-19 PBMC, ~44,000 cells
- **GSE117089**: Small dataset for fast tests, kidney, ~3,000 cells
- **GSE154763**: Large dataset for performance, tumor, ~120,000 cells

### Bulk RNA-seq (4 datasets)
- **GSE147507**: COVID-19 DE, 24 samples
- **GSE130036**: Standard counts, 8 samples
- **GSE137710**: Complex time-series design, 48 samples
- **GSE114762**: Kallisto quantification files, 12 samples

### Edge Cases (4 datasets)
- **GSE42057**: Microarray platform (GPL570) - should reject
- **GSE100618**: Mixed platforms - should reject
- **GSE48968**: SOFT format (legacy GEO)
- **Ambiguous datasets**: Generated matrices for ambiguity detection

## Storage Requirements
- Total size: ~5-10 GB (depending on extracted archives)
- Single-cell datasets: ~3-5 GB
- Bulk RNA-seq: ~1-2 GB
- Format-specific: ~500 MB
- Edge cases: ~100 MB (mostly metadata + generated files)

## Maintenance
- Datasets are downloaded ONCE and reused for all tests
- Update `datasets.yml` when adding new test scenarios
- Document dataset-specific quirks in this file
- Use `.gitignore` to exclude large data files from git tracking
