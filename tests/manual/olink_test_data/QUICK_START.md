# Quick Start: Using Olink Test Datasets

## TL;DR

**Best datasets for testing**: `olink_npx_data1_example.csv` and `olink_npx_data2_example.csv`
- Official Olink example data
- 5-6 MB each
- 17 columns, long format
- Already in this directory

## Load Data (Python)

```python
import pandas as pd

# Load dataset
df = pd.read_csv('olink_npx_data1_example.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Samples: {df['SampleID'].nunique()}")
print(f"Proteins: {df['Assay'].nunique()}")

# Preview
print(df.head())

# Key columns
npx_values = df['NPX']  # Main measurement
proteins = df['Assay']  # Protein names
samples = df['SampleID']  # Sample IDs
qc_status = df['QC_Warning']  # QC flags
```

## Convert to AnnData

```python
import pandas as pd
import anndata as ad

# Load data
df = pd.read_csv('olink_npx_data1_example.csv')

# Pivot to wide format (samples × proteins)
df_wide = df.pivot(index='SampleID', columns='Assay', values='NPX')

# Create AnnData object
adata = ad.AnnData(df_wide)

# Add sample metadata
sample_metadata = df.groupby('SampleID')[['Subject', 'Treatment', 'Site', 'Time', 'QC_Warning']].first()
adata.obs = sample_metadata

# Add protein metadata
protein_metadata = df.groupby('Assay')[['OlinkID', 'UniProt', 'LOD', 'MissingFreq', 'Panel']].first()
adata.var = protein_metadata

# Add panel info
adata.uns['panel'] = df['Panel'].iloc[0]
adata.uns['panel_version'] = df['Panel_Version'].iloc[0]

print(adata)
```

## Expected Output Structure

```
AnnData object with n_obs × n_vars = 158 × 1104
    obs: 'Subject', 'Treatment', 'Site', 'Time', 'QC_Warning'
    var: 'OlinkID', 'UniProt', 'LOD', 'MissingFreq', 'Panel'
    uns: 'panel', 'panel_version'
```

## Key Features to Test

1. **Long format parsing**: One row per protein per sample
2. **Multiple metadata levels**: Sample, protein, plate
3. **QC flags**: "Pass", "Warning" values
4. **Control samples**: Samples with NA metadata
5. **Multi-timepoint**: Longitudinal design
6. **Batch effects**: Multiple plates/sites

## Common Issues

### Issue: NA values in metadata
**Solution**: Control samples legitimately have NA for Subject/Treatment/Site/Time

### Issue: Duplicate (SampleID, Assay) pairs
**Solution**: Should NOT happen in these datasets - indicates parsing error

### Issue: NPX values out of expected range
**Solution**: Valid NPX range is approximately -5 to 25 (log2 scale)

## Validation Tests

```python
# Test 1: No duplicate measurements
assert df.duplicated(subset=['SampleID', 'Assay']).sum() == 0

# Test 2: NPX is numeric
assert pd.api.types.is_numeric_dtype(df['NPX'])

# Test 3: Expected columns present
required_cols = ['SampleID', 'NPX', 'Assay', 'OlinkID', 'UniProt']
assert all(col in df.columns for col in required_cols)

# Test 4: QC flags are valid
assert df['QC_Warning'].isin(['Pass', 'Warning', 'Fail']).all()

# Test 5: Reasonable NPX range
assert df['NPX'].between(-10, 30).all()

print("All validation tests passed!")
```

## Performance Benchmarks

Target performance metrics:
- **Load time**: <2 seconds
- **Parse time**: <3 seconds
- **Convert to AnnData**: <5 seconds
- **Total pipeline**: <10 seconds

## Next Steps

1. Implement `OlinkNPXParser` class
2. Run validation tests on both datasets
3. Test edge cases (control samples, QC warnings)
4. Integrate into `lobster/services/data_access/proteomics_parsers/`

## Files in This Directory

```
olink_test_data/
├── README.md                      # Comprehensive documentation
├── QUICK_START.md                 # This file
├── DATASET_SEARCH_SUMMARY.md      # Search methodology and findings
├── olink_npx_data1_example.csv    # Primary test dataset (5.1 MB)
└── olink_npx_data2_example.csv    # Secondary test dataset (6.2 MB)
```

## Additional Resources

- **Olink Documentation**: https://www.olink.com/resources-support/
- **OlinkAnalyze R Package**: https://cran.r-project.org/web/packages/OlinkAnalyze/
- **NPX Explanation**: https://www.olink.com/faq/what-is-npx/

