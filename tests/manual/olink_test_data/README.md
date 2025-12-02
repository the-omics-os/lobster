# Olink Test Datasets

## Overview

This directory contains publicly available Olink proteomics test datasets for parser development and validation.

## Files

### 1. olink_npx_data1_example.csv
- **Size**: 5.1 MB
- **Rows**: 29,441 (including header)
- **Columns**: 17
- **Source**: OlinkAnalyze R package (official Olink package)
- **Panel**: Olink Cardiometabolic
- **Format**: Long format (one row per protein per sample)
- **Data Type**: Simulated NPX data for demonstration

**Experimental Design**:
- 158 unique samples (156 + 2 control sample repeats)
- 1,104 unique assays (proteins)
- 3 timepoints: Baseline, Week 6, Week 12
- 2 treatment groups: Treated, Untreated
- 5 sites: Site_A through Site_E
- Project ID: data1

### 2. olink_npx_data2_example.csv
- **Size**: 6.2 MB
- **Rows**: 32,385 (including header)
- **Columns**: 17
- **Source**: OlinkAnalyze R package
- **Panel**: Olink Cardiometabolic
- **Format**: Long format
- **Data Type**: Simulated NPX data (follow-up dataset)

**Experimental Design**:
- Different subject cohort from data1
- Same panel and timepoint structure
- Project ID: data2
- Designed for bridge normalization examples

## Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `SampleID` | String | Unique sample identifier |
| `Index` | Integer | Row index |
| `OlinkID` | String | Olink-specific protein identifier |
| `UniProt` | String | UniProt protein accession |
| `Assay` | String | Protein/assay name (gene symbol) |
| `MissingFreq` | Float | Frequency of missing values for this assay |
| `Panel_Version` | String | Olink panel version (e.g., "v.1201") |
| `PlateID` | String | Plate identifier |
| `QC_Warning` | String | QC flag ("Pass", "Warning", or "Fail") |
| `LOD` | Float | Limit of detection for this assay |
| `NPX` | Float | **Normalized Protein eXpression** (main measurement) |
| `Subject` | String | Subject/participant ID |
| `Treatment` | String | Treatment group ("Treated", "Untreated") |
| `Site` | String | Study site |
| `Time` | String | Timepoint ("Baseline", "Week.6", "Week.12") |
| `Project` | String | Project identifier |
| `Panel` | String | Olink panel name |

## NPX (Normalized Protein eXpression)

NPX is Olink's proprietary unit for protein expression:
- Log2 scale
- Arbitrary units (relative, not absolute concentrations)
- Normalized using internal and external controls
- Higher values = higher protein abundance
- Typical range: -5 to 25 NPX units

## Data Characteristics

### Quality Control
- Contains both "Pass" and "Warning" QC flags
- Includes control samples (e.g., "CONTROL_SAMPLE_AS 1")
- LOD (Limit of Detection) varies by assay

### Missing Data
- Control samples have NA for Subject/Treatment/Site/Time metadata
- MissingFreq indicates proportion of samples below LOD

### Experimental Design
- Longitudinal design (3 timepoints)
- Multi-site study
- Treatment vs. control comparison
- Suitable for:
  - Differential expression testing
  - Longitudinal analysis
  - Batch effect detection
  - Quality control workflows

## Source Information

**Original Source**: OlinkAnalyze R package
- CRAN: https://cran.r-project.org/web/packages/OlinkAnalyze/
- GitHub: https://github.com/Olink-Proteomics/OlinkRPackage

**Extraction Date**: 2024-12-01

**Extraction Method**:
```r
install.packages("OlinkAnalyze")
library(OlinkAnalyze)
data(npx_data1)
data(npx_data2)
write.csv(npx_data1, "olink_npx_data1_example.csv", row.names = FALSE)
write.csv(npx_data2, "olink_npx_data2_example.csv", row.names = FALSE)
```

## License & Usage

These datasets are provided by Olink as example data within the OlinkAnalyze package for demonstration and testing purposes. The data is simulated/synthetic, not from real biological samples.

**Appropriate uses**:
- Parser development and testing
- Algorithm validation
- Documentation examples
- Unit tests
- Integration tests

**Not appropriate for**:
- Biological interpretation
- Publication of results
- Method benchmarking (use real data)

## Parser Implementation Notes

### Expected Parsing Behavior

1. **Format Detection**: Auto-detect as Olink NPX format based on:
   - Presence of "NPX" column
   - Presence of "OlinkID" column
   - Long format structure

2. **Required Columns** (must parse successfully):
   - NPX (numeric)
   - SampleID (string)
   - OlinkID (string)
   - Assay (string)
   - UniProt (string)

3. **Optional Columns** (parse if present):
   - All metadata columns (Subject, Treatment, Site, Time, etc.)
   - QC columns (QC_Warning, LOD, MissingFreq)
   - Panel information

4. **Data Type Conversion**:
   - Numeric: NPX, LOD, MissingFreq, Index
   - Categorical: QC_Warning, Panel, Treatment, Site, Time
   - String: SampleID, OlinkID, UniProt, Assay, Subject

5. **Edge Cases to Handle**:
   - Control samples with NA metadata
   - QC_Warning values (Pass/Warning)
   - Variable LOD values per assay
   - Plate batch effects (PlateID)

### Integration with Lobster

For integration into `lobster/services/data_access/proteomics_parsers/`:

1. Create `OlinkNPXParser` class
2. Implement column mapping to standardized schema
3. Handle long-to-wide format conversion if needed
4. Validate data types and ranges
5. Preserve QC metadata in AnnData.obs

Example AnnData structure:
```
adata.obs: Sample metadata (SampleID, Subject, Treatment, Site, Time)
adata.var: Protein metadata (OlinkID, UniProt, Assay, LOD, MissingFreq)
adata.X: NPX matrix (samples × proteins)
adata.uns['qc']: QC warnings per sample
adata.uns['panel']: Panel information
```

## Testing Checklist

- [ ] Successfully load npx_data1.csv
- [ ] Successfully load npx_data2.csv
- [ ] Parse all 17 columns correctly
- [ ] Handle control samples with NA metadata
- [ ] Validate NPX numeric values
- [ ] Detect QC warnings
- [ ] Convert to AnnData format
- [ ] Preserve all metadata
- [ ] Handle missing values appropriately
- [ ] Performance test (should load in <5 seconds)

## Additional Resources

- Olink NPX Documentation: https://www.olink.com/resources-support/
- OlinkAnalyze Package Vignettes: https://cran.r-project.org/web/packages/OlinkAnalyze/vignettes/
- Olink Methodology: Proximity Extension Assay (PEA)

