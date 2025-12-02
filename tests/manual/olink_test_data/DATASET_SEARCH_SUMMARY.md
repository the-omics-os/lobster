# Publicly Available Olink Datasets for Parser Testing

## Summary of Findings

After comprehensive search across multiple sources (GitHub, PRIDE, PubMed, Dryad, Zenodo, GEO), I found **limited publicly available Olink datasets** in standard formats suitable for parser testing. Most published Olink data is either:
1. Behind institutional/UK Biobank access controls
2. Embedded in supplementary materials (not raw NPX files)
3. Available in R package format (RDA) rather than CSV

---

## RECOMMENDED DATASETS

### 1. OlinkAnalyze R Package Example Data (BEST OPTION)
**Status**: ✅ Successfully extracted to CSV

**Access**: 
- R package: `install.packages("OlinkAnalyze")`
- Direct CSV extraction: Available at `/tmp/olink_npx_data1_example.csv` and `/tmp/olink_npx_data2_example.csv`

**Details**:
- **npx_data1.csv**: 5.1 MB, 29,441 rows × 17 columns
- **npx_data2.csv**: 6.2 MB, 32,385 rows × 17 columns
- **Panel**: Olink Cardiometabolic
- **Format**: Long format (one row per protein per sample)
- **Data type**: Simulated NPX data (generated for demo purposes)

**Column Structure**:
```
SampleID, Index, OlinkID, UniProt, Assay, MissingFreq, Panel_Version, 
PlateID, QC_Warning, LOD, NPX, Subject, Treatment, Site, Time, Project, Panel
```

**Why it's good for testing**:
- ✅ Clean, well-structured format
- ✅ Representative of real Olink output
- ✅ Includes QC metrics (QC_Warning, LOD, MissingFreq)
- ✅ Contains control samples
- ✅ Multiple timepoints and conditions
- ✅ Reasonable size for testing

**Download instructions**:
```r
install.packages("OlinkAnalyze")
library(OlinkAnalyze)
data(npx_data1)
data(npx_data2)
write.csv(npx_data1, "olink_npx_data1.csv", row.names = FALSE)
write.csv(npx_data2, "olink_npx_data2.csv", row.names = FALSE)
```

---

### 2. GitHub: Olink-Proteomics/OlinkRPackage
**URL**: https://github.com/Olink-Proteomics/OlinkRPackage

**Access**: Public repository

**Details**:
- Contains official Olink R package source code
- Data files: `npx_data1.rda`, `npx_data2.rda`, `manifest.rda`
- Format: R binary data (.rda)

**Why it's good for testing**:
- ✅ Official Olink-maintained example data
- ✅ Comprehensive documentation
- ✅ Representative of production data

**Download instructions**:
```bash
git clone https://github.com/Olink-Proteomics/OlinkRPackage.git
cd OlinkRPackage/OlinkAnalyze/data
# Use R to load .rda files
```

---

### 3. PRIDE Archive Olink Datasets (Limited)
**URL**: https://www.ebi.ac.uk/pride/archive/simpleSearch?q=olink

**Status**: ⚠️ Mixed - 18 results found but most are not pure Olink/NPX format

**Access**: Public (requires manual download)

**Details**:
- PRIDE accessions found: PXD020261, PXD060726, PXD059624, PAD000001-22
- Most datasets contain mass spectrometry data, not Olink NPX format
- May require additional processing to extract Olink-specific data

**Why it's challenging**:
- ❌ Mixed proteomics data types
- ❌ Not pure Olink NPX format
- ⚠️ Requires manual curation

---

### 4. Published Studies with Potential Data Access

#### A. UK Biobank Olink Pharma Proteomics Project
**Reference**: Eldjarn et al., Nature 2023 (doi:10.1038/s41586-023-06592-6)

**Details**:
- 53,022 participants
- Olink Explore 3072 platform
- NPX data for 3,000+ proteins

**Access**: ⚠️ Requires UK Biobank application (institutional access)

**Why it's not ideal for testing**:
- ❌ Restricted access (requires formal application)
- ❌ Large dataset (not suitable for quick testing)

#### B. ALSPAC Inflammation Proteomics
**Reference**: Goulding et al., Wellcome Open Research 2024

**Details**:
- Olink PEA technology
- Inflammation-related proteins

**Access**: ⚠️ Requires data access application

**Why it's not ideal**:
- ❌ Restricted access
- ⚠️ Unknown if raw NPX files available

---

## ALTERNATIVE SOURCES (NOT FOUND/INSUFFICIENT)

### Zenodo
- **Search**: "olink" + dataset filter
- **Result**: ❌ Unable to retrieve results (403 error)

### Figshare
- **Search**: "olink" 
- **Result**: ❌ Unable to retrieve specific Olink datasets

### GEO (Gene Expression Omnibus)
- **Search**: Olink proteomics
- **Result**: ❌ GEO primarily hosts transcriptomics data, not Olink proteomics

### Dryad
- **Search**: "olink"
- **Result**: ⚠️ Some results but access issues

---

## RECOMMENDATIONS FOR PARSER IMPLEMENTATION

### Primary Test Dataset
**Use OlinkAnalyze package data (npx_data1.csv and npx_data2.csv)**

**Rationale**:
1. ✅ Officially maintained by Olink
2. ✅ Representative structure and format
3. ✅ Easily accessible (no institutional barriers)
4. ✅ Appropriate size for testing (5-6 MB)
5. ✅ Includes edge cases (control samples, QC warnings)

### Test Coverage Strategy

**Essential columns to parse**:
- `NPX` (normalized protein expression) - core measurement
- `SampleID` - sample identifier
- `OlinkID` - protein identifier (Olink-specific)
- `UniProt` - standard protein ID
- `Assay` - protein name
- `QC_Warning` - quality control flag
- `LOD` - limit of detection
- `MissingFreq` - missing data frequency

**Optional metadata columns**:
- `Subject`, `Treatment`, `Site`, `Time` - experimental design
- `Panel`, `Panel_Version` - assay details
- `PlateID` - batch information

### Parser Validation Tests

1. **Basic parsing**: Load npx_data1.csv successfully
2. **Column detection**: Identify all 17 standard columns
3. **Data type validation**: Verify numeric (NPX, LOD), string (SampleID, Assay), factor (Panel)
4. **QC flag handling**: Parse "Pass"/"Warning" values
5. **Missing data**: Handle control samples with NA metadata
6. **Long format**: Verify correct handling of one-row-per-protein-per-sample format

---

## FILE LOCATIONS (Current System)

- `/tmp/olink_npx_data1_example.csv` (5.1 MB)
- `/tmp/olink_npx_data2_example.csv` (6.2 MB)

