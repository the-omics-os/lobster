# Olink NPX Data Implementation Research

**Date:** 2025-12-01
**Purpose:** Reference implementations for Olink affinity proteomics data parsing and analysis

---

## Executive Summary

This research identifies existing Python and R implementations for handling Olink Proximity Extension Assay (PEA) data. The **official OlinkRPackage** (R, 121 stars, AGPL-3.0) is the gold standard reference, with comprehensive format detection and parsing logic. Python implementations are limited but provide valuable patterns for pandas-based workflows.

---

## Reference Implementations

### 1. OlinkRPackage (R) - PRIMARY REFERENCE ⭐⭐⭐⭐⭐

**Repository:** https://github.com/Olink-Proteomics/OlinkRPackage
**Stars:** 121 | **License:** AGPL-3.0 | **Language:** R
**Maintainer:** Olink Proteomics Data Science Team (Official)
**Last Updated:** November 27, 2025 (actively maintained)

#### Key Features
- **Format Support:** Excel (Target 48/96, Flex), CSV (Explore), Parquet, ZIP
- **Version Detection:** 10+ header format versions with distance-based matching
- **Data Types:** NPX, QUANT, concentration data
- **LOD Handling:** Both FixedLOD and NCLOD (negative control-based)
- **Normalization:** Bridge samples, plate control, intensity normalization

#### Critical Files to Study
1. **`OlinkAnalyze/R/Read_NPX_data.R`** - Main router function
2. **`OlinkAnalyze/R/read_npx_csv.R`** - CSV parsing (long format)
3. **`OlinkAnalyze/R/read_npx_parquet.R`** - Parquet support
4. **`OlinkAnalyze/R/olink_lod.R`** - LOD calculation and handling
5. **`OlinkAnalyze/R/olink_normalization.R`** - Multi-dataset normalization

#### Format Detection Logic

**Platform Detection:**
```r
# CSV/TXT/ZIP/Parquet → Explore (long format)
# XLS/XLSX → Target/Flex (wide format)

if (tools::file_ext(filename) %in% c("csv", "txt", "zip", "parquet")) {
  read_NPX_explore(filename = filename)  # Long format
} else if (tools::file_ext(filename) %in% c("xls", "xlsx")) {
  read_NPX_target(filename = filename)   # Wide format
}
```

**Column Version Matching:**
```r
# Define expected headers for multiple versions
header_v <- list(
  "header_npx_v1" = c("SampleID", "OlinkID", "UniProt", "Assay",
                      "Panel", "PlateID", "NPX", "QC_Warning",
                      "Index", "MissingFreq", "LOD", "Panel_Version"),
  "header_npx_v1.1" = c(..., "Normalization", "Assay_Warning"),
  "header_npx_v3" = c(..., "Sample_Type", "Panel_Lot_Nr", "ExploreVersion"),
  # ... additional versions
)

# Calculate minimum distance to pick best matching version
header_diff_1 <- lapply(header_v, function(x) setdiff(x, colnames(out)))
header_diff_2 <- lapply(header_v, function(x) setdiff(colnames(out), x))
header_pick <- tidyr::tibble(v_name = names(header_v),
                              v = v1 + v2) %>%
  dplyr::arrange(v, v_name) %>%
  dplyr::slice_head(n = 1)
```

#### Expected Columns (Explore Format)

**Core columns:**
- `SampleID` - Sample identifier
- `OlinkID` - Olink-specific protein ID
- `UniProt` - UniProt accession
- `Assay` - Protein assay name
- `Panel` - Panel name (e.g., "Olink CARDIOMETABOLIC")
- `PlateID` - Plate identifier
- `NPX` - Normalized Protein eXpression value
- `QC_Warning` - QC status flag
- `LOD` - Limit of detection
- `MissingFreq` - Missing frequency across samples

**Additional columns (version-dependent):**
- `Index`, `Panel_Version`, `Normalization`, `Assay_Warning`
- `Sample_Type`, `Panel_Lot_Nr`, `ExploreVersion`
- `BelowLOD`, `AboveULOQ`, `BelowLQL` (logical flags)

#### LOD Handling Patterns

**NCLOD Calculation (from negative controls):**
```r
LODNPX = median(PCNormalizedNPX) + max(0.2, 3 * sd(PCNormalizedNPX))
LODCount = max(150, max(Count * 2))
```

**Key insight:** LOD values are **calculated and appended**, but measurements are **not automatically filtered**. Users choose their own censoring strategy.

#### License Implication
**AGPL-3.0:** Copyleft license requires derivative works to be open-sourced. Can reference logic patterns but must implement independently for proprietary use.

---

### 2. ProteomicsAnalysisPipeline (Python) - BEST PYTHON REFERENCE ⭐⭐⭐⭐

**Repository:** https://github.com/tjpel/ProteomicsAnalysisPipeline
**Stars:** 3 | **License:** GPL-3.0 | **Language:** Python
**Last Updated:** June 2, 2025

#### Key Features
- **Multi-Format Support:** Olink, TMT Phospho, TMT Protein List
- **Configuration-Driven:** JSON-based pipeline definition
- **Excel Input:** 4-sheet structure (Data, Sample Info, Study Groups, Notes)
- **Processing Pipeline:** Missing value filtering, normalization, imputation, transformation
- **Statistical Analysis:** Fold change, p-values, volcano plots, PCA, heatmaps

#### Critical Files
- **`analysis_pipeline/analysis_pipeline.py`** - Main pipeline orchestrator
- **`analysis_pipeline/helper_scripts/data_operations.py`** - Data loading (not public)
- **`analysis_pipeline/config.json`** - Pipeline configuration
- **`examples/Olink_Input_Example/`** - Sample data structure

#### Data Structure (Excel)

**Sheet 1: "Data"**
- Protein metadata columns + Sample columns (must start with `"Sample "`)
- Primary key for Olink: `"Assay"` column
- Values: NPX or concentration measurements

**Sheet 2: "Sample Information"**
- First column: `"Sample ID"` (matches Data sheet)
- Additional columns: Study group categories

**Sheet 3: "Study Group Information"**
- `Study Group` - Group name
- `Study Group ID` - Category identifier
- `Study Group Type` - "Case" or "Control"

**Sheet 4: "Notes"** (optional)

#### Processing Pipeline Stages

1. **Duplicate removal** - By primary key (Assay)
2. **Missing value filtering**:
   - Global: Remove proteins missing ≥X% across all samples
   - Groupwise: Remove proteins missing ≥X% in any study group
3. **Normalization**:
   - Median: `y_ij = x_ij - median(x_j)`
   - Mean: `y_ij = x_ij - mean(x_j)`
   - Total: `y_ij = x_ij / sum(x_i)`
   - Quantile: Distribution-based
4. **Imputation**: Replace NA with % of minimum (sample-wise or protein-wise)
5. **Transformation**: Log, Z-score

#### Configuration Example
```json
{
  "project_information": {
    "file_type": "Olink",
    "relative_path": "./project_dir",
    "raw_data_name": "data.xlsx"
  },
  "ordered_pipeline": [
    "Drop Duplicates",
    "Remove Proteins With >=50% Values Missing Per Group",
    "Median Normalization",
    "Impute Missing Values 5% Min Per Sample",
    "Z-Score Transformation"
  ]
}
```

#### License Implication
**GPL-3.0:** Copyleft license. Can study logic but must implement independently.

---

### 3. olink_data_preparation (Python) - SIMPLE CSV EXAMPLE ⭐⭐

**Repository:** https://github.com/RGroening/olink_data_preparation
**Stars:** 0 | **License:** MIT | **Language:** Python
**Last Updated:** January 26, 2023

#### Key Features
- **Simple CSV-based workflow**
- **Multi-file merging:** Proteomics + clinical + biomarker data
- **Temporal data handling:** Disease phase categorization
- **Patient-level aggregation**

#### Code Snippet - Data Loading
```python
# Load NPX data (wide format: samples as rows, proteins as columns)
data = pd.read_csv("20201969_Forsell_NPX_edit.csv")

# Load metadata
attributes = pd.read_csv("olink_sample_names.csv")
# Columns: Sample, PatientID, Day, Progress

# Merge and categorize
def append_attributes_new_categories(givendata, dataattributes):
    for ind in givendata.index:
        current = dataattributes.loc[
            dataattributes.Sample.astype(str) == str(givendata.Sample[ind])
        ]

        # Categorize by day
        if current["Day"].item() <= 10:
            phase = "1-10d"
        elif current["Day"].item() <= 30:
            phase = "11-30d"
        else:
            phase = ">30d"

        # Concat and append
        ...
```

#### Expected CSV Columns

**NPX data file:**
- `Sample` column + protein columns (arbitrary names)

**Attributes file:**
- `Sample`, `PatientID`, `Day`, `Progress`, `Dataset`

**Clinical data:**
- `PatientID`, demographic and clinical variables

#### License Implication
**MIT:** Permissive license. Can freely adapt and integrate patterns.

---

### 4. olink_data_analysis (Python) - MULTIVARIATE ANALYSIS ⭐⭐⭐

**Repository:** https://github.com/PrimalCerebrate/olink_data_analysis
**Stars:** 0 | **License:** Not specified | **Language:** Python
**Last Updated:** May 7, 2021

#### Key Features
- **Multivariate analysis:** PCA, t-SNE, correlation matrices
- **Clinical stratification:** Age, sex, disease severity, antibody levels
- **Pandas-based workflow**
- **Extensive visualizations:** 52 PDF outputs

#### Code Patterns - Data Loading & Transformation

**Loading:**
```python
data = pd.read_csv(
    ".../20201969_Forsell_NPX_edit.csv")
attributes = pd.read_csv(
    ".../olink_sample_names.csv")

# Additional metadata
antibodies = pd.read_csv(".../IgGIgAIgMELISAresults.csv", na_values=['NA'])
sexage = pd.read_csv(".../COVID_age_sex.csv")
```

**Column cleaning:**
```python
# Strip whitespace from column names
covum_attributes.columns = covum_attributes.columns.str.strip()
proteins = data.drop("Sample", axis=1).columns.str.strip()
```

**Merging attributes:**
```python
def append_attributes(givendata, dataattributes):
    add_data = pd.DataFrame(
        columns=["Sample", "Dataset", "Day", "Progress", "PatientID", "Phase"])

    for ind in givendata.index:
        currentattribute = dataattributes.loc[
            dataattributes.Sample.astype(str) == str(givendata.Sample[ind])
        ]

        # Categorize by day into phases
        if currentattribute["Day"].item() <= 14:
            phase = pd.Series(["Acute"], name="Phase")
        elif 15 <= currentattribute["Day"].item() < 41:
            phase = pd.Series(["Mid Convalescent"], name="Phase")
        else:
            phase = pd.Series(["Late Convalescent"], name="Phase")

        currentattribute = pd.concat([currentattribute, phase], axis=1)
        add_data = add_data.append(currentattribute, ignore_index=True)

    return add_data

# Merge
additionaldata = append_attributes(data, attributes)
data = pd.concat([data, additionaldata.drop("Sample", axis=1)], axis=1)
```

**Data selection and filtering:**
```python
# Filter by dataset
data_oneset = data[data["Dataset"] == "Umea"]

# Drop NaN for PCA
proteinlist = data.drop("Sample", axis=1).columns
only_concentration_data = data_oneset[proteinlist]
only_concentration_data = only_concentration_data.dropna(axis=1)

# Filter rows
data.drop(data[data["Progress"] == "Healthy"].index, inplace=True)
```

**Grouping operations:**
```python
groups = one_set.groupby("Progress")
patients = one_set.groupby("PatientID")

for name, group in groups:
    axis.scatter(group["Day"], group[protein], label=name)
```

**Standardization for PCA:**
```python
from sklearn.preprocessing import scale

standardised = scale(only_concentration_data)
standardised = pd.DataFrame(
    standardised,
    index=only_concentration_data.index,
    columns=only_concentration_data.columns
)
```

**Correlation analysis:**
```python
# Within-group correlation
corrmat = only_concentration_data.corr()

# Between-group correlation
corrmat_between = only_concentration_dataA.corrwith(
    only_concentration_dataB, axis=0
)
```

---

### 5. olinkpy (Python) - EMPTY REPOSITORY ⚠️

**Repository:** https://github.com/athro/olinkpy
**Stars:** 0 | **License:** Unknown | **Language:** Python
**Status:** Empty (created but not initialized)

Repository description promises "Python library for Olink proteomics data analysis" but contains no code. Monitor for future updates.

---

## Key Insights & Patterns

### Format Detection Strategy

**File extension-based routing:**
1. `.csv`, `.txt` → Long format (Explore platform)
2. `.xlsx`, `.xls` → Wide format (Target 48/96, Flex)
3. `.parquet` → Long format (modern Explore)
4. `.zip` → Compressed long format

**Column-based version detection:**
- Define expected column sets for known versions
- Calculate set difference distance (missing + extra columns)
- Pick version with minimum distance
- Warn about missing columns but proceed

### Long → Wide Transformation

**Explore data is already LONG format:**
- One row per sample-protein combination
- Columns: `SampleID`, `OlinkID`, `Assay`, `NPX`, `Panel`, etc.
- No pivot needed for analysis

**Target data requires LONG conversion:**
```r
# Wide format: samples as rows, proteins as columns
panel_list_long[[i]] <- panel_list[[i]] %>%
  dplyr::mutate(SampleID = SampleID) %>%
  tidyr::gather(Assay, NPX, -SampleID, -`QC Warning`, -`Plate ID`, -Index)
```

**Python equivalent (pandas):**
```python
# Wide → Long
df_long = df_wide.melt(
    id_vars=['SampleID', 'QC_Warning', 'PlateID'],
    var_name='Assay',
    value_name='NPX'
)
```

### Multi-Panel Handling

**OlinkRPackage approach:**
```r
# Detect number of panels
nr_panel <- readxl::read_excel(path = filename, range = 'B3', col_names = FALSE)

# Loop through panels (Base_Index = 45 for Target48, 92 for Target96)
for (i in 1:nr_panel) {
  panel_data[[i]] <- dat[,(2+((i-1)*BASE_INDEX)):((BASE_INDEX+1)+((i-1)*BASE_INDEX))]
  panel_list_long[[i]] <- panel_data[[i]] %>% tidyr::gather(...)
}

# Combine panels
combined <- dplyr::bind_rows(panel_list_long)
```

**Key insight:** Wide format files contain multiple panel blocks side-by-side (each 45 or 92 columns).

### QC Warning Handling

**Data includes but does not auto-filter:**
- `QC_Warning` column (categorical)
- `Assay_Warning` column (assay-specific flags)
- `BelowLOD`, `AboveULOQ`, `BelowLQL` (boolean flags)

**User decides filtering strategy:**
```r
# Example: Remove samples with QC warnings
clean_data <- data %>% filter(QC_Warning != "Warning")

# Example: Flag but keep below-LOD values
data <- data %>% mutate(censored = NPX < LOD)
```

### LOD Calculation Strategies

**1. FixedLOD (from reference file):**
- Pre-calculated LOD per assay
- Extracted from manufacturer data

**2. NCLOD (from negative controls):**
```r
# Requirements: ≥10 negative control samples with SampleQC = PASS
lod_data <- data %>% filter(SampleType == "NEGATIVE_CONTROL")

# Calculate per assay
LODNPX = median(PCNormalizedNPX) + max(0.2, 3 * sd(PCNormalizedNPX))
LODCount = max(150, max(Count * 2))
```

**3. Both methods (for comparison):**
- Append columns: `NCLOD`, `FixedLOD`, `NCPCNormalizedLOD`, `FixedPCNormalizedLOD`

---

## Recommended Implementation Strategy

### Phase 1: Data Loading (MVP)

**Priority:** Support Explore CSV format (most common for downloads)

1. **Format detection:**
   - Check file extension (`.csv`, `.xlsx`)
   - Validate presence of core columns: `SampleID`, `Assay`, `NPX`
   - Detect Olink by checking for `OlinkID` or `Panel` columns

2. **CSV parsing:**
   ```python
   df = pd.read_csv(filepath, na_values=['NA', ''])

   # Type conversions
   df['SampleID'] = df['SampleID'].astype(str)
   df['NPX'] = pd.to_numeric(df['NPX'], errors='coerce')

   # Handle MissingFreq as percentage
   if 'MissingFreq' in df.columns:
       df['MissingFreq'] = df['MissingFreq'].str.rstrip('%').astype(float) / 100
   ```

3. **Column standardization:**
   - Map known column variants to canonical names
   - Strip whitespace from column names
   - Validate required columns exist

4. **Convert to AnnData:**
   ```python
   # Pivot to wide format if needed
   df_wide = df.pivot(index='SampleID', columns='Assay', values='NPX')

   # Create AnnData
   adata = sc.AnnData(X=df_wide.values)
   adata.obs_names = df_wide.index
   adata.var_names = df_wide.columns

   # Store metadata
   adata.var['OlinkID'] = df.groupby('Assay')['OlinkID'].first()
   adata.var['UniProt'] = df.groupby('Assay')['UniProt'].first()
   adata.var['Panel'] = df.groupby('Assay')['Panel'].first()

   # Store LOD if present
   if 'LOD' in df.columns:
       adata.var['LOD'] = df.groupby('Assay')['LOD'].first()

   # Store QC flags
   if 'QC_Warning' in df.columns:
       adata.obs['QC_Warning'] = df.groupby('SampleID')['QC_Warning'].first()
   ```

### Phase 2: Quality Control

1. **Missing value assessment:**
   ```python
   missing_freq = (adata.X == np.nan).sum(axis=0) / adata.n_obs
   adata.var['missing_freq'] = missing_freq
   ```

2. **LOD-based filtering (optional):**
   ```python
   if 'LOD' in adata.var.columns:
       below_lod = adata.X < adata.var['LOD'].values
       adata.layers['below_LOD'] = below_lod
   ```

3. **QC warning filtering:**
   ```python
   if 'QC_Warning' in adata.obs.columns:
       adata_clean = adata[adata.obs['QC_Warning'] != 'Warning'].copy()
   ```

### Phase 3: Normalization

**Supported methods (match OlinkRPackage):**

1. **Median normalization:**
   ```python
   sample_medians = np.median(adata.X, axis=1, keepdims=True)
   adata.layers['normalized'] = adata.X - sample_medians
   ```

2. **Bridge normalization (for multi-dataset):**
   - Identify bridge samples present in both datasets
   - Calculate adjustment factor
   - Apply to align datasets

### Phase 4: Excel Support (Target/Flex)

1. **Read multi-sheet Excel:**
   ```python
   data_sheet = pd.read_excel(filepath, sheet_name='Data')
   sample_info = pd.read_excel(filepath, sheet_name='Sample Information')
   ```

2. **Detect platform (Target 48 vs 96):**
   ```python
   panel_name = pd.read_excel(filepath, sheet_name='Data',
                                usecols='B', nrows=3).iloc[2, 0]
   is_target48 = 'Target 48' in str(panel_name)
   base_index = 45 if is_target48 else 92
   ```

3. **Extract panel blocks and convert to long:**
   ```python
   nr_panels = detect_number_of_panels(data_sheet)

   panels = []
   for i in range(nr_panels):
       start_col = 2 + (i * base_index)
       end_col = start_col + base_index
       panel_data = data_sheet.iloc[:, start_col:end_col]

       # Melt to long format
       panel_long = panel_data.melt(
           id_vars=['SampleID'],
           var_name='Assay',
           value_name='NPX'
       )
       panel_long['Panel'] = f"Panel_{i+1}"
       panels.append(panel_long)

   df_long = pd.concat(panels, ignore_index=True)
   ```

---

## Code Snippets for Lobster Implementation

### 1. Format Detection

```python
def detect_olink_format(filepath: str) -> Optional[str]:
    """
    Detect Olink data format.

    Returns:
        'explore_csv': Explore platform CSV/TXT (long format)
        'target_excel': Target/Flex platform Excel (wide format)
        'explore_parquet': Explore platform Parquet (long format)
        None: Not Olink format
    """
    ext = Path(filepath).suffix.lower()

    if ext in ['.csv', '.txt']:
        # Read first few rows
        df_sample = pd.read_csv(filepath, nrows=5)

        # Check for Olink-specific columns
        olink_cols = ['OlinkID', 'Assay', 'NPX', 'Panel']
        if any(col in df_sample.columns for col in olink_cols):
            return 'explore_csv'

    elif ext in ['.xlsx', '.xls']:
        # Check for multi-sheet structure
        xl_file = pd.ExcelFile(filepath)

        if 'Data' in xl_file.sheet_names:
            data_sheet = pd.read_excel(filepath, sheet_name='Data', nrows=5)

            # Target format has samples starting with "Sample "
            sample_cols = [col for col in data_sheet.columns
                          if str(col).startswith('Sample ')]
            if sample_cols and 'Assay' in data_sheet.columns:
                return 'target_excel'

    elif ext == '.parquet':
        df_sample = pd.read_parquet(filepath, columns=None)
        df_sample = df_sample.head(5)

        olink_cols = ['OlinkID', 'Assay', 'NPX']
        if all(col in df_sample.columns for col in olink_cols):
            return 'explore_parquet'

    return None
```

### 2. CSV Loading (Explore Format)

```python
def load_olink_explore_csv(filepath: str) -> pd.DataFrame:
    """Load Olink Explore CSV format."""

    # Read CSV with NA handling
    df = pd.read_csv(
        filepath,
        na_values=['NA', '', 'nan'],
        dtype={'SampleID': str, 'Block': str}
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Type conversions
    numeric_cols = ['NPX', 'LOD', 'MissingFreq']
    for col in numeric_cols:
        if col in df.columns:
            if col == 'MissingFreq' and df[col].dtype == 'object':
                # Handle percentage strings
                df[col] = df[col].str.rstrip('%').astype(float) / 100
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Logical columns
    logical_cols = ['BelowLOD', 'AboveULOQ', 'BelowLQL']
    for col in logical_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df
```

### 3. Conversion to AnnData

```python
def olink_to_anndata(df: pd.DataFrame) -> sc.AnnData:
    """
    Convert Olink long-format DataFrame to AnnData.

    Args:
        df: Long-format DataFrame with columns:
            - SampleID, OlinkID, Assay, NPX, Panel, etc.

    Returns:
        AnnData object with:
            - X: NPX matrix (samples × proteins)
            - obs: Sample metadata
            - var: Protein metadata
            - uns: Experiment metadata
    """
    # Pivot to wide format
    df_wide = df.pivot_table(
        index='SampleID',
        columns='Assay',
        values='NPX',
        aggfunc='first'  # Handle duplicates
    )

    # Create AnnData
    adata = sc.AnnData(X=df_wide.values)
    adata.obs_names = df_wide.index
    adata.var_names = df_wide.columns

    # Add protein metadata to .var
    protein_meta_cols = ['OlinkID', 'UniProt', 'Panel', 'LOD',
                          'Panel_Version', 'Assay_Warning']

    for col in protein_meta_cols:
        if col in df.columns:
            # Take first value per Assay (should be constant)
            adata.var[col] = df.groupby('Assay')[col].first().reindex(adata.var_names)

    # Add sample metadata to .obs
    sample_meta_cols = ['PlateID', 'QC_Warning', 'Index', 'Sample_Type']

    for col in sample_meta_cols:
        if col in df.columns:
            adata.obs[col] = df.groupby('SampleID')[col].first().reindex(adata.obs_names)

    # Store MissingFreq (per sample-protein)
    if 'MissingFreq' in df.columns:
        missing_matrix = df.pivot_table(
            index='SampleID',
            columns='Assay',
            values='MissingFreq',
            aggfunc='first'
        )
        adata.layers['missing_freq'] = missing_matrix.reindex(
            index=adata.obs_names, columns=adata.var_names
        ).values

    # Store panel information in .uns
    panels = df['Panel'].unique().tolist() if 'Panel' in df.columns else []
    adata.uns['olink_panels'] = panels
    adata.uns['olink_format'] = 'explore'

    # Store QC flags as layer
    if 'BelowLOD' in df.columns:
        below_lod_matrix = df.pivot_table(
            index='SampleID',
            columns='Assay',
            values='BelowLOD',
            aggfunc='first'
        )
        adata.layers['below_LOD'] = below_lod_matrix.reindex(
            index=adata.obs_names, columns=adata.var_names
        ).fillna(False).values

    return adata
```

### 4. Version Detection

```python
def detect_olink_version(df: pd.DataFrame) -> str:
    """
    Detect Olink data format version based on columns.

    Returns version string like 'v1.0', 'v1.1', 'v3.0', etc.
    """
    header_versions = {
        'v1.0': {'SampleID', 'OlinkID', 'UniProt', 'Assay', 'Panel',
                 'PlateID', 'NPX', 'QC_Warning', 'Index', 'MissingFreq',
                 'LOD', 'Panel_Version'},

        'v1.1': {'SampleID', 'OlinkID', 'UniProt', 'Assay', 'Panel',
                 'PlateID', 'NPX', 'QC_Warning', 'Index', 'MissingFreq',
                 'LOD', 'Panel_Version', 'Normalization', 'Assay_Warning'},

        'v3.0': {'SampleID', 'OlinkID', 'UniProt', 'Assay', 'Panel',
                 'PlateID', 'NPX', 'QC_Warning', 'Index', 'MissingFreq',
                 'LOD', 'Sample_Type', 'Panel_Lot_Nr', 'Assay_Warning',
                 'Normalization', 'ExploreVersion'},
    }

    actual_cols = set(df.columns)

    # Calculate distance for each version
    distances = {}
    for version, expected_cols in header_versions.items():
        missing = len(expected_cols - actual_cols)
        extra = len(actual_cols - expected_cols)
        distances[version] = missing + extra

    # Return version with minimum distance
    best_match = min(distances, key=distances.get)

    # Warn if not exact match
    if distances[best_match] > 0:
        missing_cols = header_versions[best_match] - actual_cols
        extra_cols = actual_cols - header_versions[best_match]

        warnings.warn(
            f"Partial match to {best_match}. "
            f"Missing: {missing_cols}, Extra: {extra_cols}"
        )

    return best_match
```

### 5. LOD Handling

```python
def calculate_nclod(adata: sc.AnnData,
                    sample_type_col: str = 'Sample_Type') -> pd.Series:
    """
    Calculate LOD from negative controls (NCLOD method).

    Args:
        adata: AnnData with negative control samples
        sample_type_col: Column in .obs indicating sample type

    Returns:
        Series with LOD per protein
    """
    # Filter negative controls
    nc_mask = adata.obs[sample_type_col] == 'NEGATIVE_CONTROL'

    if 'QC_Warning' in adata.obs.columns:
        nc_mask &= (adata.obs['QC_Warning'] != 'Warning')

    if nc_mask.sum() < 10:
        raise ValueError(
            f"Need ≥10 negative controls, found {nc_mask.sum()}"
        )

    # Extract NC data
    nc_data = adata[nc_mask].X

    # Calculate LOD per protein: median + max(0.2, 3*SD)
    medians = np.median(nc_data, axis=0)
    stds = np.std(nc_data, axis=0)

    lod = medians + np.maximum(0.2, 3 * stds)

    return pd.Series(lod, index=adata.var_names, name='NCLOD')
```

---

## License Summary

| Repository | License | Commercial Use | Derivative Work Requirement |
|------------|---------|----------------|----------------------------|
| OlinkRPackage | AGPL-3.0 | ⚠️ Yes, if derivative is open-sourced | Must open-source |
| ProteomicsAnalysisPipeline | GPL-3.0 | ⚠️ Yes, if derivative is open-sourced | Must open-source |
| olink_data_preparation | MIT | ✅ Yes | None |
| olink_data_analysis | Unknown | ⚠️ Unclear | Unknown |

**Recommendation for Lobster:** Implement parsing logic independently by **referencing patterns** (not copying code) from AGPL/GPL projects. MIT-licensed code can be directly adapted.

---

## Additional Resources

### Official Olink Documentation
- **Olink.com**: https://www.olink.com/
- **NPX Manager Software**: Export tool for generating analysis files
- **Olink Explore Documentation**: Platform-specific file formats
- **Technical Notes**: LOD calculation methodologies

### Scientific Background
- **Proximity Extension Assay (PEA)**: Technology overview
- **NPX Values**: Normalized Protein eXpression scale (log2-based)
- **QC Guidelines**: Best practices for quality control

### Python Proteomics Ecosystem
- **pandas**: Core data manipulation
- **scanpy/anndata**: Single-cell/omics data structures (applicable to proteomics)
- **scikit-learn**: Normalization, PCA, clustering
- **statsmodels**: Statistical testing
- **plotly**: Interactive visualizations

---

## Recommended Next Steps for Lobster

1. **MVP Implementation:**
   - Create `OlinkModalityAdapter` in `lobster/core/interfaces/modality_adapter.py`
   - Implement CSV loading for Explore format (most common)
   - Support conversion to AnnData with proper metadata
   - Add format detection utility

2. **Service Integration:**
   - Extend `ProteomicsQualityService` with Olink-specific QC
   - Add LOD-based filtering options
   - Implement missing value assessment

3. **Normalization:**
   - Add median normalization to `ProteomicsPreprocessingService`
   - Implement bridge sample normalization for multi-dataset

4. **Excel Support (Phase 2):**
   - Add Target 48/96 format loader
   - Handle wide→long transformation
   - Support multi-panel extraction

5. **Testing:**
   - Create test fixtures with synthetic Olink data
   - Test version detection logic
   - Validate AnnData conversion

6. **Documentation:**
   - Add wiki page: "Working with Olink Data"
   - Document supported formats and limitations
   - Provide example workflows

---

## Conclusion

The official **OlinkRPackage** provides the gold-standard reference for format detection, parsing, and QC logic. Python implementations are limited but demonstrate pandas-based workflows for CSV handling, metadata merging, and multivariate analysis.

**Key takeaways:**
1. **Format detection** should be extension + column-based with fuzzy matching
2. **Long format** (Explore) is simpler to parse than wide format (Target)
3. **LOD values** are calculated but not automatically applied
4. **QC warnings** should be preserved as flags, not filters
5. **Multi-panel handling** requires parsing repeated column blocks in Excel

The Lobster implementation should prioritize **Explore CSV format** for MVP, then expand to Excel-based Target/Flex formats as needed.
