# Services API Reference

## Overview

The Services API provides stateless analysis services implementing scientific algorithms for bioinformatics workflows. All services follow the stateless pattern, accepting AnnData objects as input and returning a tuple of (processed_adata, statistics_dict). This design ensures reproducibility, testability, and easy integration with the agent system.

## Service Design Pattern

All services follow the standard stateless pattern:

```python
class ExampleService:
    """Stateless service for biological data analysis."""

    def __init__(self):
        """Initialize the service (no state stored)."""
        pass

    def analyze(
        self,
        adata: anndata.AnnData,
        **kwargs
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform analysis on AnnData object.

        Args:
            adata: Input AnnData object
            **kwargs: Analysis parameters

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Processed data and statistics
        """
        # Process data
        processed_adata = self._process_data(adata, **kwargs)

        # Calculate statistics
        statistics = self._calculate_statistics(processed_adata, adata, **kwargs)

        return processed_adata, statistics
```

## Transcriptomics Services

### PreprocessingService

Advanced preprocessing service for single-cell RNA-seq data.

```python
class PreprocessingService:
    """
    Advanced preprocessing service for single-cell RNA-seq data.

    This stateless service provides methods for ambient RNA correction, quality control filtering,
    normalization, and batch correction/integration following best practices.
    """
```

#### Methods

##### correct_ambient_rna

```python
def correct_ambient_rna(
    self,
    adata: anndata.AnnData,
    contamination_fraction: float = 0.1,
    empty_droplet_threshold: int = 100,
    method: str = "simple_decontamination"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Correct for ambient RNA contamination using simplified decontamination methods.

**Parameters:**
- `adata` (anndata.AnnData): AnnData object with raw UMI counts
- `contamination_fraction` (float): Expected fraction of ambient RNA (0.05-0.2 typical)
- `empty_droplet_threshold` (int): Minimum UMI count to consider droplet as cell-containing
- `method` (str): Method to use ('simple_decontamination', 'quantile_based')

**Returns:**
- `Tuple[anndata.AnnData, Dict[str, Any]]`: Corrected AnnData and processing stats

##### filter_cells_and_genes

```python
def filter_cells_and_genes(
    self,
    adata: anndata.AnnData,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    max_genes_per_cell: int = None,
    max_pct_mito: float = 20.0,
    max_pct_ribo: float = None
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Filter cells and genes based on quality metrics.

**Parameters:**
- `min_genes_per_cell` (int): Minimum genes expressed per cell
- `min_cells_per_gene` (int): Minimum cells expressing each gene
- `max_genes_per_cell` (int): Maximum genes per cell (removes potential doublets)
- `max_pct_mito` (float): Maximum mitochondrial gene percentage
- `max_pct_ribo` (float): Maximum ribosomal gene percentage

##### normalize_data

```python
def normalize_data(
    self,
    adata: anndata.AnnData,
    target_sum: float = 1e4,
    normalization_method: str = "log1p",
    highly_variable_genes: bool = True,
    n_top_genes: int = 2000
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Normalize expression data and identify highly variable genes.

**Parameters:**
- `target_sum` (float): Target sum for normalization
- `normalization_method` (str): Method ('log1p', 'sqrt', 'none')
- `highly_variable_genes` (bool): Whether to identify highly variable genes
- `n_top_genes` (int): Number of highly variable genes to identify

### QualityService

Quality assessment service for single-cell data.

```python
class QualityService:
    """Service for assessing data quality with comprehensive metrics."""
```

#### Methods

##### assess_quality_comprehensive

```python
def assess_quality_comprehensive(
    self,
    adata: anndata.AnnData,
    organism: str = "human",
    include_scrublet: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform comprehensive quality assessment including doublet detection.

**Parameters:**
- `organism` (str): Organism type for gene set analysis ('human', 'mouse')
- `include_scrublet` (bool): Whether to include Scrublet doublet detection

### ClusteringService

Clustering service for single-cell RNA-seq data.

```python
class ClusteringService:
    """Stateless service for clustering single-cell RNA-seq data."""
```

#### Methods

##### cluster_and_visualize

```python
def cluster_and_visualize(
    self,
    adata: anndata.AnnData,
    resolution: Optional[float] = None,
    use_rep: Optional[str] = None,
    batch_correction: bool = False,
    batch_key: Optional[str] = None,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    umap_min_dist: float = 0.5,
    random_state: int = 42
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform clustering and dimensionality reduction with UMAP visualization.

**Parameters:**
- `resolution` (float): Clustering resolution for Leiden algorithm
- `use_rep` (str): Representation to use for clustering ('X_pca', 'X_harmony')
- `batch_correction` (bool): Whether to apply batch correction
- `batch_key` (str): Column name for batch information
- `n_pcs` (int): Number of principal components
- `n_neighbors` (int): Number of neighbors for graph construction
- `umap_min_dist` (float): UMAP minimum distance parameter

### EnhancedSinglecellService

Enhanced single-cell analysis service with advanced features.

```python
class EnhancedSinglecellService:
    """Enhanced service for advanced single-cell analysis workflows."""
```

#### Methods

##### detect_doublets_comprehensive

```python
def detect_doublets_comprehensive(
    self,
    adata: anndata.AnnData,
    expected_doublet_rate: float = 0.1,
    use_scrublet: bool = True,
    use_doubletfinder_alternative: bool = True,
    n_neighbors: int = None,
    n_pcs: int = 30
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Comprehensive doublet detection using multiple methods.

##### find_marker_genes

```python
def find_marker_genes(
    self,
    adata: anndata.AnnData,
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 100,
    reference: str = "rest",
    min_fold_change: float = 1.5,
    max_pval_adj: float = 0.05
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Find marker genes for clusters or groups using statistical testing.

### BulkRNAseqService

Service for bulk RNA-seq analysis with pyDESeq2 integration.

```python
class BulkRNAseqService:
    """Service for bulk RNA-seq differential expression analysis."""
```

#### Methods

##### run_deseq2_analysis

```python
def run_deseq2_analysis(
    self,
    adata: anndata.AnnData,
    design_formula: str,
    condition_col: str,
    reference_level: str = None,
    batch_col: str = None,
    min_count: int = 10,
    alpha: float = 0.05
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Run differential expression analysis using pyDESeq2.

**Parameters:**
- `design_formula` (str): R-style formula for experimental design
- `condition_col` (str): Column name for the main condition
- `reference_level` (str): Reference level for comparison
- `batch_col` (str): Column name for batch effects
- `min_count` (int): Minimum count threshold
- `alpha` (float): Significance threshold

### DifferentialFormulaService

Service for R-style formula construction and design matrix generation.

```python
class DifferentialFormulaService:
    """Service for constructing and validating R-style formulas for differential analysis."""
```

#### Methods

##### construct_formula

```python
def construct_formula(
    self,
    adata: anndata.AnnData,
    primary_condition: str,
    covariates: List[str] = None,
    interactions: List[Tuple[str, str]] = None,
    formula_type: str = "additive"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Construct and validate R-style formula for differential analysis.

**Parameters:**
- `primary_condition` (str): Main condition of interest
- `covariates` (List[str]): Additional covariates to include
- `interactions` (List[Tuple[str, str]]): Interaction terms
- `formula_type` (str): Type of formula ('additive', 'interaction')

### PseudobulkService

Service for aggregating single-cell data to pseudobulk.

```python
class PseudobulkService:
    """Service for converting single-cell data to pseudobulk for differential expression."""
```

#### Methods

##### create_pseudobulk

```python
def create_pseudobulk(
    self,
    adata: anndata.AnnData,
    sample_col: str,
    cluster_col: str = None,
    min_cells: int = 10,
    aggregation_method: str = "sum"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Convert single-cell data to pseudobulk samples.

**Parameters:**
- `sample_col` (str): Column identifying individual samples
- `cluster_col` (str): Optional column for cell type-specific pseudobulk
- `min_cells` (int): Minimum cells required per pseudobulk sample
- `aggregation_method` (str): Method for aggregation ('sum', 'mean')

## Proteomics Services

### ProteomicsPreprocessingService

Preprocessing service for proteomics data.

```python
class ProteomicsPreprocessingService:
    """Service for preprocessing proteomics data including missing value handling."""
```

#### Methods

##### handle_missing_values

```python
def handle_missing_values(
    self,
    adata: anndata.AnnData,
    missing_strategy: str = "hybrid",
    imputation_method: str = "knn",
    filter_threshold: float = 0.7,
    min_valid_values: int = 3
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Handle missing values in proteomics data with multiple strategies.

**Parameters:**
- `missing_strategy` (str): Strategy ('filter', 'impute', 'hybrid')
- `imputation_method` (str): Method for imputation ('knn', 'mice', 'mean')
- `filter_threshold` (float): Threshold for filtering features with too many missing values
- `min_valid_values` (int): Minimum valid values required per feature

##### normalize_intensities

```python
def normalize_intensities(
    self,
    adata: anndata.AnnData,
    method: str = "tmm",
    log_transform: bool = True,
    center_median: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Normalize protein intensities using various methods.

**Parameters:**
- `method` (str): Normalization method ('tmm', 'quantile', 'vsn', 'median')
- `log_transform` (bool): Whether to apply log transformation
- `center_median` (bool): Whether to center by median

### ProteomicsQualityService

Quality assessment service for proteomics data.

```python
class ProteomicsQualityService:
    """Service for assessing proteomics data quality."""
```

#### Methods

##### assess_data_quality

```python
def assess_data_quality(
    self,
    adata: anndata.AnnData,
    cv_threshold: float = 0.3,
    missing_threshold: float = 0.5
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Comprehensive quality assessment for proteomics data.

**Parameters:**
- `cv_threshold` (float): Coefficient of variation threshold
- `missing_threshold` (float): Missing value threshold for quality flags

### ProteomicsAnalysisService

Analysis service for proteomics data.

```python
class ProteomicsAnalysisService:
    """Service for proteomics statistical analysis and pathway enrichment."""
```

#### Methods

##### perform_differential_analysis

```python
def perform_differential_analysis(
    self,
    adata: anndata.AnnData,
    group_col: str,
    reference_group: str = None,
    method: str = "limma",
    adjust_method: str = "BH"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform differential protein expression analysis.

**Parameters:**
- `group_col` (str): Column for grouping samples
- `reference_group` (str): Reference group for comparison
- `method` (str): Statistical method ('limma', 't-test', 'wilcoxon')
- `adjust_method` (str): Multiple testing correction method

## Utility Services

### GEOService

Service for downloading and processing GEO datasets.

```python
class GEOService:
    """Service for fetching and processing GEO datasets."""
```

#### Methods

##### fetch_metadata_only

```python
def fetch_metadata_only(
    self,
    geo_id: str,
    include_sample_info: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Fetch metadata for a GEO dataset without downloading expression data.

**Parameters:**
- `geo_id` (str): GEO accession number
- `include_sample_info` (bool): Whether to include detailed sample information

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`: Metadata and validation results

##### download_and_process

```python
def download_and_process(
    self,
    geo_id: str,
    sample_limit: Optional[int] = None,
    concatenation_strategy: str = "guided"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Download and process GEO dataset with guided concatenation.

### ContentAccessService ✨ (v2.4.0+ Phase 1-6 Complete)

**Unified publication access service** with capability-based provider routing, three-tier cascade logic, and comprehensive literature mining. Replaces PublicationService and UnifiedContentService with a modular provider architecture.

```python
class ContentAccessService:
    """
    Unified publication access service with capability-based routing.

    Provides 10 core methods organized into 4 categories:
    - Discovery (3): search_literature, discover_datasets, find_linked_datasets
    - Metadata (2): extract_metadata, validate_metadata
    - Content (3): get_abstract, get_full_content, extract_methods
    - System (1): query_capabilities

    Features:
    - 5 specialized providers with automatic routing
    - Three-tier cascade: PMC XML → Webpage → PDF
    - Session caching via DataManager
    - W3C-PROV provenance tracking
    """
```

**New in v2.4.0:**
- ✅ **Provider Architecture**: 5 providers (Abstract, PubMed, GEO, PMC, Webpage) with capability-based routing
- ✅ **ProviderRegistry**: Priority-based provider selection (Priority 10 = high, 50 = low)
- ✅ **Three-Tier Cascade**: PMC (500ms) → Webpage (2-5s) → PDF (3-8s) with automatic fallback
- ✅ **10 Core Methods**: Comprehensive API for discovery, metadata, content, and system queries
- ✅ **Dataset Integration**: GEO/SRA dataset discovery with validation
- ✅ **Accession Detection**: Auto-detect GSM/GSE/GDS/GPL accessions with parent series lookup
- ✅ **DataManager-First Caching**: Session cache + workspace persistence

**Architecture:**
```
ContentAccessService (Coordination Layer)
    ↓
ProviderRegistry (Capability-Based Routing)
    ↓
5 Providers:
  - AbstractProvider (Priority: 10) - Fast abstracts (200-500ms)
  - PubMedProvider (Priority: 10) - Literature search (1-3s)
  - GEOProvider (Priority: 10) - Dataset discovery (2-5s)
  - PMCProvider (Priority: 10) - PMC XML (500ms-2s, 30-40% coverage)
  - WebpageProvider (Priority: 50) - Webpage/PDF fallback (2-8s)
    ↓
DataManagerV2 (Session Caching + Provenance)
```

#### Discovery Methods (3)

##### search_literature

```python
def search_literature(
    self,
    query: str,
    max_results: int = 5,
    sources: Optional[list[str]] = None,
    filters: Optional[dict[str, any]] = None,
    **kwargs
) -> Tuple[str, Dict[str, Any], AnalysisStep]
```

Search PubMed, bioRxiv, medRxiv for literature with capability-based routing.

**Parameters:**
- `query` (str): Search query string
- `max_results` (int): Maximum results to return (default: 5)
- `sources` (Optional[list[str]]): Provider names to use (e.g., ["pubmed"]). If None, uses all SEARCH_LITERATURE providers
- `filters` (Optional[dict]): Provider-specific filters (publication_year, organism, etc.)
- `**kwargs`: Additional parameters passed to providers

**Returns:**
- `Tuple[str, Dict[str, Any], AnalysisStep]`:
  - **str**: Formatted search results with publications
  - **Dict**: Statistics (query, max_results, provider_used, results_count, execution_time_ms)
  - **AnalysisStep**: Lightweight IR for provenance (exportable=False)

**Example:**
```python
service = ContentAccessService(data_manager)

# Basic literature search
results, stats, ir = service.search_literature("BRCA1 breast cancer")
print(f"Found {stats['results_count']} papers in {stats['execution_time_ms']}ms")

# With source filter
results, stats, ir = service.search_literature("p53", sources=["pubmed"])

# With year filter
results, stats, ir = service.search_literature(
    "single-cell RNA-seq",
    max_results=10,
    filters={"publication_year": "2023"}
)
```

**Performance:** 1-3s typical (PubMedProvider)

##### discover_datasets

```python
def discover_datasets(
    self,
    query: str,
    dataset_type: "DatasetType",
    max_results: int = 5,
    filters: Optional[dict[str, str]] = None
) -> Tuple[str, Dict[str, Any], AnalysisStep]
```

Search for omics datasets with **automatic accession detection**. Auto-detects direct accessions (GSM/GSE/GDS/GPL) and provides enhanced information including parent series for sample IDs.

**Parameters:**
- `query` (str): Search query or direct accession (e.g., "GSM6204600")
- `dataset_type` (DatasetType): Type of dataset to search for (DatasetType.GEO, DatasetType.SRA, etc.)
- `max_results` (int): Maximum results (default: 5)
- `filters` (Optional[dict]): Provider-specific filters (organism, platform, etc.)

**Returns:**
- `Tuple[str, Dict[str, Any], AnalysisStep]`:
  - **str**: Formatted dataset search results
  - **Dict**: Statistics (query, dataset_type, accession_detected, normalized_accession, results_count, execution_time_ms)
  - **AnalysisStep**: Lightweight IR for provenance (exportable=False)

**Example:**
```python
from lobster.tools.providers.base_provider import DatasetType

# Direct accession (auto-detected)
results, stats, ir = service.discover_datasets("GSM6204600", DatasetType.GEO)
if stats['accession_detected']:
    print(f"Parent series: {stats.get('parent_series', 'N/A')}")

# Text search
results, stats, ir = service.discover_datasets(
    "single-cell RNA-seq breast cancer",
    DatasetType.GEO,
    max_results=10,
    filters={"organism": "human"}
)
```

**Accession Detection:**
- GSM (sample): Lookup parent series (GSE)
- GSE (series): Direct lookup
- GDS (dataset): Direct lookup
- GPL (platform): Direct lookup

**Performance:** 2-5s typical (GEOProvider)

##### find_linked_datasets

```python
def find_linked_datasets(
    self,
    identifier: str,
    dataset_types: Optional[list["DatasetType"]] = None,
    include_related: bool = True
) -> str
```

Find datasets linked to a publication via PubMed.

**Parameters:**
- `identifier` (str): Publication identifier (PMID, DOI)
- `dataset_types` (Optional[list[DatasetType]]): Filter to specific dataset types
- `include_related` (bool): Include related datasets (default: True)

**Returns:**
- `str`: Formatted linked datasets results

**Example:**
```python
# Find all linked datasets
results = service.find_linked_datasets("PMID:35042229")

# Filter to GEO and SRA
results = service.find_linked_datasets(
    "PMID:35042229",
    dataset_types=[DatasetType.GEO, DatasetType.SRA]
)
```

**Performance:** 1-3s typical (PubMedProvider)

#### Metadata Methods (2)

##### extract_metadata

```python
def extract_metadata(
    self,
    identifier: str,
    source: Optional[str] = None
) -> Union["PublicationMetadata", str]
```

Extract publication or dataset metadata with capability-based routing.

**Parameters:**
- `identifier` (str): Publication identifier (PMID, DOI, PMC ID, URL) or dataset ID (GSE, SRA, etc.)
- `source` (Optional[str]): Explicit source ("pubmed", "geo", "pmc"). If None, auto-detects from identifier

**Returns:**
- `Union[PublicationMetadata, str]`: PublicationMetadata object or error string

**PublicationMetadata Fields:**
- `title` (str): Publication title
- `authors` (List[str]): Author list
- `abstract` (str): Abstract text
- `journal` (str): Journal name
- `year` (int): Publication year
- `pmid` (Optional[str]): PubMed ID
- `doi` (Optional[str]): DOI
- `keywords` (List[str]): Keywords/MeSH terms

**Example:**
```python
# Publication metadata
metadata = service.extract_metadata("PMID:35042229")
print(f"Title: {metadata.title}")
print(f"Authors: {metadata.authors}")
print(f"Year: {metadata.year}")

# Dataset metadata
metadata = service.extract_metadata("GSE180759", source="geo")
```

**Performance:** 1-3s typical (PubMedProvider, GEOProvider)

##### validate_metadata

```python
def validate_metadata(
    self,
    dataset_id: str,
    required_fields: Optional[List[str]] = None,
    required_values: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.8
) -> str
```

Validate GEO dataset metadata completeness and quality before download.

**Parameters:**
- `dataset_id` (str): Dataset identifier (e.g., "GSE180759")
- `required_fields` (Optional[List[str]]): Required field names to check
- `required_values` (Optional[Dict]): Field → required values mapping
- `threshold` (float): Minimum fraction of samples that must have each field (default: 0.8)

**Returns:**
- `str`: Formatted validation report with recommendations:
  - **PROCEED**: Full integration possible (>90% field coverage)
  - **COHORT**: Cohort-level integration (70-90% coverage)
  - **SKIP**: Insufficient metadata (<70% coverage)

**Example:**
```python
# Validate with specific required fields
report = service.validate_metadata(
    "GSE180759",
    required_fields=["smoking_status", "treatment_response"],
    threshold=0.8
)
print(report)
# Output:
# ✅ PROCEED - 95% completeness (19/20 samples)
# - smoking_status: 100% (20/20 samples)
# - treatment_response: 90% (18/20 samples)

# Validate with required values
report = service.validate_metadata(
    "GSE111111",
    required_values={"condition": ["control", "normal"]},
    threshold=1.0  # All samples must be controls
)
```

**Validation Checks:**
- Sample count verification
- Required field presence
- Required value matching
- Completeness scoring
- Missing field identification

**Performance:** 2-5s typical (GEOProvider + MetadataValidationService)

#### Content Methods (3)

##### get_abstract

```python
def get_abstract(
    self,
    identifier: str,
    force_refresh: bool = False
) -> dict[str, any]
```

Fast abstract retrieval (Tier 1: 200-500ms) via NCBI E-utilities.

**Parameters:**
- `identifier` (str): Publication identifier (PMID, DOI, PMC ID)
- `force_refresh` (bool): Force refresh from API, bypass cache (default: False)

**Returns:**
- `dict[str, any]`: Abstract metadata
  - `title` (str): Publication title
  - `abstract` (str): Abstract text
  - `authors` (List[str]): Author list
  - `journal` (str): Journal name
  - `year` (int): Publication year
  - `pmid` (str): PubMed ID
  - `doi` (Optional[str]): DOI if available
  - `keywords` (List[str]): Keywords/MeSH terms

**Example:**
```python
abstract = service.get_abstract("PMID:35042229")
print(f"Title: {abstract['title']}")
print(f"Authors: {', '.join(abstract['authors'][:3])}")
print(f"Abstract: {abstract['abstract'][:200]}...")
```

**Performance:** 200-500ms typical (AbstractProvider)

##### get_full_content

```python
def get_full_content(
    self,
    source: str,
    prefer_webpage: bool = True,
    keywords: Optional[list[str]] = None,
    max_paragraphs: int = 100,
    max_retries: int = 2
) -> dict[str, any]
```

Full publication content (Tier 2) with **three-tier cascade**: PMC XML → Webpage → PDF.

**Cascade Flow:**
1. **Cache Check**: DataManager lookup (<100ms)
2. **Tier 1 - PMC XML**: For PMID/DOI, try PMC full-text API (500ms-2s, 95% accuracy, 30-40% coverage)
3. **Tier 2 - Webpage**: If PMC unavailable, resolve to URL and scrape HTML (2-5s, 80% success)
4. **Tier 3 - PDF**: Final fallback via DoclingService (3-8s, 70% success)

**Parameters:**
- `source` (str): Publication identifier (PMID, DOI, PMC ID, URL)
- `prefer_webpage` (bool): Try webpage before PDF for URLs (default: True)
- `keywords` (Optional[list[str]]): Section keywords for targeted extraction
- `max_paragraphs` (int): Maximum paragraphs to extract (default: 100)
- `max_retries` (int): Retry count for transient errors (default: 2)

**Returns:**
- `dict[str, any]`: Full content result
  - `content` (str): Full text markdown
  - `methods_text` (str): Methods section (if available)
  - `results_text` (str): Results section (if available)
  - `discussion_text` (str): Discussion section (if available)
  - `tier_used` (str): "full_cached", "full_pmc_xml", "full_webpage", or "full_pdf"
  - `source_type` (str): "pmc_xml", "webpage", or "pdf"
  - `extraction_time` (float): Seconds taken
  - `metadata` (dict): Tables, figures, software, GitHub repos
    - `tables` (int): Number of tables extracted
    - `figures` (int): Number of figures
    - `software` (List[str]): Detected software tools
    - `github_repos` (List[str]): GitHub repository URLs
  - `title` (str): Publication title
  - `abstract` (str): Abstract text
  - `pmc_id` (Optional[str]): PMC ID
  - `pmid` (Optional[str]): PubMed ID
  - `doi` (Optional[str]): DOI

**Example:**
```python
# PMC available (fast path)
content = service.get_full_content("PMID:35042229")
print(f"Tier: {content['tier_used']}")  # "full_pmc_xml"
print(f"Time: {content['extraction_time']:.2f}s")  # ~1s
print(f"Methods: {content['methods_text'][:200]}...")

# Webpage extraction
content = service.get_full_content("https://www.nature.com/articles/...")
print(f"Tier: {content['tier_used']}")  # "full_webpage"

# PDF fallback
content = service.get_full_content("https://biorxiv.org/.../file.pdf")
print(f"Tier: {content['tier_used']}")  # "full_pdf"

# Check software detected
print(f"Software: {content['metadata']['software']}")
print(f"GitHub repos: {content['metadata']['github_repos']}")
```

**Performance Characteristics:**
| Tier | Duration | Success Rate | Coverage |
|------|----------|--------------|----------|
| **Cache** | <100ms | 100% (if cached) | Previously accessed |
| **Tier 1 (PMC)** | 500ms-2s | 95% | 30-40% (open access) |
| **Tier 2 (Webpage)** | 2-5s | 80% | Major publishers |
| **Tier 3 (PDF)** | 3-8s | 70% | Open access PDFs, preprints |

**Error Handling:**
- Automatic PMC → Webpage → PDF fallback
- Paywall detection with suggestions
- Graceful degradation on failures

##### extract_methods

```python
def extract_methods(
    self,
    content_result: dict[str, any],
    llm: Optional[any] = None,
    include_tables: bool = True
) -> dict[str, any]
```

Extract structured methods information from full content result.

**Parameters:**
- `content_result` (dict): Result dict from `get_full_content()`
- `llm` (Optional[any]): LLM for structured extraction (future feature)
- `include_tables` (bool): Whether to include methods tables (default: True)

**Returns:**
- `dict[str, any]`: Extracted methods
  - `methods_text` (str): Raw methods section text
  - `software_used` (List[str]): Detected software tools
  - `github_repos` (List[str]): GitHub repository URLs
  - `parameters` (dict): Extracted parameters (future: LLM extraction)
  - `statistical_methods` (List[str]): Detected statistical tests (future: LLM extraction)
  - `tables` (Optional[List]): Methods-related tables (if include_tables=True)

**Example:**
```python
# Get full content first
content = service.get_full_content("PMID:35042229")

# Extract methods
methods = service.extract_methods(content, include_tables=True)
print(f"Software: {methods['software_used']}")
print(f"GitHub repos: {methods['github_repos']}")
print(f"Tables: {len(methods.get('tables', []))}")
```

**Performance:** <100ms (metadata extraction from cached content)

#### System Methods (1)

##### query_capabilities

```python
def query_capabilities(self) -> str
```

Query available capabilities and supported databases.

**Returns:**
- `str`: Formatted capability matrix showing:
  - Available operations grouped by category
  - Registered providers with capabilities
  - Supported dataset types
  - Performance tiers
  - Cascade logic

**Example:**
```python
capabilities = service.query_capabilities()
print(capabilities)
```

**Output Format:**
```
======================================================================
LOBSTER CONTENT ACCESS SERVICE - CAPABILITY MATRIX
======================================================================

📋 AVAILABLE OPERATIONS:

  Discovery & Search:
    ✅ SEARCH_LITERATURE              → PubMedProvider
    ✅ DISCOVER_DATASETS              → GEOProvider
    ✅ FIND_LINKED_DATASETS           → PubMedProvider

  Metadata & Validation:
    ✅ EXTRACT_METADATA               → PubMedProvider, GEOProvider
    ✅ VALIDATE_METADATA              → GEOProvider

  Content Retrieval:
    ✅ GET_ABSTRACT                   → AbstractProvider
    ✅ GET_FULL_CONTENT               → PMCProvider, WebpageProvider

🔧 REGISTERED PROVIDERS:

  • AbstractProvider (Priority: 10)
    Capabilities: GET_ABSTRACT

  • PubMedProvider (Priority: 10)
    Capabilities: SEARCH_LITERATURE, FIND_LINKED_DATASETS, EXTRACT_METADATA

  • GEOProvider (Priority: 10)
    Capabilities: DISCOVER_DATASETS, EXTRACT_METADATA, VALIDATE_METADATA

  • PMCProvider (Priority: 10)
    Capabilities: GET_FULL_CONTENT

  • WebpageProvider (Priority: 50)
    Capabilities: GET_FULL_CONTENT

💾 SUPPORTED DATASET TYPES:

  ✅ GEO                    → GEOProvider

⚡ PERFORMANCE TIERS:

  Tier 1 (Fast): <500ms
    - get_abstract: AbstractProvider
    - search_literature: PubMedProvider

  Tier 2 (Moderate): 500ms-2s
    - get_full_content (PMC): PMCProvider
    - extract_metadata: PubMedProvider, GEOProvider

  Tier 3 (Slow): 2-8s
    - get_full_content (Webpage): WebpageProvider
    - get_full_content (PDF): WebpageProvider + DoclingService

🔄 CASCADE LOGIC:

  Full Content Retrieval:
    1. Check DataManager cache (fastest)
    2. Try PMC XML (Priority 10, 30-40% coverage)
    3. Fallback: Webpage HTML (Priority 50)
    4. Final fallback: PDF via Docling (Priority 100)

======================================================================
```

#### Performance Benchmarks

| Provider | Operation | Mean Duration | P95 | P99 | Success Rate |
|----------|-----------|---------------|-----|-----|--------------|
| **AbstractProvider** | `get_abstract()` | 350ms | 450ms | 500ms | 95%+ |
| **PubMedProvider** | `search_literature()` | 2.1s | 3.5s | 5s | 99%+ |
| **GEOProvider** | `discover_datasets()` | 3.2s | 4.8s | 6s | 95%+ |
| **PMCProvider** | `get_full_content()` | 1.2s | 2s | 2.5s | 95% (of eligible) |
| **WebpageProvider** | `get_full_content()` | 4.5s | 7s | 10s | 70-80% |

#### Integration with Research Agent

The research_agent uses ContentAccessService through 10 tools:

| Agent Tool | ContentAccessService Method | Category |
|-----------|----------------------------|----------|
| `search_literature` | `search_literature()` | Discovery |
| `fast_dataset_search` | `discover_datasets()` | Discovery |
| `find_related_entries` | `find_linked_datasets()` | Discovery |
| `get_dataset_metadata` | `extract_metadata()` | Metadata |
| `fast_abstract_search` | `get_abstract()` | Content |
| `read_full_publication` | `get_full_content()` | Content |
| `extract_methods` | `extract_methods()` | Content |
| `validate_dataset_metadata` | `validate_metadata()` | Metadata |

#### See Also

- **Deep Dive:** [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md) - Comprehensive provider architecture guide
- **Research Agent:** [15-agents-api.md](15-agents-api.md) - Integration with literature mining agent
- **Architecture:** [18-architecture-overview.md](18-architecture-overview.md) - System design
- **Troubleshooting:** [28-troubleshooting.md](28-troubleshooting.md) - Common issues

### PublicationResolver ✨ (v2.3+ Enhanced)

Utility class for automatic PMID/DOI → PDF URL resolution using tiered waterfall strategy. **v2.4+ enhancement:** Integrated with ContentAccessService for seamless DOI/PMID auto-detection.

```python
class PublicationResolver:
    """Resolver for converting identifiers to accessible PDF URLs."""

    def resolve(self, identifier: str) -> PublicationResolutionResult:
        """
        Resolve DOI/PMID to accessible URL using tiered waterfall strategy.

        Auto-detects identifier type and applies appropriate resolution method.
        """
```

#### Auto-Detection Logic (v2.3+ Enhancement)

The resolver automatically detects identifier types without requiring format specification:

| Input Format | Detection Pattern | Example | Resolution Strategy |
|--------------|-------------------|---------|-------------------|
| **Bare DOI** | Starts with `10.` | `10.1101/2024.01.001` | bioRxiv/medRxiv → Publisher |
| **DOI with prefix** | `^DOI:10\.` | `DOI:10.1038/s41586-025-09686-5` | Publisher → PMC → Preprints |
| **PMID with prefix** | `^PMID:\d{7,8}$` | `PMID:39370688` | PMC → Publisher |
| **Numeric PMID** | `^\d{7,8}$` | `39370688` | PMC → Publisher |
| **Direct URL** | `^https?://` | `https://nature.com/articles/...` | Pass through (no resolution needed) |

#### Resolution Strategies

**1. PMC Open Access (Highest Priority)**
```python
# For PMID input
identifier = "PMID:39370688"
# → Checks PMC API for open access version
# → Returns: "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12496192/pdf/"
# → Quality: Very high (government repository, reliable)
```

**2. Preprint Servers (bioRxiv, medRxiv)**
```python
# For DOI starting with 10.1101
identifier = "10.1101/2024.08.29.610467"
# → Resolves to bioRxiv/medRxiv PDF
# → Returns: "https://www.biorxiv.org/content/10.1101/2024.08.29.610467.full.pdf"
# → Quality: High (preprints, usually accessible)
```

**3. Publisher Direct (Fallback)**
```python
# For non-preprint DOIs
identifier = "10.1038/s41586-025-09686-5"
# → Uses CrossRef API to find publisher URL
# → Returns: "https://www.nature.com/articles/s41586-025-09686-5"
# → Quality: Medium (may be paywalled)
```

#### Methods

##### resolve

```python
def resolve(self, identifier: str) -> PublicationResolutionResult
```

**Auto-detects identifier type** and resolves to accessible URL using tiered waterfall strategy.

**Parameters:**
- `identifier` (str): **Multiple formats supported:**
  - **Bare DOI:** `"10.1101/2024.08.29.610467"`
  - **DOI with prefix:** `"DOI:10.1038/s41586-025-09686-5"`
  - **PMID:** `"PMID:39370688"` or `"39370688"`
  - **Direct URL:** `"https://www.nature.com/articles/..."` (passthrough)

**Returns:** `PublicationResolutionResult` with:
- `pdf_url` (str): Accessible PDF URL (if found)
- `source` (str): Resolution source (`'pmc'`, `'biorxiv'`, `'medrxiv'`, `'publisher'`, `'paywalled'`)
- `access_type` (str): Access level (`'open_access'`, `'preprint'`, `'paywalled'`, `'error'`)
- `suggestions` (str): Alternative access strategies for paywalled content
- `alternative_urls` (List[str]): Alternative access URLs when available

**Example Usage:**
```python
from lobster.tools.providers.publication_resolver import PublicationResolver

resolver = PublicationResolver()

# Resolve bioRxiv DOI
result = resolver.resolve("10.1101/2024.08.29.610467")
print(f"PDF URL: {result.pdf_url}")  # https://www.biorxiv.org/content/...
print(f"Source: {result.source}")    # 'biorxiv'
print(f"Access: {result.access_type}")  # 'preprint'

# Resolve PMID to PMC
result = resolver.resolve("PMID:39370688")
print(f"PDF URL: {result.pdf_url}")  # https://www.ncbi.nlm.nih.gov/pmc/...
print(f"Source: {result.source}")    # 'pmc'
print(f"Access: {result.access_type}")  # 'open_access'

# Handle paywalled DOI gracefully
result = resolver.resolve("10.18632/aging.204666")
print(f"Accessible: {result.is_accessible()}")  # False
print(f"Access type: {result.access_type}")     # 'paywalled'
print(f"Suggestions: {result.suggestions}")     # Alternative access methods
```

##### batch_resolve

```python
def batch_resolve(
    self,
    identifiers: List[str],
    max_batch: int = 10
) -> List[PublicationResolutionResult]
```

Batch resolve multiple identifiers with automatic rate limiting.

**Parameters:**
- `identifiers` (List[str]): List of DOIs, PMIDs, or URLs
- `max_batch` (int): Conservative limit to avoid API rate limits (default: 10)

**Example:**
```python
identifiers = [
    "10.1101/2024.08.29.610467",  # bioRxiv DOI
    "PMID:39370688",              # PMID
    "10.18632/aging.204666",      # Potentially paywalled DOI
]

results = resolver.batch_resolve(identifiers)

for i, result in enumerate(results):
    print(f"Paper {i+1}: {result.source} ({result.access_type})")
    if result.is_accessible():
        print(f"  → {result.pdf_url}")
    else:
        print(f"  → {result.suggestions}")
```

### ConcatenationService

Service for combining multiple samples or datasets.

```python
class ConcatenationService:
    """Service for concatenating samples with batch correction and validation."""
```

#### Methods

##### concatenate_samples

```python
def concatenate_samples(
    self,
    adata_list: List[anndata.AnnData],
    batch_key: str = "batch",
    batch_correction_method: str = "harmony",
    join_method: str = "outer"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Concatenate multiple AnnData objects with batch correction.

**Parameters:**
- `adata_list` (List[anndata.AnnData]): List of AnnData objects to concatenate
- `batch_key` (str): Column name for batch information
- `batch_correction_method` (str): Method for batch correction ('harmony', 'scanorama', 'none')
- `join_method` (str): How to join variables ('outer', 'inner')

### VisualizationService

Service for creating scientific visualizations.

```python
class VisualizationService:
    """Service for creating publication-quality visualizations."""
```

#### Methods

##### create_umap_plot

```python
def create_umap_plot(
    self,
    adata: anndata.AnnData,
    color_by: str = None,
    use_raw: bool = False,
    point_size: float = 1.0,
    alpha: float = 0.8,
    color_map: str = "viridis"
) -> go.Figure
```

Create UMAP visualization with customizable styling.

##### create_volcano_plot

```python
def create_volcano_plot(
    self,
    results_df: pd.DataFrame,
    log2fc_col: str = "log2FoldChange",
    pvalue_col: str = "padj",
    significance_threshold: float = 0.05,
    fold_change_threshold: float = 1.0
) -> go.Figure
```

Create volcano plot for differential expression results.

##### create_heatmap

```python
def create_heatmap(
    self,
    adata: anndata.AnnData,
    genes: List[str],
    groupby: str = None,
    use_raw: bool = False,
    standard_scale: str = None,
    cmap: str = "RdBu_r"
) -> go.Figure
```

Create expression heatmap for selected genes.

## Advanced Services

### MLProteomicsService (ALPHA)

Machine learning service for proteomics data.

```python
class MLProteomicsService:
    """Alpha service for machine learning applications in proteomics."""
```

### MLTranscriptomicsService (ALPHA)

Machine learning service for transcriptomics data.

```python
class MLTranscriptomicsService:
    """Alpha service for machine learning applications in transcriptomics."""
```

### SCVIEmbeddingService

Service for scVI-based embeddings and batch correction.

```python
class SCVIEmbeddingService:
    """Service for scVI-based dimensionality reduction and batch correction."""
```

#### Methods

##### train_scvi_model

```python
def train_scvi_model(
    self,
    adata: anndata.AnnData,
    batch_key: str = None,
    n_latent: int = 10,
    n_epochs: int = 400,
    early_stopping: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Train scVI model for dimensionality reduction and batch correction.

## Error Handling in Services

All services implement consistent error handling:

### Exception Hierarchy

```python
class ServiceError(Exception):
    """Base exception for service operations."""
    pass

class PreprocessingError(ServiceError):
    """Exception for preprocessing operations."""
    pass

class AnalysisError(ServiceError):
    """Exception for analysis operations."""
    pass

class ValidationError(ServiceError):
    """Exception for validation operations."""
    pass
```

### Error Response Pattern

```python
def handle_service_error(func):
    """Decorator for consistent service error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Service error in {func.__name__}: {e}")
            raise ServiceError(f"Operation failed: {str(e)}") from e
    return wrapper
```

## Progress Callbacks

Services support progress callbacks for long-running operations:

```python
def set_progress_callback(self, callback: Callable[[int, str], None]) -> None:
    """
    Set a callback function to report progress.

    Args:
        callback: Function accepting (progress_percent, message)
    """
    self.progress_callback = callback
```

## Service Integration Examples

### Using Services Directly

```python
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.clustering_service import ClusteringService

# Initialize services
preprocess = PreprocessingService()
cluster = ClusteringService()

# Process data through pipeline
filtered_adata, filter_stats = preprocess.filter_cells_and_genes(adata)
normalized_adata, norm_stats = preprocess.normalize_data(filtered_adata)
clustered_adata, cluster_stats = cluster.cluster_and_visualize(normalized_adata)
```

### Service Chain Pattern

```python
def create_analysis_pipeline(services: List, params: List[Dict]) -> Callable:
    """Create a pipeline from multiple services."""
    def pipeline(adata: anndata.AnnData) -> Tuple[anndata.AnnData, Dict]:
        current_adata = adata
        all_stats = {}

        for service, param_dict in zip(services, params):
            current_adata, stats = service(**param_dict)(current_adata)
            all_stats.update(stats)

        return current_adata, all_stats

    return pipeline
```

### Validation and Quality Control

All services include built-in validation:

```python
def validate_input(self, adata: anndata.AnnData) -> None:
    """Validate AnnData input for service operations."""
    if adata is None:
        raise ValueError("AnnData object cannot be None")
    if adata.n_obs == 0:
        raise ValueError("No observations in AnnData object")
    if adata.n_vars == 0:
        raise ValueError("No variables in AnnData object")
```

The Services API provides a comprehensive set of stateless, reproducible analysis tools that form the computational backbone of the Lobster AI system. Each service is designed to be used independently or as part of larger analysis workflows, with consistent interfaces and robust error handling throughout.