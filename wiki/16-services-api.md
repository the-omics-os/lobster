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

### PublicationService

Service for literature mining and dataset discovery.

```python
class PublicationService:
    """Service for searching literature and finding associated datasets."""
```

#### Methods

##### search_literature

```python
def search_literature(
    self,
    query: str,
    max_results: int = 10,
    publication_year_range: Tuple[int, int] = None,
    journal_filter: List[str] = None
) -> Dict[str, Any]
```

Search PubMed for relevant literature.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of results
- `publication_year_range` (Tuple[int, int]): Year range filter
- `journal_filter` (List[str]): List of journals to include

##### find_datasets_from_publication

```python
def find_datasets_from_publication(
    self,
    pmid: str,
    dataset_types: List[str] = None
) -> Dict[str, Any]
```

Find datasets associated with a publication.

### PublicationIntelligenceService ✨ (v2.3+ Enhanced with Docling)

Service for extracting computational methods from publications with **structure-aware Docling PDF parsing**, automatic PMID/DOI → PDF resolution, and intelligent Methods section detection.

```python
class PublicationIntelligenceService:
    """Service for method extraction with Docling integration and automatic identifier resolution."""
```

**New in v2.3.0:**
- ✅ Structure-aware PDF parsing with Docling (replaces naive PyPDF2 truncation)
- ✅ Intelligent Methods section detection by keywords (>90% hit rate)
- ✅ Table extraction from Methods sections (parameter tables)
- ✅ Formula detection and LaTeX formatting
- ✅ Smart image filtering (removes base64 bloat, 40-60% size reduction)
- ✅ Document caching (2-5s first parse → <100ms cached)
- ✅ Comprehensive retry logic with automatic PyPDF2 fallback
- ✅ Memory management with explicit garbage collection

#### Methods

##### extract_methods_section 🆕 (v2.3+)

```python
def extract_methods_section(
    self,
    source: str,
    keywords: Optional[List[str]] = None,
    max_paragraphs: int = 50,
    max_retries: int = 2
) -> Dict[str, Any]
```

**Flagship method** for structure-aware Methods section extraction using Docling.

**Parameters:**
- `source` (str): PDF URL or local file path
- `keywords` (Optional[List[str]]): Section keywords to search (default: method-related)
- `max_paragraphs` (int): Maximum paragraphs to extract (default: 50)
- `max_retries` (int): Maximum retry attempts on failure (default: 2)

**Returns:** Dict with:
- `methods_text` (str): Full Methods section text
- `methods_markdown` (str): Markdown with tables (images filtered)
- `sections` (List[Dict]): Hierarchical document structure
- `tables` (List[DataFrame]): Extracted tables as pandas DataFrames
- `formulas` (List[str]): Mathematical formulas in LaTeX
- `software_mentioned` (List[str]): Detected bioinformatics tools (24 tools recognized)
- `provenance` (Dict): Metadata tracking (parser, version, timestamp, fallback status)

**Example:**
```python
service = PublicationIntelligenceService()

result = service.extract_methods_section(
    "https://arxiv.org/pdf/2408.09869"
)

print(f"Methods: {len(result['methods_text'])} chars")
print(f"Tables: {len(result['tables'])}")
print(f"Formulas: {len(result['formulas'])}")
print(f"Software: {result['software_mentioned']}")
print(f"Parser: {result['provenance']['parser']}")  # 'docling' or 'pypdf2'
```

**Performance:**
| Metric | First Parse | Cached Parse | Improvement |
|--------|------------|--------------|-------------|
| Time | 2-5 seconds | <100ms | 30-50x faster |
| Memory | ~500MB | ~50MB | 10x lower |

**Error Handling:**
- Automatic retry on MemoryError (with `gc.collect()`)
- Detects incompatible PDFs (page-dimensions error)
- Graceful fallback to PyPDF2 after max retries
- Non-fatal cache failures

##### extract_methods_from_paper

```python
def extract_methods_from_paper(
    self,
    url_or_pmid: str,
    llm=None,
    max_text_length: int = 10000  # DEPRECATED in v2.3+
) -> Dict[str, Any]
```

Extract computational methods from a paper with LLM analysis. **NOW uses Docling for structure-aware extraction** (v2.3+) and accepts PMIDs, DOIs, and direct URLs with automatic resolution.

**Parameters:**
- `url_or_pmid` (str): PMID, DOI, or direct PDF URL
- `llm`: LLM instance for extraction (optional, auto-created if None)
- `max_text_length` (int): **DEPRECATED** - Docling now extracts full Methods section intelligently

**Returns:** Dict with:
- `software_used` (List[str]): Detected software tools
- `parameters` (Dict): Parameter values and cutoffs
- `statistical_methods` (List[str]): Statistical approaches
- `data_sources` (List[str]): Data sources and repositories
- `sample_sizes` (Dict): Sample size information
- `normalization_methods` (List[str]): Normalization approaches
- `quality_control` (List[str]): QC steps
- **v2.3+ enhancements:**
  - `tables` (List[DataFrame]): Parameter tables from Methods
  - `formulas` (List[str]): Mathematical formulas
  - `software_detected` (List[str]): Auto-detected tools (keyword-based)
  - `extraction_metadata` (Dict): Provenance tracking

**Example:**
```python
# Automatic PMID resolution + Docling extraction + LLM analysis
methods = service.extract_methods_from_paper("PMID:38448586")

print("Software:", methods['software_used'])
print("Parameters:", methods['parameters'])

# v2.3+ enhancements
print("Tables:", len(methods['tables']))
print("Formulas:", len(methods['formulas']))
print("Parser used:", methods['extraction_metadata']['parser'])
```

**Resolution strategy:**
1. PMC → bioRxiv/medRxiv → Publisher → Suggestions
2. 70-80% automatic resolution success rate
3. Graceful fallback with 5 alternative access strategies for paywalled papers

##### _filter_images_from_markdown 🆕 (v2.3+)

```python
def _filter_images_from_markdown(self, markdown: str) -> str
```

Remove base64 image encodings from Markdown to reduce LLM context bloat.

**Performance:** 40-60% Markdown size reduction for image-heavy papers.

##### _get_cached_document 🆕 (v2.3+)

```python
def _get_cached_document(self, source: str) -> Optional[DoclingDocument]
```

Retrieve cached parsed document if available.

**Cache Location:** `.lobster_workspace/literature_cache/parsed_docs/{md5_hash}.json`

**Performance:** <100ms cache hit vs 2-5s fresh parse (30-50x faster)

##### _cache_document 🆕 (v2.3+)

```python
def _cache_document(self, source: str, doc: DoclingDocument) -> None
```

Cache parsed document as JSON for future retrieval.

**Storage:** JSON serialization via Pydantic `model_dump()` / `model_validate()`

#### Performance Characteristics (v2.3+)

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| **First parse (Docling)** | 2-5s | ~500MB | Structure analysis + tables |
| **Cache hit** | <100ms | ~50MB | JSON load + validation |
| **PyPDF2 fallback** | <1s | ~100MB | Naive text extraction |
| **Methods hit rate** | >90% | - | Docling vs ~30% PyPDF2 |
| **Table extraction** | 80%+ | - | Parameter tables detected |
| **Cache storage** | 500KB-2MB | - | Per paper JSON file |

#### Caching Behavior (v2.3+)

**Cache Strategy:**
- Cache Key: MD5 hash of source URL
- Storage Format: JSON via Pydantic serialization
- Location: `.lobster_workspace/literature_cache/parsed_docs/`
- Invalidation: Manual (delete cached file)
- Performance: 30-50x faster on cache hit

**Best Practices:**
- Cache is persistent across sessions
- Safe for batch processing (explicit `gc.collect()` after each paper)
- Non-fatal cache failures (extraction continues on cache error)
- Monitor cache directory size (~1-2MB per paper)

#### See Also

- **Deep Dive:** [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md) - Comprehensive technical guide
- **Troubleshooting:** [28-troubleshooting.md](28-troubleshooting.md) - Common issues
- **Research Agent:** [15-agents-api.md](15-agents-api.md) - Integration with literature mining agent

### PublicationResolver ✨ (v2.3+ Enhanced)

Utility class for automatic PMID/DOI → PDF URL resolution using tiered waterfall strategy. **v2.3+ enhancement:** Integrated with UnifiedContentService for seamless DOI/PMID auto-detection.

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