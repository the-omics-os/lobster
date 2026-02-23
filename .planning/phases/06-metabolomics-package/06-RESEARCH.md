# Phase 6: Metabolomics Package - Research

**Researched:** 2026-02-22
**Domain:** Metabolomics analysis (LC-MS, GC-MS, NMR), untargeted metabolomics pipelines, multivariate statistics, metabolite annotation
**Confidence:** HIGH (codebase verified, domain well-established, existing infrastructure confirmed)

## Summary

Phase 6 creates a new `packages/lobster-metabolomics/` package containing a `metabolomics_expert` parent agent with 10 tools, backed by 4 stateless services. This is the first entirely new package created from scratch (all prior phases enhanced existing packages). The codebase is well-prepared: the core already contains `MetabolomicsAdapter` (3 platform variants for LC-MS/GC-MS/NMR), `MetabolomicsSchema` with full Pydantic validation, the `metabolomics` omics type registered in `OmicsTypeRegistry`, and MetaboLights provider/download infrastructure. What does NOT exist is: any metabolomics agent, any metabolomics-specific analysis services, or any metabolomics tools.

The package follows the exact same architecture as `lobster-proteomics`: PEP 420 namespace package under `packages/`, entry points for agent discovery, stateless services returning 3-tuples, tools in `shared_tools.py`, agent factory in `metabolomics_expert.py`, and `PlatformConfig`-style configuration for LC-MS/GC-MS/NMR platform differences. The phase also fixes BUG-14 (sparse matrix zero-checking in the metabolomics schema validation functions) and creates the metabolomics_expert system prompt.

**Primary recommendation:** Model the package structure exactly on `lobster-proteomics`, reuse existing core infrastructure (adapter, schema, pathway enrichment service), implement PLS-DA via sklearn's `PLSRegression` (already available), implement OPLS-DA from scratch (~100 lines, pyopls is unmaintained since 2020), and do NOT add any new PyPI dependencies.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MET-01 | Create `packages/lobster-metabolomics/` package structure | Follow `lobster-proteomics` package layout exactly: pyproject.toml with entry points, lobster/agents/metabolomics/ modular folder, lobster/services/{quality,analysis,annotation}/ service folders, tests/ mirror. Use PEP 420 namespace (no `lobster/__init__.py`). |
| MET-02 | Create `MetabolomicsQualityService` -- RSD, TIC, QC sample evaluation | New service in `lobster/services/quality/metabolomics_quality_service.py`. Methods: `assess_quality()` for overall QC metrics (TIC distribution, CV/RSD per metabolite, QC sample reproducibility, missing value analysis), `evaluate_qc_samples()` for QC-specific pooled sample assessment. Returns 3-tuple. Use numpy/scipy only. |
| MET-03 | Create `MetabolomicsPreprocessingService` -- filter, impute, normalize, batch correct | New service in `lobster/services/quality/metabolomics_preprocessing_service.py`. Methods: `filter_features()` (prevalence, RSD, blank subtraction), `impute_missing_values()` (KNN, minimum, LOD/2, MICE), `normalize()` (PQN, TIC, IS, median, quantile), `correct_batch_effects()` (QC-RLSC, ComBat, median centering). Returns 3-tuples. |
| MET-04 | Create `MetabolomicsAnalysisService` -- univariate stats, PLS-DA, fold change, pathway enrichment | New service in `lobster/services/analysis/metabolomics_analysis_service.py`. Methods: `run_univariate_statistics()` (t-test/Wilcoxon/ANOVA with FDR), `run_pls_da()` (sklearn PLSRegression with Y encoding + VIP scores + permutation test), `run_opls_da()` (custom NIPALS-based implementation ~100 lines), `run_pca()` (sklearn PCA wrapper), `calculate_fold_changes()`. Reuse existing `PathwayEnrichmentService` from core for pathway enrichment tool. |
| MET-05 | Create `MetabolomicsAnnotationService` -- m/z matching to HMDB/KEGG, MSI levels | New service in `lobster/services/annotation/metabolomics_annotation_service.py`. Method: `annotate_by_mz()` matches observed m/z against a bundled reference database (HMDB/KEGG compound masses + common adducts), with ppm tolerance. Assigns MSI confidence levels 1-4. Method: `classify_lipids()` groups features by lipid class based on m/z ranges and annotation. |
| MET-06 | Implement `assess_metabolomics_quality` tool | Wraps `MetabolomicsQualityService.assess_quality()`. Platform-aware (LC-MS/GC-MS/NMR auto-detect from adapter). Reports TIC, RSD, missing values, QC sample CV. |
| MET-07 | Implement `filter_metabolomics_features` tool | Wraps `MetabolomicsPreprocessingService.filter_features()`. Parameters: min_prevalence, max_rsd, blank_ratio_threshold. |
| MET-08 | Implement `handle_missing_values` tool | Wraps `MetabolomicsPreprocessingService.impute_missing_values()`. Parameters: method (knn/min/lod_half/median/mice), knn_neighbors. |
| MET-09 | Implement `normalize_metabolomics` tool | Wraps `MetabolomicsPreprocessingService.normalize()`. Parameters: method (pqn/tic/is/median/quantile), log_transform, reference_sample. |
| MET-10 | Implement `correct_batch_effects` tool | Wraps `MetabolomicsPreprocessingService.correct_batch_effects()`. Parameters: batch_key, method (qc_rlsc/combat/median_centering), qc_label. |
| MET-11 | Implement `run_metabolomics_statistics` tool | Wraps `MetabolomicsAnalysisService.run_univariate_statistics()` + `calculate_fold_changes()`. Parameters: group_column, method (ttest/wilcoxon/anova), fdr_method, fold_change_threshold. |
| MET-12 | Implement `run_multivariate_analysis` tool | Wraps PCA/PLS-DA/OPLS-DA methods. Parameters: method (pca/plsda/oplsda), n_components, group_column (for supervised), permutation_test. |
| MET-13 | Implement `annotate_metabolites` tool | Wraps `MetabolomicsAnnotationService.annotate_by_mz()`. Parameters: ppm_tolerance, adducts, databases (hmdb/kegg/both). Reports MSI levels. |
| MET-14 | Implement `analyze_lipid_classes` tool | Wraps `MetabolomicsAnnotationService.classify_lipids()`. Groups metabolites by lipid class, computes class-level statistics, creates summary. |
| MET-15 | Implement `run_pathway_enrichment` tool | Wraps existing `PathwayEnrichmentService` from core (gseapy/Enrichr). Extracts significant metabolite gene/compound list, runs MSEA. Parameters: database (kegg/reactome/go), significance_threshold. |
| MET-16 | Register metabolomics_expert via entry points in pyproject.toml | Entry point: `[project.entry-points."lobster.agents"]` with `metabolomics_expert = "lobster.agents.metabolomics.metabolomics_expert:AGENT_CONFIG"`. Follow proteomics pattern. |
| MET-17 | Fix BUG-14: metabolomics schema sparse matrix zero-checking | In `lobster/core/schemas/metabolomics.py`, functions `_validate_intensity_data()` (line ~466-469) and `_validate_missing_values()` (line ~493-494) use `adata.X.data == 0` for sparse matrices. This checks only stored (non-zero) elements, not the actual zero count. For sparse matrices, zeros are implicit (not stored in `.data`). Fix: use `adata.X.nnz` vs total elements for true sparsity, and `np.isnan()` on dense conversion (or per-chunk) for missing values. |
| DOC-06 | Create metabolomics_expert prompt | System prompt in `prompts.py` following proteomics pattern: identity, capabilities, platform detection (LC-MS/GC-MS/NMR), tool inventory (10 tools), standard workflows, delegation protocol (structured for future child agents). |
</phase_requirements>

## Standard Stack

### Core (Already Available -- No New Dependencies)

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| numpy | >=1.23.0 | Array operations, statistics | Already in lobster-ai |
| pandas | >=1.5.0 | DataFrames, data manipulation | Already in lobster-ai |
| scipy | >=1.10.0 | Statistical tests (t-test, Wilcoxon, ANOVA), FDR, signal processing | Already in lobster-ai |
| scikit-learn | >=1.3.0 | PCA, PLSRegression (for PLS-DA), KNN imputation, StandardScaler | Already in lobster-ai |
| statsmodels | >=0.14.0 | Multiple testing correction, advanced statistics | Already in lobster-ai |
| anndata | >=0.9.0 | Data structure (samples x metabolites) | Already in lobster-ai |
| gseapy | >=1.1.0 | Pathway enrichment via Enrichr (MSEA) | Already in lobster-ai core |
| plotly | >=5.0.0 | Visualization (volcano, PCA scores, loading plots) | Already in lobster-ai |

### New Dependencies Needed

| Library | Version | Purpose | Where |
|---------|---------|---------|-------|
| None required | -- | All functionality built on existing stack | -- |

**Key insight:** No new dependencies. PLS-DA uses sklearn's `PLSRegression` (encode class labels as dummy matrix, compute VIP scores from weights/loadings). OPLS-DA is implemented from scratch using NIPALS algorithm (~100 lines of numpy). QC-RLSC (QC-based Robust Loess Signal Correction) uses scipy's interpolation. All statistical tests (t-test, Wilcoxon, Mann-Whitney, ANOVA, Kruskal-Wallis) are in scipy.stats. FDR correction is in statsmodels. ComBat for batch correction follows the same pattern as proteomics (parametric empirical Bayes).

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom OPLS-DA | pyopls package | pyopls last updated March 2020, Python 3.5+ requirement, unmaintained. Custom implementation is ~100 lines, no dependency. Use custom. |
| Custom PLS-DA | mbPLS, pyPLS | These are niche packages with poor maintenance. sklearn PLSRegression is battle-tested and already installed. Use sklearn. |
| Custom m/z annotation | ClassyFire API, SIRIUS | ClassyFire API is fragile. SIRIUS requires MS2 data (out of scope for v1). Use offline m/z matching against bundled mass tables. |
| Custom pathway enrichment | MetaboAnalyst API | MetaboAnalyst has rate limits and is web-only. gseapy/Enrichr is already integrated. Use existing PathwayEnrichmentService. |
| pyCombat package | Custom ComBat | pyCombat exists but adds another dependency. Proteomics package already implements ComBat logic. Reuse that pattern. |

**Installation:**
```bash
# No new packages needed -- lobster-metabolomics depends only on lobster-ai
pip install lobster-metabolomics  # Will pull lobster-ai which has all needed deps
```

## Architecture Patterns

### Recommended Package Structure

```
packages/lobster-metabolomics/
├── pyproject.toml                              # Package metadata + entry points
├── LICENSE                                     # AGPL-3.0-or-later
├── README.md                                   # Package description
├── lobster/
│   ├── agents/
│   │   └── metabolomics/
│   │       ├── __init__.py                     # Module exports + graceful imports
│   │       ├── config.py                       # PlatformConfig (lc_ms/gc_ms/nmr)
│   │       ├── prompts.py                      # create_metabolomics_expert_prompt()
│   │       ├── state.py                        # MetabolomicsExpertState
│   │       ├── shared_tools.py                 # 10 tool factory (create_shared_tools)
│   │       ├── metabolomics_expert.py          # AGENT_CONFIG + factory function
│   │       └── py.typed                        # PEP 561 marker
│   └── services/
│       ├── quality/
│       │   ├── metabolomics_quality_service.py     # MET-02: QC assessment
│       │   └── metabolomics_preprocessing_service.py # MET-03: filter/impute/normalize/batch
│       ├── analysis/
│       │   └── metabolomics_analysis_service.py     # MET-04: stats, PLS-DA, OPLS-DA
│       └── annotation/
│           └── metabolomics_annotation_service.py   # MET-05: m/z matching, lipid classes
└── tests/
    ├── agents/
    │   ├── __init__.py
    │   └── test_metabolomics_integration.py    # Agent contract tests
    └── services/
        ├── quality/
        │   ├── test_metabolomics_quality_service.py
        │   └── test_metabolomics_preprocessing_service.py
        ├── analysis/
        │   └── test_metabolomics_analysis_service.py
        └── annotation/
            └── test_metabolomics_annotation_service.py
```

### Pattern 1: PlatformConfig for LC-MS/GC-MS/NMR (follow proteomics pattern)

**What:** Platform-specific defaults for metabolomics sub-types, mirroring the `PlatformConfig` in proteomics.
**When to use:** Every tool needs platform-aware defaults (normalization, missing thresholds, QC criteria).

```python
# config.py - follows proteomics PlatformConfig pattern exactly
@dataclass
class MetabPlatformConfig:
    """Configuration for a metabolomics platform type."""
    platform_type: str  # "lc_ms", "gc_ms", or "nmr"
    display_name: str
    description: str
    # Missing value thresholds
    expected_missing_rate_range: tuple
    max_missing_per_sample: float
    max_missing_per_feature: float
    # QC thresholds
    max_rsd_qc_samples: float  # RSD threshold for QC samples
    min_features_per_sample: int
    # Normalization defaults
    default_normalization: str
    log_transform: bool
    default_imputation: str
    # Analysis defaults
    default_fold_change_threshold: float
    default_n_pca_components: int

PLATFORM_CONFIGS = {
    "lc_ms": MetabPlatformConfig(
        platform_type="lc_ms",
        display_name="LC-MS",
        description="Liquid chromatography-mass spectrometry",
        expected_missing_rate_range=(0.20, 0.60),
        max_missing_per_sample=0.60,
        max_missing_per_feature=0.80,
        max_rsd_qc_samples=30.0,
        min_features_per_sample=50,
        default_normalization="pqn",
        log_transform=True,
        default_imputation="knn",
        default_fold_change_threshold=1.5,
        default_n_pca_components=10,
    ),
    "gc_ms": MetabPlatformConfig(
        platform_type="gc_ms",
        display_name="GC-MS",
        description="Gas chromatography-mass spectrometry",
        expected_missing_rate_range=(0.10, 0.40),
        max_missing_per_sample=0.40,
        max_missing_per_feature=0.60,
        max_rsd_qc_samples=25.0,
        min_features_per_sample=30,
        default_normalization="tic",
        log_transform=True,
        default_imputation="min",
        default_fold_change_threshold=2.0,
        default_n_pca_components=10,
    ),
    "nmr": MetabPlatformConfig(
        platform_type="nmr",
        display_name="NMR",
        description="Nuclear magnetic resonance spectroscopy",
        expected_missing_rate_range=(0.0, 0.10),
        max_missing_per_sample=0.10,
        max_missing_per_feature=0.20,
        max_rsd_qc_samples=20.0,
        min_features_per_sample=100,
        default_normalization="pqn",
        log_transform=False,  # NMR data often already processed
        default_imputation="median",
        default_fold_change_threshold=1.5,
        default_n_pca_components=10,
    ),
}
```

### Pattern 2: PLS-DA via sklearn PLSRegression

**What:** PLS-DA (Partial Least Squares Discriminant Analysis) using sklearn's PLSRegression with class label encoding.
**When to use:** MET-12 multivariate analysis tool.

```python
# Source: sklearn.cross_decomposition.PLSRegression + standard VIP calculation
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def run_pls_da(X, y_labels, n_components=2):
    """PLS-DA using PLSRegression with dummy-encoded class labels."""
    # Encode class labels as dummy matrix
    lb = LabelBinarizer()
    Y = lb.fit_transform(y_labels)
    if Y.shape[1] == 1:  # Binary case: expand to 2 columns
        Y = np.hstack([1 - Y, Y])

    # Fit PLS model
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X, Y)

    # Get scores (sample coordinates in latent space)
    T = pls.x_scores_  # n_samples x n_components

    # Calculate VIP (Variable Importance in Projection) scores
    W = pls.x_weights_  # n_features x n_components
    T_scores = pls.x_scores_
    Q = pls.y_loadings_  # n_targets x n_components

    # VIP = sqrt(p * sum(q^2 * t't * w^2) / sum(q^2 * t't))
    p = X.shape[1]
    SS = np.diag(T_scores.T @ T_scores @ Q.T @ Q)
    vip = np.sqrt(p * np.sum(SS * W**2, axis=1) / np.sum(SS))

    return T, vip, pls
```

### Pattern 3: OPLS-DA (Custom NIPALS Implementation)

**What:** OPLS-DA separates predictive from orthogonal variation using the NIPALS algorithm.
**When to use:** MET-12 when user requests OPLS-DA specifically.

```python
# Custom OPLS-DA implementation (no external dependency needed)
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer

def run_opls_da(X, y_labels, n_orthogonal=1, n_predictive=1):
    """OPLS-DA: separate predictive from orthogonal variation."""
    # Encode Y
    lb = LabelBinarizer()
    Y = lb.fit_transform(y_labels).astype(float)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])

    # Center and scale
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    # NIPALS: extract orthogonal components
    X_filtered = X_scaled.copy()
    orthogonal_scores = []
    orthogonal_loadings = []

    for _ in range(n_orthogonal):
        # PLS weight vector
        w = X_filtered.T @ Y_scaled
        w = w[:, 0] / np.linalg.norm(w[:, 0])
        # Predictive score
        t = X_filtered @ w
        # Predictive loading
        p = X_filtered.T @ t / (t.T @ t)
        # Orthogonal weight
        w_orth = p - (w.T @ p / (w.T @ w)) * w
        w_orth = w_orth / np.linalg.norm(w_orth)
        # Orthogonal score and loading
        t_orth = X_filtered @ w_orth
        p_orth = X_filtered.T @ t_orth / (t_orth.T @ t_orth)
        # Remove orthogonal variation
        X_filtered = X_filtered - np.outer(t_orth, p_orth)
        orthogonal_scores.append(t_orth)
        orthogonal_loadings.append(p_orth)

    # Final predictive PLS on filtered X
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=n_predictive, scale=False)
    pls.fit(X_filtered, Y_scaled)

    return {
        "predictive_scores": pls.x_scores_,
        "orthogonal_scores": np.column_stack(orthogonal_scores) if orthogonal_scores else None,
        "model": pls,
        "X_filtered": X_filtered,
    }
```

### Pattern 4: MSI Confidence Levels for Annotation

**What:** Metabolomics Standards Initiative (MSI) identification confidence levels.
**When to use:** MET-05, MET-13 annotation tool.

```python
# MSI Confidence Level Definitions (Sumner et al. 2007, Metabolomics 3:211-221)
MSI_LEVELS = {
    1: "Identified compounds (matched to authentic standards with 2+ orthogonal properties: RT, m/z, MS2)",
    2: "Putatively annotated compounds (matched to spectral/physicochemical databases without reference standards)",
    3: "Putatively characterized compound classes (based on physicochemical properties or spectral similarity to known classes)",
    4: "Unknown compounds (detected but not annotated)",
}

# For m/z-only matching (no MS2 data), maximum achievable level is MSI-2 (putative annotation)
# With RT matching against in-house library: can reach MSI-1 if RT + m/z both match
```

### Pattern 5: Metabolomics Normalization Methods

**What:** Standard metabolomics normalization approaches.
**When to use:** MET-03, MET-09.

```python
# PQN (Probabilistic Quotient Normalization) - gold standard for metabolomics
def pqn_normalize(X):
    """PQN: divide each sample by median of quotients vs reference spectrum."""
    # Reference = median spectrum across all samples
    reference = np.nanmedian(X, axis=0)
    # Quotients = sample / reference for each feature
    quotients = X / reference
    # Normalization factor = median of quotients per sample
    norm_factors = np.nanmedian(quotients, axis=1)
    # Divide each sample by its normalization factor
    return X / norm_factors[:, np.newaxis], norm_factors

# TIC (Total Ion Current) normalization
def tic_normalize(X):
    """Normalize each sample by its total intensity."""
    tic = np.nansum(X, axis=1)
    return X / tic[:, np.newaxis] * np.nanmedian(tic), tic

# Internal Standard normalization
def is_normalize(X, is_indices):
    """Normalize each sample by the median of internal standards."""
    is_values = X[:, is_indices]
    norm_factors = np.nanmedian(is_values, axis=1)
    return X / norm_factors[:, np.newaxis], norm_factors
```

### Anti-Patterns to Avoid

- **Mixing zeros and NaN:** Metabolomics data uses NaN for missing values (not detected), NOT zeros. Zeros may represent true measurements. Never convert zeros to NaN or vice versa without explicit user request.
- **Normalizing before filtering:** Always filter (remove low-quality features/samples) BEFORE normalization. Filtering after normalization invalidates the normalization.
- **Log-transforming with zeros/negatives:** Always handle zeros (add small offset or use NaN-aware log) before log transformation. `np.log2(0)` = `-inf`.
- **Using parametric tests on non-normal metabolomics data:** Default to non-parametric (Wilcoxon) unless normality is confirmed. Metabolomics intensity distributions are typically right-skewed.
- **Imputing before QC assessment:** Always run QC FIRST to identify problematic samples, THEN impute. Imputing masks QC issues.
- **Adding heavy dependencies:** The metabolomics domain has many niche Python packages (pymetaboanalyst, metaboanalystR, etc.) but they are poorly maintained. Use scipy/sklearn/numpy for everything.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCA | Custom SVD implementation | `sklearn.decomposition.PCA` | Handles NaN, scaling, variance explained |
| PLS-DA | Custom PLS from scratch | `sklearn.cross_decomposition.PLSRegression` + label encoding | Battle-tested, handles edge cases |
| KNN imputation | Custom KNN | `sklearn.impute.KNNImputer` | Handles NaN natively, configurable |
| Statistical tests | Custom t-test/ANOVA | `scipy.stats.ttest_ind`, `scipy.stats.f_oneway`, `scipy.stats.mannwhitneyu` | Correct p-values, handles edge cases |
| FDR correction | Custom BH procedure | `statsmodels.stats.multitest.multipletests` | Multiple methods (BH, Bonferroni, etc.) |
| Pathway enrichment | Custom ORA/GSEA | `PathwayEnrichmentService` (core, using gseapy) | Already integrated, 100+ databases |
| ComBat batch correction | Custom ComBat | Follow proteomics `ProteomicsPreprocessingService.correct_batch_effects()` pattern | Proven implementation already exists |
| m/z adduct calculation | Custom adduct math | Pre-computed adduct table (dict) | Standard adduct masses are well-known constants |

**Key insight:** The metabolomics domain tempts developers into adding packages like MetaboAnalystR, XCMS, or pymetaboanalyst. Resist this. Everything needed for processed feature tables (which is Lobster's input -- NOT raw instrument data) can be built with scipy/sklearn/numpy/statsmodels.

## Common Pitfalls

### Pitfall 1: Sparse Matrix Mishandling in Metabolomics (BUG-14)
**What goes wrong:** The existing `_validate_intensity_data()` and `_validate_missing_values()` in `metabolomics.py` check `adata.X.data == 0` on sparse matrices, which only examines explicitly stored values. Sparse matrices store non-zero values in `.data`; zeros are implicit. This means `(adata.X.data == 0).sum()` counts stored zeros (from explicit assignment), not all zeros.
**Why it happens:** Confusion between sparse matrix storage format and dense array semantics.
**How to avoid:** For sparse: use `adata.X.nnz` for non-zero count, `total - nnz` for zero count. For missing values: metabolomics data should use NaN (not zero) for missing, but after sparse conversion NaN becomes 0. Use `np.isnan()` only on dense data or track missing pattern separately in layers.
**Warning signs:** Zero percentage calculations that seem too low on sparse metabolomics data.

### Pitfall 2: Confusing Zeros and Missing Values
**What goes wrong:** In metabolomics, missing values (feature not detected) are semantically different from zero intensity. If data is loaded as sparse, NaN becomes 0, losing this distinction.
**Why it happens:** AnnData sparse conversion replaces NaN with 0 for storage efficiency.
**How to avoid:** Store metabolomics data as dense numpy arrays (not sparse). If sparse is needed, track the original missing mask in `adata.layers['missing_mask']`. The `MetabolomicsAdapter` already uses `handle_missing_values="keep"` by default which preserves NaN.
**Warning signs:** Unexpectedly low missing value percentages after loading data.

### Pitfall 3: VIP Score Calculation Errors in PLS-DA
**What goes wrong:** VIP (Variable Importance in Projection) scores are frequently implemented incorrectly, especially the normalization step.
**Why it happens:** Multiple equivalent formulations exist in literature, and the matrix dimensions must align correctly.
**How to avoid:** Use the standard formulation: `VIP_j = sqrt(p * sum_a(SS_a * w_ja^2) / sum_a(SS_a))` where SS_a = sum of squares explained by component a, w_ja = weight of variable j for component a, p = number of variables. Always validate: mean(VIP^2) should equal 1.0.
**Warning signs:** VIP scores with mean != 1, or all VIP scores very similar.

### Pitfall 4: PQN Reference Spectrum with Missing Values
**What goes wrong:** PQN normalization computes quotients against a reference spectrum. If many features have NaN, the median quotient may be unreliable.
**Why it happens:** High missing rate is normal in untargeted metabolomics (30-60%).
**How to avoid:** Use `np.nanmedian` for both reference computation and quotient computation. Require minimum number of non-NaN values per sample for quotient calculation (e.g., at least 50% of features must be present). Warn if quotient is based on fewer than 20 features.
**Warning signs:** Normalization factors with extreme values (>10x or <0.1x).

### Pitfall 5: OPLS-DA Overfitting
**What goes wrong:** OPLS-DA with too many orthogonal components overfits, especially with small sample sizes.
**Why it happens:** Each orthogonal component removes variation, and with few samples the model memorizes noise.
**How to avoid:** Default to 1 orthogonal component (sufficient for most cases). Always run cross-validation (7-fold CV) and permutation tests (n=100). Report Q2, R2, and permutation p-value. Warn if Q2 < 0.5 or permutation p > 0.05.
**Warning signs:** R2 >> Q2 (large gap indicates overfitting).

### Pitfall 6: Lipid Class Analysis Without Standards
**What goes wrong:** Assigning lipid classes based solely on m/z ranges is inaccurate because many lipid classes overlap in mass.
**Why it happens:** LC-MS m/z alone cannot distinguish isomeric lipids.
**How to avoid:** Use lipid class assignment as MSI level 3 (putative class characterization). Always report confidence level. Require adduct information when available. Document limitations in tool output.
**Warning signs:** Lipid class assignments with no confidence context.

## Code Examples

### Service 3-Tuple Pattern (MetabolomicsQualityService)

```python
# Source: follows lobster-proteomics/lobster/services/quality/proteomics_quality_service.py pattern
from typing import Any, Dict, Tuple
import anndata
import numpy as np
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

class MetabolomicsQualityService:
    """Stateless quality service for metabolomics data."""

    def assess_quality(
        self,
        adata: anndata.AnnData,
        qc_label: str = "QC",
        rsd_threshold: float = 30.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """Assess metabolomics data quality."""
        adata_qc = adata.copy()
        X = adata_qc.X if not hasattr(adata_qc.X, 'toarray') else adata_qc.X.toarray()

        # Per-feature RSD
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        rsd = (stds / means) * 100
        rsd = np.nan_to_num(rsd, nan=0.0, posinf=0.0)
        adata_qc.var['rsd'] = rsd

        # QC sample evaluation
        qc_stats = {}
        if 'sample_type' in adata_qc.obs.columns:
            qc_mask = adata_qc.obs['sample_type'] == qc_label
            if qc_mask.any():
                X_qc = X[qc_mask.values]
                qc_rsd = (np.nanstd(X_qc, axis=0) / np.nanmean(X_qc, axis=0)) * 100
                qc_stats = {
                    'n_qc_samples': int(qc_mask.sum()),
                    'median_qc_rsd': float(np.nanmedian(qc_rsd)),
                    'features_below_threshold': int((qc_rsd < rsd_threshold).sum()),
                }

        stats = {
            'n_samples': adata_qc.n_obs,
            'n_features': adata_qc.n_vars,
            'median_rsd': float(np.nanmedian(rsd)),
            'high_rsd_features': int((rsd > rsd_threshold).sum()),
            'missing_pct': float(np.isnan(X).sum() / X.size * 100),
            'tic_cv': float(np.nanstd(np.nansum(X, axis=1)) / np.nanmean(np.nansum(X, axis=1)) * 100),
            **qc_stats,
        }

        ir = self._create_ir_assess_quality(qc_label, rsd_threshold)
        return adata_qc, stats, ir
```

### Tool Wrapping Pattern (shared_tools.py)

```python
# Source: follows lobster-proteomics/lobster/agents/proteomics/shared_tools.py pattern
@tool
def assess_metabolomics_quality(
    modality_name: str,
    qc_label: str = "QC",
    rsd_threshold: float = 30.0,
) -> str:
    """
    Assess quality of metabolomics data including TIC distribution,
    per-feature RSD, QC sample reproducibility, and missing value analysis.

    Args:
        modality_name: Name of the metabolomics modality to assess
        qc_label: Label for QC/pooled samples in sample_type column
        rsd_threshold: RSD threshold for flagging high-variability features

    Returns:
        str: Quality assessment report
    """
    try:
        adata = data_manager.get_modality(modality_name)
    except ValueError:
        return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

    adata_qc, stats, ir = quality_service.assess_quality(
        adata, qc_label=qc_label, rsd_threshold=rsd_threshold
    )

    assessed_name = f"{modality_name}_quality_assessed"
    data_manager.store_modality(
        name=assessed_name,
        adata=adata_qc,
        parent_name=modality_name,
        step_summary=f"Quality assessed: RSD={stats['median_rsd']:.1f}%, missing={stats['missing_pct']:.1f}%",
    )

    data_manager.log_tool_usage(
        tool_name="assess_metabolomics_quality",
        parameters={"modality_name": modality_name, "qc_label": qc_label, "rsd_threshold": rsd_threshold},
        description="Assessed metabolomics data quality",
        ir=ir,
    )

    response = f"Quality assessment complete for '{modality_name}':\n"
    response += f"- Samples: {stats['n_samples']}, Features: {stats['n_features']}\n"
    response += f"- Median RSD: {stats['median_rsd']:.1f}%\n"
    response += f"- High RSD features (>{rsd_threshold}%): {stats['high_rsd_features']}\n"
    response += f"- Missing values: {stats['missing_pct']:.1f}%\n"
    response += f"- TIC CV: {stats['tic_cv']:.1f}%\n"
    response += f"\nNew modality: '{assessed_name}'"
    return response
```

### Agent Factory Pattern (metabolomics_expert.py)

```python
# Source: follows lobster-proteomics/lobster/agents/proteomics/proteomics_expert.py pattern
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="metabolomics_expert",
    display_name="Metabolomics Expert",
    description="Metabolomics analysis: LC-MS/GC-MS/NMR QC, preprocessing, multivariate statistics, metabolite annotation",
    factory_function="lobster.agents.metabolomics.metabolomics_expert.metabolomics_expert",
    handoff_tool_name="handoff_to_metabolomics_expert",
    handoff_tool_description="Assign metabolomics analysis tasks: LC-MS/GC-MS/NMR QC, normalization, univariate/multivariate statistics, metabolite annotation, pathway enrichment",
    child_agents=None,  # No children in v1 (future: targeted_metabolomics_expert, annotation_child)
    supervisor_accessible=True,
    tier_requirement="free",
)

def metabolomics_expert(
    data_manager, callback_handler=None, agent_name="metabolomics_expert",
    delegation_tools=None, workspace_path=None,
    provider_override=None, model_override=None,
):
    """Factory for metabolomics expert agent."""
    # ... standard factory pattern following proteomics_expert ...
```

### BUG-14 Fix Pattern

```python
# BEFORE (BUG-14 - incorrect sparse zero check):
def _validate_intensity_data(adata):
    if hasattr(adata.X, "data"):  # Sparse matrix
        zero_pct = (adata.X.data == 0).sum() / adata.X.data.size * 100  # WRONG
    else:
        zero_pct = (adata.X == 0).sum() / adata.X.size * 100

# AFTER (correct):
def _validate_intensity_data(adata):
    if hasattr(adata.X, "toarray"):  # Sparse matrix
        total = adata.X.shape[0] * adata.X.shape[1]
        # nnz counts explicitly stored values; zeros are implicit
        nonzero = adata.X.nnz
        # But some stored values might be actual zeros (from explicit assignment)
        stored_zeros = (adata.X.data == 0).sum() if hasattr(adata.X, 'data') else 0
        actual_nonzero = nonzero - stored_zeros
        zero_pct = (total - actual_nonzero) / total * 100
    else:
        zero_pct = (adata.X == 0).sum() / adata.X.size * 100
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MetaboAnalyst (R/web only) | Python-native pipelines (sklearn, scipy) | 2020+ | No R dependency needed |
| XCMS for peak picking | Lobster receives processed feature tables | N/A | Out of scope -- Lobster is post-processing |
| SIMCA (commercial) for OPLS-DA | Open-source implementations (pyopls, custom) | 2018+ | No commercial license needed |
| Manual m/z annotation | Automated database matching (HMDB, KEGG) | 2015+ | Consistent MSI levels |
| Separate tools for each stat | Unified service pattern | This phase | Consistent 3-tuple provenance |

**Deprecated/outdated:**
- `pyopls` (last release 2020, unmaintained) -- implement OPLS-DA from scratch
- `pyMetaboanalyst` (API wrapper, fragile) -- use scipy/sklearn directly
- `mzML` parsing (raw data) -- out of scope for Lobster

## Open Questions

1. **m/z Reference Database Size**
   - What we know: HMDB has ~220K metabolites, KEGG has ~19.5K compounds. Full HMDB download is large.
   - What's unclear: Should we bundle a curated subset or the full database?
   - Recommendation: Bundle a curated subset of ~5K common metabolites with accurate monoisotopic masses and common adducts. This covers 90%+ of typical untargeted metabolomics features. Full database lookup can be a v2 enhancement (MET-V2-02).

2. **QC-RLSC Implementation Complexity**
   - What we know: QC-RLSC (QC-based Robust LOESS Signal Correction) is the gold standard for batch correction in metabolomics. It requires QC samples injected throughout the run.
   - What's unclear: The LOESS fitting can be tricky with few QC samples (< 5 per batch).
   - Recommendation: Implement using scipy's `UnivariateSpline` or `lowess` from statsmodels. Fall back to median centering when fewer than 5 QC samples per batch. Always warn about minimum QC requirements.

3. **Future Child Agent Architecture**
   - What we know: The package must be "structured for future child agents" (MET-01). v2 requirements list targeted metabolomics child (MET-V2-01) and annotation child (MET-V2-02).
   - What's unclear: Exactly which tools should be on the parent vs future children.
   - Recommendation: Keep all 10 tools on parent for v1. Use `child_agents=None` in AGENT_CONFIG but structure code so tools can be easily moved to separate agent files later (tools in `shared_tools.py`, services in separate files).

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `packages/lobster-proteomics/` -- complete package reference for structure, patterns, entry points
- Codebase inspection: `lobster/core/adapters/metabolomics_adapter.py` -- existing adapter infrastructure
- Codebase inspection: `lobster/core/schemas/metabolomics.py` -- existing schema + BUG-14 location
- Codebase inspection: `lobster/core/omics_registry.py` -- metabolomics omics type registered
- sklearn 1.8.0 docs: `PLSRegression` in `sklearn.cross_decomposition` -- verified available
- scipy 1.17.0: statistical tests (ttest_ind, mannwhitneyu, f_oneway, kruskal) -- verified available
- gseapy 1.1.11: pathway enrichment via Enrichr -- verified available in core

### Secondary (MEDIUM confidence)
- PyPI: pyopls 20.3.post1 (March 2020) -- confirmed unmaintained, do not use as dependency
- KEGG REST API: 19,571 compounds in database (release 117.0) -- verified via API
- MSI identification levels (Sumner et al. 2007, Metabolomics 3:211-221) -- well-established standard

### Tertiary (LOW confidence)
- HMDB compound count (~220K) -- based on training knowledge, not directly verified this session

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified as installed and available
- Architecture: HIGH -- exact patterns verified from proteomics package codebase
- Pitfalls: HIGH -- BUG-14 confirmed by code inspection, domain pitfalls well-established
- Domain science (MSI, PQN, OPLS-DA): MEDIUM -- standard metabolomics methodology, well-established in literature

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (30 days -- stable domain, no fast-moving dependencies)
