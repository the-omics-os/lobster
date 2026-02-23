# lobster-metabolomics

Metabolomics analysis for LC-MS, GC-MS, and NMR untargeted metabolomics workflows.

## Installation

```bash
pip install lobster-metabolomics
```

## Agents

| Agent | Description |
|-------|-------------|
| `metabolomics_expert` | Full metabolomics workflow orchestration with auto-detection of platform type (LC-MS, GC-MS, NMR) and appropriate preprocessing. |

## Services

| Service | Purpose |
|---------|---------|
| MetabolomicsQualityService | QC assessment: RSD, TIC CV, QC sample evaluation, missing value analysis |
| MetabolomicsPreprocessingService | Feature filtering, imputation (KNN, min, LOD/2, median, MICE), normalization (PQN, TIC, IS, median, quantile), batch correction (ComBat, median centering, QC-RLSC) |
| MetabolomicsAnalysisService | Univariate statistics (t-test, Wilcoxon, ANOVA, Kruskal-Wallis with FDR), PCA, PLS-DA with VIP scores, OPLS-DA, fold change analysis |
| MetabolomicsAnnotationService | m/z-based metabolite annotation against bundled reference database (HMDB/KEGG), MSI confidence levels, lipid class classification |

## Features

- Auto-detection of platform type from data characteristics (LC-MS, GC-MS, NMR)
- Platform-specific defaults for QC thresholds, normalization, and imputation
- Multiple normalization methods including PQN (gold standard for metabolomics)
- Supervised multivariate analysis (PLS-DA, OPLS-DA) with permutation testing
- VIP score calculation for biomarker discovery
- Metabolite annotation with MSI confidence levels (1-4)
- Lipid class classification from m/z ranges

## Platform Support

| Platform | Missing Values | Normalization | Notes |
|----------|---------------|---------------|-------|
| LC-MS | 20-60% | PQN + log2 | Most common untargeted platform |
| GC-MS | 10-40% | TIC + log2 | Lower missing rates, derivatization |
| NMR | 0-10% | PQN (no log) | Near-complete feature detection |

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0

## Tier Requirement

This is a **free** agent. Available to all Lobster AI users.

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/metabolomics](https://docs.omics-os.com/docs/agents/metabolomics)

## License

AGPL-3.0-or-later
