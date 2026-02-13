# lobster-proteomics

Unified proteomics analysis for mass spectrometry (DDA/DIA) and affinity platforms (Olink, SomaScan).

## Installation

```bash
pip install lobster-proteomics
```

## Agents

| Agent | Description |
|-------|-------------|
| `proteomics_expert` | Full proteomics workflow orchestration with auto-detection of platform type and appropriate preprocessing. |

## Services

| Service | Purpose |
|---------|---------|
| ProteomicsAnalysisService | Core analysis workflows for protein quantification |
| ProteomicsDifferentialService | Differential protein expression between conditions |
| ProteomicsSurvivalService | Survival analysis with protein biomarkers |
| ProteomicsNetworkService | Protein-protein interaction network analysis |
| ProteomicsQualityService | Quality control and missing value assessment |
| ProteomicsPreprocessingService | MNAR/MAR-aware imputation and normalization |
| ProteomicsVisualizationService | Volcano plots, heatmaps, and network graphs |
| MaxQuantParser | Parse MaxQuant proteinGroups.txt output |
| DIANNParser | Parse DIA-NN report.tsv output |
| SpectronaultParser | Parse Spectronaut report output |
| OlinkParser | Parse Olink NPX export files |

## Features

- Auto-detection of platform type from data characteristics
- Missing value handling optimized for MNAR (mass spec) vs MAR (affinity) patterns
- Platform-appropriate quality control with batch effect detection
- Median and quantile normalization with log2 transformation
- Multiple imputation strategies (MinDet, KNN, zero, median)
- Differential protein analysis with multiple testing correction
- Volcano plots and MA plots for results visualization
- Protein-protein interaction network construction and visualization
- Survival analysis integration for clinical proteomics studies
- Support for multi-plex affinity platforms (Olink, SomaScan, Luminex)

## Platform Support

| Platform | Missing Values | Normalization | Notes |
|----------|---------------|---------------|-------|
| Mass Spectrometry (DDA/DIA) | 30-70% (MNAR) | Median + log2 | Peptide mapping support |
| Affinity (Olink) | <30% (MAR) | Quantile | Plate effect correction |
| Affinity (SomaScan) | <30% (MAR) | Quantile | Antibody validation |

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0

## Tier Requirement

This is a **premium** agent. Access is controlled at runtime via Lobster AI's tier system.

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/proteomics](https://docs.omics-os.com/docs/agents/proteomics)

## License

MIT
