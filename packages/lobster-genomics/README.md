# lobster-genomics

Genomics analysis for VCF and PLINK data with GWAS, QC, and variant annotation capabilities.

## Installation

```bash
pip install lobster-genomics
```

## Agents

| Agent | Description |
|-------|-------------|
| `genomics_expert` | Full genomics workflow orchestration including data loading, quality control, GWAS analysis, and variant annotation. |

## Services

| Service | Purpose |
|---------|---------|
| GWASService | Genome-wide association study analysis with linear/logistic regression |
| VariantAnnotationService | Variant functional annotation via Ensembl VEP integration |
| GenomicsQualityService | Comprehensive QC including call rate, MAF, HWE, and heterozygosity |

## Features

- Load VCF files (whole genome sequencing) and PLINK files (SNP arrays)
- Comprehensive quality control with configurable thresholds
- Call rate filtering at both sample and variant level
- Minor allele frequency (MAF) filtering for rare variant removal
- Hardy-Weinberg equilibrium testing for genotyping errors
- Heterozygosity outlier detection for sample quality assessment
- Linear and logistic regression GWAS with covariate support
- Lambda GC calculation for population stratification assessment
- Principal component analysis (PCA) for ancestry detection
- Variant functional annotation via Ensembl VEP API
- Manhattan plots and QQ plots for GWAS visualization
- PCA scatter plots colored by phenotype or ancestry

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0
- cyvcf2 (VCF parsing)
- bed-reader (PLINK file support)
- sgkit (GWAS and PCA analysis)

## Tier Requirement

This is a **premium** agent. Access is controlled at runtime via Lobster AI's tier system.

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/genomics](https://docs.omics-os.com/docs/agents/genomics)

## License

MIT
