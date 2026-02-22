"""
Configuration for genomics analysis.

This module defines constants and defaults for WGS and SNP array analysis.
"""

__all__ = [
    "AGENT_NAME",
    "AGENT_DISPLAY_NAME",
    "AGENT_DESCRIPTION",
    "DEFAULT_MIN_CALL_RATE",
    "DEFAULT_MIN_MAF",
    "DEFAULT_HWE_PVALUE",
    "DEFAULT_LD_THRESHOLD",
    "DEFAULT_KINSHIP_THRESHOLD",
    "DEFAULT_CLUMP_KB",
    "SUPPORTED_FORMATS",
]

# Agent metadata
AGENT_NAME = "genomics_expert"
AGENT_DISPLAY_NAME = "Genomics Expert"
AGENT_DESCRIPTION = (
    "WGS and SNP array analysis: QC, filtering, GWAS, LD pruning, kinship, variant annotation"
)

# Default QC thresholds (UK Biobank standards)
DEFAULT_MIN_CALL_RATE = 0.95  # Minimum call rate (samples/variants)
DEFAULT_MIN_MAF = 0.01  # Minimum minor allele frequency
DEFAULT_HWE_PVALUE = 1e-6  # Minimum Hardy-Weinberg equilibrium p-value

# Default GWAS pipeline thresholds
DEFAULT_LD_THRESHOLD = 0.2  # r² threshold for LD pruning
DEFAULT_KINSHIP_THRESHOLD = 0.125  # Kinship coefficient for 3rd degree relatives
DEFAULT_CLUMP_KB = 250  # Clumping window in kilobases

# Supported file formats
SUPPORTED_FORMATS = [".vcf", ".vcf.gz", ".bcf", ".bed", ".bim", ".fam"]
