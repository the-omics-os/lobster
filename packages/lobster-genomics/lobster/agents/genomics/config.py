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
    "SUPPORTED_FORMATS",
]

# Agent metadata
AGENT_NAME = "genomics_expert"
AGENT_DISPLAY_NAME = "Genomics Expert"
AGENT_DESCRIPTION = (
    "WGS and SNP array analysis: QC, filtering, GWAS, variant annotation"
)

# Default QC thresholds (UK Biobank standards)
DEFAULT_MIN_CALL_RATE = 0.95  # Minimum call rate (samples/variants)
DEFAULT_MIN_MAF = 0.01  # Minimum minor allele frequency
DEFAULT_HWE_PVALUE = 1e-6  # Minimum Hardy-Weinberg equilibrium p-value

# Supported file formats
SUPPORTED_FORMATS = [".vcf", ".vcf.gz", ".bcf", ".bed", ".bim", ".fam"]
