"""
Genomics data adapters for DNA sequencing data.

This package provides adapters for genomics data formats including:
- VCF (Variant Call Format) for whole genome sequencing
- PLINK (.bed/.bim/.fam) for SNP array data
"""

from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter
from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

__all__ = ["VCFAdapter", "PLINKAdapter"]
