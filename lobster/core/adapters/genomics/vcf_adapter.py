"""
VCF adapter for genomics data with schema enforcement.

This module provides the VCFAdapter that handles loading and validation
of VCF (Variant Call Format) files for whole genome sequencing data
with appropriate schema enforcement and quality control.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.genomics import GenomicsSchema
from lobster.core.utils.h5ad_utils import sanitize_value

logger = logging.getLogger(__name__)


class VCFAdapter(BaseAdapter):
    """
    Adapter for VCF genomics data with schema enforcement.

    This adapter handles loading and validation of VCF files from whole
    genome sequencing with appropriate schema validation and quality control.
    Converts VCF format to AnnData with samples as observations and variants
    as variables.

    Key Features:
        - Handles cyvcf2 buffer reuse bug with explicit array copying
        - Supports region-based filtering and sample subsetting
        - Quality filtering (QUAL, FILTER, missing data)
        - 0/1/2 genotype encoding (homref/het/homalt)
        - Sparse matrix support for large datasets
        - Comprehensive variant and sample metadata extraction
    """

    def __init__(self, strict_validation: bool = False):
        """
        Initialize the VCF adapter.

        Args:
            strict_validation: Whether to use strict validation
        """
        super().__init__(name="VCFAdapter")
        self.strict_validation = strict_validation

        # Create validator for WGS data
        self.validator = GenomicsSchema.create_validator(
            schema_type="wgs", strict=strict_validation
        )

        # Get QC thresholds
        self.qc_thresholds = GenomicsSchema.get_recommended_qc_thresholds("wgs")

    def from_source(
        self, source: Union[str, Path, pd.DataFrame], **kwargs
    ) -> anndata.AnnData:
        """
        Convert VCF file to AnnData with genomics schema.

        Args:
            source: Path to VCF file (can be gzipped)
            **kwargs: Additional parameters:
                - region: Genomic region (e.g., "chr1:1000-2000")
                - samples: List of sample IDs to load (None = all samples)
                - filter_pass: Only load PASS variants (default True)
                - min_qual: Minimum QUAL score (default None)
                - max_missing: Max missing genotype fraction (default None)
                - max_variants: Maximum number of variants to load (default None)
                - sparse_threshold: Use sparse matrix if n_variants > threshold (default 1M)

        Returns:
            anndata.AnnData: Loaded and validated genomics data
                - X: Genotype matrix (samples × variants), 0/1/2 encoding
                - obs: Sample metadata (sample_id)
                - var: Variant metadata (CHROM, POS, REF, ALT, ID, QUAL, FILTER, AF)
                - layers['GT']: Original genotype data
                - layers['DP']: Depth (if available)
                - layers['GQ']: Quality (if available)
                - uns: vcf_header metadata, source_file, data_type='genomics', modality='wgs'

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
            ImportError: If cyvcf2 is not installed
        """
        self._log_operation("loading", source=str(source))

        # Validate source is a file path
        if not isinstance(source, (str, Path)):
            raise TypeError(
                f"VCFAdapter only supports file paths, got {type(source).__name__}"
            )

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"VCF file not found: {source_path}")

        try:
            # Import cyvcf2
            try:
                import cyvcf2
            except ImportError:
                raise ImportError(
                    "cyvcf2 is required for VCF loading. Install with: pip install cyvcf2"
                )

            # Extract parameters
            region = kwargs.get("region")
            sample_filter = kwargs.get("samples")
            filter_pass = kwargs.get("filter_pass", True)
            min_qual = kwargs.get("min_qual")
            max_missing = kwargs.get("max_missing")
            max_variants = kwargs.get("max_variants")
            sparse_threshold = kwargs.get("sparse_threshold", 1_000_000)

            # Open VCF file with gts012=True for 0/1/2 encoding
            vcf = cyvcf2.VCF(str(source_path), gts012=True)

            # Filter samples if specified
            if sample_filter:
                if isinstance(sample_filter, str):
                    sample_filter = [sample_filter]
                # Validate samples exist
                available_samples = vcf.samples
                invalid_samples = set(sample_filter) - set(available_samples)
                if invalid_samples:
                    raise ValueError(
                        f"Samples not found in VCF: {sorted(invalid_samples)}"
                    )
                vcf.set_samples(sample_filter)

            samples = vcf.samples
            n_samples = len(samples)

            # Parse variants (with region filtering if specified)
            genotypes = []
            variant_metadata = []

            if region:
                variant_iter = vcf(region)
            else:
                variant_iter = vcf

            for variant in variant_iter:
                # Check max_variants limit
                if max_variants is not None and len(genotypes) >= max_variants:
                    break

                # Apply FILTER field filtering
                if (
                    filter_pass
                    and variant.FILTER is not None
                    and variant.FILTER != "PASS"
                ):
                    continue

                # Apply QUAL filtering
                if min_qual is not None and (
                    variant.QUAL is None or variant.QUAL < min_qual
                ):
                    continue

                # CRITICAL: Copy genotype data to avoid buffer reuse bug
                # cyvcf2's variant.gt_types returns a VIEW that gets overwritten
                gt = np.array(variant.gt_types, copy=True)

                # Apply missing data filtering
                if max_missing is not None:
                    missing_fraction = (gt == -1).sum() / len(gt)
                    if missing_fraction > max_missing:
                        continue

                genotypes.append(gt)

                # Extract variant metadata
                # Create variant ID: use rsID if available, else CHROM:POS:REF>ALT
                variant_id = variant.ID if variant.ID else None
                if not variant_id or variant_id == ".":
                    alt_allele = variant.ALT[0] if variant.ALT else "."
                    variant_id = (
                        f"{variant.CHROM}:{variant.POS}:{variant.REF}>{alt_allele}"
                    )

                # Extract allele frequency (AF) from INFO field
                af = None
                if "AF" in variant.INFO:
                    af_val = variant.INFO.get("AF")
                    if af_val is not None:
                        # AF can be a list for multi-allelic sites
                        af = af_val[0] if isinstance(af_val, (list, tuple)) else af_val

                variant_metadata.append(
                    {
                        "variant_id": variant_id,
                        "CHROM": variant.CHROM,
                        "POS": variant.POS,
                        "REF": variant.REF,
                        "ALT": variant.ALT[0] if variant.ALT else ".",
                        "ID": variant.ID if variant.ID else ".",
                        "QUAL": variant.QUAL if variant.QUAL is not None else np.nan,
                        "FILTER": variant.FILTER if variant.FILTER is not None else ".",
                        "AF": af if af is not None else np.nan,
                    }
                )

            if len(genotypes) == 0:
                raise ValueError(
                    f"No variants passed filters. "
                    f"Filters: filter_pass={filter_pass}, min_qual={min_qual}, "
                    f"max_missing={max_missing}, region={region}"
                )

            # Convert to numpy array and transpose to (samples × variants)
            # Each row in genotypes is a variant, each element is a sample
            X = np.vstack(
                genotypes
            ).T  # Transpose: (n_variants, n_samples) → (n_samples, n_variants)

            n_variants = len(genotypes)
            self.logger.info(
                f"Loaded {n_samples} samples × {n_variants} variants from VCF"
            )

            # Use sparse matrix if dataset is large
            if n_variants > sparse_threshold:
                # For genotype data, sparse representation saves memory
                # Most genotypes are 0 (homozygous reference)
                sparsity = (X == 0).sum() / X.size
                if sparsity > 0.5:
                    X = csr_matrix(X.astype(np.float32))
                    self.logger.info(
                        f"Converted to sparse matrix (sparsity: {sparsity:.2%})"
                    )
                else:
                    X = X.astype(np.float32)
            else:
                X = X.astype(np.float32)

            # Create observation (sample) metadata
            obs_df = pd.DataFrame({"sample_id": samples}, index=samples)

            # Create variable (variant) metadata
            var_df = pd.DataFrame(variant_metadata)
            var_df.index = var_df["variant_id"]
            var_df = var_df.drop(columns=["variant_id"])

            # Create AnnData object
            adata = anndata.AnnData(X=X, obs=obs_df, var=var_df)

            # Add genotype layer (copy of X for reference)
            adata.layers["GT"] = X.toarray() if hasattr(X, "toarray") else X.copy()

            # Extract FORMAT fields if available (DP, GQ)
            # Note: This requires re-parsing the VCF, so we only do it if requested
            # For now, we'll skip this to optimize performance
            # Future enhancement: Add extract_format_fields parameter

            # Add VCF header metadata to uns
            vcf_metadata = {
                "fileformat": vcf.raw_header.split("\n")[0]
                if vcf.raw_header
                else "VCFv4.2",
                "source": str(source_path),
                "n_samples": n_samples,
                "n_variants": n_variants,
                "samples": samples,
            }

            # Add contigs from header
            if hasattr(vcf, "seqnames"):
                vcf_metadata["contigs"] = vcf.seqnames

            adata.uns["vcf_metadata"] = sanitize_value(vcf_metadata)
            adata.uns["source_file"] = str(source_path)
            adata.uns["data_type"] = "genomics"
            adata.uns["modality"] = "wgs"

            # Store filtering parameters
            filter_params = {
                "region": region,
                "samples": sample_filter,
                "filter_pass": filter_pass,
                "min_qual": min_qual,
                "max_missing": max_missing,
            }
            adata.uns["filter_params"] = sanitize_value(filter_params)

            # Close VCF file
            vcf.close()

            # Apply genomics-specific preprocessing
            adata = self.preprocess_data(adata, **kwargs)

            self.logger.info(
                f"Successfully loaded VCF: {adata.n_obs} samples × {adata.n_vars} variants"
            )
            return adata

        except Exception as e:
            self.logger.error(f"Failed to load VCF data from {source_path}: {e}")
            raise

    def validate(self, adata: anndata.AnnData, strict: bool = None) -> ValidationResult:
        """
        Validate AnnData against genomics WGS schema.

        Args:
            adata: AnnData object to validate
            strict: Override default strict setting

        Returns:
            ValidationResult: Validation results
        """
        if strict is None:
            strict = self.strict_validation

        # Use the configured validator
        result = self.validator.validate(adata, strict=strict)

        # Add basic structural validation
        basic_result = self._validate_basic_structure(adata)
        result = result.merge(basic_result)

        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for WGS genomics data.

        Returns:
            Dict[str, Any]: WGS schema definition
        """
        return GenomicsSchema.get_wgs_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ["vcf", "vcf.gz", "bcf"]

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply genomics-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing (numeric matrix, basic metadata)
        adata = super().preprocess_data(adata, **kwargs)

        # Add genomics-specific metadata
        adata = self._add_genomics_metadata(adata)

        return adata

    def _add_genomics_metadata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Add genomics-specific metadata to AnnData object.

        Calculates per-sample and per-variant statistics:
        - Sample-level: variant_count, missing_rate, het_rate
        - Variant-level: n_samples_called, missing_rate, maf (minor allele frequency)

        Args:
            adata: AnnData object to annotate

        Returns:
            anndata.AnnData: Annotated AnnData object
        """
        # Get genotype matrix (use layer if X is sparse)
        if "GT" in adata.layers:
            gt = adata.layers["GT"]
        else:
            gt = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # Calculate per-sample metrics
        if "variant_count" not in adata.obs.columns:
            # Count non-missing variants per sample
            adata.obs["variant_count"] = (gt != -1).sum(axis=1)

        if "missing_rate" not in adata.obs.columns:
            # Missing genotype rate per sample
            adata.obs["missing_rate"] = (gt == -1).sum(axis=1) / adata.n_vars

        if "het_rate" not in adata.obs.columns:
            # Heterozygosity rate per sample
            het_count = (gt == 1).sum(axis=1)
            called_count = (gt != -1).sum(axis=1)
            adata.obs["het_rate"] = np.where(
                called_count > 0, het_count / called_count, 0.0
            )

        # Calculate per-variant metrics
        if "n_samples_called" not in adata.var.columns:
            # Number of samples with called genotypes
            adata.var["n_samples_called"] = (gt != -1).sum(axis=0)

        if "missing_rate" not in adata.var.columns:
            # Missing genotype rate per variant
            adata.var["missing_rate"] = (gt == -1).sum(axis=0) / adata.n_obs

        if "maf" not in adata.var.columns:
            # Minor allele frequency (MAF)
            # Count alleles: 0=0 alt, 1=1 alt, 2=2 alt, -1=missing
            called_mask = gt != -1
            called_count = called_mask.sum(axis=0)

            # Count alternate alleles (1 = 1 allele, 2 = 2 alleles)
            alt_count = np.where(called_mask, gt, 0).sum(axis=0)
            total_alleles = called_count * 2  # Diploid

            # Calculate allele frequency
            af = np.where(total_alleles > 0, alt_count / total_alleles, 0.0)

            # MAF is min(AF, 1-AF)
            maf = np.minimum(af, 1 - af)
            adata.var["maf"] = maf

        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate genomics-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary with:
                - Sample-level: mean_missing_rate, mean_het_rate
                - Variant-level: mean_variant_missing_rate, mean_qual, pass_rate
                - Distribution: common_variants, rare_variants
        """
        metrics = super().get_quality_metrics(adata)

        # Sample-level metrics
        if "missing_rate" in adata.obs.columns:
            metrics["mean_sample_missing_rate"] = float(
                adata.obs["missing_rate"].mean()
            )
            metrics["max_sample_missing_rate"] = float(adata.obs["missing_rate"].max())

        if "het_rate" in adata.obs.columns:
            metrics["mean_het_rate"] = float(adata.obs["het_rate"].mean())

        # Variant-level metrics
        if "missing_rate" in adata.var.columns:
            metrics["mean_variant_missing_rate"] = float(
                adata.var["missing_rate"].mean()
            )
            metrics["high_missing_variants"] = int(
                (adata.var["missing_rate"] > 0.1).sum()
            )

        if "QUAL" in adata.var.columns:
            qual_vals = adata.var["QUAL"].dropna()
            if len(qual_vals) > 0:
                metrics["mean_qual"] = float(qual_vals.mean())
                metrics["low_qual_variants"] = int((qual_vals < 20).sum())

        if "FILTER" in adata.var.columns:
            filter_vals = adata.var["FILTER"]
            metrics["pass_variants"] = int((filter_vals == "PASS").sum())
            metrics["pass_rate"] = float(
                (filter_vals == "PASS").sum() / len(filter_vals)
            )

        # MAF distribution
        if "maf" in adata.var.columns:
            maf = adata.var["maf"]
            metrics["common_variants"] = int((maf >= 0.05).sum())
            metrics["rare_variants"] = int((maf < 0.01).sum())
            metrics["mean_maf"] = float(maf.mean())

        return metrics

    def detect_format(self, source: Union[str, Path]) -> Optional[str]:
        """
        Detect the format of a genomics file.

        Args:
            source: Path to the source file

        Returns:
            Optional[str]: Detected format name, None if unknown
        """
        if isinstance(source, (str, Path)):
            path = Path(source)

            # Check for VCF formats
            if path.suffix == ".vcf" or path.name.endswith(".vcf.gz"):
                return "vcf"
            elif path.suffix == ".bcf":
                return "bcf"

        return None
