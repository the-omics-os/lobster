"""
PLINK data adapter with schema enforcement.

This module provides the PLINKAdapter that handles loading, validation, and
preprocessing of PLINK format genomics data (.bed/.bim/.fam files) with
appropriate schema enforcement for SNP array data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.genomics import GenomicsSchema

logger = logging.getLogger(__name__)


class PLINKAdapter(BaseAdapter):
    """
    Adapter for PLINK format genomics data with schema enforcement.

    This adapter handles loading and validation of PLINK .bed/.bim/.fam files
    with appropriate schema validation for SNP array data. PLINK is a widely
    used format for genome-wide association studies (GWAS).

    PLINK Format:
        - .bed: Binary genotype data (samples × SNPs)
        - .bim: SNP information (chr, ID, cM, bp, A1, A2)
        - .fam: Sample/individual information (FID, IID, father, mother, sex, phenotype)
    """

    def __init__(self, strict_validation: bool = False):
        """
        Initialize the PLINK adapter.

        Args:
            strict_validation: Whether to use strict validation
        """
        super().__init__(name="PLINKAdapter")

        self.strict_validation = strict_validation

        # Create validator for SNP array data
        self.validator = GenomicsSchema.create_validator(
            schema_type="snp_array", strict=strict_validation
        )

        # Get QC thresholds
        self.qc_thresholds = GenomicsSchema.get_recommended_qc_thresholds("snp_array")

    def from_source(self, source: Union[str, Path], **kwargs) -> anndata.AnnData:
        """
        Convert PLINK files to AnnData with genomics schema.

        Args:
            source: Path to PLINK .bed file (or prefix without extension)
            **kwargs: Additional parameters:
                - iid_filter: List of individual IDs to load (subset samples)
                - sid_filter: List of SNP IDs to load (subset SNPs)
                - maf_min: Minimum minor allele frequency threshold
                - count_A1: If True, count A1 allele (default: True)

        Returns:
            anndata.AnnData: Loaded and validated PLINK data

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If PLINK files don't exist
            ImportError: If bed-reader package is not installed
        """
        self._log_operation("loading", source=str(source))

        try:
            # Import bed-reader (optional dependency)
            try:
                from bed_reader import open_bed
            except ImportError:
                raise ImportError(
                    "bed-reader package is required for PLINK support. "
                    "Install with: pip install bed-reader"
                )

            # Resolve PLINK file paths
            bed_path = self._resolve_plink_path(source)

            # Extract kwargs
            iid_filter = kwargs.get("iid_filter", None)
            sid_filter = kwargs.get("sid_filter", None)
            maf_min = kwargs.get("maf_min", None)
            count_A1 = kwargs.get("count_A1", True)

            # Open PLINK files with bed-reader
            bed = open_bed(str(bed_path), count_A1=count_A1)

            # Build metadata DataFrames from bed-reader properties
            # bed-reader doesn't expose .fam and .bim as DataFrames,
            # but provides individual properties that we construct into DataFrames
            # Column names match bed-reader's internal naming convention
            fam_data = pd.DataFrame(
                {
                    0: bed.fid,  # Family ID
                    1: bed.iid,  # Individual ID
                    2: bed.father,  # Father ID
                    3: bed.mother,  # Mother ID
                    4: bed.sex,  # Sex (1=male, 2=female, 0=unknown)
                    5: bed.pheno,  # Phenotype
                }
            )

            bim_data = pd.DataFrame(
                {
                    0: bed.chromosome,  # Chromosome
                    1: bed.sid,  # SNP ID
                    2: bed.cm_position,  # Genetic distance (cM)
                    3: bed.bp_position,  # Physical position (bp)
                    4: bed.allele_1,  # Allele 1
                    5: bed.allele_2,  # Allele 2
                }
            )

            # Resolve indices for filtering
            sample_idx = None
            variant_idx = None

            if iid_filter is not None:
                sample_idx = self._resolve_iid_indices(fam_data, iid_filter)
                logger.info(
                    f"Filtering to {len(sample_idx)} individuals from {len(iid_filter)} requested IDs"
                )

            if sid_filter is not None:
                variant_idx = self._resolve_sid_indices(bim_data, sid_filter)
                logger.info(
                    f"Filtering to {len(variant_idx)} SNPs from {len(sid_filter)} requested IDs"
                )

            # Read genotype data (with optional subsetting)
            if sample_idx is not None or variant_idx is not None:
                genotypes = bed.read(index=sample_idx, sid_index=variant_idx)
            else:
                genotypes = bed.read()

            # Apply MAF filter if specified
            if maf_min is not None and maf_min > 0:
                genotypes, variant_idx = self._apply_maf_filter(
                    genotypes, maf_min, bim_data, variant_idx
                )
                logger.info(
                    f"Filtered to {genotypes.shape[1]} SNPs with MAF >= {maf_min}"
                )

            # Update metadata DataFrames based on filtering
            if sample_idx is not None:
                fam_data = fam_data.iloc[sample_idx].reset_index(drop=True)
            if variant_idx is not None:
                bim_data = bim_data.iloc[variant_idx].reset_index(drop=True)

            # Create AnnData object
            adata = self._create_anndata_from_plink(
                genotypes, fam_data, bim_data, bed_path
            )

            # Apply preprocessing
            adata = self.preprocess_data(adata, **kwargs)

            self.logger.info(
                f"Loaded PLINK data: {adata.n_obs} individuals × {adata.n_vars} SNPs"
            )
            return adata

        except Exception as e:
            self.logger.error(f"Failed to load PLINK data from {source}: {e}")
            raise

    def _resolve_plink_path(self, source: Union[str, Path]) -> Path:
        """
        Resolve PLINK .bed file path from various input formats.

        Args:
            source: Path to .bed file or prefix

        Returns:
            Path: Resolved .bed file path

        Raises:
            FileNotFoundError: If PLINK files don't exist
        """
        source_path = Path(source)

        # If source has .bed extension, use it directly
        if source_path.suffix == ".bed":
            bed_path = source_path
        else:
            # Assume source is a prefix, add .bed extension
            bed_path = source_path.with_suffix(".bed")

        # Check if .bed file exists
        if not bed_path.exists():
            raise FileNotFoundError(
                f"PLINK .bed file not found: {bed_path}\n"
                f"Ensure all three PLINK files exist (.bed, .bim, .fam)"
            )

        # Check for companion .bim and .fam files
        bim_path = bed_path.with_suffix(".bim")
        fam_path = bed_path.with_suffix(".fam")

        if not bim_path.exists():
            raise FileNotFoundError(f"PLINK .bim file not found: {bim_path}")
        if not fam_path.exists():
            raise FileNotFoundError(f"PLINK .fam file not found: {fam_path}")

        return bed_path

    def _resolve_iid_indices(
        self, fam_data: pd.DataFrame, iid_filter: List[str]
    ) -> np.ndarray:
        """
        Resolve individual IDs to indices for subsetting.

        Args:
            fam_data: DataFrame from .fam file
            iid_filter: List of individual IDs to keep

        Returns:
            np.ndarray: Indices of individuals to keep
        """
        # PLINK .fam file: FID, IID, father, mother, sex, phenotype
        # bed-reader returns columns: fid, iid, father, mother, sex, pheno
        iid_column = "iid" if "iid" in fam_data.columns else 1

        # Create mapping of IID -> index
        iid_to_idx = {iid: idx for idx, iid in enumerate(fam_data[iid_column])}

        # Resolve indices
        indices = []
        for iid in iid_filter:
            if iid in iid_to_idx:
                indices.append(iid_to_idx[iid])
            else:
                logger.warning(f"Individual ID '{iid}' not found in .fam file")

        return np.array(indices, dtype=np.int64)

    def _resolve_sid_indices(
        self, bim_data: pd.DataFrame, sid_filter: List[str]
    ) -> np.ndarray:
        """
        Resolve SNP IDs to indices for subsetting.

        Args:
            bim_data: DataFrame from .bim file
            sid_filter: List of SNP IDs to keep

        Returns:
            np.ndarray: Indices of SNPs to keep
        """
        # PLINK .bim file: chr, snp_id, cm_pos, bp_pos, a1, a2
        # bed-reader returns columns: chrom, snp, cm, pos, a0, a1
        sid_column = "snp" if "snp" in bim_data.columns else 1

        # Create mapping of SNP ID -> index
        sid_to_idx = {sid: idx for idx, sid in enumerate(bim_data[sid_column])}

        # Resolve indices
        indices = []
        for sid in sid_filter:
            if sid in sid_to_idx:
                indices.append(sid_to_idx[sid])
            else:
                logger.warning(f"SNP ID '{sid}' not found in .bim file")

        return np.array(indices, dtype=np.int64)

    def _apply_maf_filter(
        self,
        genotypes: np.ndarray,
        maf_min: float,
        bim_data: pd.DataFrame,
        variant_idx: Optional[np.ndarray],
    ) -> tuple:
        """
        Filter SNPs by minimum minor allele frequency.

        Args:
            genotypes: Genotype matrix (samples × SNPs)
            maf_min: Minimum MAF threshold
            bim_data: DataFrame from .bim file
            variant_idx: Current variant indices (or None)

        Returns:
            Tuple of (filtered_genotypes, updated_variant_idx)
        """
        # Calculate MAF for each SNP
        # Genotypes: 0 (hom A1), 1 (het), 2 (hom A2), NaN (missing)
        # MAF = min(freq(A1), freq(A2))

        n_samples = genotypes.shape[0]
        maf_values = []

        for snp_idx in range(genotypes.shape[1]):
            snp_genotypes = genotypes[:, snp_idx]

            # Count alleles (ignoring missing data coded as NaN)
            valid_mask = ~np.isnan(snp_genotypes)
            valid_genotypes = snp_genotypes[valid_mask]

            if len(valid_genotypes) == 0:
                maf_values.append(0.0)
                continue

            # Count A1 alleles: 0 → 0, 1 → 1, 2 → 2
            a1_count = np.sum(valid_genotypes)
            total_alleles = 2 * len(valid_genotypes)

            # Calculate frequencies
            a1_freq = a1_count / total_alleles if total_alleles > 0 else 0.0
            a2_freq = 1.0 - a1_freq

            # MAF is the minimum
            maf = min(a1_freq, a2_freq)
            maf_values.append(maf)

        maf_array = np.array(maf_values)

        # Filter SNPs by MAF
        keep_mask = maf_array >= maf_min
        filtered_genotypes = genotypes[:, keep_mask]

        # Update variant indices
        if variant_idx is not None:
            updated_variant_idx = variant_idx[keep_mask]
        else:
            updated_variant_idx = np.where(keep_mask)[0]

        return filtered_genotypes, updated_variant_idx

    def _create_anndata_from_plink(
        self,
        genotypes: np.ndarray,
        fam_data: pd.DataFrame,
        bim_data: pd.DataFrame,
        source_path: Path,
    ) -> anndata.AnnData:
        """
        Create AnnData object from PLINK data components.

        Args:
            genotypes: Genotype matrix (samples × SNPs), float32 with NaN for missing
            fam_data: DataFrame from .fam file
            bim_data: DataFrame from .bim file
            source_path: Path to source .bed file

        Returns:
            anndata.AnnData: Created AnnData object
        """
        # PLINK .fam columns (bed-reader): fid, iid, father, mother, sex, pheno
        # PLINK .bim columns (bed-reader): chrom, snp, cm, pos, a0, a1

        # Create obs (samples) metadata
        obs = pd.DataFrame(
            {
                "individual_id": fam_data.iloc[:, 1].astype(str),  # IID
                "family_id": fam_data.iloc[:, 0].astype(str),  # FID
                "father_id": fam_data.iloc[:, 2].astype(str),  # Father
                "mother_id": fam_data.iloc[:, 3].astype(str),  # Mother
                "sex": fam_data.iloc[:, 4],  # Sex (1=male, 2=female, 0=unknown)
                "phenotype": fam_data.iloc[:, 5],  # Phenotype
            },
            index=fam_data.iloc[:, 1].astype(str),  # Use IID as index
        )

        # Create var (SNPs) metadata
        var = pd.DataFrame(
            {
                "snp_id": bim_data.iloc[:, 1].astype(str),  # SNP ID
                "chromosome": bim_data.iloc[:, 0].astype(str),  # Chromosome
                "cm_position": bim_data.iloc[:, 2],  # Genetic distance (cM)
                "bp_position": bim_data.iloc[:, 3],  # Physical position (bp)
                "allele_1": bim_data.iloc[:, 4].astype(str),  # Allele 1 (A1)
                "allele_2": bim_data.iloc[:, 5].astype(str),  # Allele 2 (A2)
            },
            index=bim_data.iloc[:, 1].astype(str),  # Use SNP ID as index
        )

        # Ensure unique indices
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)

        # Make indices unique if there are duplicates
        if obs.index.duplicated().any():
            logger.warning("Duplicate individual IDs found, making unique")
            obs.index = pd.Index([f"{iid}_{i}" for i, iid in enumerate(obs.index)])

        if var.index.duplicated().any():
            logger.warning("Duplicate SNP IDs found, making unique")
            var.index = pd.Index([f"{sid}_{i}" for i, sid in enumerate(var.index)])

        # Create AnnData object
        # X matrix: genotypes as float32 (NaN for missing)
        X = genotypes.astype(np.float32)

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Add genotypes to layers
        adata.layers["GT"] = X.copy()

        # Add metadata to uns
        adata.uns["source_file"] = str(source_path)
        adata.uns["data_type"] = "genomics"
        adata.uns["modality"] = "snp_array"

        return adata

    def validate(self, adata: anndata.AnnData, strict: bool = None) -> ValidationResult:
        """
        Validate AnnData against genomics SNP array schema.

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

        # Add PLINK-specific validation
        plink_result = self._validate_plink_specific(adata)
        result = result.merge(plink_result)

        return result

    def _validate_plink_specific(self, adata: anndata.AnnData) -> ValidationResult:
        """
        Perform PLINK-specific validation.

        Args:
            adata: AnnData object to validate

        Returns:
            ValidationResult: Validation results
        """
        result = ValidationResult()

        # Check for duplicate SNP IDs
        if "snp_id" in adata.var.columns:
            snp_ids = adata.var["snp_id"]
            duplicates = snp_ids.duplicated().sum()
            if duplicates > 0:
                result.add_warning(
                    f"Found {duplicates} duplicate SNP IDs. Consider filtering or renaming."
                )

        # Validate sex encoding (PLINK convention: 1=male, 2=female, 0=unknown)
        if "sex" in adata.obs.columns:
            sex_values = set(adata.obs["sex"].unique())
            valid_sex = {0, 1, 2}
            # Allow string representations
            try:
                sex_values_numeric = {int(v) for v in sex_values if pd.notna(v)}
                invalid_sex = sex_values_numeric - valid_sex
                if invalid_sex:
                    result.add_error(
                        f"Invalid sex values: {invalid_sex}. "
                        f"PLINK convention: 1=male, 2=female, 0=unknown"
                    )
            except (ValueError, TypeError):
                result.add_warning(
                    "Could not validate sex encoding (non-numeric values found)"
                )

        # Validate phenotype encoding (PLINK convention: -9=missing, 0=missing, 1=control, 2=case)
        if "phenotype" in adata.obs.columns:
            pheno_values = set(adata.obs["phenotype"].unique())
            valid_pheno = {-9, 0, 1, 2}
            try:
                pheno_values_numeric = {int(v) for v in pheno_values if pd.notna(v)}
                unusual_pheno = pheno_values_numeric - valid_pheno
                if unusual_pheno:
                    result.add_info(
                        f"Non-standard phenotype values found: {unusual_pheno}. "
                        f"PLINK convention: -9/0=missing, 1=control, 2=case (but other values are valid for quantitative traits)"
                    )
            except (ValueError, TypeError):
                result.add_info(
                    "Phenotype values are non-numeric (quantitative trait or custom encoding)"
                )

        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for PLINK/SNP array data.

        Returns:
            Dict[str, Any]: Schema definition
        """
        return GenomicsSchema.get_snp_array_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ["bed"]  # PLINK binary format

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply PLINK-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing
        adata = super().preprocess_data(adata, **kwargs)

        # Calculate MAF if not present
        if "maf" not in adata.var.columns:
            adata = self._calculate_maf(adata)

        # Calculate call rates
        adata = self._calculate_call_rates(adata)

        return adata

    def _calculate_maf(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Calculate minor allele frequency for each SNP.

        Args:
            adata: AnnData object

        Returns:
            anndata.AnnData: AnnData with MAF calculated
        """
        # Get genotypes from X or GT layer
        if "GT" in adata.layers:
            genotypes = adata.layers["GT"]
        else:
            genotypes = adata.X

        maf_values = []

        for snp_idx in range(adata.n_vars):
            snp_genotypes = genotypes[:, snp_idx]

            # Handle both dense and sparse matrices
            if hasattr(snp_genotypes, "toarray"):
                snp_genotypes = snp_genotypes.toarray().flatten()
            else:
                snp_genotypes = np.array(snp_genotypes).flatten()

            # Count alleles (ignoring missing data)
            valid_mask = ~np.isnan(snp_genotypes)
            valid_genotypes = snp_genotypes[valid_mask]

            if len(valid_genotypes) == 0:
                maf_values.append(np.nan)
                continue

            # Count A1 alleles
            a1_count = np.sum(valid_genotypes)
            total_alleles = 2 * len(valid_genotypes)

            # Calculate frequencies
            a1_freq = a1_count / total_alleles if total_alleles > 0 else 0.0
            a2_freq = 1.0 - a1_freq

            # MAF is the minimum
            maf = min(a1_freq, a2_freq)
            maf_values.append(maf)

        adata.var["maf"] = maf_values
        logger.info(f"Calculated MAF for {adata.n_vars} SNPs")

        return adata

    def _calculate_call_rates(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Calculate call rates for individuals and SNPs.

        Args:
            adata: AnnData object

        Returns:
            anndata.AnnData: AnnData with call rates calculated
        """
        # Get genotypes
        if "GT" in adata.layers:
            genotypes = adata.layers["GT"]
        else:
            genotypes = adata.X

        # Calculate individual (obs) call rates
        if "call_rate" not in adata.obs.columns:
            individual_call_rates = []
            for ind_idx in range(adata.n_obs):
                ind_genotypes = genotypes[ind_idx, :]
                if hasattr(ind_genotypes, "toarray"):
                    ind_genotypes = ind_genotypes.toarray().flatten()
                else:
                    ind_genotypes = np.array(ind_genotypes).flatten()

                non_missing = np.sum(~np.isnan(ind_genotypes))
                total = len(ind_genotypes)
                call_rate = non_missing / total if total > 0 else 0.0
                individual_call_rates.append(call_rate)

            adata.obs["call_rate"] = individual_call_rates
            logger.info(f"Calculated call rates for {adata.n_obs} individuals")

        # Calculate SNP (var) call rates
        if "call_rate" not in adata.var.columns:
            snp_call_rates = []
            for snp_idx in range(adata.n_vars):
                snp_genotypes = genotypes[:, snp_idx]
                if hasattr(snp_genotypes, "toarray"):
                    snp_genotypes = snp_genotypes.toarray().flatten()
                else:
                    snp_genotypes = np.array(snp_genotypes).flatten()

                non_missing = np.sum(~np.isnan(snp_genotypes))
                total = len(snp_genotypes)
                call_rate = non_missing / total if total > 0 else 0.0
                snp_call_rates.append(call_rate)

            adata.var["call_rate"] = snp_call_rates
            logger.info(f"Calculated call rates for {adata.n_vars} SNPs")

        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate PLINK-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)

        # Add PLINK-specific metrics
        if "maf" in adata.var.columns:
            maf_values = adata.var["maf"].dropna()
            if len(maf_values) > 0:
                metrics["mean_maf"] = float(maf_values.mean())
                metrics["median_maf"] = float(maf_values.median())
                metrics["common_snps"] = int((maf_values >= 0.05).sum())
                metrics["rare_snps"] = int((maf_values < 0.01).sum())

        if "call_rate" in adata.obs.columns:
            obs_call_rates = adata.obs["call_rate"].dropna()
            if len(obs_call_rates) > 0:
                metrics["mean_individual_call_rate"] = float(obs_call_rates.mean())
                metrics["low_call_rate_individuals"] = int(
                    (obs_call_rates < 0.95).sum()
                )

        if "call_rate" in adata.var.columns:
            var_call_rates = adata.var["call_rate"].dropna()
            if len(var_call_rates) > 0:
                metrics["mean_snp_call_rate"] = float(var_call_rates.mean())
                metrics["low_call_rate_snps"] = int((var_call_rates < 0.98).sum())

        # Calculate missing rate
        if "GT" in adata.layers:
            gt = adata.layers["GT"]
        else:
            gt = adata.X

        missing_count = np.sum(np.isnan(gt))
        total_count = gt.size
        metrics["missing_rate"] = (
            float(missing_count / total_count) if total_count > 0 else 0.0
        )

        return metrics
