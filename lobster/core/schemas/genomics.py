"""
Genomics schema definitions for WGS and SNP array data.

This module defines the expected structure and metadata for genomics
datasets including whole genome sequencing (WGS) VCF data and SNP array
PLINK data with appropriate validation rules.
"""

from typing import Any, Dict, List, Optional

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.validation import FlexibleValidator


class GenomicsSchema:
    """
    Schema definitions for genomics data modalities.

    This class provides schema definitions for both WGS (VCF) and SNP array
    (PLINK) data with appropriate metadata requirements and validation rules.
    """

    @staticmethod
    def get_wgs_schema() -> Dict[str, Any]:
        """
        Get schema for whole genome sequencing (WGS) VCF data.

        Returns:
            Dict[str, Any]: WGS VCF schema definition
        """
        return {
            "modality": "wgs",
            "description": "Whole genome sequencing VCF data schema",
            # obs: Observations (samples) metadata - DataFrame with samples as rows
            # Contains per-sample metadata including population info, sequencing metrics,
            # and variant statistics
            #
            # Example obs DataFrame:
            #            sample_id population ethnicity disease_status  sequencing_depth  variant_count  ti_tv_ratio  het_hom_ratio
            # Sample_1    Sample_1  EUR        Caucasian        healthy              30.5          45678         2.08           1.65
            # Sample_2    Sample_2  AFR        African           cancer              28.3          52341         2.11           1.58
            # Sample_3    Sample_3  EAS        East_Asian        healthy              32.1          43210         2.06           1.72
            "obs": {
                "required": [],  # Flexible to accommodate diverse VCF datasets
                "optional": [
                    "sample_id",  # Unique sample identifier
                    "population",  # Population group (e.g., EUR, AFR, EAS)
                    "ethnicity",  # Detailed ethnicity information
                    "disease_status",  # Disease/phenotype status
                    "sequencing_depth",  # Mean sequencing coverage depth
                    "variant_count",  # Total number of variants per sample
                    "ti_tv_ratio",  # Transition/transversion ratio (quality metric)
                    "het_hom_ratio",  # Heterozygous/homozygous ratio
                    "age",  # Subject age
                    "sex",  # Subject sex
                    "batch",  # Sequencing batch
                    "sequencing_platform",  # Platform (e.g., Illumina NovaSeq)
                    "call_rate",  # Variant call rate (0-1)
                    "inbreeding_coefficient",  # F coefficient
                ],
                "types": {
                    "sample_id": "string",
                    "population": "categorical",
                    "ethnicity": "categorical",
                    "disease_status": "categorical",
                    "sequencing_depth": "numeric",
                    "variant_count": "numeric",
                    "ti_tv_ratio": "numeric",
                    "het_hom_ratio": "numeric",
                    "age": "numeric",
                    "sex": "categorical",
                    "batch": "string",
                    "sequencing_platform": "string",
                    "call_rate": "numeric",
                    "inbreeding_coefficient": "numeric",
                },
            },
            # var: Variables (variants) metadata - DataFrame with variants as rows
            # Contains per-variant metadata including genomic coordinates, alleles,
            # quality metrics, and functional annotations
            #
            # Example var DataFrame:
            #                 CHROM       POS  REF ALT           ID   QUAL FILTER          INFO     AF gene_symbol     consequence
            # var_001            1    123456    A   G    rs123456   45.2   PASS  AC=2;AN=100  0.020         BRCA2        missense
            # var_002            X   9876543    C   T    rs987654   89.7   PASS  AC=5;AN=100  0.050          TP53         frameshift
            # var_003           17    654321    G   A          .   12.3   LowQ  AC=1;AN=100  0.010          TTN      synonymous
            "var": {
                "required": [
                    "CHROM",  # Chromosome (1-22, X, Y, MT)
                    "POS",  # Genomic position (1-based)
                    "REF",  # Reference allele
                    "ALT",  # Alternate allele
                ],
                "optional": [
                    "ID",  # Variant ID (e.g., rs number)
                    "QUAL",  # Quality score
                    "FILTER",  # Filter status (PASS, LowQual, etc.)
                    "INFO",  # INFO field from VCF
                    "AF",  # Allele frequency
                    "gene_symbol",  # Gene symbol (if variant in gene)
                    "consequence",  # Functional consequence
                    "dbSNP_id",  # dbSNP identifier
                    "gnomad_af",  # gnomAD allele frequency
                    "cadd_score",  # CADD pathogenicity score
                    "polyphen_score",  # PolyPhen score
                    "sift_score",  # SIFT score
                    "clinical_significance",  # ClinVar classification
                ],
                "types": {
                    "CHROM": "categorical",
                    "POS": "numeric",
                    "REF": "string",
                    "ALT": "string",
                    "ID": "string",
                    "QUAL": "numeric",
                    "FILTER": "categorical",
                    "INFO": "string",
                    "AF": "numeric",
                    "gene_symbol": "string",
                    "consequence": "categorical",
                    "dbSNP_id": "string",
                    "gnomad_af": "numeric",
                    "cadd_score": "numeric",
                    "polyphen_score": "numeric",
                    "sift_score": "numeric",
                    "clinical_significance": "categorical",
                },
            },
            # layers: Genotype matrices with same dimensions as X
            # Store different representations of genotype data (raw calls, dosages, quality)
            # Each layer is a 2D matrix: samples x variants, same shape as adata.X
            #
            # Example layers (3 samples x 4 variants):
            #
            # layers['GT'] (genotypes: 0=hom_ref, 1=het, 2=hom_alt, -1=missing):
            #           var_001  var_002  var_003  var_004
            # Sample_1        0        1        2       -1
            # Sample_2        1        0        1        2
            # Sample_3        2        2        0        0
            #
            # layers['DP'] (read depth):
            #           var_001  var_002  var_003  var_004
            # Sample_1       32       28       15        0
            # Sample_2       29       31       27       18
            # Sample_3       35       30       33       29
            #
            # layers['GQ'] (genotype quality):
            #           var_001  var_002  var_003  var_004
            # Sample_1       99       85       42        0
            # Sample_2       92       99       88       65
            # Sample_3       99       95       99       99
            "layers": {
                "required": [
                    "GT",  # Genotype calls (required)
                ],
                "optional": [
                    "DP",  # Read depth per variant per sample
                    "GQ",  # Genotype quality score
                    "AD",  # Allelic depth (ref, alt counts)
                    "PL",  # Phred-scaled likelihoods
                ],
            },
            # obsm: Not typically used for genomics data
            "obsm": {
                "required": [],
                "optional": [
                    "X_pca",  # PCA coordinates (if computed)
                    "X_umap",  # UMAP embedding (if computed)
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores dataset-level information, VCF metadata, and quality control results
            #
            # Example uns structure:
            # uns = {
            #     'vcf_metadata': {
            #         'fileformat': 'VCFv4.2',
            #         'source': 'GATK',
            #         'reference': 'GRCh38',
            #         'contig_info': {...}
            #     },
            #     'variant_calling': {
            #         'caller': 'GATK HaplotypeCaller',
            #         'version': '4.2.0.0',
            #         'filters_applied': ['DP < 10', 'GQ < 20']
            #     },
            #     'population_structure': {
            #         'pca_variance_explained': [0.15, 0.08, 0.06],
            #         'n_components': 10
            #     }
            # }
            "uns": {
                "required": [],
                "optional": [
                    "vcf_metadata",  # VCF header metadata
                    "variant_calling",  # Variant calling parameters
                    "population_structure",  # Population genetics results
                    "quality_control",  # QC metrics and filtering
                    "reference_genome",  # Reference genome version
                    "provenance",  # Provenance tracking
                    # Cross-database accessions
                    "bioproject_accession",  # NCBI BioProject
                    "biosample_accession",  # NCBI BioSample
                    "sra_study_accession",  # SRA Study
                    "publication_doi",  # Publication DOI
                ],
            },
        }

    @staticmethod
    def get_snp_array_schema() -> Dict[str, Any]:
        """
        Get schema for SNP array PLINK data.

        Returns:
            Dict[str, Any]: SNP array PLINK schema definition
        """
        return {
            "modality": "snp_array",
            "description": "SNP array PLINK data schema",
            # obs: Observations (samples/individuals) metadata - DataFrame with individuals as rows
            # Contains per-individual metadata from PLINK .fam file and phenotype data
            #
            # Example obs DataFrame:
            #                individual_id family_id father_id mother_id  sex phenotype population    age
            # Individual_1    Individual_1    Fam_01       0         0    M   control        EUR   45.2
            # Individual_2    Individual_2    Fam_01       0         0    F      case        EUR   52.8
            # Individual_3    Individual_3    Fam_02       0         0    M   control        AFR   38.5
            "obs": {
                "required": [],  # Flexible for different PLINK datasets
                "optional": [
                    "individual_id",  # Individual identifier (IID in PLINK)
                    "family_id",  # Family identifier (FID in PLINK)
                    "father_id",  # Paternal ID
                    "mother_id",  # Maternal ID
                    "sex",  # Sex (1=male, 2=female, 0=unknown)
                    "phenotype",  # Phenotype value
                    "population",  # Population group
                    "age",  # Subject age
                    "batch",  # Genotyping batch
                    "call_rate",  # Individual call rate
                    "heterozygosity",  # Heterozygosity rate
                ],
                "types": {
                    "individual_id": "string",
                    "family_id": "string",
                    "father_id": "string",
                    "mother_id": "string",
                    "sex": "categorical",
                    "phenotype": "categorical",
                    "population": "categorical",
                    "age": "numeric",
                    "batch": "string",
                    "call_rate": "numeric",
                    "heterozygosity": "numeric",
                },
            },
            # var: Variables (SNPs) metadata - DataFrame with SNPs as rows
            # Contains per-SNP metadata from PLINK .bim file and quality metrics
            #
            # Example var DataFrame:
            #                chromosome    snp_id  cm_position  bp_position allele_1 allele_2      maf  call_rate  hwe_pvalue
            # SNP_001                 1   rs12345          0.5      1234567        A        G    0.123       0.98     0.45
            # SNP_002                 2   rs67890          1.2      7654321        C        T    0.287       0.99     0.78
            # SNP_003                 X   rs11111          2.3      9999999        G        A    0.056       0.95     0.12
            "var": {
                "required": [],  # Flexible for different annotation levels
                "optional": [
                    "chromosome",  # Chromosome
                    "snp_id",  # SNP identifier (rsID)
                    "cm_position",  # Genetic distance (centiMorgans)
                    "bp_position",  # Physical position (base pairs)
                    "allele_1",  # First allele (usually minor)
                    "allele_2",  # Second allele (usually major)
                    "maf",  # Minor allele frequency
                    "call_rate",  # SNP call rate
                    "hwe_pvalue",  # Hardy-Weinberg equilibrium p-value
                    "info_score",  # Imputation quality (if imputed)
                    "gene_symbol",  # Nearest gene
                    "consequence",  # Functional consequence
                ],
                "types": {
                    "chromosome": "categorical",
                    "snp_id": "string",
                    "cm_position": "numeric",
                    "bp_position": "numeric",
                    "allele_1": "string",
                    "allele_2": "string",
                    "maf": "numeric",
                    "call_rate": "numeric",
                    "hwe_pvalue": "numeric",
                    "info_score": "numeric",
                    "gene_symbol": "string",
                    "consequence": "categorical",
                },
            },
            # layers: Genotype matrices
            # Store genotype calls and quality information
            #
            # Example layers (3 individuals x 4 SNPs):
            #
            # layers['GT'] (genotypes: 0/1/2 or -1 for missing):
            #                SNP_001  SNP_002  SNP_003  SNP_004
            # Individual_1         0        1        2       -1
            # Individual_2         1        0        1        2
            # Individual_3         2        2        0        0
            "layers": {
                "required": [
                    "GT",  # Genotype calls (required)
                ],
                "optional": [],
            },
            # obsm: Not typically used for SNP array data
            "obsm": {
                "required": [],
                "optional": [
                    "X_pca",  # PCA coordinates (if computed)
                    "X_umap",  # UMAP embedding (if computed)
                    "X_mds",  # MDS coordinates (if computed)
                ],
            },
            # uns: Unstructured annotations - global metadata
            # Stores dataset-level information, array chip info, and QC results
            #
            # Example uns structure:
            # uns = {
            #     'array_metadata': {
            #         'chip_type': 'Illumina Global Screening Array',
            #         'n_snps': 650000,
            #         'genome_build': 'GRCh38'
            #     },
            #     'quality_control': {
            #         'min_call_rate': 0.95,
            #         'min_maf': 0.01,
            #         'hwe_threshold': 1e-6
            #     }
            # }
            "uns": {
                "required": [],
                "optional": [
                    "array_metadata",  # Array chip metadata
                    "quality_control",  # QC parameters and results
                    "reference_genome",  # Reference genome build
                    "provenance",  # Provenance tracking
                    # Cross-database accessions
                    "bioproject_accession",  # NCBI BioProject
                    "publication_doi",  # Publication DOI
                ],
            },
        }

    @staticmethod
    def create_validator(
        schema_type: str = "wgs",
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None,
    ) -> FlexibleValidator:
        """
        Create a validator for genomics data.

        Args:
            schema_type: Type of schema ('wgs' or 'snp_array')
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator

        Raises:
            ValueError: If schema_type is not recognized
        """
        if schema_type == "wgs":
            schema = GenomicsSchema.get_wgs_schema()
        elif schema_type == "snp_array":
            schema = GenomicsSchema.get_snp_array_schema()
        else:
            raise ValueError(
                f"Unknown schema type: {schema_type}. Must be 'wgs' or 'snp_array'"
            )

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for genomics
        ignore_set.update(
            [
                "Unexpected obs columns",
                "Unexpected var columns",
                "missing values",
                "Very sparse data",
            ]
        )

        validator = FlexibleValidator(
            schema=schema,
            name=f"GenomicsValidator_{schema_type}",
            ignore_warnings=ignore_set,
        )

        # Add genomics-specific validation rules
        validator.add_custom_rule(
            "check_chromosome_format", _validate_chromosome_format
        )
        validator.add_custom_rule("check_genotype_data", _validate_genotype_data)
        validator.add_custom_rule(
            "check_variant_positions", _validate_variant_positions
        )

        if schema_type == "wgs":
            validator.add_custom_rule("check_vcf_fields", _validate_vcf_fields)
            validator.add_custom_rule(
                "check_allele_frequency", _validate_allele_frequency
            )
        elif schema_type == "snp_array":
            validator.add_custom_rule("check_plink_fields", _validate_plink_fields)
            validator.add_custom_rule(
                "check_maf_distribution", _validate_maf_distribution
            )

        return validator

    @staticmethod
    def get_recommended_qc_thresholds(
        schema_type: str = "wgs",
    ) -> Dict[str, Any]:
        """
        Get recommended quality control thresholds for genomics data.

        Args:
            schema_type: Type of schema ('wgs' or 'snp_array')

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        if schema_type == "wgs":
            return {
                "min_sequencing_depth": 10.0,  # Minimum mean coverage
                "max_sequencing_depth": 100.0,  # Maximum mean coverage (potential duplication)
                "min_variant_quality": 20.0,  # Minimum QUAL score
                "min_genotype_quality": 20.0,  # Minimum GQ score
                "min_call_rate": 0.95,  # Minimum variant call rate
                "ti_tv_ratio_range": [2.0, 2.2],  # Expected Ti/Tv ratio for WGS
                "het_hom_ratio_range": [1.3, 2.0],  # Expected Het/Hom ratio
                "max_missing_rate": 0.05,  # Maximum missing genotype rate
            }
        elif schema_type == "snp_array":
            return {
                "min_individual_call_rate": 0.95,  # Minimum call rate per individual
                "min_snp_call_rate": 0.98,  # Minimum call rate per SNP
                "min_maf": 0.01,  # Minimum minor allele frequency
                "max_maf": 0.49,  # Maximum minor allele frequency (sanity check)
                "hwe_pvalue_threshold": 1e-6,  # Hardy-Weinberg equilibrium threshold
                "max_heterozygosity_deviation": 3.0,  # SD from mean heterozygosity
                "max_missing_rate": 0.05,  # Maximum missing genotype rate
            }
        else:
            raise ValueError(
                f"Unknown schema type: {schema_type}. Must be 'wgs' or 'snp_array'"
            )


def _validate_chromosome_format(adata) -> "ValidationResult":
    """Validate chromosome format and values."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for CHROM in var
    if "CHROM" in adata.var.columns:
        chroms = adata.var["CHROM"]

        # Valid chromosome values
        valid_chroms = set(
            [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M", "chr1"]
        )

        # Check for invalid chromosomes
        unique_chroms = set(chroms.unique())
        invalid_chroms = unique_chroms - valid_chroms

        if invalid_chroms:
            result.add_warning(
                f"Found non-standard chromosome names: {sorted(list(invalid_chroms))[:10]}"
            )

    elif "chromosome" in adata.var.columns:
        chroms = adata.var["chromosome"]

        # Check chromosome format
        unique_chroms = set(chroms.unique())
        result.add_info(
            f"Found {len(unique_chroms)} unique chromosomes: {sorted(list(unique_chroms))[:10]}"
        )

    return result


def _validate_genotype_data(adata) -> "ValidationResult":
    """Validate genotype matrix characteristics."""

    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for GT layer
    if "GT" not in adata.layers:
        result.add_error("Required 'GT' (genotype) layer not found")
        return result

    gt = adata.layers["GT"]

    # Check genotype values (should be 0, 1, 2, or -1 for missing)
    if hasattr(gt, "min") and hasattr(gt, "max"):
        min_val = gt.min()
        max_val = gt.max()

        # Valid genotype values: -1 (missing), 0 (hom_ref), 1 (het), 2 (hom_alt)
        if min_val < -1:
            result.add_error(f"Invalid genotype values found (min: {min_val})")

        if max_val > 2:
            result.add_warning(
                f"Genotype values > 2 found (max: {max_val}), possible multi-allelic sites"
            )

        # Check missing data rate
        if hasattr(gt, "data"):  # Sparse matrix
            missing_count = (gt.data == -1).sum()
            total_count = gt.data.size
        else:  # Dense matrix
            missing_count = (gt == -1).sum()
            total_count = gt.size

        if total_count > 0:
            missing_rate = missing_count / total_count
            if missing_rate > 0.1:
                result.add_warning(f"High missing genotype rate: {missing_rate:.2%}")
            else:
                result.add_info(f"Missing genotype rate: {missing_rate:.2%}")

    return result


def _validate_variant_positions(adata) -> "ValidationResult":
    """Validate variant position data."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for POS in var
    if "POS" in adata.var.columns:
        positions = adata.var["POS"]

        # Check for non-positive positions
        if hasattr(positions, "min"):
            min_pos = positions.min()
            if min_pos <= 0:
                result.add_error(
                    f"Invalid variant positions found (min: {min_pos}). Positions must be >= 1"
                )

        # Check for duplicates (same chr:pos)
        if "CHROM" in adata.var.columns:
            composite_ids = (
                adata.var["CHROM"].astype(str) + ":" + adata.var["POS"].astype(str)
            )
            duplicates = composite_ids.duplicated().sum()
            if duplicates > 0:
                result.add_warning(
                    f"Found {duplicates} duplicate variant positions (same CHROM:POS)"
                )

    elif "bp_position" in adata.var.columns:
        positions = adata.var["bp_position"]

        # Check for non-positive positions
        if hasattr(positions, "min"):
            min_pos = positions.min()
            if min_pos < 0:
                result.add_error(f"Invalid bp_position values found (min: {min_pos})")

    return result


def _validate_vcf_fields(adata) -> "ValidationResult":
    """Validate VCF-specific fields."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for required VCF fields
    required_fields = ["CHROM", "POS", "REF", "ALT"]
    missing_fields = [f for f in required_fields if f not in adata.var.columns]

    if missing_fields:
        result.add_error(f"Missing required VCF fields: {missing_fields}")

    # Check REF/ALT alleles format
    if "REF" in adata.var.columns:
        ref_alleles = adata.var["REF"]
        # Check if REF alleles are valid (A, C, G, T, N)
        valid_bases = {"A", "C", "G", "T", "N"}
        if ref_alleles.dtype == "object":
            invalid_ref = sum(
                not all(base in valid_bases for base in str(allele))
                for allele in ref_alleles
                if str(allele) != "nan"
            )
            if invalid_ref > 0:
                result.add_warning(
                    f"{invalid_ref} variants with non-standard REF alleles"
                )

    if "ALT" in adata.var.columns:
        alt_alleles = adata.var["ALT"]
        # Check if ALT alleles are valid
        if alt_alleles.dtype == "object":
            invalid_alt = sum(
                not all(base in valid_bases for base in str(allele))
                for allele in alt_alleles
                if str(allele) != "nan"
            )
            if invalid_alt > 0:
                result.add_warning(
                    f"{invalid_alt} variants with non-standard ALT alleles"
                )

    return result


def _validate_allele_frequency(adata) -> "ValidationResult":
    """Validate allele frequency values."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for AF in var
    if "AF" in adata.var.columns:
        af = adata.var["AF"]

        # AF should be between 0 and 1
        if hasattr(af, "min") and hasattr(af, "max"):
            min_af = af.min()
            max_af = af.max()

            if min_af < 0 or max_af > 1:
                result.add_error(
                    f"Invalid allele frequency values (min: {min_af}, max: {max_af}). Must be in [0, 1]"
                )

            # Check for extreme frequencies
            very_rare = (af < 0.001).sum()
            if very_rare > 0:
                result.add_info(f"{very_rare} very rare variants (AF < 0.001)")

    return result


def _validate_plink_fields(adata) -> "ValidationResult":
    """Validate PLINK-specific fields."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for typical PLINK .fam fields in obs
    fam_fields = [
        "individual_id",
        "family_id",
        "father_id",
        "mother_id",
        "sex",
        "phenotype",
    ]
    present_fam_fields = [f for f in fam_fields if f in adata.obs.columns]

    if present_fam_fields:
        result.add_info(f"Found PLINK .fam fields: {present_fam_fields}")

    # Check for typical PLINK .bim fields in var
    bim_fields = [
        "chromosome",
        "snp_id",
        "cm_position",
        "bp_position",
        "allele_1",
        "allele_2",
    ]
    present_bim_fields = [f for f in bim_fields if f in adata.var.columns]

    if present_bim_fields:
        result.add_info(f"Found PLINK .bim fields: {present_bim_fields}")

    # Validate sex encoding if present (PLINK convention: 1=male, 2=female, 0=unknown)
    if "sex" in adata.obs.columns:
        sex_values = set(adata.obs["sex"].unique())
        valid_sex = {0, 1, 2, "0", "1", "2", "M", "F", "male", "female", "unknown"}
        invalid_sex = sex_values - valid_sex

        if invalid_sex:
            result.add_warning(
                f"Non-standard sex values found: {invalid_sex}. PLINK convention: 1=male, 2=female, 0=unknown"
            )

    return result


def _validate_maf_distribution(adata) -> "ValidationResult":
    """Validate minor allele frequency distribution."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for maf in var
    if "maf" in adata.var.columns:
        maf = adata.var["maf"]

        # MAF should be between 0 and 0.5
        if hasattr(maf, "min") and hasattr(maf, "max"):
            min_maf = maf.min()
            max_maf = maf.max()

            if min_maf < 0 or max_maf > 0.5:
                result.add_warning(
                    f"Unusual MAF values (min: {min_maf}, max: {max_maf}). MAF typically in [0, 0.5]"
                )

            # Report MAF distribution
            common_snps = (maf >= 0.05).sum()
            low_freq_snps = ((maf >= 0.01) & (maf < 0.05)).sum()
            rare_snps = (maf < 0.01).sum()

            result.add_info(
                f"MAF distribution: {common_snps} common (>5%), "
                f"{low_freq_snps} low-freq (1-5%), {rare_snps} rare (<1%)"
            )

    return result
