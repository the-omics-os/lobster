"""
Quality assessment service for genomics/DNA data.

This service provides methods for evaluating the quality of genomics data,
including genotype call rates, minor allele frequencies, Hardy-Weinberg
equilibrium, and heterozygosity metrics.

This service emits Intermediate Representation (IR) for automatic notebook export.
"""

from typing import Any, Dict, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import stats

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GenomicsQualityError(Exception):
    """Base exception for genomics quality assessment operations."""

    pass


class GenomicsQualityService:
    """
    Stateless service for assessing genomics/DNA data quality.

    This class provides methods to calculate quality metrics for genotype data,
    including per-sample and per-variant QC metrics based on UK Biobank
    Pan-Ancestry standards.

    QC Thresholds (UK Biobank):
        - Sample call rate: ≥0.95
        - Variant call rate: ≥0.99
        - MAF: ≥0.01 (common) or ≥0.001 (rare)
        - HWE p-value: ≥1e-10
        - Heterozygosity: within 3 SD of mean
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the genomics quality assessment service.

        Args:
            config: Optional configuration dict (ignored, for backward compatibility)
            **kwargs: Additional arguments (ignored, for backward compatibility)

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless GenomicsQualityService")
        self.config = config or {}
        logger.debug("GenomicsQualityService initialized successfully")

    def assess_quality(
        self,
        adata: anndata.AnnData,
        min_call_rate: float = 0.95,
        min_maf: float = 0.01,
        hwe_pvalue: float = 1e-10,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform quality assessment on genomics data.

        Calculates per-sample and per-variant QC metrics including:
        - Sample metrics: call_rate, heterozygosity, het_z_score
        - Variant metrics: call_rate, maf, hwe_p, qc_pass flag

        Args:
            adata: AnnData object with genotypes in layers['GT']
            min_call_rate: Minimum call rate threshold (0.95 = 95%)
            min_maf: Minimum minor allele frequency (0.01 = 1%)
            hwe_pvalue: Minimum Hardy-Weinberg equilibrium p-value (1e-10)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with QC metrics added to .obs and .var
                - Assessment statistics dictionary
                - AnalysisStep IR for notebook export

        Raises:
            GenomicsQualityError: If quality assessment fails
        """
        try:
            logger.debug("Starting genomics quality assessment")

            # Validate GT matrix exists
            if "GT" not in adata.layers:
                raise GenomicsQualityError(
                    "No genotype data found. Expected 'GT' layer in adata.layers"
                )

            # Create working copy
            adata_qc = adata.copy()

            # Get genotype matrix (samples × variants)
            gt = adata_qc.layers["GT"]

            # Calculate per-sample metrics
            logger.debug("Calculating per-sample QC metrics")
            sample_call_rate = self._calculate_sample_call_rate(gt)
            heterozygosity = self._calculate_heterozygosity(gt)
            het_z_score = self._calculate_het_z_score(heterozygosity)

            # Add to adata.obs (per-sample)
            adata_qc.obs["call_rate"] = sample_call_rate
            adata_qc.obs["heterozygosity"] = heterozygosity
            adata_qc.obs["het_z_score"] = het_z_score

            # Calculate per-variant metrics
            logger.debug("Calculating per-variant QC metrics")
            variant_call_rate = self._calculate_variant_call_rate(gt)
            maf = self._calculate_maf(gt)
            hwe_p = self._calculate_hwe(gt)

            # Add to adata.var (per-variant)
            adata_qc.var["call_rate"] = variant_call_rate
            adata_qc.var["maf"] = maf
            adata_qc.var["hwe_p"] = hwe_p

            # QC pass/fail flag for variants
            qc_pass_variant = (
                (variant_call_rate >= min_call_rate)
                & (maf >= min_maf)
                & (hwe_p >= hwe_pvalue)
            )
            adata_qc.var["qc_pass"] = qc_pass_variant

            # Compile assessment statistics
            n_samples = adata_qc.n_obs
            n_variants = adata_qc.n_vars
            n_variants_pass = qc_pass_variant.sum()

            assessment_stats = {
                "analysis_type": "genomics_quality_assessment",
                "min_call_rate": min_call_rate,
                "min_maf": min_maf,
                "hwe_pvalue": hwe_pvalue,
                "n_samples": n_samples,
                "n_variants": n_variants,
                "n_variants_pass_qc": int(n_variants_pass),
                "variants_fail_qc": int(n_variants - n_variants_pass),
                "variants_pass_pct": float((n_variants_pass / n_variants) * 100),
                "sample_metrics": {
                    "mean_call_rate": float(sample_call_rate.mean()),
                    "median_call_rate": float(np.median(sample_call_rate)),
                    "mean_heterozygosity": float(heterozygosity.mean()),
                    "median_heterozygosity": float(np.median(heterozygosity)),
                },
                "variant_metrics": {
                    "mean_call_rate": float(variant_call_rate.mean()),
                    "median_call_rate": float(np.median(variant_call_rate)),
                    "mean_maf": float(maf.mean()),
                    "median_maf": float(np.median(maf)),
                    "n_variants_low_call_rate": int(
                        (variant_call_rate < min_call_rate).sum()
                    ),
                    "n_variants_low_maf": int((maf < min_maf).sum()),
                    "n_variants_hwe_fail": int((hwe_p < hwe_pvalue).sum()),
                },
            }

            logger.info(
                f"Quality assessment completed: {n_variants_pass}/{n_variants} variants pass QC "
                f"({assessment_stats['variants_pass_pct']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_assess_quality_ir(
                min_call_rate=min_call_rate,
                min_maf=min_maf,
                hwe_pvalue=hwe_pvalue,
            )

            return adata_qc, assessment_stats, ir

        except Exception as e:
            logger.exception(f"Error in genomics quality assessment: {e}")
            raise GenomicsQualityError(f"Quality assessment failed: {str(e)}")

    def filter_samples(
        self,
        adata: anndata.AnnData,
        min_call_rate: float = 0.95,
        het_sd_threshold: float = 3.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter samples based on QC metrics.

        Removes samples with:
        - Low call rate (< min_call_rate)
        - Extreme heterozygosity (|het_z_score| > het_sd_threshold)

        Args:
            adata: AnnData object with QC metrics in .obs
            min_call_rate: Minimum call rate threshold (default: 0.95)
            het_sd_threshold: Heterozygosity z-score threshold (default: 3.0 SD)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - Filtered AnnData object
                - Filtering statistics dictionary
                - AnalysisStep IR for notebook export

        Raises:
            GenomicsQualityError: If filtering fails or QC metrics missing
        """
        try:
            logger.debug("Starting sample filtering")

            # Validate QC metrics exist
            required_cols = ["call_rate", "het_z_score"]
            missing = [col for col in required_cols if col not in adata.obs.columns]
            if missing:
                raise GenomicsQualityError(
                    f"Missing QC metrics: {missing}. Run assess_quality() first."
                )

            n_samples_before = adata.n_obs

            # Identify passing samples
            passing_samples = (adata.obs["call_rate"] >= min_call_rate) & (
                np.abs(adata.obs["het_z_score"]) <= het_sd_threshold
            )

            # Filter AnnData
            adata_filtered = adata[passing_samples, :].copy()
            n_samples_after = adata_filtered.n_obs
            n_samples_removed = n_samples_before - n_samples_after

            # Compile statistics
            stats = {
                "analysis_type": "sample_filtering",
                "min_call_rate": min_call_rate,
                "het_sd_threshold": het_sd_threshold,
                "samples_before": n_samples_before,
                "samples_after": n_samples_after,
                "samples_removed": n_samples_removed,
                "samples_retained_pct": float((n_samples_after / n_samples_before) * 100),
                "removal_reasons": {
                    "low_call_rate": int(
                        (adata.obs["call_rate"] < min_call_rate).sum()
                    ),
                    "extreme_heterozygosity": int(
                        (np.abs(adata.obs["het_z_score"]) > het_sd_threshold).sum()
                    ),
                },
            }

            logger.info(
                f"Sample filtering completed: {n_samples_after}/{n_samples_before} samples retained "
                f"({stats['samples_retained_pct']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_filter_samples_ir(
                min_call_rate=min_call_rate,
                het_sd_threshold=het_sd_threshold,
            )

            return adata_filtered, stats, ir

        except Exception as e:
            logger.exception(f"Error in sample filtering: {e}")
            raise GenomicsQualityError(f"Sample filtering failed: {str(e)}")

    def filter_variants(
        self,
        adata: anndata.AnnData,
        min_call_rate: float = 0.99,
        min_maf: float = 0.01,
        min_hwe_p: float = 1e-10,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter variants based on QC metrics.

        Removes variants with:
        - Low call rate (< min_call_rate)
        - Low minor allele frequency (< min_maf)
        - Hardy-Weinberg disequilibrium (hwe_p < min_hwe_p)

        Args:
            adata: AnnData object with QC metrics in .var
            min_call_rate: Minimum call rate threshold (default: 0.99)
            min_maf: Minimum minor allele frequency (default: 0.01)
            min_hwe_p: Minimum Hardy-Weinberg p-value (default: 1e-10)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - Filtered AnnData object
                - Filtering statistics dictionary
                - AnalysisStep IR for notebook export

        Raises:
            GenomicsQualityError: If filtering fails or QC metrics missing
        """
        try:
            logger.debug("Starting variant filtering")

            # Validate QC metrics exist
            required_cols = ["call_rate", "maf", "hwe_p"]
            missing = [col for col in required_cols if col not in adata.var.columns]
            if missing:
                raise GenomicsQualityError(
                    f"Missing QC metrics: {missing}. Run assess_quality() first."
                )

            n_variants_before = adata.n_vars

            # Identify passing variants (use qc_pass if available, else calculate)
            if "qc_pass" in adata.var.columns:
                passing_variants = adata.var["qc_pass"]
            else:
                passing_variants = (
                    (adata.var["call_rate"] >= min_call_rate)
                    & (adata.var["maf"] >= min_maf)
                    & (adata.var["hwe_p"] >= min_hwe_p)
                )

            # Filter AnnData
            adata_filtered = adata[:, passing_variants].copy()
            n_variants_after = adata_filtered.n_vars
            n_variants_removed = n_variants_before - n_variants_after

            # Compile statistics
            stats = {
                "analysis_type": "variant_filtering",
                "min_call_rate": min_call_rate,
                "min_maf": min_maf,
                "min_hwe_p": min_hwe_p,
                "variants_before": n_variants_before,
                "variants_after": n_variants_after,
                "variants_removed": n_variants_removed,
                "variants_retained_pct": float(
                    (n_variants_after / n_variants_before) * 100
                ),
                "removal_reasons": {
                    "low_call_rate": int(
                        (adata.var["call_rate"] < min_call_rate).sum()
                    ),
                    "low_maf": int((adata.var["maf"] < min_maf).sum()),
                    "hwe_fail": int((adata.var["hwe_p"] < min_hwe_p).sum()),
                },
            }

            logger.info(
                f"Variant filtering completed: {n_variants_after}/{n_variants_before} variants retained "
                f"({stats['variants_retained_pct']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_filter_variants_ir(
                min_call_rate=min_call_rate,
                min_maf=min_maf,
                min_hwe_p=min_hwe_p,
            )

            return adata_filtered, stats, ir

        except Exception as e:
            logger.exception(f"Error in variant filtering: {e}")
            raise GenomicsQualityError(f"Variant filtering failed: {str(e)}")

    # ===== Helper Methods =====

    def _calculate_sample_call_rate(self, gt: np.ndarray) -> np.ndarray:
        """
        Calculate per-sample call rates.

        Args:
            gt: Genotype matrix (samples × variants)

        Returns:
            Array of call rates per sample (length = n_samples)
        """
        # Called genotypes are >= 0 (0, 1, 2)
        # Missing genotypes are -1
        called = gt >= 0
        call_rate = called.mean(axis=1)  # Mean across variants
        return call_rate

    def _calculate_variant_call_rate(self, gt: np.ndarray) -> np.ndarray:
        """
        Calculate per-variant call rates.

        Args:
            gt: Genotype matrix (samples × variants)

        Returns:
            Array of call rates per variant (length = n_variants)
        """
        called = gt >= 0
        call_rate = called.mean(axis=0)  # Mean across samples
        return call_rate

    def _calculate_heterozygosity(self, gt: np.ndarray) -> np.ndarray:
        """
        Calculate per-sample heterozygosity.

        Args:
            gt: Genotype matrix (samples × variants)

        Returns:
            Array of heterozygosity rates per sample (length = n_samples)
        """
        # Heterozygous genotypes are coded as 1
        het = gt == 1
        called = gt >= 0

        # Heterozygosity = n_het / n_called_genotypes
        n_het = het.sum(axis=1)
        n_called = called.sum(axis=1)

        # Avoid division by zero
        heterozygosity = np.divide(
            n_het, n_called, out=np.zeros_like(n_het, dtype=float), where=n_called > 0
        )
        return heterozygosity

    def _calculate_het_z_score(self, heterozygosity: np.ndarray) -> np.ndarray:
        """
        Calculate heterozygosity z-scores (number of SD from mean).

        Args:
            heterozygosity: Array of heterozygosity values

        Returns:
            Array of z-scores
        """
        mean_het = heterozygosity.mean()
        std_het = heterozygosity.std()

        if std_het == 0:
            logger.warning("Zero standard deviation in heterozygosity. Returning zeros.")
            return np.zeros_like(heterozygosity)

        z_scores = (heterozygosity - mean_het) / std_het
        return z_scores

    def _calculate_maf(self, gt: np.ndarray) -> np.ndarray:
        """
        Calculate minor allele frequency per variant.

        MAF = frequency of the less common allele.

        Args:
            gt: Genotype matrix (samples × variants)
                Values: 0 (homref), 1 (het), 2 (homalt), -1 (missing)

        Returns:
            Array of MAF values per variant (length = n_variants)
        """
        # Count alleles (each sample contributes 2 alleles)
        # Genotype 0 → 0 alt alleles
        # Genotype 1 → 1 alt allele
        # Genotype 2 → 2 alt alleles
        # Genotype -1 → missing (exclude)

        called = gt >= 0
        n_alleles = called.sum(axis=0) * 2  # Total alleles per variant

        # Count alt alleles
        alt_alleles = np.where(called, gt, 0).sum(axis=0)

        # Allele frequency for alt allele
        alt_freq = np.divide(
            alt_alleles,
            n_alleles,
            out=np.zeros(n_alleles.shape, dtype=float),
            where=n_alleles > 0,
        )

        # MAF is the minimum of (alt_freq, 1 - alt_freq)
        maf = np.minimum(alt_freq, 1 - alt_freq)
        return maf

    def _calculate_hwe(self, gt: np.ndarray) -> np.ndarray:
        """
        Calculate Hardy-Weinberg equilibrium p-values per variant.

        Uses chi-squared test comparing observed vs expected genotype counts.

        Args:
            gt: Genotype matrix (samples × variants)

        Returns:
            Array of HWE p-values per variant (length = n_variants)
        """
        n_variants = gt.shape[1]
        hwe_p_values = np.zeros(n_variants)

        for i in range(n_variants):
            genotypes = gt[:, i]
            called = genotypes >= 0

            if called.sum() == 0:
                hwe_p_values[i] = 1.0  # No data, assume equilibrium
                continue

            # Count genotypes
            n_aa = (genotypes[called] == 0).sum()  # Homozygous ref
            n_ab = (genotypes[called] == 1).sum()  # Heterozygous
            n_bb = (genotypes[called] == 2).sum()  # Homozygous alt

            n_total = n_aa + n_ab + n_bb

            if n_total == 0:
                hwe_p_values[i] = 1.0
                continue

            # Calculate allele frequencies
            n_a = 2 * n_aa + n_ab
            n_b = 2 * n_bb + n_ab
            total_alleles = n_a + n_b

            if total_alleles == 0:
                hwe_p_values[i] = 1.0
                continue

            p = n_a / total_alleles  # Frequency of A allele
            q = n_b / total_alleles  # Frequency of B allele

            # Expected counts under HWE
            exp_aa = n_total * p * p
            exp_ab = n_total * 2 * p * q
            exp_bb = n_total * q * q

            # Chi-squared test
            observed = np.array([n_aa, n_ab, n_bb])
            expected = np.array([exp_aa, exp_ab, exp_bb])

            # Avoid division by zero
            expected = np.maximum(expected, 1e-10)

            chi2 = np.sum((observed - expected) ** 2 / expected)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)  # 1 degree of freedom

            hwe_p_values[i] = p_value

        return hwe_p_values

    # ===== IR Creation Methods =====

    def _create_assess_quality_ir(
        self,
        min_call_rate: float,
        min_maf: float,
        hwe_pvalue: float,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for quality assessment.

        Args:
            min_call_rate: Minimum call rate threshold
            min_maf: Minimum MAF threshold
            hwe_pvalue: Minimum HWE p-value

        Returns:
            AnalysisStep with complete code generation instructions
        """
        # Create parameter schema with Papermill flags
        parameter_schema = {
            "min_call_rate": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.95,
                required=False,
                validation_rule="min_call_rate > 0 and min_call_rate <= 1",
                description="Minimum call rate threshold for samples/variants",
            ),
            "min_maf": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.01,
                required=False,
                validation_rule="min_maf > 0 and min_maf < 0.5",
                description="Minimum minor allele frequency threshold",
            ),
            "hwe_pvalue": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=1e-10,
                required=False,
                validation_rule="hwe_pvalue > 0 and hwe_pvalue < 1",
                description="Minimum Hardy-Weinberg equilibrium p-value",
            ),
        }

        # Jinja2 template with parameter placeholders
        code_template = """# Calculate genomics QC metrics
import numpy as np
from scipy import stats

# Get genotype matrix (samples × variants)
gt = adata.layers['GT']

# === Per-sample metrics ===
# Call rate
called = gt >= 0
sample_call_rate = called.mean(axis=1)
adata.obs['call_rate'] = sample_call_rate

# Heterozygosity
het = gt == 1
n_het = het.sum(axis=1)
n_called = called.sum(axis=1)
heterozygosity = np.divide(n_het, n_called, out=np.zeros_like(n_het, dtype=float), where=n_called > 0)
adata.obs['heterozygosity'] = heterozygosity

# Heterozygosity z-score
mean_het = heterozygosity.mean()
std_het = heterozygosity.std()
het_z_score = (heterozygosity - mean_het) / std_het if std_het > 0 else np.zeros_like(heterozygosity)
adata.obs['het_z_score'] = het_z_score

# === Per-variant metrics ===
# Call rate
variant_call_rate = called.mean(axis=0)
adata.var['call_rate'] = variant_call_rate

# Minor allele frequency (MAF)
n_alleles = called.sum(axis=0) * 2
alt_alleles = np.where(called, gt, 0).sum(axis=0)
alt_freq = np.divide(alt_alleles, n_alleles, out=np.zeros(n_alleles.shape, dtype=float), where=n_alleles > 0)
maf = np.minimum(alt_freq, 1 - alt_freq)
adata.var['maf'] = maf

# Hardy-Weinberg equilibrium (HWE) - simplified for notebook
n_variants = gt.shape[1]
hwe_p_values = np.ones(n_variants)
for i in range(n_variants):
    genotypes = gt[:, i]
    called_i = genotypes >= 0
    if called_i.sum() > 0:
        n_aa = (genotypes[called_i] == 0).sum()
        n_ab = (genotypes[called_i] == 1).sum()
        n_bb = (genotypes[called_i] == 2).sum()
        n_total = n_aa + n_ab + n_bb
        if n_total > 0:
            p = (2 * n_aa + n_ab) / (2 * n_total)
            q = 1 - p
            exp_aa = n_total * p * p
            exp_ab = n_total * 2 * p * q
            exp_bb = n_total * q * q
            expected = np.maximum([exp_aa, exp_ab, exp_bb], 1e-10)
            chi2 = np.sum((np.array([n_aa, n_ab, n_bb]) - expected) ** 2 / expected)
            hwe_p_values[i] = 1 - stats.chi2.cdf(chi2, df=1)
adata.var['hwe_p'] = hwe_p_values

# QC pass/fail flag for variants
qc_pass = (variant_call_rate >= {{ min_call_rate }}) & (maf >= {{ min_maf }}) & (hwe_p_values >= {{ hwe_pvalue }})
adata.var['qc_pass'] = qc_pass

# Display QC summary
print(f"Samples: {adata.n_obs}")
print(f"Variants: {adata.n_vars}")
print(f"Variants passing QC: {qc_pass.sum()} ({100 * qc_pass.sum() / adata.n_vars:.1f}%)")
print(f"Mean sample call rate: {sample_call_rate.mean():.3f}")
print(f"Mean variant call rate: {variant_call_rate.mean():.3f}")
print(f"Mean MAF: {maf.mean():.4f}")
"""

        # Create AnalysisStep
        ir = AnalysisStep(
            operation="genomics.qc.assess",
            tool_name="GenomicsQualityService.assess_quality",
            description="Calculate genomics quality control metrics (call rate, MAF, HWE, heterozygosity)",
            library="scipy",
            code_template=code_template,
            imports=["import numpy as np", "from scipy import stats"],
            parameters={
                "min_call_rate": min_call_rate,
                "min_maf": min_maf,
                "hwe_pvalue": hwe_pvalue,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "qc_type": "genomics",
                "method": "uk_biobank_standards",
            },
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for genomics quality assessment: {ir.operation}")
        return ir

    def _create_filter_samples_ir(
        self,
        min_call_rate: float,
        het_sd_threshold: float,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for sample filtering.

        Args:
            min_call_rate: Minimum call rate threshold
            het_sd_threshold: Heterozygosity SD threshold

        Returns:
            AnalysisStep with complete code generation instructions
        """
        parameter_schema = {
            "min_call_rate": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.95,
                required=False,
                validation_rule="min_call_rate > 0 and min_call_rate <= 1",
                description="Minimum sample call rate threshold",
            ),
            "het_sd_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=3.0,
                required=False,
                validation_rule="het_sd_threshold > 0",
                description="Heterozygosity z-score threshold (SD from mean)",
            ),
        }

        code_template = """# Filter samples based on QC metrics
import numpy as np

n_samples_before = adata.n_obs

# Identify passing samples
passing_samples = (
    (adata.obs['call_rate'] >= {{ min_call_rate }}) &
    (np.abs(adata.obs['het_z_score']) <= {{ het_sd_threshold }})
)

# Filter AnnData
adata = adata[passing_samples, :].copy()
n_samples_after = adata.n_obs

# Display filtering summary
print(f"Samples before filtering: {n_samples_before}")
print(f"Samples after filtering: {n_samples_after}")
print(f"Samples removed: {n_samples_before - n_samples_after}")
print(f"Retention rate: {100 * n_samples_after / n_samples_before:.1f}%")
"""

        ir = AnalysisStep(
            operation="genomics.qc.filter_samples",
            tool_name="GenomicsQualityService.filter_samples",
            description="Filter samples based on call rate and heterozygosity",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={
                "min_call_rate": min_call_rate,
                "het_sd_threshold": het_sd_threshold,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"qc_type": "genomics", "filter_type": "samples"},
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for sample filtering: {ir.operation}")
        return ir

    def _create_filter_variants_ir(
        self,
        min_call_rate: float,
        min_maf: float,
        min_hwe_p: float,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for variant filtering.

        Args:
            min_call_rate: Minimum call rate threshold
            min_maf: Minimum MAF threshold
            min_hwe_p: Minimum HWE p-value

        Returns:
            AnalysisStep with complete code generation instructions
        """
        parameter_schema = {
            "min_call_rate": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.99,
                required=False,
                validation_rule="min_call_rate > 0 and min_call_rate <= 1",
                description="Minimum variant call rate threshold",
            ),
            "min_maf": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=0.01,
                required=False,
                validation_rule="min_maf > 0 and min_maf < 0.5",
                description="Minimum minor allele frequency threshold",
            ),
            "min_hwe_p": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=1e-10,
                required=False,
                validation_rule="min_hwe_p > 0 and min_hwe_p < 1",
                description="Minimum Hardy-Weinberg equilibrium p-value",
            ),
        }

        code_template = """# Filter variants based on QC metrics
n_variants_before = adata.n_vars

# Identify passing variants
if 'qc_pass' in adata.var.columns:
    passing_variants = adata.var['qc_pass']
else:
    passing_variants = (
        (adata.var['call_rate'] >= {{ min_call_rate }}) &
        (adata.var['maf'] >= {{ min_maf }}) &
        (adata.var['hwe_p'] >= {{ min_hwe_p }})
    )

# Filter AnnData
adata = adata[:, passing_variants].copy()
n_variants_after = adata.n_vars

# Display filtering summary
print(f"Variants before filtering: {n_variants_before}")
print(f"Variants after filtering: {n_variants_after}")
print(f"Variants removed: {n_variants_before - n_variants_after}")
print(f"Retention rate: {100 * n_variants_after / n_variants_before:.1f}%")
"""

        ir = AnalysisStep(
            operation="genomics.qc.filter_variants",
            tool_name="GenomicsQualityService.filter_variants",
            description="Filter variants based on call rate, MAF, and Hardy-Weinberg equilibrium",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={
                "min_call_rate": min_call_rate,
                "min_maf": min_maf,
                "min_hwe_p": min_hwe_p,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={"qc_type": "genomics", "filter_type": "variants"},
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for variant filtering: {ir.operation}")
        return ir
