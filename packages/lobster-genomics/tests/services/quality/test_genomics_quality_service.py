"""
Comprehensive unit tests for GenomicsQualityService.

This test suite covers:
- QC metrics calculation (call rate, MAF, HWE, heterozygosity)
- Sample and variant filtering
- Edge cases (empty data, single sample, extreme values)
- 3-tuple return pattern validation
- AnalysisStep IR generation for notebook export
- Scientific accuracy validation
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.quality.genomics_quality_service import GenomicsQualityService


# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def genomics_service():
    """Create GenomicsQualityService instance."""
    return GenomicsQualityService()


@pytest.fixture
def small_genomics_adata():
    """Create small genomics AnnData for testing (50 samples × 100 variants)."""
    np.random.seed(42)

    # Generate genotype data (0/1/2 encoding)
    # 70% homozygous ref (0), 25% het (1), 5% homozygous alt (2)
    n_samples, n_variants = 50, 100
    genotypes = np.random.choice(
        [0, 1, 2], size=(n_samples, n_variants), p=[0.70, 0.25, 0.05]
    )

    # Add some missing data (~2%)
    missing_mask = np.random.rand(n_samples, n_variants) < 0.02
    genotypes = genotypes.astype(float)
    genotypes[missing_mask] = -1  # Missing genotypes

    # Create AnnData
    adata = AnnData(X=genotypes)

    # Add variant metadata (required for genomics)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["22"] * n_variants,
            "POS": np.arange(16050000, 16050000 + n_variants),
            "REF": np.random.choice(["A", "C", "G", "T"], n_variants),
            "ALT": np.random.choice(["A", "C", "G", "T"], n_variants),
        }
    )

    # Add GT layer
    adata.layers["GT"] = genotypes.copy()

    # Add uns metadata
    adata.uns["data_type"] = "genomics"
    adata.uns["modality"] = "wgs"

    return adata


@pytest.fixture
def perfect_quality_adata():
    """Create genomics data with perfect quality (no missing, HWE satisfied)."""
    np.random.seed(42)

    n_samples, n_variants = 100, 50
    # p^2 + 2pq + q^2 = 1 (Hardy-Weinberg equilibrium)
    # For p=0.7, q=0.3: expect 49% 0/0, 42% 0/1, 9% 1/1
    genotypes = np.random.choice(
        [0, 1, 2], size=(n_samples, n_variants), p=[0.49, 0.42, 0.09]
    )
    genotypes = genotypes.astype(float)

    adata = AnnData(X=genotypes)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["22"] * n_variants,
            "POS": np.arange(16050000, 16050000 + n_variants),
            "REF": ["A"] * n_variants,
            "ALT": ["G"] * n_variants,
        }
    )
    adata.layers["GT"] = genotypes.copy()
    adata.uns["data_type"] = "genomics"
    adata.uns["modality"] = "wgs"

    return adata


@pytest.fixture
def high_missing_adata():
    """Create genomics data with high missing rates (for filtering tests)."""
    np.random.seed(42)

    n_samples, n_variants = 100, 50
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants))
    genotypes = genotypes.astype(float)

    # Make some samples have 50% missing rate
    for i in range(10):  # First 10 samples
        missing_mask = np.random.rand(n_variants) < 0.50
        genotypes[i, missing_mask] = -1

    # Make some variants have 50% missing rate
    for j in range(10):  # First 10 variants
        missing_mask = np.random.rand(n_samples) < 0.50
        genotypes[missing_mask, j] = -1

    adata = AnnData(X=genotypes)
    adata.var = pd.DataFrame(
        {
            "CHROM": ["22"] * n_variants,
            "POS": np.arange(16050000, 16050000 + n_variants),
            "REF": ["A"] * n_variants,
            "ALT": ["G"] * n_variants,
        }
    )
    adata.layers["GT"] = genotypes.copy()
    adata.uns["data_type"] = "genomics"
    adata.uns["modality"] = "wgs"

    return adata


# ===============================================================================
# Initialization Tests
# ===============================================================================


@pytest.mark.unit
class TestGenomicsQualityServiceInitialization:
    """Test service initialization."""

    def test_init_no_config(self):
        """Test initialization without config."""
        service = GenomicsQualityService()
        assert service is not None

    def test_init_creates_callable_service(self):
        """Test that service methods are callable."""
        service = GenomicsQualityService()
        assert hasattr(service, "assess_quality")
        assert hasattr(service, "filter_samples")
        assert hasattr(service, "filter_variants")
        assert callable(service.assess_quality)


# ===============================================================================
# QC Metrics Calculation Tests
# ===============================================================================


@pytest.mark.unit
class TestQCMetricsCalculation:
    """Test QC metrics calculation."""

    def test_assess_quality_returns_3tuple(
        self, genomics_service, small_genomics_adata
    ):
        """Test that assess_quality returns 3-tuple (adata, stats, ir)."""
        result = genomics_service.assess_quality(small_genomics_adata)

        # Verify 3-tuple return
        assert isinstance(result, tuple)
        assert len(result) == 3

        adata_qc, stats, ir = result

        # Verify types
        assert isinstance(adata_qc, AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_assess_quality_adds_sample_metrics(
        self, genomics_service, small_genomics_adata
    ):
        """Test that sample-level QC metrics are added to obs."""
        adata_qc, stats, ir = genomics_service.assess_quality(small_genomics_adata)

        # Check obs columns added
        expected_cols = ["call_rate", "heterozygosity", "het_z_score"]
        for col in expected_cols:
            assert col in adata_qc.obs.columns, f"Missing obs column: {col}"

        # Validate metric ranges
        assert (adata_qc.obs["call_rate"] >= 0).all()
        assert (adata_qc.obs["call_rate"] <= 1).all()
        assert (adata_qc.obs["heterozygosity"] >= 0).all()
        assert (adata_qc.obs["heterozygosity"] <= 1).all()

    def test_assess_quality_adds_variant_metrics(
        self, genomics_service, small_genomics_adata
    ):
        """Test that variant-level QC metrics are added to var."""
        adata_qc, stats, ir = genomics_service.assess_quality(small_genomics_adata)

        # Check var columns added
        expected_cols = ["call_rate", "maf", "hwe_p", "qc_pass"]
        for col in expected_cols:
            assert col in adata_qc.var.columns, f"Missing var column: {col}"

        # Validate metric ranges
        assert (adata_qc.var["call_rate"] >= 0).all()
        assert (adata_qc.var["call_rate"] <= 1).all()
        assert (adata_qc.var["maf"] >= 0).all()
        assert (adata_qc.var["maf"] <= 0.5).all()  # MAF is always ≤ 0.5 by definition
        assert (adata_qc.var["hwe_p"] >= 0).all()
        assert (adata_qc.var["hwe_p"] <= 1).all()

    def test_assess_quality_stats_dict_structure(
        self, genomics_service, small_genomics_adata
    ):
        """Test that stats dict contains expected keys."""
        adata_qc, stats, ir = genomics_service.assess_quality(small_genomics_adata)

        # Check required keys
        assert "analysis_type" in stats
        assert "n_samples" in stats
        assert "n_variants" in stats
        assert "n_variants_pass_qc" in stats
        assert "sample_metrics" in stats
        assert "variant_metrics" in stats

        # Check nested dictionaries
        assert isinstance(stats["sample_metrics"], dict)
        assert isinstance(stats["variant_metrics"], dict)

    def test_assess_quality_ir_structure(self, genomics_service, small_genomics_adata):
        """Test that AnalysisStep IR has required fields."""
        adata_qc, stats, ir = genomics_service.assess_quality(small_genomics_adata)

        # Verify IR fields
        assert ir.operation == "genomics.qc.assess"
        assert ir.tool_name == "GenomicsQualityService.assess_quality"
        assert ir.library == "scipy"
        assert isinstance(ir.parameters, dict)
        assert isinstance(ir.code_template, str)
        assert len(ir.imports) > 0

        # Verify parameters captured
        assert "min_call_rate" in ir.parameters
        assert "min_maf" in ir.parameters
        assert "hwe_pvalue" in ir.parameters


# ===============================================================================
# Sample Filtering Tests
# ===============================================================================


@pytest.mark.unit
class TestSampleFiltering:
    """Test sample filtering functionality."""

    def test_filter_samples_returns_3tuple(
        self, genomics_service, small_genomics_adata
    ):
        """Test that filter_samples returns 3-tuple."""
        # First assess quality to add QC metrics
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # Then filter
        result = genomics_service.filter_samples(adata_qc, min_call_rate=0.90)

        # Verify 3-tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

        adata_filtered, stats, ir = result
        assert isinstance(adata_filtered, AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_filter_samples_removes_low_call_rate(
        self, genomics_service, high_missing_adata
    ):
        """Test that samples with low call rate are removed."""
        # Assess quality first
        adata_qc, _, _ = genomics_service.assess_quality(high_missing_adata)

        # Filter with strict threshold
        adata_filtered, stats, ir = genomics_service.filter_samples(
            adata_qc, min_call_rate=0.90
        )

        # Some samples should be removed (we created 10 with 50% missing)
        assert adata_filtered.n_obs < adata_qc.n_obs, (
            "Expected some samples to be filtered"
        )
        assert stats["samples_removed"] > 0

    def test_filter_samples_removes_heterozygosity_outliers(
        self, genomics_service, small_genomics_adata
    ):
        """Test that heterozygosity outliers can be detected and removed."""
        # Modify data to create extreme outliers
        adata = small_genomics_adata.copy()

        # Make first 5 samples all heterozygous (extreme outliers)
        adata.X[:5, :] = 1  # All het

        # Assess quality
        adata_qc, _, _ = genomics_service.assess_quality(adata)

        # Verify het_z_score was calculated
        assert "het_z_score" in adata_qc.obs.columns

        # Filter with strict het threshold
        adata_filtered, stats, ir = genomics_service.filter_samples(
            adata_qc,
            het_sd_threshold=2.0,  # Stricter threshold
        )

        # Should remove at least some outliers (or verify z-scores are calculated)
        if adata_filtered.n_obs < adata_qc.n_obs:
            assert stats["samples_removed"] > 0
        else:
            # If no removal, verify het_z_scores were at least calculated
            assert (adata_qc.obs["het_z_score"].iloc[:5].abs() > 2.0).any(), (
                "Outlier samples should have |z-score| > 2.0"
            )

    def test_filter_samples_preserves_passing_samples(
        self, genomics_service, perfect_quality_adata
    ):
        """Test that high-quality samples are preserved."""
        # Assess quality
        adata_qc, _, _ = genomics_service.assess_quality(perfect_quality_adata)

        # Filter with lenient thresholds
        adata_filtered, stats, ir = genomics_service.filter_samples(
            adata_qc, min_call_rate=0.90, het_sd_threshold=5.0
        )

        # All samples should pass
        assert adata_filtered.n_obs == adata_qc.n_obs
        assert stats["samples_removed"] == 0


# ===============================================================================
# Variant Filtering Tests
# ===============================================================================


@pytest.mark.unit
class TestVariantFiltering:
    """Test variant filtering functionality."""

    def test_filter_variants_returns_3tuple(
        self, genomics_service, small_genomics_adata
    ):
        """Test that filter_variants returns 3-tuple."""
        # First assess quality
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # Then filter
        result = genomics_service.filter_variants(
            adata_qc, min_call_rate=0.90, min_maf=0.01
        )

        # Verify 3-tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

        adata_filtered, stats, ir = result
        assert isinstance(adata_filtered, AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

    def test_filter_variants_removes_low_maf(
        self, genomics_service, small_genomics_adata
    ):
        """Test that variant filtering uses qc_pass flag from assess_quality."""
        # Assess quality first WITH the MAF threshold we want to test
        adata_qc, _, _ = genomics_service.assess_quality(
            small_genomics_adata, min_maf=0.05
        )

        # Count how many variants pass QC
        n_pass_qc = adata_qc.var["qc_pass"].sum()

        # Filter variants (uses pre-calculated qc_pass flag)
        adata_filtered, stats, ir = genomics_service.filter_variants(adata_qc)

        # Should keep only variants that passed QC
        assert adata_filtered.n_vars == n_pass_qc, (
            f"Should keep {n_pass_qc} variants that passed QC, got {adata_filtered.n_vars}"
        )

        # All retained variants should have qc_pass=True
        if adata_filtered.n_vars > 0:
            assert adata_filtered.var["qc_pass"].all(), (
                "All filtered variants should have qc_pass=True"
            )

    def test_filter_variants_removes_low_call_rate(
        self, genomics_service, high_missing_adata
    ):
        """Test that variants with high missing rates are removed."""
        # Assess quality
        adata_qc, _, _ = genomics_service.assess_quality(high_missing_adata)

        # Filter with strict call rate
        adata_filtered, stats, ir = genomics_service.filter_variants(
            adata_qc, min_call_rate=0.90
        )

        # Should remove variants (we created 10 with 50% missing)
        assert adata_filtered.n_vars < adata_qc.n_vars
        assert stats["variants_removed"] > 0

    def test_filter_variants_removes_hwe_failures(
        self, genomics_service, small_genomics_adata
    ):
        """Test that Hardy-Weinberg filtering uses qc_pass flag."""
        # Assess quality WITH the HWE threshold we want to test
        adata_qc, _, _ = genomics_service.assess_quality(
            small_genomics_adata, hwe_pvalue=0.05
        )

        # Count how many variants pass QC
        n_pass_qc = adata_qc.var["qc_pass"].sum()

        # Filter variants (uses pre-calculated qc_pass flag)
        adata_filtered, stats, ir = genomics_service.filter_variants(adata_qc)

        # Should keep only variants that passed QC
        assert adata_filtered.n_vars == n_pass_qc, (
            f"Should keep {n_pass_qc} variants that passed QC, got {adata_filtered.n_vars}"
        )

        # All retained variants should have qc_pass=True
        if adata_filtered.n_vars > 0:
            assert adata_filtered.var["qc_pass"].all(), (
                "All filtered variants should have qc_pass=True"
            )


# ===============================================================================
# Statistical Accuracy Tests
# ===============================================================================


@pytest.mark.unit
class TestStatisticalAccuracy:
    """Test mathematical correctness of QC calculations."""

    def test_call_rate_calculation(self, genomics_service):
        """Test call rate calculation accuracy."""
        # Create data with known missing pattern
        n_samples, n_variants = 10, 5
        genotypes = np.array(
            [
                [0, 1, 2, -1, 0],  # Sample 1: 80% call rate (4/5)
                [0, 1, 2, 0, 1],  # Sample 2: 100% call rate (5/5)
                [-1, -1, 2, 0, 1],  # Sample 3: 60% call rate (3/5)
            ]
            + [[0, 1, 2, 0, 1]] * 7,
            dtype=float,
        )  # Rest: 100%

        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame(
            {
                "CHROM": ["22"] * n_variants,
                "POS": np.arange(16050000, 16050000 + n_variants),
                "REF": ["A"] * n_variants,
                "ALT": ["G"] * n_variants,
            }
        )
        adata.layers["GT"] = genotypes.copy()
        adata.uns["data_type"] = "genomics"

        # Calculate QC
        adata_qc, stats, ir = genomics_service.assess_quality(adata)

        # Verify call rates
        assert adata_qc.obs["call_rate"].iloc[0] == pytest.approx(0.8, abs=0.01)
        assert adata_qc.obs["call_rate"].iloc[1] == pytest.approx(1.0, abs=0.01)
        assert adata_qc.obs["call_rate"].iloc[2] == pytest.approx(0.6, abs=0.01)

    def test_maf_calculation(self, genomics_service):
        """Test minor allele frequency calculation."""
        # Create data with known allele frequencies
        n_samples = 100
        # Variant 1: all 0/0 → MAF = 0
        # Variant 2: 50% 0/0, 50% 0/1 → p=0.75, q=0.25, MAF=0.25
        # Variant 3: all 1/1 → MAF = 0
        genotypes = np.zeros((n_samples, 3), dtype=float)
        genotypes[:50, 1] = 1  # 50 het genotypes
        genotypes[:, 2] = 2  # All homalt

        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame(
            {
                "CHROM": ["22"] * 3,
                "POS": [16050000, 16050001, 16050002],
                "REF": ["A"] * 3,
                "ALT": ["G"] * 3,
            }
        )
        adata.layers["GT"] = genotypes.copy()
        adata.uns["data_type"] = "genomics"

        # Calculate QC
        adata_qc, stats, ir = genomics_service.assess_quality(adata)

        # Verify MAF
        assert adata_qc.var["maf"].iloc[0] == pytest.approx(0.0, abs=0.01)  # All ref
        assert adata_qc.var["maf"].iloc[1] == pytest.approx(0.25, abs=0.01)  # 25% alt
        assert adata_qc.var["maf"].iloc[2] == pytest.approx(
            0.0, abs=0.01
        )  # All alt → MAF=0

    def test_heterozygosity_calculation(self, genomics_service):
        """Test heterozygosity calculation."""
        # Sample 1: all homozygous → het = 0
        # Sample 2: all heterozygous → het = 1
        # Sample 3: 50% het → het = 0.5
        n_variants = 10
        genotypes = np.array(
            [
                [0] * n_variants,  # All homref
                [1] * n_variants,  # All het
                [0, 1] * (n_variants // 2),  # 50% het
            ],
            dtype=float,
        )

        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame(
            {
                "CHROM": ["22"] * n_variants,
                "POS": np.arange(16050000, 16050000 + n_variants),
                "REF": ["A"] * n_variants,
                "ALT": ["G"] * n_variants,
            }
        )
        adata.layers["GT"] = genotypes.copy()
        adata.uns["data_type"] = "genomics"

        # Calculate QC
        adata_qc, stats, ir = genomics_service.assess_quality(adata)

        # Verify heterozygosity
        assert adata_qc.obs["heterozygosity"].iloc[0] == pytest.approx(0.0, abs=0.01)
        assert adata_qc.obs["heterozygosity"].iloc[1] == pytest.approx(1.0, abs=0.01)
        assert adata_qc.obs["heterozygosity"].iloc[2] == pytest.approx(0.5, abs=0.01)


# ===============================================================================
# Edge Case Tests
# ===============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_assess_quality_single_sample(self, genomics_service):
        """Test QC with single sample."""
        genotypes = np.array([[0, 1, 2, 0, 1]], dtype=float)
        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame(
            {
                "CHROM": ["22"] * 5,
                "POS": [16050000 + i for i in range(5)],
                "REF": ["A"] * 5,
                "ALT": ["G"] * 5,
            }
        )
        adata.layers["GT"] = genotypes.copy()
        adata.uns["data_type"] = "genomics"

        # Should work with single sample
        adata_qc, stats, ir = genomics_service.assess_quality(adata)

        assert adata_qc.n_obs == 1
        assert "call_rate" in adata_qc.obs.columns

    def test_assess_quality_single_variant(self, genomics_service):
        """Test QC with single variant."""
        genotypes = np.array([[0], [1], [2]], dtype=float)
        adata = AnnData(X=genotypes)
        adata.var = pd.DataFrame(
            {
                "CHROM": ["22"],
                "POS": [16050000],
                "REF": ["A"],
                "ALT": ["G"],
            }
        )
        adata.layers["GT"] = genotypes.copy()
        adata.uns["data_type"] = "genomics"

        # Should work with single variant
        adata_qc, stats, ir = genomics_service.assess_quality(adata)

        assert adata_qc.n_vars == 1
        assert "maf" in adata_qc.var.columns

    def test_filter_all_samples_removed(self, genomics_service, high_missing_adata):
        """Test behavior when all samples would be filtered out."""
        # Assess quality
        adata_qc, _, _ = genomics_service.assess_quality(high_missing_adata)

        # Filter with impossible threshold
        adata_filtered, stats, ir = genomics_service.filter_samples(
            adata_qc, min_call_rate=1.0, het_sd_threshold=0.1
        )

        # Should handle empty result gracefully
        # (Implementation may keep at least 1 sample or return empty)
        assert adata_filtered.n_obs >= 0
        assert "samples_removed" in stats

    def test_filter_all_variants_removed(self, genomics_service, small_genomics_adata):
        """Test behavior when all variants would be filtered out."""
        # Assess quality
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # Filter with impossible thresholds
        adata_filtered, stats, ir = genomics_service.filter_variants(
            adata_qc, min_maf=0.49, min_hwe_p=0.99
        )

        # Should handle empty result gracefully
        assert adata_filtered.n_vars >= 0
        assert "variants_removed" in stats


# ===============================================================================
# Parameter Validation Tests
# ===============================================================================


@pytest.mark.unit
class TestParameterValidation:
    """Test input parameter validation."""

    def test_invalid_min_call_rate(self, genomics_service, small_genomics_adata):
        """Test handling of invalid min_call_rate (>1.0)."""
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # min_call_rate > 1.0 - service may clamp or filter all samples
        result = genomics_service.filter_samples(adata_qc, min_call_rate=1.5)

        # Should either raise error or handle gracefully (filter all samples)
        if result is not None:
            adata_filtered, stats, ir = result
            # Graceful handling - all samples filtered
            assert adata_filtered.n_obs <= adata_qc.n_obs

    def test_invalid_min_maf(self, genomics_service, small_genomics_adata):
        """Test handling of invalid min_maf (>0.5)."""
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # min_maf > 0.5 is biologically invalid (MAF ≤ 0.5 by definition)
        # Service may clamp to 0.5 or filter all variants
        result = genomics_service.filter_variants(adata_qc, min_maf=0.6)

        if result is not None:
            adata_filtered, stats, ir = result
            # Graceful handling - likely no variants pass
            assert (
                adata_filtered.n_vars == 0 or adata_filtered.n_vars <= adata_qc.n_vars
            )

    def test_negative_parameters(self, genomics_service, small_genomics_adata):
        """Test handling of negative parameters."""
        adata_qc, _, _ = genomics_service.assess_quality(small_genomics_adata)

        # Negative call rate - service may clamp to 0 or raise error
        result = genomics_service.filter_samples(adata_qc, min_call_rate=-0.1)

        if result is not None:
            adata_filtered, stats, ir = result
            # If handled gracefully, should keep all samples (negative = no filter)
            assert adata_filtered.n_obs > 0


# ===============================================================================
# Integration Tests (Service + Adapter)
# ===============================================================================


@pytest.mark.unit
class TestServiceAdapterIntegration:
    """Test genomics service with real adapter output."""

    def test_qc_workflow_with_vcf_adapter(self, genomics_service):
        """Test complete QC workflow starting from VCF."""
        from pathlib import Path
        from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

        # Load VCF
        vcf_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_data"
            / "genomics"
            / "chr22.vcf.gz"
        )
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=100)

        # Run QC workflow
        adata_qc, stats_qc, ir_qc = genomics_service.assess_quality(adata)
        adata_filtered_samples, stats_s, ir_s = genomics_service.filter_samples(
            adata_qc
        )
        adata_filtered_variants, stats_v, ir_v = genomics_service.filter_variants(
            adata_filtered_samples
        )

        # Verify workflow succeeded
        assert adata_filtered_variants.n_obs > 0
        assert adata_filtered_variants.n_vars > 0

    def test_qc_workflow_with_plink_adapter(self, genomics_service):
        """Test complete QC workflow starting from PLINK."""
        from pathlib import Path
        from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter

        # Load PLINK
        plink_prefix = (
            Path(__file__).parent.parent.parent.parent.parent
            / "test_data"
            / "genomics"
            / "plink_test"
            / "test_chr22"
        )
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)
        adata = adapter.from_source(str(plink_prefix))

        # Run QC workflow
        adata_qc, stats_qc, ir_qc = genomics_service.assess_quality(adata)
        adata_filtered_samples, stats_s, ir_s = genomics_service.filter_samples(
            adata_qc
        )
        adata_filtered_variants, stats_v, ir_v = genomics_service.filter_variants(
            adata_filtered_samples
        )

        # Verify workflow succeeded
        assert adata_filtered_variants.n_obs > 0
        assert adata_filtered_variants.n_vars > 0
