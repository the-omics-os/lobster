"""
Unit tests for genomics adapters (VCFAdapter and PLINKAdapter).

This module tests:
- VCFAdapter: VCF/BCF file loading with various parameters
- PLINKAdapter: PLINK binary format loading
- Schema validation for genomics data
- Edge cases and error handling

Test data:
- chr22.vcf.gz: 1000 Genomes Phase 3 chr22 (2504 samples)
- test_chr22.bed/bim/fam: Generated PLINK test data (100 samples, 1000 variants)
"""

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter
from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

# ===============================================================================
# Test Data Paths
# ===============================================================================


@pytest.fixture
def test_data_dir():
    """Get path to genomics test data directory."""
    return Path(__file__).parent.parent.parent.parent / "test_data" / "genomics"


@pytest.fixture
def vcf_path(test_data_dir):
    """Path to chr22.vcf.gz test file (1000 Genomes Phase 3)."""
    return test_data_dir / "chr22.vcf.gz"


@pytest.fixture
def plink_prefix(test_data_dir):
    """Path to PLINK test files (without extension)."""
    return test_data_dir / "plink_test" / "test_chr22"


# ===============================================================================
# VCFAdapter Tests
# ===============================================================================


@pytest.mark.unit
class TestVCFAdapterCore:
    """Test VCFAdapter core functionality."""

    def test_adapter_creation(self):
        """Test that VCFAdapter can be instantiated."""
        adapter = VCFAdapter(strict_validation=False)
        assert adapter is not None
        assert hasattr(adapter, "from_source")
        assert adapter.get_supported_formats() == ["vcf", "vcf.gz", "bcf"]

    def test_load_vcf_basic(self, vcf_path):
        """Test basic VCF loading."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=100)

        # Verify structure
        assert isinstance(adata, AnnData)
        assert (
            adata.n_obs > 2000
        ), f"Expected >2000 samples (1000 Genomes), got {adata.n_obs}"
        assert (
            adata.n_vars == 100
        ), f"Expected 100 variants (max_variants), got {adata.n_vars}"
        assert "GT" in adata.layers
        assert adata.X.shape == (adata.n_obs, 100)

    def test_vcf_required_columns(self, vcf_path):
        """Test that required VCF columns are present."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=10)

        # Required columns in var
        required_cols = ["CHROM", "POS", "REF", "ALT"]
        for col in required_cols:
            assert col in adata.var.columns, f"Missing required column: {col}"

    def test_load_vcf_with_max_variants(self, vcf_path):
        """Test max_variants parameter."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=50)

        assert adata.n_vars == 50, f"Expected 50 variants, got {adata.n_vars}"

    def test_load_vcf_filter_pass_only(self, vcf_path):
        """Test filter_pass parameter."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=100, filter_pass=True)

        # All variants should have FILTER == PASS
        if "FILTER" in adata.var.columns:
            # Some VCFs might not have FILTER field
            filters = adata.var["FILTER"].values
            assert all(
                f == "PASS" or f == "." for f in filters
            ), f"Non-PASS variants found: {filters[filters != 'PASS']}"

    def test_vcf_genotype_encoding(self, vcf_path):
        """Test genotype encoding (0/1/2/-1)."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=10)

        gt = adata.X
        if hasattr(gt, "toarray"):
            gt = gt.toarray()

        # Check encoding range: 0=hom ref, 1=het, 2=hom alt, -1=missing
        unique_values = np.unique(gt[~np.isnan(gt)])
        assert all(
            v in [-1, 0, 1, 2] for v in unique_values
        ), f"Invalid genotype values: {unique_values}"

    def test_vcf_sparse_matrix_conversion(self, vcf_path):
        """Test sparse matrix optimization for high sparsity data."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=100)

        # 1000 Genomes has high sparsity (many rare variants)
        if hasattr(adata.X, "toarray"):
            # Sparse matrix detected
            sparsity = 1 - np.count_nonzero(adata.X.toarray()) / adata.X.size
            assert (
                sparsity > 0.5
            ), f"Expected sparsity > 50% for rare variants, got {sparsity:.1%}"
            print(f"Sparsity: {sparsity:.1%}")

    def test_vcf_metadata_preservation(self, vcf_path):
        """Test that VCF metadata is preserved in uns."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)
        adata = adapter.from_source(str(vcf_path), max_variants=10)

        # Check uns metadata
        assert "data_type" in adata.uns
        assert adata.uns["data_type"] == "genomics"
        assert "modality" in adata.uns
        assert adata.uns["modality"] == "wgs"
        assert "source_file" in adata.uns


@pytest.mark.unit
class TestVCFAdapterEdgeCases:
    """Test VCFAdapter edge cases and error handling."""

    def test_vcf_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        adapter = VCFAdapter(strict_validation=False)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            adapter.from_source("/nonexistent/file.vcf.gz")

    def test_vcf_invalid_max_variants(self, vcf_path):
        """Test invalid max_variants parameter."""
        if not vcf_path.exists():
            pytest.skip(f"Test VCF not found: {vcf_path}")

        adapter = VCFAdapter(strict_validation=False)

        # max_variants = 0 should raise error or return empty
        with pytest.raises((ValueError, RuntimeError)) or pytest.warns():
            adata = adapter.from_source(str(vcf_path), max_variants=0)
            if adata is not None:
                assert adata.n_vars == 0


# ===============================================================================
# PLINKAdapter Tests
# ===============================================================================


@pytest.mark.unit
class TestPLINKAdapterCore:
    """Test PLINKAdapter core functionality."""

    def test_adapter_creation(self):
        """Test that PLINKAdapter can be instantiated."""
        adapter = PLINKAdapter(strict_validation=False)
        assert adapter is not None
        assert hasattr(adapter, "from_source")
        assert adapter.get_supported_formats() == ["bed"]

    def test_load_plink_basic(self, plink_prefix):
        """Test basic PLINK loading with .bed file."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)

        # PLINK adapter accepts .bed file path
        adata = adapter.from_source(str(plink_prefix) + ".bed")

        # Verify structure
        assert isinstance(adata, AnnData)
        assert adata.n_obs == 100, f"Expected 100 samples, got {adata.n_obs}"
        assert adata.n_vars == 1000, f"Expected 1000 variants, got {adata.n_vars}"
        assert "GT" in adata.layers

    def test_load_plink_with_prefix(self, plink_prefix):
        """Test PLINK loading with prefix (no extension)."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)

        # Should work with prefix only (no .bed extension)
        adata = adapter.from_source(str(plink_prefix))

        assert adata.n_obs == 100
        assert adata.n_vars == 1000

    def test_plink_fam_metadata(self, plink_prefix):
        """Test that .fam metadata is loaded into obs."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)
        adata = adapter.from_source(str(plink_prefix))

        # .fam columns: FamilyID IndividualID FatherID MotherID Sex Phenotype
        expected_cols = [
            "individual_id",
            "family_id",
            "father_id",
            "mother_id",
            "sex",
            "phenotype",
        ]
        for col in expected_cols:
            assert col in adata.obs.columns, f"Missing .fam column: {col}"

    def test_plink_bim_metadata(self, plink_prefix):
        """Test that .bim metadata is loaded into var."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)
        adata = adapter.from_source(str(plink_prefix))

        # .bim columns: chr snp_id genetic_dist bp_pos allele1 allele2
        expected_cols = ["chromosome", "snp_id", "bp_position", "allele_1", "allele_2"]
        for col in expected_cols:
            assert col in adata.var.columns, f"Missing .bim column: {col}"

    def test_plink_genotype_encoding(self, plink_prefix):
        """Test PLINK genotype encoding (0/1/2/NaN)."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)
        adata = adapter.from_source(str(plink_prefix))

        gt = adata.X
        if hasattr(gt, "toarray"):
            gt = gt.toarray()

        # Check encoding range (PLINK uses NaN for missing, not -1 like VCF)
        unique_values = np.unique(gt[~np.isnan(gt)])
        assert all(
            v in [0, 1, 2] for v in unique_values
        ), f"Invalid genotype values: {unique_values}"

    def test_plink_metadata_preservation(self, plink_prefix):
        """Test that PLINK metadata is preserved in uns."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)
        adata = adapter.from_source(str(plink_prefix))

        # Check uns metadata
        assert "data_type" in adata.uns
        assert adata.uns["data_type"] == "genomics"
        assert "modality" in adata.uns
        assert adata.uns["modality"] == "snp_array"
        assert "source_file" in adata.uns


@pytest.mark.unit
class TestPLINKAdapterEdgeCases:
    """Test PLINKAdapter edge cases and error handling."""

    def test_plink_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        adapter = PLINKAdapter(strict_validation=False)
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            adapter.from_source("/nonexistent/file.bed")

    def test_plink_missing_bim_or_fam(self, tmp_path):
        """Test error when .bim or .fam files are missing."""
        adapter = PLINKAdapter(strict_validation=False)

        # Create a fake .bed file without .bim/.fam
        fake_bed = tmp_path / "fake.bed"
        fake_bed.write_bytes(b"\x6c\x1b\x01")  # PLINK magic bytes

        # Should raise error about missing .bim or .fam
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            adapter.from_source(str(fake_bed))


@pytest.mark.unit
class TestPLINKAdapterFiltering:
    """Test PLINK adapter filtering options."""

    def test_plink_maf_filter(self, plink_prefix):
        """Test MAF filtering during loading."""
        if not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip(f"Test PLINK not found: {plink_prefix}.bed")

        adapter = PLINKAdapter(strict_validation=False)

        # Load without MAF filter
        adata_all = adapter.from_source(str(plink_prefix))
        n_variants_all = adata_all.n_vars

        # Load with MAF filter (should reduce variant count)
        adata_filtered = adapter.from_source(str(plink_prefix), maf_min=0.05)
        n_variants_filtered = adata_filtered.n_vars

        # MAF filter should reduce variants (unless all variants have MAF > 0.05)
        assert (
            n_variants_filtered <= n_variants_all
        ), "MAF filter should reduce or equal variant count"
        print(f"MAF filter: {n_variants_all} → {n_variants_filtered} variants")


# ===============================================================================
# Cross-Adapter Consistency Tests
# ===============================================================================


@pytest.mark.unit
class TestAdapterConsistency:
    """Test that both adapters produce consistent AnnData structure."""

    def test_consistent_genotype_shape(self, vcf_path, plink_prefix):
        """Test that both adapters use samples × variants shape."""
        if not vcf_path.exists() or not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip("Test data not found")

        vcf_adapter = VCFAdapter(strict_validation=False)
        plink_adapter = PLINKAdapter(strict_validation=False)

        adata_vcf = vcf_adapter.from_source(str(vcf_path), max_variants=10)
        adata_plink = plink_adapter.from_source(str(plink_prefix))

        # Both should have samples as rows, variants as columns
        assert adata_vcf.X.shape == (adata_vcf.n_obs, adata_vcf.n_vars)
        assert adata_plink.X.shape == (adata_plink.n_obs, adata_plink.n_vars)

    def test_consistent_layer_structure(self, vcf_path, plink_prefix):
        """Test that both adapters create GT layer."""
        if not vcf_path.exists() or not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip("Test data not found")

        vcf_adapter = VCFAdapter(strict_validation=False)
        plink_adapter = PLINKAdapter(strict_validation=False)

        adata_vcf = vcf_adapter.from_source(str(vcf_path), max_variants=10)
        adata_plink = plink_adapter.from_source(str(plink_prefix))

        # Both should have GT layer
        assert "GT" in adata_vcf.layers
        assert "GT" in adata_plink.layers

    def test_consistent_uns_structure(self, vcf_path, plink_prefix):
        """Test that both adapters populate uns metadata."""
        if not vcf_path.exists() or not Path(str(plink_prefix) + ".bed").exists():
            pytest.skip("Test data not found")

        vcf_adapter = VCFAdapter(strict_validation=False)
        plink_adapter = PLINKAdapter(strict_validation=False)

        adata_vcf = vcf_adapter.from_source(str(vcf_path), max_variants=10)
        adata_plink = plink_adapter.from_source(str(plink_prefix))

        # Both should have data_type and modality
        assert "data_type" in adata_vcf.uns
        assert "data_type" in adata_plink.uns
        assert adata_vcf.uns["data_type"] == "genomics"
        assert adata_plink.uns["data_type"] == "genomics"

        assert "modality" in adata_vcf.uns
        assert "modality" in adata_plink.uns
        # Modalities differ (wgs vs snp_array)
        assert adata_vcf.uns["modality"] == "wgs"
        assert adata_plink.uns["modality"] == "snp_array"
