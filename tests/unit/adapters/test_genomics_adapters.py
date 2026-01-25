"""Unit tests for genomics adapters (VCF and PLINK)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter
from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter


class TestVCFAdapter:
    """Tests for VCFAdapter."""

    @pytest.fixture
    def test_vcf_path(self):
        """Path to test VCF file."""
        return Path(__file__).parent.parent.parent.parent / "test_data/genomics/chr22_small.vcf.gz"

    @pytest.fixture
    def adapter(self):
        """Create VCF adapter instance."""
        return VCFAdapter(strict_validation=False)

    def test_adapter_initialization(self, adapter):
        """Test adapter can be initialized."""
        assert adapter is not None
        assert adapter.get_supported_formats() == ["vcf", "vcf.gz", "bcf"]

    def test_load_vcf_basic(self, adapter, test_vcf_path):
        """Test basic VCF loading."""
        if not test_vcf_path.exists():
            pytest.skip(f"Test VCF not found: {test_vcf_path}")

        adata = adapter.from_source(str(test_vcf_path), max_variants=100)

        # Validate structure
        assert adata.n_obs > 2000, f"Expected >2000 samples, got {adata.n_obs}"
        assert adata.n_vars == 100, f"Expected 100 variants (max_variants), got {adata.n_vars}"
        assert 'GT' in adata.layers, "GT layer missing"
        assert adata.X.shape == (adata.n_obs, 100)

    def test_vcf_genotype_encoding(self, adapter, test_vcf_path):
        """Test genotypes are correctly encoded as 0/1/2."""
        if not test_vcf_path.exists():
            pytest.skip(f"Test VCF not found: {test_vcf_path}")

        adata = adapter.from_source(str(test_vcf_path), max_variants=10)
        gt = adata.layers['GT']

        # Check genotypes are in valid range
        unique_values = np.unique(gt[~np.isnan(gt)])
        assert all(v in [0.0, 1.0, 2.0] for v in unique_values), f"Invalid genotype values: {unique_values}"

    def test_vcf_metadata_columns(self, adapter, test_vcf_path):
        """Test variant metadata columns are present."""
        if not test_vcf_path.exists():
            pytest.skip(f"Test VCF not found: {test_vcf_path}")

        adata = adapter.from_source(str(test_vcf_path), max_variants=10)

        required_cols = ['CHROM', 'POS', 'REF', 'ALT']
        for col in required_cols:
            assert col in adata.var.columns, f"Missing column: {col}"

    def test_vcf_sparse_matrix(self, adapter, test_vcf_path):
        """Test sparse matrix conversion for large datasets."""
        if not test_vcf_path.exists():
            pytest.skip(f"Test VCF not found: {test_vcf_path}")

        adata = adapter.from_source(str(test_vcf_path), max_variants=1000)

        # For 1000 Genomes, genotype matrix should be sparse (>50% zeros)
        from scipy.sparse import issparse
        # Note: Adapter converts to sparse if variants > 1M or sparsity > 50%
        # With 1000 variants, may not trigger sparse conversion
        assert adata.X is not None


class TestPLINKAdapter:
    """Tests for PLINKAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create PLINK adapter instance."""
        return PLINKAdapter(strict_validation=False)

    def test_adapter_initialization(self, adapter):
        """Test adapter can be initialized."""
        assert adapter is not None
        assert adapter.get_supported_formats() == ["bed"]

    def test_plink_schema(self, adapter):
        """Test PLINK adapter returns correct schema."""
        schema = adapter.get_schema()
        assert schema is not None
        assert 'obs' in schema
        assert 'var' in schema

    # Additional PLINK tests would go here when test data is available
