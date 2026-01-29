"""
Unit tests for TenXGenomicsLoader in lobster/services/data_access/geo/loaders/tenx.py.

Tests verify:
1. detect_features_format() correctly identifies column counts (2-col V2, 3-col V3, 1-col edge case)
2. load_10x_manual() handles non-standard formats (single-column genes files)
3. Dimension validation catches mismatches
4. Gene ID/symbol parsing for different formats

REGRESSION TEST: These tests specifically prevent bugs where non-standard 10X formats
fail to load or result in 0 genes.
"""

import gzip
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from lobster.services.data_access.geo.loaders.tenx import TenXGenomicsLoader


class TestTenXFeaturesFormatDetection:
    """Tests for detect_features_format() column count detection."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, temp_dir):
        """Create loader with mock downloader."""
        mock_downloader = Mock()
        return TenXGenomicsLoader(mock_downloader, cache_dir=temp_dir)

    # --- V3 Format Tests (3 columns) ---

    def test_standard_10x_3_columns_compressed(self, loader, temp_dir):
        """V3 format: 3 columns (gene_id, gene_name, feature_type) compressed."""
        features_file = temp_dir / "features.tsv.gz"
        content = "ENSG00000000001\tGeneA\tGene Expression\nENSG00000000002\tGeneB\tGene Expression\n"
        with gzip.open(features_file, "wt") as f:
            f.write(content)

        result = loader.detect_features_format(features_file)
        assert result == "standard_10x", f"3-column format should be 'standard_10x', got {result}"

    def test_standard_10x_3_columns_uncompressed(self, loader, temp_dir):
        """V3 format: 3 columns uncompressed."""
        features_file = temp_dir / "features.tsv"
        features_file.write_text("ENSG00000000001\tGeneA\tGene Expression\n")

        result = loader.detect_features_format(features_file)
        assert result == "standard_10x"

    # --- V2 Format Tests (2 columns) ---

    def test_standard_10x_2_columns_compressed(self, loader, temp_dir):
        """V2 format: 2 columns (gene_id, gene_name) compressed."""
        genes_file = temp_dir / "genes.tsv.gz"
        content = "ENSG00000000001\tGeneA\nENSG00000000002\tGeneB\n"
        with gzip.open(genes_file, "wt") as f:
            f.write(content)

        result = loader.detect_features_format(genes_file)
        assert result == "standard_10x", f"2-column V2 format should be 'standard_10x', got {result}"

    def test_standard_10x_2_columns_uncompressed(self, loader, temp_dir):
        """V2 format: 2 columns uncompressed."""
        genes_file = temp_dir / "genes.tsv"
        genes_file.write_text("ENSG00000000001\tGeneA\n")

        result = loader.detect_features_format(genes_file)
        assert result == "standard_10x"

    # --- Single-Column Edge Cases (non-standard) ---

    def test_symbols_only_1_column(self, loader, temp_dir):
        """Non-standard: 1 column with gene symbols."""
        features_file = temp_dir / "genes.txt.gz"
        content = "BRCA1\nTP53\nEGFR\n"
        with gzip.open(features_file, "wt") as f:
            f.write(content)

        result = loader.detect_features_format(features_file)
        assert result == "symbols_only", f"1-column symbols should be 'symbols_only', got {result}"

    def test_ids_only_ensembl(self, loader, temp_dir):
        """Non-standard: 1 column with Ensembl IDs."""
        features_file = temp_dir / "genes.txt.gz"
        content = "ENSG00000000001\nENSG00000000002\nENSG00000000003\n"
        with gzip.open(features_file, "wt") as f:
            f.write(content)

        result = loader.detect_features_format(features_file)
        assert result == "ids_only", f"1-column Ensembl IDs should be 'ids_only', got {result}"

    def test_ids_only_mouse_ensembl(self, loader, temp_dir):
        """Non-standard: 1 column with mouse Ensembl IDs (ENSMUSG prefix)."""
        features_file = temp_dir / "genes.txt.gz"
        content = "ENSMUSG00000000001\nENSMUSG00000000002\n"
        with gzip.open(features_file, "wt") as f:
            f.write(content)

        result = loader.detect_features_format(features_file)
        assert result == "ids_only", "Mouse Ensembl IDs should be 'ids_only'"

    def test_uncompressed_single_column(self, loader, temp_dir):
        """Single-column file uncompressed."""
        features_file = temp_dir / "genes.tsv"
        features_file.write_text("BRCA1\nTP53\n")

        result = loader.detect_features_format(features_file)
        assert result == "symbols_only"

    # --- Error Handling ---

    def test_empty_file_fallback(self, loader, temp_dir):
        """Empty file should fallback gracefully."""
        features_file = temp_dir / "genes.tsv.gz"
        with gzip.open(features_file, "wt") as f:
            f.write("")

        # Should not crash, returns safe fallback
        result = loader.detect_features_format(features_file)
        assert result in ["symbols_only", "standard_10x"]  # Fallback behavior


class TestTenXManualLoader:
    """Tests for load_10x_manual() fallback for non-standard formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, temp_dir):
        """Create loader with mock downloader."""
        mock_downloader = Mock()
        return TenXGenomicsLoader(mock_downloader, cache_dir=temp_dir)

    @pytest.fixture
    def single_column_10x_data(self, temp_dir):
        """
        Create 10X directory with single-column genes file (non-standard format).

        This simulates datasets like GSE182227 which have:
        - matrix.mtx.gz (standard)
        - barcodes.tsv.gz (standard)
        - genes.txt.gz (NON-STANDARD: single column with symbols only)
        """
        n_genes = 5
        n_cells = 3

        # Create sparse matrix (genes x cells - 10X standard orientation)
        np.random.seed(42)
        data = np.random.poisson(3, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        # Matrix file
        matrix_path = temp_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wb") as f:
            mmwrite(f, sparse_matrix)

        # Single-column genes file (non-standard)
        genes_content = "BRCA1\nTP53\nEGFR\nMYC\nKRAS"
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write(genes_content)

        # Barcodes
        barcodes_content = "AAAA-1\nBBBB-1\nCCCC-1"
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        return {
            "temp_dir": temp_dir,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "gene_names": ["BRCA1", "TP53", "EGFR", "MYC", "KRAS"],
        }

    @pytest.fixture
    def v2_standard_10x_data(self, temp_dir):
        """
        Create standard V2 10X directory (2-column genes.tsv).

        This simulates standard V2 datasets with:
        - matrix.mtx.gz
        - genes.tsv.gz (2 columns: gene_id, gene_name)
        - barcodes.tsv.gz
        """
        n_genes = 4
        n_cells = 2

        np.random.seed(123)
        data = np.random.poisson(5, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        # Matrix file
        matrix_path = temp_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wb") as f:
            mmwrite(f, sparse_matrix)

        # V2 genes file (2 columns)
        genes_content = "ENSG00000139618\tBRCA2\nENSG00000141510\tTP53\nENSG00000146648\tEGFR\nENSG00000136997\tMYC"
        with gzip.open(temp_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_content)

        # Barcodes
        barcodes_content = "AAAA-1\nBBBB-1"
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        return {
            "temp_dir": temp_dir,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "gene_names": ["BRCA2", "TP53", "EGFR", "MYC"],
            "gene_ids": ["ENSG00000139618", "ENSG00000141510", "ENSG00000146648", "ENSG00000136997"],
        }

    # --- Manual Loader Tests ---

    def test_manual_loader_single_column_symbols(self, loader, single_column_10x_data):
        """Manual loader handles single-column gene symbols file."""
        data = single_column_10x_data
        adata = loader.load_10x_manual(
            data["temp_dir"],
            features_format="symbols_only",
            gse_id="GSE_TEST",
        )

        assert adata is not None, "Manual parser should return AnnData"
        assert adata.n_obs == data["n_cells"], f"Expected {data['n_cells']} cells, got {adata.n_obs}"
        assert adata.n_vars == data["n_genes"], f"Expected {data['n_genes']} genes, got {adata.n_vars}"

        # Verify gene names were extracted correctly
        for gene in data["gene_names"]:
            assert gene in adata.var_names, f"Gene {gene} should be in var_names"

    def test_manual_loader_v2_standard_format(self, loader, v2_standard_10x_data):
        """Manual loader handles V2 standard 2-column genes file."""
        data = v2_standard_10x_data
        adata = loader.load_10x_manual(
            data["temp_dir"],
            features_format="standard_10x",
            gse_id="GSE_TEST",
        )

        assert adata is not None, "Manual parser should return AnnData"
        assert adata.n_obs == data["n_cells"], f"Expected {data['n_cells']} cells"
        assert adata.n_vars == data["n_genes"], f"Expected {data['n_genes']} genes"

        # V2 format: gene_ids stored in var['gene_ids'], var_names = gene_names
        for gene in data["gene_names"]:
            assert gene in adata.var_names, f"Gene name {gene} should be in var_names"

        # Verify gene_ids column exists
        assert "gene_ids" in adata.var.columns, "var should have 'gene_ids' column"
        for gene_id in data["gene_ids"]:
            assert gene_id in adata.var["gene_ids"].values, f"Gene ID {gene_id} should be in var['gene_ids']"

    def test_manual_loader_ids_only_format(self, temp_dir, loader):
        """Manual loader handles single-column Ensembl IDs file."""
        n_genes = 3
        n_cells = 2

        # Create data
        np.random.seed(456)
        data = np.random.poisson(2, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # Single-column Ensembl IDs
        genes_content = "ENSG00000141510\nENSG00000146648\nENSG00000136997"
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write(genes_content)

        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write("CELL_A\nCELL_B")

        adata = loader.load_10x_manual(temp_dir, features_format="ids_only", gse_id="GSE_TEST")

        assert adata.n_obs == n_cells
        assert adata.n_vars == n_genes
        # For ids_only, var_names should contain the Ensembl IDs
        assert "ENSG00000141510" in adata.var_names

    # --- Dimension Validation Tests ---

    def test_dimension_mismatch_barcodes_raises_error(self, temp_dir, loader):
        """Dimension mismatch between matrix and barcodes should raise error."""
        n_genes = 3
        n_cells_matrix = 5  # Matrix has 5 cells
        n_cells_barcodes = 3  # But only 3 barcodes

        data = np.random.poisson(2, size=(n_genes, n_cells_matrix))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write("GENE1\nGENE2\nGENE3")

        # Wrong number of barcodes
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write("CELL1\nCELL2\nCELL3")  # Only 3, but matrix has 5 cells

        with pytest.raises(ValueError, match="Dimension mismatch"):
            loader.load_10x_manual(temp_dir, features_format="symbols_only", gse_id="GSE_TEST")

    def test_dimension_mismatch_genes_raises_error(self, temp_dir, loader):
        """Dimension mismatch between matrix and genes should raise error."""
        n_genes_matrix = 5  # Matrix has 5 genes
        n_genes_file = 3  # But only 3 genes in file
        n_cells = 2

        data = np.random.poisson(2, size=(n_genes_matrix, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # Wrong number of genes
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write("GENE1\nGENE2\nGENE3")  # Only 3, but matrix has 5

        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write("CELL1\nCELL2")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            loader.load_10x_manual(temp_dir, features_format="symbols_only", gse_id="GSE_TEST")

    # --- Missing Files Tests ---

    def test_missing_matrix_raises_error(self, temp_dir, loader):
        """Missing matrix file should raise FileNotFoundError."""
        # Only barcodes and genes, no matrix
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write("GENE1\nGENE2")
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write("CELL1\nCELL2")

        with pytest.raises(FileNotFoundError, match="Could not find complete 10X trio"):
            loader.load_10x_manual(temp_dir, features_format="symbols_only", gse_id="GSE_TEST")

    def test_missing_barcodes_raises_error(self, temp_dir, loader):
        """Missing barcodes file should raise FileNotFoundError."""
        n_genes = 2
        n_cells = 2
        data = np.random.poisson(2, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write("GENE1\nGENE2")
        # No barcodes file

        with pytest.raises(FileNotFoundError, match="Could not find complete 10X trio"):
            loader.load_10x_manual(temp_dir, features_format="symbols_only", gse_id="GSE_TEST")


class TestTenXLoaderInitialization:
    """Tests for TenXGenomicsLoader initialization."""

    def test_requires_cache_dir(self):
        """TenXGenomicsLoader requires cache_dir parameter."""
        mock_downloader = Mock()

        with pytest.raises(ValueError, match="cache_dir is required"):
            TenXGenomicsLoader(mock_downloader, cache_dir=None)

    def test_accepts_valid_cache_dir(self):
        """TenXGenomicsLoader accepts valid cache_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_downloader = Mock()
            loader = TenXGenomicsLoader(mock_downloader, cache_dir=Path(tmpdir))
            assert loader.cache_dir == Path(tmpdir)


class TestTenXLoaderRegression:
    """Regression tests protecting against known bugs."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, temp_dir):
        mock_downloader = Mock()
        return TenXGenomicsLoader(mock_downloader, cache_dir=temp_dir)

    def test_zero_genes_regression_v2_format(self, temp_dir, loader):
        """
        REGRESSION TEST: V2 format (2-column genes.tsv) MUST NOT result in 0 genes.

        This prevents the bug where V2 datasets were misdetected or failed to load,
        resulting in AnnData with n_vars = 0.
        """
        n_genes = 10
        n_cells = 5

        np.random.seed(789)
        data = np.random.poisson(4, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # V2 format: 2 columns
        genes_content = "\n".join([f"ENSG{i:010d}\tGene{i}" for i in range(n_genes)])
        with gzip.open(temp_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_content)

        barcodes_content = "\n".join([f"CELL_{i}" for i in range(n_cells)])
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        adata = loader.load_10x_manual(temp_dir, features_format="standard_10x", gse_id="GSE_REGRESSION")

        # CRITICAL ASSERTIONS
        assert adata.n_vars > 0, f"REGRESSION BUG: V2 format resulted in {adata.n_vars} genes!"
        assert adata.n_vars == n_genes, f"Expected {n_genes} genes, got {adata.n_vars}"
        assert adata.n_obs == n_cells, f"Expected {n_cells} cells, got {adata.n_obs}"

    def test_zero_genes_regression_single_column(self, temp_dir, loader):
        """
        REGRESSION TEST: Single-column genes file MUST NOT result in 0 genes.

        This prevents the bug where non-standard single-column gene files
        (like GSE182227) failed to load properly.
        """
        n_genes = 8
        n_cells = 4

        np.random.seed(321)
        data = np.random.poisson(3, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(temp_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # Single-column gene symbols (non-standard)
        genes = ["BRCA1", "BRCA2", "TP53", "EGFR", "MYC", "KRAS", "PTEN", "RB1"]
        with gzip.open(temp_dir / "genes.txt.gz", "wt") as f:
            f.write("\n".join(genes))

        barcodes_content = "\n".join([f"BC_{i}" for i in range(n_cells)])
        with gzip.open(temp_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        adata = loader.load_10x_manual(temp_dir, features_format="symbols_only", gse_id="GSE182227_TEST")

        # CRITICAL ASSERTIONS
        assert adata.n_vars > 0, f"REGRESSION BUG: Single-column genes resulted in {adata.n_vars} genes!"
        assert adata.n_vars == n_genes
        assert adata.n_obs == n_cells

        # Verify gene names are accessible
        for gene in genes:
            assert gene in adata.var_names, f"Gene {gene} should be in var_names"


if __name__ == "__main__":
    """Run tests with verbose output."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
