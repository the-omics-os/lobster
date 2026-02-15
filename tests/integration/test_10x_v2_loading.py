"""
Integration tests for 10X Genomics V2 format loading.

Tests verify:
1. V2 format (genes.tsv) loads successfully via scanpy and manual fallback
2. Gene counts are correct (n_vars > 0) - CRITICAL: prevents zero-genes bug
3. Cell counts are correct (n_obs > 0)
4. Manual parser fallback works for V2 format
5. Both compressed (.gz) and uncompressed formats work

REGRESSION TEST: These tests specifically prevent the claimed bug where
V2 datasets (genes.tsv) would be misdetected and result in 0 genes loaded.
Investigation proved the code is correct, but these tests ensure it stays correct.
"""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2


class Test10XV2Loading:
    """Integration tests for 10X V2 format (genes.tsv)."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def v2_10x_archive_compressed(self, temp_workspace):
        """
        Create realistic V2 10X archive structure (compressed).

        V2 format uses genes.tsv (2 columns: gene_id, gene_name)
        NOT features.tsv (3 columns: gene_id, gene_name, feature_type)

        Directory structure mimics real GEO datasets like GSE155698:
            filtered_gene_bc_matrices/
            └── GRCh38/
                ├── matrix.mtx.gz
                ├── genes.tsv.gz      <-- V2 format (2 columns)
                └── barcodes.tsv.gz
        """
        # V2 uses filtered_gene_bc_matrices (not filtered_feature_bc_matrix)
        mtx_dir = temp_workspace / "filtered_gene_bc_matrices" / "GRCh38"
        mtx_dir.mkdir(parents=True)

        # Create sparse matrix (5 cells × 10 genes)
        n_cells = 5
        n_genes = 10
        np.random.seed(42)
        data = np.random.poisson(5, size=(n_genes, n_cells))  # genes × cells
        sparse_matrix = csr_matrix(data)

        # Write matrix.mtx.gz
        matrix_path = mtx_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wb") as f:
            mmwrite(f, sparse_matrix)

        # Write genes.tsv.gz (V2 format - 2 columns: gene_id, gene_name)
        # NOTE: V3 uses features.tsv with 3 columns including feature_type
        genes_content = "\n".join([f"ENSG0000000{i}\tGene{i}" for i in range(n_genes)])
        with gzip.open(mtx_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_content)

        # Write barcodes.tsv.gz
        barcodes_content = "\n".join([f"AAACCTGA-{i}" for i in range(n_cells)])
        with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        return {
            "root_dir": temp_workspace,
            "mtx_dir": mtx_dir,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "format": "V2",
            "gene_file": "genes.tsv.gz",
        }

    @pytest.fixture
    def v2_10x_archive_uncompressed(self, temp_workspace):
        """
        Create V2 10X archive structure (uncompressed).

        Tests that uncompressed V2 files are handled correctly.
        """
        mtx_dir = temp_workspace / "filtered_gene_bc_matrices" / "hg19"
        mtx_dir.mkdir(parents=True)

        n_cells = 8
        n_genes = 15
        np.random.seed(123)
        data = np.random.poisson(3, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        # Write uncompressed files
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), sparse_matrix)

        # genes.tsv (V2 format - 2 columns)
        genes_content = "\n".join([f"ENSG{i:010d}\tTP53_{i}" for i in range(n_genes)])
        (mtx_dir / "genes.tsv").write_text(genes_content)

        # barcodes.tsv
        barcodes_content = "\n".join([f"BARCODE_{i}" for i in range(n_cells)])
        (mtx_dir / "barcodes.tsv").write_text(barcodes_content)

        return {
            "root_dir": temp_workspace,
            "mtx_dir": mtx_dir,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "format": "V2",
            "gene_file": "genes.tsv",
        }

    @pytest.fixture
    def v3_10x_archive_for_comparison(self, temp_workspace):
        """
        Create V3 10X archive structure for baseline comparison.

        V3 format uses features.tsv (3 columns: gene_id, gene_name, feature_type)
        """
        mtx_dir = temp_workspace / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(parents=True)

        n_cells = 5
        n_genes = 10
        np.random.seed(42)
        data = np.random.poisson(5, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        matrix_path = mtx_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wb") as f:
            mmwrite(f, sparse_matrix)

        # features.tsv.gz (V3 format - 3 columns)
        features_content = "\n".join(
            [f"ENSG0000000{i}\tGene{i}\tGene Expression" for i in range(n_genes)]
        )
        with gzip.open(mtx_dir / "features.tsv.gz", "wt") as f:
            f.write(features_content)

        barcodes_content = "\n".join([f"AAACCTGA-{i}" for i in range(n_cells)])
        with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        return {
            "root_dir": temp_workspace,
            "mtx_dir": mtx_dir,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "format": "V3",
            "gene_file": "features.tsv.gz",
        }

    # === REGRESSION TESTS: Zero-Genes Bug Prevention ===

    def test_v2_format_loads_with_genes_compressed(
        self, temp_workspace, v2_10x_archive_compressed
    ):
        """
        REGRESSION TEST: V2 format (genes.tsv.gz) MUST NOT result in 0 genes.

        This is the critical test that prevents the claimed bug.
        The investigation proved the code is correct, but this test ensures
        it stays correct in future changes.
        """
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        root_dir = v2_10x_archive_compressed["root_dir"]
        expected_cells = v2_10x_archive_compressed["n_cells"]
        expected_genes = v2_10x_archive_compressed["n_genes"]

        result = client._load_10x_from_directory(root_dir, "v2_test_compressed")

        # Critical assertion: loading must succeed
        assert result["success"], f"V2 loading failed: {result.get('error')}"

        # Critical regression check: genes must NOT be zero
        actual_genes = result["data_shape"][1]
        assert actual_genes > 0, (
            f"REGRESSION BUG: V2 format resulted in {actual_genes} genes! "
            "This is the bug we're preventing."
        )

        # Verify correct dimensions
        assert (
            result["data_shape"][0] == expected_cells
        ), f"Expected {expected_cells} cells, got {result['data_shape'][0]}"
        assert (
            actual_genes == expected_genes
        ), f"Expected {expected_genes} genes, got {actual_genes}"

        print(
            f"✓ V2 compressed format loaded: {result['data_shape'][0]} cells × {actual_genes} genes"
        )
        print(f"✓ Loading method: {result.get('loading_method', 'unknown')}")

    def test_v2_format_loads_with_genes_uncompressed(
        self, temp_workspace, v2_10x_archive_uncompressed
    ):
        """
        REGRESSION TEST: V2 format (genes.tsv) uncompressed MUST NOT result in 0 genes.
        """
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        root_dir = v2_10x_archive_uncompressed["root_dir"]
        expected_cells = v2_10x_archive_uncompressed["n_cells"]
        expected_genes = v2_10x_archive_uncompressed["n_genes"]

        result = client._load_10x_from_directory(root_dir, "v2_test_uncompressed")

        assert result[
            "success"
        ], f"V2 uncompressed loading failed: {result.get('error')}"

        actual_genes = result["data_shape"][1]
        assert (
            actual_genes > 0
        ), f"REGRESSION BUG: V2 uncompressed format resulted in {actual_genes} genes!"

        assert result["data_shape"][0] == expected_cells
        assert actual_genes == expected_genes

        print(
            f"✓ V2 uncompressed format loaded: {result['data_shape'][0]} cells × {actual_genes} genes"
        )

    # === FORMAT PARITY TESTS ===

    def test_v2_v3_format_parity(
        self,
        temp_workspace,
        v2_10x_archive_compressed,
        v3_10x_archive_for_comparison,
    ):
        """
        Test that V2 and V3 formats produce equivalent results.

        Both formats should:
        1. Load successfully
        2. Have non-zero genes
        3. Produce valid AnnData objects
        """
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        # Load V2 format
        v2_result = client._load_10x_from_directory(
            v2_10x_archive_compressed["root_dir"], "v2_parity_test"
        )

        # Reset workspace for V3 test
        with tempfile.TemporaryDirectory() as tmpdir2:
            data_manager2 = DataManagerV2(workspace_path=Path(tmpdir2))
            client2 = AgentClient(
                data_manager=data_manager2, workspace_path=Path(tmpdir2)
            )

            # Load V3 format
            v3_result = client2._load_10x_from_directory(
                v3_10x_archive_for_comparison["root_dir"], "v3_parity_test"
            )

        # Both must succeed
        assert v2_result["success"], f"V2 format failed: {v2_result.get('error')}"
        assert v3_result["success"], f"V3 format failed: {v3_result.get('error')}"

        # Both must have non-zero genes
        assert v2_result["data_shape"][1] > 0, "V2 resulted in 0 genes"
        assert v3_result["data_shape"][1] > 0, "V3 resulted in 0 genes"

        print(
            f"✓ V2 format: {v2_result['data_shape'][0]} × {v2_result['data_shape'][1]}"
        )
        print(
            f"✓ V3 format: {v3_result['data_shape'][0]} × {v3_result['data_shape'][1]}"
        )

    # === MANUAL PARSER FALLBACK TESTS ===

    def test_v2_manual_parser_direct(self, temp_workspace, v2_10x_archive_compressed):
        """Test that manual parser correctly handles V2 genes.tsv format."""
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        root_dir = v2_10x_archive_compressed["root_dir"]
        expected_cells = v2_10x_archive_compressed["n_cells"]
        expected_genes = v2_10x_archive_compressed["n_genes"]

        # Call manual parser directly (bypasses scanpy)
        df = client._manual_parse_10x(root_dir, "v2_manual_test")

        # Validate result
        assert df is not None, "Manual parser returned None for V2 format"
        assert (
            df.shape[0] == expected_cells
        ), f"Expected {expected_cells} cells, got {df.shape[0]}"
        assert (
            df.shape[1] == expected_genes
        ), f"Expected {expected_genes} genes, got {df.shape[1]}"

        # Verify gene names were extracted from 2-column format
        assert "Gene0" in df.columns or any(
            "Gene" in str(c) for c in df.columns
        ), "Gene names should be extracted from V2 genes.tsv format"

        print(f"✓ Manual parser V2: {df.shape[0]} cells × {df.shape[1]} genes")

    def test_v2_manual_parser_handles_2_column_format(self, temp_workspace):
        """
        Test that manual parser correctly parses 2-column genes.tsv.

        V2 format: gene_id<TAB>gene_name
        V3 format: gene_id<TAB>gene_name<TAB>feature_type

        The parser must handle both cases.
        """
        mtx_dir = temp_workspace / "test_2col" / "mtx"
        mtx_dir.mkdir(parents=True)

        n_cells = 3
        n_genes = 5

        # Matrix
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        sparse_matrix = csr_matrix(data)
        with gzip.open(mtx_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # 2-column genes file (V2 format)
        genes_2col = (
            "ENSG001\tBRCA1\nENSG002\tTP53\nENSG003\tEGFR\nENSG004\tMYC\nENSG005\tKRAS"
        )
        with gzip.open(mtx_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_2col)

        # Barcodes
        barcodes = "CELL_A\nCELL_B\nCELL_C"
        with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes)

        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        df = client._manual_parse_10x(temp_workspace / "test_2col", "test_2col")

        assert df is not None, "Parser should handle 2-column genes file"
        assert df.shape[1] == n_genes, f"Expected {n_genes} genes"

        # Check gene names extracted correctly
        gene_names = set(df.columns)
        assert "BRCA1" in gene_names or any(
            "BRCA" in str(c) for c in gene_names
        ), "Gene symbol BRCA1 should be extracted from 2-column format"

        print(f"✓ 2-column format parsed: {df.shape[0]} × {df.shape[1]}")

    # === NESTED DIRECTORY STRUCTURE TESTS ===

    def test_v2_nested_geo_structure(self, temp_workspace):
        """
        Test V2 format in nested GEO directory structure.

        Common GEO pattern:
            GSM123456_sample/
            └── filtered_gene_bc_matrices/
                └── GRCh38/
                    ├── matrix.mtx.gz
                    ├── genes.tsv.gz
                    └── barcodes.tsv.gz
        """
        # Create nested structure
        nested_dir = (
            temp_workspace / "GSM123456_sample" / "filtered_gene_bc_matrices" / "GRCh38"
        )
        nested_dir.mkdir(parents=True)

        n_cells = 10
        n_genes = 20
        np.random.seed(99)
        data = np.random.poisson(2, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        with gzip.open(nested_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # V2 genes file
        genes_content = "\n".join([f"ENSG{i:06d}\tGENE_{i}" for i in range(n_genes)])
        with gzip.open(nested_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_content)

        barcodes_content = "\n".join([f"BC_{i}" for i in range(n_cells)])
        with gzip.open(nested_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes_content)

        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        # Load from parent directory (should find nested 10X files)
        result = client._load_10x_from_directory(
            temp_workspace / "GSM123456_sample", "GSM123456_sample"
        )

        assert result["success"], f"Nested V2 loading failed: {result.get('error')}"
        assert result["data_shape"][1] > 0, "Nested V2 resulted in 0 genes"
        assert result["data_shape"][0] == n_cells
        assert result["data_shape"][1] == n_genes

        print(
            f"✓ Nested V2 structure: {result['data_shape'][0]} × {result['data_shape'][1]}"
        )

    # === EDGE CASE TESTS ===

    def test_v2_single_column_fallback(self, temp_workspace):
        """
        Test handling of non-standard single-column genes file.

        Some older datasets have only gene symbols (no gene IDs).
        The parser should still work.
        """
        mtx_dir = temp_workspace / "single_col_test"
        mtx_dir.mkdir(parents=True)

        n_cells = 4
        n_genes = 6

        data = np.random.poisson(3, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)
        with gzip.open(mtx_dir / "matrix.mtx.gz", "wb") as f:
            mmwrite(f, sparse_matrix)

        # Single-column genes file (edge case)
        genes_1col = "BRCA1\nTP53\nEGFR\nMYC\nKRAS\nPTEN"
        with gzip.open(mtx_dir / "genes.tsv.gz", "wt") as f:
            f.write(genes_1col)

        barcodes = "\n".join([f"CELL_{i}" for i in range(n_cells)])
        with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as f:
            f.write(barcodes)

        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        df = client._manual_parse_10x(mtx_dir, "single_col")

        assert df is not None, "Parser should handle single-column genes"
        assert df.shape[1] == n_genes, f"Expected {n_genes} genes"

        print(f"✓ Single-column genes: {df.shape[0]} × {df.shape[1]}")

    def test_modality_stored_correctly(self, temp_workspace, v2_10x_archive_compressed):
        """
        Test that loaded V2 data is correctly stored in DataManagerV2.

        Verifies the full pipeline from loading to modality storage.
        """
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        root_dir = v2_10x_archive_compressed["root_dir"]
        expected_genes = v2_10x_archive_compressed["n_genes"]

        result = client._load_10x_from_directory(root_dir, "storage_test")
        assert result["success"]

        # Verify modality stored
        modality_name = result["modality_name"]
        assert (
            modality_name in data_manager.list_modalities()
        ), "Modality should be stored in DataManagerV2"

        # Retrieve and verify AnnData
        adata = data_manager.get_modality(modality_name)
        assert (
            adata.n_vars == expected_genes
        ), f"AnnData should have {expected_genes} genes"
        assert adata.n_vars > 0, "CRITICAL: AnnData must NOT have 0 genes"

        print(f"✓ Modality '{modality_name}' stored: {adata.n_obs} × {adata.n_vars}")


if __name__ == "__main__":
    """Run tests with verbose output."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
