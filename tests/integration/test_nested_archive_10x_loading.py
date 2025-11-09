"""
Integration test for nested archive 10X loading with robust fallback.

Tests the fix for the 0 genes bug where PDAC_PBMC samples loaded with 0 vars.
"""

import tempfile
from pathlib import Path
import pytest
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2


class TestNestedArchive10XLoading:
    """Test two-tier 10X loading strategy with scanpy and manual parsing fallback."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_10x_data_nested(self, temp_workspace):
        """
        Create mock 10X data with nested structure that causes scanpy to fail.

        Simulates structure:
            PDAC_PBMC_1/
            └── filtered_feature_bc_matrix/
                ├── matrix.mtx.gz
                ├── features.tsv.gz
                └── barcodes.tsv.gz
        """
        # Create nested directory structure
        sample_dir = temp_workspace / "PDAC_PBMC_1"
        matrix_dir = sample_dir / "filtered_feature_bc_matrix"
        matrix_dir.mkdir(parents=True)

        # Generate synthetic expression data
        n_cells = 1000
        n_genes = 500

        # Create sparse matrix (genes × cells format - 10X standard)
        np.random.seed(42)
        data = np.random.poisson(5, size=(n_genes, n_cells))
        sparse_matrix = csr_matrix(data)

        # Write matrix.mtx.gz
        matrix_file = matrix_dir / "matrix.mtx.gz"
        with gzip.open(matrix_file, "wb") as f:
            mmwrite(f, sparse_matrix)

        # Write barcodes.tsv.gz
        barcodes_file = matrix_dir / "barcodes.tsv.gz"
        barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]
        with gzip.open(barcodes_file, "wt") as f:
            f.write("\n".join(barcodes))

        # Write features.tsv.gz (gene_id\tgene_name\tfeature_type)
        features_file = matrix_dir / "features.tsv.gz"
        genes = [f"ENSG{i:010d}\tGENE_{i}\tGene Expression" for i in range(n_genes)]
        with gzip.open(features_file, "wt") as f:
            f.write("\n".join(genes))

        return {
            "sample_dir": sample_dir,
            "matrix_dir": matrix_dir,
            "expected_shape": (n_cells, n_genes),
        }

    def test_manual_parse_10x_nested_structure(self, temp_workspace, mock_10x_data_nested):
        """Test that manual parsing handles nested 10X structure correctly."""
        # Setup
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        sample_dir = mock_10x_data_nested["sample_dir"]
        expected_cells, expected_genes = mock_10x_data_nested["expected_shape"]

        # Act - call manual parsing directly
        df = client._manual_parse_10x(sample_dir, "PDAC_PBMC_1")

        # Assert
        assert df is not None, "Manual parsing should return a DataFrame"
        assert df.shape[0] == expected_cells, f"Expected {expected_cells} cells, got {df.shape[0]}"
        assert df.shape[1] == expected_genes, f"Expected {expected_genes} genes, got {df.shape[1]}"

        # Verify cell IDs have sample prefix
        assert df.index[0].startswith("PDAC_PBMC_1_"), "Cell IDs should have sample prefix"

        # Verify gene names extracted
        assert "GENE_0" in df.columns, "Gene names should be extracted from features file"

        print(f"✓ Manual parsing successful: {df.shape[0]:,} cells × {df.shape[1]:,} genes")

    def test_load_10x_from_directory_with_fallback(self, temp_workspace, mock_10x_data_nested):
        """Test that _load_10x_from_directory uses fallback when scanpy fails."""
        # Setup
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        sample_dir = mock_10x_data_nested["sample_dir"]
        expected_cells, expected_genes = mock_10x_data_nested["expected_shape"]

        # Act
        result = client._load_10x_from_directory(sample_dir, "PDAC_PBMC_1")

        # Assert
        assert result["success"] is True, f"Loading should succeed: {result.get('error', '')}"
        assert result["data_shape"][0] == expected_cells, f"Expected {expected_cells} cells"
        assert result["data_shape"][1] == expected_genes, f"Expected {expected_genes} genes"
        assert result["data_shape"][1] > 0, "CRITICAL: Should NOT have 0 genes!"

        # Verify loading method reported
        assert "loading_method" in result, "Should report which loading method was used"
        print(f"✓ Loading method used: {result['loading_method']}")
        print(f"✓ Data shape: {result['data_shape'][0]:,} cells × {result['data_shape'][1]:,} genes")

        # Verify modality stored in DataManager
        modality_name = result["modality_name"]
        assert modality_name in data_manager.list_modalities(), "Modality should be stored"

        adata = data_manager.get_modality(modality_name)
        assert adata.n_obs == expected_cells, "AnnData should have correct cell count"
        assert adata.n_vars == expected_genes, "AnnData should have correct gene count"
        assert adata.n_vars > 0, "CRITICAL BUG CHECK: AnnData must not have 0 genes!"

    def test_load_multiple_samples_with_auto_concatenation(self, temp_workspace):
        """Test loading multiple nested samples and auto-concatenation."""
        # Setup
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        # Create 3 mock samples
        samples = []
        for i in range(1, 4):
            sample_dir = temp_workspace / f"PDAC_PBMC_{i}"
            matrix_dir = sample_dir / "filtered_feature_bc_matrix"
            matrix_dir.mkdir(parents=True)

            # Small synthetic data
            n_cells = 100
            n_genes = 50

            # Matrix
            data = np.random.poisson(3, size=(n_genes, n_cells))
            sparse_matrix = csr_matrix(data)
            matrix_file = matrix_dir / "matrix.mtx.gz"
            with gzip.open(matrix_file, "wb") as f:
                mmwrite(f, sparse_matrix)

            # Barcodes
            barcodes = [f"CELL_{j:04d}" for j in range(n_cells)]
            barcodes_file = matrix_dir / "barcodes.tsv.gz"
            with gzip.open(barcodes_file, "wt") as f:
                f.write("\n".join(barcodes))

            # Features
            genes = [f"ENSG{j:010d}\tGENE_{j}\tGene Expression" for j in range(n_genes)]
            features_file = matrix_dir / "features.tsv.gz"
            with gzip.open(features_file, "wt") as f:
                f.write("\n".join(genes))

            samples.append({
                "dir": sample_dir,
                "name": f"PDAC_PBMC_{i}",
                "cells": n_cells,
                "genes": n_genes,
            })

        # Act - load all samples
        loaded_modalities = []
        for sample in samples:
            result = client._load_10x_from_directory(sample["dir"], sample["name"])
            assert result["success"] is True, f"Sample {sample['name']} should load"
            assert result["data_shape"][1] > 0, f"Sample {sample['name']} should have genes!"
            loaded_modalities.append(result["modality_name"])

        # Verify all samples loaded correctly
        assert len(loaded_modalities) == 3, "Should load 3 samples"

        total_cells = sum(s["cells"] for s in samples)
        print(f"✓ Loaded {len(loaded_modalities)} samples with total {total_cells} cells")

        # Verify each sample has genes
        for modality_name in loaded_modalities:
            adata = data_manager.get_modality(modality_name)
            assert adata.n_vars > 0, f"{modality_name} should have genes (not 0)!"

    def test_error_handling_missing_matrix(self, temp_workspace):
        """Test error handling when matrix file is missing."""
        # Setup
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        # Create directory with NO matrix file
        sample_dir = temp_workspace / "INCOMPLETE_SAMPLE"
        sample_dir.mkdir()

        # Act
        result = client._load_10x_from_directory(sample_dir, "INCOMPLETE_SAMPLE")

        # Assert
        assert result["success"] is False, "Should fail when matrix file missing"
        assert "matrix.mtx" in result["error"].lower(), "Error should mention missing matrix"

    def test_error_handling_corrupted_matrix(self, temp_workspace):
        """Test error handling when matrix file is corrupted."""
        # Setup
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        client = AgentClient(data_manager=data_manager, workspace_path=temp_workspace)

        # Create directory with corrupted matrix
        sample_dir = temp_workspace / "CORRUPTED_SAMPLE"
        matrix_dir = sample_dir / "filtered_feature_bc_matrix"
        matrix_dir.mkdir(parents=True)

        # Write corrupted matrix file
        matrix_file = matrix_dir / "matrix.mtx.gz"
        with gzip.open(matrix_file, "wt") as f:
            f.write("NOT A VALID MATRIX FILE")

        # Act
        result = client._load_10x_from_directory(sample_dir, "CORRUPTED_SAMPLE")

        # Assert
        assert result["success"] is False, "Should fail when matrix is corrupted"
        print(f"✓ Error handling working: {result['error']}")


if __name__ == "__main__":
    """Run tests with verbose output."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
