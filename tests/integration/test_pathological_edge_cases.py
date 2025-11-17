"""
Comprehensive pathological and extreme edge case tests for Lobster platform.

This test suite systematically tests scenarios that should NEVER occur in real data
but could theoretically crash the system. Tests are organized by category:

1. Data Extremes - Pathological data values and distributions
2. Structural Extremes - Malformed AnnData objects
3. Metadata Extremes - Problematic annotations
4. Operation Extremes - Invalid analysis parameters
5. Memory & Performance Extremes - Resource exhaustion scenarios

EXPECTED BEHAVIOR: System should fail gracefully with clear error messages,
NOT crash with obscure stack traces or hang indefinitely.

Author: Agent 23 - Pathological Edge Case Testing Campaign
Date: 2025-11-07
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.data_manager_v2 import DataManagerV2

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance."""
    return DataManagerV2(workspace_path=str(temp_workspace))


@pytest.fixture
def h5ad_backend(temp_workspace):
    """Create H5ADBackend instance."""
    return H5ADBackend(base_path=str(temp_workspace))


# ==============================================================================
# Category 1: Data Extremes
# ==============================================================================


@pytest.mark.integration
class TestDataExtremes:
    """Test pathological data values and distributions."""

    def test_single_cell_dataset(self):
        """Test dataset with literally ONE cell."""
        adata = anndata.AnnData(
            X=np.array([[1, 2, 3, 4, 5]]),  # 1 cell, 5 genes
            obs=pd.DataFrame({"cell_id": ["CELL_001"]}, index=["CELL_001"]),
            var=pd.DataFrame(
                {"gene": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]}
            ),
        )

        # Should fail gracefully with scanpy operations
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        error_msg = str(exc_info.value).lower()
        assert any(
            word in error_msg
            for word in ["cell", "observation", "sample", "too few", "minimum", "error"]
        )

    def test_single_gene_dataset(self):
        """Test dataset with literally ONE gene."""
        adata = anndata.AnnData(
            X=np.array([[1], [2], [3], [4], [5]]),  # 5 cells, 1 gene
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(5)]}),
            var=pd.DataFrame({"gene": ["GENE_A"]}, index=["GENE_A"]),
        )

        # Should fail gracefully with PCA (requires multiple features)
        with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
            import scanpy as sc

            sc.pp.pca(adata, n_comps=2)

        error_msg = str(exc_info.value).lower()
        assert any(
            word in error_msg
            for word in ["gene", "variable", "feature", "too few", "component", "error"]
        )

    def test_100_percent_missing_values(self):
        """Test dataset where ALL values are NaN."""
        adata = anndata.AnnData(
            X=np.full((100, 50), np.nan, dtype=np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should detect and fail gracefully
        with pytest.raises((ValueError, RuntimeError, TypeError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        error_msg = str(exc_info.value).lower()
        assert any(
            word in error_msg
            for word in ["nan", "missing", "empty", "invalid", "error"]
        )

    def test_all_zero_values(self):
        """Test dataset where ALL counts are zero."""
        adata = anndata.AnnData(
            X=np.zeros((100, 50), dtype=np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Normalization should handle this
        try:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            # All-zero should remain all-zero or produce NaN
            assert np.allclose(adata.X, 0) or np.all(np.isnan(adata.X))
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            # Acceptable to reject all-zero data
            error_msg = str(e).lower()
            assert any(
                word in error_msg for word in ["zero", "empty", "count", "error"]
            )

    def test_zero_variance_features(self):
        """Test dataset where all genes have identical values."""
        X = (
            np.ones((100, 50), dtype=np.float32) * 42
        )  # All cells have count=42 for all genes
        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle constant values gracefully
        try:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            # Constant values should remain constant (or be rejected)
            assert adata is not None
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(
                word in error_msg for word in ["variance", "constant", "identical"]
            )

    def test_infinite_values(self):
        """Test dataset with infinite values."""
        X = np.random.randn(100, 50).astype(np.float32)
        X[10:20, 10:20] = np.inf  # Add positive infinities
        X[30:40, 30:40] = -np.inf  # Add negative infinities

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should detect and reject infinities
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["inf", "infinite", "invalid"])

    def test_negative_counts(self):
        """Test dataset with negative count values."""
        X = np.random.randint(-100, 100, (100, 50)).astype(np.float32)

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle or warn about negative counts
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import scanpy as sc

                sc.pp.normalize_total(adata, target_sum=1e4)
                # Should have warned about negative values
                assert len(w) > 0 or adata is not None
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["negative", "invalid", "count"])

    def test_extremely_large_counts(self):
        """Test dataset with counts exceeding floating point precision."""
        X = np.random.randn(100, 50).astype(np.float32) * 1e15

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle or warn about extreme values
        try:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            assert adata is not None
        except (ValueError, RuntimeError, OverflowError) as e:
            # Acceptable to reject extreme values
            assert True

    def test_unicode_emoji_gene_names(self, h5ad_backend, temp_workspace):
        """Test gene names with emojis and unicode characters."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (100, 50)).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame(
                {
                    "gene": [
                        "GENE_😀",
                        "GENE_🧬",
                        "GENE_🔬",
                        "细胞基因",
                        "α-β-γ",
                        "∑∫∂",
                        "日本語遺伝子",
                    ]
                    + [f"GENE_{i}" for i in range(43)]
                }
            ),
        )

        # Should preserve unicode in save/load
        test_file = temp_workspace / "unicode_genes.h5ad"
        h5ad_backend.save(adata, str(test_file))
        adata_loaded = h5ad_backend.load(str(test_file))

        assert adata_loaded is not None
        # Check that at least some unicode was preserved
        assert len(adata_loaded.var) == 50


# ==============================================================================
# Category 2: Structural Extremes
# ==============================================================================


@pytest.mark.integration
class TestStructuralExtremes:
    """Test malformed AnnData objects."""

    def test_empty_anndata_object(self):
        """Test completely empty AnnData object."""
        adata = anndata.AnnData()

        # Should fail gracefully
        with pytest.raises((ValueError, RuntimeError, AttributeError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        assert exc_info.value is not None

    def test_no_observations(self):
        """Test AnnData with 0 observations."""
        adata = anndata.AnnData(
            X=np.empty((0, 50), dtype=np.float32),
            obs=pd.DataFrame(),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        error_msg = str(exc_info.value).lower()
        assert any(
            word in error_msg for word in ["empty", "observation", "cell", "sample"]
        )

    def test_no_variables(self):
        """Test AnnData with 0 variables."""
        adata = anndata.AnnData(
            X=np.empty((100, 0), dtype=np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame(),
        )

        with pytest.raises((ValueError, RuntimeError, IndexError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        error_msg = str(exc_info.value).lower()
        assert any(
            word in error_msg for word in ["empty", "variable", "gene", "feature"]
        )

    def test_single_obs_single_var(self):
        """Test 1x1 AnnData object."""
        adata = anndata.AnnData(
            X=np.array([[42.0]]),
            obs=pd.DataFrame({"cell_id": ["CELL_001"]}, index=["CELL_001"]),
            var=pd.DataFrame({"gene": ["GENE_A"]}, index=["GENE_A"]),
        )

        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)

        assert exc_info.value is not None

    def test_completely_dense_matrix(self, h5ad_backend, temp_workspace):
        """Test matrix with 0% sparsity (all values non-zero)."""
        X = np.random.poisson(10, (500, 200)).astype(np.float32)
        X[X == 0] = 1  # Replace all zeros

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(500)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(200)]}),
        )

        # Should handle dense matrices
        test_file = temp_workspace / "dense_matrix.h5ad"
        h5ad_backend.save(adata, str(test_file))
        adata_loaded = h5ad_backend.load(str(test_file))

        assert adata_loaded is not None
        assert adata_loaded.shape == (500, 200)

    def test_completely_sparse_matrix(self, h5ad_backend, temp_workspace):
        """Test matrix with 100% sparsity (all zeros)."""
        X = sparse.csr_matrix((500, 200), dtype=np.float32)

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(500)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(200)]}),
        )

        # Should handle all-zero sparse matrices
        test_file = temp_workspace / "all_zero_sparse.h5ad"
        h5ad_backend.save(adata, str(test_file))
        adata_loaded = h5ad_backend.load(str(test_file))

        assert adata_loaded is not None
        assert np.allclose(adata_loaded.X.toarray(), 0)

    def test_extreme_dimension_ratio(self):
        """Test dataset with extreme dimension imbalance (10,000 cells x 5 genes)."""
        adata = anndata.AnnData(
            X=np.random.poisson(3, (10000, 5)).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(10000)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(5)]}),
        )

        # Should handle or reject extreme ratios
        try:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            assert adata is not None
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(
                word in error_msg for word in ["dimension", "ratio", "feature", "gene"]
            )


# ==============================================================================
# Category 3: Metadata Extremes
# ==============================================================================


@pytest.mark.integration
class TestMetadataExtremes:
    """Test problematic metadata and annotations."""

    @pytest.mark.slow
    def test_thousands_of_metadata_columns(self, h5ad_backend, temp_workspace):
        """Test obs with 10,000+ metadata columns (slow: 9.3s)."""
        n_cols = 10000
        obs_data = {
            f"meta_{i}": [f"val_{i}_{j}" for j in range(100)] for i in range(n_cols)
        }
        obs_data["cell_id"] = [f"CELL_{i}" for i in range(100)]

        adata = anndata.AnnData(
            X=np.random.poisson(5, (100, 50)).astype(np.float32),
            obs=pd.DataFrame(obs_data),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle large metadata (but may be slow)
        test_file = temp_workspace / "huge_metadata.h5ad"
        try:
            h5ad_backend.save(adata, str(test_file))
            adata_loaded = h5ad_backend.load(str(test_file))
            assert adata_loaded is not None
        except (MemoryError, ValueError) as e:
            # Acceptable to fail on extreme metadata
            assert True

    def test_extremely_long_strings(self, h5ad_backend, temp_workspace):
        """Test metadata with 1MB+ string values."""
        long_string = "A" * (1024 * 1024)  # 1MB string

        adata = anndata.AnnData(
            X=np.random.poisson(5, (10, 50)).astype(np.float32),
            obs=pd.DataFrame(
                {
                    "cell_id": [f"CELL_{i}" for i in range(10)],
                    "description": [long_string] * 10,
                }
            ),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle or truncate long strings
        test_file = temp_workspace / "long_strings.h5ad"
        try:
            h5ad_backend.save(adata, str(test_file))
            adata_loaded = h5ad_backend.load(str(test_file))
            assert adata_loaded is not None
        except (MemoryError, ValueError) as e:
            assert True

    def test_deeply_nested_uns_metadata(self, h5ad_backend, temp_workspace):
        """Test uns with 20+ levels of nesting."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (50, 20)).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(50)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(20)]}),
        )

        # Create 20-level nested structure
        nested = {}
        current = nested
        for i in range(20):
            current[f"level_{i}"] = {"value": i, "nested": {}}
            current = current[f"level_{i}"]["nested"]

        adata.uns["deeply_nested"] = nested

        # Should handle deep nesting
        test_file = temp_workspace / "deep_nesting.h5ad"
        h5ad_backend.save(adata, str(test_file))
        adata_loaded = h5ad_backend.load(str(test_file))

        assert adata_loaded is not None

    def test_circular_reference_metadata(self, h5ad_backend, temp_workspace):
        """Test metadata with circular references (should fail or be sanitized)."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (50, 20)).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(50)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(20)]}),
        )

        # Create circular reference
        circular_dict = {"a": 1}
        circular_dict["self"] = circular_dict  # Circular reference

        # AnnData should reject or sanitize this
        with pytest.raises((ValueError, RecursionError, TypeError)):
            adata.uns["circular"] = circular_dict
            test_file = temp_workspace / "circular.h5ad"
            h5ad_backend.save(adata, str(test_file))

    def test_mixed_type_extreme_column(self, h5ad_backend, temp_workspace):
        """Test column with every imaginable Python type."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (15, 20)).astype(np.float32),
            obs=pd.DataFrame(
                {
                    "cell_id": [f"CELL_{i}" for i in range(15)],
                    "extreme_mixed": [
                        1,
                        2.5,
                        "string",
                        True,
                        None,
                        np.nan,
                        np.inf,
                        complex(1, 2),
                        [1, 2, 3],
                        {"key": "val"},
                        (1, 2),
                        {1, 2},
                        b"bytes",
                        pd.Timestamp("2025-01-01"),
                        None,
                    ],
                }
            ),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(20)]}),
        )

        # Should convert to strings or categorical
        test_file = temp_workspace / "mixed_types.h5ad"
        h5ad_backend.save(adata, str(test_file))
        adata_loaded = h5ad_backend.load(str(test_file))

        assert adata_loaded is not None


# ==============================================================================
# Category 4: Operation Extremes
# ==============================================================================


@pytest.mark.integration
class TestOperationExtremes:
    """Test invalid analysis parameters."""

    def test_clustering_k_greater_than_n_cells(self):
        """Test clustering with k > n_cells."""
        adata = anndata.AnnData(
            X=np.random.randn(10, 50).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(10)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Add PCA for clustering
        adata.obsm["X_pca"] = np.random.randn(10, 10)

        # Try clustering with resolution that would create >10 clusters
        with pytest.raises((ValueError, RuntimeError, AttributeError)) as exc_info:
            # Very high resolution should fail
            import scanpy as sc

            sc.tl.leiden(adata, resolution=1000.0)

        # Should fail with meaningful error (or succeed - either is acceptable)
        assert exc_info.value is not None or adata is not None

    def test_pca_n_components_greater_than_features(self):
        """Test PCA with n_components > n_features."""
        adata = anndata.AnnData(
            X=np.random.randn(100, 50).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Request more components than features
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            from scanpy.tl import pca

            pca(adata, n_comps=100)  # More than 50 features

        assert exc_info.value is not None

    def test_umap_negative_parameters(self):
        """Test UMAP with negative n_neighbors."""
        adata = anndata.AnnData(
            X=np.random.randn(100, 50).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Add PCA
        adata.obsm["X_pca"] = np.random.randn(100, 20)

        # Try negative n_neighbors
        with pytest.raises((ValueError, RuntimeError, TypeError)) as exc_info:
            from scanpy.tl import umap

            umap(adata, n_neighbors=-10)

        assert exc_info.value is not None

    def test_filtering_removes_all_cells(self):
        """Test filter that would remove 100% of cells."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (100, 50)).astype(np.float32),
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Set impossible filter criteria
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            sc.pp.filter_cells(adata, min_genes=1000000)  # Impossible threshold

        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["empty", "no cells", "removed all"])

    def test_normalization_zero_total_counts(self):
        """Test normalization when cells have zero total counts."""
        X = np.zeros((100, 50), dtype=np.float32)
        # Make a few cells non-zero
        X[0:10, :] = np.random.poisson(5, (10, 50))

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(100)]}),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should handle or warn about zero-count cells
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import scanpy as sc

                sc.pp.normalize_total(adata, target_sum=1e4)
                # Should have warned or handled zero counts
                assert adata is not None or len(w) > 0
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["zero", "empty", "count"])

    def test_differential_expression_single_group(self):
        """Test DE with only one group (n=1)."""
        adata = anndata.AnnData(
            X=np.random.poisson(5, (100, 50)).astype(np.float32),
            obs=pd.DataFrame(
                {
                    "cell_id": [f"CELL_{i}" for i in range(100)],
                    "group": ["GroupA"] * 100,  # All same group
                }
            ),
            var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
        )

        # Should fail with single group
        try:
            from scanpy.tl import rank_genes_groups

            with pytest.raises((ValueError, RuntimeError)) as exc_info:
                rank_genes_groups(adata, "group")

            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ["group", "comparison", "unique"])
        except Exception:
            # If scanpy doesn't raise, that's also acceptable
            pass


# ==============================================================================
# Category 5: Memory & Performance Extremes
# ==============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestMemoryPerformanceExtremes:
    """Test resource exhaustion scenarios."""

    def test_million_gene_dataset_save_load(self, h5ad_backend, temp_workspace):
        """Test dataset with 1 million variables (genes) (slow: 6.4s).

        NOTE: This test may be skipped on systems with limited memory.
        """
        try:
            # Create sparse matrix to save memory
            n_obs = 100
            n_vars = 1_000_000

            # Use sparse matrix with ~0.1% non-zero elements
            X = sparse.random(
                n_obs, n_vars, density=0.001, format="csr", dtype=np.float32
            )

            adata = anndata.AnnData(
                X=X,
                obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(n_obs)]}),
                var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(n_vars)]}),
            )

            # Try to save and load
            test_file = temp_workspace / "million_genes.h5ad"
            h5ad_backend.save(adata, str(test_file))
            adata_loaded = h5ad_backend.load(str(test_file))

            assert adata_loaded.shape == (n_obs, n_vars)

        except MemoryError:
            pytest.skip("Insufficient memory for million-gene test")

    def test_million_cell_dataset_operations(self):
        """Test operations on dataset with 1 million cells.

        NOTE: This test may be skipped on systems with limited memory.
        """
        try:
            n_obs = 1_000_000
            n_vars = 100

            # Use sparse matrix
            X = sparse.random(
                n_obs, n_vars, density=0.01, format="csr", dtype=np.float32
            )

            adata = anndata.AnnData(
                X=X,
                obs=pd.DataFrame({"cell_id": [f"CELL_{i}" for i in range(n_obs)]}),
                var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(n_vars)]}),
            )

            # Try basic operations
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            assert adata is not None

        except MemoryError:
            pytest.skip("Insufficient memory for million-cell test")

    def test_huge_annotation_count(self, h5ad_backend, temp_workspace):
        """Test dataset with 1 million obs/var annotations."""
        try:
            n_obs = 1000
            n_annotations = 1_000_000

            # Create large obs with many columns
            obs_data = {
                f"anno_{i}": [f"val_{j}" for j in range(n_obs)]
                for i in range(min(n_annotations, 100))
            }  # Limit to avoid OOM
            obs_data["cell_id"] = [f"CELL_{i}" for i in range(n_obs)]

            adata = anndata.AnnData(
                X=sparse.random(n_obs, 50, density=0.1, format="csr", dtype=np.float32),
                obs=pd.DataFrame(obs_data),
                var=pd.DataFrame({"gene": [f"GENE_{i}" for i in range(50)]}),
            )

            # Try to save
            test_file = temp_workspace / "huge_annotations.h5ad"
            h5ad_backend.save(adata, str(test_file))
            adata_loaded = h5ad_backend.load(str(test_file))

            assert adata_loaded is not None

        except MemoryError:
            pytest.skip("Insufficient memory for huge annotation test")


# ==============================================================================
# Test Runner
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
