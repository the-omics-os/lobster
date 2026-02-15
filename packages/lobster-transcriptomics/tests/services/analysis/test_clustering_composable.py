"""
Tests for composable clustering methods: run_pca() and compute_neighbors_and_embed().

These methods decompose the monolithic cluster_and_visualize() into standalone
steps that can be used independently for step-by-step control.
"""

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)


# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def clustering_service():
    """Create ClusteringService instance for testing."""
    return ClusteringService()


@pytest.fixture
def normalized_hvg_adata():
    """Create normalized AnnData with highly_variable genes marked."""
    n_obs, n_vars = 200, 500
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    # Normalize and find HVGs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200, flavor="seurat")

    return adata


@pytest.fixture
def deviance_adata():
    """Create AnnData with highly_deviant genes marked (raw counts)."""
    n_obs, n_vars = 200, 500
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    # Mark some genes as highly deviant (simulating deviance selection)
    adata.var["highly_deviant"] = False
    adata.var.iloc[:200, adata.var.columns.get_loc("highly_deviant")] = True

    # Normalize after marking
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


@pytest.fixture
def pca_adata(normalized_hvg_adata, clustering_service):
    """Create AnnData with PCA already computed."""
    result, _, _ = clustering_service.run_pca(normalized_hvg_adata)
    return result


@pytest.fixture
def small_adata():
    """Create small AnnData for edge case testing."""
    n_obs, n_vars = 20, 50
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=30, flavor="seurat")

    return adata


# ===============================================================================
# Test run_pca()
# ===============================================================================


@pytest.mark.unit
class TestRunPCA:
    """Test standalone PCA method."""

    def test_pca_default_params(self, clustering_service, normalized_hvg_adata):
        """Test PCA with default parameters returns proper 3-tuple."""
        result, stats, ir = clustering_service.run_pca(normalized_hvg_adata)

        assert isinstance(result, ad.AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

        # Check PCA results stored
        assert "X_pca" in result.obsm
        assert result.obsm["X_pca"].shape[0] == normalized_hvg_adata.n_obs

        # Check stats
        assert stats["analysis_type"] == "pca"
        assert stats["n_comps_computed"] <= 30
        assert stats["variance_explained"] > 0
        assert stats["scaled"] is True

    def test_pca_stores_raw(self, clustering_service, normalized_hvg_adata):
        """Test that PCA stores adata.raw before scaling."""
        result, _, _ = clustering_service.run_pca(normalized_hvg_adata)

        assert result.raw is not None
        assert result.raw.shape == normalized_hvg_adata.shape

    def test_pca_auto_reduces_n_comps(self, clustering_service, small_adata):
        """Test that n_comps is auto-reduced for small datasets."""
        # Request more components than data allows
        result, stats, _ = clustering_service.run_pca(small_adata, n_comps=100)

        # Should auto-reduce
        assert stats["n_comps_computed"] < 100
        assert stats["n_comps_requested"] == 100
        assert "X_pca" in result.obsm

    def test_pca_no_scaling(self, clustering_service, normalized_hvg_adata):
        """Test PCA without scaling."""
        result, stats, _ = clustering_service.run_pca(
            normalized_hvg_adata, scale_data=False
        )

        assert stats["scaled"] is False
        assert "X_pca" in result.obsm

    def test_pca_no_highly_variable(self, clustering_service, normalized_hvg_adata):
        """Test PCA using all genes (not just highly variable)."""
        result, stats, _ = clustering_service.run_pca(
            normalized_hvg_adata, use_highly_variable=False
        )

        assert "X_pca" in result.obsm
        # Should use all genes
        assert stats["n_features_used"] == normalized_hvg_adata.n_vars

    def test_pca_with_deviance_features(self, clustering_service, deviance_adata):
        """Test PCA correctly uses highly_deviant column."""
        result, stats, _ = clustering_service.run_pca(deviance_adata)

        assert "X_pca" in result.obsm
        assert stats["feature_column"] == "highly_deviant"
        assert stats["n_features_used"] == 200  # 200 deviant genes

    def test_pca_ir_structure(self, clustering_service, normalized_hvg_adata):
        """Test IR structure for PCA."""
        _, _, ir = clustering_service.run_pca(normalized_hvg_adata)

        assert ir.operation == "scanpy.tl.pca"
        assert ir.tool_name == "run_pca"
        assert ir.library == "scanpy"
        assert "{{" in ir.code_template
        assert ir.parameter_schema is not None
        assert "n_comps" in ir.parameter_schema

    def test_pca_preserves_uns(self, clustering_service, normalized_hvg_adata):
        """Test PCA stores variance information in uns."""
        result, _, _ = clustering_service.run_pca(normalized_hvg_adata)

        assert "pca" in result.uns
        assert "variance_ratio" in result.uns["pca"]


# ===============================================================================
# Test compute_neighbors_and_embed()
# ===============================================================================


@pytest.mark.unit
class TestComputeNeighborsAndEmbed:
    """Test standalone neighbors + embedding method."""

    def test_umap_default(self, clustering_service, pca_adata):
        """Test UMAP embedding with default parameters."""
        result, stats, ir = clustering_service.compute_neighbors_and_embed(pca_adata)

        assert isinstance(result, ad.AnnData)
        assert isinstance(stats, dict)
        assert isinstance(ir, AnalysisStep)

        # Check UMAP stored
        assert "X_umap" in result.obsm
        assert result.obsm["X_umap"].shape == (pca_adata.n_obs, 2)

        # Check stats
        assert stats["embedding_method"] == "umap"
        assert stats["embedding_key"] == "X_umap"

    def test_tsne_embedding(self, clustering_service, pca_adata):
        """Test tSNE embedding."""
        result, stats, _ = clustering_service.compute_neighbors_and_embed(
            pca_adata, embedding_method="tsne"
        )

        assert "X_tsne" in result.obsm
        assert result.obsm["X_tsne"].shape == (pca_adata.n_obs, 2)
        assert stats["embedding_method"] == "tsne"
        assert stats["embedding_key"] == "X_tsne"

    def test_missing_pca_raises_error(self, clustering_service, normalized_hvg_adata):
        """Test that missing PCA raises ClusteringError."""
        # normalized_hvg_adata has no PCA computed
        with pytest.raises(ClusteringError, match="not found in adata.obsm"):
            clustering_service.compute_neighbors_and_embed(normalized_hvg_adata)

    def test_small_dataset_auto_adjusts(self, clustering_service, small_adata):
        """Test n_neighbors auto-adjustment for small datasets."""
        # First compute PCA
        pca_result, _, _ = clustering_service.run_pca(small_adata, n_comps=10)

        # Request more neighbors than cells
        result, stats, _ = clustering_service.compute_neighbors_and_embed(
            pca_result, n_neighbors=50
        )

        # Should auto-reduce n_neighbors
        assert stats["n_neighbors_used"] < 50
        assert stats["n_neighbors_used"] <= pca_result.n_obs - 1
        assert "X_umap" in result.obsm

    def test_embed_ir_structure(self, clustering_service, pca_adata):
        """Test IR structure for neighbors + embedding."""
        _, _, ir = clustering_service.compute_neighbors_and_embed(pca_adata)

        assert "neighbors" in ir.operation
        assert "umap" in ir.operation
        assert ir.tool_name == "compute_neighbors_and_embed"
        assert ir.library == "scanpy"
        assert "{{" in ir.code_template
        assert ir.parameter_schema is not None
        assert "n_neighbors" in ir.parameter_schema

    def test_unknown_embedding_method_raises(self, clustering_service, pca_adata):
        """Test unknown embedding method raises error."""
        with pytest.raises(ClusteringError, match="Unknown embedding method"):
            clustering_service.compute_neighbors_and_embed(
                pca_adata, embedding_method="unknown"
            )

    def test_n_pcs_auto_adjustment(self, clustering_service, small_adata):
        """Test n_pcs auto-adjustment when exceeding available PCs."""
        pca_result, _, _ = clustering_service.run_pca(small_adata, n_comps=10)

        # Request more PCs than available
        result, stats, _ = clustering_service.compute_neighbors_and_embed(
            pca_result, n_pcs=50
        )

        # Should auto-reduce
        assert stats["n_pcs_used"] <= 10
        assert "X_umap" in result.obsm
