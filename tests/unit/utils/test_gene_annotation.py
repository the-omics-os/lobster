"""
Unit tests for unified gene annotation module.

Tests both the runtime callable and the string constant for notebook export.
"""

import ast
import warnings

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.utils.gene_annotation import (
    ANNOTATE_QC_GENES_HELPER,
    annotate_qc_genes,
)


def _make_adata(gene_names):
    """Create minimal AnnData with given gene names."""
    n_obs = 5
    n_vars = len(gene_names)
    X = np.random.rand(n_obs, n_vars)
    var = pd.DataFrame(index=gene_names)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    return anndata.AnnData(X=X, obs=obs, var=var)


class TestAnnotateQcGenesRuntime:
    """Tests for the runtime annotate_qc_genes callable."""

    def test_human_mt_genes(self):
        """Human MT- genes detected."""
        adata = _make_adata(["MT-CO1", "MT-ND1", "ACTB", "GAPDH"])
        annotate_qc_genes(adata)
        assert adata.var["mt"].sum() == 2
        assert adata.var["mt"]["MT-CO1"] is True or adata.var["mt"]["MT-CO1"] == True

    def test_mouse_mt_genes(self):
        """Mouse mt- genes detected."""
        adata = _make_adata(["mt-Co1", "mt-Nd1", "Actb", "Gapdh"])
        annotate_qc_genes(adata)
        assert adata.var["mt"].sum() == 2

    def test_alt_delimiter_mt_genes(self):
        """Alternative delimiter MT. genes detected."""
        adata = _make_adata(["MT.CO1", "MT.ND1", "ACTB"])
        annotate_qc_genes(adata)
        assert adata.var["mt"].sum() == 2

    def test_ensembl_mt_genes(self):
        """Ensembl MT gene IDs detected."""
        adata = _make_adata(["ENSG00000198888", "ENSG00000210049", "ENSG00000000003"])
        annotate_qc_genes(adata)
        assert adata.var["mt"].sum() == 2

    def test_generic_mt_genes(self):
        """Generic mito/mitochondr pattern detected."""
        adata = _make_adata(["mitochondrially_encoded_1", "ACTB"])
        annotate_qc_genes(adata)
        assert adata.var["mt"].sum() == 1

    def test_human_ribo_genes(self):
        """Human RPS/RPL detected."""
        adata = _make_adata(["RPS2", "RPL10", "RPS15", "ACTB", "GAPDH"])
        annotate_qc_genes(adata)
        assert adata.var["ribo"].sum() == 3

    def test_mouse_ribo_genes(self):
        """Mouse Rps/Rpl detected."""
        adata = _make_adata(["Rps2", "Rpl10", "Actb"])
        annotate_qc_genes(adata)
        assert adata.var["ribo"].sum() == 2

    def test_no_matching_genes_warns(self):
        """No matching genes produces warnings (not silent failure)."""
        adata = _make_adata(["UNKNOWN1", "UNKNOWN2", "UNKNOWN3"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            annotate_qc_genes(adata)
            warning_msgs = [str(x.message) for x in w]
            assert any("mitochondrial" in m.lower() for m in warning_msgs)
            assert any("ribosomal" in m.lower() for m in warning_msgs)
        assert adata.var["mt"].sum() == 0
        assert adata.var["ribo"].sum() == 0

    def test_idempotent(self):
        """Skips existing columns on second call."""
        adata = _make_adata(["MT-CO1", "RPS2", "ACTB"])
        annotate_qc_genes(adata)
        # Manually flip a value to verify second call doesn't overwrite
        adata.var.loc["ACTB", "mt"] = True
        annotate_qc_genes(adata)
        # Should still have the manually set value
        assert adata.var.loc["ACTB", "mt"] == True

    def test_both_mt_and_ribo_annotated(self):
        """Both mt and ribo columns created in single call."""
        adata = _make_adata(["MT-CO1", "RPS2", "ACTB"])
        annotate_qc_genes(adata)
        assert "mt" in adata.var.columns
        assert "ribo" in adata.var.columns


class TestAnnotateQcGenesHelper:
    """Tests for the ANNOTATE_QC_GENES_HELPER string constant."""

    def test_valid_python_syntax(self):
        """String constant is valid Python."""
        ast.parse(ANNOTATE_QC_GENES_HELPER)

    def test_exec_creates_callable(self):
        """exec'd string creates a callable annotate_qc_genes function."""
        ns = {}
        exec(ANNOTATE_QC_GENES_HELPER, ns)
        assert callable(ns["annotate_qc_genes"])

    def test_exec_produces_same_results(self):
        """exec'd function produces same results as runtime callable."""
        # Run runtime version
        adata_runtime = _make_adata(["MT-CO1", "MT-ND1", "RPS2", "RPL10", "ACTB"])
        annotate_qc_genes(adata_runtime)

        # Run exec'd version
        adata_exec = _make_adata(["MT-CO1", "MT-ND1", "RPS2", "RPL10", "ACTB"])
        ns = {}
        exec(ANNOTATE_QC_GENES_HELPER, ns)
        ns["annotate_qc_genes"](adata_exec)

        # Compare results
        np.testing.assert_array_equal(
            adata_runtime.var["mt"].values, adata_exec.var["mt"].values
        )
        np.testing.assert_array_equal(
            adata_runtime.var["ribo"].values, adata_exec.var["ribo"].values
        )

    def test_exec_mouse_genes(self):
        """exec'd function handles mouse genes correctly."""
        adata = _make_adata(["mt-Co1", "Rps2", "Actb"])
        ns = {}
        exec(ANNOTATE_QC_GENES_HELPER, ns)
        ns["annotate_qc_genes"](adata)
        assert adata.var["mt"].sum() == 1
        assert adata.var["ribo"].sum() == 1

    def test_exec_ensembl_genes(self):
        """exec'd function handles Ensembl IDs correctly."""
        adata = _make_adata(["ENSG00000198888", "ENSG00000000003"])
        ns = {}
        exec(ANNOTATE_QC_GENES_HELPER, ns)
        ns["annotate_qc_genes"](adata)
        assert adata.var["mt"].sum() == 1
