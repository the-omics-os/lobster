"""
Comprehensive unit tests for QualityService.

This test suite provides 40+ tests covering:
- Basic QC metrics calculation
- Edge cases (empty datasets, single cell, extreme values)
- Missing value handling
- Multi-metric assessment validation
- Gene identification patterns
- Scientific accuracy validation
- IR generation for notebook export
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.quality.quality_service import QualityError, QualityService


class TestQualityServiceInitialization:
    """Test service initialization."""

    def test_init_no_config(self):
        """Test initialization without config."""
        service = QualityService()
        assert service.config == {}

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"test_param": "value"}
        service = QualityService(config=config)
        assert service.config == config

    def test_init_with_kwargs(self):
        """Test initialization with kwargs (backward compatibility)."""
        service = QualityService(some_param="value")
        assert service.config == {}


class TestQCMetricsCalculationFromAnnData:
    """Test QC metrics calculation from AnnData objects."""

    def test_calculate_qc_metrics_dense_matrix(self):
        """Test QC metrics with dense matrix."""
        # Create test data with known patterns
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)

        # Create gene names with specific patterns
        gene_names = []
        gene_names.extend([f"MT-{i}" for i in range(5)])  # Mitochondrial genes
        gene_names.extend([f"RPS{i}" for i in range(5)])  # Ribosomal genes
        gene_names.extend(["ACTB", "GAPDH", "MALAT1"])  # Housekeeping genes
        gene_names.extend([f"GENE{i}" for i in range(n_vars - 13)])  # Other genes

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names[:n_vars]))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Validate structure
        assert len(qc_metrics) == n_obs
        assert "total_counts" in qc_metrics.columns
        assert "n_genes" in qc_metrics.columns
        assert "mt_pct" in qc_metrics.columns
        assert "ribo_pct" in qc_metrics.columns
        assert "housekeeping_score" in qc_metrics.columns

        # Validate metrics are non-negative
        assert (qc_metrics["total_counts"] >= 0).all()
        assert (qc_metrics["n_genes"] >= 0).all()
        assert (qc_metrics["mt_pct"] >= 0).all()
        assert (qc_metrics["ribo_pct"] >= 0).all()
        assert (qc_metrics["housekeeping_score"] >= 0).all()

    def test_calculate_qc_metrics_sparse_matrix(self):
        """Test QC metrics with sparse matrix."""
        n_obs, n_vars = 100, 50
        X_dense = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        X_sparse = sp.csr_matrix(X_dense)

        gene_names = []
        gene_names.extend([f"MT-{i}" for i in range(5)])
        gene_names.extend([f"RPS{i}" for i in range(5)])
        gene_names.extend(["ACTB", "GAPDH", "MALAT1"])
        gene_names.extend([f"GENE{i}" for i in range(n_vars - 13)])

        adata = AnnData(X=X_sparse, var=pd.DataFrame(index=gene_names[:n_vars]))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Validate structure
        assert len(qc_metrics) == n_obs
        assert all(
            col in qc_metrics.columns
            for col in [
                "total_counts",
                "n_genes",
                "mt_pct",
                "ribo_pct",
                "housekeeping_score",
            ]
        )

    def test_calculate_qc_metrics_no_mitochondrial_genes(self):
        """Test QC metrics when no mitochondrial genes present."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]  # No MT- genes

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # MT percentage should be zero
        assert (qc_metrics["mt_pct"] == 0).all()

    def test_calculate_qc_metrics_no_ribosomal_genes(self):
        """Test QC metrics when no ribosomal genes present."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]  # No RPS/RPL genes

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Ribo percentage should be zero
        assert (qc_metrics["ribo_pct"] == 0).all()

    def test_calculate_qc_metrics_no_housekeeping_genes(self):
        """Test QC metrics when no housekeeping genes present."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]  # No housekeeping genes

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Housekeeping score should be zero
        assert (qc_metrics["housekeeping_score"] == 0).all()

    def test_calculate_qc_metrics_lowercase_gene_names(self):
        """Test QC metrics with lowercase gene names."""
        n_obs, n_vars = 50, 20
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = (
            [f"mt-{i}" for i in range(5)]
            + [f"rps{i}" for i in range(5)]
            + [f"gene{i}" for i in range(10)]
        )

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Should detect lowercase mt- and rps genes
        assert (qc_metrics["mt_pct"] > 0).any()
        assert (qc_metrics["ribo_pct"] > 0).any()

    def test_calculate_qc_metrics_rpl_ribosomal_genes(self):
        """Test QC metrics with RPL ribosomal genes."""
        n_obs, n_vars = 50, 20
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = (
            [f"RPL{i}" for i in range(5)]
            + [f"rpl{i}" for i in range(5)]
            + [f"GENE{i}" for i in range(10)]
        )

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Should detect RPL and rpl genes as ribosomal
        assert (qc_metrics["ribo_pct"] > 0).any()

    def test_calculate_qc_metrics_zero_counts_handling(self):
        """Test QC metrics with zero total counts (division by zero safety)."""
        n_obs, n_vars = 50, 20
        X = np.zeros((n_obs, n_vars))  # All zeros
        gene_names = [f"MT-{i}" for i in range(5)] + [f"GENE{i}" for i in range(15)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Should not raise division by zero errors
        assert not np.isnan(qc_metrics["mt_pct"]).any()
        assert not np.isinf(qc_metrics["mt_pct"]).any()


class TestAssessQualityEdgeCases:
    """Test assess_quality() method edge cases."""

    def test_assess_quality_empty_dataset(self):
        """Test quality assessment with empty dataset (0 cells)."""
        X = np.array([]).reshape(0, 50)
        gene_names = [f"GENE{i}" for i in range(50)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        assert result_adata.n_obs == 0
        assert stats["cells_before_qc"] == 0
        assert stats["cells_after_qc"] == 0

    def test_assess_quality_single_cell(self):
        """Test quality assessment with single cell."""
        X = np.array([[10, 20, 5, 3, 1, 0, 0, 0, 0, 0]])
        gene_names = [
            "MT-1",
            "MT-2",
            "RPS1",
            "RPL1",
            "ACTB",
            "GENE1",
            "GENE2",
            "GENE3",
            "GENE4",
            "GENE5",
        ]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        assert result_adata.n_obs == 1
        assert stats["cells_before_qc"] == 1
        assert "mean_total_counts" in stats
        assert "mean_genes_per_cell" in stats

    def test_assess_quality_100_percent_mitochondrial(self):
        """Test quality assessment with 100% mitochondrial genes."""
        n_obs, n_vars = 20, 10
        X = np.random.poisson(10, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"MT-{i}" for i in range(n_vars)]  # All MT genes

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata, max_mt_pct=20.0)

        # All cells should fail QC due to high MT%
        assert stats["cells_after_qc"] == 0
        assert (result_adata.obs["mt_pct"] > 90).all()  # Should be ~100%

    def test_assess_quality_zero_counts_all_cells(self):
        """Test quality assessment with zero counts in all cells."""
        n_obs, n_vars = 30, 20
        X = np.zeros((n_obs, n_vars))
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata, min_genes=1)

        # All cells should fail due to zero genes
        assert stats["cells_after_qc"] == 0
        assert stats["mean_genes_per_cell"] == 0

    def test_assess_quality_custom_thresholds(self):
        """Test quality assessment with custom thresholds."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = []
        gene_names.extend([f"MT-{i}" for i in range(5)])
        gene_names.extend([f"RPS{i}" for i in range(5)])
        gene_names.extend(["ACTB", "GAPDH", "MALAT1"])
        gene_names.extend([f"GENE{i}" for i in range(n_vars - 13)])

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names[:n_vars]))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(
            adata,
            min_genes=1000,  # Very high threshold
            max_mt_pct=5.0,  # Very low threshold
            max_ribo_pct=10.0,  # Very low threshold
            min_housekeeping_score=100.0,  # Very high threshold
        )

        # Most/all cells should fail with strict thresholds
        assert stats["cells_after_qc"] < stats["cells_before_qc"]
        assert stats["min_genes"] == 1000
        assert stats["max_mt_pct"] == 5.0

    def test_assess_quality_qc_pass_flag(self):
        """Test that qc_pass flag is correctly added to observations."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # qc_pass should be added
        assert "qc_pass" in result_adata.obs.columns
        assert result_adata.obs["qc_pass"].dtype == bool

    def test_assess_quality_metrics_added_to_obs(self):
        """Test that QC metrics are added to obs."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"MT-{i}" for i in range(5)] + [f"GENE{i}" for i in range(25)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # Metrics should be in obs
        assert "mt_pct" in result_adata.obs.columns
        assert "ribo_pct" in result_adata.obs.columns
        assert "housekeeping_score" in result_adata.obs.columns

    def test_assess_quality_statistics_structure(self):
        """Test that statistics dictionary has correct structure."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # Validate statistics structure
        required_keys = [
            "analysis_type",
            "min_genes",
            "max_mt_pct",
            "max_ribo_pct",
            "min_housekeeping_score",
            "cells_before_qc",
            "cells_after_qc",
            "cells_removed",
            "cells_retained_pct",
            "quality_status",
            "mean_total_counts",
            "mean_genes_per_cell",
            "mean_mt_pct",
            "mean_ribo_pct",
            "mean_housekeeping_score",
            "qc_summary",
            "mt_stats",
            "ribo_stats",
            "housekeeping_stats",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_assess_quality_quality_status_pass(self):
        """Test quality_status is 'Pass' when >70% cells retained."""
        n_obs, n_vars = 100, 50
        # Use deterministic high counts to ensure >70% pass
        X = np.full((n_obs, n_vars), 100, dtype=float)  # All cells have high counts
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        # Use very lenient threshold to ensure most cells pass
        result_adata, stats, ir = service.assess_quality(
            adata,
            min_genes=1,
            max_mt_pct=100.0,
            max_ribo_pct=100.0,
            min_housekeeping_score=0.0,
        )

        # Should pass QC with lenient thresholds
        assert stats["quality_status"] == "Pass"
        assert stats["cells_retained_pct"] > 70

    def test_assess_quality_quality_status_warning(self):
        """Test quality_status is 'Warning' when <70% cells retained."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(2, size=(n_obs, n_vars)).astype(float)  # Low counts
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(
            adata, min_genes=500
        )  # Strict threshold

        # Should trigger warning
        if stats["cells_retained_pct"] <= 70:
            assert stats["quality_status"] == "Warning"

    def test_assess_quality_error_handling(self):
        """Test error handling with invalid input."""
        # Invalid AnnData (not an AnnData object)
        service = QualityService()

        with pytest.raises(QualityError):
            service.assess_quality("not_an_anndata")


class TestQCSummaryGeneration:
    """Test QC summary text generation."""

    def test_generate_qc_summary_low_mt(self):
        """Test summary generation with low mitochondrial content."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 5, 50),
                "ribo_pct": np.random.uniform(10, 30, 50),
                "housekeeping_score": np.random.uniform(5, 20, 50),
                "n_genes": np.random.randint(800, 1500, 50),
                "total_counts": np.random.randint(5000, 10000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "low mitochondrial gene expression (healthy)" in summary
        assert "good" in summary.lower()

    def test_generate_qc_summary_moderate_mt(self):
        """Test summary generation with moderate mitochondrial content."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(5, 10, 50),
                "ribo_pct": np.random.uniform(10, 30, 50),
                "housekeeping_score": np.random.uniform(5, 20, 50),
                "n_genes": np.random.randint(800, 1500, 50),
                "total_counts": np.random.randint(5000, 10000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "moderate mitochondrial gene expression" in summary

    def test_generate_qc_summary_high_mt(self):
        """Test summary generation with high mitochondrial content."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(15, 30, 50),
                "ribo_pct": np.random.uniform(10, 30, 50),
                "housekeeping_score": np.random.uniform(5, 20, 50),
                "n_genes": np.random.randint(800, 1500, 50),
                "total_counts": np.random.randint(5000, 10000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "high mitochondrial gene expression" in summary
        assert "stressed or dying" in summary

    def test_generate_qc_summary_high_ribosomal(self):
        """Test summary generation with high ribosomal content."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 5, 50),
                "ribo_pct": np.random.uniform(45, 60, 50),  # High ribo
                "housekeeping_score": np.random.uniform(5, 20, 50),
                "n_genes": np.random.randint(800, 1500, 50),
                "total_counts": np.random.randint(5000, 10000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "high ribosomal content" in summary
        assert "metabolic stress" in summary

    def test_generate_qc_summary_low_housekeeping(self):
        """Test summary generation with low housekeeping gene expression."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 5, 50),
                "ribo_pct": np.random.uniform(10, 30, 50),
                "housekeeping_score": np.random.uniform(0, 1, 50),  # Low housekeeping
                "n_genes": np.random.randint(800, 1500, 50),
                "total_counts": np.random.randint(5000, 10000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "low housekeeping gene expression" in summary
        assert "poor RNA quality" in summary

    def test_generate_qc_summary_low_gene_count(self):
        """Test summary generation with low gene count."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 5, 50),
                "ribo_pct": np.random.uniform(10, 30, 50),
                "housekeeping_score": np.random.uniform(5, 20, 50),
                "n_genes": np.random.randint(200, 600, 50),  # Low gene count
                "total_counts": np.random.randint(1000, 3000, 50),
            }
        )

        service = QualityService()
        summary = service._generate_qc_summary(qc_metrics)

        assert "low gene count per cell" in summary


class TestQualityIRGeneration:
    """Test Intermediate Representation (IR) generation for notebook export."""

    def test_create_quality_ir_structure(self):
        """Test that IR has correct structure."""
        service = QualityService()
        ir = service._create_quality_ir(
            min_genes=500,
            max_genes=5000,
            max_mt_pct=20.0,
            max_ribo_pct=50.0,
            min_housekeeping_score=1.0,
        )

        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "scanpy.pp.calculate_qc_metrics"
        assert ir.tool_name == "assess_quality"
        assert ir.library == "scanpy"

    def test_create_quality_ir_parameters(self):
        """Test that IR contains correct parameters."""
        service = QualityService()
        ir = service._create_quality_ir(
            min_genes=600,
            max_genes=6000,
            max_mt_pct=15.0,
            max_ribo_pct=40.0,
            min_housekeeping_score=2.0,
        )

        assert ir.parameters["min_genes"] == 600
        assert ir.parameters["max_genes"] == 6000
        assert ir.parameters["max_mt_pct"] == 15.0
        assert ir.parameters["max_ribo_pct"] == 40.0
        assert ir.parameters["min_housekeeping_score"] == 2.0

    def test_create_quality_ir_parameter_schema(self):
        """Test that IR has valid parameter schema."""
        service = QualityService()
        ir = service._create_quality_ir(
            min_genes=500,
            max_genes=5000,
            max_mt_pct=20.0,
            max_ribo_pct=50.0,
            min_housekeeping_score=1.0,
        )

        assert "min_genes" in ir.parameter_schema
        assert "max_genes" in ir.parameter_schema
        assert "max_mt_pct" in ir.parameter_schema
        assert "max_ribo_pct" in ir.parameter_schema
        assert "min_housekeeping_score" in ir.parameter_schema

        # Check parameter specs
        for param_name, param_spec in ir.parameter_schema.items():
            assert isinstance(param_spec, ParameterSpec)
            assert param_spec.papermill_injectable is True
            assert param_spec.default_value is not None

    def test_create_quality_ir_code_template(self):
        """Test that IR code template is valid."""
        service = QualityService()
        ir = service._create_quality_ir(
            min_genes=500,
            max_genes=5000,
            max_mt_pct=20.0,
            max_ribo_pct=50.0,
            min_housekeeping_score=1.0,
        )

        # Check template contains Jinja2 placeholders
        assert "{{ min_genes }}" in ir.code_template
        assert "{{ max_genes }}" in ir.code_template
        assert "{{ max_mt_pct }}" in ir.code_template
        assert "{{ max_ribo_pct }}" in ir.code_template
        assert "sc.pp.calculate_qc_metrics" in ir.code_template

    def test_create_quality_ir_imports(self):
        """Test that IR includes necessary imports."""
        service = QualityService()
        ir = service._create_quality_ir(
            min_genes=500,
            max_genes=5000,
            max_mt_pct=20.0,
            max_ribo_pct=50.0,
            min_housekeeping_score=1.0,
        )

        assert "import scanpy as sc" in ir.imports
        assert "import numpy as np" in ir.imports

    def test_assess_quality_returns_valid_ir(self):
        """Test that assess_quality returns valid IR."""
        n_obs, n_vars = 50, 30
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # Validate IR
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "scanpy.pp.calculate_qc_metrics"
        assert ir.validates_on_export is True
        assert ir.requires_validation is False


class TestScientificAccuracy:
    """Test scientific accuracy of QC metrics."""

    def test_mt_percentage_calculation_accuracy(self):
        """Test that mitochondrial percentage is calculated correctly."""
        # Create controlled dataset with simple math
        n_obs = 10
        n_vars = 10

        # Create dataset where:
        # - First 2 genes are MT genes with 10 counts each (20 total MT counts)
        # - Remaining 8 genes have 10 counts each (80 total other counts)
        # Total counts per cell = 100
        # MT% should be exactly (20 / 100) * 100 = 20%
        X = np.full((n_obs, n_vars), 10, dtype=float)

        gene_names = ["MT-1", "MT-2"] + [f"GENE{i}" for i in range(8)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # MT% should be 20% (20 MT counts / 100 total counts)
        expected_mt_pct = 20.0
        np.testing.assert_almost_equal(
            qc_metrics["mt_pct"].values, expected_mt_pct, decimal=5
        )

    def test_ribo_percentage_calculation_accuracy(self):
        """Test that ribosomal percentage is calculated correctly."""
        # Create controlled dataset with simple math
        n_obs = 10
        n_vars = 10

        # Create dataset where:
        # - First 3 genes are RPS genes with 10 counts each (30 total ribo counts)
        # - Remaining 7 genes have 10 counts each (70 total other counts)
        # Total counts per cell = 100
        # Ribo% should be exactly (30 / 100) * 100 = 30%
        X = np.full((n_obs, n_vars), 10, dtype=float)

        gene_names = ["RPS1", "RPS2", "RPS3"] + [f"GENE{i}" for i in range(7)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Ribo% should be 30% (30 ribo counts / 100 total counts)
        expected_ribo_pct = 30.0
        np.testing.assert_almost_equal(
            qc_metrics["ribo_pct"].values, expected_ribo_pct, decimal=5
        )

    def test_housekeeping_score_calculation_accuracy(self):
        """Test that housekeeping score is calculated correctly."""
        n_obs = 10
        n_vars = 20

        X = np.zeros((n_obs, n_vars))
        gene_names = ["ACTB", "GAPDH", "MALAT1"] + [f"GENE{i}" for i in range(17)]

        # Set housekeeping gene counts
        X[:, 0] = 50  # ACTB
        X[:, 1] = 30  # GAPDH
        X[:, 2] = 20  # MALAT1

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Housekeeping score should be sum = 100
        expected_score = 100
        np.testing.assert_almost_equal(
            qc_metrics["housekeeping_score"].values, expected_score, decimal=5
        )

    def test_n_genes_calculation_accuracy(self):
        """Test that number of genes is counted correctly."""
        n_obs = 5
        n_vars = 20

        X = np.zeros((n_obs, n_vars))
        # Each cell has 10 expressed genes (first 10 columns)
        X[:, :10] = 1

        gene_names = [f"GENE{i}" for i in range(n_vars)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Should detect exactly 10 genes per cell
        assert (qc_metrics["n_genes"] == 10).all()

    def test_total_counts_calculation_accuracy(self):
        """Test that total counts are calculated correctly."""
        n_obs = 5
        n_vars = 10

        # Each cell has total counts of 1000
        X = np.ones((n_obs, n_vars)) * 100

        gene_names = [f"GENE{i}" for i in range(n_vars)]
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Should calculate exactly 1000 per cell
        assert (qc_metrics["total_counts"] == 1000).all()


class TestLegacyDataFrameMethod:
    """Test legacy _calculate_qc_metrics() method for backward compatibility."""

    def test_calculate_qc_metrics_dataframe(self):
        """Test QC metrics calculation from DataFrame."""
        data = pd.DataFrame(
            np.random.poisson(5, size=(50, 30)),
            columns=[f"MT-{i}" for i in range(5)]
            + [f"RPS{i}" for i in range(5)]
            + ["ACTB", "GAPDH", "MALAT1"]
            + [f"GENE{i}" for i in range(17)],
        )

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics(data)

        assert len(qc_metrics) == 50
        assert "total_counts" in qc_metrics.columns
        assert "n_genes" in qc_metrics.columns
        assert "mt_pct" in qc_metrics.columns
        assert "ribo_pct" in qc_metrics.columns
        assert "housekeeping_score" in qc_metrics.columns

    def test_calculate_qc_metrics_dataframe_no_mt_genes(self):
        """Test DataFrame QC metrics without MT genes."""
        data = pd.DataFrame(
            np.random.poisson(5, size=(50, 20)), columns=[f"GENE{i}" for i in range(20)]
        )

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics(data)

        # MT percentage should be zero
        assert (qc_metrics["mt_pct"] == 0).all()

    def test_calculate_qc_metrics_dataframe_no_ribo_genes(self):
        """Test DataFrame QC metrics without ribosomal genes."""
        data = pd.DataFrame(
            np.random.poisson(5, size=(50, 20)), columns=[f"GENE{i}" for i in range(20)]
        )

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics(data)

        # Ribo percentage should be zero
        assert (qc_metrics["ribo_pct"] == 0).all()


class TestQualityPlots:
    """Test quality plot generation."""

    def test_create_quality_plots_structure(self):
        """Test that quality plots are created with correct structure."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 30, 100),
                "ribo_pct": np.random.uniform(0, 50, 100),
                "housekeeping_score": np.random.uniform(0, 20, 100),
                "n_genes": np.random.randint(200, 2000, 100),
                "total_counts": np.random.randint(500, 10000, 100),
            }
        )

        service = QualityService()
        plots = service._create_quality_plots(qc_metrics)

        # Should create 6 plots
        assert len(plots) == 6

    def test_create_quality_plots_types(self):
        """Test that plots are Plotly figures."""
        qc_metrics = pd.DataFrame(
            {
                "mt_pct": np.random.uniform(0, 30, 100),
                "ribo_pct": np.random.uniform(0, 50, 100),
                "housekeeping_score": np.random.uniform(0, 20, 100),
                "n_genes": np.random.randint(200, 2000, 100),
                "total_counts": np.random.randint(500, 10000, 100),
            }
        )

        service = QualityService()
        plots = service._create_quality_plots(qc_metrics)

        # All should be Plotly Figure objects
        from plotly.graph_objects import Figure

        for plot in plots:
            assert isinstance(plot, Figure)


class TestMitochondrialGeneDetection:
    """Test mitochondrial gene detection across nomenclature patterns."""

    def test_detect_mt_genes_hgnc_pattern(self):
        """Test detection of HGNC pattern (MT-)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(45)] + [
            "MT-ND1",
            "MT-CO1",
            "MT-ATP6",
            "MT-CYB",
            "MT-ND2",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find exactly 5 MT genes
        assert mt_mask.sum() == 5
        # Check correct genes identified
        assert mt_mask[-5:].all()  # Last 5 should be True
        assert not mt_mask[:-5].any()  # First 45 should be False

    def test_detect_mt_genes_mouse_pattern(self):
        """Test detection of mouse pattern (mt- lowercase)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"Gene{i}" for i in range(45)] + [
            "mt-Nd1",
            "mt-Co1",
            "mt-Atp6",
            "mt-Cytb",
            "mt-Nd2",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find exactly 5 MT genes
        assert mt_mask.sum() == 5
        assert mt_mask[-5:].all()

    def test_detect_mt_genes_alternative_delimiter(self):
        """Test detection of alternative delimiter (MT.)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(45)] + [
            "MT.ND1",
            "MT.CO1",
            "MT.ATP6",
            "MT.CYB",
            "MT.ND2",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find exactly 5 MT genes
        assert mt_mask.sum() == 5
        assert mt_mask[-5:].all()

    def test_detect_mt_genes_ensembl_pattern(self):
        """Test detection of Ensembl ID pattern."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"ENSG0000012345{i}" for i in range(45)] + [
            "ENSG00000198888",  # MT-ND1
            "ENSG00000198763",  # MT-ND2
            "ENSG00000198804",  # MT-CO1
            "ENSG00000210082",  # MT-RNR2
            "ENSG00000210049",  # MT-TF
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find exactly 5 MT genes
        assert mt_mask.sum() == 5
        assert mt_mask[-5:].all()

    def test_detect_mt_genes_generic_fallback(self):
        """Test detection using generic fallback pattern."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(45)] + [
            "mitochondrial_gene1",
            "mito_protein",
            "mitochondria_related",
            "gene_with_mito",
            "mitochondrial",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find all 5 genes with "mito" in name
        assert mt_mask.sum() == 5
        assert mt_mask[-5:].all()

    def test_detect_mt_genes_none_found(self):
        """Test graceful handling when no MT genes present."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should return all False
        assert mt_mask.sum() == 0
        assert len(mt_mask) == n_vars
        assert mt_mask.dtype == bool

    def test_detect_mt_genes_cascade_order(self):
        """Test that pattern cascade stops at first match."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        # Mix patterns: HGNC should win
        gene_names = (
            [f"GENE{i}" for i in range(40)]
            + [
                "MT-ND1",
                "MT-CO1",
                "mt-nd2",  # Lowercase (should not be counted if HGNC present)
                "MT.ATP6",  # Dot delimiter (should not be counted if HGNC present)
                "mitochondrial",  # Generic (should not be counted if HGNC present)
            ]
            + [f"GENE{i}" for i in range(45, 50)]
        )

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        mt_mask = service._detect_mitochondrial_genes(adata)

        # Should find only the 2 HGNC pattern genes (MT-ND1, MT-CO1)
        assert mt_mask.sum() == 2

    def test_detect_mt_genes_integration_with_qc(self):
        """Test MT detection integrates correctly with QC metrics."""
        n_obs, n_vars = 100, 50
        # Create dataset with known MT percentage
        X = np.full((n_obs, n_vars), 10, dtype=float)
        gene_names = ["MT-ND1", "MT-CO1"] + [f"GENE{i}" for i in range(48)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # MT% should be 4% (2 MT genes / 50 total genes * 100)
        expected_mt_pct = (2 / 50) * 100
        np.testing.assert_almost_equal(
            qc_metrics["mt_pct"].values, expected_mt_pct, decimal=5
        )


class TestRibosomalGeneDetection:
    """Test ribosomal gene detection across nomenclature patterns."""

    def test_detect_ribo_genes_hgnc_pattern(self):
        """Test detection of HGNC pattern (RPS*, RPL*)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(43)] + [
            "RPS3",
            "RPS27",
            "RPL5",
            "RPL23",
            "RPS29",
            "RPL11",
            "RPS6",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should find exactly 7 ribosomal genes
        assert ribo_mask.sum() == 7
        assert ribo_mask[-7:].all()

    def test_detect_ribo_genes_mouse_pattern(self):
        """Test detection of mouse pattern (Rps*, Rpl*)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"Gene{i}" for i in range(45)] + [
            "Rps3",
            "Rps27",
            "Rpl5",
            "Rpl23",
            "Rps29",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should find exactly 5 ribosomal genes
        assert ribo_mask.sum() == 5
        assert ribo_mask[-5:].all()

    def test_detect_ribo_genes_lowercase_pattern(self):
        """Test detection of lowercase pattern (rps*, rpl*)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"gene{i}" for i in range(45)] + [
            "rps3",
            "rps27",
            "rpl5",
            "rpl23",
            "rps29",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should find exactly 5 ribosomal genes
        assert ribo_mask.sum() == 5
        assert ribo_mask[-5:].all()

    def test_detect_ribo_genes_compact_notation(self):
        """Test detection of compact notation (RP[SL]*)."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(45)] + [
            "RPS3",
            "RPL5",
            "RPS27",
            "RPL23",
            "RPS29",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should find exactly 5 ribosomal genes
        assert ribo_mask.sum() == 5

    def test_detect_ribo_genes_generic_fallback(self):
        """Test detection using generic fallback pattern."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(45)] + [
            "ribosomal_protein_s3",
            "ribosome_protein",
            "gene_ribosom",
            "ribosomal",
            "ribosomes",
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should find all 5 genes with "ribosom" in name
        assert ribo_mask.sum() == 5
        assert ribo_mask[-5:].all()

    def test_detect_ribo_genes_none_found(self):
        """Test graceful handling when no ribosomal genes present."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)
        gene_names = [f"GENE{i}" for i in range(n_vars)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        ribo_mask = service._detect_ribosomal_genes(adata)

        # Should return all False
        assert ribo_mask.sum() == 0
        assert len(ribo_mask) == n_vars
        assert ribo_mask.dtype == bool

    def test_detect_ribo_genes_integration_with_qc(self):
        """Test ribosomal detection integrates correctly with QC metrics."""
        n_obs, n_vars = 100, 50
        # Create dataset with known ribo percentage
        X = np.full((n_obs, n_vars), 10, dtype=float)
        gene_names = ["RPS3", "RPS27", "RPL5"] + [f"GENE{i}" for i in range(47)]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        qc_metrics = service._calculate_qc_metrics_from_adata(adata)

        # Ribo% should be 6% (3 ribo genes / 50 total genes * 100)
        expected_ribo_pct = (3 / 50) * 100
        np.testing.assert_almost_equal(
            qc_metrics["ribo_pct"].values, expected_ribo_pct, decimal=5
        )


class TestGeneDetectionEndToEnd:
    """End-to-end tests for gene detection in full QC workflow."""

    def test_assess_quality_with_mixed_nomenclature(self):
        """Test full QC assessment with mixed gene nomenclature."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)

        # Mix of patterns
        gene_names = (
            ["MT-ND1", "MT-CO1"]  # HGNC MT
            + ["RPS3", "RPL5"]  # HGNC ribo
            + ["ACTB", "GAPDH"]  # Housekeeping
            + [f"GENE{i}" for i in range(44)]
        )

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # Should detect MT and ribo genes
        assert stats["mean_mt_pct"] > 0
        assert stats["mean_ribo_pct"] > 0
        assert "mt_pct" in result_adata.obs.columns
        assert "ribo_pct" in result_adata.obs.columns

    def test_assess_quality_before_after_comparison(self):
        """Test that multi-pattern detection fixes 0.0% issue."""
        n_obs, n_vars = 100, 50
        X = np.random.poisson(10, size=(n_obs, n_vars)).astype(float)

        # Use Ensembl IDs (would have failed with old implementation)
        gene_names = [f"ENSG0000012345{i}" for i in range(45)] + [
            "ENSG00000198888",  # MT gene
            "ENSG00000198763",  # MT gene
            "ENSG00000198804",  # MT gene
            "RPS3",  # Ribo gene
            "RPL5",  # Ribo gene
        ]

        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

        service = QualityService()
        result_adata, stats, ir = service.assess_quality(adata)

        # Old implementation would report 0.0%, new should detect
        assert stats["mean_mt_pct"] > 0, "MT genes should be detected (was 0.0% before)"
        assert stats["mean_ribo_pct"] > 0, "Ribo genes should be detected"

    def test_assess_quality_no_false_negatives(self):
        """Test that all common nomenclature systems are detected."""
        test_cases = [
            # (gene_names, expected_mt_count, expected_ribo_count)
            (["MT-ND1", "MT-CO1", "RPS3", "RPL5"] + [f"G{i}" for i in range(46)], 2, 2),
            (["mt-nd1", "mt-co1", "rps3", "rpl5"] + [f"G{i}" for i in range(46)], 2, 2),
            (["MT.ND1", "MT.CO1", "Rps3", "Rpl5"] + [f"G{i}" for i in range(46)], 2, 2),
            (
                ["ENSG00000198888", "ENSG00000210082", "RPS3", "RPL5"]
                + [f"G{i}" for i in range(46)],
                2,
                2,
            ),
        ]

        service = QualityService()

        for gene_names, expected_mt, expected_ribo in test_cases:
            X = np.random.poisson(10, size=(100, 50)).astype(float)
            adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))

            mt_mask = service._detect_mitochondrial_genes(adata)
            ribo_mask = service._detect_ribosomal_genes(adata)

            assert (
                mt_mask.sum() == expected_mt
            ), f"Failed to detect {expected_mt} MT genes in {gene_names[:4]}"
            assert (
                ribo_mask.sum() == expected_ribo
            ), f"Failed to detect {expected_ribo} ribo genes in {gene_names[:4]}"
