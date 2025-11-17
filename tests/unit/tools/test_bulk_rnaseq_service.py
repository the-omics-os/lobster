"""
Comprehensive unit tests for bulk RNA-seq service.

This module provides thorough testing of the bulk RNA-seq service including
quality control, quantification, differential expression analysis,
and pathway enrichment for bulk RNA-seq data analysis.

Test coverage target: 95%+ with meaningful tests for bulk RNA-seq operations.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, mock_open, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core import DesignMatrixError, FormulaError, PseudobulkError
from lobster.tools.bulk_rnaseq_service import (
    BulkRNASeqError,
    BulkRNASeqService,
    PyDESeq2Error,
)
from lobster.tools.differential_formula_service import DifferentialFormulaService
from tests.mock_data.base import LARGE_DATASET_CONFIG, SMALL_DATASET_CONFIG
from tests.mock_data.factories import BulkRNASeqDataFactory, SingleCellDataFactory

# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================


@pytest.fixture
def mock_bulk_data():
    """Create mock bulk RNA-seq data for testing."""
    return BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)


@pytest.fixture
def mock_fastq_files():
    """Create mock FASTQ file paths for testing."""
    return [
        "/path/to/sample1_R1.fastq.gz",
        "/path/to/sample1_R2.fastq.gz",
        "/path/to/sample2_R1.fastq.gz",
        "/path/to/sample2_R2.fastq.gz",
    ]


@pytest.fixture
def mock_salmon_results():
    """Create mock Salmon quantification results."""
    n_genes = 1000
    n_samples = 6

    # Generate realistic count matrix
    np.random.seed(42)
    counts = np.random.negative_binomial(100, 0.3, size=(n_genes, n_samples))

    genes = [f"ENSG{str(i).zfill(11)}" for i in range(n_genes)]
    samples = [f"sample_{i}" for i in range(n_samples)]

    count_matrix = pd.DataFrame(counts, index=genes, columns=samples)

    return {
        "count_matrix": count_matrix,
        "tpm_matrix": count_matrix / count_matrix.sum() * 1e6,
        "gene_lengths": pd.Series(np.random.uniform(500, 5000, n_genes), index=genes),
    }


@pytest.fixture
def mock_design_matrix():
    """Create mock experimental design matrix."""
    return pd.DataFrame(
        {
            "condition": [
                "control",
                "control",
                "control",
                "treatment",
                "treatment",
                "treatment",
            ],
            "batch": ["batch1", "batch1", "batch2", "batch1", "batch2", "batch2"],
            "replicate": [1, 2, 3, 1, 2, 3],
        },
        index=["sample_0", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"],
    )


@pytest.fixture
def bulk_adata_with_groups():
    """Create AnnData for bulk RNA-seq with group assignments (NEW API)."""
    n_samples, n_genes = 6, 1000

    # Generate realistic count data
    np.random.seed(42)
    counts = np.random.negative_binomial(100, 0.3, size=(n_samples, n_genes))

    # Create AnnData
    adata = ad.AnnData(X=counts.astype(float))
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"ENSG{str(i).zfill(11)}" for i in range(n_genes)]

    # Add group assignments (required for new API)
    adata.obs["condition"] = pd.Categorical(
        ["control", "control", "control", "treatment", "treatment", "treatment"]
    )
    adata.obs["batch"] = pd.Categorical(
        ["batch1", "batch1", "batch2", "batch1", "batch2", "batch2"]
    )
    adata.obs["replicate"] = [1, 2, 3, 1, 2, 3]

    # Add gene metadata
    adata.var["gene_id"] = adata.var_names
    adata.var["gene_name"] = [f"GENE_{i}" for i in range(n_genes)]

    return adata


@pytest.fixture
def bulk_service():
    """Create BulkRNASeqService instance for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = BulkRNASeqService(results_dir=Path(tmp_dir))
        yield service


@pytest.fixture
def mock_pydeseq2_results():
    """Create mock pyDESeq2 results."""
    n_genes = 1000
    np.random.seed(42)

    return pd.DataFrame(
        {
            "gene_id": [f"ENSG{str(i).zfill(11)}" for i in range(n_genes)],
            "baseMean": np.random.uniform(10, 1000, n_genes),
            "log2FoldChange": np.random.normal(0, 1.5, n_genes),
            "lfcSE": np.random.uniform(0.1, 0.5, n_genes),
            "stat": np.random.normal(0, 2, n_genes),
            "pvalue": np.random.uniform(0, 1, n_genes),
            "padj": np.random.uniform(0, 1, n_genes),
        }
    )


# ===============================================================================
# BulkRNASeqService Core Tests
# ===============================================================================


@pytest.mark.unit
class TestBulkRNASeqServiceCore:
    """Test bulk RNA-seq service core functionality."""

    def test_service_initialization_default(self):
        """Test BulkRNASeqService initialization with default parameters."""
        with patch("pathlib.Path.mkdir"):
            service = BulkRNASeqService()

            assert hasattr(service, "results_dir")
            assert hasattr(service, "formula_service")
            assert isinstance(service.formula_service, DifferentialFormulaService)
            assert service.results_dir.name == "bulk_results"

    def test_service_initialization_custom_dir(self):
        """Test BulkRNASeqService initialization with custom directory."""
        custom_dir = Path("/custom/results")

        with patch("pathlib.Path.mkdir"):
            service = BulkRNASeqService(results_dir=custom_dir)

            assert service.results_dir == custom_dir

    def test_service_initialization_creates_directories(self):
        """Test that service creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "bulk_results"
            service = BulkRNASeqService(results_dir=results_dir)

            assert results_dir.exists()
            assert results_dir.is_dir()


# ===============================================================================
# FastQC Quality Control Tests
# ===============================================================================


@pytest.mark.unit
class TestFastQCAnalysis:
    """Test FastQC quality control functionality."""

    def test_run_fastqc_valid_files(self, bulk_service, mock_fastq_files):
        """Test FastQC execution with valid files."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
            patch("subprocess.run") as mock_run,
            patch.object(
                bulk_service, "_parse_fastqc_results", return_value="Quality: Good"
            ),
        ):

            mock_run.return_value = Mock(
                returncode=0, stdout="FastQC completed", stderr=""
            )

            result = bulk_service.run_fastqc(mock_fastq_files)

            assert "FastQC Analysis Complete!" in result
            assert "**Files Analyzed:** 4" in result
            assert "Quality: Good" in result
            mock_run.assert_called_once()

    def test_run_fastqc_missing_files(self, bulk_service):
        """Test FastQC with missing files."""
        missing_files = ["/nonexistent/file1.fastq", "/nonexistent/file2.fastq"]

        with patch("os.path.exists", return_value=False):
            result = bulk_service.run_fastqc(missing_files)

            assert "No valid FASTQ files found" in result

    def test_run_fastqc_command_failure(self, bulk_service, mock_fastq_files):
        """Test FastQC command failure handling."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
            patch("subprocess.run") as mock_run,
        ):

            mock_run.return_value = Mock(
                returncode=1, stdout="", stderr="FastQC error: Invalid file format"
            )

            result = bulk_service.run_fastqc(mock_fastq_files)

            assert "Error running FastQC" in result

    def test_run_fastqc_timeout(self, bulk_service, mock_fastq_files):
        """Test FastQC timeout handling."""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
            patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("fastqc", 300)
            ),
        ):

            result = bulk_service.run_fastqc(mock_fastq_files)

            assert "timed out" in result.lower()

    def test_parse_fastqc_results(self, bulk_service):
        """Test FastQC results parsing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            qc_dir = Path(tmp_dir)

            # Create mock FastQC output files
            (qc_dir / "sample1_fastqc.html").touch()
            (qc_dir / "sample2_fastqc.html").touch()

            result = bulk_service._parse_fastqc_results(qc_dir)

            assert isinstance(result, str)
            assert len(result) > 0


# ===============================================================================
# MultiQC Aggregation Tests
# ===============================================================================


@pytest.mark.unit
class TestMultiQCAnalysis:
    """Test MultiQC aggregation functionality."""

    def test_run_multiqc_default_dir(self, bulk_service):
        """Test MultiQC with default input directory."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0, stdout="MultiQC complete", stderr=""
            )

            result = bulk_service.run_multiqc()

            assert "MultiQC Analysis Complete!" in result
            mock_run.assert_called_once()

    def test_run_multiqc_custom_dir(self, bulk_service):
        """Test MultiQC with custom input directory."""
        custom_dir = "/custom/qc/dir"

        with (
            patch("os.path.exists", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(
                returncode=0, stdout="MultiQC complete", stderr=""
            )

            result = bulk_service.run_multiqc(input_dir=custom_dir)

            assert "MultiQC Analysis Complete!" in result
            # Verify custom directory was used in command
            args, kwargs = mock_run.call_args
            assert custom_dir in args[0]

    def test_run_multiqc_failure(self, bulk_service):
        """Test MultiQC command failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1, stdout="", stderr="MultiQC error: No valid input files"
            )

            result = bulk_service.run_multiqc()

            assert "Error running MultiQC" in result


# ===============================================================================
# Salmon Quantification Tests
# ===============================================================================


@pytest.mark.unit
class TestSalmonQuantification:
    """Test Salmon quantification functionality."""

    def test_run_salmon_quantification_basic(self, bulk_service, mock_fastq_files):
        """Test basic Salmon quantification."""
        transcriptome_index = "/path/to/transcriptome/index"
        sample_names = ["sample1", "sample2"]

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
            patch("subprocess.run") as mock_run,
            patch.object(bulk_service, "_combine_salmon_results") as mock_combine,
        ):

            mock_run.return_value = Mock(
                returncode=0, stdout="Salmon complete", stderr=""
            )
            mock_combine.return_value = "Combined results successfully"

            result = bulk_service.run_salmon_quantification(
                fastq_files=mock_fastq_files,
                transcriptome_index=transcriptome_index,
                sample_names=sample_names,
            )

            assert "Salmon Quantification Complete!" in result
            assert len(sample_names) == mock_run.call_count

    def test_run_salmon_quantification_mismatched_inputs(self, bulk_service):
        """Test Salmon with mismatched FASTQ files and sample names."""
        fastq_files = ["/path/to/sample1.fastq"]
        sample_names = ["sample1", "sample2"]  # More names than files

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
        ):
            result = bulk_service.run_salmon_quantification(
                fastq_files=fastq_files,
                transcriptome_index="/path/to/index",
                sample_names=sample_names,
            )

        # Should get an error due to mismatched inputs, but since we mock file existence
        # the service continues and checks other aspects
        assert (
            "Error" in result
            or "mismatch" in result.lower()
            or "Salmon index not found" in result
        )

    def test_combine_salmon_results(self, bulk_service):
        """Test combining Salmon quantification results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            salmon_dir = Path(tmp_dir)
            sample_names = ["sample1", "sample2"]

            # Create mock Salmon output directories
            for sample in sample_names:
                sample_dir = salmon_dir / sample
                sample_dir.mkdir()

                # Create mock quant.sf files
                quant_data = pd.DataFrame(
                    {
                        "Name": [f"transcript_{i}" for i in range(100)],
                        "Length": np.random.randint(200, 3000, 100),
                        "EffectiveLength": np.random.randint(150, 2900, 100),
                        "TPM": np.random.uniform(0, 100, 100),
                        "NumReads": np.random.randint(0, 1000, 100),
                    }
                )
                quant_data.to_csv(sample_dir / "quant.sf", sep="\t", index=False)

            result = bulk_service._combine_salmon_results(salmon_dir, sample_names)

            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] > 0  # Should have transcripts (rows)
            assert result.shape[1] == len(sample_names)  # Should have samples (columns)


# ===============================================================================
# Differential Expression Tests
# ===============================================================================


@pytest.mark.unit
class TestDifferentialExpression:
    """Test differential expression analysis."""

    def test_run_differential_expression_basic(self, bulk_service, bulk_adata_with_groups):
        """Test basic differential expression analysis (NEW API)."""
        # New API: returns tuple (adata_de, de_stats) instead of string
        adata_de, de_stats = bulk_service.run_differential_expression_analysis(
            adata=bulk_adata_with_groups,
            groupby="condition",
            group1="control",
            group2="treatment",
            method="deseq2_like",
            min_expression_threshold=1.0,
        )

        # Verify return types
        assert isinstance(adata_de, ad.AnnData), "Should return AnnData object"
        assert isinstance(de_stats, dict), "Should return stats dict"

        # Verify DE results are stored in AnnData (key includes comparison name)
        de_keys = [k for k in adata_de.uns.keys() if k.startswith("de_results")]
        assert len(de_keys) > 0, "DE results should be in .uns with 'de_results_*' key"

        # Verify stats dict structure
        assert "n_genes_tested" in de_stats or "n_de_genes" in de_stats
        assert "group1" in de_stats and "group2" in de_stats
        assert de_stats["group1"] == "control" and de_stats["group2"] == "treatment"

    def test_run_differential_expression_invalid_group(
        self, bulk_service, bulk_adata_with_groups
    ):
        """Test differential expression with invalid group name (NEW API)."""
        # Test with non-existent group name - should raise error
        with pytest.raises((BulkRNASeqError, KeyError, ValueError)):
            bulk_service.run_differential_expression_analysis(
                adata=bulk_adata_with_groups,
                groupby="condition",
                group1="control",
                group2="invalid_group",  # This group doesn't exist
                method="deseq2_like",
            )

    def test_deseq2_like_analysis(self, bulk_service, bulk_adata_with_groups):
        """Test DESeq2-like analysis implementation (NEW API)."""
        # Split data into groups (new private method signature)
        control_mask = bulk_adata_with_groups.obs["condition"] == "control"
        treatment_mask = bulk_adata_with_groups.obs["condition"] == "treatment"

        group1_data = bulk_adata_with_groups[control_mask].copy()
        group2_data = bulk_adata_with_groups[treatment_mask].copy()

        # Call private method with new signature
        result = bulk_service._run_deseq2_like_analysis(
            group1_data=group1_data,
            group2_data=group2_data,
            group1_name="control",
            group2_name="treatment",
        )

        # Verify result is a DataFrame (new return type)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Should have results"
        assert "log2FoldChange" in result.columns or "logFC" in result.columns
        assert "pvalue" in result.columns or "p_value" in result.columns

    def test_wilcoxon_test_analysis(self, bulk_service, bulk_adata_with_groups):
        """Test Wilcoxon rank-sum test for differential expression (NEW API)."""
        # Split data into groups
        control_mask = bulk_adata_with_groups.obs["condition"] == "control"
        treatment_mask = bulk_adata_with_groups.obs["condition"] == "treatment"

        group1_data = bulk_adata_with_groups[control_mask].copy()
        group2_data = bulk_adata_with_groups[treatment_mask].copy()

        # Call private method with new signature
        result = bulk_service._run_wilcoxon_test(
            group1_data=group1_data,
            group2_data=group2_data,
            group1_name="control",
            group2_name="treatment",
        )

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Should have results"
        assert "pvalue" in result.columns or "p_value" in result.columns

    def test_ttest_analysis(self, bulk_service, bulk_adata_with_groups):
        """Test t-test analysis for differential expression (NEW API)."""
        # Split data into groups
        control_mask = bulk_adata_with_groups.obs["condition"] == "control"
        treatment_mask = bulk_adata_with_groups.obs["condition"] == "treatment"

        group1_data = bulk_adata_with_groups[control_mask].copy()
        group2_data = bulk_adata_with_groups[treatment_mask].copy()

        # Call private method with new signature
        result = bulk_service._run_ttest_analysis(
            group1_data=group1_data,
            group2_data=group2_data,
            group1_name="control",
            group2_name="treatment",
        )

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Should have results"
        assert "pvalue" in result.columns or "p_value" in result.columns


# ===============================================================================
# PyDESeq2 Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestPyDESeq2Integration:
    """Test pyDESeq2 integration functionality."""

    def test_validate_pydeseq2_setup(self, bulk_service):
        """Test pyDESeq2 setup validation."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = Mock()  # Package found

            result = bulk_service.validate_pydeseq2_setup()

            assert isinstance(result, dict)
            assert "pydeseq2_available" in result

    def test_run_pydeseq2_analysis(
        self, bulk_service, mock_salmon_results, mock_design_matrix
    ):
        """Test running pyDESeq2 analysis."""
        with patch.object(bulk_service, "validate_pydeseq2_setup") as mock_validate:
            mock_validate.return_value = {
                "pydeseq2": True,
                "pydeseq2_inference": True,
                "numba": True,
                "statsmodels": True,
            }

            with patch("pydeseq2.dds.DeseqDataSet") as mock_dds_class:
                with patch("pydeseq2.ds.DeseqStats") as mock_ds_class:
                    with patch("pydeseq2.default_inference.DefaultInference"):
                        # Mock the results
                        mock_ds = Mock()
                        mock_ds.results_df = pd.DataFrame(
                            {
                                "baseMean": [100, 200],
                                "log2FoldChange": [1.5, -0.8],
                                "pvalue": [0.01, 0.05],
                                "padj": [0.05, 0.15],
                            }
                        )
                        mock_ds_class.return_value = mock_ds

                        result = bulk_service.run_pydeseq2_analysis(
                            count_matrix=mock_salmon_results["count_matrix"],
                            metadata=mock_design_matrix,
                            formula="~ condition",
                            contrast=["condition", "treatment", "control"],
                        )

                        # The method returns a DataFrame, not a string
                        assert isinstance(result, pd.DataFrame)
                        assert len(result) == 2
                        assert "baseMean" in result.columns
                        assert "log2FoldChange" in result.columns

    def test_run_pydeseq2_from_pseudobulk(self, bulk_service):
        """Test running pyDESeq2 from pseudobulk data."""
        # Create mock pseudobulk data
        pseudobulk_data = ad.AnnData(
            X=np.random.negative_binomial(
                10, 0.3, size=(20, 100)
            ),  # 20 samples × 100 genes
            obs=pd.DataFrame(
                {
                    "sample_id": [f"sample_{i}" for i in range(20)],
                    "condition": ["control"] * 10 + ["treatment"] * 10,
                    "cell_type": ["T_cells"] * 20,
                }
            ),
        )

        with patch.object(bulk_service, "run_pydeseq2_analysis") as mock_pydeseq2:
            mock_results_df = pd.DataFrame(
                {
                    "baseMean": [100, 200],
                    "log2FoldChange": [1.5, -0.8],
                    "pvalue": [0.01, 0.05],
                    "padj": [0.05, 0.15],
                }
            )
            mock_pydeseq2.return_value = mock_results_df

            result_df, stats = bulk_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata=pseudobulk_data,
                formula="~ condition",
                contrast=["condition", "treatment", "control"],
            )

            assert isinstance(result_df, pd.DataFrame)
            assert isinstance(stats, dict)

    def test_pydeseq2_unavailable_handling(
        self, bulk_service, mock_salmon_results, mock_design_matrix
    ):
        """Test handling when pyDESeq2 is not available."""
        with patch(
            "importlib.import_module", side_effect=ImportError("pyDESeq2 not found")
        ):

            with pytest.raises(PyDESeq2Error):
                bulk_service.run_pydeseq2_analysis(
                    count_matrix=mock_salmon_results["count_matrix"],
                    metadata=mock_design_matrix,
                    formula="~ condition",
                )


# ===============================================================================
# Formula and Design Matrix Tests
# ===============================================================================


@pytest.mark.unit
class TestFormulaDesign:
    """Test formula construction and design matrix functionality."""

    def test_create_formula_design(self, bulk_service, mock_design_matrix):
        """Test formula and design matrix creation."""
        result = bulk_service.create_formula_design(
            metadata=mock_design_matrix,
            condition_column="condition",
            batch_column="batch",
        )

        # Result contains design matrix information from formula_service
        assert isinstance(result, dict)
        assert "design_df" in result or "coefficient_names" in result

    def test_validate_experimental_design(self, bulk_service, mock_design_matrix):
        """Test experimental design validation."""
        result = bulk_service.validate_experimental_design(
            metadata=mock_design_matrix, formula="~ condition"
        )

        assert "valid" in result
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_validate_deseq2_inputs(
        self, bulk_service, mock_salmon_results, mock_design_matrix
    ):
        """Test DESeq2 input validation."""
        # Should not raise any exceptions for valid inputs
        result = bulk_service._validate_deseq2_inputs(
            count_matrix=mock_salmon_results["count_matrix"],
            metadata=mock_design_matrix,
            formula="~ condition",
            contrast=["condition", "treatment", "control"],
        )

        # Method returns None on success, raises exception on failure
        assert result is None

    def test_formula_design_invalid_column(self, bulk_service, mock_design_matrix):
        """Test formula design with invalid column."""
        # This should raise a FormulaError for invalid column
        with pytest.raises(FormulaError, match="Variables not found in metadata"):
            bulk_service.create_formula_design(
                metadata=mock_design_matrix, condition_col="nonexistent_column"
            )


# ===============================================================================
# Pathway Enrichment Tests
# ===============================================================================


@pytest.mark.unit
class TestPathwayEnrichment:
    """Test pathway enrichment analysis."""

    def test_run_pathway_enrichment(self, bulk_service, mock_pydeseq2_results):
        """Test pathway enrichment analysis with GSEApy."""
        from unittest.mock import MagicMock

        gene_list = mock_pydeseq2_results["gene_id"].head(50).tolist()

        # Test empty gene list
        with pytest.raises(BulkRNASeqError, match="Empty gene list"):
            bulk_service.run_pathway_enrichment(gene_list=[], organism="human")

        # Test with GSEApy installed (mock the enrichr function and imports)
        mock_enrichr_result = MagicMock()
        mock_enrichr_result.results = pd.DataFrame(
            {
                "Term": ["Pathway1", "Pathway2"],
                "Overlap": ["10/100", "5/50"],
                "P-value": [0.001, 0.01],
                "Adjusted P-value": [0.01, 0.05],
                "Genes": ["GENE1;GENE2", "GENE3;GENE4"],
            }
        )

        # Mock the gseapy module and enrichr function
        mock_gseapy = MagicMock()
        mock_gseapy.enrichr = MagicMock(return_value=mock_enrichr_result)

        with patch.dict("sys.modules", {"gseapy": mock_gseapy}):
            enrichment_df, stats = bulk_service.run_pathway_enrichment(
                gene_list=gene_list, analysis_type="GO", organism="human"
            )

            assert isinstance(enrichment_df, pd.DataFrame)
            assert isinstance(stats, dict)
            assert "n_significant_terms" in stats
            assert "enrichment_database" in stats
            assert stats["enrichment_database"] == "GO"
            assert stats["n_genes_input"] == 50
            # Verify gseapy.enrichr was called
            mock_gseapy.enrichr.assert_called()


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================


@pytest.mark.unit
class TestBulkRNASeqErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_count_matrix(self, bulk_service):
        """Test handling of empty AnnData."""
        # Create truly empty AnnData (0 samples, 0 genes)
        empty_adata = ad.AnnData(X=np.zeros((0, 0)))
        empty_adata.obs["condition"] = pd.Categorical([])

        # Should raise error (IndexError or KeyError from empty data)
        with pytest.raises((ValueError, IndexError, BulkRNASeqError, KeyError)):
            bulk_service.run_differential_expression_analysis(
                adata=empty_adata,
                groupby="condition",
                group1="control",
                group2="treatment",
            )

    def test_mismatched_samples(self, bulk_service):
        """Test handling of mismatched group labels."""
        # Create AnnData where group labels don't match obs values
        n_samples, n_genes = 6, 100
        X = np.random.negative_binomial(100, 0.3, size=(n_samples, n_genes))
        adata = ad.AnnData(X=X.astype(float))
        adata.obs["condition"] = pd.Categorical(
            ["control", "control", "control", "treatment", "treatment", "treatment"]
        )

        # Try to use non-existent group name
        with pytest.raises((BulkRNASeqError, KeyError, ValueError)):
            bulk_service.run_differential_expression_analysis(
                adata=adata,
                groupby="condition",
                group1="control",
                group2="invalid_group",  # This group doesn't exist
            )

    def test_insufficient_replicates(self, bulk_service):
        """Test handling of insufficient replicates."""
        # Create AnnData with only 1 replicate per group
        n_samples, n_genes = 2, 100
        X = np.random.negative_binomial(100, 0.3, size=(n_samples, n_genes))
        adata = ad.AnnData(X=X.astype(float))
        adata.obs["condition"] = pd.Categorical(["control", "treatment"])

        # Should still run (returns tuple) but may have warnings
        result = bulk_service.run_differential_expression_analysis(
            adata=adata,
            groupby="condition",
            group1="control",
            group2="treatment",
        )

        # Should return tuple even with insufficient replicates
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], dict)  # stats dict

    def test_invalid_results_directory(self):
        """Test handling of invalid results directory."""
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                BulkRNASeqService(results_dir=Path("/invalid/path"))

    def test_concurrent_analysis_safety(
        self, bulk_service, bulk_adata_with_groups
    ):
        """Test thread safety for concurrent analyses."""
        import threading
        import time

        results = []
        errors = []

        def analysis_worker(worker_id):
            """Worker function for concurrent analysis."""
            try:
                with patch.object(
                    bulk_service, "_run_deseq2_like_analysis"
                ) as mock_analysis:
                    # Mock return must include all expected columns
                    mock_analysis.return_value = pd.DataFrame({
                        "gene_id": [f"GENE_{worker_id}"],
                        "log2FoldChange": [1.5],
                        "pvalue": [0.01],
                        "padj": [0.05],  # Required column
                        "baseMean": [100.0],
                    })

                    result = bulk_service.run_differential_expression_analysis(
                        adata=bulk_adata_with_groups.copy(),
                        groupby="condition",
                        group1="control",
                        group2="treatment",
                    )
                    results.append((worker_id, result))
                    time.sleep(0.01)

            except Exception as e:
                errors.append((worker_id, e))

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=analysis_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent analysis errors: {errors}"
        assert len(results) == 3


# ===============================================================================
# Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestBulkRNASeqIntegration:
    """Test integration between different service components."""

    def test_end_to_end_workflow_simulation(
        self, bulk_service, mock_fastq_files, mock_design_matrix
    ):
        """Test simulated end-to-end workflow."""
        # Mock all external dependencies
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000000),
            patch("subprocess.run") as mock_subprocess,
            patch.object(
                bulk_service, "_parse_fastqc_results", return_value="Good quality"
            ),
            patch.object(
                bulk_service,
                "_combine_salmon_results",
                return_value="Combined successfully",
            ),
        ):

            mock_subprocess.return_value = Mock(
                returncode=0, stdout="Success", stderr=""
            )

            # Step 1: QC
            qc_result = bulk_service.run_fastqc(mock_fastq_files)
            assert "FastQC Analysis Complete!" in qc_result

            # Step 2: MultiQC
            multiqc_result = bulk_service.run_multiqc()
            assert "MultiQC Analysis Complete!" in multiqc_result

            # Step 3: Quantification
            quant_result = bulk_service.run_salmon_quantification(
                fastq_files=mock_fastq_files,
                transcriptome_index="/path/to/index",
                sample_names=["sample1", "sample2"],
            )
            assert "Salmon Quantification Complete!" in quant_result

    def test_formula_service_integration(self, bulk_service, mock_design_matrix):
        """Test integration with DifferentialFormulaService."""
        # The service should use the formula service for design validation
        assert isinstance(bulk_service.formula_service, DifferentialFormulaService)

        # Test that formula service methods are accessible
        result = bulk_service.create_formula_design(
            metadata=mock_design_matrix, condition_column="condition"
        )

        # Result contains design matrix information
        assert isinstance(result, dict)
        assert "design_df" in result or "coefficient_names" in result

    def test_enhance_deseq2_results(self, bulk_service, mock_pydeseq2_results):
        """Test enhancement of DESeq2 results with additional annotations."""
        # Mock the dds object
        mock_dds = Mock()
        contrast = ["condition", "treatment", "control"]

        enhanced = bulk_service._enhance_deseq2_results(
            results_df=mock_pydeseq2_results,
            dds=mock_dds,
            contrast=contrast,
        )

        assert isinstance(enhanced, pd.DataFrame)
        assert len(enhanced) > 0
        assert "contrast" in enhanced.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
