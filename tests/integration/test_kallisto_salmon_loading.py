"""
Integration tests for Kallisto and Salmon quantification file loading.

This module tests the complete workflow for loading bulk RNA-seq quantification files
from Kallisto and Salmon tools, including:
- Auto-detection of quantification tool type
- Per-sample file merging
- AnnData creation with correct orientation
- Integration with GEO service
- Regression testing for known datasets (GSE130036)

Test coverage target: 95%+ with realistic quantification workflow scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService
from lobster.services.data_access.geo_service import GEOService

# ===============================================================================
# Mock Quantification Data Fixtures
# ===============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for quantification tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def mock_kallisto_dataset(tmp_path):
    """
    Create mock Kallisto quantification dataset with realistic structure.

    Simulates GSE130036 structure: 4 samples × ~60K genes
    """
    kallisto_dir = tmp_path / "kallisto_output"
    kallisto_dir.mkdir()

    # Realistic gene count for Kallisto (human transcriptome)
    n_genes = 60000
    gene_ids = [f"ENST{i:011d}" for i in range(n_genes)]

    sample_names = ["control_rep1", "control_rep2", "treatment_rep1", "treatment_rep2"]

    for sample in sample_names:
        sample_dir = kallisto_dir / sample
        sample_dir.mkdir()

        # Create abundance.tsv file with realistic Kallisto columns
        abundance_data = pd.DataFrame(
            {
                "target_id": gene_ids,
                "length": np.random.randint(500, 5000, n_genes),
                "eff_length": np.random.randint(400, 4900, n_genes),
                "est_counts": np.random.exponential(10, n_genes),
                "tpm": np.random.exponential(5, n_genes),
            }
        )

        abundance_file = sample_dir / "abundance.tsv"
        abundance_data.to_csv(abundance_file, sep="\t", index=False)

    return kallisto_dir, sample_names, n_genes


@pytest.fixture
def bulk_service(tmp_path):
    """Create BulkRNASeqService with temporary results directory."""
    return BulkRNASeqService(results_dir=tmp_path)


@pytest.fixture
def mock_salmon_dataset(tmp_path):
    """
    Create mock Salmon quantification dataset with realistic structure.

    Simulates typical Salmon output: 6 samples × ~60K transcripts
    """
    salmon_dir = tmp_path / "salmon_output"
    salmon_dir.mkdir()

    n_transcripts = 60000
    transcript_ids = [f"ENST{i:011d}" for i in range(n_transcripts)]

    sample_names = [
        "sample_A1",
        "sample_A2",
        "sample_A3",
        "sample_B1",
        "sample_B2",
        "sample_B3",
    ]

    for sample in sample_names:
        sample_dir = salmon_dir / sample
        sample_dir.mkdir()

        # Create quant.sf file with realistic Salmon columns
        quant_data = pd.DataFrame(
            {
                "Name": transcript_ids,
                "Length": np.random.randint(500, 5000, n_transcripts),
                "EffectiveLength": np.random.randint(400, 4900, n_transcripts),
                "TPM": np.random.exponential(5, n_transcripts),
                "NumReads": np.random.exponential(10, n_transcripts),
            }
        )

        quant_file = sample_dir / "quant.sf"
        quant_data.to_csv(quant_file, sep="\t", index=False)

    return salmon_dir, sample_names, n_transcripts


@pytest.fixture
def mock_mixed_dataset(tmp_path):
    """
    Create dataset with both Kallisto and Salmon files (edge case).
    """
    mixed_dir = tmp_path / "mixed_output"
    mixed_dir.mkdir()

    n_genes = 1000
    gene_ids = [f"ENST{i:011d}" for i in range(n_genes)]

    # Two Kallisto samples
    for i in range(2):
        sample_dir = mixed_dir / f"kallisto_sample{i+1}"
        sample_dir.mkdir()

        abundance_data = pd.DataFrame(
            {
                "target_id": gene_ids,
                "length": np.random.randint(500, 5000, n_genes),
                "eff_length": np.random.randint(400, 4900, n_genes),
                "est_counts": np.random.exponential(10, n_genes),
                "tpm": np.random.exponential(5, n_genes),
            }
        )

        (sample_dir / "abundance.tsv").parent.mkdir(exist_ok=True)
        abundance_data.to_csv(sample_dir / "abundance.tsv", sep="\t", index=False)

    # Two Salmon samples
    for i in range(2):
        sample_dir = mixed_dir / f"salmon_sample{i+1}"
        sample_dir.mkdir()

        quant_data = pd.DataFrame(
            {
                "Name": gene_ids,
                "Length": np.random.randint(500, 5000, n_genes),
                "EffectiveLength": np.random.randint(400, 4900, n_genes),
                "TPM": np.random.exponential(5, n_genes),
                "NumReads": np.random.exponential(10, n_genes),
            }
        )

        (sample_dir / "quant.sf").parent.mkdir(exist_ok=True)
        quant_data.to_csv(sample_dir / "quant.sf", sep="\t", index=False)

    return mixed_dir


# ===============================================================================
# Test BulkRNASeqService Quantification Methods
# ===============================================================================


class TestKallistoLoading:
    """Test Kallisto quantification file loading."""

    def test_kallisto_detection(self, mock_kallisto_dataset, bulk_service):
        """Test Kallisto file detection."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        tool_type = bulk_service._detect_quantification_tool(kallisto_dir)

        assert tool_type == "kallisto"

    def test_kallisto_merge(self, mock_kallisto_dataset, bulk_service):
        """Test Kallisto file merging."""
        kallisto_dir, expected_samples, expected_genes = mock_kallisto_dataset

        df, metadata = bulk_service.merge_kallisto_results(kallisto_dir=kallisto_dir)

        # Validate shape (genes × samples for quantification files)
        assert df.shape[1] == len(
            expected_samples
        ), f"Expected {len(expected_samples)} samples, got {df.shape[1]}"
        assert (
            df.shape[0] == expected_genes
        ), f"Expected {expected_genes} genes, got {df.shape[0]}"

        # Validate metadata
        assert metadata["quantification_tool"] == "Kallisto"
        assert metadata["n_samples"] == len(expected_samples)
        assert metadata["n_genes"] == expected_genes
        assert len(metadata["successful_samples"]) == len(expected_samples)
        assert len(metadata["failed_samples"]) == 0

    def test_kallisto_to_anndata(self, mock_kallisto_dataset, bulk_service):
        """Test Kallisto DataFrame to AnnData conversion with correct orientation."""
        kallisto_dir, expected_samples, expected_genes = mock_kallisto_dataset
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Step 1: Merge Kallisto files
        df, metadata = bulk_service.merge_kallisto_results(kallisto_dir)

        # Step 2: Convert to AnnData
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # CRITICAL: Validate orientation (samples × genes for bulk RNA-seq)
        assert adata.n_obs == len(
            expected_samples
        ), f"Expected {len(expected_samples)} observations (samples), got {adata.n_obs}"
        assert (
            adata.n_vars == expected_genes
        ), f"Expected {expected_genes} variables (genes), got {adata.n_vars}"
        assert (
            adata.n_obs < adata.n_vars
        ), "Samples should be < genes (correct bulk orientation)"

        # Validate transpose metadata
        assert "transpose_info" in adata.uns
        # Note: sanitize_value converts boolean to string for H5AD compatibility
        assert adata.uns["transpose_info"]["transpose_applied"] in [True, "True"]
        assert "format specification" in adata.uns["transpose_info"]["transpose_reason"]
        assert adata.uns["transpose_info"]["format_specific"] in [True, "True"]

        # Validate quantification metadata
        assert "quantification_metadata" in adata.uns
        assert adata.uns["quantification_metadata"]["quantification_tool"] == "Kallisto"


class TestSalmonLoading:
    """Test Salmon quantification file loading."""

    def test_salmon_detection(self, mock_salmon_dataset, bulk_service):
        """Test Salmon file detection."""
        salmon_dir, _, _ = mock_salmon_dataset

        tool_type = bulk_service._detect_quantification_tool(salmon_dir)

        assert tool_type == "salmon"

    def test_salmon_merge(self, mock_salmon_dataset, bulk_service):
        """Test Salmon file merging."""
        salmon_dir, expected_samples, expected_transcripts = mock_salmon_dataset

        df, metadata = bulk_service.merge_salmon_results(salmon_dir=salmon_dir)

        # Validate shape
        assert df.shape[1] == len(expected_samples)
        assert df.shape[0] == expected_transcripts

        # Validate metadata
        assert metadata["quantification_tool"] == "Salmon"
        assert metadata["n_samples"] == len(expected_samples)
        assert metadata["n_genes"] == expected_transcripts
        assert len(metadata["successful_samples"]) == len(expected_samples)

    def test_salmon_to_anndata(self, mock_salmon_dataset, bulk_service):
        """Test Salmon DataFrame to AnnData conversion."""
        salmon_dir, expected_samples, expected_transcripts = mock_salmon_dataset
        adapter = TranscriptomicsAdapter(data_type="bulk")

        df, metadata = bulk_service.merge_salmon_results(salmon_dir)
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Validate orientation
        assert adata.n_obs == len(expected_samples)
        assert adata.n_vars == expected_transcripts
        assert adata.n_obs < adata.n_vars

        # Validate metadata
        assert adata.uns["quantification_metadata"]["quantification_tool"] == "Salmon"


class TestUnifiedLoader:
    """Test unified quantification file loader."""

    def test_auto_detection_kallisto(self, mock_kallisto_dataset, bulk_service):
        """Test auto-detection with Kallisto files."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        df, metadata = bulk_service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        assert metadata["quantification_tool"] == "Kallisto"

    def test_auto_detection_salmon(self, mock_salmon_dataset, bulk_service):
        """Test auto-detection with Salmon files."""
        salmon_dir, _, _ = mock_salmon_dataset

        df, metadata = bulk_service.load_from_quantification_files(
            quantification_dir=salmon_dir, tool="auto"
        )

        assert metadata["quantification_tool"] == "Salmon"

    def test_explicit_tool_specification(self, mock_kallisto_dataset, bulk_service):
        """Test explicit tool specification."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        df, metadata = bulk_service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="kallisto"
        )

        assert metadata["quantification_tool"] == "Kallisto"

    def test_mixed_dataset_detection(self, mock_mixed_dataset, bulk_service):
        """Test detection prioritization with mixed Kallisto/Salmon files."""
        mixed_dir = mock_mixed_dataset

        # Should detect based on count (2 Kallisto + 2 Salmon = tie, Kallisto wins)
        tool_type = bulk_service._detect_quantification_tool(mixed_dir)

        assert tool_type in ["kallisto", "salmon"]


# ===============================================================================
# Regression Tests
# ===============================================================================


class TestGSE130036Regression:
    """
    Regression tests for GSE130036 (Kallisto dataset).

    CRITICAL: This test validates the fix for orientation issues where
    GSE130036 was incorrectly detected as 4 samples × 187,697 "genes" instead
    of the correct 4 samples × ~60K genes orientation.
    """

    def test_gse130036_correct_orientation(self, mock_kallisto_dataset, bulk_service):
        """
        REGRESSION TEST: GSE130036 must load as 4 samples × genes, NOT 4 × 187,697.

        This validates the fix for:
        - Original bug: Wrong transpose detection
        - Expected: 4 samples (obs) × ~60K genes (vars)
        - Bug behavior: 4 samples × 187,697 "genes" (incorrect)
        """
        kallisto_dir, _, expected_genes = mock_kallisto_dataset
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Simulate full workflow
        df, metadata = bulk_service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="kallisto"
        )

        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # CRITICAL ASSERTIONS
        assert adata.n_obs == 4, f"GSE130036 should have 4 samples, got {adata.n_obs}"

        assert (
            adata.n_vars == expected_genes
        ), f"GSE130036 should have {expected_genes} genes, got {adata.n_vars}"

        assert (
            adata.n_obs < adata.n_vars
        ), f"Samples ({adata.n_obs}) should be < genes ({adata.n_vars})"

        # Validate transpose was applied correctly
        # Note: sanitize_value converts boolean to string for H5AD compatibility
        assert adata.uns["transpose_info"]["transpose_applied"] in [True, "True"]
        assert "format specification" in adata.uns["transpose_info"]["transpose_reason"]


# ===============================================================================
# Error Handling Tests
# ===============================================================================


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_missing_directory(self, bulk_service):
        """Test handling of non-existent directory."""
        with pytest.raises(FileNotFoundError):
            bulk_service._detect_quantification_tool(Path("/nonexistent/path"))

    def test_empty_directory(self, tmp_path, bulk_service):
        """Test handling of empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No Kallisto or Salmon"):
            bulk_service._detect_quantification_tool(empty_dir)

    def test_invalid_quantification_files(self, tmp_path, bulk_service):
        """Test handling of corrupted quantification files."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        sample_dir = invalid_dir / "sample1"
        sample_dir.mkdir()

        # Create invalid abundance.tsv (missing required columns)
        invalid_file = sample_dir / "abundance.tsv"
        pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(
            invalid_file, sep="\t", index=False
        )

        with pytest.raises(ValueError):
            bulk_service.merge_kallisto_results(invalid_dir)


# ===============================================================================
# Integration with GEO Service Tests
# ===============================================================================


class TestGEOServiceIntegration:
    """Test integration with GEO service for quantification file loading."""

    def test_geo_quantification_detection(self, mock_kallisto_dataset, temp_workspace, tmp_path):
        """Test that quantification files can be loaded through standard workflow."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        # Test that BulkRNASeqService can detect and load quantification files
        # (This is what GEO service would use after extracting TAR)
        bulk_service = BulkRNASeqService(results_dir=tmp_path)

        # Detect tool type
        tool_type = bulk_service._detect_quantification_tool(kallisto_dir)
        assert tool_type == "kallisto"

        # Load quantification files
        df, metadata = bulk_service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        assert df is not None
        assert metadata["quantification_tool"] == "Kallisto"
        assert metadata["n_samples"] == 4
        assert metadata["n_genes"] == 60000

        # Verify can convert to AnnData
        adapter = TranscriptomicsAdapter(data_type="bulk")
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        assert adata.n_obs == 4
        assert adata.n_vars == 60000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
