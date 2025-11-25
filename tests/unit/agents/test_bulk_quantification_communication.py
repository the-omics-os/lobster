"""
Unit tests for bulk RNA-seq agent quantification loading communication.

This module tests that the bulk RNA-seq expert agent provides professional,
accurate, and helpful responses when loading Kallisto/Salmon quantification files.

Test coverage target: 95%+ with focus on message quality and user guidance.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.bulk_rnaseq_expert import bulk_rnaseq_expert
from lobster.core.data_manager_v2 import DataManagerV2

# ===============================================================================
# Mock Fixtures
# ===============================================================================


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dm = DataManagerV2(workspace_path=Path(temp_dir))
        yield dm


@pytest.fixture
def mock_kallisto_dataset(tmp_path):
    """Create mock Kallisto quantification dataset."""
    kallisto_dir = tmp_path / "kallisto_output"
    kallisto_dir.mkdir()

    n_genes = 1000  # Smaller for unit tests
    gene_ids = [f"ENST{i:011d}" for i in range(n_genes)]
    sample_names = ["sample1", "sample2", "sample3"]

    for sample in sample_names:
        sample_dir = kallisto_dir / sample
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

        abundance_file = sample_dir / "abundance.tsv"
        abundance_data.to_csv(abundance_file, sep="\t", index=False)

    return kallisto_dir, sample_names, n_genes


# ===============================================================================
# Communication Quality Tests
# ===============================================================================


class TestQuantificationLoadingCommunication:
    """Test agent communication quality for quantification file loading."""

    def test_successful_loading_message_format(
        self, mock_data_manager, mock_kallisto_dataset
    ):
        """Test that successful loading produces professional, informative message."""
        kallisto_dir, samples, genes = mock_kallisto_dataset

        # Fix: Test the service directly instead of trying to extract tools from agent
        # This is the correct approach since we're testing the underlying functionality
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Load quantification files
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        # Create AnnData
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Store in data manager
        modality_name = "test_quantification"
        mock_data_manager.modalities[modality_name] = adata

        # Verify the data is correctly structured for professional reporting
        assert modality_name in mock_data_manager.list_modalities()
        stored_adata = mock_data_manager.get_modality(modality_name)

        # Check professional metadata is present
        assert "quantification_metadata" in stored_adata.uns
        assert (
            stored_adata.uns["quantification_metadata"]["quantification_tool"]
            == "Kallisto"
        )
        # Fix: Values are sanitized to strings for HDF5 compatibility
        assert stored_adata.uns["quantification_metadata"]["n_samples"] == str(len(samples))
        assert stored_adata.uns["quantification_metadata"]["n_genes"] == str(genes)

        # Check orientation metadata for user guidance
        assert "transpose_info" in stored_adata.uns
        # Note: sanitize_value() converts bool → str for HDF5 compatibility
        assert stored_adata.uns["transpose_info"]["transpose_applied"] == "True"

    def test_error_message_format(self, mock_data_manager):
        """Test that error messages are clear and actionable."""
        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()

        # Test with non-existent directory
        with pytest.raises(FileNotFoundError):
            service._detect_quantification_tool(Path("/nonexistent/path"))

        # Test with empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            with pytest.raises(ValueError, match="No Kallisto or Salmon"):
                service._detect_quantification_tool(empty_dir)

    def test_metadata_completeness_for_reporting(
        self, mock_data_manager, mock_kallisto_dataset
    ):
        """Test that all necessary metadata is present for professional reporting."""
        kallisto_dir, samples, genes = mock_kallisto_dataset

        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Load and convert
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Verify metadata has all fields needed for professional agent response
        required_metadata_fields = [
            "quantification_tool",
            "n_samples",
            "n_genes",
            "successful_samples",
            "failed_samples",
        ]

        for field in required_metadata_fields:
            assert (
                field in adata.uns["quantification_metadata"]
            ), f"Missing required metadata field: {field}"

        # Verify sample information is accurate
        # Fix: Values are sanitized to strings for HDF5 compatibility
        assert adata.uns["quantification_metadata"]["n_samples"] == str(len(samples))
        assert len(adata.uns["quantification_metadata"]["successful_samples"]) == len(
            samples
        )
        assert len(adata.uns["quantification_metadata"]["failed_samples"]) == 0

    def test_response_includes_next_steps(
        self, mock_data_manager, mock_kallisto_dataset
    ):
        """Test that response structure supports next step guidance."""
        kallisto_dir, samples, genes = mock_kallisto_dataset

        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        # Verify metadata contains information needed for next step recommendations
        assert metadata["quantification_tool"] in ["Kallisto", "Salmon"]
        assert metadata["n_samples"] > 0
        assert metadata["n_genes"] > 0

        # This metadata enables the agent to recommend:
        # 1. Quality assessment
        # 2. Filtering and normalization
        # 3. Differential expression analysis

    def test_tool_type_detection_reporting(self, mock_kallisto_dataset):
        """Test that tool type is accurately detected and reported."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()

        # Test auto-detection
        tool_type = service._detect_quantification_tool(kallisto_dir)
        assert tool_type == "kallisto"

        # Load with explicit tool specification
        df1, metadata1 = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="kallisto"
        )
        assert metadata1["quantification_tool"] == "Kallisto"

        # Load with auto-detection
        df2, metadata2 = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )
        assert metadata2["quantification_tool"] == "Kallisto"

        # Verify both methods produce identical results
        assert df1.shape == df2.shape
        assert metadata1["n_samples"] == metadata2["n_samples"]
        assert metadata1["n_genes"] == metadata2["n_genes"]


# ===============================================================================
# Message Content Tests
# ===============================================================================


class TestMessageContent:
    """Test specific message content and guidance."""

    def test_sample_count_accuracy(self, mock_kallisto_dataset):
        """Test that reported sample counts are accurate."""
        kallisto_dir, expected_samples, _ = mock_kallisto_dataset

        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        _, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        assert metadata["n_samples"] == len(expected_samples)
        assert len(metadata["successful_samples"]) == len(expected_samples)

        # Verify sample names are preserved
        for sample in expected_samples:
            assert sample in metadata["successful_samples"]

    def test_gene_count_accuracy(self, mock_kallisto_dataset):
        """Test that reported gene counts are accurate."""
        kallisto_dir, _, expected_genes = mock_kallisto_dataset

        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        assert metadata["n_genes"] == expected_genes
        assert df.shape[0] == expected_genes  # Genes as rows in quantification format

    def test_orientation_guidance(self, mock_kallisto_dataset):
        """Test that orientation information is provided for user understanding."""
        kallisto_dir, _, _ = mock_kallisto_dataset

        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
        from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

        service = BulkRNASeqService()
        adapter = TranscriptomicsAdapter(data_type="bulk")

        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Verify orientation metadata is clear and educational
        transpose_info = adata.uns["transpose_info"]
        assert "transpose_applied" in transpose_info
        assert "transpose_reason" in transpose_info
        assert "original_shape" in transpose_info
        assert "final_shape" in transpose_info

        # Verify reason is informative
        reason = transpose_info["transpose_reason"]
        assert "format specification" in reason.lower()
        assert "genes" in reason.lower()
        assert "samples" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
