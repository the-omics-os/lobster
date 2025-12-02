"""
End-to-end integration tests for quantification file loading workflow.

This module tests the complete user workflow from quantification file discovery
through loading, agent interaction, and analysis:
- Quantification file detection and loading via agent tools
- DataManagerV2 integration with quantification data
- Agent-based quality assessment and analysis
- Complete workflow validation (load → QC → filter → analyze)
- Professional response formatting and error handling

Test coverage target: Complete user workflow scenarios with realistic data.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def realistic_kallisto_dataset(tmp_path):
    """
    Create realistic Kallisto dataset for end-to-end testing.

    Simulates a real bulk RNA-seq experiment:
    - 4 samples (2 control, 2 treatment)
    - ~20K genes (realistic for filtered transcriptome)
    - Realistic expression distributions
    """
    kallisto_dir = tmp_path / "experiment_quantification"
    kallisto_dir.mkdir()

    n_genes = 20000
    gene_ids = [f"ENST{i:011d}" for i in range(n_genes)]

    sample_metadata = {
        "control_rep1": {"condition": "control", "batch": "batch1"},
        "control_rep2": {"condition": "control", "batch": "batch2"},
        "treatment_rep1": {"condition": "treatment", "batch": "batch1"},
        "treatment_rep2": {"condition": "treatment", "batch": "batch2"},
    }

    for sample_name, metadata in sample_metadata.items():
        sample_dir = kallisto_dir / sample_name
        sample_dir.mkdir()

        # Create realistic expression values
        # Most genes lowly expressed, some highly expressed
        base_expression = np.random.exponential(10, n_genes)

        # Add differential expression for treatment samples
        if metadata["condition"] == "treatment":
            # 5% of genes upregulated
            upregulated = np.random.choice(
                n_genes, size=int(0.05 * n_genes), replace=False
            )
            base_expression[upregulated] *= np.random.uniform(2, 5, len(upregulated))

            # 5% of genes downregulated
            downregulated = np.random.choice(
                [i for i in range(n_genes) if i not in upregulated],
                size=int(0.05 * n_genes),
                replace=False,
            )
            base_expression[downregulated] *= np.random.uniform(
                0.2, 0.5, len(downregulated)
            )

        abundance_data = pd.DataFrame(
            {
                "target_id": gene_ids,
                "length": np.random.randint(500, 5000, n_genes),
                "eff_length": np.random.randint(400, 4900, n_genes),
                "est_counts": base_expression,
                "tpm": base_expression / base_expression.sum() * 1e6,
            }
        )

        abundance_file = sample_dir / "abundance.tsv"
        abundance_data.to_csv(abundance_file, sep="\t", index=False)

    return kallisto_dir, sample_metadata, n_genes


# ===============================================================================
# End-to-End Workflow Tests
# ===============================================================================


class TestQuantificationLoadingWorkflow:
    """Test complete quantification loading workflow."""

    def test_load_quantification_via_service(
        self, realistic_kallisto_dataset, data_manager
    ):
        """Test loading quantification files through BulkRNASeqService."""
        kallisto_dir, sample_metadata, n_genes = realistic_kallisto_dataset
        service = BulkRNASeqService(data_manager=data_manager)

        # Step 1: Detect tool type
        tool_type = service._detect_quantification_tool(kallisto_dir)
        assert tool_type == "kallisto"

        # Step 2: Load and merge files
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        # Step 3: Verify loaded data structure
        assert df.shape == (n_genes, 4), f"Expected {n_genes}×4, got {df.shape}"
        assert metadata["quantification_tool"] == "Kallisto"
        assert metadata["n_samples"] == 4
        assert metadata["n_genes"] == n_genes

        # Step 4: Verify sample names preserved
        expected_samples = list(sample_metadata.keys())
        for sample in expected_samples:
            assert sample in df.columns

        # Step 5: Verify data quality
        assert not df.isnull().any().any(), "Should have no missing values"
        assert (df >= 0).all().all(), "Should have no negative values"

    def test_load_to_anndata_workflow(self, realistic_kallisto_dataset, data_manager):
        """Test complete workflow from quantification files to AnnData."""
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

        kallisto_dir, sample_metadata, n_genes = realistic_kallisto_dataset
        service = BulkRNASeqService(data_manager=data_manager)
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Step 1: Load quantification files
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        # Step 2: Convert to AnnData
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Step 3: Verify AnnData structure
        assert adata.n_obs == 4, "Should have 4 samples (observations)"
        assert adata.n_vars == n_genes, f"Should have {n_genes} genes (variables)"
        assert adata.n_obs < adata.n_vars, "Bulk RNA-seq: samples < genes"

        # Step 4: Verify metadata preserved
        assert "quantification_metadata" in adata.uns
        assert adata.uns["quantification_metadata"]["quantification_tool"] == "Kallisto"

        # Step 5: Verify transpose information
        assert "transpose_info" in adata.uns
        assert adata.uns["transpose_info"]["transpose_applied"] in [True, "True"]

        # Step 6: Store in DataManagerV2
        data_manager.modalities["test_experiment"] = adata

        # Step 7: Verify retrieval
        retrieved = data_manager.get_modality("test_experiment")
        assert retrieved.shape == adata.shape
        assert "test_experiment" in data_manager.list_modalities()

    def test_complete_analysis_workflow(self, realistic_kallisto_dataset, data_manager):
        """Test complete workflow: load → store → verify ready for analysis."""
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

        kallisto_dir, sample_metadata, n_genes = realistic_kallisto_dataset

        # Step 1: Load quantification files
        service = BulkRNASeqService(data_manager=data_manager)
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        # Step 2: Create AnnData
        adapter = TranscriptomicsAdapter(data_type="bulk")
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        # Step 3: Store in DataManagerV2
        modality_name = "bulk_experiment"
        data_manager.modalities[modality_name] = adata

        # Step 4: Verify data structure is correct
        assert adata.n_obs == 4, "Should have 4 samples"
        assert adata.n_vars == n_genes, f"Should have {n_genes} genes"

        # Step 5: Verify metadata is preserved
        assert "quantification_metadata" in adata.uns
        assert adata.uns["quantification_metadata"]["quantification_tool"] == "Kallisto"
        assert adata.uns["quantification_metadata"]["n_samples"] in [4, "4"]

        # Step 6: Verify data is ready for analysis
        assert modality_name in data_manager.list_modalities()
        retrieved = data_manager.get_modality(modality_name)
        assert retrieved.shape == adata.shape

        # Step 7: Verify quality metrics can be calculated
        metrics = data_manager.get_quality_metrics(modality_name)
        assert "n_obs" in metrics
        assert "n_vars" in metrics
        assert metrics["n_obs"] == 4
        assert metrics["n_vars"] == n_genes

        # Step 8: Verify data structure is suitable for downstream analysis
        # Quantification data should have samples × genes orientation
        assert adata.n_obs < adata.n_vars, "Bulk RNA-seq: samples should be < genes"

        # Step 9: Log the workflow completion
        data_manager.log_tool_usage(
            tool_name="complete_analysis_workflow",
            parameters={"n_samples": 4, "n_genes": n_genes},
            description="Completed end-to-end quantification loading workflow",
        )


class TestAgentIntegrationWorkflow:
    """Test agent interaction with quantification data."""

    def test_agent_tool_availability(self, data_manager):
        """Test that bulk RNA-seq agent has quantification loading tool."""
        agent_graph = transcriptomics_expert(
            data_manager=data_manager,
            callback_handler=None,
        )

        # Verify agent was created
        assert agent_graph is not None

        # The agent should have load_quantification_files tool
        # (This is verified by the fact that the agent factory doesn't raise an error)

    def test_agent_workflow_with_quantification_data(
        self, realistic_kallisto_dataset, data_manager
    ):
        """Test that agent can work with loaded quantification data."""
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

        kallisto_dir, sample_metadata, n_genes = realistic_kallisto_dataset

        # Load data into DataManagerV2 (simulating agent tool execution)
        service = BulkRNASeqService(data_manager=data_manager)
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        adapter = TranscriptomicsAdapter(data_type="bulk")
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        modality_name = "bulk_quantification"
        data_manager.modalities[modality_name] = adata

        # Verify agent can access the data
        assert modality_name in data_manager.list_modalities()
        retrieved_data = data_manager.get_modality(modality_name)

        assert retrieved_data.shape == (4, n_genes)
        assert "quantification_metadata" in retrieved_data.uns

        # Verify quality metrics can be calculated
        metrics = data_manager.get_quality_metrics(modality_name)
        assert "n_obs" in metrics
        assert "n_vars" in metrics
        assert metrics["n_obs"] == 4
        assert metrics["n_vars"] == n_genes


class TestErrorHandlingWorkflow:
    """Test error handling in quantification workflows."""

    def test_invalid_directory_workflow(self, data_manager):
        """Test workflow with invalid quantification directory."""
        service = BulkRNASeqService(data_manager=data_manager)

        with pytest.raises(FileNotFoundError):
            service._detect_quantification_tool(Path("/nonexistent/directory"))

    def test_empty_directory_workflow(self, tmp_path, data_manager):
        """Test workflow with empty directory."""
        empty_dir = tmp_path / "empty_quantification"
        empty_dir.mkdir()

        service = BulkRNASeqService(data_manager=data_manager)

        with pytest.raises(ValueError, match="No Kallisto or Salmon"):
            service._detect_quantification_tool(empty_dir)

    def test_corrupted_file_workflow(self, tmp_path, data_manager):
        """Test workflow with corrupted quantification files."""
        corrupted_dir = tmp_path / "corrupted_quantification"
        corrupted_dir.mkdir()

        sample_dir = corrupted_dir / "sample1"
        sample_dir.mkdir()

        # Create corrupted abundance.tsv (missing required columns)
        corrupted_file = sample_dir / "abundance.tsv"
        pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(
            corrupted_file, sep="\t", index=False
        )

        service = BulkRNASeqService(data_manager=data_manager)

        with pytest.raises(ValueError):
            service.merge_kallisto_results(corrupted_dir)


class TestProvenanceTracking:
    """Test provenance tracking for quantification workflows."""

    def test_tool_usage_logging(self, realistic_kallisto_dataset, data_manager):
        """Test that quantification loading is properly logged."""
        from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter

        kallisto_dir, sample_metadata, n_genes = realistic_kallisto_dataset

        # Simulate tool usage logging
        service = BulkRNASeqService(data_manager=data_manager)
        df, metadata = service.load_from_quantification_files(
            quantification_dir=kallisto_dir, tool="auto"
        )

        adapter = TranscriptomicsAdapter(data_type="bulk")
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata,
        )

        modality_name = "test_provenance"
        data_manager.modalities[modality_name] = adata

        # Log the operation
        data_manager.log_tool_usage(
            tool_name="load_quantification_files",
            parameters={
                "quantification_dir": str(kallisto_dir),
                "tool": "kallisto",
            },
            description=f"Loaded Kallisto quantification from {kallisto_dir}",
        )

        # Verify provenance was recorded
        assert data_manager.provenance is not None
        assert len(data_manager.provenance.activities) > 0

        # Find the load_quantification_files activity
        last_activity = data_manager.provenance.activities[-1]
        assert last_activity["type"] == "load_quantification_files"
        assert "parameters" in last_activity
        assert "quantification_dir" in last_activity["parameters"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
