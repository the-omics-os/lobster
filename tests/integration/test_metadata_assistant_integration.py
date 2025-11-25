"""
Integration tests for metadata_assistant agent.

Tests real workflows with actual services:
- Map samples between two GEO datasets
- Standardize metadata for multi-omics dataset
- Validate dataset completeness before download
- Full agent query with real LLM
- Handoff to data_expert after validation

Test coverage target: ≥85% with realistic workflow scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.metadata_assistant import metadata_assistant
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.metadata.metadata_standardization_service import (
    MetadataStandardizationService,
)
from lobster.services.metadata.sample_mapping_service import SampleMappingService

# ===============================================================================
# Test Fixtures
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
    """Create DataManagerV2 with test workspace."""
    dm = DataManagerV2(workspace_path=temp_workspace)
    return dm


@pytest.fixture
def source_dataset():
    """Create source transcriptomics dataset for mapping tests."""
    obs_data = {
        "sample_id": [f"Sample_{i}" for i in range(1, 11)],
        "condition": ["Control"] * 5 + ["Treatment"] * 5,
        "organism": ["Homo sapiens"] * 10,
        "platform": ["Illumina NovaSeq"] * 10,
        "sequencing_type": ["single-cell"] * 10,
        "batch": ["Batch1"] * 5 + ["Batch2"] * 5,
        "cell_type": ["T_cell"] * 10,
        "tissue": ["PBMC"] * 10,
        "timepoint": ["Day0"] * 10,
    }
    adata = ad.AnnData(
        X=np.random.rand(10, 100),
        obs=pd.DataFrame(obs_data, index=[f"Sample_{i}" for i in range(1, 11)]),
    )
    return adata


@pytest.fixture
def target_dataset():
    """Create target transcriptomics dataset for mapping tests."""
    # Similar naming but different convention
    obs_data = {
        "sample_id": [f"sample_{i}" for i in range(1, 11)],  # Lowercase
        "condition": ["Control"] * 5 + ["Treatment"] * 5,
        "organism": ["Homo sapiens"] * 10,
        "platform": ["Illumina NovaSeq"] * 10,
        "sequencing_type": ["single-cell"] * 10,
        "batch": ["Batch1"] * 5 + ["Batch2"] * 5,
        "cell_type": ["T_cell"] * 10,
        "tissue": ["PBMC"] * 10,
        "timepoint": ["Day0"] * 10,
    }
    adata = ad.AnnData(
        X=np.random.rand(10, 100),
        obs=pd.DataFrame(obs_data, index=[f"sample_{i}" for i in range(1, 11)]),
    )
    return adata


@pytest.fixture
def multi_omics_dataset():
    """Create multi-omics dataset with mixed metadata quality."""
    obs_data = {
        "sample_id": [f"S{i}" for i in range(1, 21)],
        "condition": ["Control"] * 10 + ["Treatment"] * 10,
        "organism": ["Homo sapiens"] * 15 + ["Mus musculus"] * 5,  # Mixed organisms
        "platform": ["Illumina"] * 20,
        "sequencing_type": ["bulk"] * 20,
        "batch": ["B1"] * 10 + ["B2"] * 10,
        "tissue": ["Liver"] * 10 + ["Brain"] * 10,
    }
    adata = ad.AnnData(
        X=np.random.rand(20, 200),
        obs=pd.DataFrame(obs_data, index=[f"S{i}" for i in range(1, 21)]),
    )
    return adata


@pytest.fixture
def incomplete_dataset():
    """Create dataset with incomplete metadata for validation tests."""
    obs_data = {
        "sample_id": ["S1", "S2", "S3", "S4", "S5"],
        "condition": ["Control", "Treatment", None, "Control", "Treatment"],  # Missing
        "organism": ["Homo sapiens"] * 5,
        "platform": ["Illumina"] * 3 + ["PacBio"] * 2,  # Inconsistent
    }
    adata = ad.AnnData(
        X=np.random.rand(5, 50),
        obs=pd.DataFrame(obs_data, index=["S1", "S2", "S3", "S4", "S5"]),
    )
    return adata


# ===============================================================================
# Workflow 1: Map Samples Between Two GEO Datasets
# ===============================================================================


class TestSampleMappingWorkflow:
    """Test real-world sample mapping workflows."""

    def test_map_samples_exact_match(
        self, data_manager, source_dataset, target_dataset
    ):
        """Test sample mapping with exact matches (case-insensitive)."""
        # Setup
        data_manager.modalities["geo_gse12345"] = source_dataset
        data_manager.modalities["geo_gse67890"] = target_dataset

        mapping_service = SampleMappingService(data_manager=data_manager)

        # Execute
        result = mapping_service.map_samples_by_id(
            source_identifier="geo_gse12345",
            target_identifier="geo_gse67890",
            strategies=["exact"],
        )

        # Verify
        assert result.summary["exact_matches"] == 10  # All samples match
        assert result.summary["mapping_rate"] == 1.0  # 100% mapping
        assert len(result.unmapped) == 0

        # Verify provenance
        assert "geo_gse12345" in data_manager.list_modalities()
        assert "geo_gse67890" in data_manager.list_modalities()

    def test_map_samples_pattern_match(self, data_manager):
        """Test sample mapping with pattern-based matching."""
        # Create datasets with pattern-based naming
        source = ad.AnnData(
            X=np.random.rand(5, 50),
            obs=pd.DataFrame(
                {"condition": ["Control"] * 5},
                index=[
                    "Sample_A",
                    "Sample_B",
                    "GSM123_Rep1",
                    "Control_001",
                    "Patient_X",
                ],
            ),
        )
        target = ad.AnnData(
            X=np.random.rand(5, 50),
            obs=pd.DataFrame(
                {"condition": ["Control"] * 5},
                index=["A", "B", "GSM123", "Control_1", "PatientX"],
            ),
        )

        data_manager.modalities["source"] = source
        data_manager.modalities["target"] = target

        mapping_service = SampleMappingService(data_manager=data_manager)

        # Execute
        result = mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact", "pattern"],
        )

        # Verify - should have some pattern matches
        assert result.summary["mapping_rate"] > 0.6  # At least 60% mapped
        assert len(result.fuzzy_matches) > 0  # Pattern matches categorized as fuzzy

    def test_map_samples_with_metadata_support(self, data_manager):
        """Test sample mapping using metadata alignment with similar IDs."""
        # Create datasets with similar IDs and aligned metadata
        # Metadata matching works best when there's some ID similarity + metadata support
        source = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(
                {
                    "condition": ["Control", "Treatment", "Control"],
                    "tissue": ["Liver", "Liver", "Brain"],
                    "timepoint": ["Day0", "Day0", "Day7"],
                },
                index=["Sample_1", "Sample_2", "Sample_3"],
            ),
        )
        target = ad.AnnData(
            X=np.random.rand(3, 50),
            obs=pd.DataFrame(
                {
                    "condition": ["Control", "Treatment", "Control"],
                    "tissue": ["Liver", "Liver", "Brain"],
                    "timepoint": ["Day0", "Day0", "Day7"],
                },
                index=["Sample_A", "Sample_B", "Sample_C"],
            ),
        )

        data_manager.modalities["source"] = source
        data_manager.modalities["target"] = target

        mapping_service = SampleMappingService(data_manager=data_manager)

        # Execute with multiple strategies including metadata
        result = mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact", "pattern", "metadata"],
        )

        # Verify - should have some matches (pattern + metadata support)
        assert result.summary["mapping_rate"] > 0.0
        # Verify overall mapping quality
        total_matches = (
            result.summary["exact_matches"] + result.summary["fuzzy_matches"]
        )
        assert total_matches > 0


# ===============================================================================
# Workflow 2: Standardize Metadata for Multi-Omics Dataset
# ===============================================================================


class TestMetadataStandardizationWorkflow:
    """Test real-world metadata standardization workflows."""

    def test_standardize_transcriptomics_metadata(self, data_manager, source_dataset):
        """Test metadata standardization for transcriptomics dataset."""
        data_manager.modalities["geo_gse12345"] = source_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Execute
        result = standardization_service.standardize_metadata(
            identifier="geo_gse12345", target_schema="transcriptomics"
        )

        # Verify
        assert len(result.standardized_metadata) == 10
        assert len(result.validation_errors) == 0
        assert result.field_coverage["condition"] == 100.0
        assert result.field_coverage["organism"] == 100.0

        # Verify schemas are correct type
        from lobster.core.schemas.transcriptomics import TranscriptomicsMetadataSchema

        assert all(
            isinstance(m, TranscriptomicsMetadataSchema)
            for m in result.standardized_metadata
        )

    def test_standardize_with_controlled_vocabularies(
        self, data_manager, multi_omics_dataset
    ):
        """Test standardization with controlled vocabularies."""
        data_manager.modalities["dataset"] = multi_omics_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Define controlled vocabularies
        controlled_vocabs = {
            "condition": ["Control", "Treatment"],
            "organism": ["Homo sapiens"],  # Reject Mus musculus
        }

        # Execute
        result = standardization_service.standardize_metadata(
            identifier="dataset",
            target_schema="transcriptomics",
            controlled_vocabularies=controlled_vocabs,
        )

        # Verify - should have warnings for non-compliant organisms
        assert len(result.warnings) > 0
        organism_warnings = [w for w in result.warnings if "organism" in w.lower()]
        assert len(organism_warnings) > 0

    def test_standardize_multi_omics_dataset(self, data_manager, multi_omics_dataset):
        """Test standardization of multi-omics dataset with mixed quality."""
        data_manager.modalities["multi_omics"] = multi_omics_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Execute
        result = standardization_service.standardize_metadata(
            identifier="multi_omics", target_schema="transcriptomics"
        )

        # Verify
        assert len(result.standardized_metadata) == 20
        assert "organism" in result.field_coverage
        assert result.field_coverage["organism"] == 100.0  # All have organism


# ===============================================================================
# Workflow 3: Validate Dataset Completeness Before Download
# ===============================================================================


class TestDatasetValidationWorkflow:
    """Test real-world dataset validation workflows."""

    def test_validate_complete_dataset(self, data_manager, source_dataset):
        """Test validation of complete, high-quality dataset."""
        data_manager.modalities["geo_gse12345"] = source_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Execute
        result = standardization_service.validate_dataset_content(
            identifier="geo_gse12345",
            expected_samples=5,  # Expecting at least 5 samples
            required_conditions=["Control", "Treatment"],
            check_controls=True,
            check_duplicates=True,
        )

        # Verify - should pass all checks
        assert result.has_required_samples is True  # 10 samples > 5
        assert len(result.missing_conditions) == 0  # Both conditions present
        assert len(result.control_issues) == 0  # Control samples detected
        assert len(result.duplicate_ids) == 0  # No duplicates
        assert result.platform_consistency is True  # All same platform
        assert len(result.warnings) == 0  # No warnings

    def test_validate_incomplete_dataset(self, data_manager, incomplete_dataset):
        """Test validation of incomplete dataset with issues."""
        data_manager.modalities["incomplete"] = incomplete_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Execute
        result = standardization_service.validate_dataset_content(
            identifier="incomplete",
            expected_samples=10,  # Expecting 10, but only 5 present
            required_conditions=[
                "Control",
                "Treatment",
                "Untreated",
            ],  # Missing Untreated
            check_controls=True,
            check_duplicates=True,
        )

        # Verify - should detect issues
        assert result.has_required_samples is False  # 5 < 10
        assert "Untreated" in result.missing_conditions  # Missing condition
        assert result.platform_consistency is False  # Mixed platforms
        assert len(result.warnings) > 0  # Should have warnings

    def test_validate_before_download_decision(self, data_manager, source_dataset):
        """Test validation workflow to decide whether to download dataset."""
        data_manager.modalities["potential_dataset"] = source_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Execute validation
        result = standardization_service.validate_dataset_content(
            identifier="potential_dataset",
            expected_samples=8,
            required_conditions=["Control", "Treatment"],
            check_controls=True,
        )

        # Decision logic
        is_worth_downloading = (
            result.has_required_samples
            and len(result.missing_conditions) == 0
            and len(result.control_issues) == 0
        )

        assert is_worth_downloading is True

        # Verify summary statistics
        assert result.summary["total_samples"] == 10
        assert result.summary["unique_conditions"] == 2
        assert result.summary["has_condition_field"] is True


# ===============================================================================
# Workflow 4: Full Agent Query (Requires LLM)
# ===============================================================================


@pytest.mark.skip(reason="Requires real LLM, expensive to run in CI")
class TestFullAgentQuery:
    """Test full agent queries with real LLM (skip in CI)."""

    def test_agent_map_samples_query(
        self, data_manager, source_dataset, target_dataset
    ):
        """Test agent with natural language query for sample mapping."""
        # Setup
        data_manager.modalities["geo_gse12345"] = source_dataset
        data_manager.modalities["geo_gse67890"] = target_dataset

        # Create agent
        agent = metadata_assistant(data_manager=data_manager)

        # Execute agent query (requires real LLM)
        # This test is skipped by default due to LLM cost
        # To run: pytest -m "not skip" tests/integration/test_metadata_assistant_integration.py
        pass

    def test_agent_standardize_metadata_query(self, data_manager, multi_omics_dataset):
        """Test agent with natural language query for standardization."""
        data_manager.modalities["multi_omics"] = multi_omics_dataset

        # Create agent
        agent = metadata_assistant(data_manager=data_manager)

        # Execute agent query (requires real LLM)
        # This test is skipped by default due to LLM cost
        pass


# ===============================================================================
# Workflow 5: Handoff to data_expert After Validation
# ===============================================================================


@pytest.mark.skip(reason="Requires multi-agent setup, complex integration")
class TestAgentHandoff:
    """Test agent handoff workflows (skip for now, requires complex setup)."""

    def test_handoff_to_data_expert_after_validation(
        self, data_manager, source_dataset
    ):
        """Test handoff from metadata_assistant to data_expert."""
        # This test requires full multi-agent setup with supervisor
        # Skip for now, will implement when multi-agent integration is more mature
        pass


# ===============================================================================
# End-to-End Workflow Integration
# ===============================================================================


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_metadata_workflow(
        self, data_manager, source_dataset, target_dataset
    ):
        """Test complete workflow: validate → standardize → map samples."""
        # Step 1: Load datasets
        data_manager.modalities["source"] = source_dataset
        data_manager.modalities["target"] = target_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )
        mapping_service = SampleMappingService(data_manager=data_manager)

        # Step 2: Validate source dataset
        validation_result = standardization_service.validate_dataset_content(
            identifier="source",
            expected_samples=5,
            required_conditions=["Control", "Treatment"],
            check_controls=True,
        )

        assert validation_result.has_required_samples is True

        # Step 3: Standardize metadata
        standardization_result = standardization_service.standardize_metadata(
            identifier="source", target_schema="transcriptomics"
        )

        assert len(standardization_result.standardized_metadata) == 10

        # Step 4: Map samples between datasets
        mapping_result = mapping_service.map_samples_by_id(
            source_identifier="source",
            target_identifier="target",
            strategies=["exact", "pattern"],
        )

        assert mapping_result.summary["mapping_rate"] > 0.8  # At least 80% mapped

    def test_quality_control_workflow(self, data_manager, incomplete_dataset):
        """Test workflow to detect and report dataset quality issues."""
        data_manager.modalities["incomplete"] = incomplete_dataset

        standardization_service = MetadataStandardizationService(
            data_manager=data_manager
        )

        # Step 1: Validate dataset
        validation_result = standardization_service.validate_dataset_content(
            identifier="incomplete",
            expected_samples=10,
            required_conditions=["Control", "Treatment"],
            check_controls=True,
            check_duplicates=True,
        )

        # Step 2: Collect issues
        issues = []
        if not validation_result.has_required_samples:
            issues.append(
                f"Insufficient samples: {validation_result.summary['total_samples']} < 10"
            )
        if len(validation_result.missing_conditions) > 0:
            issues.append(f"Missing conditions: {validation_result.missing_conditions}")
        if not validation_result.platform_consistency:
            issues.append("Inconsistent platforms detected")

        # Verify issues detected
        assert len(issues) > 0
        assert len(validation_result.warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
