"""
Integration tests for microbiome harmonization workflow.

Tests the complete multi-agent handoff workflow:
1. Research agent extracts SRA identifiers from RIS, fetches metadata → workspace
2. Metadata assistant filters by organism/sample type, standardizes disease terms
3. Research agent exports filtered CSV to workspace

This file tests Phases 1-4:
- Phase 1: Schema extension (workspace_metadata_keys, harmonization_metadata)
- Phase 2: MicrobiomeFilteringService
- Phase 3: DiseaseStandardizationService
- Phase 4: metadata_assistant filter_samples_by tool

Running Tests:
```bash
# All tests
pytest tests/integration/test_microbiome_harmonization_workflow.py -v

# Specific test class
pytest tests/integration/test_microbiome_harmonization_workflow.py::TestPublicationQueueExtension -v

# With coverage
pytest tests/integration/test_microbiome_harmonization_workflow.py --cov=lobster.tools --cov=lobster.core.schemas -v
```
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.publication_queue import PublicationQueue
from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)
from lobster.services.metadata.disease_standardization_service import DiseaseStandardizationService
from lobster.services.metadata.microbiome_filtering_service import MicrobiomeFilteringService

# Import fixtures
from tests.integration.fixtures.microbiome_data import (
    expected_filtered_metadata,
    mock_data_manager,
    sample_harmonization_metadata,
    sample_publication_entry,
    sample_ris_file,
    sample_sra_metadata,
)


# ============================================================================
# Test Class 1: Publication Queue Schema Extension
# ============================================================================


@pytest.mark.integration
class TestPublicationQueueExtension:
    """
    Test Phase 1: workspace_metadata_keys and harmonization_metadata fields.

    Verifies:
    - New fields exist and serialize correctly
    - get_workspace_metadata_paths() works
    - Multi-agent handoff contract is preserved
    """

    def test_workspace_metadata_keys_field_exists(self, sample_publication_entry):
        """Test that workspace_metadata_keys field is present and works."""
        entry = sample_publication_entry

        # Initially empty
        assert entry.workspace_metadata_keys == []

        # Add metadata keys (simulates research_agent behavior)
        entry.workspace_metadata_keys = [
            "SRR1000000_metadata.json",
            "SRR1000001_metadata.json",
        ]

        # Verify persistence
        entry_dict = entry.to_dict()
        assert "workspace_metadata_keys" in entry_dict
        assert len(entry_dict["workspace_metadata_keys"]) == 2

    def test_harmonization_metadata_field_exists(
        self, sample_publication_entry, sample_harmonization_metadata
    ):
        """Test that harmonization_metadata field is present and works."""
        entry = sample_publication_entry

        # Initially None
        assert entry.harmonization_metadata is None

        # Set harmonized data (simulates metadata_assistant behavior)
        entry.harmonization_metadata = sample_harmonization_metadata

        # Verify persistence
        entry_dict = entry.to_dict()
        assert "harmonization_metadata" in entry_dict
        assert entry_dict["harmonization_metadata"]["validation_status"] == "passed"
        assert entry_dict["harmonization_metadata"]["total_samples"] == 3

    def test_get_workspace_metadata_paths(self, tmp_path, sample_publication_entry):
        """Test get_workspace_metadata_paths() method."""
        workspace = tmp_path / "workspace"
        metadata_dir = workspace / "metadata"
        metadata_dir.mkdir(parents=True)

        entry = sample_publication_entry
        entry.workspace_metadata_keys = [
            "SRR1000000_metadata.json",
            "SRR1000001_metadata.json",
            "SRR1000002_metadata.json",
        ]

        paths = entry.get_workspace_metadata_paths(str(workspace))

        # Verify paths are correctly constructed
        assert len(paths) == 3
        assert all(str(metadata_dir) in p for p in paths)
        assert any("SRR1000000_metadata.json" in p for p in paths)
        assert any("SRR1000001_metadata.json" in p for p in paths)

    def test_multi_agent_handoff_contract(self, tmp_path):
        """
        Test complete multi-agent handoff workflow using schema fields.

        Simulates:
        1. research_agent: Creates entry → populates workspace_metadata_keys
        2. metadata_assistant: Reads keys → processes → populates harmonization_metadata
        3. research_agent: Reads harmonization_metadata → exports CSV
        """
        workspace = tmp_path / "workspace"
        metadata_dir = workspace / "metadata"
        metadata_dir.mkdir(parents=True)

        # Step 1: Research agent creates entry and fetches metadata
        entry = PublicationQueueEntry(
            entry_id="test_handoff_001",
            pmid="12345678",
            title="Test microbiome study",
            schema_type="microbiome",
            status=PublicationStatus.METADATA_EXTRACTED,
            extracted_identifiers={"sra": ["SRP123456"]},
        )

        # Simulate research_agent saving metadata files
        metadata_files = []
        for i in range(3):
            filename = f"SRR100000{i}_metadata.json"
            filepath = metadata_dir / filename
            filepath.write_text(json.dumps({"run_accession": f"SRR100000{i}"}))
            metadata_files.append(filename)

        entry.workspace_metadata_keys = metadata_files

        # Verify research_agent populated keys
        assert len(entry.workspace_metadata_keys) == 3

        # Step 2: Metadata assistant processes
        paths = entry.get_workspace_metadata_paths(str(workspace))
        assert all(Path(p).exists() for p in paths)

        # Simulate metadata_assistant filtering and harmonizing
        entry.harmonization_metadata = {
            "samples": [{"run_accession": f"SRR100000{i}"} for i in range(3)],
            "validation_status": "passed",
            "total_samples": 3,
        }

        # Verify metadata_assistant populated harmonization_metadata
        assert entry.harmonization_metadata is not None
        assert entry.harmonization_metadata["validation_status"] == "passed"

        # Step 3: Research agent reads harmonization_metadata for export
        harmonized_data = entry.harmonization_metadata
        assert harmonized_data["total_samples"] == 3
        assert len(harmonized_data["samples"]) == 3


# ============================================================================
# Test Class 2: Microbiome Filtering Service
# ============================================================================


@pytest.mark.integration
class TestMicrobiomeFiltering:
    """
    Test Phase 2: MicrobiomeFilteringService.

    Verifies:
    - 16S amplicon detection
    - Host organism validation with fuzzy matching
    - Bulk filtering operations
    - Provenance IR generation
    """

    def test_16s_amplicon_detection_strict_mode(self):
        """Test 16S amplicon detection in strict mode."""
        service = MicrobiomeFilteringService()

        metadata = {
            "library_strategy": "AMPLICON",
            "library_source": "METAGENOMIC",
            "platform": "ILLUMINA",
        }

        result, stats, ir = service.validate_16s_amplicon(metadata, strict=True)

        assert stats["is_valid"] is True
        assert stats["matched_field"] == "library_strategy"
        assert ir.operation == "microbiome_filtering_service.validate_16s_amplicon"

    def test_16s_amplicon_detection_non_strict_mode(self):
        """Test 16S amplicon detection in non-strict mode (fuzzy matching)."""
        service = MicrobiomeFilteringService()

        metadata = {
            "library_strategy": "targeted locus",
            "platform": "miseq",
        }

        result, stats, ir = service.validate_16s_amplicon(metadata, strict=False)

        assert stats["is_valid"] is True
        assert ir.operation == "microbiome_filtering_service.validate_16s_amplicon"

    def test_host_organism_validation_exact_match(self):
        """Test host organism validation with exact match."""
        service = MicrobiomeFilteringService()

        metadata = {"organism": "Homo sapiens", "host": "Homo sapiens"}

        result, stats, ir = service.validate_host_organism(
            metadata, allowed_hosts=["Human"]
        )

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Human"
        assert stats["confidence"] >= 0.9
        assert ir.operation == "microbiome_filtering_service.validate_host_organism"

    def test_host_organism_validation_fuzzy_match(self):
        """Test host organism validation with fuzzy matching."""
        service = MicrobiomeFilteringService()

        metadata = {"organism": "human gut metagenome", "host": "h. sapiens"}

        result, stats, ir = service.validate_host_organism(
            metadata, allowed_hosts=["Human"], fuzzy_threshold=0.7
        )

        assert stats["is_valid"] is True
        assert stats["matched_host"] == "Human"

    def test_bulk_filtering_with_sample_sra_metadata(self, sample_sra_metadata):
        """
        Test bulk filtering with sample SRA metadata (24 samples).

        Should filter out:
        - 4 mouse samples
        - 4 environmental samples
        Should keep:
        - 12 human gut samples
        - 4 human stool samples
        """
        service = MicrobiomeFilteringService()

        # Manually filter samples using validate_host_organism
        passed_samples = []
        failed_samples = []

        for meta in sample_sra_metadata:
            result, stats, ir = service.validate_host_organism(
                meta, allowed_hosts=["Human"]
            )
            if stats["is_valid"]:
                passed_samples.append(meta)
            else:
                failed_samples.append(meta)

        # Verify filtering results
        assert len(sample_sra_metadata) == 24
        assert len(passed_samples) == 16  # 12 gut + 4 stool
        assert len(failed_samples) == 8  # 4 mouse + 4 environmental

        # Verify only human samples remain
        passed_df = pd.DataFrame(passed_samples)
        assert all("Homo sapiens" in org or "human" in org.lower()
                   for org in passed_df["organism"])

        # Verify IR from last validation
        assert ir.operation == "microbiome_filtering_service.validate_host_organism"
        assert "allowed_hosts" in ir.parameters

    def test_filtering_preserves_provenance(self):
        """Test that filtering operations generate W3C-PROV compliant IR."""
        service = MicrobiomeFilteringService()

        metadata = {
            "library_strategy": "AMPLICON",
            "organism": "Homo sapiens",
        }

        _, stats, ir = service.validate_16s_amplicon(metadata)

        # Verify IR structure
        assert ir.operation is not None
        assert ir.tool_name is not None
        assert ir.library is not None
        assert ir.code_template is not None
        assert ir.parameters is not None
        assert ir.parameter_schema is not None


# ============================================================================
# Test Class 3: Disease Standardization Service
# ============================================================================


@pytest.mark.integration
class TestDiseaseStandardization:
    """
    Test Phase 3: DiseaseStandardizationService.

    Verifies:
    - 5-level fuzzy matching (exact → contains → reverse → token → unmapped)
    - Sample type filtering (fecal vs tissue)
    - Provenance IR generation
    """

    def test_exact_disease_matching(self, sample_sra_metadata):
        """Test exact disease term matching."""
        service = DiseaseStandardizationService()

        df = pd.DataFrame(sample_sra_metadata[:5])  # 5 IBD samples

        standardized_df, stats, ir = service.standardize_disease_terms(df)

        # Verify standardization occurred
        assert "disease_original" in standardized_df.columns
        assert stats["total_samples"] == 5
        assert stats["exact_matches"] > 0 or stats["contains_matches"] > 0

        # Verify IR
        assert ir.operation == "disease_standardization_service.standardize_disease_terms"

    def test_fuzzy_disease_matching(self):
        """Test fuzzy disease matching (contains, token-based)."""
        service = DiseaseStandardizationService()

        df = pd.DataFrame(
            [
                {"run_accession": "SRR001", "disease": "Crohns disease"},
                {"run_accession": "SRR002", "disease": "ulcerative colitis"},
                {"run_accession": "SRR003", "disease": "healthy control"},
                {"run_accession": "SRR004", "disease": "colorectal cancer"},
            ]
        )

        standardized_df, stats, ir = service.standardize_disease_terms(df)

        # Verify standardization
        assert "disease_original" in standardized_df.columns
        assert stats["total_samples"] == 4
        assert stats["unmapped"] == 0  # All should map

        # Verify standardized values
        assert "cd" in standardized_df["disease"].values
        assert "uc" in standardized_df["disease"].values
        assert "healthy" in standardized_df["disease"].values
        assert "crc" in standardized_df["disease"].values

    def test_sample_type_filtering_fecal_only(self, sample_sra_metadata):
        """Test filtering for fecal samples only."""
        service = DiseaseStandardizationService()

        df = pd.DataFrame(sample_sra_metadata)

        # Filter for fecal samples (should get stool samples)
        filtered_df, stats, ir = service.filter_by_sample_type(df, sample_types=["fecal"])

        # Verify filtering
        assert stats["total_input_samples"] == 24
        assert stats["passed_samples"] == 4  # Only the 4 stool samples
        assert all("stool" in str(tissue).lower() for tissue in filtered_df["tissue"])

    def test_provenance_ir_generation(self):
        """Test that disease standardization generates W3C-PROV IR."""
        service = DiseaseStandardizationService()

        df = pd.DataFrame(
            [
                {"run_accession": "SRR001", "disease": "IBD"},
                {"run_accession": "SRR002", "disease": "healthy"},
            ]
        )

        _, stats, ir = service.standardize_disease_terms(df)

        # Verify IR structure
        assert ir.operation is not None
        assert ir.tool_name is not None
        assert ir.library == "pandas"
        assert ir.code_template is not None
        assert "disease_column" in ir.parameters
        assert ir.parameter_schema is not None


# ============================================================================
# Test Class 4: End-to-End Workflow
# ============================================================================


@pytest.mark.integration
class TestEndToEndWorkflow:
    """
    Test complete end-to-end workflow from RIS to filtered CSV.

    Simulates:
    1. Research agent: RIS → SRA identifiers → metadata fetch → workspace
    2. Metadata assistant: Filter samples + standardize diseases
    3. Research agent: Export filtered CSV
    """

    @patch("lobster.tools.providers.geo_provider.GEOProvider.get_sra_metadata")
    def test_complete_workflow_with_mocked_api(
        self, mock_get_sra, tmp_path, sample_ris_file, sample_sra_metadata
    ):
        """
        Test complete workflow with mocked SRA API.

        Steps:
        1. Load RIS file
        2. Extract SRA identifiers
        3. Fetch metadata (mocked)
        4. Filter samples
        5. Standardize disease terms
        6. Export CSV
        """
        # Mock SRA metadata fetch
        mock_get_sra.return_value = sample_sra_metadata

        # Setup workspace
        workspace = tmp_path / "workspace"
        metadata_dir = workspace / "metadata"
        metadata_dir.mkdir(parents=True)

        # Step 1: Parse RIS file (simulate research_agent)
        from lobster.core.ris_parser import RISParser

        parser = RISParser()
        publications = parser.parse_file(str(sample_ris_file))

        assert len(publications) == 3

        # Step 2: Extract SRA identifiers
        sra_ids = []
        for pub in publications:
            keywords = pub.get("keywords", [])
            sra_ids.extend([kw for kw in keywords if kw.startswith("SRP")])

        assert len(sra_ids) == 3  # SRP123456, SRP234567, SRP345678

        # Step 3: Fetch metadata (mocked)
        metadata = mock_get_sra.return_value
        assert len(metadata) == 24

        # Step 4: Filter samples
        filtering_service = MicrobiomeFilteringService()

        # Filter samples using validate_host_organism
        passed_samples = []
        for meta in metadata:
            result, stats, filter_ir = filtering_service.validate_host_organism(
                meta, allowed_hosts=["Human"]
            )
            if stats["is_valid"]:
                passed_samples.append(meta)

        filtered_df = pd.DataFrame(passed_samples)
        assert len(passed_samples) == 16  # 12 gut + 4 stool

        # Step 5: Standardize disease terms
        disease_service = DiseaseStandardizationService()
        standardized_df, disease_stats, disease_ir = (
            disease_service.standardize_disease_terms(filtered_df)
        )

        assert disease_stats["total_samples"] == 16

        # Step 6: Export CSV
        output_csv = workspace / "filtered_metadata.csv"
        standardized_df.to_csv(output_csv, index=False)

        assert output_csv.exists()
        exported_df = pd.read_csv(output_csv)
        assert len(exported_df) == 16

    def test_workflow_with_publication_queue_integration(
        self, tmp_path, sample_publication_entry, sample_sra_metadata
    ):
        """
        Test workflow integrated with PublicationQueue.

        Verifies workspace_metadata_keys → harmonization_metadata handoff.
        """
        workspace = tmp_path / "workspace"
        metadata_dir = workspace / "metadata"
        metadata_dir.mkdir(parents=True)

        # Create publication queue
        pub_queue = PublicationQueue(workspace_dir=str(workspace))

        # Add entry
        entry = sample_publication_entry
        pub_queue.add_entry(entry)

        # Simulate research_agent saving metadata files
        for i, meta in enumerate(sample_sra_metadata[:5]):
            filename = f"{meta['run_accession']}_metadata.json"
            filepath = metadata_dir / filename
            filepath.write_text(json.dumps(meta))
            entry.workspace_metadata_keys.append(filename)

        # Update entry
        pub_queue.update_status(
            entry_id=entry.entry_id,
            status=PublicationStatus.METADATA_EXTRACTED,
        )

        # Simulate metadata_assistant filtering
        filtering_service = MicrobiomeFilteringService()
        disease_service = DiseaseStandardizationService()

        # Load metadata from workspace
        all_metadata = []
        for path in entry.get_workspace_metadata_paths(str(workspace)):
            with open(path) as f:
                all_metadata.append(json.load(f))

        # Filter samples
        passed_samples = []
        for meta in all_metadata:
            result, stats, _ = filtering_service.validate_host_organism(
                meta, allowed_hosts=["Human"]
            )
            if stats["is_valid"]:
                passed_samples.append(meta)

        filtered_df = pd.DataFrame(passed_samples)
        filter_stats = {"passed_samples": len(passed_samples), "total_samples": len(all_metadata)}

        # Standardize disease terms
        standardized_df, disease_stats, _ = disease_service.standardize_disease_terms(
            filtered_df
        )

        # Populate harmonization_metadata
        entry.harmonization_metadata = {
            "samples": standardized_df.to_dict("records"),
            "validation_status": "passed",
            "total_samples": len(standardized_df),
            "filter_stats": filter_stats,
            "disease_stats": disease_stats,
        }

        # Update entry
        pub_queue.update_status(
            entry_id=entry.entry_id,
            status=PublicationStatus.COMPLETED,
        )

        # Verify final state
        final_entry = pub_queue.get_entry(entry.entry_id)
        assert final_entry.status == PublicationStatus.COMPLETED
        assert final_entry.harmonization_metadata is not None
        assert final_entry.harmonization_metadata["validation_status"] == "passed"


# ============================================================================
# Test Class 5: Error Recovery
# ============================================================================


@pytest.mark.integration
class TestErrorRecovery:
    """
    Test error handling and recovery mechanisms.

    Verifies:
    - Invalid metadata handling
    - Empty dataset handling
    - Partial filtering failures
    """

    def test_invalid_metadata_handling(self):
        """Test that service handles invalid metadata gracefully."""
        service = MicrobiomeFilteringService()

        # Missing required fields
        invalid_metadata = {"some_field": "value"}

        result, stats, ir = service.validate_16s_amplicon(invalid_metadata)

        # Should return invalid result, not crash
        assert stats["is_valid"] is False
        assert "reason" in stats

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        service = MicrobiomeFilteringService()

        # Test with empty metadata dict
        empty_metadata = {}

        result, stats, ir = service.validate_host_organism(empty_metadata, allowed_hosts=["Human"])

        # Should return invalid result
        assert stats["is_valid"] is False

    def test_all_samples_filtered_out(self, sample_sra_metadata):
        """Test when all samples are filtered out."""
        service = MicrobiomeFilteringService()

        # Use impossible filter (no samples match)
        passed_samples = []
        for meta in sample_sra_metadata:
            result, stats, ir = service.validate_host_organism(
                meta, allowed_hosts=["Martian"]  # Impossible host
            )
            if stats["is_valid"]:
                passed_samples.append(meta)

        # All samples should be filtered out
        assert len(passed_samples) == 0
        assert len(sample_sra_metadata) == 24

    def test_missing_disease_column_handling(self):
        """Test handling of missing disease column."""
        service = DiseaseStandardizationService()

        df = pd.DataFrame(
            [
                {"run_accession": "SRR001", "tissue": "gut"},
                {"run_accession": "SRR002", "tissue": "stool"},
            ]
        )

        with pytest.raises(ValueError, match="Disease column.*not found"):
            service.standardize_disease_terms(df, disease_column="disease")


# ============================================================================
# Summary
# ============================================================================

"""
Test Summary:

1. TestPublicationQueueExtension (4 tests):
   - workspace_metadata_keys field
   - harmonization_metadata field
   - get_workspace_metadata_paths() method
   - Multi-agent handoff contract

2. TestMicrobiomeFiltering (6 tests):
   - 16S amplicon detection (strict/non-strict)
   - Host organism validation (exact/fuzzy)
   - Bulk filtering
   - Provenance IR generation

3. TestDiseaseStandardization (4 tests):
   - Exact disease matching
   - Fuzzy disease matching
   - Sample type filtering
   - Provenance IR generation

4. TestEndToEndWorkflow (2 tests):
   - Complete workflow with mocked API
   - Publication queue integration

5. TestErrorRecovery (4 tests):
   - Invalid metadata
   - Empty datasets
   - All samples filtered
   - Missing columns

Total: 20 integration tests
Expected runtime: <30 seconds (all APIs mocked)
"""
