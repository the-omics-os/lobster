"""
Regression tests for metadata_assistant with actual production data.

These tests use the full 247-sample PRJNA891765 dataset to ensure:
- Sample extraction works with real-world data structures
- Validation handles production data correctly
- Filtering preserves all required fields
- CSV export formats correctly
- Performance meets requirements (<2s for 247 samples)

Tests marked with @pytest.mark.regression and @pytest.mark.slow
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.publication_queue import (
    HandoffStatus,
    PublicationQueueEntry,
    PublicationStatus,
)


@pytest.mark.regression
@pytest.mark.slow
class TestProductionDataWorkflow:
    """Test complete workflow with production data."""

    def test_full_workflow_with_247_samples(
        self, tmp_path, production_sra_samples
    ):
        """
        Complete workflow test with actual production data (247 samples).

        This is a regression test to ensure the full pipeline works
        with real-world data structures and volumes.
        """
        # Setup workspace
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        (workspace / "metadata").mkdir()

        data_manager = DataManagerV2(workspace_path=str(workspace))

        # Create queue entry
        queue = data_manager.publication_queue
        entry = PublicationQueueEntry(
            entry_id="test_prod_001",
            pmid="37249910",
            doi="10.1093/infdis/jiad190",
            title="Microbiome Predict Infections in Pediatric HCT",
            status=PublicationStatus.HANDOFF_READY,
            workspace_metadata_keys=["sra_PRJNA891765_samples"],
            dataset_ids=["PRJNA891765"],
            extracted_identifiers={"bioproject": ["PRJNA891765"]},
        )
        queue.add_entry(entry)

        # Write production data to workspace
        metadata_dir = workspace / "metadata"
        sra_file = metadata_dir / "sra_PRJNA891765_samples.json"
        sra_file.write_text(json.dumps(production_sra_samples, indent=2))

        # Create metadata_assistant agent with mocked LLM
        with patch("lobster.agents.metadata_assistant.create_react_agent") as mock_agent, \
             patch("lobster.agents.metadata_assistant.create_llm") as mock_llm, \
             patch("lobster.agents.metadata_assistant.get_settings") as mock_settings, \
             patch("lobster.agents.metadata_assistant.MetadataStandardizationService"), \
             patch("lobster.agents.metadata_assistant.SampleMappingService"):

            # Setup mocks
            mock_settings_instance = Mock()
            mock_settings_instance.get_agent_llm_params.return_value = {}
            mock_settings.return_value = mock_settings_instance

            mock_llm_instance = Mock()
            mock_llm_instance.with_config = Mock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance

            mock_agent.return_value = Mock()

            from lobster.agents.metadata_assistant import metadata_assistant

            agent = metadata_assistant(data_manager=data_manager)

            # Extract process_metadata_entry tool
            tools = mock_agent.call_args[1]["tools"]
            process_entry_tool = next(
                t for t in tools if t.name == "process_metadata_entry"
            )

            # Process entry (no filter - test validation only)
            start_time = time.time()
            result = process_entry_tool.func(entry_id="test_prod_001")
            elapsed = time.time() - start_time

            # Verify performance (<2s for 247 samples)
            assert elapsed < 2.0, f"Processing took {elapsed:.2f}s (should be <2s)"

            # Verify result message
            assert "Entry Processed" in result
            assert "test_prod_001" in result
            assert "247" in result or "Samples Extracted: 247" in result

            # Verify queue updated
            updated_entry = queue.get_entry("test_prod_001")
            assert updated_entry.handoff_status == HandoffStatus.METADATA_COMPLETE
            assert updated_entry.harmonization_metadata is not None

            # Verify harmonization data structure
            harmonization = updated_entry.harmonization_metadata
            assert "samples" in harmonization
            assert "stats" in harmonization

            # Verify stats
            stats = harmonization["stats"]
            assert "samples_extracted" in stats
            assert "samples_valid" in stats
            assert stats["samples_extracted"] == 247
            assert stats["samples_valid"] >= 200  # Allow some validation failures

            # Verify sample structure preserved
            for sample in harmonization["samples"][:10]:  # Check first 10
                assert "run_accession" in sample
                assert "library_strategy" in sample
                assert "publication_entry_id" in sample
                assert sample["publication_entry_id"] == "test_prod_001"

    def test_production_data_validation_stats(self, production_sra_samples):
        """Test validation statistics on production data."""
        from lobster.core.schemas.sra import validate_sra_samples_batch

        samples = production_sra_samples["data"]["samples"]

        result = validate_sra_samples_batch(samples)

        # Log statistics
        print(f"\n=== Production Data Validation Stats ===")
        print(f"Total samples: {result.metadata['total_samples']}")
        print(f"Valid samples: {result.metadata['valid_samples']}")
        print(f"Validation rate: {result.metadata['validation_rate']:.1f}%")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")

        # Assertions
        assert result.metadata["total_samples"] == 247
        assert result.metadata["validation_rate"] > 90.0  # >90% should be valid

        # If there are errors, print first few for debugging
        if result.has_errors:
            print(f"\nFirst 5 errors:")
            for error in result.errors[:5]:
                print(f"  - {error}")

    def test_production_data_field_coverage(self, production_sra_samples):
        """Test field coverage in production data."""
        samples = production_sra_samples["data"]["samples"]

        # Count field coverage
        field_coverage = {}
        for sample in samples:
            for field in sample.keys():
                field_coverage[field] = field_coverage.get(field, 0) + 1

        # Verify critical fields have 100% coverage
        critical_fields = [
            "run_accession",
            "experiment_accession",
            "sample_accession",
            "bioproject",
            "library_strategy",
            "organism_name",
        ]

        print(f"\n=== Field Coverage (247 samples) ===")
        for field in critical_fields:
            coverage = field_coverage.get(field, 0)
            pct = (coverage / 247) * 100
            print(f"{field}: {coverage}/247 ({pct:.1f}%)")
            assert pct == 100.0, f"{field} should have 100% coverage"

        # Check download URL coverage
        url_fields = ["public_url", "ncbi_url", "aws_url", "gcp_url"]
        samples_with_urls = 0
        for sample in samples:
            has_url = any(sample.get(f) for f in url_fields)
            if has_url:
                samples_with_urls += 1

        url_coverage = (samples_with_urls / 247) * 100
        print(f"\nSamples with download URLs: {samples_with_urls}/247 ({url_coverage:.1f}%)")
        assert url_coverage > 95.0, "Most samples should have download URLs"

    @pytest.mark.slow
    def test_queue_processing_with_production_data(
        self, tmp_path, production_sra_samples
    ):
        """Test process_metadata_queue with production data."""
        # Setup workspace
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        (workspace / "metadata").mkdir()

        data_manager = DataManagerV2(workspace_path=str(workspace))

        # Create 3 queue entries with same production data
        queue = data_manager.publication_queue
        for i in range(3):
            entry = PublicationQueueEntry(
                entry_id=f"test_prod_{i:03d}",
                pmid=f"3724991{i}",
                doi=f"10.1093/test.{i}",
                title=f"Test Publication {i}",
                status=PublicationStatus.HANDOFF_READY,
                workspace_metadata_keys=["sra_PRJNA891765_samples"],
                dataset_ids=["PRJNA891765"],
            )
            queue.add_entry(entry)

        # Write production data to workspace
        metadata_dir = workspace / "metadata"
        sra_file = metadata_dir / "sra_PRJNA891765_samples.json"
        sra_file.write_text(json.dumps(production_sra_samples, indent=2))

        # Create agent
        with patch("lobster.agents.metadata_assistant.create_react_agent") as mock_agent, \
             patch("lobster.agents.metadata_assistant.create_llm") as mock_llm, \
             patch("lobster.agents.metadata_assistant.get_settings") as mock_settings, \
             patch("lobster.agents.metadata_assistant.MetadataStandardizationService"), \
             patch("lobster.agents.metadata_assistant.SampleMappingService"):

            # Setup mocks
            mock_settings_instance = Mock()
            mock_settings_instance.get_agent_llm_params.return_value = {}
            mock_settings.return_value = mock_settings_instance
            mock_llm_instance = Mock()
            mock_llm_instance.with_config = Mock(return_value=mock_llm_instance)
            mock_llm.return_value = mock_llm_instance
            mock_agent.return_value = Mock()

            from lobster.agents.metadata_assistant import metadata_assistant

            agent = metadata_assistant(data_manager=data_manager)

            # Extract process_metadata_queue tool
            tools = mock_agent.call_args[1]["tools"]
            process_queue_tool = next(
                t for t in tools if t.name == "process_metadata_queue"
            )

            # Process queue
            start_time = time.time()
            result = process_queue_tool.func(
                status_filter="handoff_ready", output_key="test_aggregated"
            )
            elapsed = time.time() - start_time

            # Verify performance (<6s for 3 entries x 247 samples)
            assert (
                elapsed < 6.0
            ), f"Queue processing took {elapsed:.2f}s (should be <6s)"

            # Verify result
            assert "Queue Processing Complete" in result
            assert "Entries Processed**: 3" in result  # Markdown formatted
            assert "741" in result  # 3 x 247 samples

            # Verify all entries updated
            for i in range(3):
                entry = queue.get_entry(f"test_prod_{i:03d}")
                assert entry.handoff_status == HandoffStatus.METADATA_COMPLETE

            # Verify aggregated output
            assert "test_aggregated" in data_manager.metadata_store
            aggregated = data_manager.metadata_store["test_aggregated"]
            assert "samples" in aggregated
            assert len(aggregated["samples"]) >= 600  # Most samples should be valid


@pytest.mark.regression
class TestProductionDataComparison:
    """Test production data against known-good baselines."""

    def test_production_sample_structure_unchanged(self, production_sra_samples):
        """Verify production data structure hasn't changed."""
        # Known structure from PRJNA891765
        assert "identifier" in production_sra_samples
        assert "data" in production_sra_samples
        assert "samples" in production_sra_samples["data"]
        assert "sample_count" in production_sra_samples["data"]

        # Verify sample count
        assert production_sra_samples["data"]["sample_count"] == 247
        assert len(production_sra_samples["data"]["samples"]) == 247

        # Verify first sample has expected structure
        first_sample = production_sra_samples["data"]["samples"][0]
        expected_fields = [
            "run_accession",
            "experiment_accession",
            "sample_accession",
            "study_accession",
            "bioproject",
            "biosample",
            "library_strategy",
            "library_source",
            "library_selection",
            "library_layout",
            "organism_name",
            "organism_taxid",
            "instrument",
            "instrument_model",
        ]

        for field in expected_fields:
            assert field in first_sample, f"Expected field '{field}' missing"

    def test_production_bioproject_consistency(self, production_sra_samples):
        """Verify all samples belong to PRJNA891765."""
        samples = production_sra_samples["data"]["samples"]

        # All samples should have bioproject = PRJNA891765
        for sample in samples:
            assert (
                sample.get("bioproject") == "PRJNA891765"
            ), f"Sample {sample.get('run_accession')} has wrong bioproject"

    def test_production_library_strategy_distribution(self, production_sra_samples):
        """Analyze library strategy distribution in production data."""
        samples = production_sra_samples["data"]["samples"]

        # Count library strategies
        strategies = {}
        for sample in samples:
            strategy = sample.get("library_strategy", "UNKNOWN")
            strategies[strategy] = strategies.get(strategy, 0) + 1

        print(f"\n=== Library Strategy Distribution ===")
        for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
            pct = (count / 247) * 100
            print(f"{strategy}: {count} ({pct:.1f}%)")

        # PRJNA891765 has mixed strategies (WGS: 57.9%, AMPLICON: 42.1%)
        assert "AMPLICON" in strategies, "Should have AMPLICON samples"
        assert "WGS" in strategies, "Should have WGS samples"
        assert strategies["WGS"] > 100  # ~143 WGS samples
        assert strategies["AMPLICON"] > 100  # ~104 AMPLICON samples


@pytest.mark.regression
@pytest.mark.slow
class TestProductionPerformance:
    """Performance benchmarks with production data."""

    def test_validation_performance(self, production_sra_samples):
        """Test validation completes in <100ms per sample."""
        from lobster.core.schemas.sra import validate_sra_samples_batch

        samples = production_sra_samples["data"]["samples"]

        # Benchmark batch validation
        start = time.time()
        result = validate_sra_samples_batch(samples)
        elapsed = time.time() - start

        # Should complete in <250ms for 247 samples (<1ms per sample)
        assert elapsed < 0.25, f"Validation took {elapsed:.3f}s (should be <0.25s)"

        avg_time_per_sample = (elapsed / 247) * 1000  # Convert to ms
        print(
            f"\nValidation performance: {elapsed:.3f}s total "
            f"({avg_time_per_sample:.2f}ms per sample)"
        )

        assert avg_time_per_sample < 2.0, "Should validate in <2ms per sample"

    def test_extraction_performance(self, production_sra_samples):
        """Test sample extraction performance."""
        # Test nested structure extraction
        ws_data = production_sra_samples

        # Time the extraction (manual implementation since function is in closure)
        start = time.time()

        # Simulate extraction logic
        samples_list = []
        if isinstance(ws_data, dict):
            if "data" in ws_data:
                data_dict = ws_data["data"]
                if "samples" in data_dict:
                    samples_list = data_dict["samples"]

        elapsed = time.time() - start

        # Extraction should be very fast (<10ms for 247 samples)
        assert elapsed < 0.01, f"Extraction took {elapsed:.3f}s (should be <0.01s)"
        assert len(samples_list) == 247


@pytest.mark.regression
class TestProductionDataQuality:
    """Test data quality metrics on production data."""

    def test_download_url_availability(self, production_sra_samples):
        """Verify download URLs are available for all samples."""
        from lobster.core.schemas.sra import SRASampleSchema

        samples = production_sra_samples["data"]["samples"]

        samples_with_urls = 0
        samples_without_urls = []

        for sample in samples:
            try:
                validated = SRASampleSchema.from_dict(sample)
                if validated.has_download_url():
                    samples_with_urls += 1
                else:
                    samples_without_urls.append(validated.run_accession)
            except Exception:
                # Skip validation failures for this test
                pass

        url_coverage = (samples_with_urls / 247) * 100
        print(f"\nDownload URL coverage: {samples_with_urls}/247 ({url_coverage:.1f}%)")

        if samples_without_urls:
            print(f"Samples without URLs: {samples_without_urls[:10]}")

        assert url_coverage > 95.0, "At least 95% of samples should have download URLs"

    def test_environmental_context_for_amplicon(self, production_sra_samples):
        """Verify AMPLICON samples have environmental context."""
        from lobster.core.schemas.sra import SRASampleSchema

        samples = production_sra_samples["data"]["samples"]

        amplicon_samples = []
        amplicon_with_env = 0

        for sample in samples:
            if sample.get("library_strategy") == "AMPLICON":
                amplicon_samples.append(sample)
                if sample.get("env_medium"):
                    amplicon_with_env += 1

        if amplicon_samples:
            env_coverage = (amplicon_with_env / len(amplicon_samples)) * 100
            print(
                f"\nAMPLICON samples with env_medium: "
                f"{amplicon_with_env}/{len(amplicon_samples)} ({env_coverage:.1f}%)"
            )

            # For microbiome studies, env_medium should be present
            assert (
                env_coverage > 80.0
            ), "Most AMPLICON samples should have env_medium for filtering"


@pytest.mark.regression
class TestBackwardCompatibility:
    """Ensure changes don't break existing workflows."""

    def test_harmonization_metadata_structure(self, tmp_path, minimal_sra_samples):
        """Verify harmonization_metadata structure is backward compatible."""
        # Setup
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()
        (workspace / "metadata").mkdir()

        data_manager = DataManagerV2(workspace_path=str(workspace))

        queue = data_manager.publication_queue
        entry = PublicationQueueEntry(
            entry_id="test_compat_001",
            pmid="12345678",
            status=PublicationStatus.HANDOFF_READY,
            workspace_metadata_keys=["sra_test_samples"],
        )
        queue.add_entry(entry)

        # Write minimal data
        sra_file = workspace / "metadata" / "sra_test_samples.json"
        sra_file.write_text(json.dumps(minimal_sra_samples, indent=2))

        # Process entry
        with patch("lobster.agents.metadata_assistant.create_react_agent") as mock_agent, \
             patch("lobster.agents.metadata_assistant.create_llm"), \
             patch("lobster.agents.metadata_assistant.get_settings") as mock_settings, \
             patch("lobster.agents.metadata_assistant.MetadataStandardizationService"), \
             patch("lobster.agents.metadata_assistant.SampleMappingService"):

            mock_settings_instance = Mock()
            mock_settings_instance.get_agent_llm_params.return_value = {}
            mock_settings.return_value = mock_settings_instance

            from lobster.agents.metadata_assistant import metadata_assistant

            agent = metadata_assistant(data_manager=data_manager)

            tools = mock_agent.call_args[1]["tools"]
            process_entry_tool = next(
                t for t in tools if t.name == "process_metadata_entry"
            )

            result = process_entry_tool.func(entry_id="test_compat_001")

            # Verify backward-compatible structure
            updated_entry = queue.get_entry("test_compat_001")
            harmonization = updated_entry.harmonization_metadata

            # Required fields for backward compatibility
            assert "samples" in harmonization
            assert "filter_criteria" in harmonization
            assert "stats" in harmonization

            # New validation stats should be present
            assert "samples_extracted" in harmonization["stats"]
            assert "samples_valid" in harmonization["stats"]
            assert "validation_errors" in harmonization["stats"]
            assert "validation_warnings" in harmonization["stats"]
