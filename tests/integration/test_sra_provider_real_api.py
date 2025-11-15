"""
Integration tests for SRAProvider with real API calls.

This test suite validates SRAProvider functionality using real pysradb API calls to:
- NCBI Sequence Read Archive (SRA)
- SRA-to-PubMed linking
- SRA metadata retrieval and parsing
- Microbiome/metagenomics dataset discovery

**Performance Expectations:**
- Metadata retrieval: <5s per accession
- PMID-to-SRA linking: <5s
- Microbiome search: <10s

**Test Strategy:**
- Use known stable SRA accessions (SRP, SRX, SRR)
- Test both happy paths and error cases
- Validate filter application (organism, strategy, platform)
- Test microbiome-specific functionality
- Verify graceful degradation for unsupported features

**Markers:**
- @pytest.mark.real_api: All tests (requires pysradb + network)
- @pytest.mark.slow: Tests >30s (performance benchmarks, batch operations)
- @pytest.mark.integration: Multi-component tests

**Environment Requirements:**
- pysradb package installed
- Internet connectivity for NCBI SRA access
- Optional: NCBI_API_KEY for PMID linking rate limits

Phase 3 - Task 9: SRA Provider Real API Integration Tests
"""

import os
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import pytest

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.content_access_service import ContentAccessService
from lobster.tools.providers.base_provider import (
    ProviderCapability,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.sra_provider import (
    SRANotFoundError,
    SRAProvider,
    SRAProviderConfig,
    SRAProviderError,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for test session."""
    workspace = tmp_path_factory.mktemp("test_sra_provider_real_api")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    settings = get_settings()
    dm = DataManagerV2(workspace_dir=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def real_sra_provider(data_manager):
    """Create SRAProvider instance with real pysradb connection."""
    try:
        import pysradb
    except ImportError:
        pytest.skip("pysradb not installed - cannot run SRA integration tests")

    config = SRAProviderConfig(max_results=5)
    provider = SRAProvider(data_manager=data_manager, config=config)
    return provider


@pytest.fixture(scope="module")
def content_service(data_manager):
    """Create ContentAccessService instance for registry integration tests."""
    return ContentAccessService(data_manager=data_manager)


@pytest.fixture(scope="module")
def known_accessions():
    """Known stable SRA accessions for testing."""
    return {
        # Well-known GEUVADIS RNA-seq study
        "study": "SRP033351",
        "experiment": "SRX358615",  # Single experiment from GEUVADIS
        "run": "SRR1039508",  # Single run from GEUVADIS
        # PMID with known SRA datasets
        "pmid": "23845297",  # GEUVADIS publication
        # Human Microbiome Project study
        "microbiome_study": "SRP002163",
        # Alternative accessions for testing
        "rna_seq_study": "SRP012682",  # Another well-known RNA-seq study
    }


@pytest.fixture(scope="module")
def check_pysradb():
    """Verify pysradb is installed and functional."""
    try:
        import pysradb

        # Test basic import
        from pysradb import SRAweb

        logger.info("pysradb successfully imported")
    except ImportError:
        pytest.skip("pysradb not installed - run: pip install pysradb")
    except Exception as e:
        pytest.skip(f"pysradb import error: {e}")


# ============================================================================
# Test 1: Basic SRA Search with Accessions
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealSRASearch:
    """Test real SRA searches with actual API calls."""

    def test_search_by_study_accession(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test search with real SRA study accession (SRP)."""
        result = real_sra_provider.search_publications(known_accessions["study"])

        # Verify response format
        assert "SRA Database Search Results" in result
        assert known_accessions["study"] in result
        assert "Total Results:" in result

        # Should have metadata fields
        assert any(
            keyword in result.lower()
            for keyword in ["organism", "strategy", "platform", "layout"]
        )

        logger.info(f"✓ Study search test passed for {known_accessions['study']}")

    def test_search_by_experiment_accession(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test search with real SRA experiment accession (SRX)."""
        result = real_sra_provider.search_publications(known_accessions["experiment"])

        # Verify response format
        assert "SRA Database Search Results" in result
        assert known_accessions["experiment"] in result

        logger.info(
            f"✓ Experiment search test passed for {known_accessions['experiment']}"
        )

    def test_search_by_run_accession(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test search with real SRA run accession (SRR)."""
        result = real_sra_provider.search_publications(known_accessions["run"])

        # Verify response format
        assert "SRA Database Search Results" in result
        assert known_accessions["run"] in result

        logger.info(f"✓ Run search test passed for {known_accessions['run']}")

    def test_search_with_keyword_performs_real_search(
        self, real_sra_provider, check_pysradb
    ):
        """Test that keyword search performs actual SRA search using pysradb SraSearch."""
        result = real_sra_provider.search_publications(
            "RNA sequencing human cancer", max_results=5
        )

        # Should return formatted search results, not guidance message
        assert (
            "SRA Database Search Results" in result or "No SRA Results Found" in result
        )
        assert "RNA sequencing human cancer" in result
        # Should NOT contain old guidance message
        assert "Keyword Search Limitation" not in result

        logger.info("✓ Keyword search real API test passed")


# ============================================================================
# Test 2: Metadata Extraction
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealMetadataExtraction:
    """Test real metadata extraction from SRA."""

    def test_extract_metadata_study(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test metadata extraction for real SRA study."""
        metadata = real_sra_provider.extract_publication_metadata(
            known_accessions["study"]
        )

        # Verify metadata structure
        assert isinstance(metadata, PublicationMetadata)
        assert metadata.uid == known_accessions["study"]
        assert metadata.journal == "Sequence Read Archive (SRA)"
        assert len(metadata.title) > 0
        assert len(metadata.keywords) > 0

        # Should have organism in keywords (GEUVADIS is human data)
        assert any(
            "sapiens" in kw.lower() or "human" in kw.lower()
            for kw in metadata.keywords
            if kw
        )

        logger.info(f"✓ Study metadata extraction passed: {metadata.title[:50]}...")

    def test_extract_metadata_run(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test metadata extraction for real SRA run."""
        metadata = real_sra_provider.extract_publication_metadata(
            known_accessions["run"]
        )

        # Verify metadata structure
        assert isinstance(metadata, PublicationMetadata)
        assert metadata.uid == known_accessions["run"]
        assert metadata.journal == "Sequence Read Archive (SRA)"

        logger.info(f"✓ Run metadata extraction passed: {metadata.title[:50]}...")

    def test_extract_metadata_invalid_accession(self, real_sra_provider, check_pysradb):
        """Test metadata extraction with invalid accession."""
        # Use clearly invalid accession
        with pytest.raises(SRANotFoundError):
            real_sra_provider.extract_publication_metadata("SRP0000000000")

        logger.info("✓ Invalid accession handling test passed")


# ============================================================================
# Test 3: Publication-to-Dataset Linking
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealPublicationLinking:
    """Test real publication to SRA dataset linking."""

    @pytest.mark.slow
    def test_pmid_to_sra(self, real_sra_provider, known_accessions, check_pysradb):
        """Test finding SRA datasets from real PMID."""
        # Rate limiting - be respectful to NCBI
        time.sleep(0.5)

        result = real_sra_provider.find_datasets_from_publication(
            f"PMID:{known_accessions['pmid']}"
        )

        # Should either find datasets or report none found (both are valid)
        assert (
            "SRA Datasets Linked to Publication" in result
            or "No SRA Datasets Found" in result
        )
        assert "PMID" in result

        # If datasets found, should have proper formatting
        if "SRP" in result:
            assert "ncbi.nlm.nih.gov/sra/" in result
            assert "Total Datasets:" in result

        logger.info(
            f"✓ PMID-to-SRA linking test passed for PMID:{known_accessions['pmid']}"
        )

    def test_pmid_no_results(self, real_sra_provider, check_pysradb):
        """Test PMID with no linked SRA datasets."""
        # Use a very old PMID unlikely to have SRA data
        result = real_sra_provider.find_datasets_from_publication("PMID:10000000")

        # Should handle gracefully
        assert "No SRA Datasets Found" in result or "Error" in result

        logger.info("✓ PMID no results handling test passed")

    def test_doi_returns_guidance(self, real_sra_provider, check_pysradb):
        """Test that DOI returns conversion guidance."""
        result = real_sra_provider.find_datasets_from_publication("10.1038/nature12345")

        # Should provide guidance for DOI → PMID conversion
        assert "DOI Linking" in result
        assert "Convert DOI to PMID" in result

        logger.info("✓ DOI guidance test passed")

    def test_pmc_returns_guidance(self, real_sra_provider, check_pysradb):
        """Test that PMC ID returns conversion guidance."""
        result = real_sra_provider.find_datasets_from_publication("PMC1234567")

        # Should provide guidance for PMC → PMID conversion
        assert "PMC Linking" in result or "PMC" in result
        assert "PMID" in result

        logger.info("✓ PMC guidance test passed")


# ============================================================================
# Test 4: Microbiome/Metagenomics Search
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealMicrobiomeSearch:
    """Test real microbiome-specific searches."""

    @pytest.mark.slow
    def test_microbiome_search_16s(self, real_sra_provider, check_pysradb):
        """Test real 16S microbiome search with accession."""
        # Rate limiting
        time.sleep(0.5)

        # Use known microbiome study accession
        result = real_sra_provider.search_microbiome_datasets(
            "SRP002163",  # Human Microbiome Project
            amplicon_region="16S",
            body_site="gut",
            max_results=3,
        )

        # Should return results or guidance
        assert "microbiome" in result.lower() or "16s" in result.lower()
        assert "Microbiome Analysis Tips" in result

        logger.info("✓ 16S microbiome search test passed")

    def test_microbiome_search_with_accession(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test microbiome search with known microbiome accession."""
        result = real_sra_provider.search_microbiome_datasets(
            known_accessions["microbiome_study"],
            amplicon_region="16S",
            max_results=5,
        )

        # Should have results with microbiome guidance
        assert (
            known_accessions["microbiome_study"] in result or "Keyword Search" in result
        )
        assert "Microbiome Analysis Tips" in result

        logger.info(
            f"✓ Microbiome accession search test passed for {known_accessions['microbiome_study']}"
        )

    def test_microbiome_shotgun_metagenomics(self, real_sra_provider, check_pysradb):
        """Test shotgun metagenomics search (amplicon_region=None)."""
        # Use known microbiome study
        result = real_sra_provider.search_microbiome_datasets(
            "SRP002163",
            amplicon_region=None,  # Shotgun
            body_site="gut",
            max_results=3,
        )

        # Should have shotgun metagenomics guidance
        assert "Microbiome Analysis Tips" in result
        # Should mention shotgun or WGS
        assert "Shotgun" in result or "metagenomics" in result.lower()

        logger.info("✓ Shotgun metagenomics search test passed")


# ============================================================================
# Test 5: Filter Application
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealFilterSupport:
    """Test real filter application on SRA searches."""

    def test_organism_filter_human(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test organism filter on real human data."""
        filters = {"organism": "Homo sapiens"}
        result = real_sra_provider.search_publications(
            known_accessions["study"], filters=filters
        )

        # Should show filters applied
        assert "Homo sapiens" in result
        # GEUVADIS is human data, so should have results
        assert "Total Results:" in result

        logger.info("✓ Organism filter (human) test passed")

    def test_strategy_filter_rnaseq(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test library strategy filter on real RNA-seq data."""
        filters = {"strategy": "RNA-Seq"}
        result = real_sra_provider.search_publications(
            known_accessions["study"], filters=filters
        )

        # GEUVADIS is RNA-Seq, should have results
        assert "RNA-Seq" in result or "Total Results:" in result

        logger.info("✓ Strategy filter (RNA-Seq) test passed")

    def test_platform_filter_illumina(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test platform filter on real Illumina data."""
        filters = {"platform": "ILLUMINA"}
        result = real_sra_provider.search_publications(
            known_accessions["study"], filters=filters
        )

        # GEUVADIS uses Illumina, should have results
        assert "ILLUMINA" in result or "Illumina" in result

        logger.info("✓ Platform filter (Illumina) test passed")

    def test_multiple_filters(self, real_sra_provider, known_accessions, check_pysradb):
        """Test multiple filters applied simultaneously."""
        filters = {
            "organism": "Homo sapiens",
            "strategy": "RNA-Seq",
            "layout": "PAIRED",
        }
        result = real_sra_provider.search_publications(
            known_accessions["study"], filters=filters
        )

        # Should show all filters applied
        assert "Filters:" in result or all(
            f in result for f in ["Homo sapiens", "RNA-Seq"]
        )
        assert "Total Results:" in result

        logger.info("✓ Multiple filters test passed")


# ============================================================================
# Test 6: Performance Benchmarking
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestRealPerformance:
    """Test performance characteristics with real API."""

    def test_metadata_retrieval_performance(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test that metadata retrieval is reasonably fast."""
        start_time = time.time()

        metadata = real_sra_provider.extract_publication_metadata(
            known_accessions["study"]
        )

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Metadata retrieval too slow: {elapsed:.2f}s"
        assert isinstance(metadata, PublicationMetadata)

        logger.info(
            f"✓ Metadata retrieval performance test passed: {elapsed:.2f}s (target: <5s)"
        )

    def test_search_performance(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test that accession search is reasonably fast."""
        start_time = time.time()

        result = real_sra_provider.search_publications(known_accessions["run"])

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Search too slow: {elapsed:.2f}s"
        assert "SRA Database Search Results" in result

        logger.info(f"✓ Search performance test passed: {elapsed:.2f}s (target: <5s)")

    def test_batch_metadata_retrieval(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test batch metadata retrieval performance."""
        accessions = [
            known_accessions["study"],
            known_accessions["experiment"],
            known_accessions["run"],
        ]

        latencies = []
        for accession in accessions:
            start_time = time.time()
            metadata = real_sra_provider.extract_publication_metadata(accession)
            elapsed = time.time() - start_time
            latencies.append(elapsed)

            assert isinstance(metadata, PublicationMetadata)

            # Respectful rate limiting
            time.sleep(0.2)

        # Calculate statistics
        mean_latency = mean(latencies)
        max_latency = max(latencies)

        # Mean should be < 3s, max < 5s
        assert mean_latency < 3.0, f"Mean latency {mean_latency:.2f}s exceeds 3s target"
        assert max_latency < 5.0, f"Max latency {max_latency:.2f}s exceeds 5s target"

        logger.info(
            f"✓ Batch metadata retrieval test passed: mean={mean_latency:.2f}s, max={max_latency:.2f}s"
        )


# ============================================================================
# Test 7: Error Handling
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealErrorRecovery:
    """Test error handling with real API."""

    def test_invalid_accession_format(self, real_sra_provider, check_pysradb):
        """Test handling of invalid accession format."""
        # Should perform keyword search, not crash
        result = real_sra_provider.search_publications("INVALID123")

        # Should return search results or no results message (not guidance)
        assert (
            "SRA Database Search Results" in result or "No SRA Results Found" in result
        )
        # Should NOT contain old guidance message
        assert "Keyword Search Limitation" not in result

        logger.info("✓ Invalid accession format test passed")

    def test_non_existent_accession(self, real_sra_provider, check_pysradb):
        """Test handling of non-existent but valid-format accession."""
        # Valid format but non-existent
        with pytest.raises(SRANotFoundError):
            real_sra_provider.extract_publication_metadata("SRP0000000000")

        logger.info("✓ Non-existent accession test passed")

    def test_empty_query(self, real_sra_provider, check_pysradb):
        """Test handling of empty query."""
        # Empty string should attempt search and return no results
        result = real_sra_provider.search_publications("")

        # Should return no results message or search results (not guidance)
        assert (
            "No SRA Results Found" in result or "SRA Database Search Results" in result
        )
        # Should NOT contain old guidance message
        assert "Keyword Search Limitation" not in result

        logger.info("✓ Empty query test passed")

    def test_unsupported_identifier_type(self, real_sra_provider, check_pysradb):
        """Test handling of unsupported identifier types."""
        # Try with GEO accession (unsupported)
        result = real_sra_provider.find_datasets_from_publication("GSE123456")

        # Should provide guidance
        assert (
            "Unsupported Identifier Format" in result
            or "Keyword Search" in result
            or "PMID" in result
        )

        logger.info("✓ Unsupported identifier test passed")


# ============================================================================
# Test 8: Provider Registration
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestRealContentAccessServiceIntegration:
    """Test integration with ContentAccessService."""

    def test_provider_registration(self, content_service, check_pysradb):
        """Test that SRAProvider registers with ContentAccessService."""
        # Check that SRAProvider is registered
        providers = content_service.registry.get_all_providers()
        sra_providers = [p for p in providers if p.__class__.__name__ == "SRAProvider"]

        assert len(sra_providers) == 1, "SRAProvider should be registered once"

        logger.info("✓ SRAProvider registration test passed")

    def test_provider_capabilities(self, content_service, check_pysradb):
        """Test that SRAProvider reports correct capabilities."""
        providers = content_service.registry.get_all_providers()
        sra_provider = next(
            (p for p in providers if p.__class__.__name__ == "SRAProvider"), None
        )

        assert sra_provider is not None, "SRAProvider should be registered"

        # Check capabilities
        caps = sra_provider.get_supported_capabilities()

        # Should support these capabilities
        assert caps[ProviderCapability.DISCOVER_DATASETS] is True
        assert caps[ProviderCapability.FIND_LINKED_DATASETS] is True
        assert caps[ProviderCapability.EXTRACT_METADATA] is True
        assert caps[ProviderCapability.QUERY_CAPABILITIES] is True

        # Should NOT support these capabilities
        assert caps[ProviderCapability.SEARCH_LITERATURE] is False
        assert caps[ProviderCapability.GET_ABSTRACT] is False
        assert caps[ProviderCapability.GET_FULL_CONTENT] is False

        logger.info("✓ Provider capabilities test passed")

    def test_provider_priority(self, content_service, check_pysradb):
        """Test that SRAProvider has correct priority."""
        providers = content_service.registry.get_all_providers()
        sra_provider = next(
            (p for p in providers if p.__class__.__name__ == "SRAProvider"), None
        )

        assert sra_provider is not None
        assert sra_provider.priority == 10, "SRAProvider should have high priority (10)"

        logger.info("✓ Provider priority test passed")

    def test_provider_source(self, content_service, check_pysradb):
        """Test that SRAProvider reports correct source."""
        providers = content_service.registry.get_all_providers()
        sra_provider = next(
            (p for p in providers if p.__class__.__name__ == "SRAProvider"), None
        )

        assert sra_provider is not None
        assert sra_provider.source == PublicationSource.SRA

        logger.info("✓ Provider source test passed")


# ============================================================================
# Test 9: End-to-End Workflows
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestRealEndToEndWorkflows:
    """Test complete end-to-end workflows with real data."""

    def test_pmid_to_metadata_workflow(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test complete workflow: PMID → SRA accessions → metadata."""
        # Step 1: Find datasets from publication
        time.sleep(0.5)  # Rate limiting
        datasets_result = real_sra_provider.find_datasets_from_publication(
            f"PMID:{known_accessions['pmid']}"
        )

        # Should find datasets or report none found
        assert "SRA Datasets" in datasets_result

        # If datasets found, extract metadata for one
        if "SRP" in datasets_result:
            # Extract first SRP accession from result
            srp_match = re.search(r"(SRP\d{6,})", datasets_result)

            if srp_match:
                srp = srp_match.group(1)

                time.sleep(0.5)  # Rate limiting
                metadata = real_sra_provider.extract_publication_metadata(srp)

                assert isinstance(metadata, PublicationMetadata)
                assert metadata.journal == "Sequence Read Archive (SRA)"
                assert len(metadata.keywords) > 0

                logger.info(f"✓ PMID-to-metadata workflow passed: {srp}")
        else:
            logger.info(
                "✓ PMID-to-metadata workflow passed (no datasets found - valid)"
            )

    def test_accession_search_to_metadata_workflow(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test workflow: Accession search → Metadata extraction."""
        # Step 1: Search by accession
        search_result = real_sra_provider.search_publications(known_accessions["study"])

        assert "SRA Database Search Results" in search_result
        assert known_accessions["study"] in search_result

        # Step 2: Extract detailed metadata
        time.sleep(0.3)
        metadata = real_sra_provider.extract_publication_metadata(
            known_accessions["study"]
        )

        assert isinstance(metadata, PublicationMetadata)
        assert metadata.uid == known_accessions["study"]
        assert len(metadata.title) > 0

        logger.info(
            f"✓ Search-to-metadata workflow passed for {known_accessions['study']}"
        )

    def test_microbiome_discovery_workflow(
        self, real_sra_provider, known_accessions, check_pysradb
    ):
        """Test microbiome discovery: Search → Metadata → Analysis guidance."""
        # Step 1: Microbiome-specific search
        time.sleep(0.5)
        search_result = real_sra_provider.search_microbiome_datasets(
            known_accessions["microbiome_study"],
            amplicon_region="16S",
            body_site="gut",
            max_results=3,
        )

        assert "Microbiome Analysis Tips" in search_result

        # Step 2: Extract metadata for found study
        time.sleep(0.3)
        metadata = real_sra_provider.extract_publication_metadata(
            known_accessions["microbiome_study"]
        )

        assert isinstance(metadata, PublicationMetadata)
        assert metadata.uid == known_accessions["microbiome_study"]

        # Should have organism in keywords
        assert len(metadata.keywords) > 0

        logger.info(
            f"✓ Microbiome discovery workflow passed for {known_accessions['microbiome_study']}"
        )


# ============================================================================
# Test Summary
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestSRAProviderSummary:
    """Summary test to validate overall SRAProvider functionality."""

    def test_provider_summary(self, real_sra_provider, check_pysradb):
        """Comprehensive summary test of SRAProvider capabilities."""
        # Test 1: Provider initialization
        assert real_sra_provider is not None
        assert real_sra_provider.source == PublicationSource.SRA

        # Test 2: Capability reporting
        caps = real_sra_provider.get_supported_capabilities()
        assert caps[ProviderCapability.DISCOVER_DATASETS] is True
        assert caps[ProviderCapability.EXTRACT_METADATA] is True

        # Test 3: Accession pattern detection
        assert real_sra_provider._is_sra_accession("SRP123456") is True
        assert real_sra_provider._is_sra_accession("SRX123456") is True
        assert real_sra_provider._is_sra_accession("SRR123456") is True
        assert real_sra_provider._is_sra_accession("keyword search") is False

        # Test 4: Basic search functionality
        result = real_sra_provider.search_publications("SRP033351")
        assert (
            "SRA Database Search Results" in result or "No SRA Results Found" in result
        )

        logger.info("✓ SRAProvider comprehensive summary test passed")
        logger.info("=" * 70)
        logger.info("SRAProvider Integration Test Suite Complete")
        logger.info("All core functionality validated with real pysradb API calls")
        logger.info("=" * 70)
