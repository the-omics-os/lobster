"""
Unit tests for DiseaseOntologyService

Phase 1: Keyword-based matching from JSON config
Tests the migration-stable API that will work with Phase 2 (embeddings) backend swap.
"""

import pytest

from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test to ensure fresh state."""
    DiseaseOntologyService.reset_instance()
    yield
    DiseaseOntologyService.reset_instance()


@pytest.fixture
def service():
    """Create a service instance."""
    return DiseaseOntologyService()


class TestMatchDiseaseAPI:
    """Test the match_disease() API - the Phase 2 compatible interface."""

    def test_match_disease_exact_keyword(self, service):
        """Test Phase 1 keyword matching returns correct disease."""
        matches = service.match_disease("colorectal cancer")

        assert len(matches) >= 1
        assert matches[0].disease_id == "crc"
        assert matches[0].confidence == 1.0
        assert matches[0].match_type == "exact_keyword"
        assert matches[0].name == "Colorectal Cancer"

    def test_match_disease_no_match(self, service):
        """Test no match returns empty list."""
        matches = service.match_disease("diabetes mellitus")

        assert matches == []

    def test_match_disease_case_insensitive(self, service):
        """Test case-insensitive matching works."""
        # Test uppercase
        matches_upper = service.match_disease("CROHN'S DISEASE")
        assert len(matches_upper) >= 1
        assert matches_upper[0].disease_id == "cd"

        # Test mixed case
        matches_mixed = service.match_disease("Ulcerative Colitis")
        assert len(matches_mixed) >= 1
        assert matches_mixed[0].disease_id == "uc"

        # Test lowercase
        matches_lower = service.match_disease("crc")
        assert len(matches_lower) >= 1
        assert matches_lower[0].disease_id == "crc"

    def test_metadata_fields_present(self, service):
        """Test MONDO IDs and other metadata present for Phase 2."""
        matches = service.match_disease("crc")

        assert len(matches) >= 1
        assert matches[0].metadata["mondo_id"] == "MONDO:0005575"
        assert matches[0].metadata["umls_cui"] == "C0009402"
        assert matches[0].metadata["mesh_terms"] == ["D015179"]

    def test_match_disease_k_parameter(self, service):
        """Test k parameter limits results (Phase 2 compatibility)."""
        # In Phase 1, we typically get 1 result, but k should be respected
        matches = service.match_disease("colorectal cancer", k=1)
        assert len(matches) <= 1

        matches = service.match_disease("colorectal cancer", k=5)
        assert len(matches) <= 5

    def test_match_disease_min_confidence(self, service):
        """Test min_confidence parameter (Phase 2 compatibility)."""
        # In Phase 1, all matches are 1.0 confidence
        matches_high = service.match_disease("crc", min_confidence=0.99)
        assert len(matches_high) >= 1  # Should pass since Phase 1 is always 1.0

        matches_low = service.match_disease("crc", min_confidence=0.1)
        assert len(matches_low) >= 1  # Should also pass


class TestLegacyAPI:
    """Test legacy APIs for backward compatibility during Phase 1 migration."""

    def test_legacy_api_backward_compat(self, service):
        """Test get_extraction_keywords() works for migration period."""
        keywords = service.get_extraction_keywords()

        assert "crc" in keywords
        assert "uc" in keywords
        assert "cd" in keywords
        assert "healthy" in keywords

        # Check specific keywords
        assert "colorectal cancer" in keywords["crc"]
        assert "ulcerative colitis" in keywords["uc"]
        assert "crohn's disease" in keywords["cd"]
        assert "healthy control" in keywords["healthy"]

    def test_get_standardization_variants(self, service):
        """Test get_standardization_variants() for DiseaseStandardizationService."""
        variants = service.get_standardization_variants()

        assert "crc" in variants
        assert isinstance(variants["crc"], list)
        assert len(variants["crc"]) > 0

    def test_get_all_disease_ids(self, service):
        """Test get_all_disease_ids() returns valid IDs."""
        disease_ids = service.get_all_disease_ids()

        assert "crc" in disease_ids
        assert "uc" in disease_ids
        assert "cd" in disease_ids
        assert "healthy" in disease_ids
        assert len(disease_ids) == 4

    def test_validate_disease_id(self, service):
        """Test validate_disease_id() works correctly."""
        assert service.validate_disease_id("crc") is True
        assert service.validate_disease_id("uc") is True
        assert service.validate_disease_id("invalid_id") is False

    def test_get_disease_by_id(self, service):
        """Test get_disease_by_id() returns disease concept."""
        disease = service.get_disease_by_id("crc")

        assert disease is not None
        assert disease.id == "crc"
        assert disease.name == "Colorectal Cancer"
        assert disease.mondo_id == "MONDO:0005575"

        # Test invalid ID
        invalid = service.get_disease_by_id("invalid_id")
        assert invalid is None


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        # Reset to ensure clean state
        DiseaseOntologyService.reset_instance()

        service1 = DiseaseOntologyService.get_instance()
        service2 = DiseaseOntologyService.get_instance()

        assert service1 is service2

    def test_reset_instance(self):
        """Test reset_instance() creates new singleton."""
        service1 = DiseaseOntologyService.get_instance()
        DiseaseOntologyService.reset_instance()
        service2 = DiseaseOntologyService.get_instance()

        assert service1 is not service2


class TestConfigLoading:
    """Test JSON config loading and validation."""

    def test_config_version(self, service):
        """Test config version is loaded."""
        assert service._config.version == "1.0.0"
        assert service._config.backend == "json"

    def test_all_diseases_loaded(self, service):
        """Test all 4 diseases are loaded from config."""
        assert len(service._diseases) == 4

        disease_ids = [d.id for d in service._diseases]
        assert "crc" in disease_ids
        assert "uc" in disease_ids
        assert "cd" in disease_ids
        assert "healthy" in disease_ids

    def test_keywords_merged_correctly(self, service):
        """Test keywords from both sources are merged."""
        crc_keywords = service.get_extraction_keywords()["crc"]

        # Keywords from disease_standardization_service.py
        assert "colorectal cancer" in crc_keywords
        assert "colon cancer" in crc_keywords

        # Keywords from metadata_assistant.py
        assert "colorectal" in crc_keywords
        assert "crc" in crc_keywords


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, service):
        """Test empty query returns empty list."""
        matches = service.match_disease("")
        assert matches == []

    def test_whitespace_query(self, service):
        """Test whitespace-only query returns empty list."""
        matches = service.match_disease("   ")
        assert matches == []

    def test_partial_keyword_match(self, service):
        """Test partial keyword in query string."""
        # "I have colorectal cancer stage II" should match CRC
        matches = service.match_disease("I have colorectal cancer stage II")
        assert len(matches) >= 1
        assert matches[0].disease_id == "crc"

    def test_multiple_keywords_in_query(self, service):
        """Test query with multiple disease keywords."""
        # First match wins in Phase 1
        matches = service.match_disease("colorectal cancer and crohn's disease")
        assert len(matches) >= 1
        # Should match one of them (order depends on disease iteration)
        assert matches[0].disease_id in ["crc", "cd"]

    def test_healthy_control_variants(self, service):
        """Test healthy control keyword variants."""
        test_cases = [
            ("healthy", "healthy"),
            ("healthy control", "healthy"),
            ("healthy volunteer", "healthy"),
            ("non-ibd", "healthy"),
            ("disease-free", "healthy"),
        ]

        for query, expected_id in test_cases:
            matches = service.match_disease(query)
            assert len(matches) >= 1, f"Failed to match '{query}'"
            assert matches[0].disease_id == expected_id, f"Wrong match for '{query}'"
