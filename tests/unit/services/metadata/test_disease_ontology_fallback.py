"""
Unit tests for DiseaseOntologyService fallback behavior (public version).

Tests that DiseaseStandardizationService and metadata_assistant work correctly
when DiseaseOntologyService is not available (simulating lobster-local public package).
"""

import sys
from unittest.mock import patch

import pandas as pd
import pytest


class TestDiseaseStandardizationServiceFallback:
    """Test DiseaseStandardizationService graceful fallback."""

    def test_fallback_to_hardcoded_mappings(self):
        """Test that service works when DiseaseOntologyService is unavailable."""
        # Simulate public version by hiding the ontology service
        with patch.dict(
            sys.modules, {"lobster.services.metadata.disease_ontology_service": None}
        ):
            # Force reimport with service unavailable
            import importlib

            if (
                "lobster.services.metadata.disease_standardization_service"
                in sys.modules
            ):
                del sys.modules[
                    "lobster.services.metadata.disease_standardization_service"
                ]

            from lobster.services.metadata.disease_standardization_service import (
                DiseaseStandardizationService,
                HAS_ONTOLOGY_SERVICE,
            )

            # Verify fallback is active
            assert HAS_ONTOLOGY_SERVICE is False

            # Create service (should use fallback)
            service = DiseaseStandardizationService()

            # Verify DISEASE_MAPPINGS populated from fallback
            assert service.DISEASE_MAPPINGS is not None
            assert "crc" in service.DISEASE_MAPPINGS
            assert "colorectal cancer" in service.DISEASE_MAPPINGS["crc"]

            # Test standardization works with fallback
            metadata = pd.DataFrame(
                {
                    "disease": [
                        "colorectal cancer",
                        "ulcerative colitis",
                        "crohn's disease",
                    ]
                }
            )
            result, stats, ir = service.standardize_disease_terms(metadata, "disease")

            assert result["disease"].tolist() == ["crc", "uc", "cd"]
            assert stats["total_samples"] == 3
            assert stats["standardization_rate"] == 100.0


class TestMetadataAssistantFallback:
    """Test metadata_assistant._phase1_column_rescan() fallback."""

    def test_fallback_to_hardcoded_keywords(self):
        """Test that _phase1_column_rescan works when DiseaseOntologyService is unavailable."""
        # Simulate public version
        with patch.dict(
            sys.modules, {"lobster.services.metadata.disease_ontology_service": None}
        ):
            # Force reimport
            if "lobster.agents.metadata_assistant.config" in sys.modules:
                del sys.modules["lobster.agents.metadata_assistant.config"]

            from lobster.agents.metadata_assistant.config import (
                HAS_ONTOLOGY_SERVICE,
                phase1_column_rescan as _phase1_column_rescan,
            )

            # Verify fallback is active
            assert HAS_ONTOLOGY_SERVICE is False

            # Test with sample data
            samples = [
                {"run_accession": "SRR123", "tissue": "colorectal tissue"},
                {"run_accession": "SRR124", "disease_type": "ulcerative colitis"},
                {"run_accession": "SRR125", "condition": "healthy control"},
            ]

            enriched_count, log = _phase1_column_rescan(samples)

            # Verify all samples enriched
            assert enriched_count == 3
            assert samples[0]["disease"] == "crc"
            assert samples[1]["disease"] == "uc"
            assert samples[2]["disease"] == "healthy"

            # Verify all have disease_confidence
            assert all(s.get("disease_confidence") == 1.0 for s in samples)


class TestBothPathsProduceSameResults:
    """Verify premium and public paths produce identical results for Phase 1."""

    def test_disease_standardization_parity(self):
        """Test that both premium and public paths produce same standardization results."""
        # Test data
        test_diseases = [
            "colorectal cancer",
            "ulcerative colitis",
            "crohn's disease",
            "healthy",
        ]

        # Test with premium path (service available)
        from lobster.services.metadata.disease_standardization_service import (
            DiseaseStandardizationService,
        )

        service_premium = DiseaseStandardizationService()
        metadata = pd.DataFrame({"disease": test_diseases})
        result_premium, stats_premium, _ = service_premium.standardize_disease_terms(
            metadata, "disease"
        )

        # Test with public path (simulated)
        with patch.dict(
            sys.modules, {"lobster.services.metadata.disease_ontology_service": None}
        ):
            if (
                "lobster.services.metadata.disease_standardization_service"
                in sys.modules
            ):
                del sys.modules[
                    "lobster.services.metadata.disease_standardization_service"
                ]

            from lobster.services.metadata.disease_standardization_service import (
                DiseaseStandardizationService as DiseaseStandardizationServicePublic,
            )

            service_public = DiseaseStandardizationServicePublic()
            metadata_public = pd.DataFrame({"disease": test_diseases})
            result_public, stats_public, _ = service_public.standardize_disease_terms(
                metadata_public, "disease"
            )

            # Verify identical results
            assert (
                result_premium["disease"].tolist() == result_public["disease"].tolist()
            )
            assert (
                stats_premium["standardization_rate"]
                == stats_public["standardization_rate"]
            )
