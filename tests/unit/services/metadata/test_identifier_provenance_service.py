"""
Unit tests for IdentifierProvenanceService.

Tests the 2-layer provenance validation system:
1. Section-based extraction (Data Availability → primary, Methods/Body → uncertain)
2. E-Link validation (BioProject → PubMed publication link)
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.services.metadata.identifier_provenance_service import (
    IdentifierProvenanceService,
    IdentifierWithProvenance,
)


class TestSectionBasedProvenance:
    """Test section-based provenance detection (Layer 1)."""

    def test_data_availability_section_is_primary(self):
        """Identifiers in Data Availability section should be marked as primary."""
        service = IdentifierProvenanceService()

        full_text = """
        Methods: We used data from PRJNA436359 for comparison.

        Data Availability: Raw sequencing data have been deposited at
        EGAD50000000740 and GSE12345.
        """
        data_availability = (
            "Raw sequencing data have been deposited at EGAD50000000740 and GSE12345."
        )

        results = service.extract_and_validate(
            full_text=full_text,
            data_availability_text=data_availability,
            validate_elink=False,
        )

        # Group by accession
        by_acc = {r.accession: r for r in results}

        # DA section identifiers should be primary
        assert by_acc["EGAD50000000740"].provenance == "primary"
        assert by_acc["EGAD50000000740"].source_section == "data_availability"
        assert by_acc["EGAD50000000740"].confidence >= 0.9

        assert by_acc["GSE12345"].provenance == "primary"
        assert by_acc["GSE12345"].source_section == "data_availability"

        # Methods section identifier should be uncertain
        assert by_acc["PRJNA436359"].provenance == "uncertain"
        assert by_acc["PRJNA436359"].source_section == "methods_or_body"
        assert by_acc["PRJNA436359"].confidence < 0.5

    def test_methods_body_section_is_uncertain(self):
        """Identifiers outside Data Availability should be marked as uncertain."""
        service = IdentifierProvenanceService()

        full_text = """
        We downloaded public data from GSE12345 and PRJNA123456 to
        compare with our results. Statistical analysis was performed
        using data from SRP789012.
        """

        results = service.extract_and_validate(
            full_text=full_text,
            data_availability_text="",  # No DA section
            validate_elink=False,
        )

        # All should be uncertain (no DA section)
        for r in results:
            assert r.provenance == "uncertain"
            assert r.source_section == "methods_or_body"
            assert r.confidence < 0.5

    def test_empty_data_availability_section(self):
        """Test with empty Data Availability section."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data from GSE12345 was analyzed.",
            data_availability_text="",
            validate_elink=False,
        )

        assert len(results) == 1
        assert results[0].accession == "GSE12345"
        assert results[0].provenance == "uncertain"


class TestAccessTypeDetection:
    """Test access type detection (open, controlled, embargoed)."""

    def test_ega_is_controlled_access(self):
        """EGA identifiers should be marked as controlled access."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data deposited at EGAD50000000740",
            data_availability_text="Data deposited at EGAD50000000740",
            validate_elink=False,
        )

        assert len(results) == 1
        assert results[0].access_type == "controlled"
        assert results[0].is_downloadable is False  # controlled = not auto-downloadable

    def test_geo_is_open_access(self):
        """GEO identifiers should be marked as open access."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data deposited at GSE12345",
            data_availability_text="Data deposited at GSE12345",
            validate_elink=False,
        )

        assert len(results) == 1
        assert results[0].access_type == "open"
        assert results[0].is_downloadable is True  # open + primary = downloadable

    def test_downloadability_requires_open_and_primary(self):
        """Only open access + primary provenance = downloadable."""
        service = IdentifierProvenanceService()

        full_text = """
        We used public data from GSE99999 in our methods section.

        Data Availability: Our data is at GSE12345 and EGAD50000000740.
        """
        da_text = "Our data is at GSE12345 and EGAD50000000740."

        results = service.extract_and_validate(
            full_text=full_text,
            data_availability_text=da_text,
            validate_elink=False,
        )

        by_acc = {r.accession: r for r in results}

        # GSE12345: open + primary → downloadable
        assert by_acc["GSE12345"].access_type == "open"
        assert by_acc["GSE12345"].provenance == "primary"
        assert by_acc["GSE12345"].is_downloadable is True

        # EGAD50000000740: controlled + primary → NOT downloadable
        assert by_acc["EGAD50000000740"].access_type == "controlled"
        assert by_acc["EGAD50000000740"].provenance == "primary"
        assert by_acc["EGAD50000000740"].is_downloadable is False

        # GSE99999: open + uncertain → NOT downloadable
        assert by_acc["GSE99999"].access_type == "open"
        assert by_acc["GSE99999"].provenance == "uncertain"
        assert by_acc["GSE99999"].is_downloadable is False


class TestELinkValidation:
    """Test E-Link validation for BioProject identifiers (Layer 2)."""

    def test_elink_validates_bioproject_publication_link(self):
        """Test that E-Link validation updates provenance for BioProjects."""
        # Mock PubMed provider
        mock_provider = MagicMock()
        mock_provider.validate_bioproject_publication_link.return_value = {
            "provenance": "referenced",
            "confidence": 0.1,
            "linked_pmids": ["12345678"],
            "error": None,
        }

        service = IdentifierProvenanceService(pubmed_provider=mock_provider)

        # BioProject in methods section (not DA)
        results = service.extract_and_validate(
            full_text="We used PRJNA436359 from a previous study.",
            data_availability_text="",
            source_pmid="39380095",
            validate_elink=True,
        )

        # Should have called E-Link validation
        mock_provider.validate_bioproject_publication_link.assert_called_once_with(
            "PRJNA436359", "39380095"
        )

        # Should be marked as referenced
        assert len(results) == 1
        assert results[0].provenance == "referenced"
        assert results[0].confidence == 0.1
        assert results[0].linked_pmid == "12345678"

    def test_elink_skipped_for_data_availability_identifiers(self):
        """E-Link validation should be skipped for DA section identifiers."""
        mock_provider = MagicMock()

        service = IdentifierProvenanceService(pubmed_provider=mock_provider)

        # BioProject in Data Availability section
        results = service.extract_and_validate(
            full_text="Data at PRJNA123456",
            data_availability_text="Data at PRJNA123456",
            source_pmid="12345678",
            validate_elink=True,
        )

        # Should NOT call E-Link (DA section already has high confidence)
        mock_provider.validate_bioproject_publication_link.assert_not_called()

        # Should be primary from DA section
        assert results[0].provenance == "primary"
        assert results[0].confidence >= 0.9

    def test_elink_skipped_when_no_pmid(self):
        """E-Link validation requires source PMID."""
        mock_provider = MagicMock()

        service = IdentifierProvenanceService(pubmed_provider=mock_provider)

        results = service.extract_and_validate(
            full_text="We used PRJNA436359 from a previous study.",
            data_availability_text="",
            source_pmid=None,  # No PMID
            validate_elink=True,
        )

        # Should NOT call E-Link (no PMID)
        mock_provider.validate_bioproject_publication_link.assert_not_called()

        # Should remain uncertain
        assert results[0].provenance == "uncertain"

    def test_elink_error_handling(self):
        """E-Link validation errors should be logged but not fail."""
        mock_provider = MagicMock()
        mock_provider.validate_bioproject_publication_link.side_effect = Exception(
            "API error"
        )

        service = IdentifierProvenanceService(pubmed_provider=mock_provider)

        # Should not raise
        results = service.extract_and_validate(
            full_text="We used PRJNA436359 from a previous study.",
            data_availability_text="",
            source_pmid="12345678",
            validate_elink=True,
        )

        # Should remain uncertain with error note
        assert len(results) == 1
        assert results[0].provenance == "uncertain"
        assert "E-Link validation failed" in results[0].validation_notes


class TestFilterMethods:
    """Test filter methods for identifier lists."""

    def test_filter_downloadable(self):
        """Test filtering to only downloadable identifiers."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data from GSE99999 in methods. Data at GSE12345 and EGAD50000000740.",
            data_availability_text="Data at GSE12345 and EGAD50000000740.",
            validate_elink=False,
        )

        downloadable = service.filter_downloadable(results)

        # Only GSE12345 should be downloadable (open + primary)
        assert len(downloadable) == 1
        assert downloadable[0].accession == "GSE12345"

    def test_filter_controlled_access(self):
        """Test filtering to only controlled-access identifiers."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data at GSE12345 and EGAD50000000740.",
            data_availability_text="Data at GSE12345 and EGAD50000000740.",
            validate_elink=False,
        )

        controlled = service.filter_controlled_access(results)

        assert len(controlled) == 1
        assert controlled[0].accession == "EGAD50000000740"

    def test_filter_by_provenance(self):
        """Test filtering by provenance type."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Methods used GSE99999. Data Availability: GSE12345.",
            data_availability_text="GSE12345.",
            validate_elink=False,
        )

        primary = service.filter_by_provenance(results, "primary")
        uncertain = service.filter_by_provenance(results, "uncertain")

        assert len(primary) == 1
        assert primary[0].accession == "GSE12345"

        assert len(uncertain) == 1
        assert uncertain[0].accession == "GSE99999"


class TestUserNotification:
    """Test user notification generation."""

    def test_notification_for_controlled_access(self):
        """Test notification generation for controlled-access identifiers."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data at EGAD50000000740",
            data_availability_text="Data at EGAD50000000740",
            validate_elink=False,
        )

        notification = service.get_user_notification(results)

        assert notification is not None
        assert "controlled-access" in notification.lower()
        assert "EGAD50000000740" in notification
        assert "DAC" in notification

    def test_notification_for_referenced_identifiers(self):
        """Test notification for referenced (excluded) identifiers."""
        service = IdentifierProvenanceService()

        # Create a mock result with referenced identifier
        results = [
            IdentifierWithProvenance(
                accession="PRJNA436359",
                database="BioProject",
                field_name="bioproject_accession",
                access_type="open",
                source_section="methods_or_body",
                provenance="referenced",
                confidence=0.1,
                is_downloadable=False,
                linked_pmid="12345678",
                validation_notes="",
            )
        ]

        notification = service.get_user_notification(results)

        assert notification is not None
        assert "referenced" in notification.lower()
        assert "PRJNA436359" in notification
        assert "Excluded" in notification

    def test_no_notification_when_all_downloadable(self):
        """Test no notification when all identifiers are downloadable."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data at GSE12345",
            data_availability_text="Data at GSE12345",
            validate_elink=False,
        )

        notification = service.get_user_notification(results)

        # No notification needed (all open + primary)
        assert notification is None


class TestSerialization:
    """Test serialization to dict format."""

    def test_to_dict_list(self):
        """Test conversion to list of dictionaries."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data at GSE12345 and EGAD50000000740",
            data_availability_text="Data at GSE12345 and EGAD50000000740",
            validate_elink=False,
        )

        dict_list = service.to_dict_list(results)

        assert len(dict_list) == 2
        assert all(isinstance(d, dict) for d in dict_list)

        # Check required fields
        for d in dict_list:
            assert "accession" in d
            assert "database" in d
            assert "access_type" in d
            assert "provenance" in d
            assert "confidence" in d
            assert "is_downloadable" in d


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self):
        """Test with empty input text."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="",
            data_availability_text="",
            validate_elink=False,
        )

        assert results == []

    def test_no_identifiers_found(self):
        """Test with text containing no identifiers."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="This paper discusses cancer biology.",
            data_availability_text="Data available on request.",
            validate_elink=False,
        )

        assert results == []

    def test_duplicate_identifiers_deduplicated(self):
        """Test that duplicate identifiers are deduplicated."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Data at GSE12345. We analyzed GSE12345. Results from GSE12345.",
            data_availability_text="Data at GSE12345",
            validate_elink=False,
        )

        assert len(results) == 1
        assert results[0].accession == "GSE12345"

    def test_sorting_by_confidence(self):
        """Test that results are sorted by confidence (highest first)."""
        service = IdentifierProvenanceService()

        results = service.extract_and_validate(
            full_text="Methods used GSE99999. Data Availability: GSE12345.",
            data_availability_text="GSE12345.",
            validate_elink=False,
        )

        # Should be sorted by confidence (primary first)
        assert results[0].confidence > results[1].confidence
        assert results[0].accession == "GSE12345"  # primary (0.95)
        assert results[1].accession == "GSE99999"  # uncertain (0.3)
