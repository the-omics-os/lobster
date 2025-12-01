"""
Unit tests for AccessionResolver.

Tests comprehensive identifier detection, validation, and text extraction
across all supported biobank databases (GEO, SRA, ENA, DDBJ, PRIDE, MassIVE, etc.).
"""

import pytest

from lobster.core.identifiers import (
    AccessionResolver,
    get_accession_resolver,
    reset_resolver,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test for isolation."""
    reset_resolver()
    yield
    reset_resolver()


class TestAccessionDetection:
    """Test database detection for various accession types."""

    @pytest.mark.parametrize(
        "accession,expected_db_contains",
        [
            # GEO accessions
            ("GSE12345", "Gene Expression Omnibus"),
            ("GSE194247", "Gene Expression Omnibus"),
            ("GSM1234567", "Gene Expression Omnibus"),
            ("GPL570", "Gene Expression Omnibus"),
            ("GDS5093", "Gene Expression Omnibus"),
            # NCBI SRA accessions
            ("SRP116709", "Sequence Read Archive"),
            ("SRX123456", "Sequence Read Archive"),
            ("SRR1234567", "Sequence Read Archive"),
            ("SRS123456", "Sequence Read Archive"),
            # ENA accessions
            ("ERP123456", "ENA"),
            ("ERX123456", "ENA"),
            ("ERR123456", "ENA"),
            ("ERS123456", "ENA"),
            # DDBJ accessions
            ("DRP123456", "DDBJ"),
            ("DRX123456", "DDBJ"),
            ("DRR123456", "DDBJ"),
            ("DRS123456", "DDBJ"),
            # BioProject/BioSample
            ("PRJNA123456", "BioProject"),
            ("PRJEB83385", "BioProject"),
            ("PRJDB12345", "BioProject"),
            ("SAMN12345678", "BioSample"),
            ("SAMEA123456", "BioSample"),
            ("SAMD12345678", "BioSample"),
            # Proteomics
            ("PXD012345", "PRIDE"),
            ("MSV000012345", "MassIVE"),
            # Metabolomics
            ("MTBLS1234", "MetaboLights"),
            ("ST001234", "Metabolomics Workbench"),
            # Metagenomics
            ("MGYS00001234", "MGnify"),
            # Cross-platform
            ("E-MTAB-12345", "ArrayExpress"),
            ("10.1038/nature12345", "Digital Object Identifier"),
        ],
    )
    def test_detect_database(self, accession, expected_db_contains):
        """Test that accessions are correctly detected."""
        resolver = get_accession_resolver()
        result = resolver.detect_database(accession)

        assert result is not None, f"Failed to detect database for {accession}"
        assert (
            expected_db_contains.lower() in result.lower()
        ), f"Expected '{expected_db_contains}' in '{result}' for {accession}"

    def test_detect_database_case_insensitive(self):
        """Test case-insensitive detection."""
        resolver = get_accession_resolver()

        assert resolver.detect_database("gse12345") is not None
        assert resolver.detect_database("GSE12345") is not None
        assert resolver.detect_database("Gse12345") is not None

    def test_detect_database_with_whitespace(self):
        """Test detection with leading/trailing whitespace."""
        resolver = get_accession_resolver()

        assert resolver.detect_database("  GSE12345  ") is not None
        assert resolver.detect_database("\tPXD012345\n") is not None

    def test_detect_database_invalid(self):
        """Test that invalid accessions return None."""
        resolver = get_accession_resolver()

        assert resolver.detect_database("INVALID123") is None
        assert resolver.detect_database("ABC") is None
        assert resolver.detect_database("") is None
        assert resolver.detect_database("   ") is None


class TestAccessionValidation:
    """Test validation functionality."""

    def test_validate_any_database(self):
        """Test validation against any database."""
        resolver = get_accession_resolver()

        assert resolver.validate("GSE12345") is True
        assert resolver.validate("PXD012345") is True
        assert resolver.validate("INVALID") is False

    def test_validate_specific_database(self):
        """Test validation against specific database."""
        resolver = get_accession_resolver()

        # Should pass when database matches
        assert resolver.validate("GSE12345", database="GEO") is True
        assert resolver.validate("PXD012345", database="PRIDE") is True
        assert resolver.validate("SRP123456", database="SRA") is True

        # Should fail when database doesn't match
        assert resolver.validate("GSE12345", database="PRIDE") is False
        assert resolver.validate("PXD012345", database="GEO") is False


class TestTextExtraction:
    """Test accession extraction from text."""

    def test_extract_from_abstract(self):
        """Test extraction from sample abstract text."""
        resolver = get_accession_resolver()

        text = """
        RNA-seq data has been deposited in GEO under accession GSE123456.
        Raw sequencing data is available from SRA (SRP789012).
        Proteomics data was submitted to PRIDE (PXD012345).
        """

        result = resolver.extract_accessions_by_type(text)

        assert "GEO" in result
        assert "GSE123456" in result["GEO"]
        assert "SRA" in result
        assert "SRP789012" in result["SRA"]
        assert "PRIDE" in result
        assert "PXD012345" in result["PRIDE"]

    def test_extract_multiple_same_type(self):
        """Test extraction of multiple accessions of same type."""
        resolver = get_accession_resolver()

        text = "Samples GSE111, GSE222, and GSE333 were analyzed."
        result = resolver.extract_accessions_by_type(text)

        assert "GEO" in result
        assert len(result["GEO"]) == 3

    def test_extract_international_accessions(self):
        """Test extraction of ENA/DDBJ accessions."""
        resolver = get_accession_resolver()

        text = "Data from PRJEB12345 (ENA) and PRJDB67890 (DDBJ) were combined."
        result = resolver.extract_accessions_by_type(text)

        assert "BioProject" in result
        assert "PRJEB12345" in result["BioProject"]
        assert "PRJDB67890" in result["BioProject"]

    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        resolver = get_accession_resolver()

        assert resolver.extract_accessions_by_type("") == {}
        assert resolver.extract_accessions_by_type(None) == {}

    def test_extract_no_accessions(self):
        """Test extraction from text without accessions."""
        resolver = get_accession_resolver()

        text = "This is a paper about cancer biology."
        result = resolver.extract_accessions_by_type(text)

        assert result == {}

    def test_extract_preserves_case(self):
        """Test that extracted accessions are normalized to uppercase."""
        resolver = get_accession_resolver()

        text = "Data at gse12345 and Pxd012345"
        result = resolver.extract_accessions_by_type(text)

        assert "GEO" in result
        assert "GSE12345" in result["GEO"]
        assert "PRIDE" in result
        assert "PXD012345" in result["PRIDE"]


class TestURLGeneration:
    """Test URL generation functionality."""

    def test_get_url_geo(self):
        """Test GEO URL generation."""
        resolver = get_accession_resolver()

        url = resolver.get_url("GSE12345")
        assert url is not None
        assert "ncbi.nlm.nih.gov/geo" in url
        assert "GSE12345" in url

    def test_get_url_pride(self):
        """Test PRIDE URL generation."""
        resolver = get_accession_resolver()

        url = resolver.get_url("PXD012345")
        assert url is not None
        assert "ebi.ac.uk/pride" in url
        assert "PXD012345" in url

    def test_get_url_invalid(self):
        """Test URL generation for invalid accession."""
        resolver = get_accession_resolver()

        assert resolver.get_url("INVALID") is None


class TestNormalization:
    """Test identifier normalization."""

    def test_normalize_lowercase(self):
        """Test normalization of lowercase accessions."""
        resolver = get_accession_resolver()

        assert resolver.normalize_identifier("gse12345") == "GSE12345"
        assert resolver.normalize_identifier("pxd012345") == "PXD012345"
        assert resolver.normalize_identifier("srp123456") == "SRP123456"

    def test_normalize_whitespace(self):
        """Test normalization removes whitespace."""
        resolver = get_accession_resolver()

        assert resolver.normalize_identifier("  GSE12345  ") == "GSE12345"


class TestHelperMethods:
    """Test helper methods."""

    def test_is_geo_identifier(self):
        """Test GEO identifier detection."""
        resolver = get_accession_resolver()

        assert resolver.is_geo_identifier("GSE12345") is True
        assert resolver.is_geo_identifier("GSM123456") is True
        assert resolver.is_geo_identifier("GPL570") is True
        assert resolver.is_geo_identifier("GDS5093") is True
        assert resolver.is_geo_identifier("PXD012345") is False

    def test_is_sra_identifier(self):
        """Test SRA/ENA/DDBJ identifier detection."""
        resolver = get_accession_resolver()

        assert resolver.is_sra_identifier("SRP123456") is True
        assert resolver.is_sra_identifier("ERR123456") is True
        assert resolver.is_sra_identifier("DRR123456") is True
        assert resolver.is_sra_identifier("GSE12345") is False

    def test_is_proteomics_identifier(self):
        """Test proteomics identifier detection."""
        resolver = get_accession_resolver()

        assert resolver.is_proteomics_identifier("PXD012345") is True
        assert resolver.is_proteomics_identifier("MSV000012345") is True
        assert resolver.is_proteomics_identifier("GSE12345") is False

    def test_get_supported_databases(self):
        """Test getting list of supported databases."""
        resolver = get_accession_resolver()

        databases = resolver.get_supported_databases()

        assert len(databases) > 20
        assert "NCBI Gene Expression Omnibus" in databases
        assert "ProteomeXchange/PRIDE" in databases

    def test_get_supported_types(self):
        """Test getting simplified type names."""
        resolver = get_accession_resolver()

        types = resolver.get_supported_types()

        assert "GEO" in types
        assert "SRA" in types
        assert "PRIDE" in types
        assert "ENA" in types


class TestSingleton:
    """Test singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Test that get_accession_resolver returns same instance."""
        resolver1 = get_accession_resolver()
        resolver2 = get_accession_resolver()

        assert resolver1 is resolver2

    def test_reset_creates_new_instance(self):
        """Test that reset creates new instance."""
        resolver1 = get_accession_resolver()
        reset_resolver()
        resolver2 = get_accession_resolver()

        assert resolver1 is not resolver2


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_partial_matches_not_extracted(self):
        """Test that partial matches in words are not extracted."""
        resolver = get_accession_resolver()

        # "GSE" in "GSETEST" should not match (word boundary)
        # This depends on regex pattern design
        text = "The study GSETEST123 is not a real accession"
        result = resolver.extract_accessions_by_type(text)

        # GSE patterns require \d{3,} so GSETEST shouldn't match
        assert "GEO" not in result or "GSETEST123" not in result.get("GEO", [])

    def test_doi_extraction(self):
        """Test DOI extraction from text."""
        resolver = get_accession_resolver()

        text = "See publication at 10.1038/s41586-024-00001-x"
        result = resolver.extract_accessions_by_type(text)

        assert "DOI" in result
        assert "10.1038/S41586-024-00001-X" in result["DOI"]

    def test_arrayexpress_extraction(self):
        """Test ArrayExpress accession extraction."""
        resolver = get_accession_resolver()

        text = "Data available at ArrayExpress E-MTAB-12345"
        result = resolver.extract_accessions_by_type(text)

        assert "ArrayExpress" in result
        assert "E-MTAB-12345" in result["ArrayExpress"]
