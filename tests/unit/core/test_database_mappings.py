"""
Unit tests for database accession mapping and validation.

Tests the DatabaseAccession dataclass, validation functions, and modality-specific
accession filtering for cross-database integration.
"""

import pytest

from lobster.core.schemas.database_mappings import (
    DATABASE_ACCESSION_REGISTRY,
    DatabaseAccession,
    get_accession_url,
    get_accessions_for_modality,
    get_database_summary,
    list_required_accessions,
    validate_accession,
    validate_doi,
    validate_geo_accession,
    validate_ncbi_accession,
)


@pytest.mark.unit
class TestDatabaseAccessionDataclass:
    """Test DatabaseAccession dataclass structure."""

    def test_database_accession_attributes(self):
        """Test all required attributes are present."""
        accession = DATABASE_ACCESSION_REGISTRY["bioproject_accession"]

        assert isinstance(accession, DatabaseAccession)
        assert accession.field_name == "bioproject_accession"
        assert accession.database_name == "NCBI BioProject"
        assert "{accession}" in accession.database_url_template
        assert accession.prefix_pattern == "PRJNA"
        assert accession.full_pattern == r"^PRJNA\d{6,}$"
        assert accession.example == "PRJNA123456"
        assert len(accession.description) > 0
        assert "transcriptomics" in accession.modalities
        assert isinstance(accession.required, bool)


@pytest.mark.unit
class TestAccessionValidation:
    """Test accession validation functions."""

    def test_validate_bioproject_valid(self):
        """Test validation of valid BioProject accessions."""
        assert validate_accession("bioproject_accession", "PRJNA123456")
        assert validate_accession("bioproject_accession", "PRJNA000001")
        assert validate_accession("bioproject_accession", "PRJNA9999999")

    def test_validate_bioproject_invalid(self):
        """Test rejection of invalid BioProject accessions."""
        assert not validate_accession("bioproject_accession", "GSE123456")
        assert not validate_accession("bioproject_accession", "PRJNA12345")  # Too short
        assert not validate_accession("bioproject_accession", "PRJNAabc123")
        assert not validate_accession("bioproject_accession", "")

    def test_validate_biosample_valid(self):
        """Test validation of valid BioSample accessions."""
        assert validate_accession("biosample_accession", "SAMN12345678")
        assert validate_accession("biosample_accession", "SAMN00000001")

    def test_validate_biosample_invalid(self):
        """Test rejection of invalid BioSample accessions."""
        assert not validate_accession("biosample_accession", "SAMN123")  # Too short
        assert not validate_accession("biosample_accession", "SAMNabc12345")

    def test_validate_sra_study_valid(self):
        """Test validation of SRA Study accessions."""
        assert validate_accession("sra_study_accession", "SRP123456")
        assert validate_accession("sra_study_accession", "SRP000001")

    def test_validate_sra_experiment_valid(self):
        """Test validation of SRA Experiment accessions."""
        assert validate_accession("sra_experiment_accession", "SRX123456")
        assert validate_accession("sra_experiment_accession", "SRX999999")

    def test_validate_sra_run_valid(self):
        """Test validation of SRA Run accessions."""
        assert validate_accession("sra_run_accession", "SRR123456")
        assert validate_accession("sra_run_accession", "SRR000001")

    def test_validate_geo_valid(self):
        """Test validation of GEO Series accessions."""
        assert validate_accession("geo_accession", "GSE194247")
        assert validate_accession("geo_accession", "GSE123")
        assert validate_accession("geo_accession", "GSE999999")

    def test_validate_geo_invalid(self):
        """Test rejection of invalid GEO accessions."""
        assert not validate_accession(
            "geo_accession", "GSE12"
        )  # Too short (need 3+ digits)
        assert not validate_accession("geo_accession", "PRJNA123456")
        assert not validate_accession("geo_accession", "GSEabc123")

    def test_validate_pride_valid(self):
        """Test validation of PRIDE accessions."""
        assert validate_accession("pride_accession", "PXD012345")
        assert validate_accession("pride_accession", "PXD000001")
        assert validate_accession("pride_accession", "PXD999999")

    def test_validate_pride_invalid(self):
        """Test rejection of invalid PRIDE accessions."""
        assert not validate_accession("pride_accession", "PXD12345")  # Too short
        assert not validate_accession("pride_accession", "PXD0123456")  # Too long
        assert not validate_accession("pride_accession", "MSV000012345")

    def test_validate_massive_valid(self):
        """Test validation of MassIVE accessions."""
        assert validate_accession("massive_accession", "MSV000012345")
        assert validate_accession("massive_accession", "MSV000000001")

    def test_validate_massive_invalid(self):
        """Test rejection of invalid MassIVE accessions."""
        assert not validate_accession("massive_accession", "MSV12345")  # Too short
        assert not validate_accession("massive_accession", "PXD012345")

    def test_validate_metabolights_valid(self):
        """Test validation of MetaboLights accessions."""
        assert validate_accession("metabolights_accession", "MTBLS1234")
        assert validate_accession("metabolights_accession", "MTBLS1")
        assert validate_accession("metabolights_accession", "MTBLS999999")

    def test_validate_metabolights_invalid(self):
        """Test rejection of invalid MetaboLights accessions."""
        assert not validate_accession("metabolights_accession", "MTBLS")
        assert not validate_accession("metabolights_accession", "MTBLSabc")

    def test_validate_metabolomics_workbench_valid(self):
        """Test validation of Metabolomics Workbench accessions."""
        assert validate_accession("metabolomics_workbench_accession", "ST001234")
        assert validate_accession("metabolomics_workbench_accession", "ST000001")

    def test_validate_metabolomics_workbench_invalid(self):
        """Test rejection of invalid Metabolomics Workbench accessions."""
        assert not validate_accession(
            "metabolomics_workbench_accession", "ST12345"
        )  # Too long
        assert not validate_accession(
            "metabolomics_workbench_accession", "ST123"
        )  # Too short

    def test_validate_mgnify_valid(self):
        """Test validation of MGnify accessions."""
        assert validate_accession("mgnify_accession", "MGYS00001234")
        assert validate_accession("mgnify_accession", "MGYS99999999")

    def test_validate_mgnify_invalid(self):
        """Test rejection of invalid MGnify accessions."""
        assert not validate_accession("mgnify_accession", "MGYS1234")  # Too short
        assert not validate_accession("mgnify_accession", "MGYSabcd1234")

    @pytest.mark.skip(reason="Qiita disabled due to pure numeric ID false positives")
    def test_validate_qiita_valid(self):
        """Test validation of Qiita study IDs."""
        assert validate_accession("qiita_accession", "10317")
        assert validate_accession("qiita_accession", "1")
        assert validate_accession("qiita_accession", "123456")

    @pytest.mark.skip(reason="Qiita disabled due to pure numeric ID false positives")
    def test_validate_qiita_invalid(self):
        """Test rejection of invalid Qiita IDs."""
        assert not validate_accession("qiita_accession", "abc123")
        assert not validate_accession("qiita_accession", "")

    def test_validate_arrayexpress_valid(self):
        """Test validation of ArrayExpress accessions."""
        assert validate_accession("arrayexpress_accession", "E-MTAB-12345")
        assert validate_accession("arrayexpress_accession", "E-GEOD-1")
        assert validate_accession("arrayexpress_accession", "E-MEXP-999999")

    def test_validate_arrayexpress_invalid(self):
        """Test rejection of invalid ArrayExpress accessions."""
        assert not validate_accession(
            "arrayexpress_accession", "E-MT-12345"
        )  # Prefix too short
        assert not validate_accession(
            "arrayexpress_accession", "E-MTABCD-12345"
        )  # Prefix too long
        assert not validate_accession("arrayexpress_accession", "GSE123456")

    def test_validate_doi_valid(self):
        """Test validation of DOI format."""
        assert validate_accession("publication_doi", "10.1038/nature12345")
        assert validate_accession("publication_doi", "10.1234/example.2024.01.001")
        assert validate_accession("publication_doi", "10.12345/abcd")

    def test_validate_doi_invalid(self):
        """Test rejection of invalid DOI format."""
        assert not validate_accession(
            "publication_doi", "10.123/test"
        )  # Registrant too short
        assert not validate_accession("publication_doi", "11.1234/test")  # Wrong prefix
        assert not validate_accession("publication_doi", "10.1234")  # Missing suffix

    def test_validate_unknown_field(self):
        """Test validation returns False for unknown fields."""
        assert not validate_accession("unknown_field", "VALUE123")


@pytest.mark.unit
class TestSpecializedValidators:
    """Test specialized validation functions."""

    def test_validate_ncbi_accession(self):
        """Test NCBI-specific validation helper."""
        assert validate_ncbi_accession("PRJNA123456", "PRJNA")
        assert validate_ncbi_accession("SRR123456", "SRR")
        assert not validate_ncbi_accession("PRJNA123", "PRJNA")  # Too short
        assert not validate_ncbi_accession("GSE123456", "PRJNA")  # Wrong prefix

    def test_validate_doi_helper(self):
        """Test DOI-specific validation helper."""
        assert validate_doi("10.1038/nature12345")
        assert validate_doi("10.12345/test")
        assert not validate_doi("10.123/test")
        assert not validate_doi("11.1234/test")

    def test_validate_geo_accession_helper(self):
        """Test GEO-specific validation helper."""
        assert validate_geo_accession("GSE194247")
        assert validate_geo_accession("GSE123")
        assert not validate_geo_accession("GSE12")
        assert not validate_geo_accession("PRJNA123456")


@pytest.mark.unit
class TestModalityFiltering:
    """Test filtering accessions by modality."""

    def test_get_transcriptomics_accessions(self):
        """Test retrieval of transcriptomics accessions."""
        accessions = get_accessions_for_modality("transcriptomics")

        assert "bioproject_accession" in accessions
        assert "biosample_accession" in accessions
        assert "geo_accession" in accessions
        assert "sra_study_accession" in accessions
        assert "sra_experiment_accession" in accessions
        assert "sra_run_accession" in accessions
        assert "arrayexpress_accession" in accessions
        assert "publication_doi" in accessions

        # Should NOT include proteomics-specific
        assert "pride_accession" not in accessions
        assert "massive_accession" not in accessions

    def test_get_proteomics_accessions(self):
        """Test retrieval of proteomics accessions."""
        accessions = get_accessions_for_modality("proteomics")

        assert "bioproject_accession" in accessions
        assert "biosample_accession" in accessions
        assert "pride_accession" in accessions
        assert "massive_accession" in accessions
        assert "publication_doi" in accessions

        # Should NOT include transcriptomics-specific
        assert "geo_accession" not in accessions
        assert "sra_study_accession" not in accessions

    def test_get_metabolomics_accessions(self):
        """Test retrieval of metabolomics accessions."""
        accessions = get_accessions_for_modality("metabolomics")

        assert "bioproject_accession" in accessions
        assert "biosample_accession" in accessions
        assert "metabolights_accession" in accessions
        assert "metabolomics_workbench_accession" in accessions
        assert "publication_doi" in accessions

    def test_get_metagenomics_accessions(self):
        """Test retrieval of metagenomics accessions."""
        accessions = get_accessions_for_modality("metagenomics")

        assert "bioproject_accession" in accessions
        assert "biosample_accession" in accessions
        assert "sra_study_accession" in accessions
        assert "mgnify_accession" in accessions
        # Qiita disabled due to pure numeric ID false positives
        assert "qiita_accession" not in accessions
        assert "publication_doi" in accessions

    def test_modality_accession_counts(self):
        """Test expected number of accessions per modality."""
        # Counts reflect inclusion of NCBI, ENA, DDBJ, and EGA accessions
        assert len(get_accessions_for_modality("transcriptomics")) >= 8
        assert len(get_accessions_for_modality("proteomics")) >= 5
        assert len(get_accessions_for_modality("metabolomics")) >= 5
        # Metagenomics = 28 (includes NCBI, ENA, DDBJ, EGA, MGnify; Qiita disabled)
        assert len(get_accessions_for_modality("metagenomics")) == 28


@pytest.mark.unit
class TestURLGeneration:
    """Test database URL generation."""

    def test_get_bioproject_url(self):
        """Test BioProject URL generation."""
        url = get_accession_url("bioproject_accession", "PRJNA123456")
        assert url == "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA123456"

    def test_get_geo_url(self):
        """Test GEO URL generation."""
        url = get_accession_url("geo_accession", "GSE194247")
        assert url == "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194247"

    def test_get_pride_url(self):
        """Test PRIDE URL generation."""
        url = get_accession_url("pride_accession", "PXD012345")
        assert url == "https://www.ebi.ac.uk/pride/archive/projects/PXD012345"

    def test_get_metabolights_url(self):
        """Test MetaboLights URL generation."""
        url = get_accession_url("metabolights_accession", "MTBLS1234")
        assert url == "https://www.ebi.ac.uk/metabolights/MTBLS1234"

    def test_get_mgnify_url(self):
        """Test MGnify URL generation."""
        url = get_accession_url("mgnify_accession", "MGYS00001234")
        assert url == "https://www.ebi.ac.uk/metagenomics/studies/MGYS00001234"

    def test_get_doi_url(self):
        """Test DOI URL generation."""
        url = get_accession_url("publication_doi", "10.1038/nature12345")
        assert url == "https://doi.org/10.1038/nature12345"

    def test_get_url_invalid_accession(self):
        """Test URL generation returns None for invalid accession."""
        url = get_accession_url("bioproject_accession", "INVALID")
        assert url is None

    def test_get_url_unknown_field(self):
        """Test URL generation returns None for unknown field."""
        url = get_accession_url("unknown_field", "VALUE123")
        assert url is None


@pytest.mark.unit
class TestRequiredAccessions:
    """Test required accession retrieval."""

    def test_list_required_accessions_all_optional(self):
        """Test that all accessions are currently optional."""
        for modality in [
            "transcriptomics",
            "proteomics",
            "metabolomics",
            "metagenomics",
        ]:
            required = list_required_accessions(modality)
            assert len(required) == 0  # All cross-database accessions are optional

    def test_list_required_accessions_returns_list(self):
        """Test function returns list type."""
        result = list_required_accessions("transcriptomics")
        assert isinstance(result, list)


@pytest.mark.unit
class TestDatabaseSummary:
    """Test database mapping summary statistics."""

    def test_get_database_summary_structure(self):
        """Test summary returns expected structure."""
        summary = get_database_summary()

        assert isinstance(summary, dict)
        assert "transcriptomics" in summary
        assert "proteomics" in summary
        assert "metabolomics" in summary
        assert "metagenomics" in summary

    def test_get_database_summary_counts(self):
        """Test summary counts match expected values."""
        summary = get_database_summary()

        # Counts reflect inclusion of NCBI, ENA, DDBJ, and EGA accessions
        assert summary["transcriptomics"] >= 8
        assert summary["proteomics"] >= 5
        assert summary["metabolomics"] >= 5
        # Metagenomics = 28 (includes NCBI, ENA, DDBJ, EGA, MGnify; Qiita disabled)
        assert summary["metagenomics"] == 28


@pytest.mark.unit
class TestRegistryCompleteness:
    """Test registry structure and completeness."""

    def test_registry_contains_all_categories(self):
        """Test registry includes accessions from all categories."""
        # NCBI
        assert "bioproject_accession" in DATABASE_ACCESSION_REGISTRY
        assert "biosample_accession" in DATABASE_ACCESSION_REGISTRY
        assert "sra_study_accession" in DATABASE_ACCESSION_REGISTRY

        # GEO
        assert "geo_accession" in DATABASE_ACCESSION_REGISTRY

        # Proteomics
        assert "pride_accession" in DATABASE_ACCESSION_REGISTRY
        assert "massive_accession" in DATABASE_ACCESSION_REGISTRY

        # Metabolomics
        assert "metabolights_accession" in DATABASE_ACCESSION_REGISTRY
        assert "metabolomics_workbench_accession" in DATABASE_ACCESSION_REGISTRY

        # Metagenomics
        assert "mgnify_accession" in DATABASE_ACCESSION_REGISTRY
        # Qiita disabled due to pure numeric ID false positives
        assert "qiita_accession" not in DATABASE_ACCESSION_REGISTRY

        # Cross-platform
        assert "arrayexpress_accession" in DATABASE_ACCESSION_REGISTRY
        assert "publication_doi" in DATABASE_ACCESSION_REGISTRY

    def test_registry_total_count(self):
        """Test total number of unique accession types."""
        # 41 = 37 base + 4 KNOWLEDGEBASE_ACCESSIONS (uniprot, ensembl gene/transcript/protein)
        assert len(DATABASE_ACCESSION_REGISTRY) == 41

    def test_all_accessions_have_examples(self):
        """Test all accessions have valid example values."""
        for field_name, accession in DATABASE_ACCESSION_REGISTRY.items():
            assert len(accession.example) > 0
            # Example should validate against pattern
            assert validate_accession(
                field_name, accession.example
            ), f"Example for {field_name} ({accession.example}) does not validate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
