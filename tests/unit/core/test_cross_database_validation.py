"""
Unit tests for cross-database accession validation across all modalities.

Tests the _validate_cross_database_accessions function integration with
transcriptomics, proteomics, metabolomics, and metagenomics schemas.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.metabolomics import (
    MetabolomicsSchema,
)
from lobster.core.schemas.metabolomics import (
    _validate_cross_database_accessions as metabolomics_validate,
)
from lobster.core.schemas.metagenomics import (
    MetagenomicsSchema,
)
from lobster.core.schemas.metagenomics import (
    _validate_cross_database_accessions as metagenomics_validate,
)
from lobster.core.schemas.proteomics import (
    ProteomicsSchema,
)
from lobster.core.schemas.proteomics import (
    _validate_cross_database_accessions as proteomics_validate,
)
from lobster.core.schemas.transcriptomics import (
    TranscriptomicsSchema,
)
from lobster.core.schemas.transcriptomics import (
    _validate_cross_database_accessions as transcriptomics_validate,
)


@pytest.fixture
def basic_adata():
    """Create basic AnnData object for testing."""
    return ad.AnnData(
        X=np.random.randn(10, 20),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
    )


@pytest.mark.unit
class TestTranscriptomicsAccessionValidation:
    """Test cross-database accession validation for transcriptomics."""

    def test_validate_valid_bioproject(self, basic_adata):
        """Test validation with valid BioProject accession."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        # Should have info message with URL
        assert any("Valid" in msg for msg in result.info)
        assert any("PRJNA123456" in msg for msg in result.info)

    def test_validate_invalid_bioproject(self, basic_adata):
        """Test validation with invalid BioProject accession."""
        basic_adata.uns["bioproject_accession"] = "GSE123456"  # Wrong format

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert isinstance(result, ValidationResult)
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings)
        assert "Invalid" in warning_msg
        assert "BioProject" in warning_msg
        assert "GSE123456" in warning_msg

    def test_validate_valid_geo_accession(self, basic_adata):
        """Test validation with valid GEO accession."""
        basic_adata.uns["geo_accession"] = "GSE194247"

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        assert any("Valid" in msg for msg in result.info)
        assert any("GSE194247" in msg for msg in result.info)

    def test_validate_invalid_geo_accession(self, basic_adata):
        """Test validation with invalid GEO accession."""
        basic_adata.uns["geo_accession"] = "PRJNA123456"  # Wrong format

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert result.has_warnings()
        assert "Invalid" in " ".join(result.warnings)

    def test_validate_multiple_accessions(self, basic_adata):
        """Test validation with multiple accessions."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"
        basic_adata.uns["biosample_accession"] = "SAMN12345678"
        basic_adata.uns["geo_accession"] = "GSE194247"

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert not result.has_errors()
        # Should have info messages for all three valid accessions
        assert len([msg for msg in result.info if "Valid" in msg]) == 3

    def test_validate_mixed_valid_invalid(self, basic_adata):
        """Test validation with mix of valid and invalid accessions."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"  # Valid
        basic_adata.uns["geo_accession"] = "INVALID"  # Invalid

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert result.has_warnings()
        # Should have one info (valid) and one warning (invalid)
        assert len([msg for msg in result.info if "Valid" in msg]) == 1
        assert len([msg for msg in result.warnings if "Invalid" in msg]) == 1

    def test_validate_empty_string(self, basic_adata):
        """Test validation with empty string (should be skipped)."""
        basic_adata.uns["bioproject_accession"] = ""

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        # Empty strings should be skipped - no errors or warnings
        assert not result.has_errors()
        assert not result.has_warnings()

    def test_validate_none_value(self, basic_adata):
        """Test validation with None value (should be skipped)."""
        basic_adata.uns["bioproject_accession"] = None

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        # None values should be skipped
        assert not result.has_errors()
        assert not result.has_warnings()

    def test_validator_integration(self, basic_adata):
        """Test that validation is integrated with create_validator."""
        basic_adata.uns["bioproject_accession"] = "INVALID"

        validator = TranscriptomicsSchema.create_validator(schema_type="single_cell")

        # Check that the custom rule is registered
        assert "check_cross_database_accessions" in validator.custom_rules

        # Run validation
        result = validator.validate(basic_adata)

        # Should detect the invalid accession
        assert result.has_warnings()


@pytest.mark.unit
class TestProteomicsAccessionValidation:
    """Test cross-database accession validation for proteomics."""

    def test_validate_valid_pride_accession(self, basic_adata):
        """Test validation with valid PRIDE accession."""
        basic_adata.uns["pride_accession"] = "PXD012345"

        result = proteomics_validate(basic_adata, modality="proteomics")

        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        assert any("Valid" in msg and "PXD012345" in msg for msg in result.info)

    def test_validate_invalid_pride_accession(self, basic_adata):
        """Test validation with invalid PRIDE accession."""
        basic_adata.uns["pride_accession"] = "PXD12345"  # Too short

        result = proteomics_validate(basic_adata, modality="proteomics")

        assert result.has_warnings()
        assert "Invalid" in " ".join(result.warnings)

    def test_validate_valid_massive_accession(self, basic_adata):
        """Test validation with valid MassIVE accession."""
        basic_adata.uns["massive_accession"] = "MSV000012345"

        result = proteomics_validate(basic_adata, modality="proteomics")

        assert not result.has_errors()
        assert any("Valid" in msg and "MSV000012345" in msg for msg in result.info)

    def test_validator_integration(self, basic_adata):
        """Test that validation is integrated with create_validator."""
        basic_adata.uns["pride_accession"] = "INVALID"

        validator = ProteomicsSchema.create_validator(schema_type="mass_spectrometry")

        # Check that the custom rule is registered
        assert "check_cross_database_accessions" in validator.custom_rules

        result = validator.validate(basic_adata)
        assert result.has_warnings()


@pytest.mark.unit
class TestMetabolomicsAccessionValidation:
    """Test cross-database accession validation for metabolomics."""

    def test_validate_valid_metabolights_accession(self, basic_adata):
        """Test validation with valid MetaboLights accession."""
        basic_adata.uns["metabolights_accession"] = "MTBLS1234"

        result = metabolomics_validate(basic_adata, modality="metabolomics")

        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        assert any("Valid" in msg and "MTBLS1234" in msg for msg in result.info)

    def test_validate_invalid_metabolights_accession(self, basic_adata):
        """Test validation with invalid MetaboLights accession."""
        basic_adata.uns["metabolights_accession"] = "MTBLS"  # Missing number

        result = metabolomics_validate(basic_adata, modality="metabolomics")

        assert result.has_warnings()
        assert "Invalid" in " ".join(result.warnings)

    def test_validate_valid_workbench_accession(self, basic_adata):
        """Test validation with valid Metabolomics Workbench accession."""
        basic_adata.uns["metabolomics_workbench_accession"] = "ST001234"

        result = metabolomics_validate(basic_adata, modality="metabolomics")

        assert not result.has_errors()
        assert any("Valid" in msg and "ST001234" in msg for msg in result.info)

    def test_validator_integration(self, basic_adata):
        """Test that validation is integrated with create_validator."""
        basic_adata.uns["metabolights_accession"] = "INVALID"

        validator = MetabolomicsSchema.create_validator()

        # Check that the custom rule is registered
        assert "check_cross_database_accessions" in validator.custom_rules

        result = validator.validate(basic_adata)
        assert result.has_warnings()


@pytest.mark.unit
class TestMetagenomicsAccessionValidation:
    """Test cross-database accession validation for metagenomics."""

    def test_validate_valid_mgnify_accession(self, basic_adata):
        """Test validation with valid MGnify accession."""
        basic_adata.uns["mgnify_accession"] = "MGYS00001234"

        result = metagenomics_validate(basic_adata, modality="metagenomics")

        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        assert any("Valid" in msg and "MGYS00001234" in msg for msg in result.info)

    def test_validate_invalid_mgnify_accession(self, basic_adata):
        """Test validation with invalid MGnify accession."""
        basic_adata.uns["mgnify_accession"] = "MGYS1234"  # Too short

        result = metagenomics_validate(basic_adata, modality="metagenomics")

        assert result.has_warnings()
        assert "Invalid" in " ".join(result.warnings)

    def test_validate_valid_qiita_accession(self, basic_adata):
        """Test validation with valid Qiita accession."""
        basic_adata.uns["qiita_accession"] = "10317"

        result = metagenomics_validate(basic_adata, modality="metagenomics")

        assert not result.has_errors()
        assert any("Valid" in msg and "10317" in msg for msg in result.info)

    def test_validator_integration_16s(self, basic_adata):
        """Test that validation is integrated with create_validator for 16S."""
        basic_adata.uns["mgnify_accession"] = "INVALID"

        validator = MetagenomicsSchema.create_validator(schema_type="16s_amplicon")

        # Check that the custom rule is registered
        assert "check_cross_database_accessions" in validator.custom_rules

        result = validator.validate(basic_adata)
        assert result.has_warnings()

    def test_validator_integration_shotgun(self, basic_adata):
        """Test that validation is integrated with create_validator for shotgun."""
        basic_adata.uns["mgnify_accession"] = "INVALID"

        validator = MetagenomicsSchema.create_validator(schema_type="shotgun")

        # Check that the custom rule is registered
        assert "check_cross_database_accessions" in validator.custom_rules

        result = validator.validate(basic_adata)
        assert result.has_warnings()


@pytest.mark.unit
class TestCrossModalityAccessions:
    """Test accessions that are shared across multiple modalities."""

    def test_bioproject_in_transcriptomics(self, basic_adata):
        """Test BioProject validation in transcriptomics."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"

        result = transcriptomics_validate(basic_adata, modality="transcriptomics")

        assert not result.has_errors()
        assert any("Valid" in msg for msg in result.info)

    def test_bioproject_in_proteomics(self, basic_adata):
        """Test BioProject validation in proteomics."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"

        result = proteomics_validate(basic_adata, modality="proteomics")

        assert not result.has_errors()
        assert any("Valid" in msg for msg in result.info)

    def test_bioproject_in_metabolomics(self, basic_adata):
        """Test BioProject validation in metabolomics."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"

        result = metabolomics_validate(basic_adata, modality="metabolomics")

        assert not result.has_errors()
        assert any("Valid" in msg for msg in result.info)

    def test_bioproject_in_metagenomics(self, basic_adata):
        """Test BioProject validation in metagenomics."""
        basic_adata.uns["bioproject_accession"] = "PRJNA123456"

        result = metagenomics_validate(basic_adata, modality="metagenomics")

        assert not result.has_errors()
        assert any("Valid" in msg for msg in result.info)

    def test_publication_doi_all_modalities(self, basic_adata):
        """Test publication DOI validation across all modalities."""
        doi = "10.1038/nature12345"
        basic_adata.uns["publication_doi"] = doi

        # Test in all modalities
        for modality, validate_func in [
            ("transcriptomics", transcriptomics_validate),
            ("proteomics", proteomics_validate),
            ("metabolomics", metabolomics_validate),
            ("metagenomics", metagenomics_validate),
        ]:
            result = validate_func(basic_adata, modality=modality)
            assert not result.has_errors(), f"Failed for {modality}"
            assert any(
                doi in msg for msg in result.info
            ), f"Missing info for {modality}"


@pytest.mark.unit
class TestModalitySpecificAccessions:
    """Test that modality-specific accessions are only validated in correct modality."""

    def test_geo_only_in_transcriptomics(self, basic_adata):
        """Test that GEO accession is only validated in transcriptomics."""
        basic_adata.uns["geo_accession"] = "GSE194247"

        # Should validate in transcriptomics
        result = transcriptomics_validate(basic_adata, modality="transcriptomics")
        assert any("Valid" in msg for msg in result.info)

        # Should be ignored in proteomics (not in expected accessions)
        result = proteomics_validate(basic_adata, modality="proteomics")
        assert len(result.info) == 0  # No info messages because field not checked

    def test_pride_only_in_proteomics(self, basic_adata):
        """Test that PRIDE accession is only validated in proteomics."""
        basic_adata.uns["pride_accession"] = "PXD012345"

        # Should validate in proteomics
        result = proteomics_validate(basic_adata, modality="proteomics")
        assert any("Valid" in msg for msg in result.info)

        # Should be ignored in transcriptomics
        result = transcriptomics_validate(basic_adata, modality="transcriptomics")
        assert len(result.info) == 0

    def test_metabolights_only_in_metabolomics(self, basic_adata):
        """Test that MetaboLights accession is only validated in metabolomics."""
        basic_adata.uns["metabolights_accession"] = "MTBLS1234"

        # Should validate in metabolomics
        result = metabolomics_validate(basic_adata, modality="metabolomics")
        assert any("Valid" in msg for msg in result.info)

        # Should be ignored in other modalities
        result = transcriptomics_validate(basic_adata, modality="transcriptomics")
        assert len(result.info) == 0

    def test_mgnify_only_in_metagenomics(self, basic_adata):
        """Test that MGnify accession is only validated in metagenomics."""
        basic_adata.uns["mgnify_accession"] = "MGYS00001234"

        # Should validate in metagenomics
        result = metagenomics_validate(basic_adata, modality="metagenomics")
        assert any("Valid" in msg for msg in result.info)

        # Should be ignored in other modalities
        result = transcriptomics_validate(basic_adata, modality="transcriptomics")
        assert len(result.info) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
