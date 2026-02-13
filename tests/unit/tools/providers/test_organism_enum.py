"""
Unit tests for organism enum module.

Tests the centralized organism name mapping for NCBI database queries.
"""

import pytest

from lobster.tools.providers.organism_enum import (
    OrganismEnum,
    get_scientific_name,
    list_organisms,
    list_organisms_with_scientific,
    to_sci_name,
    validate_organism,
)


class TestOrganismEnum:
    """Test OrganismEnum enumeration."""

    def test_enum_has_expected_organisms(self):
        """Test that enum contains key model organisms."""
        expected = [
            "HUMAN",
            "MOUSE",
            "RAT",
            "ZEBRAFISH",
            "FRUIT_FLY",
            "C_ELEGANS",
            "YEAST",
            "E_COLI",
        ]
        for organism in expected:
            assert organism in OrganismEnum.__members__

    def test_enum_values_are_scientific_names(self):
        """Test that enum values are proper scientific names."""
        assert OrganismEnum.HUMAN.value == "Homo sapiens"
        assert OrganismEnum.MOUSE.value == "Mus musculus"
        assert OrganismEnum.E_COLI.value == "Escherichia coli"
        assert (
            OrganismEnum.SARS_COV_2.value
            == "Severe acute respiratory syndrome coronavirus 2"
        )

    def test_enum_has_minimum_45_organisms(self):
        """Test that enum has at least 45 organisms (from SRAgent research)."""
        assert len(OrganismEnum) >= 45


class TestToSciName:
    """Test to_sci_name() function."""

    def test_converts_human_to_quoted_scientific_name(self):
        """Test conversion of 'human' to quoted NCBI format."""
        result = to_sci_name("human")
        assert result == '"Homo sapiens"'

    def test_converts_mouse_to_quoted_scientific_name(self):
        """Test conversion of 'mouse' to quoted NCBI format."""
        result = to_sci_name("mouse")
        assert result == '"Mus musculus"'

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        assert to_sci_name("human") == to_sci_name("HUMAN")
        assert to_sci_name("human") == to_sci_name("Human")
        assert to_sci_name("mouse") == to_sci_name("MOUSE")

    def test_handles_spaces_in_input(self):
        """Test that spaces are converted to underscores."""
        result = to_sci_name("fruit fly")
        assert result == '"Drosophila melanogaster"'

        result = to_sci_name("c elegans")
        assert result == '"Caenorhabditis elegans"'

    def test_handles_underscores_in_input(self):
        """Test that underscores are preserved."""
        result = to_sci_name("fruit_fly")
        assert result == '"Drosophila melanogaster"'

        result = to_sci_name("c_elegans")
        assert result == '"Caenorhabditis elegans"'

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        result = to_sci_name("  human  ")
        assert result == '"Homo sapiens"'

    def test_raises_valueerror_for_unknown_organism(self):
        """Test that unknown organism raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            to_sci_name("unknown_organism")

        error_msg = str(exc_info.value)
        assert "Organism 'unknown_organism' not found" in error_msg
        assert "Available organisms:" in error_msg
        assert "human" in error_msg  # Should suggest alternatives

    def test_bacteria_conversion(self):
        """Test conversion of bacterial organisms."""
        assert to_sci_name("e coli") == '"Escherichia coli"'
        assert to_sci_name("salmonella") == '"Salmonella enterica"'

    def test_virus_conversion(self):
        """Test conversion of viral organisms."""
        assert (
            to_sci_name("sars cov 2")
            == '"Severe acute respiratory syndrome coronavirus 2"'
        )
        assert to_sci_name("hiv") == '"Human immunodeficiency virus 1"'

    def test_plant_conversion(self):
        """Test conversion of plant organisms."""
        assert to_sci_name("arabidopsis") == '"Arabidopsis thaliana"'
        assert to_sci_name("rice") == '"Oryza sativa"'

    def test_fungal_conversion(self):
        """Test conversion of fungal organisms."""
        assert to_sci_name("yeast") == '"Saccharomyces cerevisiae"'
        assert to_sci_name("fission yeast") == '"Schizosaccharomyces pombe"'


class TestValidateOrganism:
    """Test validate_organism() function."""

    def test_returns_true_for_valid_organisms(self):
        """Test that valid organisms return True."""
        assert validate_organism("human") is True
        assert validate_organism("mouse") is True
        assert validate_organism("fruit fly") is True

    def test_returns_false_for_invalid_organisms(self):
        """Test that invalid organisms return False."""
        assert validate_organism("unknown") is False
        assert validate_organism("fake_organism") is False
        assert validate_organism("") is False

    def test_case_insensitive_validation(self):
        """Test that validation is case-insensitive."""
        assert validate_organism("human") is True
        assert validate_organism("HUMAN") is True
        assert validate_organism("Human") is True

    def test_handles_spaces_and_underscores(self):
        """Test that spaces and underscores are handled."""
        assert validate_organism("fruit fly") is True
        assert validate_organism("fruit_fly") is True
        assert validate_organism("c elegans") is True
        assert validate_organism("c_elegans") is True

    def test_no_exception_on_invalid(self):
        """Test that no exception is raised for invalid organisms."""
        # Should return False, not raise exception
        result = validate_organism("this_is_not_an_organism")
        assert result is False


class TestGetScientificName:
    """Test get_scientific_name() function."""

    def test_returns_unquoted_scientific_name(self):
        """Test that scientific names are returned without quotes."""
        assert get_scientific_name("human") == "Homo sapiens"
        assert get_scientific_name("mouse") == "Mus musculus"

    def test_returns_none_for_invalid_organism(self):
        """Test that None is returned for invalid organisms."""
        assert get_scientific_name("unknown") is None
        assert get_scientific_name("fake_organism") is None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert get_scientific_name("human") == get_scientific_name("HUMAN")
        assert get_scientific_name("mouse") == get_scientific_name("Mouse")

    def test_handles_spaces_and_underscores(self):
        """Test space and underscore handling."""
        assert get_scientific_name("fruit fly") == "Drosophila melanogaster"
        assert get_scientific_name("fruit_fly") == "Drosophila melanogaster"


class TestListOrganisms:
    """Test list_organisms() function."""

    def test_returns_list_of_common_names(self):
        """Test that list of common names is returned."""
        organisms = list_organisms()
        assert isinstance(organisms, list)
        assert len(organisms) >= 45

    def test_names_are_lowercase_with_spaces(self):
        """Test that names are lowercase with spaces."""
        organisms = list_organisms()
        assert "human" in organisms
        assert "mouse" in organisms
        assert "fruit fly" in organisms
        assert "c elegans" in organisms

    def test_no_uppercase_or_underscores_in_output(self):
        """Test that output names don't contain uppercase or underscores."""
        organisms = list_organisms()
        for organism in organisms:
            assert organism == organism.lower()
            assert "_" not in organism


class TestListOrganismsWithScientific:
    """Test list_organisms_with_scientific() function."""

    def test_returns_dict_mapping(self):
        """Test that dict mapping is returned."""
        mapping = list_organisms_with_scientific()
        assert isinstance(mapping, dict)
        assert len(mapping) >= 45

    def test_mapping_has_correct_format(self):
        """Test that mapping has common name → scientific name format."""
        mapping = list_organisms_with_scientific()
        assert mapping["human"] == "Homo sapiens"
        assert mapping["mouse"] == "Mus musculus"
        assert mapping["fruit fly"] == "Drosophila melanogaster"

    def test_common_names_are_lowercase_with_spaces(self):
        """Test that common names (keys) are lowercase with spaces."""
        mapping = list_organisms_with_scientific()
        for common_name in mapping.keys():
            assert common_name == common_name.lower()
            assert "_" not in common_name

    def test_scientific_names_are_unquoted(self):
        """Test that scientific names (values) are unquoted."""
        mapping = list_organisms_with_scientific()
        for scientific_name in mapping.values():
            assert not scientific_name.startswith('"')
            assert not scientific_name.endswith('"')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_organism(self):
        """Test handling of empty string."""
        assert validate_organism("") is False
        assert get_scientific_name("") is None

        with pytest.raises(ValueError):
            to_sci_name("")

    def test_whitespace_only_organism(self):
        """Test handling of whitespace-only strings."""
        assert validate_organism("   ") is False
        assert get_scientific_name("   ") is None

        with pytest.raises(ValueError):
            to_sci_name("   ")

    def test_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        # Should treat as single space
        result = to_sci_name("fruit  fly")  # Double space
        assert result == '"Drosophila melanogaster"'

    def test_mixed_case_with_spaces(self):
        """Test mixed case with spaces."""
        result = to_sci_name("Fruit Fly")
        assert result == '"Drosophila melanogaster"'

        result = to_sci_name("C Elegans")
        assert result == '"Caenorhabditis elegans"'


class TestIntegration:
    """Integration tests using multiple functions together."""

    def test_validate_before_convert(self):
        """Test typical pattern: validate before converting."""
        organism = "human"
        if validate_organism(organism):
            result = to_sci_name(organism)
            assert result == '"Homo sapiens"'

    def test_get_all_organisms_and_validate(self):
        """Test that all listed organisms are valid."""
        organisms = list_organisms()
        for organism in organisms:
            assert validate_organism(organism) is True

    def test_get_all_organisms_and_convert(self):
        """Test that all listed organisms can be converted."""
        organisms = list_organisms()
        for organism in organisms:
            result = to_sci_name(organism)
            assert result.startswith('"')
            assert result.endswith('"')
            assert len(result) > 2  # More than just quotes

    def test_mapping_consistency(self):
        """Test that mapping is consistent with to_sci_name()."""
        mapping = list_organisms_with_scientific()
        for common_name, scientific_name in mapping.items():
            # to_sci_name adds quotes
            result = to_sci_name(common_name)
            assert result == f'"{scientific_name}"'
