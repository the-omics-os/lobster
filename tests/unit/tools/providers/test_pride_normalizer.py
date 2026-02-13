"""
Unit tests for PRIDE API response normalizer.

Tests defensive type handling for inconsistent PRIDE API v2 responses.
"""

import pytest

from lobster.tools.providers.pride_normalizer import PRIDENormalizer


class TestOrganismNormalization:
    """Test organism field normalization across all input types."""

    def test_list_of_dicts(self):
        """Test already-normalized list of dicts (API v3 format)."""
        input_data = [{"name": "Homo sapiens"}, {"name": "Mus musculus"}]
        result = PRIDENormalizer.normalize_organisms(input_data)
        assert result == input_data
        assert len(result) == 2

    def test_list_of_strings(self):
        """Test list of strings → list of dicts (API v2 format)."""
        input_data = ["Homo sapiens", "Mus musculus"]
        result = PRIDENormalizer.normalize_organisms(input_data)
        expected = [{"name": "Homo sapiens"}, {"name": "Mus musculus"}]
        assert result == expected

    def test_single_dict(self):
        """Test single dict → list of dicts."""
        input_data = {"name": "Homo sapiens"}
        result = PRIDENormalizer.normalize_organisms(input_data)
        assert result == [{"name": "Homo sapiens"}]
        assert len(result) == 1

    def test_single_string(self):
        """Test single string → list of dicts (legacy format)."""
        input_data = "Homo sapiens"
        result = PRIDENormalizer.normalize_organisms(input_data)
        assert result == [{"name": "Homo sapiens"}]

    def test_none(self):
        """Test None → empty list."""
        result = PRIDENormalizer.normalize_organisms(None)
        assert result == []

    def test_empty_list(self):
        """Test empty list → empty list."""
        result = PRIDENormalizer.normalize_organisms([])
        assert result == []

    def test_mixed_list(self):
        """Test list with mixed dict/string types."""
        input_data = [{"name": "Homo sapiens"}, "Mus musculus", {"name": "Rattus"}]
        result = PRIDENormalizer.normalize_organisms(input_data)
        expected = [
            {"name": "Homo sapiens"},
            {"name": "Mus musculus"},
            {"name": "Rattus"},
        ]
        assert result == expected

    def test_safe_get_organism_name(self):
        """Test safe name extraction helper."""
        assert PRIDENormalizer.safe_get_organism_name({"name": "Human"}) == "Human"
        assert PRIDENormalizer.safe_get_organism_name("Mouse") == "Mouse"
        assert PRIDENormalizer.safe_get_organism_name({}) == "Unknown"
        assert PRIDENormalizer.safe_get_organism_name(None) == "Unknown"


class TestReferencesNormalization:
    """Test references field normalization."""

    def test_list_of_dicts(self):
        """Test already-normalized list of reference dicts."""
        input_data = [
            {"pubmedId": "12345", "doi": "10.1234/abc"},
            {"doi": "10.5678/def"},
        ]
        result = PRIDENormalizer.normalize_references(input_data)
        assert result == input_data

    def test_list_of_strings(self):
        """Test list of string identifiers."""
        input_data = ["PMID:12345", "DOI:10.1234/abc"]
        result = PRIDENormalizer.normalize_references(input_data)
        expected = [{"id": "PMID:12345"}, {"id": "DOI:10.1234/abc"}]
        assert result == expected

    def test_single_dict(self):
        """Test single reference dict."""
        input_data = {"pubmedId": "12345", "doi": "10.1234/abc"}
        result = PRIDENormalizer.normalize_references(input_data)
        assert result == [input_data]

    def test_single_string(self):
        """Test single string identifier."""
        input_data = "PMID:12345"
        result = PRIDENormalizer.normalize_references(input_data)
        assert result == [{"id": "PMID:12345"}]

    def test_none(self):
        """Test None → empty list."""
        result = PRIDENormalizer.normalize_references(None)
        assert result == []

    def test_empty_list(self):
        """Test empty list → empty list."""
        result = PRIDENormalizer.normalize_references([])
        assert result == []

    def test_mixed_list(self):
        """Test list with mixed dict/string references."""
        input_data = [
            {"pubmedId": "12345"},
            "DOI:10.1234/abc",
            {"doi": "10.5678/def", "referenceLine": "Smith et al. 2023"},
        ]
        result = PRIDENormalizer.normalize_references(input_data)
        assert len(result) == 3
        assert result[0] == {"pubmedId": "12345"}
        assert result[1] == {"id": "DOI:10.1234/abc"}
        assert result[2]["doi"] == "10.5678/def"


class TestPeopleNormalization:
    """Test submitters/labPIs field normalization."""

    def test_list_of_dicts(self):
        """Test already-normalized list of people dicts."""
        input_data = [
            {"firstName": "John", "lastName": "Doe"},
            {"firstName": "Jane", "lastName": "Smith"},
        ]
        result = PRIDENormalizer.normalize_people(input_data)
        assert result == input_data

    def test_list_of_strings(self):
        """Test list of full names as strings."""
        input_data = ["John Doe", "Jane Smith"]
        result = PRIDENormalizer.normalize_people(input_data)
        expected = [{"fullName": "John Doe"}, {"fullName": "Jane Smith"}]
        assert result == expected

    def test_single_dict(self):
        """Test single person dict."""
        input_data = {"firstName": "John", "lastName": "Doe"}
        result = PRIDENormalizer.normalize_people(input_data)
        assert result == [input_data]

    def test_single_string(self):
        """Test single full name string."""
        input_data = "John Doe"
        result = PRIDENormalizer.normalize_people(input_data)
        assert result == [{"fullName": "John Doe"}]

    def test_none(self):
        """Test None → empty list."""
        result = PRIDENormalizer.normalize_people(None)
        assert result == []

    def test_empty_list(self):
        """Test empty list → empty list."""
        result = PRIDENormalizer.normalize_people([])
        assert result == []

    def test_safe_get_person_name_with_dict(self):
        """Test safe name extraction from dict with firstName/lastName."""
        person = {"firstName": "John", "lastName": "Doe"}
        result = PRIDENormalizer.safe_get_person_name(person)
        assert result == "John Doe"

    def test_safe_get_person_name_with_fullname(self):
        """Test safe name extraction from dict with fullName."""
        person = {"fullName": "Jane Smith"}
        result = PRIDENormalizer.safe_get_person_name(person)
        assert result == "Jane Smith"

    def test_safe_get_person_name_with_string(self):
        """Test safe name extraction from string."""
        result = PRIDENormalizer.safe_get_person_name("Bob Jones")
        assert result == "Bob Jones"

    def test_safe_get_person_name_with_partial_data(self):
        """Test name extraction with only firstName."""
        person = {"firstName": "John"}
        result = PRIDENormalizer.safe_get_person_name(person)
        assert result == "John"


class TestFileLocationsNormalization:
    """Test publicFileLocations field normalization."""

    def test_list_of_dicts(self):
        """Test already-normalized list of location dicts."""
        input_data = [
            {"name": "FTP Protocol", "value": "ftp://example.com/file1.raw"},
            {"name": "S3 Protocol", "value": "s3://bucket/file1.raw"},
        ]
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == input_data

    def test_list_of_strings(self):
        """Test list of URL strings → list of dicts."""
        input_data = ["ftp://example.com/file.raw", "https://example.com/file.mzML"]
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert len(result) == 2
        assert result[0]["name"] == "FTP Protocol"
        assert result[0]["value"] == "ftp://example.com/file.raw"
        assert result[1]["name"] == "HTTP Protocol"
        assert result[1]["value"] == "https://example.com/file.mzML"

    def test_single_dict(self):
        """Test single location dict."""
        input_data = {"name": "FTP Protocol", "value": "ftp://example.com"}
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == [input_data]

    def test_single_string_ftp(self):
        """Test single FTP URL string."""
        input_data = "ftp://example.com/data.raw"
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == [{"name": "FTP Protocol", "value": input_data}]

    def test_single_string_http(self):
        """Test single HTTP URL string."""
        input_data = "https://example.com/data.mzML"
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == [{"name": "HTTP Protocol", "value": input_data}]

    def test_single_string_s3(self):
        """Test single S3 URL string."""
        input_data = "s3://bucket/data.raw"
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == [{"name": "S3 Protocol", "value": input_data}]

    def test_none(self):
        """Test None → empty list."""
        result = PRIDENormalizer.normalize_file_locations(None)
        assert result == []

    def test_empty_list(self):
        """Test empty list → empty list."""
        result = PRIDENormalizer.normalize_file_locations([])
        assert result == []

    def test_extract_ftp_url_with_multiple_protocols(self):
        """Test FTP URL extraction with multiple protocols."""
        locations = [
            {"name": "S3 Protocol", "value": "s3://bucket/file.raw"},
            {"name": "FTP Protocol", "value": "ftp://example.com/file.raw"},
            {"name": "HTTP Protocol", "value": "https://example.com/file.raw"},
        ]
        result = PRIDENormalizer.extract_ftp_url(locations)
        # Should prioritize FTP over S3/HTTP
        assert result == "ftp://example.com/file.raw"

    def test_extract_ftp_url_with_only_http(self):
        """Test FTP URL extraction falls back to HTTP."""
        locations = [{"name": "HTTP Protocol", "value": "https://example.com/file.raw"}]
        result = PRIDENormalizer.extract_ftp_url(locations)
        assert result == "https://example.com/file.raw"

    def test_extract_ftp_url_from_empty_list(self):
        """Test FTP URL extraction from empty list."""
        result = PRIDENormalizer.extract_ftp_url([])
        assert result is None


class TestProjectNormalization:
    """Test full project normalization."""

    def test_normalize_project_with_all_fields(self):
        """Test normalization of project with all problematic fields."""
        raw_project = {
            "accession": "PXD012345",
            "title": "Test Project",
            "organisms": "Homo sapiens",  # String instead of list
            "references": [{"pubmedId": "12345"}, "DOI:10.1234/abc"],  # Mixed types
            "submitters": [
                {"firstName": "John", "lastName": "Doe"},
                "Jane Smith",
            ],  # Mixed
            "labPIs": "Bob Jones",  # String instead of list
            "publicationDate": "2023-01-15",
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        # Check organisms normalized
        assert result["organisms"] == [{"name": "Homo sapiens"}]

        # Check references normalized
        assert len(result["references"]) == 2
        assert result["references"][0] == {"pubmedId": "12345"}
        assert result["references"][1] == {"id": "DOI:10.1234/abc"}

        # Check submitters normalized
        assert len(result["submitters"]) == 2
        assert result["submitters"][0] == {"firstName": "John", "lastName": "Doe"}
        assert result["submitters"][1] == {"fullName": "Jane Smith"}

        # Check labPIs normalized
        assert result["labPIs"] == [{"fullName": "Bob Jones"}]

        # Check other fields preserved
        assert result["accession"] == "PXD012345"
        assert result["title"] == "Test Project"
        assert result["publicationDate"] == "2023-01-15"

    def test_normalize_project_with_missing_fields(self):
        """Test normalization when problematic fields are absent."""
        raw_project = {
            "accession": "PXD012345",
            "title": "Test Project",
            # No organisms, references, submitters, labPIs
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        # Should preserve all original fields
        assert result["accession"] == "PXD012345"
        assert result["title"] == "Test Project"

        # Should not add missing fields
        assert "organisms" not in result
        assert "references" not in result

    def test_normalize_project_with_none_values(self):
        """Test normalization when fields are explicitly None."""
        raw_project = {
            "accession": "PXD012345",
            "organisms": None,
            "references": None,
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        # Should normalize None to empty lists
        assert result["organisms"] == []
        assert result["references"] == []

    def test_normalize_project_with_invalid_input(self):
        """Test normalization with non-dict input."""
        result = PRIDENormalizer.normalize_project("not a dict")
        assert result == {}

        result = PRIDENormalizer.normalize_project(None)
        assert result == {}

        result = PRIDENormalizer.normalize_project([])
        assert result == {}


class TestSearchResultsNormalization:
    """Test bulk normalization of search results."""

    def test_normalize_search_results_with_multiple_projects(self):
        """Test normalizing list of projects."""
        raw_results = [
            {"accession": "PXD001", "organisms": "Human"},
            {"accession": "PXD002", "organisms": ["Mouse", "Rat"]},
            {"accession": "PXD003", "organisms": [{"name": "Yeast"}]},
        ]

        result = PRIDENormalizer.normalize_search_results(raw_results)

        assert len(result) == 3
        # All organisms should be List[Dict]
        assert result[0]["organisms"] == [{"name": "Human"}]
        assert result[1]["organisms"] == [{"name": "Mouse"}, {"name": "Rat"}]
        assert result[2]["organisms"] == [{"name": "Yeast"}]

    def test_normalize_search_results_with_empty_list(self):
        """Test normalizing empty search results."""
        result = PRIDENormalizer.normalize_search_results([])
        assert result == []

    def test_normalize_search_results_with_invalid_input(self):
        """Test normalizing non-list input."""
        result = PRIDENormalizer.normalize_search_results("not a list")
        assert result == []

        result = PRIDENormalizer.normalize_search_results(None)
        assert result == []

    def test_normalize_search_results_filters_non_dicts(self):
        """Test that non-dict items are filtered out."""
        raw_results = [
            {"accession": "PXD001", "organisms": "Human"},
            "invalid entry",
            None,
            {"accession": "PXD002", "organisms": "Mouse"},
        ]

        result = PRIDENormalizer.normalize_search_results(raw_results)

        # Should only have 2 valid projects
        assert len(result) == 2
        assert result[0]["accession"] == "PXD001"
        assert result[1]["accession"] == "PXD002"


class TestFileMetadataNormalization:
    """Test file metadata normalization."""

    def test_normalize_file_metadata_with_locations(self):
        """Test normalizing file list with publicFileLocations."""
        raw_files = [
            {
                "fileName": "data.raw",
                "publicFileLocations": "ftp://example.com/data.raw",
            },
            {
                "fileName": "results.txt",
                "publicFileLocations": [
                    {"name": "FTP Protocol", "value": "ftp://example.com/results.txt"}
                ],
            },
        ]

        result = PRIDENormalizer.normalize_file_metadata(raw_files)

        assert len(result) == 2
        # First file: string URL → normalized
        assert result[0]["publicFileLocations"] == [
            {"name": "FTP Protocol", "value": "ftp://example.com/data.raw"}
        ]
        # Second file: already normalized
        assert result[1]["publicFileLocations"] == [
            {"name": "FTP Protocol", "value": "ftp://example.com/results.txt"}
        ]

    def test_normalize_file_metadata_without_locations(self):
        """Test normalizing files without publicFileLocations field."""
        raw_files = [
            {"fileName": "data.raw", "fileSizeBytes": 1024},
        ]

        result = PRIDENormalizer.normalize_file_metadata(raw_files)

        assert len(result) == 1
        assert result[0]["fileName"] == "data.raw"
        # publicFileLocations field should not be added
        assert "publicFileLocations" not in result[0]

    def test_normalize_file_metadata_with_empty_list(self):
        """Test normalizing empty file list."""
        result = PRIDENormalizer.normalize_file_metadata([])
        assert result == []

    def test_normalize_file_metadata_with_invalid_input(self):
        """Test normalizing non-list input."""
        result = PRIDENormalizer.normalize_file_metadata("not a list")
        assert result == []

        result = PRIDENormalizer.normalize_file_metadata(None)
        assert result == []

    def test_normalize_file_metadata_filters_non_dicts(self):
        """Test that non-dict file entries are filtered out."""
        raw_files = [
            {"fileName": "valid.raw"},
            "invalid entry",
            None,
            {"fileName": "valid2.mzML"},
        ]

        result = PRIDENormalizer.normalize_file_metadata(raw_files)

        # Should only have 2 valid files
        assert len(result) == 2
        assert result[0]["fileName"] == "valid.raw"
        assert result[1]["fileName"] == "valid2.mzML"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_organism_with_numeric_type(self):
        """Test organism field with unexpected numeric type."""
        result = PRIDENormalizer.normalize_organisms(12345)
        assert result == []  # Should return empty list with warning log

    def test_references_with_boolean(self):
        """Test references field with boolean."""
        result = PRIDENormalizer.normalize_references(True)
        assert result == []

    def test_people_with_empty_string(self):
        """Test people field with empty string."""
        result = PRIDENormalizer.normalize_people("")
        # Empty string becomes list with empty fullName (could be filtered)
        assert result == [{"fullName": ""}]

    def test_file_locations_with_malformed_url(self):
        """Test file locations with non-standard URL."""
        input_data = "not-a-url"
        result = PRIDENormalizer.normalize_file_locations(input_data)
        assert result == [{"name": "Unknown Protocol", "value": "not-a-url"}]

    def test_deeply_nested_mixed_types(self):
        """Test project with deeply nested mixed types."""
        raw_project = {
            "accession": "PXD012345",
            "organisms": [
                {"name": "Human"},  # Dict
                "Mouse",  # String
                {"name": "Rat", "taxonomy": "10116"},  # Dict with extra field
            ],
            "references": [
                "PMID:12345",  # String
                {"pubmedId": "67890", "doi": "10.1234/abc"},  # Full dict
            ],
            "submitters": "John Doe",  # Single string
            "labPIs": [
                {"firstName": "Jane", "lastName": "Smith"},
                "Bob Jones",
            ],
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        # All fields should be normalized to List[Dict]
        assert all(isinstance(org, dict) for org in result["organisms"])
        assert all(isinstance(ref, dict) for ref in result["references"])
        assert all(isinstance(sub, dict) for sub in result["submitters"])
        assert all(isinstance(pi, dict) for pi in result["labPIs"])

        # Check specific values preserved
        assert result["organisms"][0]["name"] == "Human"
        assert result["organisms"][1]["name"] == "Mouse"
        assert result["organisms"][2]["taxonomy"] == "10116"  # Extra field preserved


class TestHelperMethods:
    """Test helper extraction methods."""

    def test_extract_ftp_url_priority_order(self):
        """Test FTP URL extraction respects protocol priority."""
        locations = [
            {"name": "S3 Protocol", "value": "s3://bucket/file.raw"},
            {"name": "HTTP Protocol", "value": "https://example.com/file.raw"},
            {"name": "FTP Protocol", "value": "ftp://example.com/file.raw"},
            {"name": "Aspera Protocol", "value": "aspera://example.com/file.raw"},
        ]

        result = PRIDENormalizer.extract_ftp_url(locations)
        # Should prioritize FTP over others
        assert result == "ftp://example.com/file.raw"

    def test_extract_ftp_url_fallback(self):
        """Test FTP URL extraction falls back to first available."""
        locations = [
            {"name": "Custom Protocol", "value": "custom://example.com"},
            {"name": "Unknown", "value": "https://example.com/backup.raw"},
        ]

        result = PRIDENormalizer.extract_ftp_url(locations)
        # Should fall back to first available URL
        assert result == "custom://example.com"

    def test_extract_ftp_url_with_no_value(self):
        """Test FTP URL extraction with missing 'value' field."""
        locations = [{"name": "FTP Protocol"}]  # No value field

        result = PRIDENormalizer.extract_ftp_url(locations)
        assert result is None

    def test_extract_ftp_url_with_invalid_input(self):
        """Test FTP URL extraction with non-list input."""
        result = PRIDENormalizer.extract_ftp_url("not a list")
        assert result is None

        result = PRIDENormalizer.extract_ftp_url(None)
        assert result is None


class TestRealWorldScenarios:
    """Test scenarios based on actual PRIDE API responses."""

    def test_legacy_dataset_format(self):
        """Test format from pre-2015 PRIDE dataset."""
        # Simulates old PRIDE API format with minimal metadata
        raw_project = {
            "accession": "PXD000001",
            "title": "Legacy Dataset",
            "organisms": "Human",  # Single string
            "submitters": "John Doe",  # Single string
            "references": "PMID:11111111",  # Single string
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        assert result["organisms"] == [{"name": "Human"}]
        assert result["submitters"] == [{"fullName": "John Doe"}]
        assert result["references"] == [{"id": "PMID:11111111"}]

    def test_modern_dataset_format(self):
        """Test format from recent PRIDE dataset (2020+)."""
        raw_project = {
            "accession": "PXD012345",
            "title": "Modern Dataset",
            "organisms": [{"name": "Homo sapiens", "taxonomy": "9606"}],
            "references": [
                {
                    "pubmedId": "12345678",
                    "doi": "10.1038/s41586-023-12345-1",
                    "referenceLine": "Smith J et al. Nature 2023",
                }
            ],
            "submitters": [{"firstName": "John", "lastName": "Doe"}],
            "labPIs": [{"firstName": "Jane", "lastName": "Smith"}],
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        # Modern format should pass through unchanged
        assert result["organisms"] == raw_project["organisms"]
        assert result["references"] == raw_project["references"]
        assert result["submitters"] == raw_project["submitters"]
        assert result["labPIs"] == raw_project["labPIs"]

    def test_partial_metadata_project(self):
        """Test project with sparse metadata (common in private datasets)."""
        raw_project = {
            "accession": "PXD012345",
            "title": "Minimal Metadata",
            "organisms": [],  # Empty list
            "references": None,  # Explicitly None
            # submitters and labPIs missing entirely
        }

        result = PRIDENormalizer.normalize_project(raw_project)

        assert result["organisms"] == []
        assert result["references"] == []
        assert "submitters" not in result  # Should not be added
        assert "labPIs" not in result


class TestIntegrationPatterns:
    """Test usage patterns for pride_provider.py integration."""

    def test_search_results_integration_pattern(self):
        """Test pattern for integrating into search_publications()."""
        # Simulate PRIDE API search response
        api_response = {
            "_embedded": {
                "projects": [
                    {"accession": "PXD001", "organisms": "Human"},
                    {"accession": "PXD002", "organisms": ["Mouse"]},
                ]
            },
            "page": {"totalElements": 2},
        }

        # Integration pattern
        raw_projects = api_response.get("_embedded", {}).get("projects", [])
        normalized_projects = PRIDENormalizer.normalize_search_results(raw_projects)

        # Verify all organisms are List[Dict]
        for proj in normalized_projects:
            assert isinstance(proj["organisms"], list)
            if proj["organisms"]:
                assert isinstance(proj["organisms"][0], dict)

    def test_file_listing_integration_pattern(self):
        """Test pattern for integrating into get_project_files()."""
        # Simulate PRIDE API file listing response
        api_response = {
            "_embedded": {
                "files": [
                    {
                        "fileName": "data.raw",
                        "publicFileLocations": [
                            {"name": "FTP Protocol", "value": "ftp://..."}
                        ],
                    },
                    {
                        "fileName": "results.txt",
                        "publicFileLocations": "ftp://...",  # String instead of list
                    },
                ]
            }
        }

        # Integration pattern
        raw_files = api_response.get("_embedded", {}).get("files", [])
        normalized_files = PRIDENormalizer.normalize_file_metadata(raw_files)

        # Extract FTP URLs safely
        for file_dict in normalized_files:
            locations = file_dict.get("publicFileLocations", [])
            ftp_url = PRIDENormalizer.extract_ftp_url(locations)
            assert ftp_url is not None
            assert ftp_url.startswith("ftp://")


# Run tests with: pytest tests/unit/tools/providers/test_pride_normalizer.py -v
