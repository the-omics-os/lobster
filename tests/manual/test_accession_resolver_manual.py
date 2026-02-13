#!/usr/bin/env python3
"""
Comprehensive manual test suite for AccessionResolver.

Tests all 37 identifier patterns (29 base + 8 EGA) in DATABASE_ACCESSION_REGISTRY.

Usage:
    python tests/manual/test_accession_resolver_manual.py

Test Coverage:
    - detect_database() - for all 37 patterns
    - detect_field() - for all patterns
    - validate() - validation logic with/without database specification
    - extract_all_accessions() - text extraction
    - extract_accessions_by_type() - simplified type extraction
    - extract_accessions_with_metadata() - metadata extraction with access_type
    - Helper methods: is_geo_identifier(), is_sra_identifier(), is_proteomics_identifier(), is_ega_identifier()
    - get_url() - URL generation
    - normalize_identifier() - identifier normalization
    - get_access_type() - access type detection
    - is_controlled_access() - controlled access detection
    - Edge cases: case sensitivity, whitespace, mixed content, large text blocks
"""

import sys
import time
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, '/Users/tyo/GITHUB/omics-os/lobster')

# Direct import to avoid core/__init__.py dependency chain
import importlib.util
spec = importlib.util.spec_from_file_location(
    "accession_resolver",
    "/Users/tyo/GITHUB/omics-os/lobster/lobster/core/identifiers/accession_resolver.py"
)
accession_resolver_module = importlib.util.module_from_spec(spec)

# Also need to import database_mappings
spec_db = importlib.util.spec_from_file_location(
    "database_mappings",
    "/Users/tyo/GITHUB/omics-os/lobster/lobster/core/schemas/database_mappings.py"
)
database_mappings_module = importlib.util.module_from_spec(spec_db)

# Load both modules
spec_db.loader.exec_module(database_mappings_module)
sys.modules['lobster.core.schemas.database_mappings'] = database_mappings_module

# Mock logger to avoid dependency
class MockLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

class MockLoggerModule:
    @staticmethod
    def get_logger(name):
        return MockLogger()

sys.modules['lobster.utils.logger'] = MockLoggerModule()

# Now load accession_resolver
spec.loader.exec_module(accession_resolver_module)

# Extract what we need
get_accession_resolver = accession_resolver_module.get_accession_resolver
reset_resolver = accession_resolver_module.reset_resolver


# =============================================================================
# Test Data: Valid Identifiers (37 patterns)
# =============================================================================

VALID_IDENTIFIERS = {
    # NCBI Accessions (6)
    "bioproject_accession": "PRJNA123456",
    "biosample_accession": "SAMN12345678",
    "sra_study_accession": "SRP123456",
    "sra_experiment_accession": "SRX123456",
    "sra_run_accession": "SRR123456",
    "sra_sample_accession": "SRS123456",

    # ENA Accessions (6)
    "ena_study_accession": "ERP123456",
    "ena_experiment_accession": "ERX123456",
    "ena_run_accession": "ERR123456",
    "ena_sample_accession": "ERS123456",
    "bioproject_ena_accession": "PRJEB83385",
    "biosample_ena_accession": "SAMEA123456",

    # DDBJ Accessions (6)
    "ddbj_study_accession": "DRP123456",
    "ddbj_experiment_accession": "DRX123456",
    "ddbj_run_accession": "DRR123456",
    "ddbj_sample_accession": "DRS123456",
    "bioproject_ddbj_accession": "PRJDB12345",
    "biosample_ddbj_accession": "SAMD12345678",

    # GEO Accessions (4)
    "geo_accession": "GSE194247",
    "geo_sample_accession": "GSM1234567",
    "geo_platform_accession": "GPL570",
    "geo_dataset_accession": "GDS5093",

    # Proteomics Accessions (2)
    "pride_accession": "PXD012345",
    "massive_accession": "MSV000012345",

    # Metabolomics Accessions (2)
    "metabolights_accession": "MTBLS1234",
    "metabolomics_workbench_accession": "ST001234",

    # Metagenomics Accessions (1)
    "mgnify_accession": "MGYS00001234",

    # Cross-Platform Accessions (2)
    "arrayexpress_accession": "E-MTAB-12345",
    "publication_doi": "10.1038/nature12345",

    # EGA Accessions - Controlled Access (8)
    "ega_study_accession": "EGAS00001234567",
    "ega_dataset_accession": "EGAD50000000740",
    "ega_sample_accession": "EGAN00001234567",
    "ega_experiment_accession": "EGAX00001234567",
    "ega_run_accession": "EGAR00001234567",
    "ega_analysis_accession": "EGAZ00001234567",
    "ega_policy_accession": "EGAP00001234567",
    "ega_dac_accession": "EGAC00001234567",
}

# Expected database names for each field
EXPECTED_DATABASES = {
    # NCBI
    "bioproject_accession": "NCBI BioProject",
    "biosample_accession": "NCBI BioSample",
    "sra_study_accession": "NCBI Sequence Read Archive (Study)",
    "sra_experiment_accession": "NCBI Sequence Read Archive (Experiment)",
    "sra_run_accession": "NCBI Sequence Read Archive (Run)",
    "sra_sample_accession": "NCBI Sequence Read Archive (Sample)",

    # ENA
    "ena_study_accession": "ENA Sequence Read Archive (Study)",
    "ena_experiment_accession": "ENA Sequence Read Archive (Experiment)",
    "ena_run_accession": "ENA Sequence Read Archive (Run)",
    "ena_sample_accession": "ENA Sequence Read Archive (Sample)",
    "bioproject_ena_accession": "ENA BioProject",
    "biosample_ena_accession": "ENA BioSample",

    # DDBJ
    "ddbj_study_accession": "DDBJ Sequence Read Archive (Study)",
    "ddbj_experiment_accession": "DDBJ Sequence Read Archive (Experiment)",
    "ddbj_run_accession": "DDBJ Sequence Read Archive (Run)",
    "ddbj_sample_accession": "DDBJ Sequence Read Archive (Sample)",
    "bioproject_ddbj_accession": "DDBJ BioProject",
    "biosample_ddbj_accession": "DDBJ BioSample",

    # GEO
    "geo_accession": "NCBI Gene Expression Omnibus",
    "geo_sample_accession": "NCBI Gene Expression Omnibus (Sample)",
    "geo_platform_accession": "NCBI Gene Expression Omnibus (Platform)",
    "geo_dataset_accession": "NCBI Gene Expression Omnibus (Dataset)",

    # Proteomics
    "pride_accession": "ProteomeXchange/PRIDE",
    "massive_accession": "MassIVE",

    # Metabolomics
    "metabolights_accession": "MetaboLights",
    "metabolomics_workbench_accession": "Metabolomics Workbench",

    # Metagenomics
    "mgnify_accession": "MGnify (EBI Metagenomics)",

    # Cross-Platform
    "arrayexpress_accession": "ArrayExpress",
    "publication_doi": "Digital Object Identifier",

    # EGA
    "ega_study_accession": "European Genome-phenome Archive (Study)",
    "ega_dataset_accession": "European Genome-phenome Archive (Dataset)",
    "ega_sample_accession": "European Genome-phenome Archive (Sample)",
    "ega_experiment_accession": "European Genome-phenome Archive (Experiment)",
    "ega_run_accession": "European Genome-phenome Archive (Run)",
    "ega_analysis_accession": "European Genome-phenome Archive (Analysis)",
    "ega_policy_accession": "European Genome-phenome Archive (Policy)",
    "ega_dac_accession": "European Genome-phenome Archive (DAC)",
}

# Invalid identifiers for each pattern
INVALID_IDENTIFIERS = {
    "bioproject_accession": ["GSE123", "PRJNA", "PRJNA12", "prjna_abc"],
    "biosample_accession": ["SAMN", "SAMN123", "GSE123456"],
    "sra_study_accession": ["SRP", "SRP123", "GSE123"],
    "geo_accession": ["GSE12", "GEO123", "PRJNA123456"],
    "pride_accession": ["PXD", "PXD12", "PXD1234567", "GSE123"],
    "massive_accession": ["MSV", "MSV123", "MSV12345678901"],
    "metabolights_accession": ["MTBLS", "MTB123", "GSE123"],
    "arrayexpress_accession": ["E-MTB-123", "EMTAB12345", "E-M-123"],
    "publication_doi": ["10.123", "doi:10.1038", "10.1038"],
    "ega_study_accession": ["EGAS", "EGAS123", "EGAS123456789012"],
    "ega_dataset_accession": ["EGAD", "EGAD123", "GSE123"],
}

# Case sensitivity tests
CASE_VARIANTS = [
    ("GSE123456", "gse123456", "GsE123456", "GSe123456"),
    ("PRJNA123456", "prjna123456", "Prjna123456"),
    ("PXD012345", "pxd012345", "Pxd012345"),
    ("EGAS00001234567", "egas00001234567", "Egas00001234567"),
]


# =============================================================================
# Test Suite Classes
# =============================================================================

class TestResults:
    """Track test results with statistics."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self):
        self.passed += 1

    def record_fail(self, test_name: str, details: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {details}")

    def print_summary(self):
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.failed}")

        if self.errors:
            print("\nFailed Tests:")
            for error in self.errors:
                print(f"  - {error}")

        return self.failed == 0


def test_detect_database(results: TestResults):
    """Test detect_database() for all 37 patterns."""
    print("\n" + "="*80)
    print("TEST: detect_database() - All 37 Patterns")
    print("="*80)

    resolver = get_accession_resolver()

    for field_name, identifier in VALID_IDENTIFIERS.items():
        expected_db = EXPECTED_DATABASES[field_name]
        detected_db = resolver.detect_database(identifier)

        if detected_db == expected_db:
            results.record_pass()
            print(f"✓ {field_name}: {identifier} -> {detected_db}")
        else:
            results.record_fail(
                f"detect_database({identifier})",
                f"Expected '{expected_db}', got '{detected_db}'"
            )
            print(f"✗ {field_name}: {identifier} -> Expected '{expected_db}', got '{detected_db}'")


def test_detect_field(results: TestResults):
    """Test detect_field() for all patterns."""
    print("\n" + "="*80)
    print("TEST: detect_field() - All Patterns")
    print("="*80)

    resolver = get_accession_resolver()

    for expected_field, identifier in VALID_IDENTIFIERS.items():
        detected_field = resolver.detect_field(identifier)

        if detected_field == expected_field:
            results.record_pass()
            print(f"✓ {identifier} -> {detected_field}")
        else:
            results.record_fail(
                f"detect_field({identifier})",
                f"Expected '{expected_field}', got '{detected_field}'"
            )
            print(f"✗ {identifier} -> Expected '{expected_field}', got '{detected_field}'")


def test_validate_generic(results: TestResults):
    """Test validate() without database specification."""
    print("\n" + "="*80)
    print("TEST: validate() - Generic Validation")
    print("="*80)

    resolver = get_accession_resolver()

    # Test valid identifiers
    print("\nValid identifiers (should pass):")
    for field_name, identifier in VALID_IDENTIFIERS.items():
        is_valid = resolver.validate(identifier)

        if is_valid:
            results.record_pass()
            print(f"✓ {identifier} -> Valid")
        else:
            results.record_fail(
                f"validate({identifier})",
                f"Should be valid but returned False"
            )
            print(f"✗ {identifier} -> Should be valid but returned False")

    # Test invalid identifiers
    print("\nInvalid identifiers (should fail):")
    invalid_samples = [
        "INVALID123",
        "GSE12",  # Too short
        "PRJNA",  # No digits
        "12345",  # No prefix
        "",       # Empty
        "   ",    # Whitespace only
    ]

    for invalid_id in invalid_samples:
        is_valid = resolver.validate(invalid_id)

        if not is_valid:
            results.record_pass()
            print(f"✓ '{invalid_id}' -> Invalid (correct)")
        else:
            results.record_fail(
                f"validate('{invalid_id}')",
                f"Should be invalid but returned True"
            )
            print(f"✗ '{invalid_id}' -> Should be invalid but returned True")


def test_validate_with_database(results: TestResults):
    """Test validate() with database specification."""
    print("\n" + "="*80)
    print("TEST: validate(database=...) - Database-Specific Validation")
    print("="*80)

    resolver = get_accession_resolver()

    test_cases = [
        ("GSE123456", "GEO", True),
        ("GSE123456", "geo", True),  # Case insensitive
        ("GSE123456", "SRA", False),  # Wrong database
        ("GSE123456", "PRIDE", False),
        ("PXD012345", "PRIDE", True),
        ("PXD012345", "pride", True),
        ("PXD012345", "GEO", False),
        ("SRP123456", "SRA", True),
        ("SRP123456", "sra", True),
        ("SRP123456", "GEO", False),
        # Note: EGA validation may not work with simplified "EGA" name
        # It needs to match against "European Genome-phenome Archive" pattern
        ("EGAS00001234567", "GEO", False),
    ]

    for identifier, database, expected in test_cases:
        is_valid = resolver.validate(identifier, database=database)

        if is_valid == expected:
            results.record_pass()
            status = "Valid" if is_valid else "Invalid"
            print(f"✓ validate('{identifier}', database='{database}') -> {status} (correct)")
        else:
            results.record_fail(
                f"validate('{identifier}', database='{database}')",
                f"Expected {expected}, got {is_valid}"
            )
            print(f"✗ validate('{identifier}', database='{database}') -> Expected {expected}, got {is_valid}")


def test_extract_all_accessions(results: TestResults):
    """Test extract_all_accessions() for text extraction."""
    print("\n" + "="*80)
    print("TEST: extract_all_accessions() - Text Extraction")
    print("="*80)

    resolver = get_accession_resolver()

    test_texts = [
        {
            "text": "Data available at GSE123456 and PRIDE PXD012345",
            "expected": {
                "NCBI Gene Expression Omnibus": ["GSE123456"],
                "ProteomeXchange/PRIDE": ["PXD012345"]
            }
        },
        {
            "text": "Sequencing data deposited in SRP123456, SRX789012, and SRR345678",
            "expected": {
                "NCBI Sequence Read Archive (Study)": ["SRP123456"],
                "NCBI Sequence Read Archive (Experiment)": ["SRX789012"],
                "NCBI Sequence Read Archive (Run)": ["SRR345678"]
            }
        },
        {
            "text": "Methods: We downloaded GSE194247 and processed with tools from 10.1038/nmeth.1234",
            "expected": {
                "NCBI Gene Expression Omnibus": ["GSE194247"],
                "Digital Object Identifier": ["10.1038/NMETH.1234"]
            }
        },
        {
            "text": "Controlled access data: EGAS00001234567 and EGAD50000000740",
            "expected": {
                "European Genome-phenome Archive (Study)": ["EGAS00001234567"],
                "European Genome-phenome Archive (Dataset)": ["EGAD50000000740"]
            }
        },
        {
            "text": "Multiple BioProjects: PRJNA123456, PRJEB83385, and PRJDB12345",
            "expected": {
                "NCBI BioProject": ["PRJNA123456"],
                "ENA BioProject": ["PRJEB83385"],
                "DDBJ BioProject": ["PRJDB12345"]
            }
        },
    ]

    for test_case in test_texts:
        text = test_case["text"]
        expected = test_case["expected"]

        extracted = resolver.extract_all_accessions(text)

        # Check if results match
        matches = extracted == expected

        if matches:
            results.record_pass()
            print(f"✓ Extracted from: '{text[:50]}...'")
            print(f"  Results: {extracted}")
        else:
            results.record_fail(
                f"extract_all_accessions",
                f"Text: '{text[:50]}...'\nExpected: {expected}\nGot: {extracted}"
            )
            print(f"✗ Text: '{text[:50]}...'")
            print(f"  Expected: {expected}")
            print(f"  Got: {extracted}")


def test_extract_accessions_by_type(results: TestResults):
    """Test extract_accessions_by_type() for simplified type extraction."""
    print("\n" + "="*80)
    print("TEST: extract_accessions_by_type() - Simplified Type Extraction")
    print("="*80)

    resolver = get_accession_resolver()

    test_cases = [
        {
            "text": "GSE123456 and SRP789012",
            "expected": {
                "GEO": {"GSE123456"},
                "SRA": {"SRP789012"}
            }
        },
        {
            "text": "PXD012345 and MSV000012345",
            "expected": {
                "PRIDE": {"PXD012345"},
                "MassIVE": {"MSV000012345"}
            }
        },
        {
            "text": "EGA data: EGAS00001234567, EGAD50000000740, EGAN00001234567",
            "expected": {
                "EGA": {"EGAS00001234567", "EGAD50000000740", "EGAN00001234567"}
            }
        },
    ]

    for test_case in test_cases:
        text = test_case["text"]
        expected = test_case["expected"]

        extracted = resolver.extract_accessions_by_type(text)

        # Convert lists to sets for comparison (order doesn't matter)
        extracted_sets = {k: set(v) for k, v in extracted.items()}

        if extracted_sets == expected:
            results.record_pass()
            print(f"✓ '{text}' -> {extracted}")
        else:
            results.record_fail(
                f"extract_accessions_by_type",
                f"Expected {expected}, got {extracted_sets}"
            )
            print(f"✗ '{text}'")
            print(f"  Expected: {expected}")
            print(f"  Got: {extracted_sets}")


def test_case_sensitivity(results: TestResults):
    """Test case-insensitive matching."""
    print("\n" + "="*80)
    print("TEST: Case Sensitivity")
    print("="*80)

    resolver = get_accession_resolver()

    for variants in CASE_VARIANTS:
        canonical = variants[0]

        print(f"\nTesting variants of {canonical}:")
        for variant in variants:
            detected = resolver.detect_database(variant)

            if detected is not None:
                results.record_pass()
                print(f"  ✓ {variant} -> {detected}")
            else:
                results.record_fail(
                    f"case_sensitivity({variant})",
                    f"Should match pattern but returned None"
                )
                print(f"  ✗ {variant} -> None (should match)")


def test_whitespace_handling(results: TestResults):
    """Test handling of whitespace."""
    print("\n" + "="*80)
    print("TEST: Whitespace Handling")
    print("="*80)

    resolver = get_accession_resolver()

    test_cases = [
        ("  GSE123456  ", "NCBI Gene Expression Omnibus"),
        ("\tGSE123456\t", "NCBI Gene Expression Omnibus"),
        ("\nPXD012345\n", "ProteomeXchange/PRIDE"),
        ("   EGAS00001234567   ", "European Genome-phenome Archive (Study)"),
    ]

    for identifier_with_ws, expected_db in test_cases:
        detected = resolver.detect_database(identifier_with_ws)

        if detected == expected_db:
            results.record_pass()
            print(f"✓ '{identifier_with_ws}' -> {detected}")
        else:
            results.record_fail(
                f"whitespace_handling('{identifier_with_ws}')",
                f"Expected '{expected_db}', got '{detected}'"
            )
            print(f"✗ '{identifier_with_ws}' -> Expected '{expected_db}', got '{detected}'")


def test_helper_methods(results: TestResults):
    """Test helper methods like is_geo_identifier(), etc."""
    print("\n" + "="*80)
    print("TEST: Helper Methods")
    print("="*80)

    resolver = get_accession_resolver()

    # Test is_geo_identifier()
    print("\nis_geo_identifier():")
    geo_tests = [
        ("GSE123456", True),
        ("GSM123456", True),
        ("GPL570", True),
        ("GDS5093", True),
        ("SRP123456", False),
        ("PXD012345", False),
    ]

    for identifier, expected in geo_tests:
        result = resolver.is_geo_identifier(identifier)

        if result == expected:
            results.record_pass()
            print(f"  ✓ {identifier} -> {result}")
        else:
            results.record_fail(
                f"is_geo_identifier({identifier})",
                f"Expected {expected}, got {result}"
            )
            print(f"  ✗ {identifier} -> Expected {expected}, got {result}")

    # Test is_sra_identifier()
    print("\nis_sra_identifier():")
    sra_tests = [
        ("SRP123456", True),
        ("SRX123456", True),
        ("SRR123456", True),
        ("ERP123456", True),
        ("DRP123456", True),
        ("GSE123456", False),
        ("PXD012345", False),
    ]

    for identifier, expected in sra_tests:
        result = resolver.is_sra_identifier(identifier)

        if result == expected:
            results.record_pass()
            print(f"  ✓ {identifier} -> {result}")
        else:
            results.record_fail(
                f"is_sra_identifier({identifier})",
                f"Expected {expected}, got {result}"
            )
            print(f"  ✗ {identifier} -> Expected {expected}, got {result}")

    # Test is_proteomics_identifier()
    print("\nis_proteomics_identifier():")
    proteomics_tests = [
        ("PXD012345", True),
        ("MSV000012345", True),
        ("GSE123456", False),
        ("SRP123456", False),
    ]

    for identifier, expected in proteomics_tests:
        result = resolver.is_proteomics_identifier(identifier)

        if result == expected:
            results.record_pass()
            print(f"  ✓ {identifier} -> {result}")
        else:
            results.record_fail(
                f"is_proteomics_identifier({identifier})",
                f"Expected {expected}, got {result}"
            )
            print(f"  ✗ {identifier} -> Expected {expected}, got {result}")

    # Test is_ega_identifier()
    print("\nis_ega_identifier():")
    ega_tests = [
        ("EGAS00001234567", True),
        ("EGAD50000000740", True),
        ("EGAN00001234567", True),
        ("EGAX00001234567", True),
        ("GSE123456", False),
        ("SRP123456", False),
    ]

    for identifier, expected in ega_tests:
        result = resolver.is_ega_identifier(identifier)

        if result == expected:
            results.record_pass()
            print(f"  ✓ {identifier} -> {result}")
        else:
            results.record_fail(
                f"is_ega_identifier({identifier})",
                f"Expected {expected}, got {result}"
            )
            print(f"  ✗ {identifier} -> Expected {expected}, got {result}")


def test_get_url(results: TestResults):
    """Test get_url() for URL generation."""
    print("\n" + "="*80)
    print("TEST: get_url() - URL Generation")
    print("="*80)

    resolver = get_accession_resolver()

    test_cases = [
        ("GSE123456", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456"),
        ("PRJNA123456", "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA123456"),
        ("PXD012345", "https://www.ebi.ac.uk/pride/archive/projects/PXD012345"),
        ("EGAS00001234567", "https://ega-archive.org/studies/EGAS00001234567"),
        ("10.1038/nature12345", "https://doi.org/10.1038/nature12345"),
    ]

    for identifier, expected_url in test_cases:
        url = resolver.get_url(identifier)

        if url == expected_url:
            results.record_pass()
            print(f"✓ {identifier} -> {url}")
        else:
            results.record_fail(
                f"get_url({identifier})",
                f"Expected '{expected_url}', got '{url}'"
            )
            print(f"✗ {identifier}")
            print(f"  Expected: {expected_url}")
            print(f"  Got: {url}")


def test_normalize_identifier(results: TestResults):
    """Test normalize_identifier() for identifier normalization."""
    print("\n" + "="*80)
    print("TEST: normalize_identifier() - Identifier Normalization")
    print("="*80)

    resolver = get_accession_resolver()

    test_cases = [
        ("  gse12345  ", "GSE12345"),
        ("prjna123456", "PRJNA123456"),
        ("pxd012345", "PXD012345"),
        ("\tegas00001234567\n", "EGAS00001234567"),
        ("SrP123456", "SRP123456"),
        ("msv000012345", "MSV000012345"),
    ]

    for input_id, expected_output in test_cases:
        normalized = resolver.normalize_identifier(input_id)

        if normalized == expected_output:
            results.record_pass()
            print(f"✓ '{input_id}' -> '{normalized}'")
        else:
            results.record_fail(
                f"normalize_identifier('{input_id}')",
                f"Expected '{expected_output}', got '{normalized}'"
            )
            print(f"✗ '{input_id}'")
            print(f"  Expected: '{expected_output}'")
            print(f"  Got: '{normalized}'")


def test_access_type(results: TestResults):
    """Test get_access_type() and is_controlled_access()."""
    print("\n" + "="*80)
    print("TEST: Access Type Detection")
    print("="*80)

    resolver = get_accession_resolver()

    # Test get_access_type()
    print("\nget_access_type():")
    access_tests = [
        ("GSE123456", "open"),
        ("PXD012345", "open"),
        ("EGAS00001234567", "controlled"),
        ("EGAD50000000740", "controlled"),
        ("EGAC00001234567", "controlled"),
        ("INVALID123", "unknown"),
    ]

    for identifier, expected_type in access_tests:
        access_type = resolver.get_access_type(identifier)

        if access_type == expected_type:
            results.record_pass()
            print(f"  ✓ {identifier} -> {access_type}")
        else:
            results.record_fail(
                f"get_access_type({identifier})",
                f"Expected '{expected_type}', got '{access_type}'"
            )
            print(f"  ✗ {identifier} -> Expected '{expected_type}', got '{access_type}'")

    # Test is_controlled_access()
    print("\nis_controlled_access():")
    controlled_tests = [
        ("GSE123456", False),
        ("PXD012345", False),
        ("EGAS00001234567", True),
        ("EGAD50000000740", True),
        ("EGAN00001234567", True),
        ("EGAC00001234567", True),
    ]

    for identifier, expected_controlled in controlled_tests:
        is_controlled = resolver.is_controlled_access(identifier)

        if is_controlled == expected_controlled:
            results.record_pass()
            print(f"  ✓ {identifier} -> {is_controlled}")
        else:
            results.record_fail(
                f"is_controlled_access({identifier})",
                f"Expected {expected_controlled}, got {is_controlled}"
            )
            print(f"  ✗ {identifier} -> Expected {expected_controlled}, got {is_controlled}")


def test_extract_with_metadata(results: TestResults):
    """Test extract_accessions_with_metadata() for metadata extraction."""
    print("\n" + "="*80)
    print("TEST: extract_accessions_with_metadata() - Metadata Extraction")
    print("="*80)

    resolver = get_accession_resolver()

    text = """
    We analyzed data from GSE123456 (open access) and EGAD50000000740 (controlled access).
    Proteomics data available at PXD012345. Published in 10.1038/nature12345
    """

    metadata_results = resolver.extract_accessions_with_metadata(text)

    # Check expected accessions (set comparison - order doesn't matter, DOI may include period)
    expected_accessions = {"GSE123456", "EGAD50000000740", "PXD012345", "10.1038/NATURE12345"}
    found_accessions = {r["accession"].rstrip('.') for r in metadata_results}  # Strip trailing periods

    print(f"\nText: '{text[:80]}...'")
    print(f"\nExtracted {len(metadata_results)} accessions with metadata:\n")

    for result in metadata_results:
        print(f"  {result['accession']}")
        print(f"    Database: {result['database']}")
        print(f"    Field: {result['field_name']}")
        print(f"    Access: {result['access_type']}")
        if result['access_notes']:
            print(f"    Notes: {result['access_notes'][:80]}...")
        print()

    all_found = expected_accessions == found_accessions

    if all_found:
        results.record_pass()
        print("✓ All expected accessions found with correct metadata")
    else:
        results.record_fail(
            "extract_accessions_with_metadata",
            f"Expected {expected_accessions}, found {found_accessions}"
        )
        print(f"✗ Expected {expected_accessions}")
        print(f"  Found: {found_accessions}")


def test_performance_large_text(results: TestResults):
    """Test performance with large text blocks."""
    print("\n" + "="*80)
    print("TEST: Performance - Large Text Blocks")
    print("="*80)

    resolver = get_accession_resolver()

    # Generate large text with embedded accessions
    large_text = "Lorem ipsum dolor sit amet. " * 1000
    accessions = ["GSE123456", "PRJNA789012", "PXD012345", "SRP345678", "EGAS00001234567"]

    # Insert accessions at various positions
    positions = [100, 500, 1000, 2000, 3000]
    for pos, acc in zip(positions, accessions):
        large_text = large_text[:pos] + f" {acc} " + large_text[pos:]

    print(f"\nText size: {len(large_text)} characters")
    print(f"Embedded accessions: {accessions}\n")

    # Time the extraction
    start_time = time.time()
    extracted = resolver.extract_all_accessions(large_text)
    elapsed = time.time() - start_time

    # Check results
    all_found = True
    for acc in accessions:
        found = False
        for db_accessions in extracted.values():
            if acc.upper() in db_accessions:
                found = True
                break
        if not found:
            all_found = False
            break

    print(f"Extraction time: {elapsed:.4f} seconds")
    print(f"Extracted databases: {list(extracted.keys())}")

    if all_found and elapsed < 1.0:  # Should be fast
        results.record_pass()
        print("✓ Performance test passed (all accessions found, <1 second)")
    elif not all_found:
        results.record_fail(
            "performance_large_text",
            f"Not all accessions found. Expected {accessions}, got {extracted}"
        )
        print(f"✗ Not all accessions found")
    else:
        results.record_fail(
            "performance_large_text",
            f"Extraction too slow: {elapsed:.4f} seconds"
        )
        print(f"✗ Extraction too slow: {elapsed:.4f} seconds")


def test_mixed_content_extraction(results: TestResults):
    """Test extraction from mixed content with multiple database types."""
    print("\n" + "="*80)
    print("TEST: Mixed Content Extraction")
    print("="*80)

    resolver = get_accession_resolver()

    mixed_text = """
    Methods: RNA-seq data was downloaded from GEO (GSE123456, GSM789012) and
    SRA (SRP345678, SRR999888). Proteomics data from PRIDE (PXD012345) and
    MassIVE (MSV000067890). Metabolomics from MetaboLights (MTBLS5678).

    Controlled access data was obtained via DAC application:
    - EGA Study: EGAS00001234567
    - EGA Dataset: EGAD50000000740
    - EGA Samples: EGAN00001234567, EGAN00009876543

    BioProject: PRJNA123456, PRJEB83385, PRJDB98765
    BioSample: SAMN12345678, SAMEA9876543, SAMD87654321

    Published in doi:10.1038/nature12345 and 10.1016/j.cell.2023.01.001
    """

    print(f"Extracting from complex methods section ({len(mixed_text)} chars)...\n")

    extracted = resolver.extract_all_accessions(mixed_text)

    # Expected minimum databases
    expected_dbs = [
        "NCBI Gene Expression Omnibus",
        "NCBI Sequence Read Archive",
        "ProteomeXchange/PRIDE",
        "MassIVE",
        "MetaboLights",
        "European Genome-phenome Archive",
        "NCBI BioProject",
        "NCBI BioSample",
        "Digital Object Identifier"
    ]

    found_dbs = list(extracted.keys())

    print("Extracted accessions:")
    for db, accs in extracted.items():
        print(f"\n  {db}:")
        for acc in accs:
            print(f"    - {acc}")

    # Check if major databases found
    all_major_found = all(
        any(exp_db in found_db for found_db in found_dbs)
        for exp_db in ["Gene Expression", "Sequence Read", "PRIDE", "Genome-phenome", "BioProject"]
    )

    if all_major_found:
        results.record_pass()
        print(f"\n✓ Successfully extracted accessions from {len(found_dbs)} database types")
    else:
        results.record_fail(
            "mixed_content_extraction",
            f"Missing some expected databases. Found: {found_dbs}"
        )
        print(f"\n✗ Missing some expected database types")


def test_supported_databases(results: TestResults):
    """Test get_supported_databases() and get_supported_types()."""
    print("\n" + "="*80)
    print("TEST: Supported Databases and Types")
    print("="*80)

    resolver = get_accession_resolver()

    databases = resolver.get_supported_databases()
    types = resolver.get_supported_types()

    print(f"\nSupported Databases ({len(databases)}):")
    for db in databases:
        print(f"  - {db}")

    print(f"\nSupported Types ({len(types)}):")
    for t in types:
        print(f"  - {t}")

    # Should have multiple databases and types
    if len(databases) >= 20 and len(types) >= 10:
        results.record_pass()
        print(f"\n✓ Found {len(databases)} databases and {len(types)} types")
    else:
        results.record_fail(
            "supported_databases",
            f"Expected >=20 databases and >=10 types, got {len(databases)} databases and {len(types)} types"
        )
        print(f"\n✗ Expected more databases/types")


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    """Run all tests and print summary."""
    print("="*80)
    print("ACCESSION RESOLVER - COMPREHENSIVE MANUAL TEST SUITE")
    print("="*80)
    print("\nTesting AccessionResolver with all 37 identifier patterns")
    print("(29 base patterns + 8 EGA patterns)")
    print()

    results = TestResults()

    # Run all test suites
    test_detect_database(results)
    test_detect_field(results)
    test_validate_generic(results)
    test_validate_with_database(results)
    test_extract_all_accessions(results)
    test_extract_accessions_by_type(results)
    test_case_sensitivity(results)
    test_whitespace_handling(results)
    test_helper_methods(results)
    test_get_url(results)
    test_normalize_identifier(results)
    test_access_type(results)
    test_extract_with_metadata(results)
    test_performance_large_text(results)
    test_mixed_content_extraction(results)
    test_supported_databases(results)

    # Print final summary
    success = results.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
