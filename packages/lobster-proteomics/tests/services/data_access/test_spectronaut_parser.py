"""
Comprehensive unit tests for the Spectronaut parser.

This module tests all aspects of the SpectronautParser including:
- Long format parsing
- Matrix format parsing
- Format auto-detection
- Q-value filtering
- Log2 transformation
- Contaminant/reverse hit filtering
- Gene symbol indexing
- Error handling for invalid files
- Edge cases (single sample, single protein, etc.)
"""

import os
import tempfile
from pathlib import Path

import anndata
import numpy as np
import pytest

from lobster.services.data_access.proteomics_parsers import (
    FileValidationError,
    ParsingError,
    SpectronautParser,
    get_parser_for_file,
)

# Fixtures directory path
FIXTURES_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "fixtures"
    / "proteomics"
    / "spectronaut"
)


class TestSpectronautParserInit:
    """Tests for SpectronautParser initialization."""

    def test_init_default_values(self):
        """Test parser initializes with correct default values."""
        parser = SpectronautParser()
        assert parser.name == "Spectronaut"
        assert parser.version == "1.0.0"

    def test_get_supported_formats(self):
        """Test supported formats include expected extensions."""
        parser = SpectronautParser()
        formats = parser.get_supported_formats()
        assert ".tsv" in formats
        assert ".txt" in formats
        assert ".csv" in formats
        assert ".xlsx" in formats

    def test_repr(self):
        """Test string representation."""
        parser = SpectronautParser()
        repr_str = repr(parser)
        assert "SpectronautParser" in repr_str
        assert "Spectronaut" in repr_str


class TestSpectronautParserValidation:
    """Tests for file validation."""

    def test_validate_long_format_file(self):
        """Test validation of valid long-format Spectronaut file."""
        parser = SpectronautParser()
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        assert parser.validate_file(str(file_path)) is True

    def test_validate_matrix_format_file(self):
        """Test validation of valid matrix-format Spectronaut file."""
        parser = SpectronautParser()
        file_path = FIXTURES_DIR / "spectronaut_matrix_format.tsv"
        assert parser.validate_file(str(file_path)) is True

    def test_validate_invalid_file(self):
        """Test validation rejects invalid file format."""
        parser = SpectronautParser()
        file_path = FIXTURES_DIR / "spectronaut_invalid.tsv"
        assert parser.validate_file(str(file_path)) is False

    def test_validate_nonexistent_file(self):
        """Test validation returns False for nonexistent file."""
        parser = SpectronautParser()
        assert parser.validate_file("/nonexistent/path/file.tsv") is False

    def test_validate_unsupported_extension(self):
        """Test validation rejects unsupported file extensions."""
        parser = SpectronautParser()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"PG.ProteinGroups\tSample1\nP12345\t1000\n")
            temp_path = f.name
        try:
            assert parser.validate_file(temp_path) is False
        finally:
            os.unlink(temp_path)


class TestSpectronautParserLongFormat:
    """Tests for long-format Spectronaut parsing."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    @pytest.fixture
    def long_format_file(self):
        """Path to long-format fixture file."""
        return FIXTURES_DIR / "spectronaut_long_format.tsv"

    def test_parse_long_format_basic(self, parser, long_format_file):
        """Test basic parsing of long-format file."""
        adata, stats, _ = parser.parse(str(long_format_file))

        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs > 0  # Has samples
        assert adata.n_vars > 0  # Has proteins
        assert "n_samples" in stats
        assert "n_proteins" in stats

    def test_parse_long_format_sample_detection(self, parser, long_format_file):
        """Test samples are correctly detected from R.FileName."""
        adata, _, _ = parser.parse(str(long_format_file))

        # Should have 5 unique samples (Sample1-5)
        assert adata.n_obs == 5
        assert "sample_name" in adata.obs.columns

    def test_parse_long_format_qvalue_filtering(self, parser, long_format_file):
        """Test Q-value filtering removes low-confidence proteins."""
        # Parse with strict Q-value threshold
        adata_strict, stats_strict, _ = parser.parse(
            str(long_format_file), qvalue_threshold=0.005
        )

        # Parse with lenient threshold
        adata_lenient, stats_lenient, _ = parser.parse(
            str(long_format_file), qvalue_threshold=0.05
        )

        # Strict should have fewer or equal proteins
        assert adata_strict.n_vars <= adata_lenient.n_vars

    def test_parse_long_format_log_transform(self, parser, long_format_file):
        """Test log2 transformation is applied correctly."""
        # With log transform
        adata_log, _, _ = parser.parse(str(long_format_file), log_transform=True)

        # Without log transform
        adata_raw, _, _ = parser.parse(str(long_format_file), log_transform=False)

        # Log-transformed values should be much smaller
        # (Spectronaut raw values are typically in millions)
        mean_log = np.nanmean(adata_log.X)
        mean_raw = np.nanmean(adata_raw.X)

        assert mean_log < mean_raw
        # Log2 of ~1M is ~20
        assert mean_log < 30

    def test_parse_long_format_pseudocount(self, parser):
        """Test pseudocount parameter for log transformation."""
        # Use small intensity values where pseudocount actually matters
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t10\t0.001\n")  # Small value
            f.write("Sample1.raw\tP67890\tKRAS\t100\t0.002\n")
            f.write("Sample2.raw\tP12345\tEGFR\t20\t0.001\n")
            f.write("Sample2.raw\tP67890\tKRAS\t200\t0.002\n")
            temp_path = f.name

        try:
            adata_p1, _, _ = parser.parse(
                temp_path, log_transform=True, pseudocount=1.0
            )
            adata_p10, _, _ = parser.parse(
                temp_path, log_transform=True, pseudocount=10.0
            )

            # Different pseudocounts should produce noticeably different values
            # for small intensities (log2(10+1) ≈ 3.46, log2(10+10) ≈ 4.32)
            assert not np.allclose(adata_p1.X, adata_p10.X, equal_nan=True)
        finally:
            os.unlink(temp_path)

    def test_parse_long_format_contaminant_filtering(self, parser, long_format_file):
        """Test contaminant proteins are filtered."""
        # With contaminant filtering
        adata_filtered, stats_filtered, _ = parser.parse(
            str(long_format_file), filter_contaminants=True
        )

        # Without contaminant filtering
        adata_unfiltered, stats_unfiltered, _ = parser.parse(
            str(long_format_file), filter_contaminants=False
        )

        # Unfiltered should have more proteins (includes CON__)
        assert adata_unfiltered.n_vars >= adata_filtered.n_vars

        # Check is_contaminant flag exists in unfiltered
        assert "is_contaminant" in adata_unfiltered.var.columns
        assert adata_unfiltered.var["is_contaminant"].any()

    def test_parse_long_format_reverse_filtering(self, parser, long_format_file):
        """Test reverse/decoy hits are filtered."""
        # With reverse filtering
        adata_filtered, _, _ = parser.parse(str(long_format_file), filter_reverse=True)

        # Without reverse filtering
        adata_unfiltered, _, _ = parser.parse(
            str(long_format_file), filter_reverse=False
        )

        # Unfiltered should have more proteins (includes REV__)
        assert adata_unfiltered.n_vars >= adata_filtered.n_vars

        # Check is_reverse flag exists in unfiltered
        assert "is_reverse" in adata_unfiltered.var.columns
        assert adata_unfiltered.var["is_reverse"].any()

    def test_parse_long_format_gene_index(self, parser, long_format_file):
        """Test gene symbols are used as index when available."""
        adata, _, _ = parser.parse(str(long_format_file), use_genes_as_index=True)

        # Index should contain gene symbols (EGFR, KRAS, etc.)
        var_index = list(adata.var_names)
        assert any("EGFR" in str(idx) for idx in var_index)

    def test_parse_long_format_protein_metadata(self, parser, long_format_file):
        """Test protein metadata is correctly extracted."""
        adata, _, _ = parser.parse(str(long_format_file))

        # Check expected var columns
        assert (
            "protein_groups" in adata.var.columns or "protein_id" in adata.var.columns
        )

        # Check gene symbols column if present
        if "gene_symbols" in adata.var.columns:
            assert not adata.var["gene_symbols"].empty

    def test_parse_long_format_uns_metadata(self, parser, long_format_file):
        """Test parser metadata is stored in uns."""
        adata, _, _ = parser.parse(str(long_format_file))

        assert "parser" in adata.uns
        assert adata.uns["parser"]["name"] == "Spectronaut"
        assert "format_type" in adata.uns["parser"]
        assert adata.uns["parser"]["format_type"] == "long"


class TestSpectronautParserMatrixFormat:
    """Tests for matrix-format Spectronaut parsing."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    @pytest.fixture
    def matrix_format_file(self):
        """Path to matrix-format fixture file."""
        return FIXTURES_DIR / "spectronaut_matrix_format.tsv"

    def test_parse_matrix_format_basic(self, parser, matrix_format_file):
        """Test basic parsing of matrix-format file."""
        adata, stats, _ = parser.parse(str(matrix_format_file))

        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs == 10  # 10 sample columns
        assert adata.n_vars > 0  # Has proteins

    def test_parse_matrix_format_sample_detection(self, parser, matrix_format_file):
        """Test samples are correctly detected from column headers."""
        adata, _, _ = parser.parse(str(matrix_format_file))

        # Should have 10 samples (Sample1-10)
        assert adata.n_obs == 10
        assert "sample_name" in adata.obs.columns

    def test_parse_matrix_format_format_detection(self, parser, matrix_format_file):
        """Test matrix format is correctly auto-detected."""
        adata, _, _ = parser.parse(str(matrix_format_file))

        assert adata.uns["parser"]["format_type"] == "matrix"

    def test_parse_matrix_format_contaminant_filtering(
        self, parser, matrix_format_file
    ):
        """Test contaminant filtering in matrix format."""
        adata_filtered, _, _ = parser.parse(
            str(matrix_format_file), filter_contaminants=True
        )
        adata_unfiltered, _, _ = parser.parse(
            str(matrix_format_file), filter_contaminants=False
        )

        # Unfiltered should have CON__ protein
        assert adata_unfiltered.n_vars >= adata_filtered.n_vars


class TestSpectronautParserErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_parse_file_not_found(self, parser):
        """Test FileValidationError for nonexistent file."""
        # Parser validates file first, so non-existent files raise FileValidationError
        with pytest.raises(FileValidationError):
            parser.parse("/nonexistent/path/file.tsv")

    def test_parse_invalid_format(self, parser):
        """Test FileValidationError for invalid file format."""
        invalid_file = FIXTURES_DIR / "spectronaut_invalid.tsv"
        with pytest.raises(FileValidationError):
            parser.parse(str(invalid_file))

    def test_parse_all_filtered_by_qvalue(self, parser):
        """Test error when all data is filtered by Q-value."""
        high_qvalue_file = FIXTURES_DIR / "spectronaut_high_qvalue.tsv"
        with pytest.raises(ParsingError) as exc_info:
            parser.parse(str(high_qvalue_file), qvalue_threshold=0.001)
        assert "No data remaining" in str(exc_info.value)


class TestSpectronautParserEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_single_sample_long_format(self, parser):
        """Test parsing file with single sample."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t0.002\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path)
            assert adata.n_obs == 1
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_single_protein_long_format(self, parser):
        """Test parsing file with single protein."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("Sample2.raw\tP12345\tEGFR\t2000000\t0.001\n")
            f.write("Sample3.raw\tP12345\tEGFR\t3000000\t0.001\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path)
            assert adata.n_obs == 3
            assert adata.n_vars == 1
        finally:
            os.unlink(temp_path)

    def test_duplicate_protein_ids(self, parser):
        """Test handling of duplicate protein IDs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            f.write("P12345\tEGFR\t1000000\t2000000\n")
            f.write("P12345\tEGFR_2\t3000000\t4000000\n")  # Duplicate protein ID
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path)
            # Should handle duplicates by appending suffix
            assert len(set(adata.var_names)) == adata.n_vars
        finally:
            os.unlink(temp_path)

    def test_missing_gene_column(self, parser):
        """Test parsing file without gene symbols column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\t1000000\t0.001\n")
            f.write("Sample1.raw\tP67890\t2000000\t0.002\n")
            f.write("Sample2.raw\tP12345\t1500000\t0.001\n")
            f.write("Sample2.raw\tP67890\t2500000\t0.002\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path, use_genes_as_index=True)
            # Should fall back to protein IDs
            assert adata.n_obs == 2
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_file_extensions_removal(self, parser):
        """Test various file extensions are removed from sample names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("Sample2.d\tP12345\tEGFR\t2000000\t0.001\n")
            f.write("Sample3.wiff\tP12345\tEGFR\t3000000\t0.001\n")
            f.write("Sample4.mzML\tP12345\tEGFR\t4000000\t0.001\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path)
            sample_names = adata.obs["sample_name"].tolist()
            # Extensions should be removed
            assert "Sample1" in sample_names
            assert "Sample2" in sample_names
            assert "Sample3" in sample_names
            assert "Sample4" in sample_names
            assert not any(".raw" in s for s in sample_names)
        finally:
            os.unlink(temp_path)

    def test_csv_format(self, parser):
        """Test parsing CSV format file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("R.FileName,PG.ProteinGroups,PG.Genes,PG.Quantity,PG.Qvalue\n")
            f.write("Sample1,P12345,EGFR,1000000,0.001\n")
            f.write("Sample2,P12345,EGFR,2000000,0.001\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path)
            assert adata.n_obs == 2
            assert adata.n_vars == 1
        finally:
            os.unlink(temp_path)


class TestSpectronautParserStatistics:
    """Tests for parsing statistics."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_stats_contain_required_fields(self, parser):
        """Test statistics contain all required fields."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        adata, stats, _ = parser.parse(str(file_path))

        required_fields = [
            "n_samples",
            "n_proteins",
            "n_proteins_raw",
            "missing_percentage",
            "parser_name",
            "parser_version",
        ]
        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"

    def test_stats_values_are_valid(self, parser):
        """Test statistics values are valid."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        adata, stats, _ = parser.parse(str(file_path))

        assert stats["n_samples"] > 0
        assert stats["n_proteins"] > 0
        assert stats["n_proteins_raw"] >= stats["n_proteins"]
        assert 0 <= stats["missing_percentage"] <= 100
        assert stats["parser_name"] == "Spectronaut"

    def test_stats_track_filtering(self, parser):
        """Test statistics track filtering operations."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        adata, stats, _ = parser.parse(
            str(file_path),
            filter_contaminants=True,
            filter_reverse=True,
        )

        # Should have tracked contaminant and reverse filtering
        assert "n_contaminants" in stats or stats.get("n_proteins_filtered", 0) > 0


class TestSpectronautParserIntegration:
    """Integration tests with auto-detection and other parsers."""

    def test_auto_detection_spectronaut_file(self):
        """Test auto-detection correctly identifies Spectronaut files."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        parser = get_parser_for_file(str(file_path))
        assert isinstance(parser, SpectronautParser)

    def test_column_mapping(self):
        """Test column mapping is comprehensive."""
        parser = SpectronautParser()
        mapping = parser.get_column_mapping()

        expected_columns = [
            "PG.ProteinGroups",
            "PG.Genes",
            "PG.Quantity",
            "PG.Qvalue",
            "R.FileName",
        ]
        for col in expected_columns:
            assert col in mapping

    def test_provenance_ir_generation(self):
        """Test that AnalysisStep IR is correctly generated for notebook export."""
        parser = SpectronautParser()
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"

        adata, stats, ir = parser.parse(str(file_path))

        # Verify IR is generated
        assert ir is not None
        from lobster.core.analysis_ir import AnalysisStep

        assert isinstance(ir, AnalysisStep)

        # Verify IR contains key fields for notebook export
        assert ir.operation == "spectronaut.parse"
        assert ir.tool_name == "SpectronautParser.parse"
        assert "Spectronaut" in ir.description
        assert ir.library == "lobster"

        # Verify code template is reproducible
        assert (
            "from lobster.services.data_access.proteomics_parsers import SpectronautParser"
            in ir.code_template
        )
        assert "parser.parse(" in ir.code_template
        assert "{{ file_path }}" in ir.code_template

        # Verify parameters are captured
        assert "file_path" in ir.parameters
        assert "qvalue_threshold" in ir.parameters
        assert "log_transform" in ir.parameters

        # Verify parameter schema for documentation
        assert "file_path" in ir.parameter_schema
        assert "qvalue_threshold" in ir.parameter_schema
        assert ir.parameter_schema["qvalue_threshold"]["default"] == 0.01

        # Verify provenance entities
        assert len(ir.input_entities) > 0
        assert ir.input_entities[0]["type"] == "spectronaut_report"
        assert len(ir.output_entities) > 0


class TestSpectronautParserAggregation:
    """Tests for aggregation methods."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    @pytest.fixture
    def long_format_file(self):
        """Path to long-format fixture file."""
        return FIXTURES_DIR / "spectronaut_long_format.tsv"

    def test_aggregation_sum(self, parser, long_format_file):
        """Test sum aggregation method."""
        adata, _, _ = parser.parse(
            str(long_format_file),
            aggregation_method="sum",
            log_transform=False,
        )
        assert adata.n_vars > 0

    def test_aggregation_mean(self, parser, long_format_file):
        """Test mean aggregation method."""
        adata, _, _ = parser.parse(
            str(long_format_file),
            aggregation_method="mean",
            log_transform=False,
        )
        assert adata.n_vars > 0

    def test_aggregation_median(self, parser, long_format_file):
        """Test median aggregation method."""
        adata, _, _ = parser.parse(
            str(long_format_file),
            aggregation_method="median",
            log_transform=False,
        )
        assert adata.n_vars > 0

    def test_aggregation_max(self, parser, long_format_file):
        """Test max aggregation method."""
        adata, _, _ = parser.parse(
            str(long_format_file),
            aggregation_method="max",
            log_transform=False,
        )
        assert adata.n_vars > 0

    def test_invalid_aggregation_method(self, parser, long_format_file):
        """Test error for invalid aggregation method."""
        with pytest.raises(ParsingError) as exc_info:
            parser.parse(
                str(long_format_file),
                aggregation_method="invalid",
            )
        assert "aggregation method" in str(exc_info.value).lower()


class TestSpectronautParserDataQuality:
    """Tests for data quality aspects."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_missing_values_as_nan(self, parser):
        """Test missing values are represented as NaN."""
        # Create file with zeros (which should become NaN in matrix format)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\tSample3\n")
            f.write("P12345\tEGFR\t1000000\t0\t2000000\n")
            f.write("P67890\tKRAS\t0\t500000\t0\n")
            temp_path = f.name

        try:
            adata, stats, _ = parser.parse(temp_path, log_transform=False)
            # Zeros should be converted to NaN
            assert np.isnan(adata.X).any()
        finally:
            os.unlink(temp_path)

    def test_intensity_values_positive(self, parser):
        """Test intensity values are positive after log transform."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        adata, _, _ = parser.parse(str(file_path), log_transform=True)

        # All non-NaN values should be positive (log of positive number)
        non_nan_values = adata.X[~np.isnan(adata.X)]
        assert (non_nan_values > 0).all()

    def test_float32_dtype(self, parser):
        """Test intensity matrix uses float32 dtype."""
        file_path = FIXTURES_DIR / "spectronaut_long_format.tsv"
        adata, _, _ = parser.parse(str(file_path))

        assert adata.X.dtype == np.float32


class TestSpectronautParserCriticalEdgeCases:
    """Critical edge case tests identified by QA review."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_parse_empty_file(self, parser):
        """Test parsing file with header only - no data rows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            temp_path = f.name
        try:
            with pytest.raises(ParsingError) as exc_info:
                parser.parse(temp_path)
            assert (
                "no data" in str(exc_info.value).lower()
                or "empty" in str(exc_info.value).lower()
            )
        finally:
            os.unlink(temp_path)

    def test_parse_header_only_matrix_format(self, parser):
        """Test parsing matrix format file with header only."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            temp_path = f.name
        try:
            # Empty matrix file may be detected as long format and error differently
            # Just verify it raises ParsingError and doesn't crash
            with pytest.raises(ParsingError):
                parser.parse(temp_path)
        finally:
            os.unlink(temp_path)

    def test_nan_protein_groups(self, parser):
        """Test handling of NaN/empty protein groups."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            f.write("P12345\tEGFR\t1000000\t2000000\n")
            f.write("\tKRAS\t3000000\t4000000\n")  # Empty protein group
            f.write("P67890\tTP53\t5000000\t6000000\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Should have parsed successfully, empty protein gets "UNKNOWN"
            assert adata.n_vars >= 2
            assert "UNKNOWN" in list(adata.var_names) or any(
                "UNKNOWN" in str(x) for x in adata.var["protein_id"]
            )
        finally:
            os.unlink(temp_path)

    def test_zero_to_nan_consistency_between_formats(self, parser):
        """Test that zeros are handled consistently between long and matrix formats."""
        # Long format with zeros
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("Sample1.raw\tP67890\tKRAS\t0\t0.001\n")  # Zero value
            f.write("Sample2.raw\tP12345\tEGFR\t2000000\t0.001\n")
            f.write("Sample2.raw\tP67890\tKRAS\t500000\t0.001\n")
            long_path = f.name

        # Matrix format with same data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            f.write("P12345\tEGFR\t1000000\t2000000\n")
            f.write("P67890\tKRAS\t0\t500000\n")  # Zero value
            matrix_path = f.name

        try:
            adata_long, _, _ = parser.parse(long_path, log_transform=False)
            adata_matrix, _, _ = parser.parse(matrix_path, log_transform=False)

            # NaN counts should be consistent
            nan_count_long = np.isnan(adata_long.X).sum()
            nan_count_matrix = np.isnan(adata_matrix.X).sum()

            assert nan_count_long == nan_count_matrix, (
                f"Inconsistent NaN handling: long={nan_count_long}, matrix={nan_count_matrix}"
            )
        finally:
            os.unlink(long_path)
            os.unlink(matrix_path)

    def test_qvalue_greater_than_one(self, parser):
        """Test handling of invalid Q-values > 1.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t1.5\n")  # Invalid Q-value
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t0.01\n")  # Valid
            f.write("Sample2.raw\tP12345\tEGFR\t1500000\t1.5\n")
            f.write("Sample2.raw\tP67890\tKRAS\t2500000\t0.01\n")
            temp_path = f.name
        try:
            # With threshold 1.0, invalid Q-values (>1.0) should still be filtered
            adata, stats, _ = parser.parse(
                temp_path, qvalue_threshold=1.0, log_transform=False
            )
            # Should only have 1 protein (KRAS with valid Q-value)
            assert adata.n_vars == 1
            assert stats["n_qvalue_filtered"] > 0
        finally:
            os.unlink(temp_path)

    def test_duplicate_sample_names_after_cleaning(self, parser):
        """Test Sample1.raw and Sample1.d both become Sample1 - should get unique names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write(
                "Sample1.d\tP12345\tEGFR\t2000000\t0.001\n"
            )  # Becomes "Sample1" after cleaning
            f.write("Sample1.raw\tP67890\tKRAS\t3000000\t0.001\n")
            f.write("Sample1.d\tP67890\tKRAS\t4000000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Should have 2 unique sample names
            assert adata.n_obs == 2
            # All sample names should be unique (no duplicate index)
            assert len(set(adata.obs_names)) == adata.n_obs
        finally:
            os.unlink(temp_path)

    def test_all_samples_nan(self, parser):
        """Test handling when all sample names are NaN (all become 'unknown')."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Note: pandas will read empty string as NaN
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("\tP12345\tEGFR\t2000000\t0.001\n")
            f.write("\tP67890\tKRAS\t3000000\t0.001\n")
            f.write("\tP67890\tKRAS\t4000000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Should have parsed with unique "unknown" sample names
            assert adata.n_obs >= 1
            # All sample names should be unique
            assert len(set(adata.obs_names)) == adata.n_obs
        finally:
            os.unlink(temp_path)


class TestSpectronautParserHighPriorityEdgeCases:
    """High priority edge case tests identified by QA review."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return SpectronautParser()

    def test_negative_intensity_values(self, parser):
        """Test handling of negative intensity values (invalid in MS)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t-1000\t0.001\n")  # Negative value
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t0.001\n")
            f.write("Sample2.raw\tP12345\tEGFR\t1000000\t0.001\n")
            f.write("Sample2.raw\tP67890\tKRAS\t2500000\t0.001\n")
            temp_path = f.name
        try:
            # With log transform, negative values will produce problematic results
            # but should not crash
            adata, _, _ = parser.parse(temp_path, log_transform=True, pseudocount=1.0)
            # Should have parsed without crashing
            assert adata.n_obs == 2
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_infinity_intensity_values(self, parser):
        """Test handling of Inf values in data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            f.write("P12345\tEGFR\tinf\t2000000\n")  # Infinity value
            f.write("P67890\tKRAS\t3000000\t4000000\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Should parse, infinity may be preserved or become NaN
            assert adata.n_obs == 2
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_scientific_notation_intensity(self, parser):
        """Test parsing of intensities in scientific notation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write(
                "Sample1.raw\tP12345\tEGFR\t1.23E+06\t0.001\n"
            )  # Scientific notation
            f.write("Sample1.raw\tP67890\tKRAS\t4.56E+05\t0.001\n")
            f.write("Sample2.raw\tP12345\tEGFR\t2.34E+06\t0.001\n")
            f.write("Sample2.raw\tP67890\tKRAS\t5.67E+05\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Scientific notation should be parsed correctly
            assert adata.n_obs == 2
            assert adata.n_vars == 2
            # Values should be in expected range (>100000)
            assert np.nanmean(adata.X) > 100000
        finally:
            os.unlink(temp_path)

    def test_qvalue_exactly_at_threshold(self, parser):
        """Test Q-value exactly at threshold (boundary condition)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write(
                "Sample1.raw\tP12345\tEGFR\t1000000\t0.01\n"
            )  # Exactly at threshold
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t0.009\n")  # Below threshold
            f.write("Sample2.raw\tP12345\tEGFR\t1500000\t0.01\n")
            f.write("Sample2.raw\tP67890\tKRAS\t2500000\t0.009\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(
                temp_path, qvalue_threshold=0.01, log_transform=False
            )
            # Both proteins should be included (Q-value <= threshold)
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_qvalue_zero(self, parser):
        """Test Q-value of 0.0 (perfect confidence) is handled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t0.0\n")  # Zero Q-value
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t0.001\n")
            f.write("Sample2.raw\tP12345\tEGFR\t1500000\t0.0\n")
            f.write("Sample2.raw\tP67890\tKRAS\t2500000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(
                temp_path, qvalue_threshold=0.01, log_transform=False
            )
            # Both proteins should be included
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_qvalue_all_nan(self, parser):
        """Test handling when Q-value column exists but all values are NaN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\tEGFR\t1000000\t\n")  # NaN Q-value
            f.write("Sample1.raw\tP67890\tKRAS\t2000000\t\n")
            f.write("Sample2.raw\tP12345\tEGFR\t1500000\t\n")
            f.write("Sample2.raw\tP67890\tKRAS\t2500000\t\n")
            temp_path = f.name
        try:
            # Should either parse successfully (skipping filtering) or raise clear error
            # NaN Q-values will be filtered out by >= 0 check
            with pytest.raises(ParsingError):
                parser.parse(temp_path, qvalue_threshold=0.01, log_transform=False)
        finally:
            os.unlink(temp_path)

    def test_sample_name_with_spaces(self, parser):
        """Test sample names with spaces are handled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample 1.raw\tP12345\tEGFR\t1000000\t0.001\n")  # Space in name
            f.write("Sample 1.raw\tP67890\tKRAS\t2000000\t0.001\n")
            f.write("Sample 2.raw\tP12345\tEGFR\t1500000\t0.001\n")
            f.write("Sample 2.raw\tP67890\tKRAS\t2500000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Spaces should be preserved in sample names
            assert adata.n_obs == 2
            sample_names = list(adata.obs["sample_name"])
            assert any(" " in name for name in sample_names)
        finally:
            os.unlink(temp_path)

    def test_protein_id_with_special_chars(self, parser):
        """Test protein IDs with special characters like sp|P12345|EGFR_HUMAN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write(
                "Sample1.raw\tsp|P12345|EGFR_HUMAN\tEGFR\t1000000\t0.001\n"
            )  # UniProt format
            f.write("Sample1.raw\tP67890-2\tKRAS\t2000000\t0.001\n")  # Isoform notation
            f.write("Sample2.raw\tsp|P12345|EGFR_HUMAN\tEGFR\t1500000\t0.001\n")
            f.write("Sample2.raw\tP67890-2\tKRAS\t2500000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            assert adata.n_obs == 2
            assert adata.n_vars == 2
        finally:
            os.unlink(temp_path)

    def test_semicolon_only_protein_groups(self, parser):
        """Test handling of protein groups that are only semicolons."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("PG.ProteinGroups\tPG.Genes\tSample1\tSample2\n")
            f.write("P12345\tEGFR\t1000000\t2000000\n")
            f.write(";;;\tKRAS\t3000000\t4000000\n")  # Only semicolons
            f.write("P67890\tTP53\t5000000\t6000000\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(temp_path, log_transform=False)
            # Should handle gracefully, empty splits become UNKNOWN
            assert adata.n_vars >= 2
        finally:
            os.unlink(temp_path)

    def test_gene_symbols_empty_strings(self, parser):
        """Test handling when gene symbols are all empty strings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345\t\t1000000\t0.001\n")  # Empty gene
            f.write("Sample1.raw\tP67890\t\t2000000\t0.001\n")  # Empty gene
            f.write("Sample2.raw\tP12345\t\t1500000\t0.001\n")
            f.write("Sample2.raw\tP67890\t\t2500000\t0.001\n")
            temp_path = f.name
        try:
            adata, _, _ = parser.parse(
                temp_path, use_genes_as_index=True, log_transform=False
            )
            # Should fall back to protein IDs as index
            assert adata.n_obs == 2
            assert adata.n_vars == 2
            # Index should not be empty or all "UNKNOWN"
            assert all(idx for idx in adata.var_names)
        finally:
            os.unlink(temp_path)

    def test_contaminant_case_sensitivity(self, parser):
        """Test contaminant detection is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tCON__P12345\tKERA\t1000000\t0.001\n")  # Uppercase
            f.write("Sample1.raw\tcon__P67890\tKERB\t2000000\t0.001\n")  # Lowercase
            f.write("Sample1.raw\tP11111\tEGFR\t3000000\t0.001\n")  # Normal protein
            f.write("Sample2.raw\tCON__P12345\tKERA\t1500000\t0.001\n")
            f.write("Sample2.raw\tcon__P67890\tKERB\t2500000\t0.001\n")
            f.write("Sample2.raw\tP11111\tEGFR\t3500000\t0.001\n")
            temp_path = f.name
        try:
            # Without filtering
            adata_unfiltered, _, _ = parser.parse(
                temp_path, filter_contaminants=False, log_transform=False
            )
            # Both uppercase and lowercase should be flagged
            assert adata_unfiltered.var["is_contaminant"].sum() == 2

            # With filtering
            adata_filtered, _, _ = parser.parse(
                temp_path, filter_contaminants=True, log_transform=False
            )
            # Only 1 protein should remain
            assert adata_filtered.n_vars == 1
        finally:
            os.unlink(temp_path)

    def test_reverse_patterns_suffix(self, parser):
        """Test reverse pattern with _rev at end of string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("R.FileName\tPG.ProteinGroups\tPG.Genes\tPG.Quantity\tPG.Qvalue\n")
            f.write("Sample1.raw\tP12345_rev\tGENE1\t1000000\t0.001\n")  # Suffix _rev
            f.write("Sample1.raw\tREV__P67890\tGENE2\t2000000\t0.001\n")  # Prefix REV__
            f.write("Sample1.raw\tP11111\tEGFR\t3000000\t0.001\n")  # Normal protein
            f.write("Sample2.raw\tP12345_rev\tGENE1\t1500000\t0.001\n")
            f.write("Sample2.raw\tREV__P67890\tGENE2\t2500000\t0.001\n")
            f.write("Sample2.raw\tP11111\tEGFR\t3500000\t0.001\n")
            temp_path = f.name
        try:
            # Without filtering
            adata_unfiltered, _, _ = parser.parse(
                temp_path, filter_reverse=False, log_transform=False
            )
            # Both patterns should be flagged
            assert adata_unfiltered.var["is_reverse"].sum() == 2

            # With filtering
            adata_filtered, _, _ = parser.parse(
                temp_path, filter_reverse=True, log_transform=False
            )
            # Only 1 protein should remain
            assert adata_filtered.n_vars == 1
        finally:
            os.unlink(temp_path)
