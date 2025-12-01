"""
Unit tests for DownloadUrlResult schema.

Tests comprehensive URL handling, conversion, and backward compatibility
across all supported database formats (GEO, SRA, PRIDE, MassIVE).
"""

import pytest

from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult


class TestDownloadFile:
    """Test DownloadFile model."""

    def test_basic_creation(self):
        """Test basic file creation."""
        f = DownloadFile(
            url="ftp://example.com/file.txt",
            filename="file.txt",
        )
        assert f.url == "ftp://example.com/file.txt"
        assert f.filename == "file.txt"
        assert f.size_bytes is None
        assert f.checksum is None

    def test_full_creation(self):
        """Test file creation with all fields."""
        f = DownloadFile(
            url="ftp://example.com/data.fastq.gz",
            filename="data.fastq.gz",
            size_bytes=1024000,
            checksum="abc123def456",
            checksum_type="md5",
            file_type="raw",
            description="Raw FASTQ file",
        )
        assert f.size_bytes == 1024000
        assert f.checksum == "abc123def456"
        assert f.checksum_type == "md5"
        assert f.file_type == "raw"

    def test_to_simple_url(self):
        """Test URL extraction."""
        f = DownloadFile(
            url="ftp://example.com/file.txt",
            filename="file.txt",
        )
        assert f.to_simple_url() == "ftp://example.com/file.txt"

    def test_to_dict(self):
        """Test dictionary conversion."""
        f = DownloadFile(
            url="ftp://example.com/file.txt",
            filename="file.txt",
            size_bytes=1024,
            checksum="abc123",
            checksum_type="md5",
        )
        d = f.to_dict()
        assert d["url"] == "ftp://example.com/file.txt"
        assert d["filename"] == "file.txt"
        assert d["size"] == 1024
        assert d["md5"] == "abc123"


class TestDownloadUrlResult:
    """Test DownloadUrlResult model."""

    def test_basic_creation(self):
        """Test minimal result creation."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
        )
        assert result.accession == "GSE12345"
        assert result.database == "geo"
        assert result.primary_files == []
        assert result.raw_files == []

    def test_with_files(self):
        """Test result with multiple file categories."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            primary_files=[
                DownloadFile(url="ftp://a/matrix.txt", filename="matrix.txt")
            ],
            raw_files=[
                DownloadFile(url="ftp://a/raw1.gz", filename="raw1.gz"),
                DownloadFile(url="ftp://a/raw2.gz", filename="raw2.gz"),
            ],
            supplementary_files=[
                DownloadFile(url="ftp://a/supp.xlsx", filename="supp.xlsx")
            ],
        )
        assert len(result.primary_files) == 1
        assert len(result.raw_files) == 2
        assert len(result.supplementary_files) == 1

    def test_get_all_files(self):
        """Test getting all files."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            primary_files=[
                DownloadFile(url="ftp://a/matrix.txt", filename="matrix.txt")
            ],
            raw_files=[
                DownloadFile(url="ftp://a/raw.gz", filename="raw.gz"),
            ],
        )
        all_files = result.get_all_files()
        assert len(all_files) == 2

    def test_get_all_urls(self):
        """Test getting all URLs."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            primary_files=[
                DownloadFile(url="ftp://a/matrix.txt", filename="matrix.txt")
            ],
            raw_files=[
                DownloadFile(url="ftp://a/raw.gz", filename="raw.gz"),
            ],
        )
        urls = result.get_all_urls()
        assert "ftp://a/matrix.txt" in urls
        assert "ftp://a/raw.gz" in urls

    def test_get_total_size(self):
        """Test total size calculation."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            raw_files=[
                DownloadFile(url="ftp://a/1.gz", filename="1.gz", size_bytes=100),
                DownloadFile(url="ftp://a/2.gz", filename="2.gz", size_bytes=200),
            ],
        )
        assert result.get_total_size() == 300

    def test_get_total_size_uses_stored(self):
        """Test that stored total_size_bytes is preferred."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            total_size_bytes=999,
            raw_files=[
                DownloadFile(url="ftp://a/1.gz", filename="1.gz", size_bytes=100),
            ],
        )
        assert result.get_total_size() == 999

    def test_to_queue_entry_fields(self):
        """Test conversion to DownloadQueueEntry fields."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            primary_files=[
                DownloadFile(url="ftp://a/matrix.txt", filename="matrix.txt")
            ],
            raw_files=[
                DownloadFile(url="ftp://a/raw.gz", filename="raw.gz"),
            ],
            supplementary_files=[
                DownloadFile(url="ftp://a/supp.xlsx", filename="supp.xlsx"),
            ],
            ftp_base="ftp://a/",
            total_size_bytes=1024,
        )
        fields = result.to_queue_entry_fields()

        assert fields["dataset_id"] == "GSE12345"
        assert fields["database"] == "geo"
        assert fields["matrix_url"] == "ftp://a/matrix.txt"
        assert fields["raw_urls"] == ["ftp://a/raw.gz"]
        assert fields["supplementary_urls"] == ["ftp://a/supp.xlsx"]
        assert fields["ftp_base"] == "ftp://a/"
        assert fields["download_size_bytes"] == 1024

    def test_to_queue_entry_fields_h5_detection(self):
        """Test H5AD URL detection."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            primary_files=[
                DownloadFile(url="ftp://a/data.h5ad", filename="data.h5ad")
            ],
        )
        fields = result.to_queue_entry_fields()
        assert fields["h5_url"] == "ftp://a/data.h5ad"


class TestFromSRAResponse:
    """Test conversion from SRA provider response."""

    def test_basic_conversion(self):
        """Test basic SRA response conversion."""
        response = {
            "accession": "SRP123456",
            "database": "sra",
            "raw_urls": [
                {
                    "url": "ftp://ena/SRR001.fastq.gz",
                    "filename": "SRR001.fastq.gz",
                    "size": 1024000,
                    "md5": "abc123",
                }
            ],
            "ftp_base": "ftp://ena/",
            "total_size_bytes": 1024000,
            "mirror": "ena",
            "layout": "PAIRED",
            "platform": "ILLUMINA",
            "run_count": 1,
        }

        result = DownloadUrlResult.from_sra_response(response)

        assert result.accession == "SRP123456"
        assert result.database == "sra"
        assert len(result.raw_files) == 1
        assert result.raw_files[0].url == "ftp://ena/SRR001.fastq.gz"
        assert result.raw_files[0].checksum == "abc123"
        assert result.mirror == "ena"
        assert result.layout == "PAIRED"

    def test_handles_string_urls(self):
        """Test handling simple string URLs."""
        response = {
            "accession": "SRP123456",
            "database": "sra",
            "raw_urls": ["ftp://ena/file1.gz", "ftp://ena/file2.gz"],
        }

        result = DownloadUrlResult.from_sra_response(response)
        assert len(result.raw_files) == 2


class TestFromGEOResponse:
    """Test conversion from GEO provider response."""

    def test_basic_conversion(self):
        """Test basic GEO response conversion."""
        response = {
            "geo_id": "GSE12345",  # GEO provider uses geo_id key
            "matrix_url": "ftp://geo/matrix.txt.gz",
            "h5_url": "ftp://geo/data.h5",
            "raw_urls": ["ftp://geo/raw1.gz", "ftp://geo/raw2.gz"],
            "supplementary_urls": ["ftp://geo/supp.xlsx"],
            "ftp_base": "ftp://geo/",
        }

        result = DownloadUrlResult.from_geo_response(response)

        assert result.accession == "GSE12345"
        assert result.database == "geo"
        assert len(result.primary_files) == 2  # matrix + h5
        assert len(result.raw_files) == 2
        assert len(result.supplementary_files) == 1
        assert result.recommended_strategy == "matrix"


class TestFromPRIDEResponse:
    """Test conversion from PRIDE provider response."""

    def test_basic_conversion(self):
        """Test basic PRIDE response conversion."""
        response = {
            "accession": "PXD012345",
            "raw_urls": [
                {"downloadLink": "ftp://pride/raw.raw", "fileName": "raw.raw"},
            ],
            "processed_urls": [
                {"downloadLink": "ftp://pride/result.txt", "fileName": "result.txt"},
            ],
        }

        result = DownloadUrlResult.from_pride_response(response)

        assert result.accession == "PXD012345"
        assert result.database == "pride"
        assert len(result.raw_files) == 1
        assert len(result.processed_files) == 1
        assert result.raw_files[0].url == "ftp://pride/raw.raw"

    def test_handles_result_files(self):
        """Test that result_files are merged into processed_files."""
        response = {
            "accession": "PXD012345",
            "raw_urls": [{"url": "ftp://pride/raw.raw", "filename": "raw.raw"}],
            "result_files": [
                {"url": "ftp://pride/result.txt", "filename": "result.txt"}
            ],
        }

        result = DownloadUrlResult.from_pride_response(response)
        assert len(result.processed_files) == 1
        assert result.processed_files[0].filename == "result.txt"


class TestFromMassIVEResponse:
    """Test conversion from MassIVE provider response."""

    def test_basic_conversion(self):
        """Test basic MassIVE response conversion."""
        response = {
            "accession": "MSV000012345",
            "ftp_base": "ftp://massive/MSV000012345/",
            "raw_urls": [
                {"url": "ftp://massive/raw.raw", "filename": "raw.raw"},
            ],
            "result_files": [
                {"url": "ftp://massive/result.txt", "filename": "result.txt"},
            ],
        }

        result = DownloadUrlResult.from_massive_response(response)

        assert result.accession == "MSV000012345"
        assert result.database == "massive"
        assert len(result.raw_files) == 1
        assert len(result.processed_files) == 1


class TestToLegacyDict:
    """Test backward compatibility conversion."""

    def test_legacy_dict_format(self):
        """Test legacy dictionary output."""
        result = DownloadUrlResult(
            accession="GSE12345",
            database="geo",
            raw_files=[
                DownloadFile(
                    url="ftp://a/raw.gz",
                    filename="raw.gz",
                    size_bytes=100,
                    checksum="abc",
                )
            ],
            mirror="ncbi",
            layout="SINGLE",
        )

        legacy = result.to_legacy_dict()

        assert legacy["accession"] == "GSE12345"
        assert legacy["database"] == "geo"
        assert len(legacy["raw_urls"]) == 1
        assert legacy["raw_urls"][0]["url"] == "ftp://a/raw.gz"
        assert legacy["mirror"] == "ncbi"
        assert legacy["layout"] == "SINGLE"
