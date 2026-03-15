"""Tests for archive format detection in geo_service.

Verifies that ARCHIVE_EXTENSIONS constant and _is_archive_url helper
correctly identify .tar, .tar.gz, .tgz, and .tar.bz2 archive URLs.
"""

import pytest

from lobster.services.data_access.geo_service import ARCHIVE_EXTENSIONS, _is_archive_url


class TestArchiveExtensionsConstant:
    """Tests for the ARCHIVE_EXTENSIONS module-level constant."""

    def test_archive_extensions_is_tuple(self):
        assert isinstance(ARCHIVE_EXTENSIONS, tuple)

    def test_archive_extensions_contains_tar(self):
        assert ".tar" in ARCHIVE_EXTENSIONS

    def test_archive_extensions_contains_tar_gz(self):
        assert ".tar.gz" in ARCHIVE_EXTENSIONS

    def test_archive_extensions_contains_tgz(self):
        assert ".tgz" in ARCHIVE_EXTENSIONS

    def test_archive_extensions_contains_tar_bz2(self):
        assert ".tar.bz2" in ARCHIVE_EXTENSIONS

    def test_archive_extensions_has_four_entries(self):
        assert len(ARCHIVE_EXTENSIONS) == 4


class TestIsArchiveUrl:
    """Tests for the _is_archive_url helper function."""

    def test_tar_url_is_archive(self):
        assert _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.tar") is True

    def test_tar_gz_url_is_archive(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.tar.gz") is True
        )

    def test_tgz_url_is_archive(self):
        assert _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.tgz") is True

    def test_tar_bz2_url_is_archive(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.tar.bz2") is True
        )

    def test_txt_gz_is_not_archive(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_counts.txt.gz")
            is False
        )

    def test_csv_is_not_archive(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_data.csv") is False
        )

    def test_h5ad_is_not_archive(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_data.h5ad") is False
        )

    def test_case_insensitive_tar(self):
        assert _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.TAR") is True

    def test_case_insensitive_tar_gz(self):
        assert (
            _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.TAR.GZ") is True
        )

    def test_case_insensitive_tgz(self):
        assert _is_archive_url("https://ftp.ncbi.nlm.nih.gov/GSE12345_RAW.TGZ") is True
