"""GEO TAR extraction refuses members that would write outside the target."""

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.services.data_access.geo.archive_processing import (
    ArchiveProcessor,
    _extract_tar_all_or_nothing,
)
from lobster.services.data_access.geo.downloader import GEODownloadError


def _file(name: str, data: bytes = b"x") -> tuple:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, io.BytesIO(data)


def _link(name: str, target: str, hard: bool = False) -> tuple:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _dev(name: str, member_type: bytes = tarfile.CHRTYPE) -> tuple:
    info = tarfile.TarInfo(name)
    info.type = member_type
    return info, None


def _dir(name: str) -> tuple:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _write_tar(
    path: Path, members, mode: str = "w", fmt: int = tarfile.PAX_FORMAT
) -> Path:
    with tarfile.open(path, mode, format=fmt) as tar:
        for info, data in members:
            tar.addfile(info, data)
    return path


# Each case gets the scratch root, so the absolute-path member points into it.
UNSAFE = {
    "parent traversal": lambda root: [_file("ok.txt"), _file("../escaped.txt")],
    "absolute path": lambda root: [_file("ok.txt"), _file(str(root / "abs.txt"))],
    "absolute symlink": lambda root: [_file("ok.txt"), _link("pw", "/etc/passwd")],
    "escaping symlink": lambda root: [_file("ok.txt"), _link("up", "../../out")],
    "escaping hardlink": lambda root: [
        _file("ok.txt"),
        _link("hard", "../out", hard=True),
    ],
    "device file": lambda root: [_file("ok.txt"), _dev("null")],
}


LONG = "GSM1_" + "very_long_sample_name_" * 6 + "matrix.mtx"  # > 100 characters

# Safe layouts that must extract: (members, tar format, files expected afterwards).
ACCEPTED = {
    "dot-slash prefix": (lambda: [_file("./x.txt")], tarfile.PAX_FORMAT, ["x.txt"]),
    "directory entry": (
        lambda: [_dir("GSM1/"), _file("GSM1/x.txt")],
        tarfile.PAX_FORMAT,
        ["GSM1/x.txt"],
    ),
    "hardlink inside": (
        lambda: [_file("ok.txt", b"data"), _link("same.txt", "ok.txt", hard=True)],
        tarfile.PAX_FORMAT,
        ["ok.txt", "same.txt"],
    ),
    "relative symlink inside": (
        lambda: [_file("GSM1/ok.txt"), _link("GSM1/link.txt", "ok.txt")],
        tarfile.PAX_FORMAT,
        ["GSM1/ok.txt", "GSM1/link.txt"],
    ),
    "GNU long name": (lambda: [_file(LONG)], tarfile.GNU_FORMAT, [LONG]),
    "PAX long name": (lambda: [_file(LONG)], tarfile.PAX_FORMAT, [LONG]),
    "spaces": (
        lambda: [_file("GSM1 sample/matrix 1.mtx")],
        tarfile.PAX_FORMAT,
        ["GSM1 sample/matrix 1.mtx"],
    ),
    "percent sign kept literally": (
        lambda: [_file("GSM1%20sample.txt")],
        tarfile.PAX_FORMAT,
        ["GSM1%20sample.txt"],
    ),
}


@pytest.fixture
def layout(tmp_path):
    work = tmp_path / "work"
    dest = work / "dest"
    dest.mkdir(parents=True)
    return tmp_path, work, dest


@pytest.mark.parametrize("case", sorted(UNSAFE))
def test_unsafe_archive_is_refused_and_writes_nothing(layout, case):
    root, work, dest = layout
    archive = _write_tar(root / "a.tar", UNSAFE[case](root))
    with tarfile.open(archive) as tar:
        with pytest.raises(GEODownloadError, match="Archive refused"):
            _extract_tar_all_or_nothing(tar, dest)
    # Not even the safe member is written.
    assert list(dest.iterdir()) == []
    assert not (work / "escaped.txt").exists()
    assert not (root / "abs.txt").exists()


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_accepted_layouts_extract(layout, case):
    root, _, dest = layout
    members, fmt, expected = ACCEPTED[case]
    archive = _write_tar(root / "a.tar", members(), fmt=fmt)
    with tarfile.open(archive) as tar:
        _extract_tar_all_or_nothing(tar, dest)
    for name in expected:
        assert (dest / name).is_file(), name


def test_setuid_bit_is_cleared_not_refused(layout):
    root, _, dest = layout
    info, data = _file("tool.sh", b"#!/bin/sh\n")
    info.mode = 0o4755
    archive = _write_tar(root / "a.tar", [(info, data)])
    with tarfile.open(archive) as tar:
        _extract_tar_all_or_nothing(tar, dest)
    assert not (dest / "tool.sh").stat().st_mode & 0o4000


def test_safe_archive_extracts_everything(layout):
    root, _, dest = layout
    archive = _write_tar(
        root / "a.tar",
        [_file("GSM1/matrix.mtx", b"m"), _file("GSM1/barcodes.tsv", b"b")],
    )
    with tarfile.open(archive) as tar:
        assert _extract_tar_all_or_nothing(tar, dest) == 2
    assert (dest / "GSM1" / "matrix.mtx").read_bytes() == b"m"


def _processor(cache_dir: Path) -> ArchiveProcessor:
    service = MagicMock()
    service.cache_dir = cache_dir
    return ArchiveProcessor(service)


def _no_quant_files():
    return patch(
        "lobster.services.analysis.bulk_rnaseq_service.BulkRNASeqService"
        "._detect_quantification_tool",
        side_effect=ValueError("no quantification files"),
    )


def test_unsafe_nested_sample_archive_is_skipped(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    nested = _write_tar(
        tmp_path / "GSM1_sample.tar.gz",
        [_file("matrix.mtx"), _file("../../nested-escape.txt")],
        mode="w:gz",
    )
    with tarfile.open(cache / "GSE1_RAW.tar", "w") as tar:
        tar.add(nested, arcname=nested.name)

    processor = _processor(cache)
    with _no_quant_files():
        result = processor._process_tar_file("https://example.org/x.tar", "GSE1")

    assert result is None
    processor.service.geo_parser.parse_10x_data.assert_not_called()
    assert not any(tmp_path.rglob("nested-escape.txt"))
    # The refused sample's extraction folder is removed.
    assert not any(cache.rglob("GSM1_sample"))


def test_nameless_nested_archive_does_not_remove_siblings(tmp_path):
    """A refused member named ".tar.gz" must not clear the shared extraction folder."""
    cache = tmp_path / "cache"
    cache.mkdir()
    nameless = _write_tar(
        tmp_path / "nameless.tar.gz", [_file("../../escape.txt")], mode="w:gz"
    )
    good = _write_tar(tmp_path / "GSM2_ok.tar.gz", [_file("matrix.mtx")], mode="w:gz")
    with tarfile.open(cache / "GSE1_RAW.tar", "w") as tar:
        tar.add(nameless, arcname=".tar.gz")
        tar.add(good, arcname=good.name)

    processor = _processor(cache)
    processor.service.geo_parser.parse_10x_data.return_value = None
    with _no_quant_files():
        processor._process_tar_file("https://example.org/x.tar", "GSE1")

    nested_root = cache / "GSE1_nested_extracted"
    sibling = nested_root / "GSM2_ok"
    processor.service.geo_parser.parse_10x_data.assert_called_once_with(
        sibling, "GSM2_ok"
    )
    assert (sibling / "matrix.mtx").is_file()
    assert not (nested_root / "sample_1").exists()
    assert not any(tmp_path.rglob("escape.txt"))


@pytest.mark.parametrize("case", sorted(UNSAFE))
def test_unsafe_outer_raw_tar_is_refused(tmp_path, case):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_tar(cache / "GSE1_RAW.tar", UNSAFE[case](tmp_path))

    processor = _processor(cache)
    with _no_quant_files():
        result = processor._process_tar_file("https://example.org/x.tar", "GSE1")

    assert result is None
    # The failed extraction directory is removed, and nothing escaped.
    assert not (cache / "GSE1_extracted").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert not (cache / "escaped.txt").exists()
    assert not (tmp_path / "abs.txt").exists()
    processor.service.geo_parser.parse_expression_file.assert_not_called()
