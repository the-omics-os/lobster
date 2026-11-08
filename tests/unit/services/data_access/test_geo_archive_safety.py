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


def _write_tar(path: Path, members, mode: str = "w") -> Path:
    with tarfile.open(path, mode) as tar:
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
