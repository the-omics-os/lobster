"""A TAR of per-sample 10x file trios is imported as every sample in it, or says which it did not."""

import gzip
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from lobster.services.data_access.geo.archive_processing import ArchiveProcessor
from lobster.services.data_access.geo.parser import GEOParser

pytestmark = pytest.mark.unit

GENES = [f"MT-{i}" if i < 5 else f"G{i}" for i in range(300)]


def _gz(data: bytes) -> bytes:
    return gzip.compress(data)


def _trio(cells: int, seed: int, prefix: str = ""):
    rng = np.random.default_rng(seed)
    counts = sp.csr_matrix(rng.poisson(1.2, (len(GENES), cells)).astype("int32"))
    mtx = io.BytesIO()
    scipy.io.mmwrite(mtx, counts)
    barcodes = "".join(f"AAAC{seed}{i:03d}-1\n" for i in range(cells)).encode()
    features = "".join(f"{g}\t{g}\tGene Expression\n" for g in GENES).encode()
    return {
        f"{prefix}barcodes.tsv.gz": _gz(barcodes),
        f"{prefix}features.tsv.gz": _gz(features),
        f"{prefix}matrix.mtx.gz": _gz(mtx.getvalue()),
    }


def _tar(tmp_path: Path, files: dict, name: str = "GSE1_RAW.tar") -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    with tarfile.open(cache / name, "w") as tar:
        for member, data in files.items():
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return cache


def _process(tmp_path, files, multimodal_info=None):
    cache = _tar(tmp_path, files)
    service = SimpleNamespace(
        cache_dir=cache, geo_parser=GEOParser(), geo_downloader=None
    )
    processor = ArchiveProcessor(service)
    return processor._process_tar_file(
        "http://example.test/x.tar", "GSE1", multimodal_info
    )


def _three(corrupt=None):
    files = {}
    for gsm, tag, cells, seed in (
        ("GSM1", "BM290", 5, 1),
        ("GSM2", "BM474", 7, 2),
        ("GSM3", "BM808", 9, 3),
    ):
        trio = _trio(cells, seed, f"{gsm}_{tag}_")
        files.update(trio)
    if corrupt:
        key = f"GSM2_BM474_{corrupt}"
        files[key] = files[key][: len(files[key]) // 2]  # truncated
    return files


def _not_loaded(result):
    return list(result.uns["geo_import"]["not_loaded"])


def test_three_flat_trios_are_three_samples(tmp_path):
    result = _process(tmp_path, _three())
    assert result is not None and result.n_obs == 5 + 7 + 9
    assert set(result.obs["sample_id"]) == {
        "GSM1_BM290",
        "GSM2_BM474",
        "GSM3_BM808",
    }
    assert result.n_vars == len(GENES)
    assert "geo_import" not in result.uns or not result.uns["geo_import"].get(
        "not_loaded"
    )


def test_a_corrupt_trio_is_named_with_its_reason_and_the_others_load(tmp_path):
    result = _process(tmp_path, _three(corrupt="matrix.mtx.gz"))
    assert result.n_obs == 5 + 9
    (line,) = _not_loaded(result)
    assert line.startswith("GSM2_BM474") and len(line) > len("GSM2_BM474: ")


def test_an_incomplete_trio_is_named(tmp_path):
    files = _three()
    del files["GSM2_BM474_features.tsv.gz"]
    result = _process(tmp_path, files)
    assert result.n_obs == 5 + 9
    (line,) = _not_loaded(result)
    assert line.startswith("GSM2_BM474") and "features" in line


def test_an_unprefixed_single_trio_is_one_sample(tmp_path):
    result = _process(tmp_path, _trio(6, 4))
    assert result.n_obs == 6
    assert set(result.obs["sample_id"]) == {"GSE1"}


def test_nested_sample_archives_are_unchanged(tmp_path):
    files = {}
    for gsm, cells, seed in (("GSM1", 5, 1), ("GSM2", 7, 2)):
        inner = io.BytesIO()
        with tarfile.open(fileobj=inner, mode="w:gz") as tar:
            for member, data in _trio(cells, seed).items():
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        files[f"{gsm}.tar.gz"] = inner.getvalue()
    result = _process(tmp_path, files)
    assert result.n_obs == 12 and set(result.obs["sample_id"]) == {"GSM1", "GSM2"}


def test_the_first_success_fallback_with_several_files_says_which_it_left_out(
    tmp_path,
):
    rng = np.random.default_rng(0)

    def table(rows):
        body = "\n".join(
            f"G{i}\t" + "\t".join(str(v) for v in rng.integers(0, 100000, 60))
            for i in range(rows)
        )
        return ("gene\t" + "\t".join(f"c{i}" for i in range(60)) + "\n" + body).encode()

    big, small = table(2500), table(2000)
    header = ""
    files = {
        "a_counts.txt.gz": _gz(big),
        "b_counts.txt.gz": _gz(small),
    }
    result = _process(tmp_path, files)
    assert result is not None
    (line,) = result.attrs["not_loaded"]
    assert "only" in line and "was loaded" in line
