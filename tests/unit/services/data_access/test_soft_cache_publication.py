"""SOFT cache correctness, not GEO candidate selection or scientific validation."""

import gzip
import io
import os
from unittest.mock import Mock

import pytest

from lobster.services.data_access.geo.soft_download import pre_download_soft_file

GEO_ID = "GSE12345"


def soft_bytes(geo_id=GEO_ID):
    entity = "SAMPLE" if geo_id.startswith("GSM") else "SERIES"
    text = (
        f"^{entity} = {geo_id}\n"
        f"!{entity.title()}_title = synthetic metadata\n"
        f"!{entity.title()}_geo_accession = {geo_id}\n"
    )
    return gzip.compress(text.encode(), mtime=0)


class Response(io.BytesIO):
    def __init__(self, body, final_path):
        super().__init__(body)
        self.final_path = final_path
        self.final_visible_during_reads = []

    def read(self, *args):
        self.final_visible_during_reads.append(self.final_path.exists())
        return super().read(*args)


def fake_download(monkeypatch, final_path, body):
    response = Response(body, final_path)
    opener = Mock(return_value=response)
    monkeypatch.setattr(
        "lobster.services.data_access.geo.soft_download.urllib.request.urlopen",
        opener,
    )
    return opener, response


@pytest.mark.parametrize(
    "cached",
    [
        b"",
        b"not gzip",
        gzip.compress(b"not SOFT"),
        soft_bytes()[:12],
        soft_bytes("GSE99999"),
    ],
    ids=["empty", "raw", "non-soft-gzip", "truncated-gzip", "wrong-accession"],
)
def test_invalid_existing_cache_is_discarded_and_refetched(
    tmp_path, monkeypatch, cached
):
    final = tmp_path / f"{GEO_ID}_family.soft.gz"
    final.write_bytes(cached)
    valid = soft_bytes()
    opener, _ = fake_download(monkeypatch, final, valid)
    result = pre_download_soft_file(GEO_ID, tmp_path)
    opener.assert_called_once()
    assert result == final
    assert final.read_bytes() == valid
    assert list(tmp_path.iterdir()) == [final]


@pytest.mark.parametrize("geo_id", [GEO_ID, "GSM12345"])
def test_valid_cache_hit_does_not_contact_network(tmp_path, monkeypatch, geo_id):
    final = tmp_path / f"{geo_id}_family.soft.gz"
    valid = soft_bytes(geo_id)
    final.write_bytes(valid)
    opener = Mock(side_effect=AssertionError("valid cache must be reused"))
    monkeypatch.setattr(
        "lobster.services.data_access.geo.soft_download.urllib.request.urlopen",
        opener,
    )
    assert pre_download_soft_file(geo_id, tmp_path) == final
    assert final.read_bytes() == valid
    opener.assert_not_called()


def test_interrupted_read_leaves_no_final_or_temporary_cache(tmp_path, monkeypatch):
    final = tmp_path / f"{GEO_ID}_family.soft.gz"

    class Interrupted(Response):
        def read(self, *args):
            raise OSError("synthetic interrupted response")

    opener = Mock(return_value=Interrupted(soft_bytes(), final))
    monkeypatch.setattr(
        "lobster.services.data_access.geo.soft_download.urllib.request.urlopen",
        opener,
    )
    assert pre_download_soft_file(GEO_ID, tmp_path) is None
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not gzip",
        gzip.compress(b"not SOFT"),
        soft_bytes()[:-8],
        soft_bytes("GSE99999"),
        gzip.compress(b"^SERIES GSE12345\n!Series_title = malformed\n"),
    ],
    ids=[
        "empty",
        "raw",
        "non-soft-gzip",
        "missing-gzip-footer",
        "wrong-accession",
        "malformed-soft",
    ],
)
def test_invalid_response_is_never_published(tmp_path, monkeypatch, body):
    final = tmp_path / f"{GEO_ID}_family.soft.gz"
    fake_download(monkeypatch, final, body)
    assert pre_download_soft_file(GEO_ID, tmp_path) is None
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


def test_final_cache_is_not_visible_until_response_is_complete(tmp_path, monkeypatch):
    final = tmp_path / f"{GEO_ID}_family.soft.gz"
    valid = soft_bytes()
    _, response = fake_download(monkeypatch, final, valid)
    assert pre_download_soft_file(GEO_ID, tmp_path) == final
    assert response.final_visible_during_reads
    assert not any(response.final_visible_during_reads)
    assert final.read_bytes() == valid
    assert list(tmp_path.iterdir()) == [final]


def test_atomic_replace_failure_returns_no_false_success_and_cleans_temp(
    tmp_path, monkeypatch
):
    final = tmp_path / f"{GEO_ID}_family.soft.gz"
    fake_download(monkeypatch, final, soft_bytes())
    original_replace = os.replace
    refused = []

    def fail_cache_replace(source, destination, *args, **kwargs):
        if os.fspath(destination) == os.fspath(final):
            refused.append(source)
            raise OSError("synthetic cache rename failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_cache_replace)
    assert pre_download_soft_file(GEO_ID, tmp_path) is None
    assert len(refused) == 1
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []
