"""Download file names from remote listings stay inside the download directory."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.utils.safe_paths import safe_download_path


@pytest.mark.parametrize(
    "name,expected",
    [
        ("data.raw", "data.raw"),
        ("../../evil", "evil"),
        ("/etc/passwd", "passwd"),
        ("sub/dir/file.txt", "file.txt"),
        ("..\\..\\win.txt", "win.txt"),
    ],
)
def test_names_are_reduced_to_a_base_name(tmp_path, name, expected):
    path = safe_download_path(tmp_path, name)
    assert path == tmp_path.resolve() / expected
    assert path.parent == tmp_path.resolve()


@pytest.mark.parametrize("name", ["", ".", "..", "../", "/", "a/.."])
def test_unusable_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError):
        safe_download_path(tmp_path, name)


def test_existing_symlink_out_of_the_directory_is_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    inside = tmp_path / "downloads"
    inside.mkdir()
    (inside / "f.txt").symlink_to(outside)
    with pytest.raises(ValueError):
        safe_download_path(inside, "f.txt")


# Each download service keeps ``../../evil`` from a listing inside download_dir.


def _dm(tmp_path):
    dm = Mock()
    del dm.workspace_dir
    dm.workspace_path = tmp_path / "ws"
    dm.cache_dir = tmp_path / "ws" / "cache"
    dm.cache_dir.mkdir(parents=True)
    dm.modalities = {}
    return dm


def _entry(database, dataset_id):
    return Mock(database=database, dataset_id=dataset_id, recommended_strategy=None)


LISTED = [{"url": "ftp://example.org/x/evil", "filename": "../../evil"}]


def _assert_inside(calls, download_dir):
    targets = [Path(c.args[1]) for c in calls]
    assert targets == [download_dir.resolve() / "evil"]


def test_pride_listing_name_stays_inside(tmp_path):
    from lobster.services.data_access.pride_download_service import (
        PRIDEDownloadService,
    )

    service = PRIDEDownloadService(_dm(tmp_path))
    service.pride_provider = Mock()
    service.pride_provider.get_download_urls.return_value = Mock(
        error=None, raw_files=[], processed_files=[], search_files=[]
    )
    service._select_files_by_strategy = Mock(return_value=LISTED)
    service._download_ftp_file = Mock(return_value=None)
    with pytest.raises(RuntimeError):
        service.download_dataset(_entry("pride", "PXD000001"))
    _assert_inside(
        service._download_ftp_file.call_args_list,
        tmp_path / "ws" / "downloads" / "PXD000001",
    )


def test_massive_listing_name_stays_inside(tmp_path):
    from lobster.services.data_access.massive_download_service import (
        MassIVEDownloadService,
    )

    service = MassIVEDownloadService(_dm(tmp_path))
    service._scan_ftp_directory = Mock(return_value=LISTED)
    service._select_files_by_strategy_massive = Mock(return_value=LISTED)
    service._download_ftp_file = Mock(return_value=None)
    with pytest.raises(RuntimeError):
        service.download_dataset(_entry("massive", "MSV000000001"))
    _assert_inside(
        service._download_ftp_file.call_args_list,
        tmp_path / "ws" / "downloads" / "MSV000000001",
    )


def test_metabolights_listing_name_stays_inside(tmp_path):
    from lobster.services.data_access.metabolights_download_service import (
        MetaboLightsDownloadService,
    )

    service = MetaboLightsDownloadService(_dm(tmp_path))
    download_dir = tmp_path / "a" / "b" / "dl"
    download_dir.mkdir(parents=True)
    response = Mock()
    response.iter_content.return_value = [b"data"]
    service.session.get = Mock(return_value=response)

    path = service._download_file(
        "MTBLS1", {"filename": "../../evil"}, download_dir, max_retries=1
    )

    assert path == download_dir.resolve() / "evil"
    assert path.read_bytes() == b"data"
    assert [p for p in tmp_path.rglob("evil")] == [path]


def test_geo_supplementary_name_stays_inside(tmp_path):
    from lobster.services.data_access.geo.downloader import GEODownloadManager

    manager = GEODownloadManager(cache_dir=str(tmp_path / "cache"))
    manager.get_supplementary_files = Mock(
        return_value=["https://example.org/suppl/..%2F..%2Fx_counts.txt.gz"]
    )
    targets = []
    manager.download_file = Mock(side_effect=lambda url, p: targets.append(Path(p)))
    manager.download_supplementary_data("../../GSE1")
    assert targets, "a supplementary file should have been selected"
    assert all(t.parent == (tmp_path / "cache").resolve() for t in targets)
