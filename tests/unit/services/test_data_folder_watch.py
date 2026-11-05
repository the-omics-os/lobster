"""The data folder watched by default, and quick fingerprints for big files.
Synthetic files only."""

import os

import pytest

import lobster.services.interaction.component_schemas as schemas
from lobster.services.interaction.component_schemas import (
    FileSnapshot,
    _changed_files,
    data_folder_files,
    fingerprint,
    watch_data_folder,
)


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / ".cache").mkdir()
    (data / "layout.csv").write_text("sample,well\nS1,A1\n")
    (data / "raw" / "counts.tsv").write_text("gene\tS1\nG1\t5\n")
    (data / ".hidden.csv").write_text("x\n")
    (data / ".cache" / "c.bin").write_text("x\n")
    (tmp_path / "notes.txt").write_text("outside data/\n")
    return tmp_path


def test_the_data_folder_is_listed_without_hidden_files(workspace):
    assert data_folder_files(workspace) == ["data/layout.csv", "data/raw/counts.tsv"]


def test_links_are_skipped(workspace):
    os.symlink(workspace / "notes.txt", workspace / "data" / "link.txt")
    assert "data/link.txt" not in data_folder_files(workspace)


def test_the_listing_stops_at_the_limit(workspace):
    for n in range(10):
        (workspace / "data" / f"f{n}.csv").write_text("x\n")
    assert len(data_folder_files(workspace, limit=4)) == 4


def test_no_data_folder_or_an_empty_one_watches_nothing(tmp_path):
    assert watch_data_folder(tmp_path) is None
    (tmp_path / "data").mkdir()
    assert watch_data_folder(tmp_path) is None


def test_a_big_file_is_fingerprinted_by_size_and_time(workspace, monkeypatch):
    monkeypatch.setattr(schemas, "QUICK_FINGERPRINT_BYTES", 10)
    big = workspace / "data" / "layout.csv"  # 18 bytes: "big" here
    (entry,) = fingerprint(["data/layout.csv"], workspace).files
    stat = big.stat()
    assert entry.sha256 is None
    assert entry.quick == f"{stat.st_size} bytes, modified {stat.st_mtime_ns}"
    asked = FileSnapshot(files=[entry])
    assert _changed_files(asked, fingerprint(["data/layout.csv"], workspace)) == []
    os.utime(big, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    now = fingerprint(["data/layout.csv"], workspace)
    assert _changed_files(asked, now) == ["data/layout.csv"]


def test_a_small_file_is_still_hashed(workspace):
    (entry,) = fingerprint(["data/layout.csv"], workspace).files
    assert entry.sha256 and entry.quick is None
