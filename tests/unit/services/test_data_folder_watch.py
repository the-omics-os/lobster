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


# --- limits: named files in full up to 1 GB, the folder watch above 1 MB quick


def _two_mb(workspace):
    big = workspace / "data" / "matrix.csv"
    big.write_bytes(b"0," * (1024 * 1024))  # 2 MB: above the folder limit
    return big


def test_the_limits_are_1_gb_for_named_files_and_1_mb_for_the_folder():
    assert schemas.QUICK_FINGERPRINT_BYTES == 1024**3
    assert schemas.FOLDER_QUICK_BYTES == 1024**2


def test_the_folder_watch_reads_only_small_files_in_full(workspace):
    _two_mb(workspace)
    entries = {e.path: e for e in watch_data_folder(workspace).files}
    assert (
        entries["data/matrix.csv"].quick and entries["data/matrix.csv"].sha256 is None
    )
    assert (
        entries["data/layout.csv"].sha256 and entries["data/layout.csv"].quick is None
    )


def test_a_named_file_is_read_in_full_whatever_the_folder_limit(workspace):
    _two_mb(workspace)
    (entry,) = fingerprint(["data/matrix.csv"], workspace).files
    assert entry.sha256 and entry.quick is None


def test_a_file_is_read_again_the_way_it_was_recorded(workspace):
    """The folder recorded matrix.csv by size and time: reading it again in
    full would differ every time and make every answer look stale."""
    big = _two_mb(workspace)
    asked = watch_data_folder(workspace)
    assert _changed_files(asked, schemas.refingerprint(asked, workspace)) == []
    with big.open("ab") as handle:
        handle.write(b"1,")
    assert _changed_files(asked, schemas.refingerprint(asked, workspace)) == [
        "data/matrix.csv"
    ]


def test_a_named_file_is_read_again_in_full(workspace):
    asked = fingerprint(["data/layout.csv"], workspace)
    again = schemas.refingerprint(asked, workspace)
    assert again.files[0].sha256 == asked.files[0].sha256
    (workspace / "data" / "layout.csv").write_text("sample,well\nS1,A2\n")
    assert _changed_files(asked, schemas.refingerprint(asked, workspace)) == [
        "data/layout.csv"
    ]


# --- a deleted file in the folder watch is reported once, then forgotten ----


def test_the_folder_watch_is_marked_and_named_files_are_not(workspace):
    assert watch_data_folder(workspace).folder is True
    named = fingerprint(["data/layout.csv"], workspace)
    assert named.folder is False
    assert "folder" not in named.model_dump()  # the wire is unchanged


def test_a_deleted_unrelated_file_blocks_the_answer_only_once(workspace):
    """Review: deleting old_notes.txt made every answer stale, forever."""
    from lobster.services.interaction.component_schemas import (
        DecisionRequest,
        reask,
        refingerprint,
    )

    (workspace / "data" / "old_notes.txt").write_text("old\n")
    request = DecisionRequest(
        question="Which source?",
        fields=[{"key": "s", "type": "choice", "options": ["A", "B"]}],
        snapshot=watch_data_folder(workspace),
    )
    (workspace / "data" / "old_notes.txt").unlink()
    now = refingerprint(request.snapshot, workspace)
    assert _changed_files(request.snapshot, now) == ["data/old_notes.txt"]
    again = reask(request, [], {"s": {"id": "A"}}, snapshot=now)
    assert "data/old_notes.txt" not in [e.path for e in again.snapshot.files]
    assert (
        _changed_files(again.snapshot, refingerprint(again.snapshot, workspace)) == []
    )


def test_a_deleted_named_file_still_stays_stale(workspace):
    """P-11: the decision is about a named file, so it stays stale until back."""
    from lobster.services.interaction.component_schemas import (
        DecisionRequest,
        reask,
        refingerprint,
    )

    request = DecisionRequest(
        question="Which source?",
        fields=[{"key": "s", "type": "choice", "options": ["A", "B"]}],
        snapshot=fingerprint(["data/layout.csv"], workspace),
    )
    (workspace / "data" / "layout.csv").unlink()
    again = reask(
        request, [], None, snapshot=refingerprint(request.snapshot, workspace)
    )
    assert _changed_files(again.snapshot, refingerprint(again.snapshot, workspace)) == [
        "data/layout.csv"
    ]
