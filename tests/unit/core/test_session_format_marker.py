"""A session record says whether this version of Lobster wrote it.

The record of a session (``metadata.json`` beside ``provenance.jsonl``) carries a
``record_format`` key from this version on. A record without it was written by an
earlier version: it is "migrated", whatever version string this build reports.
"""

import json
from pathlib import Path

import pytest

from lobster.core.provenance.provenance import ProvenanceTracker

pytestmark = pytest.mark.unit

OLD_VERSIONS = {"scanpy": "1.12.1", "lobster": "1.2.2+release.f32a8be"}


def _old_session(folder: Path, metadata="written") -> Path:
    """A session folder as an earlier version left it (no format key)."""
    folder.mkdir(parents=True)
    tracker = ProvenanceTracker(session_dir=folder / "scratch")
    tracker.create_activity("load_dataset", "a", description="old step")
    line = (folder / "scratch" / "provenance.jsonl").read_text()
    (folder / "provenance.jsonl").write_text(line)
    if metadata == "written":
        (folder / "metadata.json").write_text(
            json.dumps({"created_at": "2026-11-08", "software_versions": OLD_VERSIONS})
        )
    elif metadata == "unreadable":
        (folder / "metadata.json").write_text("{not json")
    return folder


def _written(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


def test_a_fresh_session_writes_the_format_key_and_is_not_migrated(tmp_path):
    tracker = ProvenanceTracker(session_dir=tmp_path / "s")
    tracker.create_activity("tool", "a", description="d")
    assert _written(tmp_path / "s")["record_format"]
    assert tracker.migrated_from() is None


def test_a_native_session_resumed_in_a_new_process_is_not_migrated(tmp_path):
    first = ProvenanceTracker(session_dir=tmp_path / "s")
    first.create_activity("tool", "a", description="d")
    again = ProvenanceTracker(session_dir=tmp_path / "s")
    again.create_activity("tool", "a", description="more")
    assert again.migrated_from() is None
    assert _written(tmp_path / "s")["record_format"]


def test_a_native_session_attached_late_is_not_migrated(tmp_path):
    tracker = ProvenanceTracker()
    tracker.create_activity("tool", "a", description="d")
    tracker.attach_session_dir(tmp_path / "s")
    tracker.create_activity("tool", "a", description="more")
    assert tracker.migrated_from() is None
    assert _written(tmp_path / "s")["record_format"]


def test_an_earlier_record_is_migrated_with_its_source_versions(tmp_path):
    folder = _old_session(tmp_path / "old")
    tracker = ProvenanceTracker(session_dir=folder)
    assert tracker.migrated_from() == {
        "source": "metadata.json",
        "lobster": "1.2.2+release.f32a8be",
        "software_versions": OLD_VERSIONS,
    }
    tracker.create_activity("tool", "a", description="after the migration")
    assert tracker.migrated_from()["lobster"] == "1.2.2+release.f32a8be"
    kept = _written(folder)
    assert "record_format" not in kept
    assert kept["software_versions"] == OLD_VERSIONS


@pytest.mark.parametrize("metadata", ["unreadable", "absent"])
def test_a_record_without_readable_metadata_is_migrated_from_unknown(
    tmp_path, metadata
):
    folder = _old_session(tmp_path / "old", metadata=metadata)
    tracker = ProvenanceTracker(session_dir=folder)
    assert tracker.migrated_from() == {
        "source": "unknown (metadata unreadable)",
        "lobster": None,
        "software_versions": None,
    }


def test_an_earlier_session_restored_into_a_new_one_keeps_the_new_one_migrated(
    tmp_path,
):
    old = _old_session(tmp_path / "old")
    tracker = ProvenanceTracker(session_dir=tmp_path / "new")
    tracker._load_from_disk(old / "provenance.jsonl")
    tracker.create_activity("tool", "a", description="d")
    assert tracker.migrated_from()["lobster"] == "1.2.2+release.f32a8be"
    assert "record_format" not in _written(tmp_path / "new")


def test_a_native_session_restored_into_a_new_one_stays_native(tmp_path):
    first = ProvenanceTracker(session_dir=tmp_path / "a")
    first.create_activity("tool", "a", description="d")
    tracker = ProvenanceTracker(session_dir=tmp_path / "b")
    tracker._load_from_disk(tmp_path / "a" / "provenance.jsonl")
    tracker.create_activity("tool", "a", description="d")
    assert tracker.migrated_from() is None
    assert _written(tmp_path / "b")["record_format"]
