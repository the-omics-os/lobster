"""A session restored in another process still says which file it loaded.

Entities are not written to disk; the load step is, and it keeps the file's name,
SHA-256 and size, so the restored session names the same file.
"""

import hashlib

import numpy as np
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.runtime.data_manager import DataManagerV2

pytestmark = pytest.mark.unit


def _session(tmp_path):
    import anndata as ad

    file = tmp_path / "in.h5ad"
    ad.AnnData(
        np.random.default_rng(1).poisson(1, (20, 10)).astype("float32")
    ).write_h5ad(file)
    session_dir = tmp_path / "ws" / ".lobster" / "sessions" / "s1"
    first = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    first.attach_session_dir(session_dir)
    first.load_modality(
        "m", file, adapter="transcriptomics_single_cell", validate=False
    )
    return file, session_dir, first


def _loaded(manager):
    return NotebookExporter(manager.provenance, manager)._get_input_roles({})["loaded"]


def test_a_restored_session_names_the_file_it_loaded(tmp_path):
    file, session_dir, _first = _session(tmp_path)
    second = DataManagerV2(workspace_path=tmp_path / "ws2", auto_scan=False)
    second.provenance._load_from_disk(session_dir / "provenance.jsonl")
    (entry,) = _loaded(second)
    assert entry["name"] == "in.h5ad" and entry["modality"] == "m"
    assert entry["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert entry["size"] == file.stat().st_size and isinstance(entry["size"], int)


def test_the_same_session_does_not_gain_a_second_entity(tmp_path):
    _file, session_dir, first = _session(tmp_path)
    before = len(first.provenance.entities)
    first.provenance._load_from_disk(session_dir / "provenance.jsonl")
    assert len(first.provenance.entities) == before
    assert len(_loaded(first)) == 1


def test_a_load_that_names_no_file_restores_no_entity(tmp_path):
    session_dir = tmp_path / "s"
    first = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    first.attach_session_dir(session_dir)
    first.log_tool_usage("load_dataset", {"name": "m", "source": "AnnData object"})
    second = DataManagerV2(workspace_path=tmp_path / "ws2", auto_scan=False)
    second.provenance._load_from_disk(session_dir / "provenance.jsonl")
    assert _loaded(second) == []


def test_a_restored_file_import_step_brings_back_the_file_it_read():
    """import_bulk_counts reads its table itself; its step keeps the file's
    record, and a later process lists that file as loaded, under the modality."""
    from lobster.core.provenance.provenance import ProvenanceTracker

    file = {"name": "counts.csv", "sha256": "a" * 64, "size": "1234"}
    tracker = ProvenanceTracker()
    activity = {
        "type": "import_bulk_counts",
        "parameters": {"modality_name": "bulk", "loaded_file": file},
    }
    tracker._entity_of_loaded_file(activity)
    tracker._entity_of_loaded_file(activity)  # restored twice: one entity

    entities = [
        e["metadata"]
        for e in tracker.entities.values()
        if (e.get("metadata") or {}).get("loaded_file")
    ]
    assert entities == [
        {
            "modality_name": "bulk",
            "adapter": None,
            "loaded_file": {**file, "size": 1234},
        }
    ]
