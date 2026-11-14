"""Loading a saved session must not duplicate the provenance the client already holds."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.client import AgentClient
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.provenance.provenance import ProvenanceTracker

SESSION_ID = "idempotent_load"


def _graph():
    metadata = Mock()
    metadata.subscription_tier = "free"
    metadata.available_agents = []
    metadata.supervisor_accessible_agents = []
    metadata.filtered_out_agents = []
    metadata.to_dict.return_value = {
        "subscription_tier": "free",
        "available_agents": [],
        "supervisor_accessible_agents": [],
        "filtered_out_agents": [],
    }
    return Mock(), metadata


def _client(workspace: Path, session_id: str) -> AgentClient:
    with patch(
        "lobster.core.client.create_bioinformatics_graph", return_value=_graph()
    ):
        return AgentClient(workspace_path=workspace, session_id=session_id)


def _step() -> AnalysisStep:
    return AnalysisStep(
        operation="scanpy.pp.normalize_total",
        tool_name="normalize_data",
        description="normalize",
        library="scanpy",
        code_template="sc.pp.normalize_total(adata)",
        imports=["import scanpy as sc"],
        parameters={},
        parameter_schema={},
    )


@pytest.fixture
def saved_session(tmp_path):
    """A workspace holding a saved session with two recorded activities."""
    first = _client(tmp_path, SESSION_ID)
    for i in range(2):
        first.data_manager.provenance.create_activity(
            activity_type=f"step_{i}", agent="agent", ir=_step()
        )
    first._save_session_json()
    path = tmp_path / f"session_{SESSION_ID}.json"
    assert path.exists()
    del first
    return tmp_path, path


def test_load_into_a_new_client_on_the_same_session_does_not_double(saved_session):
    workspace, path = saved_session
    client = _client(workspace, SESSION_ID)
    held = [a["id"] for a in client.data_manager.provenance.activities]
    assert len(held) == 2  # the constructor already restored them from disk

    result = client.load_session(path)

    assert result["provenance_restored"] == 2
    assert [a["id"] for a in client.data_manager.provenance.activities] == held


def test_loading_the_same_session_twice_is_the_same_as_once(saved_session):
    workspace, path = saved_session
    client = _client(workspace, "another_session")
    client.load_session(path)
    once = [a["id"] for a in client.data_manager.provenance.activities]
    assert len(once) == 2

    client.load_session(path)

    assert [a["id"] for a in client.data_manager.provenance.activities] == once


def test_the_activity_file_is_not_rewritten_by_a_load(saved_session):
    workspace, path = saved_session
    file = workspace / ".lobster" / "sessions" / SESSION_ID / "provenance.jsonl"
    before = file.read_text()
    _client(workspace, SESSION_ID).load_session(path)
    assert file.read_text() == before


def test_activities_with_the_same_id_are_kept_once_by_the_tracker(tmp_path):
    source = ProvenanceTracker(session_dir=tmp_path / "a")
    source.create_activity(activity_type="one", agent="agent")
    target = ProvenanceTracker()
    target._load_from_disk(tmp_path / "a" / "provenance.jsonl")
    target._load_from_disk(tmp_path / "a" / "provenance.jsonl")
    assert [a["type"] for a in target.activities] == ["one"]


def test_attaching_a_directory_that_already_holds_the_steps_does_not_repeat_them(
    tmp_path,
):
    directory = tmp_path / "session"
    writer = ProvenanceTracker(session_dir=directory)
    for i in range(2):
        writer.create_activity(activity_type=f"step_{i}", agent="agent")
    file = directory / "provenance.jsonl"

    late = ProvenanceTracker()  # persistence is attached after the steps were loaded
    late._load_from_disk(file)
    late.attach_session_dir(directory)

    assert [a["type"] for a in late.activities] == ["step_0", "step_1"]
    assert len(file.read_text().splitlines()) == 2


def test_attach_still_flushes_steps_the_file_does_not_hold(tmp_path):
    directory = tmp_path / "session"
    late = ProvenanceTracker()
    late.create_activity(activity_type="only_in_memory", agent="agent")

    late.attach_session_dir(directory)

    assert len((directory / "provenance.jsonl").read_text().splitlines()) == 1
    assert [a["type"] for a in ProvenanceTracker(session_dir=directory).activities] == [
        "only_in_memory"
    ]
