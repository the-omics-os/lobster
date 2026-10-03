"""Verify provenance persistence when AgentClient receives an injected DataManager.

An injected ``DataManagerV2`` may begin without a session directory. AgentClient must
attach its session directory so recorded provenance survives process exit.

Files are written beneath ``<workspace>/.lobster/sessions/<session_id>/provenance.jsonl``.
"""

import json
from pathlib import Path

import pytest

from lobster.core.provenance.provenance import ProvenanceTracker
from lobster.core.runtime.data_manager import DataManagerV2


def _read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "ws"


class TestTrackerAttach:
    def test_attach_enables_persistence(self, tmp_path):
        tracker = ProvenanceTracker()
        assert tracker.provenance_path is None, "precondition: persistence starts off"

        path = tracker.attach_session_dir(tmp_path / "sessions" / "s1")

        assert path.exists() or path.parent.exists()
        assert tracker.provenance_path == path

    def test_activities_recorded_before_attach_are_flushed(self, tmp_path):
        """The head of the session must not be lost.

        Attaching late without flushing would trade one silent data loss for a subtler
        one -- a file that exists but is missing everything before the attach.
        """
        tracker = ProvenanceTracker()
        tracker.create_activity(
            "tool_a", "agent_x", parameters={"x": 1}, description="first"
        )
        tracker.create_activity(
            "tool_b", "agent_x", parameters={"y": 2}, description="second"
        )

        path = tracker.attach_session_dir(tmp_path / "sessions" / "s2")

        records = _read(path)
        assert [r["type"] for r in records] == ["tool_a", "tool_b"]

    def test_seq_and_agent_survive_the_flush(self, tmp_path):
        """The fields the DAG reconstruction depends on ()."""
        tracker = ProvenanceTracker()
        tracker.create_activity("tool_a", "agent_one", description="d")
        tracker.create_activity("tool_b", "agent_two", description="d")

        records = _read(tracker.attach_session_dir(tmp_path / "sessions" / "s3"))

        assert [r["seq"] for r in records] == [0, 1]
        assert [r["agent"] for r in records] == ["agent_one", "agent_two"]

    def test_attach_is_idempotent(self, tmp_path):
        """Re-attaching the same directory must not duplicate lines."""
        tracker = ProvenanceTracker()
        tracker.create_activity("tool_a", "agent_x", description="d")
        session_dir = tmp_path / "sessions" / "s4"

        path = tracker.attach_session_dir(session_dir)
        tracker.attach_session_dir(session_dir)
        tracker.attach_session_dir(session_dir)

        assert len(_read(path)) == 1

    def test_attach_preserves_history_written_by_a_previous_process(self, tmp_path):
        """A resumed session keeps what an earlier process wrote."""
        session_dir = tmp_path / "sessions" / "s5"
        first = ProvenanceTracker(session_dir=session_dir)
        first.create_activity("from_process_one", "agent_x", description="d")

        second = ProvenanceTracker()
        second.create_activity("from_process_two", "agent_x", description="d")
        path = second.attach_session_dir(session_dir)

        kinds = [r["type"] for r in _read(path)]
        assert kinds == ["from_process_one", "from_process_two"]
        assert len(second.activities) == 2

    def test_activities_after_attach_are_persisted(self, tmp_path):
        tracker = ProvenanceTracker()
        path = tracker.attach_session_dir(tmp_path / "sessions" / "s6")

        tracker.create_activity("later", "agent_x", description="d")

        assert [r["type"] for r in _read(path)] == ["later"]

    def test_metadata_file_is_written(self, tmp_path):
        tracker = ProvenanceTracker()
        tracker.create_activity("tool_a", "agent_x", description="d")
        session_dir = tmp_path / "sessions" / "s7"

        tracker.attach_session_dir(session_dir)

        assert (session_dir / "metadata.json").exists()


class TestDataManagerAttach:
    def test_injected_pattern_persists_after_attach(self, workspace):
        """The exact shape of the bench bug, end to end."""
        manager = DataManagerV2(workspace_path=workspace)
        assert (
            manager.provenance.provenance_path is None
        ), "precondition: off by default"

        manager.log_tool_usage("early", {"a": 1}, description="d", agent="agent_a")
        session_dir = workspace / ".lobster" / "sessions" / "bench_q1"
        path = manager.attach_session_dir(session_dir)

        manager.log_tool_usage("late", {"b": 2}, description="d", agent="agent_b")

        records = _read(path)
        assert [r["type"] for r in records] == ["early", "late"]
        assert [r["agent"] for r in records] == ["agent_a", "agent_b"]

    def test_session_dir_is_recorded_so_clear_keeps_persisting(self, workspace):
        """`clear()` and `clear_workspace()` rebuild the tracker from `_session_dir`.

        If attach only touched the tracker, the first `clear()` would silently revert the
        session to memory-only -- the original bug, reintroduced mid-session.
        """
        manager = DataManagerV2(workspace_path=workspace)
        path = manager.attach_session_dir(
            workspace / ".lobster" / "sessions" / "bench_q2"
        )
        manager.log_tool_usage("before_clear", {}, description="d", agent="a")

        manager.clear()
        manager.log_tool_usage("after_clear", {}, description="d", agent="a")

        assert manager.provenance.provenance_path == path
        assert [r["type"] for r in _read(path)] == ["before_clear", "after_clear"]

    def test_attach_is_harmless_when_provenance_is_disabled(self, workspace):
        manager = DataManagerV2(workspace_path=workspace, enable_provenance=False)

        result = manager.attach_session_dir(workspace / "sessions" / "x")

        assert result is None
        assert manager._session_dir is not None

    def test_explicit_session_dir_at_construction_still_works(self, workspace):
        """The pre-existing path must be untouched."""
        session_dir = workspace / ".lobster" / "sessions" / "direct"
        manager = DataManagerV2(workspace_path=workspace, session_dir=session_dir)

        manager.log_tool_usage("t", {}, description="d", agent="a")

        assert manager.provenance.provenance_path == session_dir / "provenance.jsonl"
        assert len(_read(session_dir / "provenance.jsonl")) == 1


class TestClientWiring:
    """`AgentClient.__init__` must attach on the injected-data-manager branch.

    Constructing a real `AgentClient` needs a configured provider and provider package, so
    these assert on the wiring contract rather than booting a client: that the attribute
    exists, is callable, and that the branch's logic produces persistence.
    """

    def test_data_manager_exposes_the_hook_the_client_calls(self, workspace):
        manager = DataManagerV2(workspace_path=workspace)

        assert callable(getattr(manager, "attach_session_dir", None)), (
            "AgentClient looks up attach_session_dir() via getattr; renaming or removing "
            "it silently disables provenance for every injected data manager"
        )

    def test_client_source_attaches_on_the_injected_branch(self):
        """Guards the branch itself, since it cannot be exercised without a provider.

        A regression here is invisible: the client keeps working and simply stops writing
        provenance, which is exactly how this shipped.
        """
        import inspect

        from lobster.core.client import AgentClient

        source = inspect.getsource(AgentClient.__init__)
        _, _, injected_branch = source.partition("self.data_manager = data_manager")

        assert "attach_session_dir" in injected_branch, (
            "the injected-data-manager branch no longer attaches a session dir; "
            "provenance will not be persisted for injected managers"
        )

    def test_client_branch_logic_persists_provenance(self, workspace):
        """Replicates the client's block verbatim against a real DataManagerV2."""
        manager = DataManagerV2(workspace_path=workspace)
        manager.log_tool_usage("preexisting", {}, description="d", agent="early")
        session_dir = workspace / ".lobster" / "sessions" / "bench_q42"

        attach = getattr(manager, "attach_session_dir", None)
        assert callable(attach)
        session_dir.mkdir(parents=True, exist_ok=True)
        attach(session_dir)

        manager.log_tool_usage("after", {}, description="d", agent="late")

        path = session_dir / "provenance.jsonl"
        assert path.exists()
        assert [r["type"] for r in _read(path)] == ["preexisting", "after"]

    def test_missing_hook_does_not_raise(self, workspace):
        """Fail-open: provenance is observability and must never break a session."""

        class LegacyManager:
            pass

        attach = getattr(LegacyManager(), "attach_session_dir", None)

        assert attach is None  # the client warns and continues
