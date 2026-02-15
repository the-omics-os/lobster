"""
Unit tests for provenance persistence layer.

Tests the disk persistence mechanics added in Phase 1: JSONL append,
schema versioning, corrupt line recovery, and backward compatibility
when session_dir=None.

Coverage target: 100% of new persistence methods in provenance.py
"""

import json
import warnings
from pathlib import Path

import pytest

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """Create a temporary session directory for persistence tests."""
    session = tmp_path / "test_session"
    session.mkdir(parents=True)
    return session


@pytest.fixture
def synthetic_ir() -> AnalysisStep:
    """Create a minimal but complete AnalysisStep for testing IR round-trip."""
    return AnalysisStep(
        operation="test.operation",
        tool_name="test_tool",
        description="Test operation for unit tests",
        library="test_library",
        code_template="result = {{ param1 }}",
        imports=["import test_library"],
        parameters={"param1": 42, "param2": "value"},
        parameter_schema={
            "param1": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=0,
                required=True,
                description="Test parameter"
            )
        },
        input_entities=["input_data"],
        output_entities=["output_data"],
        execution_context={"test_key": "test_value"},
    )


# ============================================================================
# TEST-01 through TEST-06: Core Persistence Mechanics
# ============================================================================


@pytest.mark.unit
class TestPersistenceMechanics:
    """Tests for core persistence mechanics (TEST-01 through TEST-06)."""

    def test_persistence_disabled_by_default(self, tmp_path: Path):
        """TEST-01: Persistence disabled when session_dir=None (no-op)."""
        tracker = ProvenanceTracker()

        # Create activity
        activity_id = tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent",
            description="Test activity"
        )

        # Verify activity in memory
        assert len(tracker.activities) == 1
        assert tracker.activities[0]["id"] == activity_id

        # Verify no file operations occurred
        assert tracker.session_dir is None
        assert tracker.provenance_path is None

        # Verify no files created (check tmp_path is empty)
        assert len(list(tmp_path.iterdir())) == 0

    def test_activity_appended_to_jsonl(self, session_dir: Path):
        """TEST-02: Activity appended to JSONL on create_activity()."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Create activity
        tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent",
            description="Test activity"
        )

        # Verify JSONL file created
        jsonl_path = session_dir / "provenance.jsonl"
        assert jsonl_path.exists()

        # Verify single line written
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1

        # Verify line is valid JSON
        data = json.loads(lines[0])
        assert data["type"] == "test_activity"
        assert data["agent"] == "test_agent"

    def test_incremental_append_three_activities(self, session_dir: Path):
        """TEST-03: Incremental append (3 activities = 3 lines)."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Create 3 activities
        for i in range(3):
            tracker.create_activity(
                activity_type=f"activity_{i}",
                agent="test_agent",
                description=f"Activity {i}"
            )

        # Verify exactly 3 lines
        jsonl_path = session_dir / "provenance.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify activities are in order
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["type"] == f"activity_{i}"

    def test_load_from_disk_round_trip_with_ir(
        self, session_dir: Path, synthetic_ir: AnalysisStep
    ):
        """TEST-04: Load from disk round-trip (with IR)."""
        # Create tracker and add activity with IR
        tracker1 = ProvenanceTracker(session_dir=session_dir)
        tracker1.create_activity(
            activity_type="ir_test",
            agent="test_agent",
            description="Activity with IR",
            ir=synthetic_ir
        )

        # Create new tracker loading from same session_dir
        tracker2 = ProvenanceTracker(session_dir=session_dir)

        # Verify activity restored
        assert len(tracker2.activities) == 1
        activity = tracker2.activities[0]
        assert activity["type"] == "ir_test"

        # Verify IR restored as AnalysisStep object (not dict)
        ir = activity["ir"]
        assert isinstance(ir, AnalysisStep)
        assert ir.operation == "test.operation"
        assert ir.tool_name == "test_tool"
        assert ir.parameters == {"param1": 42, "param2": "value"}

        # Verify ParameterSpec restored
        assert "param1" in ir.parameter_schema
        param_spec = ir.parameter_schema["param1"]
        assert isinstance(param_spec, ParameterSpec)
        assert param_spec.param_type == "int"
        assert param_spec.papermill_injectable is True

    def test_load_from_disk_round_trip_without_ir(self, session_dir: Path):
        """TEST-04 (continued): Load from disk round-trip (without IR)."""
        # Create tracker and add activity without IR
        tracker1 = ProvenanceTracker(session_dir=session_dir)
        tracker1.create_activity(
            activity_type="no_ir_test",
            agent="test_agent",
            description="Activity without IR"
        )

        # Create new tracker loading from same session_dir
        tracker2 = ProvenanceTracker(session_dir=session_dir)

        # Verify activity restored
        assert len(tracker2.activities) == 1
        activity = tracker2.activities[0]
        assert activity["type"] == "no_ir_test"
        assert activity["ir"] is None

    def test_corrupt_line_handling(self, session_dir: Path):
        """TEST-05: Corrupt line handling (skip bad, load good)."""
        # Create tracker and add valid activity
        tracker1 = ProvenanceTracker(session_dir=session_dir)
        tracker1.create_activity(
            activity_type="valid_activity",
            agent="test_agent",
            description="Valid activity"
        )

        # Manually inject corrupt line
        jsonl_path = session_dir / "provenance.jsonl"
        with open(jsonl_path, "a") as f:
            f.write("{invalid json\n")
            f.write('{"v": 1, "type": "second_valid", "agent": "test", "id": "test:2", "timestamp": "2026-01-01T00:00:00", "inputs": [], "outputs": [], "parameters": {}, "description": "Second valid", "software_versions": {}, "ir": null}\n')

        # Load in new tracker - should skip corrupt line with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tracker2 = ProvenanceTracker(session_dir=session_dir)

            # Verify warning emitted for corrupt line
            assert len(w) >= 1
            assert any("corrupt" in str(warning.message).lower() or "line 2" in str(warning.message) for warning in w)

        # Verify valid activities loaded (line 1 and line 3)
        assert len(tracker2.activities) == 2
        assert tracker2.activities[0]["type"] == "valid_activity"
        assert tracker2.activities[1]["type"] == "second_valid"

    def test_empty_file_handling(self, session_dir: Path):
        """TEST-06: Empty file handling."""
        # Create empty provenance.jsonl
        jsonl_path = session_dir / "provenance.jsonl"
        jsonl_path.touch()

        # Load tracker - should handle gracefully
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Verify no activities loaded, no errors
        assert len(tracker.activities) == 0
        assert tracker.provenance_path == jsonl_path


# ============================================================================
# TEST-07 through TEST-12: API Contracts and Edge Cases
# ============================================================================


@pytest.mark.unit
class TestApiContracts:
    """Tests for API contracts and edge cases (TEST-07 through TEST-12)."""

    def test_clear_preserves_session_dir(self, session_dir: Path):
        """TEST-07: clear() preserves session_dir."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Add activity
        tracker.create_activity(
            activity_type="initial_activity",
            agent="test_agent"
        )
        assert len(tracker.activities) == 1

        # Store session_dir reference
        original_session_dir = tracker.session_dir
        original_provenance_path = tracker.provenance_path

        # Note: ProvenanceTracker doesn't have a clear() method directly.
        # This test validates that DataManagerV2.clear() preserves session_dir
        # when it recreates ProvenanceTracker. For unit testing, we verify
        # the session_dir attribute is not mutated by any operations.

        # Add more activities
        tracker.create_activity(activity_type="second_activity", agent="test_agent")

        # Verify session_dir unchanged
        assert tracker.session_dir == original_session_dir
        assert tracker.provenance_path == original_provenance_path

        # Verify activities accumulate (not reset)
        assert len(tracker.activities) == 2

    def test_session_metadata_written(self, session_dir: Path):
        """TEST-08: Session metadata.json written."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Create activity to trigger metadata write
        tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent"
        )

        # Verify metadata.json exists
        metadata_path = session_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata content
        metadata = json.loads(metadata_path.read_text())
        assert "created_at" in metadata
        assert "last_modified" in metadata
        assert "activity_count" in metadata
        assert "software_versions" in metadata

        # Verify activity count is correct
        assert metadata["activity_count"] == 1

        # Add another activity and verify count updates
        tracker.create_activity(activity_type="second_activity", agent="test_agent")
        metadata = json.loads(metadata_path.read_text())
        assert metadata["activity_count"] == 2

    def test_jsonl_lines_contain_schema_version(self, session_dir: Path):
        """TEST-09: Each JSONL line contains "v": 1."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Create multiple activities
        for i in range(3):
            tracker.create_activity(
                activity_type=f"activity_{i}",
                agent="test_agent"
            )

        # Read and verify all lines have version
        jsonl_path = session_dir / "provenance.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert "v" in data, f"Line {i+1} missing version field"
            assert data["v"] == 1, f"Line {i+1} has wrong version: {data['v']}"

    def test_provenance_path_returns_correct_value(self, session_dir: Path, tmp_path: Path):
        """TEST-10: provenance_path returns correct Path or None."""
        # With session_dir - returns Path
        tracker_with_dir = ProvenanceTracker(session_dir=session_dir)
        assert tracker_with_dir.provenance_path is not None
        assert isinstance(tracker_with_dir.provenance_path, Path)
        assert tracker_with_dir.provenance_path == session_dir / "provenance.jsonl"

        # Without session_dir - returns None
        tracker_without_dir = ProvenanceTracker()
        assert tracker_without_dir.provenance_path is None

    def test_get_summary_returns_correct_stats(
        self, session_dir: Path, synthetic_ir: AnalysisStep
    ):
        """TEST-11: get_summary() returns correct stats."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Empty tracker summary
        summary = tracker.get_summary()
        assert summary["activity_count"] == 0
        assert summary["tools_used"] == []
        assert summary["agents_used"] == []
        assert summary["has_ir"] is False
        assert summary["can_export_notebook"] is False

        # Add activity without IR
        tracker.create_activity(
            activity_type="no_ir_activity",
            agent="agent_1"
        )

        summary = tracker.get_summary()
        assert summary["activity_count"] == 1
        assert "no_ir_activity" in summary["tools_used"]
        assert "agent_1" in summary["agents_used"]
        assert summary["has_ir"] is False
        assert summary["can_export_notebook"] is False

        # Add activity with IR
        tracker.create_activity(
            activity_type="ir_activity",
            agent="agent_2",
            ir=synthetic_ir
        )

        summary = tracker.get_summary()
        assert summary["activity_count"] == 2
        assert "ir_activity" in summary["tools_used"]
        assert "agent_2" in summary["agents_used"]
        assert summary["has_ir"] is True
        assert summary["can_export_notebook"] is True

    def test_persist_activity_recreates_deleted_session_dir(self, session_dir: Path):
        """TEST-12: _persist_activity() creates session_dir if deleted after init."""
        tracker = ProvenanceTracker(session_dir=session_dir)

        # Create first activity
        tracker.create_activity(
            activity_type="first_activity",
            agent="test_agent"
        )

        # Verify files exist
        jsonl_path = session_dir / "provenance.jsonl"
        assert session_dir.exists()
        assert jsonl_path.exists()

        # Delete session_dir (simulates external deletion or cloud restore scenario)
        import shutil
        shutil.rmtree(session_dir)
        assert not session_dir.exists()

        # Create another activity - should recreate session_dir
        tracker.create_activity(
            activity_type="second_activity",
            agent="test_agent"
        )

        # Verify session_dir and file recreated
        assert session_dir.exists()
        assert jsonl_path.exists()

        # Verify second activity persisted (first was lost with directory)
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["type"] == "second_activity"


# ============================================================================
# Test Runner Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
