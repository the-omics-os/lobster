"""
Integration tests for Two-Tier State lifecycle through DataManagerV2.

Tests cover:
- State persistence across DataManagerV2 instances
- Request lifecycle through DataManagerV2
- Session state accumulation across multiple requests
- .session_state.json file format and content
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2


class TestStateLifecycleThroughDataManager:
    """Integration tests for state management via DataManagerV2."""

    def test_data_manager_has_state_manager(self):
        """Test that DataManagerV2 initializes with StateManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            assert dm._state_manager is not None
            assert dm._session_state is None  # Lazy loaded
            assert dm._request_state is None  # No active request

    def test_data_manager_start_request(self):
        """Test starting a request through DataManagerV2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            request = dm.start_request(request_id="req-123", source="test")

            assert request is not None
            assert request.request_id == "req-123"
            assert request.source == "test"
            assert dm.request_state is request

    def test_data_manager_request_state_property(self):
        """Test request_state property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # No request yet
            assert dm.request_state is None

            # Start request
            dm.start_request()
            assert dm.request_state is not None

    def test_data_manager_end_request(self):
        """Test ending a request through DataManagerV2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.start_request(request_id="req-123")
            dm.request_state.start_step("test_step")
            time.sleep(0.01)
            dm.request_state.end_step("test_step")

            summary = dm.end_request()

            assert summary["request_id"] == "req-123"
            assert summary["total_duration_ms"] >= 10
            assert dm.request_state is None

    def test_data_manager_session_state_lazy_load(self):
        """Test that session_state is lazily loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # Not loaded yet
            assert dm._session_state is None

            # Access triggers load
            session = dm.session_state
            assert session is not None
            assert dm._session_state is session

    def test_data_manager_init_session_state(self):
        """Test initializing session state with objective."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            session = dm.init_session_state(objective="Analyze cancer data")

            assert session.objective == "Analyze cancer data"
            assert dm.session_state is session

    def test_data_manager_add_insight(self):
        """Test adding insights through DataManagerV2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.add_insight("Gene X is upregulated")
            dm.add_insight("Gene Y is downregulated")

            insights = dm.get_key_insights()
            assert len(insights) == 2
            assert "Gene X is upregulated" in insights

    def test_data_manager_update_hypothesis(self):
        """Test updating hypothesis through DataManagerV2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.update_hypothesis("Gene X causes cancer")

            hypothesis = dm.get_current_hypothesis()
            assert hypothesis == "Gene X causes cancer"

    def test_data_manager_save_session_state(self):
        """Test explicitly saving session state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.add_insight("Test insight")
            dm.save_session_state()

            # Verify file exists
            state_file = Path(tmpdir) / ".session_state.json"
            assert state_file.exists()

    def test_data_manager_get_session_context(self):
        """Test getting session context summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.init_session_state(objective="Test objective")
            dm.add_insight("Insight 1")
            dm.update_hypothesis("Test hypothesis")

            context = dm.get_session_context()

            assert context["objective"] == "Test objective"
            assert context["current_hypothesis"] == "Test hypothesis"
            assert len(context["key_insights"]) == 1


class TestStatePersistenceAcrossInstances:
    """Test state persistence across DataManagerV2 instances."""

    def test_session_state_persists_across_instances(self):
        """Test that session state survives DataManagerV2 recreation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            dm1 = DataManagerV2(workspace_path=tmpdir)
            dm1.init_session_state(objective="Original objective")
            dm1.add_insight("Insight from session 1")
            dm1.update_hypothesis("Initial hypothesis")
            dm1.save_session_state()

            # Second instance (same workspace)
            dm2 = DataManagerV2(workspace_path=tmpdir)

            # Session state should be loaded from disk
            # Note: This will create a new session_id, but load existing state file
            session = dm2.session_state

            # The state file was saved, so it should have the insights
            # However, session_id will be different
            assert "Insight from session 1" in session.key_insights
            assert session.current_hypothesis == "Initial hypothesis"

    def test_request_state_does_not_persist(self):
        """Test that request state is NOT persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance with request
            dm1 = DataManagerV2(workspace_path=tmpdir)
            dm1.start_request(request_id="req-123")
            dm1.request_state.start_step("test")
            # Don't end request - simulate crash

            # Second instance
            dm2 = DataManagerV2(workspace_path=tmpdir)

            # Request state should be None (transient)
            assert dm2.request_state is None

    def test_multiple_requests_accumulate_insights(self):
        """Test that insights accumulate across multiple requests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # First request
            dm.start_request(request_id="req-1")
            dm.add_insight("Insight from request 1")
            dm.end_request()

            # Second request
            dm.start_request(request_id="req-2")
            dm.add_insight("Insight from request 2")
            dm.end_request()

            # Third request
            dm.start_request(request_id="req-3")
            dm.add_insight("Insight from request 3")
            dm.end_request()

            insights = dm.get_key_insights()
            assert len(insights) == 3
            assert "Insight from request 1" in insights
            assert "Insight from request 2" in insights
            assert "Insight from request 3" in insights


class TestSessionStateFileFormat:
    """Test .session_state.json file format and content."""

    def test_session_state_file_format(self):
        """Test that session state file has expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.init_session_state(objective="Test objective")
            dm.add_insight("Test insight")
            dm.update_hypothesis("Test hypothesis")
            dm.session_state.update_methodology("Test methodology")
            dm.session_state.add_discovery({"title": "Discovery 1"})
            dm.save_session_state()

            # Read file
            state_file = Path(tmpdir) / ".session_state.json"
            with open(state_file) as f:
                data = json.load(f)

            # Verify structure
            assert "session_id" in data
            assert "created_at" in data
            assert "updated_at" in data
            assert "objective" in data
            assert "key_insights" in data
            assert "methodology" in data
            assert "current_hypothesis" in data
            assert "discoveries" in data
            assert "progress" in data
            assert "loaded_modalities" in data

            # Verify content
            assert data["objective"] == "Test objective"
            assert "Test insight" in data["key_insights"]
            assert data["current_hypothesis"] == "Test hypothesis"
            assert data["methodology"] == "Test methodology"
            assert len(data["discoveries"]) == 1

    def test_session_state_file_is_valid_json(self):
        """Test that session state file is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # Add various types of data
            dm.init_session_state(objective="Test")
            dm.add_insight("Insight")
            dm.session_state.update_loaded_modalities(["mod1", "mod2"])
            dm.session_state.increment_iteration()
            dm.save_session_state()

            # Should not raise
            state_file = Path(tmpdir) / ".session_state.json"
            with open(state_file) as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_session_state_progress_tracking(self):
        """Test that progress tracking is persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.session_state.increment_iteration()
            dm.session_state.increment_iteration()
            dm.session_state.mark_converged("Found answer")
            dm.save_session_state()

            # Read file
            state_file = Path(tmpdir) / ".session_state.json"
            with open(state_file) as f:
                data = json.load(f)

            assert data["progress"]["iteration_count"] == 2
            assert data["progress"]["is_converged"] is True
            assert data["progress"]["convergence_reason"] == "Found answer"


class TestRequestTimingIntegration:
    """Test request timing through DataManagerV2."""

    def test_request_timing_single_step(self):
        """Test timing a single step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.start_request(request_id="req-123")
            dm.request_state.start_step("analysis")
            time.sleep(0.05)  # 50ms
            dm.request_state.end_step("analysis")

            summary = dm.end_request()

            assert summary["total_duration_ms"] >= 50
            assert len(summary["steps"]) == 1
            assert summary["steps"][0]["name"] == "analysis"
            assert summary["steps"][0]["duration_ms"] >= 50

    def test_request_timing_multiple_steps(self):
        """Test timing multiple sequential steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.start_request(request_id="req-123")

            dm.request_state.start_step("planning")
            time.sleep(0.02)
            dm.request_state.end_step("planning")

            dm.request_state.start_step("execution")
            time.sleep(0.03)
            dm.request_state.end_step("execution")

            dm.request_state.start_step("export")
            time.sleep(0.01)
            dm.request_state.end_step("export")

            summary = dm.end_request()

            assert len(summary["steps"]) == 3
            assert summary["total_duration_ms"] >= 60  # At least 60ms total

    def test_request_error_tracking(self):
        """Test error tracking in request state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.start_request(request_id="req-123")
            dm.request_state.add_error("Error 1")
            dm.request_state.add_error("Error 2")
            dm.request_state.add_warning("Warning 1")

            summary = dm.end_request()

            assert len(summary["errors"]) == 2
            assert "Error 1" in summary["errors"]
            assert "Warning 1" in summary["warnings"]

    def test_request_tool_call_counting(self):
        """Test tool call counting in request state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.start_request(request_id="req-123")
            dm.request_state.increment_tool_call()
            dm.request_state.increment_tool_call()
            dm.request_state.increment_tool_call()

            summary = dm.end_request()

            assert summary["tool_calls"] == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_end_request_without_start(self):
        """Test ending request when none is active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            summary = dm.end_request()

            assert summary == {}

    def test_multiple_start_requests_overwrites(self):
        """Test that starting a new request overwrites the previous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            request1 = dm.start_request(request_id="req-1")
            request2 = dm.start_request(request_id="req-2")

            assert dm.request_state.request_id == "req-2"
            assert dm.request_state is request2

    def test_session_state_handles_corrupted_file(self):
        """Test that session state handles corrupted file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write corrupted file
            state_file = Path(tmpdir) / ".session_state.json"
            with open(state_file, "w") as f:
                f.write("{ invalid json }")

            # Should create new session without raising
            dm = DataManagerV2(workspace_path=tmpdir)
            session = dm.session_state

            assert session is not None
            assert session.session_id is not None

    def test_insight_limit_enforced(self):
        """Test that insight limit is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # Add 20 insights (more than max 10)
            for i in range(20):
                dm.add_insight(f"Insight {i}")

            insights = dm.get_key_insights()
            assert len(insights) == 10

            # Should keep most recent
            assert "Insight 19" in insights
            assert "Insight 0" not in insights

    def test_empty_workspace_creates_state_file(self):
        """Test that state file is created in empty workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            dm.init_session_state(objective="Test")
            dm.save_session_state()

            state_file = Path(tmpdir) / ".session_state.json"
            assert state_file.exists()

    def test_session_state_with_special_characters(self):
        """Test session state handles special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManagerV2(workspace_path=tmpdir)

            # Add insights with special characters
            dm.add_insight("Gene expression: p-value < 0.05 (FDR)")
            dm.add_insight("Pathway: NF-\u03baB signaling")  # Unicode
            dm.update_hypothesis("H\u2080: No difference exists")

            dm.save_session_state()

            # Reload and verify
            dm2 = DataManagerV2(workspace_path=tmpdir)
            insights = dm2.get_key_insights()

            assert "Gene expression: p-value < 0.05 (FDR)" in insights
            assert "Pathway: NF-\u03baB signaling" in insights
