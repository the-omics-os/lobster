"""
Unit tests for Two-Tier State Management.

Tests cover:
- StepTiming: timing calculations
- RequestState: lifecycle, timing, error tracking
- SessionState: persistence, insight accumulation, hypothesis updates
- StateManager: orchestration, file I/O, caching
"""

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from lobster.core.schemas.state import (
    RequestState,
    ResearchProgress,
    SessionState,
    StateManager,
    StepTiming,
    create_request_state,
    create_session_state,
)


# =============================================================================
# StepTiming Tests
# =============================================================================


class TestStepTiming:
    """Tests for StepTiming model."""

    def test_step_timing_creation(self):
        """Test creating a StepTiming instance."""
        now = datetime.utcnow()
        step = StepTiming(name="analysis", start=now)

        assert step.name == "analysis"
        assert step.start == now
        assert step.end is None
        assert step.duration_ms is None

    def test_step_timing_complete(self):
        """Test completing a step and computing duration."""
        step = StepTiming(name="analysis", start=datetime.utcnow())
        time.sleep(0.01)  # Small delay to ensure measurable duration
        step.complete()

        assert step.end is not None
        assert step.duration_ms is not None
        assert step.duration_ms >= 10  # At least 10ms

    def test_step_timing_complete_duration_calculation(self):
        """Test duration calculation is accurate."""
        start = datetime.utcnow()
        step = StepTiming(name="test", start=start)

        # Manually set end time to test calculation
        step.end = start + timedelta(milliseconds=500)
        step.duration_ms = int((step.end - step.start).total_seconds() * 1000)

        assert step.duration_ms == 500


# =============================================================================
# RequestState Tests
# =============================================================================


class TestRequestState:
    """Tests for RequestState (Tier 1 - Transient)."""

    def test_request_state_creation(self):
        """Test creating a RequestState instance."""
        state = RequestState(request_id="req-123", session_id="sess-456")

        assert state.request_id == "req-123"
        assert state.session_id == "sess-456"
        assert state.user_id is None
        assert state.source is None
        assert state.is_deep_research is False
        assert state.is_streaming is False
        assert state.steps == []
        assert state.current_step is None
        assert state.tool_call_count == 0
        assert state.errors == []
        assert state.warnings == []

    def test_request_state_with_optional_fields(self):
        """Test RequestState with optional fields set."""
        state = RequestState(
            request_id="req-123",
            session_id="sess-456",
            user_id="user-1",
            source="cli",
            is_deep_research=True,
            is_streaming=True,
        )

        assert state.user_id == "user-1"
        assert state.source == "cli"
        assert state.is_deep_research is True
        assert state.is_streaming is True

    def test_request_state_start_step(self):
        """Test starting a step."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        state.start_step("analysis")

        assert state.current_step == "analysis"
        assert len(state.steps) == 1
        assert state.steps[0].name == "analysis"
        assert state.steps[0].start is not None
        assert state.steps[0].end is None

    def test_request_state_end_step(self):
        """Test ending a step."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        state.start_step("analysis")
        time.sleep(0.01)
        duration = state.end_step("analysis")

        assert state.current_step is None
        assert state.steps[0].end is not None
        assert duration is not None
        assert duration >= 10

    def test_request_state_end_nonexistent_step(self):
        """Test ending a step that doesn't exist."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        duration = state.end_step("nonexistent")

        assert duration is None

    def test_request_state_multiple_steps(self):
        """Test tracking multiple steps."""
        state = RequestState(request_id="req-123", session_id="sess-456")

        state.start_step("planning")
        time.sleep(0.01)
        state.end_step("planning")

        state.start_step("execution")
        time.sleep(0.01)
        state.end_step("execution")

        assert len(state.steps) == 2
        assert state.steps[0].name == "planning"
        assert state.steps[1].name == "execution"
        assert state.steps[0].duration_ms is not None
        assert state.steps[1].duration_ms is not None

    def test_request_state_get_step_duration(self):
        """Test getting duration of a specific step."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        state.start_step("analysis")
        time.sleep(0.01)
        state.end_step("analysis")

        duration = state.get_step_duration("analysis")
        assert duration is not None
        assert duration >= 10

    def test_request_state_get_step_duration_not_found(self):
        """Test getting duration of nonexistent step."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        duration = state.get_step_duration("nonexistent")
        assert duration is None

    def test_request_state_get_total_duration(self):
        """Test getting total duration of all steps."""
        state = RequestState(request_id="req-123", session_id="sess-456")

        state.start_step("step1")
        time.sleep(0.01)
        state.end_step("step1")

        state.start_step("step2")
        time.sleep(0.01)
        state.end_step("step2")

        total = state.get_total_duration()
        assert total >= 20  # At least 20ms total

    def test_request_state_add_error(self):
        """Test adding errors."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        state.add_error("Error 1")
        state.add_error("Error 2")

        assert len(state.errors) == 2
        assert state.errors[0] == "Error 1"
        assert state.errors[1] == "Error 2"

    def test_request_state_add_warning(self):
        """Test adding warnings."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        state.add_warning("Warning 1")

        assert len(state.warnings) == 1
        assert state.warnings[0] == "Warning 1"

    def test_request_state_increment_tool_call(self):
        """Test incrementing tool call counter."""
        state = RequestState(request_id="req-123", session_id="sess-456")
        assert state.tool_call_count == 0

        count = state.increment_tool_call()
        assert count == 1
        assert state.tool_call_count == 1

        count = state.increment_tool_call()
        assert count == 2


# =============================================================================
# ResearchProgress Tests
# =============================================================================


class TestResearchProgress:
    """Tests for ResearchProgress model."""

    def test_research_progress_defaults(self):
        """Test default values."""
        progress = ResearchProgress()

        assert progress.iteration_count == 0
        assert progress.max_iterations == 5
        assert progress.mode == "semi-autonomous"
        assert progress.is_converged is False
        assert progress.convergence_reason is None

    def test_research_progress_custom_values(self):
        """Test custom values."""
        progress = ResearchProgress(
            iteration_count=3,
            max_iterations=10,
            mode="fully-autonomous",
            is_converged=True,
            convergence_reason="Found answer",
        )

        assert progress.iteration_count == 3
        assert progress.max_iterations == 10
        assert progress.mode == "fully-autonomous"
        assert progress.is_converged is True
        assert progress.convergence_reason == "Found answer"


# =============================================================================
# SessionState Tests
# =============================================================================


class TestSessionState:
    """Tests for SessionState (Tier 2 - Persistent)."""

    def test_session_state_creation(self):
        """Test creating a SessionState instance."""
        state = SessionState(session_id="sess-123")

        assert state.session_id == "sess-123"
        assert state.objective == ""
        assert state.current_objective is None
        assert state.session_title is None
        assert state.key_insights == []
        assert state.methodology is None
        assert state.current_hypothesis is None
        assert state.discoveries == []
        assert state.current_plan == []
        assert state.suggested_next_steps == []
        assert state.loaded_modalities == []
        assert state.uploaded_files == []

    def test_session_state_with_objective(self):
        """Test SessionState with objective set."""
        state = SessionState(
            session_id="sess-123",
            objective="Analyze gene expression in cancer cells",
        )

        assert state.objective == "Analyze gene expression in cancer cells"

    def test_session_state_add_insight(self):
        """Test adding insights."""
        state = SessionState(session_id="sess-123")
        state.add_insight("Gene X is upregulated")
        state.add_insight("Gene Y is downregulated")

        assert len(state.key_insights) == 2
        assert state.key_insights[0] == "Gene X is upregulated"
        assert state.key_insights[1] == "Gene Y is downregulated"

    def test_session_state_add_insight_no_duplicates(self):
        """Test that duplicate insights are not added."""
        state = SessionState(session_id="sess-123")
        state.add_insight("Gene X is upregulated")
        state.add_insight("Gene X is upregulated")  # Duplicate

        assert len(state.key_insights) == 1

    def test_session_state_add_insight_max_limit(self):
        """Test that insights are limited to max_insights."""
        state = SessionState(session_id="sess-123")

        # Add 15 insights (more than default max of 10)
        for i in range(15):
            state.add_insight(f"Insight {i}")

        assert len(state.key_insights) == 10
        # Should keep most recent
        assert state.key_insights[0] == "Insight 5"
        assert state.key_insights[-1] == "Insight 14"

    def test_session_state_add_insight_custom_max(self):
        """Test custom max_insights limit."""
        state = SessionState(session_id="sess-123")

        for i in range(10):
            state.add_insight(f"Insight {i}", max_insights=5)

        assert len(state.key_insights) == 5

    def test_session_state_add_insight_empty_string(self):
        """Test that empty insights are not added."""
        state = SessionState(session_id="sess-123")
        state.add_insight("")
        state.add_insight("   ")  # Whitespace is kept if non-empty

        # Empty string should not be added, but whitespace will be
        assert "" not in state.key_insights

    def test_session_state_update_hypothesis(self):
        """Test updating hypothesis."""
        state = SessionState(session_id="sess-123")
        state.update_hypothesis("Gene X causes cancer")

        assert state.current_hypothesis == "Gene X causes cancer"

    def test_session_state_update_methodology(self):
        """Test updating methodology."""
        state = SessionState(session_id="sess-123")
        state.update_methodology("Use differential expression analysis")

        assert state.methodology == "Use differential expression analysis"

    def test_session_state_add_discovery(self):
        """Test adding discoveries."""
        state = SessionState(session_id="sess-123")
        discovery = {
            "title": "Gene X Discovery",
            "claim": "Gene X is a key regulator",
            "summary": "Detailed analysis shows...",
            "evidence": ["Figure 1", "Table 2"],
        }
        state.add_discovery(discovery)

        assert len(state.discoveries) == 1
        assert state.discoveries[0]["title"] == "Gene X Discovery"

    def test_session_state_increment_iteration(self):
        """Test incrementing iteration count."""
        state = SessionState(session_id="sess-123")

        # Should return True while under max
        assert state.increment_iteration() is True
        assert state.progress.iteration_count == 1

        # Continue until max
        for _ in range(3):
            state.increment_iteration()

        assert state.progress.iteration_count == 4

        # At max (5 iterations), should return False
        assert state.increment_iteration() is False
        assert state.progress.iteration_count == 5

    def test_session_state_mark_converged(self):
        """Test marking research as converged."""
        state = SessionState(session_id="sess-123")
        state.mark_converged("Found conclusive evidence")

        assert state.progress.is_converged is True
        assert state.progress.convergence_reason == "Found conclusive evidence"

    def test_session_state_update_loaded_modalities(self):
        """Test updating loaded modalities list."""
        state = SessionState(session_id="sess-123")
        state.update_loaded_modalities(["geo_gse123", "geo_gse456"])

        assert state.loaded_modalities == ["geo_gse123", "geo_gse456"]

    def test_session_state_get_context_summary(self):
        """Test getting context summary for agent prompts."""
        state = SessionState(
            session_id="sess-123",
            objective="Analyze cancer data",
        )
        state.add_insight("Insight 1")
        state.add_insight("Insight 2")
        state.update_hypothesis("Gene X is key")
        state.update_methodology("Use DESeq2")
        state.update_loaded_modalities(["modality1"])
        state.increment_iteration()

        summary = state.get_context_summary()

        assert summary["objective"] == "Analyze cancer data"
        assert summary["current_hypothesis"] == "Gene X is key"
        assert summary["methodology"] == "Use DESeq2"
        assert len(summary["key_insights"]) == 2
        assert summary["iteration"] == 1
        assert summary["is_converged"] is False
        assert summary["loaded_modalities"] == ["modality1"]

    def test_session_state_updated_at_timestamp(self):
        """Test that updated_at is updated on mutations."""
        state = SessionState(session_id="sess-123")
        original_time = state.updated_at

        time.sleep(0.01)
        state.add_insight("New insight")

        assert state.updated_at > original_time


# =============================================================================
# StateManager Tests
# =============================================================================


class TestStateManager:
    """Tests for StateManager orchestration."""

    def test_state_manager_creation(self):
        """Test creating a StateManager instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)

            assert manager.workspace_path == Path(tmpdir)
            assert manager._current_request is None
            assert manager._session_cache == {}

    def test_state_manager_start_request(self):
        """Test starting a request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            request = manager.start_request(
                request_id="req-123",
                session_id="sess-456",
                source="cli",
            )

            assert request.request_id == "req-123"
            assert request.session_id == "sess-456"
            assert request.source == "cli"
            assert manager._current_request is request

    def test_state_manager_start_request_auto_id(self):
        """Test starting a request with auto-generated ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            request = manager.start_request(session_id="sess-456")

            assert request.request_id is not None
            assert len(request.request_id) == 36  # UUID format

    def test_state_manager_get_current_request(self):
        """Test getting current request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)

            # No request yet
            assert manager.get_current_request() is None

            # Start request
            manager.start_request(session_id="sess-456")
            assert manager.get_current_request() is not None

    def test_state_manager_end_request(self):
        """Test ending a request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            manager.start_request(request_id="req-123", session_id="sess-456")

            # Add a step
            manager._current_request.start_step("test")
            time.sleep(0.01)
            manager._current_request.end_step("test")

            # End request
            summary = manager.end_request()

            assert summary["request_id"] == "req-123"
            assert summary["total_duration_ms"] >= 10
            assert len(summary["steps"]) == 1
            assert summary["steps"][0]["name"] == "test"
            assert manager._current_request is None

    def test_state_manager_end_request_no_active(self):
        """Test ending request when none is active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            summary = manager.end_request()

            assert summary == {}

    def test_state_manager_load_session_new(self):
        """Test loading a new session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            session = manager.load_session(
                session_id="sess-123",
                objective="Test objective",
            )

            assert session.session_id == "sess-123"
            assert session.objective == "Test objective"
            assert "sess-123" in manager._session_cache

    def test_state_manager_load_session_from_cache(self):
        """Test loading session from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)

            # First load
            session1 = manager.load_session(session_id="sess-123")

            # Second load should return cached
            session2 = manager.load_session(session_id="sess-123")

            assert session1 is session2

    def test_state_manager_save_session(self):
        """Test saving session to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            session = manager.load_session(session_id="sess-123")
            session.add_insight("Test insight")
            session.update_hypothesis("Test hypothesis")

            manager.save_session(session)

            # Verify file exists
            state_file = Path(tmpdir) / ".session_state.json"
            assert state_file.exists()

            # Verify content
            with open(state_file) as f:
                data = json.load(f)

            assert data["session_id"] == "sess-123"
            assert "Test insight" in data["key_insights"]
            assert data["current_hypothesis"] == "Test hypothesis"

    def test_state_manager_load_session_from_disk(self):
        """Test loading session from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save session
            manager1 = StateManager(tmpdir)
            session1 = manager1.load_session(session_id="sess-123")
            session1.add_insight("Test insight")
            manager1.save_session(session1)

            # Clear cache and reload
            manager1.clear_session_cache()
            session2 = manager1.load_session(session_id="sess-123")

            assert session2.session_id == "sess-123"
            assert "Test insight" in session2.key_insights

    def test_state_manager_clear_session_cache(self):
        """Test clearing session cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            manager.load_session(session_id="sess-123")
            assert len(manager._session_cache) == 1

            manager.clear_session_cache()
            assert len(manager._session_cache) == 0

    def test_state_manager_delete_session_file(self):
        """Test deleting session file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            session = manager.load_session(session_id="sess-123")
            manager.save_session(session)

            state_file = Path(tmpdir) / ".session_state.json"
            assert state_file.exists()

            result = manager.delete_session_file()
            assert result is True
            assert not state_file.exists()

    def test_state_manager_delete_session_file_nonexistent(self):
        """Test deleting nonexistent session file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            result = manager.delete_session_file()
            assert result is False

    def test_state_manager_thread_safety(self):
        """Test thread-safe session file access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            session = manager.load_session(session_id="sess-123")

            errors = []

            def save_session():
                try:
                    for _ in range(10):
                        session.add_insight(f"Insight from {threading.current_thread().name}")
                        manager.save_session(session)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=save_session, name=f"Thread-{i}")
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should occur
            assert len(errors) == 0

            # File should be valid JSON
            state_file = Path(tmpdir) / ".session_state.json"
            with open(state_file) as f:
                data = json.load(f)
            assert data["session_id"] == "sess-123"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_request_state(self):
        """Test create_request_state convenience function."""
        state = create_request_state(
            session_id="sess-123",
            source="api",
        )

        assert state.session_id == "sess-123"
        assert state.source == "api"
        assert state.request_id is not None

    def test_create_request_state_with_id(self):
        """Test create_request_state with custom request_id."""
        state = create_request_state(
            session_id="sess-123",
            request_id="custom-req-id",
        )

        assert state.request_id == "custom-req-id"

    def test_create_session_state(self):
        """Test create_session_state convenience function."""
        state = create_session_state(objective="Test objective")

        assert state.session_id is not None
        assert state.objective == "Test objective"

    def test_create_session_state_with_id(self):
        """Test create_session_state with custom session_id."""
        state = create_session_state(
            session_id="custom-sess-id",
            objective="Test objective",
        )

        assert state.session_id == "custom-sess-id"


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJsonSerialization:
    """Tests for JSON serialization of state objects."""

    def test_request_state_serialization(self):
        """Test RequestState can be serialized to JSON."""
        state = RequestState(
            request_id="req-123",
            session_id="sess-456",
            source="cli",
        )
        state.start_step("test")
        state.end_step("test")
        state.add_error("Test error")

        # Should not raise
        json_data = state.model_dump(mode="json")
        json_str = json.dumps(json_data, default=str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["request_id"] == "req-123"

    def test_session_state_serialization(self):
        """Test SessionState can be serialized to JSON."""
        state = SessionState(
            session_id="sess-123",
            objective="Test objective",
        )
        state.add_insight("Insight 1")
        state.update_hypothesis("Hypothesis 1")
        state.add_discovery({"title": "Discovery 1"})

        # Should not raise
        json_data = state.model_dump(mode="json")
        json_str = json.dumps(json_data, default=str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "sess-123"
        assert "Insight 1" in parsed["key_insights"]

    def test_session_state_round_trip(self):
        """Test SessionState survives JSON round-trip."""
        original = SessionState(
            session_id="sess-123",
            objective="Test objective",
        )
        original.add_insight("Insight 1")
        original.update_hypothesis("Hypothesis 1")

        # Serialize and deserialize
        json_data = original.model_dump(mode="json")
        json_str = json.dumps(json_data, default=str)
        parsed = json.loads(json_str)

        # Handle datetime fields
        if "created_at" in parsed and isinstance(parsed["created_at"], str):
            parsed["created_at"] = datetime.fromisoformat(
                parsed["created_at"].replace("Z", "+00:00")
            )
        if "updated_at" in parsed and isinstance(parsed["updated_at"], str):
            parsed["updated_at"] = datetime.fromisoformat(
                parsed["updated_at"].replace("Z", "+00:00")
            )

        restored = SessionState(**parsed)

        assert restored.session_id == original.session_id
        assert restored.objective == original.objective
        assert restored.key_insights == original.key_insights
        assert restored.current_hypothesis == original.current_hypothesis
