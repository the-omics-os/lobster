"""
Two-Tier State Management for Lobster AI.

Inspired by BioAgents' state architecture, this module implements:
- Tier 1: RequestState - Transient, per-request, discarded after processing
- Tier 2: SessionState - Persistent, per-session, saved to `.session_state.json`

The StateManager class orchestrates both tiers, handling lifecycle and persistence.

Example Usage:
    state_manager = StateManager(workspace_path)

    # Start request
    request_state = state_manager.start_request(request_id="req-123")
    session_state = state_manager.load_session(session_id="sess-456")

    # During processing
    request_state.start_step("analysis")
    # ... agent work ...
    request_state.end_step("analysis")
    session_state.add_insight("Found 100 differentially expressed genes")

    # End request
    state_manager.save_session(session_state)
    summary = state_manager.end_request()  # request_state is discarded
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# TIER 1: REQUEST STATE (Transient)
# =============================================================================


class StepTiming(BaseModel):
    """
    Timing information for a single execution step.

    Used to track duration of individual operations within a request
    (e.g., "planning", "analysis", "export").
    """

    name: str = Field(description="Step identifier (e.g., 'planning', 'clustering')")
    start: datetime = Field(description="UTC timestamp when step started")
    end: Optional[datetime] = Field(default=None, description="UTC timestamp when step ended")
    duration_ms: Optional[int] = Field(
        default=None, description="Duration in milliseconds (computed on end)"
    )

    def complete(self) -> None:
        """Mark step as complete and compute duration."""
        self.end = datetime.utcnow()
        self.duration_ms = int((self.end - self.start).total_seconds() * 1000)


class RequestState(BaseModel):
    """
    Transient state for a single request.

    Lifecycle: Created at request start, discarded at request end.
    Purpose: Track request metadata, timing, and intermediate results.

    IMPORTANT: This state is NEVER persisted to disk.
    It is garbage collected after the request completes.

    Attributes:
        request_id: Unique identifier for this request
        session_id: Parent session identifier
        user_id: Optional user identifier
        source: Request source ("cli", "api", "cloud")
        is_deep_research: Whether this is a deep research request
        is_streaming: Whether streaming mode is enabled
        steps: List of timing information for each step
        current_step: Name of currently executing step
        intermediate_response: Temporary response buffer
        llm_thought: LLM reasoning (for thinking-enabled models)
        tool_call_count: Number of tool invocations
        errors: List of errors encountered
        warnings: List of warnings generated
    """

    # Request identification
    request_id: str = Field(description="Unique request identifier")
    session_id: str = Field(description="Parent session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    source: Optional[str] = Field(
        default=None, description="Request source: 'cli', 'api', or 'cloud'"
    )

    # Request type flags
    is_deep_research: bool = Field(default=False, description="Deep research mode enabled")
    is_streaming: bool = Field(default=False, description="Streaming mode enabled")

    # Execution tracking
    steps: List[StepTiming] = Field(
        default_factory=list, description="Timing for each execution step"
    )
    current_step: Optional[str] = Field(
        default=None, description="Currently executing step name"
    )

    # Intermediate results (not persisted)
    intermediate_response: Optional[str] = Field(
        default=None, description="Temporary response buffer"
    )
    llm_thought: Optional[str] = Field(
        default=None, description="LLM reasoning for thinking-enabled models"
    )
    tool_call_count: int = Field(default=0, description="Number of tool invocations")

    # Error tracking
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")

    def start_step(self, step_name: str) -> None:
        """
        Start timing a new step.

        Args:
            step_name: Identifier for the step (e.g., "analysis", "export")
        """
        self.current_step = step_name
        self.steps.append(StepTiming(name=step_name, start=datetime.utcnow()))

    def end_step(self, step_name: str) -> Optional[int]:
        """
        End timing for a step and compute duration.

        Args:
            step_name: The step to complete

        Returns:
            Duration in milliseconds, or None if step not found
        """
        for step in reversed(self.steps):
            if step.name == step_name and step.end is None:
                step.complete()
                if self.current_step == step_name:
                    self.current_step = None
                return step.duration_ms
        return None

    def get_step_duration(self, step_name: str) -> Optional[int]:
        """
        Get duration of a completed step in milliseconds.

        Args:
            step_name: The step to query

        Returns:
            Duration in milliseconds, or None if step not found/incomplete
        """
        for step in self.steps:
            if step.name == step_name:
                return step.duration_ms
        return None

    def get_total_duration(self) -> int:
        """
        Get total duration of all completed steps in milliseconds.

        Returns:
            Sum of all step durations
        """
        return sum(s.duration_ms or 0 for s in self.steps)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def increment_tool_call(self) -> int:
        """Increment tool call counter and return new count."""
        self.tool_call_count += 1
        return self.tool_call_count


# =============================================================================
# TIER 2: SESSION STATE (Persistent)
# =============================================================================


class ResearchProgress(BaseModel):
    """
    Track research progress across iterations.

    Used for multi-turn research sessions where the system
    iteratively refines hypotheses and accumulates insights.
    """

    iteration_count: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=5, description="Maximum allowed iterations")
    mode: str = Field(
        default="semi-autonomous",
        description="Research mode: 'semi-autonomous', 'fully-autonomous', or 'steering'",
    )
    is_converged: bool = Field(default=False, description="Whether research has converged")
    convergence_reason: Optional[str] = Field(
        default=None, description="Reason for convergence (if converged)"
    )


class SessionState(BaseModel):
    """
    Persistent state for a research session.

    Lifecycle: Created at session start, persisted across requests.
    Purpose: Accumulate knowledge, track progress, enable multi-turn research.

    Storage: Saved to `workspace/.session_state.json`

    Attributes:
        session_id: Unique session identifier
        created_at: When session was created
        updated_at: When session was last updated
        objective: Main research question/goal
        current_objective: Current iteration's focus (may evolve)
        session_title: Human-readable session title
        key_insights: Accumulated insights (max 10, most recent kept)
        methodology: Current research methodology/approach
        current_hypothesis: Latest generated hypothesis
        discoveries: Structured scientific discoveries
        progress: Research iteration progress
        current_plan: Currently executing plan tasks
        suggested_next_steps: Suggested tasks for next iteration
        loaded_modalities: Names of loaded data modalities
        uploaded_files: Uploaded file metadata
    """

    # Session identification
    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Session creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Research objectives
    objective: str = Field(default="", description="Main research question/goal")
    current_objective: Optional[str] = Field(
        default=None, description="Current iteration's focus (may evolve)"
    )
    session_title: Optional[str] = Field(
        default=None, description="Human-readable session title"
    )

    # Accumulated knowledge (CRITICAL - this is what persists)
    key_insights: List[str] = Field(
        default_factory=list, description="Max 10 prioritized insights from reflection"
    )
    methodology: Optional[str] = Field(
        default=None, description="Current research methodology/approach"
    )
    current_hypothesis: Optional[str] = Field(
        default=None, description="Latest generated hypothesis"
    )
    discoveries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Structured scientific discoveries"
    )

    # Research progress
    progress: ResearchProgress = Field(
        default_factory=ResearchProgress, description="Research iteration progress"
    )

    # Plan state
    current_plan: List[Dict[str, Any]] = Field(
        default_factory=list, description="Currently executing plan tasks"
    )
    suggested_next_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Suggested tasks for next iteration"
    )

    # Data references (not the data itself, just names/metadata)
    loaded_modalities: List[str] = Field(
        default_factory=list, description="Names of loaded modalities"
    )
    uploaded_files: List[Dict[str, str]] = Field(
        default_factory=list, description="Uploaded file metadata"
    )

    def add_insight(self, insight: str, max_insights: int = 10) -> None:
        """
        Add insight with maximum limit enforcement.

        If insight already exists, it won't be duplicated.
        If over limit, oldest insights are removed.

        Args:
            insight: The insight to add
            max_insights: Maximum number of insights to keep (default: 10)
        """
        if insight and insight not in self.key_insights:
            self.key_insights.append(insight)
            # Keep only most recent if over limit
            if len(self.key_insights) > max_insights:
                self.key_insights = self.key_insights[-max_insights:]
        self.updated_at = datetime.utcnow()

    def update_hypothesis(self, hypothesis: str) -> None:
        """
        Update the current hypothesis.

        Args:
            hypothesis: The new hypothesis
        """
        self.current_hypothesis = hypothesis
        self.updated_at = datetime.utcnow()

    def update_methodology(self, methodology: str) -> None:
        """
        Update the research methodology.

        Args:
            methodology: The new methodology description
        """
        self.methodology = methodology
        self.updated_at = datetime.utcnow()

    def add_discovery(self, discovery: Dict[str, Any]) -> None:
        """
        Add a structured discovery.

        Args:
            discovery: Discovery dictionary with title, claim, summary, evidence
        """
        self.discoveries.append(discovery)
        self.updated_at = datetime.utcnow()

    def increment_iteration(self) -> bool:
        """
        Increment iteration count.

        Returns:
            True if can continue (under max), False if at limit
        """
        self.progress.iteration_count += 1
        self.updated_at = datetime.utcnow()
        return self.progress.iteration_count < self.progress.max_iterations

    def mark_converged(self, reason: str) -> None:
        """
        Mark research as converged.

        Args:
            reason: Why research converged
        """
        self.progress.is_converged = True
        self.progress.convergence_reason = reason
        self.updated_at = datetime.utcnow()

    def update_loaded_modalities(self, modality_names: List[str]) -> None:
        """
        Update the list of loaded modalities.

        Args:
            modality_names: List of modality names currently loaded
        """
        self.loaded_modalities = modality_names
        self.updated_at = datetime.utcnow()

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of session context for agent prompts.

        Returns:
            Dictionary with key context information
        """
        return {
            "objective": self.objective,
            "current_objective": self.current_objective,
            "key_insights": self.key_insights[-5:] if self.key_insights else [],
            "current_hypothesis": self.current_hypothesis,
            "methodology": self.methodology,
            "iteration": self.progress.iteration_count,
            "is_converged": self.progress.is_converged,
            "loaded_modalities": self.loaded_modalities,
        }


# =============================================================================
# STATE MANAGER (Orchestrates Both Tiers)
# =============================================================================


class StateManager:
    """
    Manages two-tier state lifecycle.

    Responsibilities:
    - Create and discard transient RequestState (Tier 1)
    - Load, save, and cache persistent SessionState (Tier 2)
    - Handle thread-safe file I/O for session persistence

    Usage:
        state_manager = StateManager(workspace_path)

        # Start request
        request_state = state_manager.start_request(request_id)
        session_state = state_manager.load_session(session_id)

        # During processing
        request_state.start_step("planning")
        # ... agent work ...
        request_state.end_step("planning")
        session_state.add_insight("New finding about X")

        # End request
        state_manager.save_session(session_state)
        summary = state_manager.end_request()  # request_state is discarded

    Thread Safety:
        - Session file I/O is protected by threading.Lock
        - Session cache is thread-safe for read operations
    """

    STATE_FILE_NAME = ".session_state.json"

    def __init__(self, workspace_path: str):
        """
        Initialize StateManager.

        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = Path(workspace_path)
        self._current_request: Optional[RequestState] = None
        self._session_cache: Dict[str, SessionState] = {}
        self._file_lock = threading.Lock()

    def start_request(
        self,
        request_id: Optional[str] = None,
        session_id: str = "",
        **kwargs,
    ) -> RequestState:
        """
        Create new transient request state.

        Args:
            request_id: Unique request ID (auto-generated if not provided)
            session_id: Parent session ID
            **kwargs: Additional RequestState fields (user_id, source, etc.)

        Returns:
            New RequestState instance
        """
        request_id = request_id or str(uuid.uuid4())
        self._current_request = RequestState(
            request_id=request_id, session_id=session_id, **kwargs
        )
        logger.debug(f"Started request: {request_id}")
        return self._current_request

    def get_current_request(self) -> Optional[RequestState]:
        """
        Get current request state.

        Returns:
            Current RequestState or None if no active request
        """
        return self._current_request

    def end_request(self) -> Dict[str, Any]:
        """
        End request and return timing summary.

        The request state is discarded after this call.

        Returns:
            Dictionary with timing summary:
            - request_id: The request identifier
            - total_duration_ms: Total duration of all steps
            - steps: List of step timing info
            - tool_calls: Number of tool invocations
            - errors: List of errors encountered
        """
        if self._current_request is None:
            return {}

        summary = {
            "request_id": self._current_request.request_id,
            "total_duration_ms": self._current_request.get_total_duration(),
            "steps": [
                {"name": s.name, "duration_ms": s.duration_ms}
                for s in self._current_request.steps
            ],
            "tool_calls": self._current_request.tool_call_count,
            "errors": self._current_request.errors,
            "warnings": self._current_request.warnings,
        }

        logger.debug(
            f"Ended request: {self._current_request.request_id} "
            f"(duration: {summary['total_duration_ms']}ms)"
        )

        # Discard transient state
        self._current_request = None
        return summary

    def load_session(self, session_id: str, objective: str = "") -> SessionState:
        """
        Load or create persistent session state.

        If session exists in cache or on disk, it is loaded.
        Otherwise, a new session is created.

        Args:
            session_id: Session identifier
            objective: Initial research objective (for new sessions)

        Returns:
            SessionState instance
        """
        # Check cache first
        if session_id in self._session_cache:
            logger.debug(f"Loaded session from cache: {session_id}")
            return self._session_cache[session_id]

        # Try to load from disk
        state_file = self.workspace_path / self.STATE_FILE_NAME
        if state_file.exists():
            try:
                with self._file_lock:
                    with open(state_file, "r") as f:
                        data = json.load(f)

                # Handle datetime fields
                if "created_at" in data and isinstance(data["created_at"], str):
                    data["created_at"] = datetime.fromisoformat(
                        data["created_at"].replace("Z", "+00:00")
                    )
                if "updated_at" in data and isinstance(data["updated_at"], str):
                    data["updated_at"] = datetime.fromisoformat(
                        data["updated_at"].replace("Z", "+00:00")
                    )

                session = SessionState(**data)
                self._session_cache[session_id] = session
                logger.debug(f"Loaded session from disk: {session_id}")
                return session

            except Exception as e:
                logger.warning(f"Could not load session from disk: {e}")
                # Fall through to create new

        # Create new session state
        session = SessionState(session_id=session_id, objective=objective)
        self._session_cache[session_id] = session
        logger.debug(f"Created new session: {session_id}")
        return session

    def save_session(self, session: SessionState) -> None:
        """
        Persist session state to disk.

        Args:
            session: The SessionState to save
        """
        session.updated_at = datetime.utcnow()
        state_file = self.workspace_path / self.STATE_FILE_NAME

        # Ensure directory exists
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        with self._file_lock:
            with open(state_file, "w") as f:
                json.dump(
                    session.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,
                )

        # Update cache
        self._session_cache[session.session_id] = session
        logger.debug(f"Saved session to disk: {session.session_id}")

    def clear_session_cache(self) -> None:
        """Clear the in-memory session cache."""
        self._session_cache.clear()
        logger.debug("Cleared session cache")

    def delete_session_file(self) -> bool:
        """
        Delete the session state file from disk.

        Returns:
            True if file was deleted, False if it didn't exist
        """
        state_file = self.workspace_path / self.STATE_FILE_NAME
        if state_file.exists():
            with self._file_lock:
                state_file.unlink()
            logger.debug("Deleted session state file")
            return True
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_request_state(
    session_id: str,
    request_id: Optional[str] = None,
    source: str = "cli",
) -> RequestState:
    """
    Convenience function to create a RequestState.

    Args:
        session_id: Parent session ID
        request_id: Unique request ID (auto-generated if not provided)
        source: Request source ("cli", "api", "cloud")

    Returns:
        New RequestState instance
    """
    return RequestState(
        request_id=request_id or str(uuid.uuid4()),
        session_id=session_id,
        source=source,
    )


def create_session_state(
    session_id: Optional[str] = None,
    objective: str = "",
) -> SessionState:
    """
    Convenience function to create a SessionState.

    Args:
        session_id: Session ID (auto-generated if not provided)
        objective: Initial research objective

    Returns:
        New SessionState instance
    """
    return SessionState(
        session_id=session_id or str(uuid.uuid4()),
        objective=objective,
    )
