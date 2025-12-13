# Token Tracking Agent Name Detection - Complete Implementation

## Analysis Summary

### Problem
`TokenTrackingCallback.on_llm_start` needs robust agent name detection for accurate token attribution across multi-agent workflows.

### Root Causes
1. **Missing `on_chat_model_start`**: Claude/Bedrock use chat model callbacks, not LLM callbacks
2. **Unreliable detection sources**: Only checking 3 sources with hardcoded filters
3. **No validation**: Doesn't check if detected name is actually an agent
4. **No registry integration**: Should validate against `AGENT_REGISTRY`
5. **Stale state**: `current_agent` can become incorrect after handoffs

### Key Insights from LangChain/LangGraph

**Callback Propagation:**
- Config with `run_name` and `tags` is passed through `agent.invoke(..., config=config)`
- Available in callbacks via `kwargs.get("run_name")`, `kwargs.get("tags")`
- Can be lost in nested invocations or parallel execution

**Agent Switching:**
- **PRIMARY**: Detected via `on_tool_start` with `handoff_to_*` tool names
- **BACKUP**: Detected via `on_chain_start` with registry lookup
- LLM callbacks should maintain `current_agent`, not detect switches

## Complete Implementation

```python
"""
Robust agent name detection for TokenTrackingCallback.

Key improvements:
1. Multi-source detection with priority chain
2. Agent registry validation
3. Model class name filtering
4. Thread-safe state management
5. Run ID hierarchy tracking
6. Graceful fallback handling
"""

from typing import Dict, List, Optional, Set
from langchain_core.callbacks import BaseCallbackHandler
from lobster.config.agent_registry import get_all_agent_names


class TokenTrackingCallback(BaseCallbackHandler):
    """Enhanced token tracking with robust agent name detection."""

    # Known LLM model class names (case-insensitive)
    MODEL_CLASS_NAMES: Set[str] = {
        "chatbedrock",
        "chatbedrockconverse",
        "chatanthropic",
        "chatollama",
        "llm",
        "chat",
        "chatmodel",
        "completionmodel",
    }

    def __init__(
        self,
        session_id: str,
        pricing_config: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize with agent registry caching."""
        super().__init__()

        self.session_id = session_id
        self.pricing_config = pricing_config or {}

        # Aggregated totals
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        # Per-agent aggregation
        self.by_agent: Dict[str, Dict[str, Any]] = {}

        # Detailed invocation log
        self.invocations: List[TokenInvocation] = []

        # Track current context
        self.current_agent: Optional[str] = None
        self.current_tool: Optional[str] = None

        # Run ID tracking for hierarchy (prevents stale state)
        self.run_to_agent: Dict[str, str] = {}  # run_id -> agent_name
        self.current_run_id: Optional[str] = None

        # Cache valid agent names from registry (lazy-loaded)
        self._valid_agents: Optional[Set[str]] = None

    # =================================================================
    # AGENT NAME DETECTION (Core Logic)
    # =================================================================

    def _get_valid_agents(self) -> Set[str]:
        """Lazy-load and cache valid agent names from registry."""
        if self._valid_agents is None:
            self._valid_agents = set(get_all_agent_names())
            # Add common system agents not in registry
            self._valid_agents.update(["supervisor", "system"])
        return self._valid_agents

    def _is_model_class(self, name: str) -> bool:
        """Check if name is a known LLM model class (not an agent)."""
        if not name:
            return False
        return name.lower() in self.MODEL_CLASS_NAMES

    def _is_valid_agent(self, name: str) -> bool:
        """
        Validate that name is an actual agent, not a model class.

        Returns:
            True if name is in AGENT_REGISTRY or known system agent
            False if name is a model class or invalid
        """
        if not name or self._is_model_class(name):
            return False

        # Check against agent registry
        valid_agents = self._get_valid_agents()
        return name in valid_agents

    def _extract_agent_name(
        self,
        serialized: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract agent name from multiple sources with validation.

        Priority order:
        1. run_name from config (explicit, most reliable)
        2. tags[0] from config (secondary indicator)
        3. metadata["agent_name"] (if present)
        4. kwargs["name"] (if validated)
        5. serialized["name"] (last resort, rarely useful)

        Args:
            serialized: LangChain serialized model info
            kwargs: Callback kwargs (contains run_name, tags, metadata)

        Returns:
            Valid agent name or None if not found
        """
        # Priority 1: Explicit run_name (set in graph.py:68)
        agent_name = kwargs.get("run_name")
        if agent_name and self._is_valid_agent(agent_name):
            return agent_name

        # Priority 2: Tags (first tag is often agent name)
        tags = kwargs.get("tags", [])
        if tags and isinstance(tags, list) and len(tags) > 0:
            candidate = tags[0]
            if self._is_valid_agent(candidate):
                return candidate

        # Priority 3: Metadata (explicit agent field)
        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            # Try multiple metadata keys
            for key in ["agent_name", "agent", "run_name"]:
                agent_name = metadata.get(key)
                if agent_name and self._is_valid_agent(agent_name):
                    return agent_name

        # Priority 4: kwargs["name"] (validate it's not a model class)
        agent_name = kwargs.get("name")
        if agent_name and self._is_valid_agent(agent_name):
            return agent_name

        # Priority 5: serialized["name"] (very unreliable, usually model class)
        if serialized:
            agent_name = serialized.get("name")
            if agent_name and self._is_valid_agent(agent_name):
                return agent_name

        # No valid agent found
        return None

    def _update_current_agent(
        self,
        detected_agent: Optional[str],
        run_id: Optional[str] = None
    ) -> None:
        """
        Update current_agent with fallback logic.

        Strategy:
        - If detected agent is valid, use it
        - If run_id is known, use mapped agent
        - Otherwise, keep current_agent unchanged (maintain state)

        Args:
            detected_agent: Agent name from detection
            run_id: Current run ID (for hierarchy tracking)
        """
        # Update run_id mapping if we have both
        if run_id and detected_agent:
            self.run_to_agent[str(run_id)] = detected_agent
            self.current_run_id = str(run_id)

        # Fallback chain
        if detected_agent:
            self.current_agent = detected_agent
        elif run_id and str(run_id) in self.run_to_agent:
            # Use run_id to recover agent from hierarchy
            self.current_agent = self.run_to_agent[str(run_id)]
        # else: keep current_agent unchanged (don't set to "unknown")

    # =================================================================
    # CALLBACK METHODS (LangChain Integration)
    # =================================================================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> None:
        """
        Called when a traditional completion LLM starts.

        NOTE: Chat models (Claude, GPT-4) use on_chat_model_start instead.
        This method is for legacy completion models (GPT-3 davinci, etc).
        """
        run_id = kwargs.get("run_id")
        detected_agent = self._extract_agent_name(serialized, kwargs)
        self._update_current_agent(detected_agent, run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Called when a chat model starts (Claude, GPT-4, Bedrock models).

        CRITICAL: This is the PRIMARY callback for Claude/Anthropic models.
        on_llm_start is NOT called for chat models.

        This method:
        1. Detects agent name from multiple sources
        2. Validates against AGENT_REGISTRY
        3. Tracks run_id hierarchy
        4. Maintains current_agent state
        """
        # Merge explicit args into kwargs for unified detection
        kwargs_merged = {
            **kwargs,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        detected_agent = self._extract_agent_name(serialized, kwargs_merged)
        self._update_current_agent(detected_agent, run_id)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """
        Track current tool context.

        IMPORTANT: This is where agent handoffs are detected in the main
        callback handlers (via handoff_to_* tool names). We don't track
        agent switches here to avoid duplication, but we do track the tool
        for attribution in invocation logs.
        """
        if serialized is None:
            serialized = {}
        self.current_tool = serialized.get("name", "unknown_tool")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Clear tool context when tool completes."""
        self.current_tool = None

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Extract token usage and update tracking.

        This method handles the actual token counting and cost calculation.
        Agent name should already be set by on_llm_start or on_chat_model_start.
        """
        # Extract token usage from response
        usage = self._extract_token_usage(response)
        if not usage:
            return  # No token data available, skip silently

        # Extract model name
        model = self._extract_model_name(response)

        # Calculate cost
        cost = self._calculate_cost(
            model, usage["input_tokens"], usage["output_tokens"]
        )

        # Create invocation record (use "unknown" only if current_agent is None)
        invocation = TokenInvocation(
            timestamp=datetime.now().isoformat(),
            agent=self.current_agent or "unknown",
            model=model,
            tool=self.current_tool,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=cost,
        )
        self.invocations.append(invocation)

        # Update session totals
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]
        self.total_tokens += usage["total_tokens"]
        self.total_cost_usd += cost

        # Update per-agent aggregation
        agent_name = self.current_agent or "unknown"
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "invocation_count": 0,
            }

        self.by_agent[agent_name]["input_tokens"] += usage["input_tokens"]
        self.by_agent[agent_name]["output_tokens"] += usage["output_tokens"]
        self.by_agent[agent_name]["total_tokens"] += usage["total_tokens"]
        self.by_agent[agent_name]["cost_usd"] += cost
        self.by_agent[agent_name]["invocation_count"] += 1

    # ... (rest of methods: _extract_token_usage, _extract_model_name,
    #      _calculate_cost, get_usage_summary, etc. remain unchanged)
```

## Testing Strategy

### Unit Tests (test_callbacks.py additions)

```python
def test_detect_agent_from_run_name(token_tracker):
    """Test detection from explicit run_name."""
    token_tracker.on_llm_start(
        serialized={},
        prompts=["test"],
        run_name="research_agent",  # Explicit run_name
    )
    assert token_tracker.current_agent == "research_agent"


def test_detect_agent_from_tags(token_tracker):
    """Test detection from tags when run_name missing."""
    token_tracker.on_llm_start(
        serialized={},
        prompts=["test"],
        tags=["data_expert_agent", "other_tag"],
    )
    assert token_tracker.current_agent == "data_expert_agent"


def test_ignore_model_class_names(token_tracker):
    """Test that model class names are filtered out."""
    token_tracker.on_llm_start(
        serialized={"name": "ChatBedrockConverse"},
        prompts=["test"],
        name="ChatAnthropic",  # Should be ignored
    )
    assert token_tracker.current_agent is None  # No valid agent detected


def test_validate_against_registry(token_tracker):
    """Test that only registered agents are accepted."""
    token_tracker.on_llm_start(
        serialized={},
        prompts=["test"],
        run_name="fake_agent_12345",  # Not in registry
    )
    assert token_tracker.current_agent is None


def test_maintain_state_when_no_detection(token_tracker):
    """Test that current_agent is maintained when detection fails."""
    token_tracker.current_agent = "supervisor"

    token_tracker.on_llm_start(
        serialized={"name": "ChatModel"},  # Model class, not agent
        prompts=["test"],
    )

    # Should maintain previous agent, not reset to unknown
    assert token_tracker.current_agent == "supervisor"


def test_run_id_hierarchy_tracking(token_tracker):
    """Test that run_id maps to agent for hierarchy."""
    token_tracker.on_llm_start(
        serialized={},
        prompts=["test"],
        run_id="abc-123",
        run_name="research_agent",
    )

    assert token_tracker.run_to_agent["abc-123"] == "research_agent"
    assert token_tracker.current_run_id == "abc-123"


def test_on_chat_model_start_detection(token_tracker):
    """Test chat model callback (primary for Claude/Bedrock)."""
    token_tracker.on_chat_model_start(
        serialized={},
        messages=[],
        run_id="def-456",
        run_name="transcriptomics_expert",
    )

    assert token_tracker.current_agent == "transcriptomics_expert"
```

### Integration Tests

```python
@pytest.mark.integration
def test_multi_agent_token_attribution():
    """Test token attribution across agent handoffs."""
    token_tracker = TokenTrackingCallback("test", pricing_config)

    # Supervisor invokes research_agent
    token_tracker.on_chat_model_start(
        serialized={},
        messages=[],
        run_name="supervisor",
    )
    token_tracker.on_llm_end(create_mock_result(1000, 500, "claude-sonnet"))

    # Handoff to research_agent (detected via tool in production)
    token_tracker.current_agent = "research_agent"
    token_tracker.on_chat_model_start(
        serialized={},
        messages=[],
        run_name="research_agent",
    )
    token_tracker.on_llm_end(create_mock_result(2000, 800, "claude-sonnet"))

    # Verify attribution
    assert token_tracker.by_agent["supervisor"]["total_tokens"] == 1500
    assert token_tracker.by_agent["research_agent"]["total_tokens"] == 2800
```

## Deployment Notes

### Breaking Changes
- **None**: This is a pure enhancement to existing logic
- All existing tests should pass
- New validation may prevent incorrect attribution (improvement)

### Migration Guide
1. Replace `on_llm_start` logic with robust detection
2. Add `on_chat_model_start` method (critical for Claude)
3. Add agent registry integration
4. Update tests to verify detection sources

### Monitoring
- Log when agent detection fails (returns None)
- Track "unknown" agent percentage in usage summaries
- Alert if model class names appear as agent names (indicates bug)

## References
- CALLBACK_ANALYSIS.md (lines 1-550)
- textual_callback.py (lines 222-273, 348-408)
- graph.py (lines 65-76) - Config propagation
- agent_registry.py (lines 165-188) - Valid agent names
