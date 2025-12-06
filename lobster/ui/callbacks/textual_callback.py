"""
Textual-aware callback handler for bridging LangChain events to UI.

This callback handler extends TerminalCallbackHandler but instead of
printing to console, it posts Textual Messages that widgets can handle.

CRITICAL: Claude/Anthropic models use chat model callbacks, not LLM callbacks.
- on_llm_start → Traditional completion models (GPT-3 davinci)
- on_chat_model_start → Chat models (Claude, GPT-4, etc.)

We implement BOTH to ensure we capture all agent activity.

Agent Detection Strategy:
- on_chain_start is the PRIMARY method for detecting agent handoffs
- Uses get_all_agent_names() from agent registry to match chain names
- This mirrors TerminalCallbackHandler's proven approach
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from textual.message import Message

from lobster.config.agent_registry import get_all_agent_names
from lobster.utils.callbacks import EventType


class AgentActivityMessage(Message):
    """
    Posted when agent activity changes.

    Widgets can handle this via on_agent_activity_message().
    """

    bubble = True  # Allow message to bubble up to parent widgets/screens

    def __init__(
        self,
        event_type: str,
        agent_name: str,
        tool_name: Optional[str] = None,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__()
        self.event_type = event_type
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.content = content
        self.metadata = metadata or {}
        self.duration_ms = duration_ms
        self.timestamp = timestamp or datetime.now()


class TokenUsageMessage(Message):
    """
    Posted when token usage is updated.

    Widgets can handle this via on_token_usage_message().
    """

    bubble = True  # Allow message to bubble up to parent widgets/screens

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        agent_name: str = "",
        model_name: str = "",
    ):
        super().__init__()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd
        self.agent_name = agent_name
        self.model_name = model_name


class TextualCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that posts Textual Messages for UI updates.

    This handler tracks:
    - Agent starts/completions
    - Tool starts/completions
    - Agent handoffs
    - Token usage

    All events are posted as Textual Messages via call_from_thread
    for thread-safe UI updates.

    Usage:
        handler = TextualCallbackHandler(app)
        client = AgentClient(custom_callbacks=[handler])
    """

    def __init__(
        self,
        app=None,
        show_reasoning: bool = False,
        show_tools: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the Textual callback handler.

        Args:
            app: Textual App instance for posting messages
            show_reasoning: Whether to post agent thinking events
            show_tools: Whether to post tool usage events
            debug: Enable debug logging to see all callback events
        """
        self.app = app
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.debug = debug

        # State tracking
        self.current_agent: Optional[str] = None
        self.agent_stack: List[str] = []
        self.start_times: Dict[str, datetime] = {}
        self.current_tool: Optional[str] = None  # Track current tool name

        # Run ID tracking for agent hierarchy
        self.run_to_agent: Dict[str, str] = {}  # run_id -> agent_name
        self.current_run_id: Optional[str] = None

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    def _debug_log(self, method: str, **kwargs) -> None:
        """Log debug information if debug mode is enabled."""
        if self.debug and self.app:
            import json
            info = {k: str(v)[:100] for k, v in kwargs.items() if v is not None}
            try:
                self.app.call_from_thread(
                    self.app.notify,
                    f"[DEBUG] {method}: {json.dumps(info, default=str)[:200]}",
                    timeout=3,
                )
            except Exception:
                pass

    def _post_message(self, message: Message) -> None:
        """Thread-safe post message to Textual app."""
        if self.app:
            try:
                self.app.call_from_thread(self.app.post_message, message)
                # Debug: confirm message posted
                if self.debug and isinstance(message, AgentActivityMessage):
                    self._debug_log("MESSAGE_POSTED", event_type=message.event_type, agent=message.agent_name)
            except Exception as e:
                if self.debug:
                    self._debug_log("POST_ERROR", error=str(e)[:100])

    def _post_activity(
        self,
        event_type: EventType,
        agent_name: str,
        tool_name: Optional[str] = None,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Post an agent activity message."""
        self._post_message(
            AgentActivityMessage(
                event_type=event_type.value,
                agent_name=agent_name,
                tool_name=tool_name,
                content=content,
                metadata=metadata or {},
                duration_ms=duration_ms,
            )
        )

    def _post_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        agent_name: str = "",
        model_name: str = "",
    ) -> None:
        """Post a token usage update message."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

        self._post_message(
            TokenUsageMessage(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
                total_tokens=self.total_tokens,
                cost_usd=0.0,  # Cost calculated by TokenTrackingCallback
                agent_name=agent_name,
                model_name=model_name,
            )
        )

    # LangChain Callback Methods

    def _handle_model_start(
        self,
        serialized: Dict[str, Any],
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        """
        Common handler for both LLM and chat model starts.

        NOTE: This method only tracks run_ids for hierarchy.
        Agent detection happens in on_chain_start using the agent registry.
        The serialized "name" here is the MODEL name (e.g., "ChatAnthropic"),
        NOT the agent name (e.g., "research_agent").
        """
        if serialized is None:
            serialized = {}

        # Track run_id -> current_agent mapping (for hierarchy tracking)
        # Use current_agent (set by on_chain_start) rather than serialized name
        if run_id and self.current_agent:
            run_id_str = str(run_id)
            self.run_to_agent[run_id_str] = self.current_agent
            self.current_run_id = run_id_str

        # Track start time for current agent if set
        if self.current_agent:
            self.start_times[self.current_agent] = datetime.now()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when a completion LLM starts (GPT-3 davinci, etc)."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        self._handle_model_start(serialized, run_id, parent_run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Called when a chat model starts (Claude, GPT-4, etc).

        CRITICAL: This is what Claude/Anthropic models fire, NOT on_llm_start.
        """
        self._handle_model_start(serialized, run_id, parent_run_id, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when an LLM finishes."""
        if not self.current_agent:
            return

        # Calculate duration
        duration_ms = None
        if self.current_agent in self.start_times:
            delta = datetime.now() - self.start_times[self.current_agent]
            duration_ms = delta.total_seconds() * 1000
            del self.start_times[self.current_agent]

        # Extract token usage if available
        self._extract_and_post_tokens(response)

        # Show reasoning if enabled
        if self.show_reasoning and response.generations:
            if response.generations[0]:
                content = response.generations[0][0].text
                if content:
                    self._post_activity(
                        event_type=EventType.AGENT_THINKING,
                        agent_name=self.current_agent,
                        content=content[:200],  # Truncate
                    )

        # Post completion
        self._post_activity(
            event_type=EventType.AGENT_COMPLETE,
            agent_name=self.current_agent,
            duration_ms=duration_ms,
        )

    def _extract_and_post_tokens(self, response: LLMResult) -> None:
        """Extract token usage from LLMResult and post update."""
        # Try newest LangChain format
        if response.generations and len(response.generations) > 0:
            first_gen_list = response.generations[0]
            if first_gen_list and len(first_gen_list) > 0:
                generation = first_gen_list[0]
                if hasattr(generation, "message") and hasattr(
                    generation.message, "usage_metadata"
                ):
                    usage = generation.message.usage_metadata
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        if input_tokens or output_tokens:
                            self._post_token_usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                agent_name=self.current_agent or "",
                            )
                            return

        # Try llm_output format
        llm_output = response.llm_output or {}
        if "usage" in llm_output:
            usage = llm_output["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            if input_tokens or output_tokens:
                self._post_token_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agent_name=self.current_agent or "",
                )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when an LLM errors."""
        self._post_activity(
            event_type=EventType.TOOL_ERROR,
            agent_name=self.current_agent or "system",
            content=str(error)[:100],
        )

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts."""
        if serialized is None:
            serialized = {}
        tool_name = serialized.get("name", "unknown_tool")

        # Track current tool and start time
        self.current_tool = tool_name
        self.start_times[f"tool_{tool_name}"] = datetime.now()

        # Detect handoff tools - this is the PRIMARY handoff detection method
        # LangGraph supervisor uses: handoff_to_<agent_name> and transfer_back_to_supervisor
        is_handoff = False
        if tool_name.startswith("handoff_to_"):
            is_handoff = True
            # Extract target agent name from tool name
            target_agent = tool_name.replace("handoff_to_", "")
            self._debug_log("HANDOFF", from_agent=self.current_agent or "none", to_agent=target_agent, tool=tool_name)
            self._handle_handoff(target_agent)
        elif tool_name.startswith("transfer_back_to_"):
            is_handoff = True
            # Returning to supervisor
            self._debug_log("HANDOFF_BACK", from_agent=self.current_agent or "none", to_agent="supervisor", tool=tool_name)
            self._handle_handoff("supervisor")

        # Debug log all tool calls
        if not is_handoff:
            self._debug_log("TOOL", name=tool_name, agent=self.current_agent or "none")

        # Post tool start event (if show_tools enabled)
        if self.show_tools:
            # Safely convert input to string
            input_content = ""
            if input_str:
                try:
                    input_content = str(input_str)[:100]
                except Exception:
                    pass

            self._post_activity(
                event_type=EventType.TOOL_START,
                agent_name=self.current_agent or "system",
                tool_name=tool_name,
                content=input_content,
            )

    def _handle_handoff(self, target_agent: str) -> None:
        """Handle agent handoff detected from tool call."""
        if target_agent == self.current_agent:
            return  # No change

        # Post handoff event
        self._post_activity(
            event_type=EventType.HANDOFF,
            agent_name="system",
            metadata={
                "from": self.current_agent or "system",
                "to": target_agent,
            },
        )

        # Update current agent tracking
        old_agent = self.current_agent
        self.current_agent = target_agent
        if target_agent not in self.agent_stack:
            self.agent_stack.append(target_agent)

        # Track start time for new agent
        self.start_times[target_agent] = datetime.now()

        # Post agent start event
        self._post_activity(
            event_type=EventType.AGENT_START,
            agent_name=target_agent,
        )

    def on_tool_end(self, output: Any, **kwargs) -> None:
        """Called when a tool finishes."""
        if not self.show_tools:
            return

        # Use tracked tool name (more reliable than kwargs)
        tool_name = self.current_tool or "unknown_tool"

        # Calculate duration
        duration_ms = None
        tool_key = f"tool_{tool_name}"
        if tool_key in self.start_times:
            delta = datetime.now() - self.start_times[tool_key]
            duration_ms = delta.total_seconds() * 1000
            del self.start_times[tool_key]

        # Safely convert output to string
        output_content = ""
        if output:
            try:
                output_content = str(output)[:100]
            except Exception:
                pass

        self._post_activity(
            event_type=EventType.TOOL_COMPLETE,
            agent_name=self.current_agent or "system",
            tool_name=tool_name,
            content=output_content,
            duration_ms=duration_ms,
        )

        # Clear current tool
        self.current_tool = None

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when a tool errors."""
        tool_name = kwargs.get("name", "unknown_tool")
        self._post_activity(
            event_type=EventType.TOOL_ERROR,
            agent_name=self.current_agent or "system",
            tool_name=tool_name,
            content=str(error)[:100],
        )

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when an agent takes an action."""
        self._post_activity(
            event_type=EventType.AGENT_ACTION,
            agent_name=self.current_agent or "system",
            tool_name=action.tool,
            content=str(action.tool_input)[:100],
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when an agent finishes."""
        if self.agent_stack:
            self.agent_stack.pop()
            self.current_agent = self.agent_stack[-1] if self.agent_stack else None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """
        Called when a chain starts - PRIMARY method for detecting agent handoffs.

        This mirrors TerminalCallbackHandler's approach using the agent registry.
        """
        if serialized is None:
            serialized = {}
        if inputs is None:
            inputs = {}

        chain_name = serialized.get("name", "")

        # Debug logging to understand what chain names we're receiving
        self._debug_log(
            "on_chain_start",
            chain_name=chain_name,
            serialized_keys=list(serialized.keys()) if serialized else [],
        )

        # Detect agent transitions using the agent registry (dynamic, not hardcoded)
        agent_names = get_all_agent_names()

        for agent_name in agent_names:
            if agent_name in chain_name.lower():
                if agent_name != self.current_agent:
                    # This is a handoff - post the handoff event
                    self._post_activity(
                        event_type=EventType.HANDOFF,
                        agent_name="system",
                        content=inputs.get("task", "")[:100] if inputs.get("task") else "",
                        metadata={
                            "from": self.current_agent or "system",
                            "to": agent_name,
                        },
                    )

                    # Update current agent tracking
                    self.current_agent = agent_name
                    if agent_name not in self.agent_stack:
                        self.agent_stack.append(agent_name)

                    # Track start time for this agent
                    self.start_times[agent_name] = datetime.now()

                    # Post agent start event
                    self._post_activity(
                        event_type=EventType.AGENT_START,
                        agent_name=agent_name,
                    )
                break

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes."""
        pass

    def reset(self) -> None:
        """Reset all tracking state."""
        self.current_agent = None
        self.agent_stack = []
        self.start_times = {}
        self.current_tool = None
        self.run_to_agent = {}
        self.current_run_id = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
