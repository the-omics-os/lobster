"""
Mock LLM with configurable responses for testing.

This module provides a MockLLM class that mimics the interface of real LLM
objects (like ChatAnthropic, ChatBedrock) for testing agent behavior without
making actual API calls.

Example:
    >>> from lobster.testing import MockLLM
    >>> llm = MockLLM(default_response='Test response')
    >>> llm.set_response_sequence(['First', 'Second', 'Third'])
    >>> r1 = llm.invoke('prompt1')
    >>> r2 = llm.invoke('prompt2')
    >>> assert r1.content == 'First'
    >>> assert r2.content == 'Second'
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class MockLLMResponse:
    """Mock response object matching LangChain message interface.

    Attributes:
        content: The text content of the response.
        additional_kwargs: Extra data (tool calls, etc).
        response_metadata: Metadata about the response.
    """

    content: str
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Return message type (for LangChain compatibility)."""
        return "ai"


class MockLLM:
    """Mock LLM with configurable responses for testing.

    Supports multiple modes of response configuration:
    1. Default response: Simple fixed response for all prompts
    2. Response map: Keyword-based response selection
    3. Response sequence: Ordered responses for multi-turn tests

    Attributes:
        default_response: Default text to return if no match found.
        response_map: Dict mapping keywords to responses.
        response_sequence: Ordered list of responses.
        call_history: List of all prompts received.

    Example:
        >>> llm = MockLLM(default_response='Default')
        >>> llm.set_response_sequence(['A', 'B', 'C'])
        >>> llm.invoke('first')   # Returns MockLLMResponse(content='A')
        >>> llm.invoke('second')  # Returns MockLLMResponse(content='B')
        >>> len(llm.call_history)  # 2
    """

    def __init__(
        self,
        default_response: str = "Mock LLM response",
        *,
        response_map: Optional[Dict[str, str]] = None,
        model_name: str = "mock-model",
    ):
        """Initialize MockLLM.

        Args:
            default_response: Default response text for unmatched prompts.
            response_map: Optional dict mapping keywords to responses.
            model_name: Name of the mock model (for logging).
        """
        self.default_response = default_response
        self.response_map: Dict[str, str] = response_map or {}
        self.model_name = model_name

        # Response sequence for multi-turn tests
        self._response_sequence: List[str] = []
        self._sequence_index: int = 0

        # Call tracking
        self.call_history: List[Dict[str, Any]] = []

        # Config (for chaining support)
        self._config: Dict[str, Any] = {}

    def set_response_sequence(self, responses: List[str]) -> None:
        """Set an ordered sequence of responses.

        When set, invoke() will return responses in order from this sequence,
        falling back to default_response when exhausted.

        Args:
            responses: List of response strings in order.
        """
        self._response_sequence = list(responses)
        self._sequence_index = 0

    def add_response(self, keyword: str, response: str) -> None:
        """Add a keyword-based response mapping.

        Args:
            keyword: Keyword to match in prompts (case-insensitive).
            response: Response to return when keyword is found.
        """
        self.response_map[keyword.lower()] = response

    def invoke(
        self,
        prompt: Union[str, List[Any]],
        *,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> MockLLMResponse:
        """Invoke the mock LLM with a prompt.

        Response selection priority:
        1. If response sequence is set and not exhausted, use next in sequence
        2. If keyword from response_map is found in prompt, use that response
        3. Fall back to default_response

        Args:
            prompt: The prompt text or message list.
            config: Optional config dict (for LangChain compatibility).
            **kwargs: Additional arguments (ignored).

        Returns:
            MockLLMResponse with the selected content.
        """
        # Extract text from prompt
        prompt_text = self._extract_prompt_text(prompt)

        # Record call
        self.call_history.append(
            {
                "prompt": prompt_text,
                "config": config,
                "kwargs": kwargs,
            }
        )

        # Determine response
        response_text = self._select_response(prompt_text)

        return MockLLMResponse(
            content=response_text,
            response_metadata={"model": self.model_name},
        )

    def _extract_prompt_text(self, prompt: Union[str, List[Any]]) -> str:
        """Extract text content from various prompt formats.

        Args:
            prompt: String prompt or list of messages.

        Returns:
            Extracted text content.
        """
        if isinstance(prompt, str):
            return prompt

        # Handle list of messages (LangChain format)
        if isinstance(prompt, list):
            texts = []
            for msg in prompt:
                if hasattr(msg, "content"):
                    texts.append(str(msg.content))
                elif isinstance(msg, dict) and "content" in msg:
                    texts.append(str(msg["content"]))
                elif isinstance(msg, str):
                    texts.append(msg)
            return "\n".join(texts)

        return str(prompt)

    def _select_response(self, prompt_text: str) -> str:
        """Select appropriate response based on configuration.

        Args:
            prompt_text: The prompt text.

        Returns:
            Selected response text.
        """
        # Priority 1: Response sequence
        if self._response_sequence and self._sequence_index < len(
            self._response_sequence
        ):
            response = self._response_sequence[self._sequence_index]
            self._sequence_index += 1
            return response

        # Priority 2: Keyword matching
        prompt_lower = prompt_text.lower()
        for keyword, response in self.response_map.items():
            if keyword in prompt_lower:
                return response

        # Priority 3: Default response
        return self.default_response

    def with_config(self, **config) -> "MockLLM":
        """Return self with config stored (for LangChain chaining).

        Args:
            **config: Configuration options.

        Returns:
            Self (for chaining).
        """
        self._config.update(config)
        return self

    def bind_tools(self, tools: List[Any], **kwargs) -> "MockLLM":
        """Mock bind_tools for tool-calling LLMs.

        Args:
            tools: List of tools to bind (stored but not used).
            **kwargs: Additional arguments (ignored).

        Returns:
            Self (for chaining).
        """
        self._config["tools"] = tools
        return self

    def reset(self) -> None:
        """Reset the mock LLM state.

        Clears call history and resets sequence index.
        """
        self.call_history.clear()
        self._sequence_index = 0

    def get_call_count(self) -> int:
        """Get the number of calls made.

        Returns:
            Number of invoke calls.
        """
        return len(self.call_history)

    def get_last_prompt(self) -> Optional[str]:
        """Get the last prompt received.

        Returns:
            Last prompt text or None if no calls made.
        """
        if self.call_history:
            return self.call_history[-1]["prompt"]
        return None
