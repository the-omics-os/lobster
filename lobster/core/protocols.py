"""
Protocol definitions for Lobster AI core SDK.

These protocols define the contracts that agent packages implement.
Services use structural subtyping (duck typing) - no inheritance required.

Example:
    class MyService:
        def analyze(self, data, **params) -> Tuple[Any, Dict, AnalysisStep]:
            return result, stats, ir

    # MyService implicitly satisfies ServiceProtocol - no inheritance needed
"""

from typing import TYPE_CHECKING, Any, Dict, Protocol, Tuple, TypeVar, runtime_checkable

# Forward reference for AnalysisStep to avoid circular import
# Actual type comes from lobster.core.provenance
if TYPE_CHECKING:
    from lobster.core.provenance import AnalysisStep

T = TypeVar("T")


@runtime_checkable
class ServiceProtocol(Protocol):
    """
    Protocol for services that follow the 3-tuple return contract.

    All analysis services return (result, stats, ir) where:
    - result: The primary output (AnnData, DataFrame, processed data)
    - stats: Dict of human-readable statistics for logging/display
    - ir: AnalysisStep for provenance tracking and notebook export

    Services implement this implicitly via duck typing.
    No inheritance required - just match the signature.

    Example:
        class ClusteringService:
            def cluster(self, adata, resolution=1.0):
                # ... processing ...
                return processed_adata, {"n_clusters": 10}, ir

        # ClusteringService satisfies ServiceProtocol automatically
    """

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[T, Dict[str, Any], "AnalysisStep"]:
        """Execute the service operation, returning 3-tuple."""
        ...


@runtime_checkable
class StateProtocol(Protocol):
    """
    Protocol for agent state schemas.

    Agent packages extend OverallState by adding domain-specific fields.
    This protocol defines the minimum contract for state interoperability
    between supervisor and specialist agents.

    All states must have:
    - messages: Conversation history (from AgentState)
    - last_active_agent: Which agent last handled the conversation
    - conversation_id: Unique identifier for the session
    """

    messages: list
    last_active_agent: str
    conversation_id: str


__all__ = ["ServiceProtocol", "StateProtocol"]
