"""Output-token bound for Bedrock-family models, and the signal for a cut-off reply.

Without ``maxTokens`` Bedrock applies a default of its own that is far below the
model's real maximum, so a long answer could end at ``stopReason == "max_tokens"``
in the middle of a table row. Every Bedrock-family request therefore carries a
bound. It comes from the model's documented output maximum:

* Claude 5.5 models: 64000, the Omics output ceiling for them (the provider
  allows 128000).
* Claude 4.5 models: 64000, their documented maximum.
* Any other model: 8192, which every Bedrock chat model accepts. Set
  ``LOBSTER_BEDROCK_MAX_TOKENS`` or ``max_tokens`` in the agent config for more.

An explicit ``max_tokens`` always wins, then the setting, then this table.
"""

import logging
import os
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

OUTPUT_BOUND_ENV = "LOBSTER_BEDROCK_MAX_TOKENS"

_FALLBACK_BOUND = 8192
_DOCUMENTED_BOUNDS = (
    ("claude-sonnet-5-5", 64000),
    ("claude-haiku-5-5", 64000),
    ("claude-opus-5-5", 64000),
    ("claude-sonnet-4-5", 64000),
    ("claude-haiku-4-5", 64000),
    ("claude-opus-4-5", 64000),
)
# Stop reasons that mean the reply ended at the length limit, by provider.
_LENGTH_STOPS = frozenset({"max_tokens", "max_output_tokens", "length"})


def default_output_bound(model_id: str) -> int:
    """The output-token bound to send when the caller sets none."""
    raw = os.environ.get(OUTPUT_BOUND_ENV, "").strip()
    if raw:
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        logger.warning(
            "Ignoring invalid %s=%r (need a positive integer)", OUTPUT_BOUND_ENV, raw
        )
    for family, bound in _DOCUMENTED_BOUNDS:
        if family in model_id:
            return bound
    return _FALLBACK_BOUND


def above_thinking_budget(bound: int, additional_fields: Optional[dict]) -> int:
    """Keep a defaulted bound above an extended-thinking budget (providers refuse less)."""
    thinking = (additional_fields or {}).get("thinking")
    budget = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
    if isinstance(budget, int) and bound <= budget:
        return budget + _FALLBACK_BOUND
    return bound


def stop_reason_of(response_metadata: Any) -> Optional[str]:
    """The stop reason a message's metadata carries, if any (Bedrock or Anthropic key)."""
    if not isinstance(response_metadata, dict):
        return None
    return (
        response_metadata.get("stopReason")
        or response_metadata.get("stop_reason")
        or response_metadata.get("finish_reason")
    )


def stopped_at_length(response_metadata: Any) -> bool:
    """Whether a message's metadata says the reply ended at the length limit."""
    return stop_reason_of(response_metadata) in _LENGTH_STOPS


def truncation_note() -> str:
    """The plain note that ends a turn whose final reply was cut off."""
    return "\n\n[The answer was cut off at the length limit. Ask me to continue.]"


class OutputLimitWatch(BaseCallbackHandler):
    """Logs a WARNING, naming the agent and the bound, when a reply is cut off."""

    def __init__(self, agent_name: Optional[str], bound: Optional[int]):
        self.agent_name = agent_name
        self.bound = bound

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                if stopped_at_length(getattr(message, "response_metadata", None)):
                    logger.warning(
                        "Reply cut off at the output limit: agent=%s bound=%s tokens",
                        self.agent_name or "unknown",
                        self.bound if self.bound is not None else "unknown",
                    )
                    return


def attach_output_limit_watch(llm: Any, agent_name: Optional[str]) -> None:
    """Add the watch to a model's own callbacks; a model that cannot take one is left alone."""
    callbacks = getattr(llm, "callbacks", None)
    if callbacks is not None and not isinstance(callbacks, list):
        return
    bound = getattr(llm, "max_tokens", None)
    watch = OutputLimitWatch(agent_name, bound if isinstance(bound, int) else None)
    try:
        llm.callbacks = [*(callbacks or []), watch]
    except Exception:
        logger.debug("Could not attach the output-limit watch", exc_info=True)
