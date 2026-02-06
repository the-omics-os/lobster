# Hypothesis Expert Agent Module
# Synthesizes research findings into novel, evidence-linked hypotheses

from lobster.agents.hypothesis_expert.config import AGENT_CONFIG
from lobster.agents.hypothesis_expert.hypothesis_expert import hypothesis_expert
from lobster.agents.hypothesis_expert.prompts import (
    HYPOTHESIS_EXPERT_SYSTEM_PROMPT,
    create_hypothesis_expert_prompt,
)
from lobster.agents.hypothesis_expert.state import HypothesisExpertState

__all__ = [
    "hypothesis_expert",
    "AGENT_CONFIG",
    "HYPOTHESIS_EXPERT_SYSTEM_PROMPT",
    "create_hypothesis_expert_prompt",
    "HypothesisExpertState",
]
