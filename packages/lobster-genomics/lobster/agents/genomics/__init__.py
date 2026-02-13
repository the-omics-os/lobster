# Genomics Agent Module
# Agent for WGS and SNP array genomics analysis

from lobster.agents.genomics.config import (
    AGENT_DISPLAY_NAME,
    AGENT_NAME,
    DEFAULT_HWE_PVALUE,
    DEFAULT_MIN_CALL_RATE,
    DEFAULT_MIN_MAF,
    SUPPORTED_FORMATS,
)
from lobster.agents.genomics.genomics_expert import genomics_expert
from lobster.agents.genomics.prompts import create_genomics_expert_prompt

__all__ = [
    # Main agent
    "genomics_expert",
    # Configuration
    "AGENT_NAME",
    "AGENT_DISPLAY_NAME",
    "DEFAULT_MIN_CALL_RATE",
    "DEFAULT_MIN_MAF",
    "DEFAULT_HWE_PVALUE",
    "SUPPORTED_FORMATS",
    # Prompts
    "create_genomics_expert_prompt",
]
