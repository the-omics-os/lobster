"""Pydantic models for prompt YAML configurations."""

from typing import Dict, List

from pydantic import BaseModel, Field


class SectionSpec(BaseModel):
    """Specification for a single prompt section.

    Attributes:
        source: Path to the markdown file (e.g., "shared/role_identity.md")
        variables: Static variables to render in this section
    """

    source: str = Field(
        ...,
        description="Path like 'shared/role_identity.md' or 'transcriptomics/capabilities.md'",
    )
    variables: Dict[str, str] = Field(
        default_factory=dict, description="Static variables for this section"
    )


class PromptConfig(BaseModel):
    """Configuration for an agent's prompt assembly.

    Attributes:
        version: Schema version for future migrations
        agent_name: Unique identifier for the agent
        description: Human-readable description of the agent
        sections: Ordered list of sections to assemble
    """

    version: str = Field(default="1.0.0", description="Prompt config schema version")
    agent_name: str = Field(..., description="Unique agent identifier")
    description: str = Field(default="", description="Human-readable agent description")
    sections: List[SectionSpec] = Field(
        default_factory=list, description="Ordered sections to assemble"
    )
