"""
Hypothesis schema definitions for the HypothesisExpert agent.

This module provides Pydantic models for structured hypothesis representation,
evidence tracking, and hypothesis generation requests. Follows the BioAgents
hypothesis pattern adapted for Lobster's architecture.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class HypothesisEvidence(BaseModel):
    """
    Evidence supporting a hypothesis claim.

    Each piece of evidence comes from a specific source (literature, analysis,
    or dataset) and contributes to the overall hypothesis rationale.

    Attributes:
        source_type: Type of evidence source - "literature", "analysis", or "dataset"
        source_id: Identifier for the source (DOI, modality name, or workspace key)
        claim: Specific claim or finding from this source
        confidence: Confidence score for this evidence (0.0-1.0)
        metadata: Optional additional context about the evidence
    """

    source_type: str = Field(
        ...,
        description="Type of evidence: 'literature', 'analysis', or 'dataset'",
    )
    source_id: str = Field(
        ...,
        description="Identifier: DOI, modality name, or workspace key",
    )
    claim: str = Field(
        ...,
        description="Specific claim or finding from this source",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for this evidence",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the evidence",
    )


class Hypothesis(BaseModel):
    """
    Structured scientific hypothesis with evidence and provenance.

    A hypothesis synthesizes multiple evidence sources into a novel, testable
    scientific claim. It includes rationale, novelty assessment, experimental
    design suggestions, and follow-up analysis recommendations.

    Citation format: (claim)[DOI or URL]
    Example: "KRAS mutations drive resistance (studies confirm)[10.1038/xxx]"

    Attributes:
        id: Unique identifier for this hypothesis
        hypothesis_text: The main hypothesis statement (2-4 sentences)
        rationale: Evidence synthesis and logical reasoning (3-5 sentences)
        novelty_statement: What makes this hypothesis novel (2-3 sentences)
        experimental_design: Suggested validation approach
        follow_up_analyses: Recommended additional analyses
        evidence: List of supporting evidence sources
        mode: "create" for new hypothesis, "update" for refinement
        created_at: Timestamp when hypothesis was created
        updated_at: Timestamp of last update (None if never updated)
        iteration: Version number (increments on update)
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this hypothesis",
    )
    hypothesis_text: str = Field(
        ...,
        description="The main hypothesis statement (2-4 sentences)",
    )
    rationale: str = Field(
        ...,
        description="Evidence synthesis and logical reasoning (3-5 sentences)",
    )
    novelty_statement: str = Field(
        default="",
        description="What makes this hypothesis novel (2-3 sentences)",
    )
    experimental_design: Optional[str] = Field(
        default=None,
        description="Suggested experimental validation approach",
    )
    follow_up_analyses: Optional[str] = Field(
        default=None,
        description="Recommended additional analyses",
    )
    evidence: List[HypothesisEvidence] = Field(
        default_factory=list,
        description="List of supporting evidence sources",
    )
    mode: str = Field(
        default="create",
        description="'create' for new hypothesis, 'update' for refinement",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when hypothesis was created",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last update",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Version number (increments on update)",
    )

    def get_formatted_output(self) -> str:
        """
        Format hypothesis as markdown for display.

        Returns:
            Formatted markdown string with all hypothesis sections
        """
        sections = [
            f"## Hypothesis\n{self.hypothesis_text}",
            f"## Rationale\n{self.rationale}",
        ]

        if self.novelty_statement:
            sections.append(f"## Novelty Statement\n{self.novelty_statement}")

        if self.experimental_design:
            sections.append(f"## Experimental Design\n{self.experimental_design}")

        if self.follow_up_analyses:
            sections.append(f"## Follow-Up Analyses\n{self.follow_up_analyses}")

        # Add metadata footer
        sections.append(
            f"---\n**Mode:** {self.mode} | **Evidence Sources:** {len(self.evidence)} | "
            f"**Iteration:** {self.iteration}"
        )

        return "\n\n".join(sections)


class HypothesisGenerationRequest(BaseModel):
    """
    Request parameters for hypothesis generation.

    Encapsulates all inputs needed to generate or update a hypothesis,
    including the research objective, evidence sources, and context.

    Attributes:
        objective: Research question or objective
        evidence_sources: List of evidence dictionaries with source info
        current_hypothesis: Existing hypothesis to update (None for create)
        key_insights: Previously accumulated research insights
        methodology: Current research methodology description
    """

    objective: str = Field(
        ...,
        description="Research question or objective",
    )
    evidence_sources: List[Dict[str, Any]] = Field(
        ...,
        description="List of evidence with source_type, source_id, content",
    )
    current_hypothesis: Optional[str] = Field(
        default=None,
        description="Existing hypothesis to update (None for new)",
    )
    key_insights: Optional[List[str]] = Field(
        default=None,
        description="Previously accumulated research insights",
    )
    methodology: Optional[str] = Field(
        default=None,
        description="Current research methodology",
    )
