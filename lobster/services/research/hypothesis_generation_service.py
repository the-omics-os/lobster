"""
Hypothesis Generation Service

Stateless service that synthesizes research findings into novel, evidence-linked
hypotheses. Follows Lobster's service pattern returning 3-tuple:
(Hypothesis, stats_dict, AnalysisStep)

Adapted from BioAgents hypothesis agent architecture.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.schemas.hypothesis import Hypothesis, HypothesisEvidence
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# Prompt templates for hypothesis generation
HYPOTHESIS_GENERATION_PROMPT = """<Identity_And_Role>
You are a HYPOTHESIS GENERATION EXPERT within Lobster AI.
You synthesize research findings into novel, evidence-linked scientific hypotheses.
</Identity_And_Role>

<Your_Task>
Using the Evidence Set provided, generate ONE comprehensive hypothesis that:
- Is specific (population/system, intervention/exposure, comparator, endpoint)
- Is falsifiable (clear direction; measurable outcome)
- Is experimentally actionable (feasible assay or protocol)
- Is significantly novel: synthesize across multiple sources to propose genuinely new directions
- Addresses the research goals: {question}
</Your_Task>

<Novelty_Requirements>
CRITICAL: This is deep research; aim for HIGH novelty.
- Do not just restate existing findings
- Synthesize across multiple papers/sources to identify gaps, contradictions, or unexplored combinations
- Propose new mechanistic links, intervention strategies, biomarker approaches, or translational pathways
- If combining interventions, explain synergistic rationale with evidence
- Explicitly note what makes this hypothesis novel compared to existing literature
</Novelty_Requirements>

<Citation_Rules>
- Cite DOIs or URLs that appear verbatim in the Evidence Set (from literature sources)
- For analysis results (computational data, statistics, gene expression), reference directly without DOIs
- Place inline citations immediately after the clause they support: (claim)[DOI or URL]
- Example: "KRAS mutations drive therapy resistance (multiple studies confirm this)[10.1038/nature12345]"
</Citation_Rules>

<Output_Format>
Write exactly these sections in markdown:

## Hypothesis
2-4 sentences. Name the system/population, variables, direction of effect, and experimental method.
Include inline citations in (claim)[DOI or URL] format when available.

## Rationale
3-5 sentences that:
- Connect specific findings from multiple sources
- Explain the logical synthesis enabling this novel hypothesis
- Identify the gap or opportunity this hypothesis addresses
- Include inline citations for literature claims
- Reference analysis results directly (e.g., "Based on differential expression analysis showing...")

## Novelty Statement
2-3 sentences explicitly describing:
- What is novel about this hypothesis
- What gap it fills or new angle it explores
- Why this has not been tested before (if applicable)

## Experimental Design
3-5 sentences including:
- Experimental unit/system and groups (with controls)
- Primary endpoint(s) and measurement methods
- Secondary endpoints or exploratory analyses
- Planned statistical test (e.g., t-test, ANOVA, mixed-effects)
- Sample size considerations

## Follow-Up Analyses
1-3 sentences suggesting:
- Molecular/proteomic/genomic analyses for mechanism validation
- Computational analyses for outcome prediction
- Precedent searches to confirm novelty
</Output_Format>

<Evidence_Set>
{evidence_docs}
</Evidence_Set>

<Constraints>
- Use ONLY the Evidence Set for factual claims and citations
- Novelty must be HIGH and arise from multi-source synthesis
- No extra sections or explanations outside the format above
- If insufficient evidence, respond: "Unable to generate hypothesis - Insufficient evidence"
</Constraints>
"""

HYPOTHESIS_UPDATE_PROMPT = """<Identity_And_Role>
You are a HYPOTHESIS GENERATION EXPERT within Lobster AI.
You are UPDATING an existing hypothesis with new evidence.
</Identity_And_Role>

<Your_Task>
You have an existing hypothesis that needs refinement based on new evidence.
Your task is to UPDATE (not replace) the hypothesis to incorporate new findings while:
- Maintaining consistency with previously established claims
- Integrating new evidence that supports, refines, or challenges the hypothesis
- Strengthening the novelty statement with additional context
- Updating experimental design if new methods are suggested
</Your_Task>

<Current_Hypothesis>
{current_hypothesis}
</Current_Hypothesis>

<New_Evidence>
{evidence_docs}
</New_Evidence>

<Research_Question>
{question}
</Research_Question>

<Output_Format>
Write exactly these sections in markdown (same as creation, but UPDATED):

## Hypothesis
[Updated 2-4 sentences incorporating new evidence]

## Rationale
[Updated 3-5 sentences with new supporting evidence]

## Novelty Statement
[Updated 2-3 sentences, noting what new evidence adds]

## Experimental Design
[Updated if new methods suggested]

## Follow-Up Analyses
[Updated based on new findings]
</Output_Format>

<Constraints>
- Preserve valid claims from original hypothesis
- Add new evidence citations using (claim)[DOI or URL] format
- Explicitly note what changed and why in the Rationale
- If new evidence contradicts hypothesis, note the contradiction
</Constraints>
"""


class HypothesisGenerationService:
    """
    Service for generating and updating research hypotheses.

    Follows Lobster service pattern:
    - Stateless (no side effects beyond LLM calls)
    - Returns (Hypothesis, stats_dict, AnalysisStep) tuple
    - AnalysisStep enables provenance tracking and notebook export
    """

    def __init__(self):
        """Initialize the hypothesis generation service."""
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM to avoid import-time initialization."""
        if self._llm is None:
            from lobster.config.llm_factory import create_llm
            from lobster.config.settings import get_settings

            settings = get_settings()
            model_params = settings.get_agent_llm_params("hypothesis_expert")
            self._llm = create_llm("hypothesis_expert", model_params)
        return self._llm

    def generate_hypothesis(
        self,
        objective: str,
        evidence_sources: List[Dict[str, Any]],
        current_hypothesis: Optional[str] = None,
        key_insights: Optional[List[str]] = None,
        methodology: Optional[str] = None,
    ) -> Tuple[Hypothesis, Dict[str, Any], AnalysisStep]:
        """
        Generate or update a research hypothesis from evidence.

        Args:
            objective: Research objective/question
            evidence_sources: List of evidence dicts with keys:
                - source_type: "literature" | "analysis" | "dataset"
                - source_id: DOI, modality name, or workspace key
                - content: Text content from source
                - metadata: Optional additional context
            current_hypothesis: Existing hypothesis to update (None for create)
            key_insights: Previously accumulated insights
            methodology: Current research methodology

        Returns:
            Tuple of (Hypothesis, stats_dict, AnalysisStep)
        """
        mode = "update" if current_hypothesis else "create"

        logger.info(
            f"Generating hypothesis (mode={mode}, evidence_count={len(evidence_sources)})"
        )

        # Build evidence documents for LLM
        evidence_docs = self._build_evidence_docs(
            evidence_sources, current_hypothesis, key_insights, methodology
        )

        # Select and format prompt based on mode
        if mode == "update":
            formatted_prompt = HYPOTHESIS_UPDATE_PROMPT.format(
                question=objective,
                evidence_docs=self._format_evidence_docs(evidence_docs),
                current_hypothesis=current_hypothesis or "",
            )
        else:
            formatted_prompt = HYPOTHESIS_GENERATION_PROMPT.format(
                question=objective,
                evidence_docs=self._format_evidence_docs(evidence_docs),
            )

        # Generate hypothesis via LLM
        llm = self._get_llm()
        response = llm.invoke(formatted_prompt)

        # Extract content from response
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse structured output
        hypothesis = self._parse_hypothesis_response(
            response_text, evidence_sources, mode
        )

        # Build statistics
        stats = {
            "mode": mode,
            "evidence_count": len(evidence_sources),
            "literature_sources": sum(
                1 for e in evidence_sources if e.get("source_type") == "literature"
            ),
            "analysis_sources": sum(
                1 for e in evidence_sources if e.get("source_type") == "analysis"
            ),
            "dataset_sources": sum(
                1 for e in evidence_sources if e.get("source_type") == "dataset"
            ),
            "hypothesis_length": len(hypothesis.hypothesis_text),
            "has_novelty_statement": bool(hypothesis.novelty_statement),
            "has_experimental_design": bool(hypothesis.experimental_design),
            "iteration": hypothesis.iteration,
        }

        # Build IR for provenance/notebook export
        ir = self._create_ir(objective, evidence_sources, mode, stats)

        logger.info(
            f"Hypothesis generated: {len(hypothesis.hypothesis_text)} chars, "
            f"{len(hypothesis.evidence)} evidence sources"
        )

        return hypothesis, stats, ir

    def _build_evidence_docs(
        self,
        evidence_sources: List[Dict[str, Any]],
        current_hypothesis: Optional[str],
        key_insights: Optional[List[str]],
        methodology: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build document list for LLM context."""
        docs = []

        # Add evidence sources
        for i, source in enumerate(evidence_sources):
            source_type = source.get("source_type", "Source").title()
            source_id = source.get("source_id", "unknown")
            content = source.get("content", "")

            docs.append(
                {
                    "title": f"{source_type} {i + 1}",
                    "text": content,
                    "context": f"Source ID: {source_id}",
                }
            )

        # Add current hypothesis if updating
        if current_hypothesis:
            docs.append(
                {
                    "title": "Current Hypothesis",
                    "text": current_hypothesis,
                    "context": "Existing hypothesis to be updated with new findings",
                }
            )

        # Add research context
        context_parts = []
        if key_insights:
            insights_text = "\n".join(f"- {insight}" for insight in key_insights)
            context_parts.append(f"Key Insights:\n{insights_text}")
        if methodology:
            context_parts.append(f"Methodology: {methodology}")

        if context_parts:
            docs.append(
                {
                    "title": "Research Context",
                    "text": "\n\n".join(context_parts),
                    "context": "Overall research context",
                }
            )

        return docs

    def _format_evidence_docs(self, docs: List[Dict[str, str]]) -> str:
        """Format documents for LLM prompt."""
        formatted = []
        for doc in docs:
            formatted.append(
                f"### {doc['title']}\n**Context:** {doc['context']}\n\n{doc['text']}"
            )
        return "\n\n---\n\n".join(formatted)

    def _parse_hypothesis_response(
        self,
        response: str,
        evidence_sources: List[Dict[str, Any]],
        mode: str,
    ) -> Hypothesis:
        """Parse LLM response into structured Hypothesis."""
        # Extract sections from markdown response
        sections = self._extract_markdown_sections(response)

        # Build evidence list
        evidence = [
            HypothesisEvidence(
                source_type=src.get("source_type", "unknown"),
                source_id=src.get("source_id", "unknown"),
                claim=src.get("content", "")[:200],  # First 200 chars
                confidence=0.8,  # Default confidence
                metadata=src.get("metadata"),
            )
            for src in evidence_sources
        ]

        return Hypothesis(
            id=str(uuid4()),
            hypothesis_text=sections.get("Hypothesis", ""),
            rationale=sections.get("Rationale", ""),
            novelty_statement=sections.get("Novelty Statement", ""),
            experimental_design=sections.get("Experimental Design"),
            follow_up_analyses=sections.get("Follow-Up Analyses"),
            evidence=evidence,
            mode=mode,
            created_at=datetime.utcnow(),
            iteration=1,
        )

    def _extract_markdown_sections(self, text: str) -> Dict[str, str]:
        """Extract ## sections from markdown text."""
        sections = {}
        pattern = r"##\s+(.+?)\n([\s\S]*?)(?=\n##|\Z)"
        for match in re.finditer(pattern, text):
            section_name = match.group(1).strip()
            section_content = match.group(2).strip()
            sections[section_name] = section_content
        return sections

    def _create_ir(
        self,
        objective: str,
        evidence_sources: List[Dict[str, Any]],
        mode: str,
        stats: Dict[str, Any],
    ) -> AnalysisStep:
        """Create AnalysisStep IR for provenance tracking."""
        return AnalysisStep(
            operation="hypothesis_generation",
            tool_name="HypothesisGenerationService.generate_hypothesis",
            description=(
                f"Generated {'updated ' if mode == 'update' else ''}hypothesis "
                f"from {len(evidence_sources)} evidence sources"
            ),
            library="lobster.services.research",
            code_template="""# Hypothesis Generation
# This step synthesizes evidence into a research hypothesis
# Mode: {{ mode }}
# Evidence sources: {{ evidence_count }}
#
# The hypothesis was generated using LLM synthesis of:
# - Literature sources: {{ literature_sources }}
# - Analysis results: {{ analysis_sources }}
# - Dataset sources: {{ dataset_sources }}
#
# Output stored in session state as current_hypothesis
""",
            imports=[],
            parameters={
                "objective": objective,
                "mode": mode,
                "evidence_count": len(evidence_sources),
            },
            parameter_schema={
                "objective": ParameterSpec(
                    param_type="str",
                    papermill_injectable=False,
                    default_value=objective,
                    required=True,
                    description="Research objective/question",
                ),
                "mode": ParameterSpec(
                    param_type="str",
                    papermill_injectable=False,
                    default_value=mode,
                    required=True,
                    description="create or update",
                ),
            },
            input_entities=["evidence_sources"],
            output_entities=["hypothesis"],
            execution_context={"stats": stats},
            exportable=False,  # Orchestration step, not executable code
        )
