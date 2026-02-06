"""
Unit tests for HypothesisGenerationService.

Tests evidence doc building, markdown parsing, and IR creation without LLM calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from lobster.services.research.hypothesis_generation_service import (
    HypothesisGenerationService,
    HYPOTHESIS_GENERATION_PROMPT,
    HYPOTHESIS_UPDATE_PROMPT,
)
from lobster.core.schemas.hypothesis import Hypothesis, HypothesisEvidence


class TestHypothesisGenerationService:
    """Test suite for HypothesisGenerationService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return HypothesisGenerationService()

    @pytest.fixture
    def sample_evidence(self):
        """Sample evidence sources for testing."""
        return [
            {
                "source_type": "literature",
                "source_id": "10.1038/nature12345",
                "content": "KRAS mutations in G12C position drive drug resistance...",
            },
            {
                "source_type": "analysis",
                "source_id": "de_results_treatment_vs_control",
                "content": "Differential expression analysis revealed 500 DEGs...",
            },
            {
                "source_type": "dataset",
                "source_id": "GSE12345",
                "content": "Single-cell RNA-seq of 50,000 cells from lung adenocarcinoma...",
            },
        ]

    def test_build_evidence_docs_basic(self, service, sample_evidence):
        """Test evidence doc building with basic input."""
        docs = service._build_evidence_docs(
            evidence_sources=sample_evidence,
            current_hypothesis=None,
            key_insights=None,
            methodology=None,
        )

        assert len(docs) == 3
        assert docs[0]["title"] == "Literature 1"
        assert docs[1]["title"] == "Analysis 2"
        assert docs[2]["title"] == "Dataset 3"
        assert "10.1038/nature12345" in docs[0]["context"]

    def test_build_evidence_docs_with_hypothesis(self, service, sample_evidence):
        """Test evidence doc building includes current hypothesis for update."""
        current = "KRAS G12C inhibitors may show synergy with MEK inhibitors."

        docs = service._build_evidence_docs(
            evidence_sources=sample_evidence,
            current_hypothesis=current,
            key_insights=None,
            methodology=None,
        )

        assert len(docs) == 4
        assert docs[3]["title"] == "Current Hypothesis"
        assert "KRAS G12C" in docs[3]["text"]

    def test_build_evidence_docs_with_context(self, service, sample_evidence):
        """Test evidence doc building includes research context."""
        docs = service._build_evidence_docs(
            evidence_sources=sample_evidence,
            current_hypothesis=None,
            key_insights=["KRAS mutations common in NSCLC", "Sotorasib shows 37% ORR"],
            methodology="Literature review + single-cell analysis",
        )

        assert len(docs) == 4
        assert docs[3]["title"] == "Research Context"
        assert "KRAS mutations" in docs[3]["text"]
        assert "Literature review" in docs[3]["text"]

    def test_format_evidence_docs(self, service):
        """Test evidence docs are formatted correctly for prompt."""
        docs = [
            {"title": "Literature 1", "context": "DOI: 10.1038/xxx", "text": "Content here"},
            {"title": "Analysis 1", "context": "modality_name", "text": "Results here"},
        ]

        formatted = service._format_evidence_docs(docs)

        assert "### Literature 1" in formatted
        assert "**Context:** DOI: 10.1038/xxx" in formatted
        assert "---" in formatted  # Separator between docs

    def test_extract_markdown_sections(self, service):
        """Test markdown section extraction from LLM response."""
        response = """## Hypothesis
This is the main hypothesis statement.

## Rationale
This explains the reasoning behind the hypothesis.

## Novelty Statement
This describes what is novel.

## Experimental Design
This describes how to test it.

## Follow-Up Analyses
These are recommended next steps.
"""
        sections = service._extract_markdown_sections(response)

        assert len(sections) == 5
        assert "Hypothesis" in sections
        assert "This is the main hypothesis" in sections["Hypothesis"]
        assert "Rationale" in sections
        assert "Novelty Statement" in sections
        assert "Experimental Design" in sections
        assert "Follow-Up Analyses" in sections

    def test_extract_markdown_sections_partial(self, service):
        """Test extraction with only some sections present."""
        response = """## Hypothesis
Minimal hypothesis.

## Rationale
Minimal rationale.
"""
        sections = service._extract_markdown_sections(response)

        assert len(sections) == 2
        assert "Hypothesis" in sections
        assert "Rationale" in sections
        assert "Novelty Statement" not in sections

    def test_parse_hypothesis_response(self, service, sample_evidence):
        """Test parsing LLM response into Hypothesis object."""
        response = """## Hypothesis
KRAS G12C inhibitors combined with MEK inhibitors may improve response rates.

## Rationale
Literature shows KRAS-MEK pathway crosstalk. Analysis confirms DEGs in pathway.

## Novelty Statement
Combination approach not tested in clinical trials.

## Experimental Design
A549 cells, 4 groups (control, sotorasib, trametinib, combination).

## Follow-Up Analyses
Validate with organoid models.
"""
        hypothesis = service._parse_hypothesis_response(
            response=response,
            evidence_sources=sample_evidence,
            mode="create",
        )

        assert isinstance(hypothesis, Hypothesis)
        assert "KRAS G12C" in hypothesis.hypothesis_text
        assert "crosstalk" in hypothesis.rationale
        assert hypothesis.mode == "create"
        assert len(hypothesis.evidence) == 3

    def test_create_ir(self, service, sample_evidence):
        """Test AnalysisStep IR creation."""
        stats = {
            "mode": "create",
            "evidence_count": 3,
            "literature_sources": 1,
            "analysis_sources": 1,
            "dataset_sources": 1,
        }

        ir = service._create_ir(
            objective="Identify drug targets",
            evidence_sources=sample_evidence,
            mode="create",
            stats=stats,
        )

        assert ir.operation == "hypothesis_generation"
        assert ir.tool_name == "HypothesisGenerationService.generate_hypothesis"
        assert "3 evidence sources" in ir.description
        assert ir.parameters["objective"] == "Identify drug targets"
        assert ir.exportable is False  # Orchestration step

    @patch.object(HypothesisGenerationService, "_get_llm")
    def test_generate_hypothesis_create_mode(self, mock_get_llm, service, sample_evidence):
        """Test hypothesis generation in create mode."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="""## Hypothesis
Test hypothesis statement.

## Rationale
Test rationale.

## Novelty Statement
Test novelty.

## Experimental Design
Test design.

## Follow-Up Analyses
Test follow-up.
"""
        )
        mock_get_llm.return_value = mock_llm

        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Test objective",
            evidence_sources=sample_evidence,
            current_hypothesis=None,
        )

        assert isinstance(hypothesis, Hypothesis)
        assert stats["mode"] == "create"
        assert stats["evidence_count"] == 3
        assert stats["literature_sources"] == 1
        assert ir.operation == "hypothesis_generation"
        mock_llm.invoke.assert_called_once()

    @patch.object(HypothesisGenerationService, "_get_llm")
    def test_generate_hypothesis_update_mode(self, mock_get_llm, service, sample_evidence):
        """Test hypothesis generation in update mode with existing hypothesis."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="""## Hypothesis
Updated hypothesis statement.

## Rationale
Updated rationale with new evidence.
"""
        )
        mock_get_llm.return_value = mock_llm

        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Test objective",
            evidence_sources=sample_evidence,
            current_hypothesis="Previous hypothesis about KRAS",
        )

        assert stats["mode"] == "update"
        # Verify update prompt was used (contains current_hypothesis)
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Previous hypothesis about KRAS" in call_args or "UPDATING" in call_args


class TestHypothesisSchema:
    """Test Hypothesis Pydantic model."""

    def test_hypothesis_creation(self):
        """Test basic hypothesis creation."""
        hypothesis = Hypothesis(
            hypothesis_text="Test hypothesis",
            rationale="Test rationale",
        )

        assert hypothesis.hypothesis_text == "Test hypothesis"
        assert hypothesis.mode == "create"
        assert hypothesis.iteration == 1
        assert hypothesis.id is not None

    def test_hypothesis_with_evidence(self):
        """Test hypothesis with evidence list."""
        evidence = [
            HypothesisEvidence(
                source_type="literature",
                source_id="10.1038/nature12345",
                claim="KRAS drives resistance",
            ),
        ]

        hypothesis = Hypothesis(
            hypothesis_text="Test",
            rationale="Test",
            evidence=evidence,
        )

        assert len(hypothesis.evidence) == 1
        assert hypothesis.evidence[0].source_type == "literature"

    def test_hypothesis_formatted_output(self):
        """Test markdown formatting of hypothesis."""
        hypothesis = Hypothesis(
            hypothesis_text="Main claim",
            rationale="Supporting logic",
            novelty_statement="What is new",
            experimental_design="How to test",
            follow_up_analyses="Next steps",
            mode="create",
            iteration=1,
        )

        output = hypothesis.get_formatted_output()

        assert "## Hypothesis" in output
        assert "Main claim" in output
        assert "## Rationale" in output
        assert "## Novelty Statement" in output
        assert "## Experimental Design" in output
        assert "## Follow-Up Analyses" in output
        assert "Mode:** create" in output


class TestHypothesisPromptTemplates:
    """Test prompt template content."""

    def test_generation_prompt_has_required_sections(self):
        """Test generation prompt contains required sections."""
        assert "<Identity_And_Role>" in HYPOTHESIS_GENERATION_PROMPT
        assert "<Your_Task>" in HYPOTHESIS_GENERATION_PROMPT
        assert "<Citation_Rules>" in HYPOTHESIS_GENERATION_PROMPT
        assert "<Output_Format>" in HYPOTHESIS_GENERATION_PROMPT
        assert "<Evidence_Set>" in HYPOTHESIS_GENERATION_PROMPT

    def test_update_prompt_has_current_hypothesis_section(self):
        """Test update prompt contains current hypothesis placeholder."""
        assert "<Current_Hypothesis>" in HYPOTHESIS_UPDATE_PROMPT
        assert "{current_hypothesis}" in HYPOTHESIS_UPDATE_PROMPT

    def test_citation_format_documented(self):
        """Test citation format is documented in prompts."""
        assert "(claim)[DOI or URL]" in HYPOTHESIS_GENERATION_PROMPT
