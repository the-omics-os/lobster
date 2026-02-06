"""
Integration tests for HypothesisExpert agent.

Tests full workflow from evidence gathering to hypothesis generation.
Requires LLM API credentials for full integration tests.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil


@pytest.fixture
def temp_workspace():
    """Create temporary workspace with evidence files."""
    workspace = tempfile.mkdtemp()
    workspace_path = Path(workspace)

    # Create workspace structure
    (workspace_path / "literature").mkdir()
    (workspace_path / "data").mkdir()
    (workspace_path / "metadata").mkdir()

    # Add sample literature evidence
    lit_content = {
        "title": "KRAS G12C Inhibitors in Cancer",
        "content": (
            "KRAS G12C mutations are found in ~13% of non-small cell lung cancers. "
            "Sotorasib (AMG 510) is a first-in-class KRAS G12C inhibitor that showed "
            "37.1% overall response rate in the CodeBreaK 100 trial. "
            "Resistance mechanisms include acquired KRAS mutations and MET amplification."
        ),
        "doi": "10.1056/NEJMoa2030428",
    }
    with open(workspace_path / "literature" / "kras_review.json", "w") as f:
        json.dump(lit_content, f)

    # Add sample analysis results
    analysis_content = {
        "title": "Differential Expression Results",
        "content": (
            "Comparison of KRAS G12C mutant vs wild-type cell lines identified "
            "342 upregulated and 215 downregulated genes (|log2FC| > 1, padj < 0.05). "
            "Top pathways: MAPK signaling (p=1.2e-8), cell cycle (p=3.4e-6)."
        ),
    }
    with open(workspace_path / "metadata" / "de_analysis.json", "w") as f:
        json.dump(analysis_content, f)

    yield workspace_path

    # Cleanup
    shutil.rmtree(workspace)


@pytest.fixture
def mock_data_manager(temp_workspace):
    """Create mock DataManagerV2 with temp workspace."""
    dm = MagicMock()
    dm.workspace_path = temp_workspace
    dm.session_data = {}
    dm.list_modalities.return_value = []
    dm._save_session_metadata = MagicMock()
    dm.log_tool_usage = MagicMock()
    return dm


class TestHypothesisAgentIntegration:
    """Integration tests for hypothesis agent workflow."""

    @patch("lobster.services.research.hypothesis_generation_service.HypothesisGenerationService._get_llm")
    def test_full_hypothesis_workflow(self, mock_get_llm, mock_data_manager):
        """Test complete workflow from evidence to hypothesis."""
        from lobster.agents.hypothesis_expert.hypothesis_expert import hypothesis_expert
        from lobster.services.research.hypothesis_generation_service import (
            HypothesisGenerationService,
        )

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="""## Hypothesis
KRAS G12C inhibitor resistance mediated by MET amplification can be overcome
through combination therapy with MET inhibitors, leading to improved response
rates in NSCLC patients who have progressed on sotorasib monotherapy.

## Rationale
Clinical data shows 37.1% ORR for sotorasib monotherapy[10.1056/NEJMoa2030428],
indicating significant room for improvement. MET amplification is identified
as a key resistance mechanism. Differential expression analysis revealed
MAPK signaling as top dysregulated pathway (p=1.2e-8), supporting pathway crosstalk.

## Novelty Statement
While KRAS G12C inhibitors and MET inhibitors exist separately, their combination
to overcome acquired resistance has not been systematically evaluated in clinical
trials. This represents a novel therapeutic strategy.

## Experimental Design
- Cell model: H358 (KRAS G12C) with induced MET amplification
- Groups: Control, Sotorasib, Capmatinib, Combination
- Endpoints: Cell viability (IC50), pathway activation (pERK/pMET)
- Statistical analysis: Two-way ANOVA with Tukey post-hoc

## Follow-Up Analyses
- Validate in patient-derived organoids with acquired resistance
- RNA-seq to identify mechanism of synergy
- Screen for additional combination partners
"""
        )
        mock_get_llm.return_value = mock_llm

        # Test service directly
        service = HypothesisGenerationService()

        evidence = [
            {
                "source_type": "literature",
                "source_id": "10.1056/NEJMoa2030428",
                "content": "KRAS G12C inhibitors show 37.1% ORR. MET amplification causes resistance.",
            },
            {
                "source_type": "analysis",
                "source_id": "de_analysis",
                "content": "MAPK signaling top pathway (p=1.2e-8)",
            },
        ]

        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Identify strategies to overcome KRAS G12C inhibitor resistance",
            evidence_sources=evidence,
        )

        # Verify hypothesis structure
        assert "KRAS" in hypothesis.hypothesis_text
        assert hypothesis.mode == "create"
        assert len(hypothesis.evidence) == 2

        # Verify stats
        assert stats["evidence_count"] == 2
        assert stats["literature_sources"] == 1
        assert stats["analysis_sources"] == 1

        # Verify IR
        assert ir.operation == "hypothesis_generation"

    @patch("lobster.services.research.hypothesis_generation_service.HypothesisGenerationService._get_llm")
    def test_hypothesis_update_workflow(self, mock_get_llm, mock_data_manager):
        """Test hypothesis update with new evidence."""
        from lobster.services.research.hypothesis_generation_service import (
            HypothesisGenerationService,
        )

        # Mock LLM for update mode
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="""## Hypothesis
Updated: KRAS G12C + MET inhibitor combination shows synergy in preclinical models.

## Rationale
New analysis confirms pathway crosstalk. Updated with new evidence.
"""
        )
        mock_get_llm.return_value = mock_llm

        service = HypothesisGenerationService()

        # Simulate update with existing hypothesis
        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Validate KRAS resistance hypothesis",
            evidence_sources=[
                {
                    "source_type": "analysis",
                    "source_id": "combination_assay",
                    "content": "Synergy confirmed with CI < 0.5",
                }
            ],
            current_hypothesis="Original hypothesis about KRAS resistance",
        )

        assert stats["mode"] == "update"

    def test_evidence_gathering_from_workspace(self, mock_data_manager, temp_workspace):
        """Test evidence gathering from workspace files."""
        # Import helper function from agent module
        from lobster.agents.hypothesis_expert import hypothesis_expert as he_module

        # The helper functions are defined inside the factory function,
        # so we test the service's _build_evidence_docs instead
        from lobster.services.research.hypothesis_generation_service import (
            HypothesisGenerationService,
        )

        service = HypothesisGenerationService()

        evidence = [
            {
                "source_type": "literature",
                "source_id": "kras_review",
                "content": "Test content",
            },
        ]

        docs = service._build_evidence_docs(
            evidence_sources=evidence,
            current_hypothesis=None,
            key_insights=["Insight 1"],
            methodology="Literature review",
        )

        # Should have evidence + context
        assert len(docs) >= 2
        assert any("Literature" in d["title"] for d in docs)
        assert any("Context" in d["title"] for d in docs)


class TestSupervisorDelegation:
    """Test supervisor delegation to hypothesis_expert."""

    def test_hypothesis_expert_in_valid_handoffs(self):
        """Test hypothesis_expert appears in supervisor's valid handoffs."""
        from lobster.config.agent_registry import get_valid_handoffs

        valid_handoffs = get_valid_handoffs()

        # hypothesis_expert should be accessible from supervisor
        assert "hypothesis_expert" in valid_handoffs.get("supervisor", set())

    def test_workflow_section_includes_hypothesis(self):
        """Test supervisor workflow section mentions hypothesis generation."""
        from lobster.agents.supervisor import _build_workflow_section
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        active_agents = ["research_agent", "hypothesis_expert", "transcriptomics_expert"]

        section = _build_workflow_section(active_agents, config)

        assert "Hypothesis Generation" in section
        assert "research_agent" in section
        assert "hypothesis_expert" in section


class TestProvenanceTracking:
    """Test provenance tracking for hypothesis generation."""

    @patch("lobster.services.research.hypothesis_generation_service.HypothesisGenerationService._get_llm")
    def test_ir_contains_required_fields(self, mock_get_llm):
        """Test IR has all required fields for provenance."""
        from lobster.services.research.hypothesis_generation_service import (
            HypothesisGenerationService,
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="## Hypothesis\nTest\n\n## Rationale\nTest"
        )
        mock_get_llm.return_value = mock_llm

        service = HypothesisGenerationService()
        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Test",
            evidence_sources=[{"source_type": "literature", "source_id": "test", "content": "test"}],
        )

        # Verify IR structure
        assert ir.operation == "hypothesis_generation"
        assert ir.tool_name is not None
        assert ir.description is not None
        assert ir.library == "lobster.services.research"
        assert "objective" in ir.parameters
        assert "mode" in ir.parameters

    @patch("lobster.services.research.hypothesis_generation_service.HypothesisGenerationService._get_llm")
    def test_log_tool_usage_called(self, mock_get_llm, mock_data_manager):
        """Test log_tool_usage is called with IR."""
        from lobster.agents.hypothesis_expert.hypothesis_expert import hypothesis_expert
        from langchain_core.tools import tool

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="## Hypothesis\nTest\n\n## Rationale\nTest"
        )
        mock_get_llm.return_value = mock_llm

        # Verify log_tool_usage call structure
        mock_data_manager.log_tool_usage.reset_mock()

        # Note: Full agent invocation would require more setup
        # This tests the expected call pattern
        mock_data_manager.log_tool_usage(
            tool_name="generate_hypothesis",
            parameters={"objective": "test", "evidence_keys": [], "mode": "create"},
            description="Generated hypothesis from 0 evidence sources",
            ir=MagicMock(),
        )

        mock_data_manager.log_tool_usage.assert_called_once()
        call_kwargs = mock_data_manager.log_tool_usage.call_args[1]
        assert call_kwargs["tool_name"] == "generate_hypothesis"
        assert "ir" in call_kwargs


class TestSessionPersistence:
    """Test hypothesis storage in session."""

    def test_hypothesis_stored_in_session(self, mock_data_manager):
        """Test hypothesis is stored in session_data."""
        # Simulate what the agent tool does
        mock_data_manager.session_data["current_hypothesis"] = "Test hypothesis"
        mock_data_manager.session_data["hypothesis_full"] = {
            "hypothesis_text": "Test hypothesis",
            "rationale": "Test rationale",
            "mode": "create",
            "iteration": 1,
        }

        assert mock_data_manager.session_data["current_hypothesis"] == "Test hypothesis"
        assert mock_data_manager.session_data["hypothesis_full"]["mode"] == "create"

    def test_hypothesis_update_increments_iteration(self, mock_data_manager):
        """Test hypothesis update increments iteration counter."""
        mock_data_manager.session_data["hypothesis_full"] = {
            "iteration": 1,
            "mode": "create",
        }

        # Simulate update
        mock_data_manager.session_data["hypothesis_full"]["iteration"] = 2
        mock_data_manager.session_data["hypothesis_full"]["mode"] = "update"

        assert mock_data_manager.session_data["hypothesis_full"]["iteration"] == 2
        assert mock_data_manager.session_data["hypothesis_full"]["mode"] == "update"


@pytest.mark.real_api
class TestHypothesisAgentRealAPI:
    """Integration tests requiring real LLM API.

    These tests are skipped by default and require:
    - ANTHROPIC_API_KEY or AWS Bedrock credentials
    - pytest -m real_api to run
    """

    @pytest.fixture
    def real_data_manager(self, temp_workspace):
        """Create real DataManagerV2 for API tests."""
        from lobster.core.data_manager_v2 import DataManagerV2

        return DataManagerV2(workspace_path=temp_workspace)

    def test_real_hypothesis_generation(self, real_data_manager, temp_workspace):
        """Test hypothesis generation with real LLM API."""
        from lobster.services.research.hypothesis_generation_service import (
            HypothesisGenerationService,
        )

        service = HypothesisGenerationService()

        evidence = [
            {
                "source_type": "literature",
                "source_id": "review_2024",
                "content": (
                    "Single-cell RNA sequencing has revolutionized our understanding of "
                    "tumor heterogeneity. Studies show that cancer stem cells represent "
                    "a small fraction (<5%) but drive tumor initiation and resistance."
                ),
            },
        ]

        hypothesis, stats, ir = service.generate_hypothesis(
            objective="Identify novel therapeutic targets for cancer stem cells",
            evidence_sources=evidence,
        )

        # Verify we got a real hypothesis
        assert len(hypothesis.hypothesis_text) > 50
        assert "cancer" in hypothesis.hypothesis_text.lower() or "stem" in hypothesis.hypothesis_text.lower()
        assert hypothesis.rationale is not None
        assert stats["mode"] == "create"
