"""
HypothesisExpert Agent for synthesizing research findings into scientific hypotheses.

This agent:
- Generates novel, evidence-linked hypotheses from literature and analysis results
- Uses the (claim)[DOI or URL] citation format
- Stores hypotheses in DataManagerV2 session metadata
- Supports both creation and update modes
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.hypothesis_expert.prompts import create_hypothesis_expert_prompt
from lobster.agents.hypothesis_expert.state import HypothesisExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.research.hypothesis_generation_service import (
    HypothesisGenerationService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class HypothesisAgentError(Exception):
    """Base exception for hypothesis agent operations."""

    pass


class EvidenceNotFoundError(HypothesisAgentError):
    """Raised when requested evidence is not found in workspace."""

    pass


def hypothesis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "hypothesis_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
):
    """
    Factory function for the hypothesis expert agent.

    This agent synthesizes research findings into novel, evidence-linked
    scientific hypotheses using the HypothesisGenerationService.

    Args:
        data_manager: DataManagerV2 instance for workspace and session management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools (not used, no sub-agents)
        workspace_path: Optional workspace path override

    Returns:
        Configured ReAct agent with hypothesis generation capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("hypothesis_expert")
    llm = create_llm("hypothesis_expert", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize service
    hypothesis_service = HypothesisGenerationService()

    # =========================================================================
    # HYPOTHESIS TOOLS
    # =========================================================================

    @tool
    def generate_hypothesis(
        objective: str,
        evidence_workspace_keys: List[str],
    ) -> str:
        """
        Generate a novel hypothesis from collected evidence sources.

        This tool synthesizes literature findings, analysis results, and dataset
        metadata into a structured scientific hypothesis with citations.

        Args:
            objective: Research question or goal for hypothesis generation.
                      Example: "Identify therapeutic targets for KRAS-mutant lung cancer"
            evidence_workspace_keys: List of workspace keys containing evidence to use.
                                    Supported key patterns:
                                    - literature_*: Literature content from research_agent
                                    - analysis_*: Analysis results from domain experts
                                    - dataset_*: Dataset metadata from data_expert
                                    - modality names: Direct modality references

        Returns:
            Formatted hypothesis with sections: Hypothesis, Rationale, Novelty Statement,
            Experimental Design, and Follow-Up Analyses. Uses (claim)[DOI] citation format.

        Example:
            generate_hypothesis(
                objective="Identify mechanisms of drug resistance in melanoma",
                evidence_workspace_keys=["literature_melanoma_2024", "analysis_de_results"]
            )
        """
        try:
            # Gather evidence from workspace
            evidence_sources = _gather_evidence(data_manager, evidence_workspace_keys)

            if not evidence_sources:
                return (
                    "Unable to generate hypothesis - No evidence found.\n\n"
                    f"Requested keys: {evidence_workspace_keys}\n\n"
                    "Use `list_evidence_sources` to see available evidence."
                )

            # Check for existing hypothesis (update mode)
            current_hypothesis = data_manager.session_data.get("current_hypothesis")

            # Get accumulated context
            key_insights = data_manager.session_data.get("key_insights")
            methodology = data_manager.session_data.get("methodology")

            # Generate hypothesis
            hypothesis, stats, ir = hypothesis_service.generate_hypothesis(
                objective=objective,
                evidence_sources=evidence_sources,
                current_hypothesis=current_hypothesis,
                key_insights=key_insights,
                methodology=methodology,
            )

            # Store hypothesis in session
            data_manager.session_data["current_hypothesis"] = hypothesis.hypothesis_text
            data_manager.session_data["hypothesis_full"] = hypothesis.model_dump()
            data_manager._save_session_metadata()

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="generate_hypothesis",
                parameters={
                    "objective": objective,
                    "evidence_keys": evidence_workspace_keys,
                    "mode": stats["mode"],
                },
                description=f"Generated hypothesis from {stats['evidence_count']} evidence sources",
                ir=ir,
            )

            # Format response
            response_parts = [
                hypothesis.get_formatted_output(),
                "",
                "---",
                "**Generation Statistics:**",
                f"- Mode: {stats['mode']}",
                f"- Evidence sources: {stats['evidence_count']}",
                f"  - Literature: {stats['literature_sources']}",
                f"  - Analysis: {stats['analysis_sources']}",
                f"  - Datasets: {stats['dataset_sources']}",
            ]

            return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return f"Hypothesis generation failed: {str(e)}"

    @tool
    def get_current_hypothesis() -> str:
        """
        Retrieve the current hypothesis from session state.

        Returns the most recently generated hypothesis if one exists,
        or a message indicating no hypothesis has been generated.

        Returns:
            Formatted hypothesis text or status message
        """
        current = data_manager.session_data.get("current_hypothesis")
        full_hypothesis = data_manager.session_data.get("hypothesis_full")

        if not current:
            return (
                "No hypothesis has been generated yet.\n\n"
                "Use `generate_hypothesis(objective, evidence_workspace_keys)` to create one."
            )

        # Format output
        if full_hypothesis:
            # Reconstruct from stored data
            iteration = full_hypothesis.get("iteration", 1)
            mode = full_hypothesis.get("mode", "create")
            evidence_count = len(full_hypothesis.get("evidence", []))

            return (
                f"**Current Hypothesis (Iteration {iteration}, Mode: {mode})**\n\n"
                f"{current}\n\n"
                f"---\n"
                f"Evidence sources used: {evidence_count}"
            )

        return f"**Current Hypothesis:**\n\n{current}"

    @tool
    def list_evidence_sources() -> str:
        """
        List available evidence sources in workspace for hypothesis generation.

        Scans workspace for literature content, analysis results, and dataset
        metadata that can be used as evidence for hypothesis generation.

        Returns:
            Categorized list of available evidence with workspace keys
        """
        evidence = {"literature": [], "analysis": [], "datasets": [], "modalities": []}

        # Check workspace for cached content
        workspace_path = data_manager.workspace_path

        # Literature workspace
        lit_path = workspace_path / "literature"
        if lit_path.exists():
            for item in lit_path.iterdir():
                if item.is_dir() or item.suffix in [".json", ".md", ".txt"]:
                    evidence["literature"].append(
                        f"literature_{item.stem}"
                    )

        # Data workspace
        data_path = workspace_path / "data"
        if data_path.exists():
            for item in data_path.iterdir():
                if item.is_dir() or item.suffix in [".json", ".md", ".txt"]:
                    evidence["datasets"].append(f"dataset_{item.stem}")

        # Metadata workspace
        meta_path = workspace_path / "metadata"
        if meta_path.exists():
            for item in meta_path.iterdir():
                if item.suffix in [".json", ".md", ".txt"]:
                    evidence["analysis"].append(f"metadata_{item.stem}")

        # Available modalities (from data_manager)
        modalities = data_manager.list_modalities()
        evidence["modalities"] = modalities

        # Format response
        response_parts = ["**Available Evidence Sources**\n"]

        if evidence["literature"]:
            response_parts.append(
                f"**Literature** ({len(evidence['literature'])}):\n"
                + "\n".join(f"  - {k}" for k in evidence["literature"][:10])
            )
            if len(evidence["literature"]) > 10:
                response_parts.append(
                    f"  ... and {len(evidence['literature']) - 10} more"
                )

        if evidence["analysis"]:
            response_parts.append(
                f"\n**Analysis Results** ({len(evidence['analysis'])}):\n"
                + "\n".join(f"  - {k}" for k in evidence["analysis"][:10])
            )

        if evidence["datasets"]:
            response_parts.append(
                f"\n**Datasets** ({len(evidence['datasets'])}):\n"
                + "\n".join(f"  - {k}" for k in evidence["datasets"][:10])
            )

        if evidence["modalities"]:
            response_parts.append(
                f"\n**Loaded Modalities** ({len(evidence['modalities'])}):\n"
                + "\n".join(f"  - {m}" for m in evidence["modalities"][:10])
            )

        if not any(evidence.values()):
            response_parts.append(
                "\nNo evidence sources found in workspace.\n\n"
                "Use research_agent to search literature or domain experts to run analyses."
            )

        return "\n".join(response_parts)

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _gather_evidence(
        dm: DataManagerV2, workspace_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """Gather evidence from workspace based on keys."""
        evidence = []
        workspace_path = dm.workspace_path

        for key in workspace_keys:
            source_data = None
            source_type = "unknown"

            # Determine source type and path
            if key.startswith("literature_"):
                source_type = "literature"
                identifier = key.replace("literature_", "")
                content = _read_workspace_content(
                    workspace_path / "literature", identifier
                )
                if content:
                    source_data = {
                        "source_type": source_type,
                        "source_id": identifier,
                        "content": content,
                    }

            elif key.startswith("analysis_") or key.startswith("metadata_"):
                source_type = "analysis"
                identifier = key.replace("analysis_", "").replace("metadata_", "")
                content = _read_workspace_content(
                    workspace_path / "metadata", identifier
                )
                if content:
                    source_data = {
                        "source_type": source_type,
                        "source_id": identifier,
                        "content": content,
                    }

            elif key.startswith("dataset_"):
                source_type = "dataset"
                identifier = key.replace("dataset_", "")
                content = _read_workspace_content(workspace_path / "data", identifier)
                if content:
                    source_data = {
                        "source_type": source_type,
                        "source_id": identifier,
                        "content": content,
                    }

            else:
                # Check if it's a modality name
                if key in dm.list_modalities():
                    adata = dm.get_modality(key)
                    source_data = {
                        "source_type": "analysis",
                        "source_id": key,
                        "content": _summarize_modality(adata, key),
                    }
                else:
                    logger.warning(f"Evidence key not found: {key}")

            if source_data:
                evidence.append(source_data)

        return evidence

    def _read_workspace_content(base_path: Path, identifier: str) -> Optional[str]:
        """Read content from workspace file or directory."""
        import json

        # Try various file patterns
        patterns = [
            base_path / f"{identifier}.json",
            base_path / f"{identifier}.md",
            base_path / f"{identifier}.txt",
            base_path / identifier / "content.json",
            base_path / identifier / "content.md",
        ]

        for path in patterns:
            if path.exists():
                try:
                    content = path.read_text()
                    if path.suffix == ".json":
                        data = json.loads(content)
                        # Extract text content from JSON
                        if isinstance(data, dict):
                            return data.get("content") or data.get("text") or str(data)
                        return str(data)
                    return content
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")

        return None

    def _summarize_modality(adata, name: str) -> str:
        """Create a text summary of a modality for evidence."""
        summary_parts = [
            f"## Modality: {name}",
            f"- Observations: {adata.n_obs}",
            f"- Variables: {adata.n_vars}",
        ]

        if hasattr(adata, "obs") and len(adata.obs.columns) > 0:
            summary_parts.append(f"- Observation columns: {list(adata.obs.columns)[:10]}")

        if hasattr(adata, "var") and len(adata.var.columns) > 0:
            summary_parts.append(f"- Variable columns: {list(adata.var.columns)[:10]}")

        if "uns" in dir(adata) and adata.uns:
            summary_parts.append(f"- Unstructured data keys: {list(adata.uns.keys())[:10]}")

        return "\n".join(summary_parts)

    # =========================================================================
    # BUILD AGENT
    # =========================================================================

    # Collect tools
    tools = [
        generate_hypothesis,
        get_current_hypothesis,
        list_evidence_sources,
    ]

    # Add delegation tools if provided (for future sub-agent support)
    if delegation_tools:
        tools.extend(delegation_tools)

    # Create system prompt
    system_prompt = create_hypothesis_expert_prompt()

    # Create ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        state_schema=HypothesisExpertState,
        name=agent_name,
    )

    return agent
