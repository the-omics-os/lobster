"""
Feature Selection Expert Agent for biomarker discovery.

This sub-agent specializes in feature selection methods for high-dimensional
biological data, including stability-based selection, LASSO, and variance filtering.
"""

# Agent configuration for entry point discovery (must be at top)
from lobster.agents.machine_learning.config import FEATURE_SELECTION_EXPERT_CONFIG

AGENT_CONFIG = FEATURE_SELECTION_EXPERT_CONFIG

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.machine_learning.prompts import (
    create_feature_selection_expert_prompt,
)
from lobster.agents.machine_learning.shared_tools import create_feature_selection_tools
from lobster.agents.machine_learning.state import FeatureSelectionExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_tool import create_list_modalities_tool
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def feature_selection_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "feature_selection_expert_agent",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Create feature selection expert agent.

    This sub-agent handles biomarker discovery through various
    feature selection methods optimized for high-dimensional omics data.

    Args:
        data_manager: DataManagerV2 instance for data access
        callback_handler: Optional callback for LLM events
        agent_name: Name for this agent instance
        delegation_tools: List of delegation tools for handoffs
        workspace_path: Optional workspace path
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params(agent_name)
    llm = create_llm(
        agent_name,
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Get feature selection tools
    fs_tools = create_feature_selection_tools(data_manager)

    # Use shared tool from workspace_tool.py (consistent with data_expert and supervisor)
    list_available_modalities = create_list_modalities_tool(data_manager)

    @tool
    def get_feature_selection_results(modality_name: str) -> str:
        """Get feature selection results from a processed modality."""
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found."

        adata = data_manager.get_modality(modality_name)

        response = f"Feature selection results for '{modality_name}':\n\n"

        # Check for selection results in var
        selection_cols = [
            col
            for col in adata.var.columns
            if "select" in col.lower() or "importance" in col.lower()
        ]

        if not selection_cols:
            return f"No feature selection results found in '{modality_name}'. Run selection first."

        for col in selection_cols:
            response += f"- {col}: {adata.var[col].dtype}\n"

        # Show top features if importance available
        if "feature_mean_importance" in adata.var.columns:
            top_features = adata.var.nlargest(10, "feature_mean_importance")
            response += "\n**Top 10 Features by Importance**:\n"
            for i, (feat, row) in enumerate(top_features.iterrows(), 1):
                response += f"  {i}. {feat}: {row['feature_mean_importance']:.4f}\n"

        return response

    get_feature_selection_results.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_feature_selection_results.tags = ["UTILITY"]

    # Combine tools
    tools = (
        fs_tools
        + [list_available_modalities, get_feature_selection_results]
        + (delegation_tools or [])
    )

    # System prompt
    system_prompt = create_feature_selection_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=FeatureSelectionExpertState,
    )
