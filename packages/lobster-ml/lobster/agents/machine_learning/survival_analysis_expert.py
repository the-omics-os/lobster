"""
Survival Analysis Expert Agent for time-to-event modeling.

This sub-agent specializes in survival analysis including Cox proportional hazards,
Kaplan-Meier estimation, and risk stratification for biomedical data.
"""

# Agent configuration for entry point discovery (must be at top)
from lobster.agents.machine_learning.config import SURVIVAL_ANALYSIS_EXPERT_CONFIG

AGENT_CONFIG = SURVIVAL_ANALYSIS_EXPERT_CONFIG

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.machine_learning.prompts import (
    create_survival_analysis_expert_prompt,
)
from lobster.agents.machine_learning.shared_tools import create_survival_analysis_tools
from lobster.agents.machine_learning.state import SurvivalAnalysisExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def survival_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "survival_analysis_expert_agent",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Create survival analysis expert agent.

    This sub-agent handles time-to-event analysis including Cox models,
    Kaplan-Meier curves, and clinical risk stratification.

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

    # Get survival analysis tools
    survival_tools = create_survival_analysis_tools(data_manager)

    # Add helper tools
    @tool
    def check_survival_data(modality_name: str) -> str:
        """
        Check if a modality has the required columns for survival analysis.

        Args:
            modality_name: Name of the modality to check

        Returns:
            Summary of survival-relevant columns
        """
        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found."

        adata = data_manager.get_modality(modality_name)

        response = f"Survival data check for '{modality_name}':\n\n"
        response += f"**Shape**: {adata.n_obs} samples x {adata.n_vars} features\n\n"

        # Look for time columns
        time_keywords = [
            "time",
            "days",
            "months",
            "years",
            "survival",
            "duration",
            "follow",
        ]
        time_cols = [
            col
            for col in adata.obs.columns
            if any(kw in col.lower() for kw in time_keywords)
        ]

        response += "**Potential time columns**:\n"
        if time_cols:
            for col in time_cols:
                values = adata.obs[col]
                response += f"  - {col}: dtype={values.dtype}, range=[{values.min():.1f}, {values.max():.1f}]\n"
        else:
            response += (
                "  None found. Expected columns like 'time', 'days', 'survival_time'.\n"
            )

        # Look for event columns
        event_keywords = ["event", "status", "censor", "death", "disease", "outcome"]
        event_cols = [
            col
            for col in adata.obs.columns
            if any(kw in col.lower() for kw in event_keywords)
        ]

        response += "\n**Potential event columns**:\n"
        if event_cols:
            for col in event_cols:
                values = adata.obs[col]
                unique = values.unique()
                response += f"  - {col}: dtype={values.dtype}, unique values={list(unique)[:5]}\n"
        else:
            response += (
                "  None found. Expected columns like 'event', 'status', 'censored'.\n"
            )

        # Look for grouping columns (for stratified analysis)
        group_keywords = ["group", "arm", "treatment", "stage", "grade", "risk"]
        group_cols = [
            col
            for col in adata.obs.columns
            if any(kw in col.lower() for kw in group_keywords)
        ]

        response += "\n**Potential grouping columns**:\n"
        if group_cols:
            for col in group_cols:
                n_groups = adata.obs[col].nunique()
                response += f"  - {col}: {n_groups} groups\n"
        else:
            response += "  None found.\n"

        # Check for existing survival analysis results
        if "cox_risk_score" in adata.obs.columns:
            response += (
                "\n**Existing Cox model results**: Yes (obs['cox_risk_score'])\n"
            )
        if "risk_category" in adata.obs.columns:
            risk_counts = adata.obs["risk_category"].value_counts()
            response += f"\n**Risk categories**: {dict(risk_counts)}\n"
        if "kaplan_meier" in adata.uns:
            response += "\n**Kaplan-Meier curves**: Available in adata.uns\n"

        return response

    check_survival_data.metadata = {"categories": ["UTILITY"], "provenance": False}
    check_survival_data.tags = ["UTILITY"]

    @tool
    def get_hazard_ratios(modality_name: str, top_n: int = 20) -> str:
        """
        Get hazard ratios from a trained Cox model.

        Args:
            modality_name: Name of modality with Cox model results
            top_n: Number of top features to return

        Returns:
            Summary of hazard ratios
        """
        from lobster.services.analysis.survival_analysis_service import (
            SurvivalAnalysisService,
        )

        if modality_name not in data_manager.list_modalities():
            return f"Modality '{modality_name}' not found."

        adata = data_manager.get_modality(modality_name)

        if "cox_coefficient" not in adata.var.columns:
            return "No Cox model found. Train a Cox model first."

        service = SurvivalAnalysisService()
        adata_result, stats, ir = service.get_hazard_ratios(adata, top_n=top_n)

        response = f"Top {top_n} hazard ratios:\n\n"
        response += f"**Non-zero features**: {stats['n_nonzero_features']}\n\n"

        response += "| Rank | Feature | Coef | HR | Effect |\n"
        response += "|------|---------|------|-----|--------|\n"

        for f in stats["top_features"]:
            response += f"| {f['rank']} | {f['feature'][:20]} | {f['coefficient']:.3f} | {f['hazard_ratio']:.2f} | {f['effect']} |\n"

        response += "\n**Interpretation**: HR > 1 = increased risk, HR < 1 = protective"

        return response

    get_hazard_ratios.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_hazard_ratios.tags = ["UTILITY"]

    @tool
    def check_survival_availability() -> str:
        """Check if survival analysis dependencies are available."""
        from lobster.services.analysis.survival_analysis_service import (
            SurvivalAnalysisService,
        )

        service = SurvivalAnalysisService()
        avail = service.check_availability()

        if avail["ready"]:
            return "Survival analysis dependencies available. Ready to run Cox models and Kaplan-Meier analysis."
        else:
            return f"Survival analysis not available. Install with: {avail['install_command']}"

    check_survival_availability.metadata = {"categories": ["UTILITY"], "provenance": False}
    check_survival_availability.tags = ["UTILITY"]

    # Combine tools
    tools = (
        survival_tools
        + [
            check_survival_data,
            get_hazard_ratios,
            check_survival_availability,
        ]
        + (delegation_tools or [])
    )

    # System prompt
    system_prompt = create_survival_analysis_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=SurvivalAnalysisExpertState,
    )
