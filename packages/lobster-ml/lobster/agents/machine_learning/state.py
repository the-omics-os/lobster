"""
State definitions for the machine learning expert agents.

This module defines the state classes for ML agents which handle
feature engineering, data splitting, survival analysis, and ML framework exports.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "MachineLearningExpertState",
    "FeatureSelectionExpertState",
    "SurvivalAnalysisExpertState",
]


class MachineLearningExpertState(AgentState):
    """
    State for the machine learning expert agent.

    This agent prepares biological data for ML model training, handling
    feature engineering, data splitting, and framework-specific exports.
    """

    next: str = ""

    # Machine learning specific context
    task_description: str = ""  # Description of the current task
    ml_ready_modalities: Dict[str, Any] = {}  # Assessment of modalities ready for ML
    feature_engineering_results: Dict[str, Any] = {}  # Feature preparation outcomes
    data_splits: Dict[str, Any] = {}  # Train/test/validation split information
    exported_datasets: Dict[str, Any] = {}  # Framework export results and paths
    ml_metadata: Dict[str, Any] = {}  # ML-specific metadata and preprocessing info
    framework_exports: List[str] = []  # List of export formats and paths
    file_paths: List[str] = []  # Paths to ML-ready files
    methodology_parameters: Dict[str, Any] = {}  # ML method parameters used
    data_context: str = ""  # ML data context and characteristics
    intermediate_outputs: Dict[str, Any] = {}  # For partial ML computations


class FeatureSelectionExpertState(AgentState):
    """
    State for the feature selection expert agent.

    This agent handles biomarker discovery through stability-based
    feature selection, LASSO/Elastic Net, and importance ranking.
    """

    next: str = ""

    # Feature selection context
    task_description: str = ""
    selection_method: str = ""  # stability, lasso, elastic_net, xgboost
    selected_features: List[str] = []  # Final selected feature names
    feature_rankings: Dict[str, float] = {}  # Feature -> importance score
    stability_scores: Dict[str, float] = {}  # Feature -> stability across bootstraps
    selection_metadata: Dict[str, Any] = {}  # Method parameters, thresholds
    intermediate_outputs: Dict[str, Any] = {}


class SurvivalAnalysisExpertState(AgentState):
    """
    State for the survival analysis expert agent.

    This agent handles time-to-event analysis including Cox PH models,
    Kaplan-Meier curves, and risk stratification.
    """

    next: str = ""

    # Survival analysis context
    task_description: str = ""
    time_column: str = ""  # Column with time-to-event
    event_column: str = ""  # Column with event indicator (0/1)
    model_type: str = ""  # coxph, kaplan_meier, gradient_boosting
    hazard_ratios: Dict[str, float] = {}  # Feature -> hazard ratio
    survival_curves: Dict[str, Any] = {}  # Group -> survival function
    risk_scores: Dict[str, float] = {}  # Sample -> risk score
    model_metrics: Dict[str, Any] = {}  # C-index, time-dependent AUC
    threshold_info: Dict[str, Any] = {}  # Optimal threshold, sensitivity/specificity
    intermediate_outputs: Dict[str, Any] = {}
