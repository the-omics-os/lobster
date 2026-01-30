"""
Analysis services for Lobster AI.

This module provides stateless analysis services for various omics data types.
All services return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking.
"""

from lobster.services.analysis.proteomics_network_service import (
    WGCNALiteService,
    ProteomicsNetworkError,
)
from lobster.services.analysis.proteomics_survival_service import (
    ProteomicsSurvivalService,
    ProteomicsSurvivalError,
)

__all__ = [
    # Proteomics network (WGCNA)
    "WGCNALiteService",
    "ProteomicsNetworkError",
    # Proteomics survival
    "ProteomicsSurvivalService",
    "ProteomicsSurvivalError",
]
