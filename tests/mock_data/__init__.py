"""
Mock data generation utilities for Lobster AI testing framework.

This module provides comprehensive synthetic biological data generation
that maintains statistical properties of real datasets while being
completely reproducible for testing purposes.
"""

from .base import MockDataConfig, TestDataRegistry
from .factories import (
    BulkRNASeqDataFactory,
    MultiModalDataFactory,
    ProteomicsDataFactory,
    SingleCellDataFactory,
)
from .generators import (
    generate_mock_geo_response,
    generate_synthetic_bulk_rnaseq,
    generate_synthetic_proteomics,
    generate_synthetic_single_cell,
    generate_test_workspace_state,
)

__all__ = [
    # Factories
    "SingleCellDataFactory",
    "BulkRNASeqDataFactory",
    "ProteomicsDataFactory",
    "MultiModalDataFactory",
    # Generators
    "generate_synthetic_single_cell",
    "generate_synthetic_bulk_rnaseq",
    "generate_synthetic_proteomics",
    "generate_mock_geo_response",
    "generate_test_workspace_state",
    # Configuration
    "MockDataConfig",
    "TestDataRegistry",
]
