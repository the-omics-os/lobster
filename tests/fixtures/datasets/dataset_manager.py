"""
DatasetManager helper class for test access.

Provides convenient methods to access datasets from YAML config in tests.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class DatasetManager:
    """Helper class for accessing test datasets from configuration."""

    def __init__(self, config_path: Path):
        """Initialize manager from YAML config."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.datasets_dir = config_path.parent

    def get(self, path: str) -> Path:
        """
        Get dataset path by hierarchical key.

        Args:
            path: Hierarchical path to dataset
                  Format: 'omics_type/analysis_type/platform/dataset_id'
                  Example: 'single_cell/clustering/10x_chromium/GSE132044'

        Returns:
            Path to dataset directory or file

        Raises:
            ValueError: If path format is invalid or dataset not found
        """
        parts = path.split("/")
        if len(parts) != 4:
            raise ValueError(
                f"Path must have 4 parts: omics_type/analysis_type/platform/dataset_id\n"
                f"Got: {path}"
            )

        omics_type, analysis_type, platform, dataset_id = parts

        # Find dataset in config
        dataset = self._find_dataset(omics_type, analysis_type, platform, dataset_id)
        if not dataset:
            raise ValueError(
                f"Dataset not found: {path}\n"
                f"Check datasets.yml for available datasets"
            )

        # Return path based on structure
        return self.datasets_dir / omics_type / analysis_type / platform / dataset_id

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """
        Get full metadata for dataset from config.

        Args:
            path: Hierarchical path (same format as get())

        Returns:
            Dictionary with all dataset metadata from YAML

        Example:
            >>> dm.get_metadata('single_cell/clustering/10x_chromium/GSE132044')
            {
                'id': 'GSE132044',
                'description': 'Melanoma, ~15,000 cells',
                'format': '10x_mtx',
                'expected_cells': 15000,
                ...
            }
        """
        parts = path.split("/")
        if len(parts) != 4:
            raise ValueError(f"Invalid path format: {path}")

        omics_type, analysis_type, platform, dataset_id = parts
        dataset = self._find_dataset(omics_type, analysis_type, platform, dataset_id)

        if not dataset:
            raise ValueError(f"Dataset not found: {path}")

        return dataset

    def list_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Find all datasets with specific test tag.

        Args:
            tag: Tag to search for (e.g., 'clustering', 'transpose_correctness')

        Returns:
            List of dataset metadata dictionaries with matching tag

        Example:
            >>> dm.list_by_tag('transpose_correctness')
            [
                {'id': 'GSE132044', 'test_tags': ['format_preservation', ...]},
                {'id': 'GSE147507', 'test_tags': ['bulk_rnaseq', ...]},
                ...
            ]
        """
        results = []
        for omics_type, omics_data in self.config.items():
            for analysis_type, analysis_data in omics_data.items():
                for platform, datasets in analysis_data.items():
                    for dataset in datasets:
                        tags = dataset.get("test_tags", [])
                        if tag in tags:
                            # Add hierarchical path for easy get() access
                            dataset_with_path = dataset.copy()
                            dataset_with_path["_path"] = (
                                f"{omics_type}/{analysis_type}/{platform}/{dataset['id']}"
                            )
                            results.append(dataset_with_path)
        return results

    def list_by_type(
        self, omics_type: str, analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all datasets for omics/analysis type.

        Args:
            omics_type: Omics type ('single_cell', 'bulk_rnaseq', 'edge_cases')
            analysis_type: Optional analysis type filter

        Returns:
            List of dataset metadata dictionaries

        Example:
            >>> dm.list_by_type('bulk_rnaseq', 'differential_expression')
            [
                {'id': 'GSE147507', ...},
                {'id': 'GSE130036', ...}
            ]
        """
        results = []
        omics_data = self.config.get(omics_type, {})

        for analysis_key, analysis_data in omics_data.items():
            if analysis_type and analysis_key != analysis_type:
                continue

            for platform, datasets in analysis_data.items():
                for dataset in datasets:
                    dataset_with_path = dataset.copy()
                    dataset_with_path["_path"] = (
                        f"{omics_type}/{analysis_key}/{platform}/{dataset['id']}"
                    )
                    results.append(dataset_with_path)

        return results

    def _find_dataset(
        self, omics_type: str, analysis_type: str, platform: str, dataset_id: str
    ) -> Optional[Dict[str, Any]]:
        """Internal: Find dataset in config by hierarchical keys."""
        try:
            omics_data = self.config[omics_type]
            analysis_data = omics_data[analysis_type]
            datasets = analysis_data[platform]

            for dataset in datasets:
                if dataset["id"] == dataset_id:
                    return dataset
        except KeyError:
            return None
        return None


# Global singleton instance
_manager: Optional[DatasetManager] = None


def get_dataset_manager(config_path: Optional[Path] = None) -> DatasetManager:
    """
    Get global DatasetManager instance.

    Args:
        config_path: Optional path to datasets.yml (auto-detected if None)

    Returns:
        Singleton DatasetManager instance
    """
    global _manager

    if _manager is None:
        if config_path is None:
            # Auto-detect config path relative to this file
            config_path = Path(__file__).parent / "datasets.yml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"datasets.yml not found at {config_path}\n"
                f"Run download_datasets.py to set up test data"
            )

        _manager = DatasetManager(config_path)

    return _manager
