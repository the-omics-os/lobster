"""
Mock DataManagerV2 with realistic in-memory behavior for testing.

This module provides a MockDataManager class that implements the same interface
as DataManagerV2 but operates entirely in-memory for fast, isolated testing.

CRITICAL: workspace_path is Path type (not str) per Phase 5 decision 05-02.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

try:
    import anndata as ad
except ImportError:
    ad = None  # Will fail at runtime if anndata not available


class MockProvenanceTracker:
    """Mock ProvenanceTracker for IR tracking in tests."""

    def __init__(self):
        self.steps: List[Any] = []
        self.current_entity_id: Optional[str] = None

    def add_step(self, step: Any) -> None:
        """Record a provenance step."""
        self.steps.append(step)

    def get_history(self) -> List[Any]:
        """Return recorded provenance history."""
        return self.steps.copy()

    def clear(self) -> None:
        """Clear all recorded steps."""
        self.steps.clear()


class MockDataManager:
    """Mock DataManagerV2 with realistic in-memory behavior.

    Provides a fully mocked DataManagerV2 instance with working methods
    for testing agents and services without real data persistence.

    Attributes:
        workspace_path: Path to workspace directory (MUST be Path type)
        modalities: Dict of loaded datasets (name -> AnnData)
        metadata_store: Dict of dataset metadata
        latest_plots: List of generated plot data
        tool_usage_history: List of tool usage records
        provenance_tracker: Mock provenance tracker for IR

    Example:
        >>> from lobster.testing import MockDataManager
        >>> from pathlib import Path
        >>> dm = MockDataManager(workspace_path=Path('/tmp/test'))
        >>> dm.add_modality('test_data', adata)
        >>> assert 'test_data' in dm.list_modalities()
    """

    def __init__(
        self,
        workspace_path: Path,
        *,
        modalities: Optional[Dict[str, Any]] = None,
        metadata_store: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MockDataManager.

        Args:
            workspace_path: Path to workspace directory. MUST be Path type.
            modalities: Optional pre-populated modalities dict.
            metadata_store: Optional pre-populated metadata dict.

        Raises:
            TypeError: If workspace_path is not a Path instance.
        """
        if not isinstance(workspace_path, Path):
            raise TypeError(
                f"workspace_path must be Path, got {type(workspace_path).__name__}. "
                "Per Phase 5 decision 05-02, workspace_path is always Path type."
            )

        # Core properties
        self.workspace_path: Path = workspace_path
        self.modalities: Dict[str, Any] = modalities if modalities is not None else {}
        self.metadata_store: Dict[str, Any] = (
            metadata_store if metadata_store is not None else {}
        )

        # Tracking collections
        self.latest_plots: List[Dict[str, Any]] = []
        self.tool_usage_history: List[Dict[str, Any]] = []

        # Provenance tracking
        self.provenance_tracker = MockProvenanceTracker()

        # Mock callback handler (used by some agents)
        self.callback_handler = MagicMock()

    # =========================================================================
    # Modality Management
    # =========================================================================

    def list_modalities(self) -> List[str]:
        """List all available modality names.

        Returns:
            List of modality names currently in memory.
        """
        return list(self.modalities.keys())

    def get_modality_ids(self) -> List[str]:
        """Get all modality IDs (alias for list_modalities).

        Returns:
            List of modality IDs.
        """
        return self.list_modalities()

    def get_modality(self, name: str) -> Optional[Any]:
        """Get a modality by name.

        Args:
            name: Name of the modality to retrieve.

        Returns:
            AnnData object or None if not found.
        """
        return self.modalities.get(name)

    def add_modality(self, name: str, data: Any) -> None:
        """Add a modality to the store.

        Args:
            name: Name for the modality.
            data: AnnData object to store.
        """
        self.modalities[name] = data

    def remove_modality(self, name: str) -> Optional[Any]:
        """Remove a modality from the store.

        Args:
            name: Name of the modality to remove.

        Returns:
            Removed AnnData object or None if not found.
        """
        return self.modalities.pop(name, None)

    def has_modality(self, name: str) -> bool:
        """Check if a modality exists.

        Args:
            name: Name of the modality.

        Returns:
            True if modality exists, False otherwise.
        """
        return name in self.modalities

    # =========================================================================
    # File Operations (Mocked)
    # =========================================================================

    def save_modality(self, name: str, **kwargs) -> bool:
        """Mock save modality to disk.

        Args:
            name: Name of the modality to save.
            **kwargs: Additional save parameters (ignored in mock).

        Returns:
            True (always succeeds in mock).
        """
        return True

    def load_modality(self, name: str, **kwargs) -> bool:
        """Mock load modality from disk.

        Args:
            name: Name of the modality to load.
            **kwargs: Additional load parameters (ignored in mock).

        Returns:
            True (always succeeds in mock).
        """
        return True

    def export_workspace(self, **kwargs) -> Path:
        """Mock export workspace to archive.

        Returns:
            Path to mock export file.
        """
        return self.workspace_path / "export.zip"

    # =========================================================================
    # Provenance & Tool Usage
    # =========================================================================

    def log_tool_usage(
        self,
        tool_name: str,
        params: Dict[str, Any],
        stats: Dict[str, Any],
        *,
        ir: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Log tool usage for provenance tracking.

        Args:
            tool_name: Name of the tool that was used.
            params: Parameters passed to the tool.
            stats: Statistics/results from the tool.
            ir: Optional AnalysisStep IR for provenance.
            **kwargs: Additional metadata.
        """
        entry = {
            "tool_name": tool_name,
            "params": params,
            "stats": stats,
            "ir": ir,
            **kwargs,
        }
        self.tool_usage_history.append(entry)

        if ir is not None:
            self.provenance_tracker.add_step(ir)

    def get_tool_usage_history(self) -> List[Dict[str, Any]]:
        """Get the tool usage history.

        Returns:
            List of tool usage records.
        """
        return self.tool_usage_history.copy()

    # =========================================================================
    # Plot Management
    # =========================================================================

    def add_plot(self, plot_data: Dict[str, Any]) -> None:
        """Add a plot to the latest plots list.

        Args:
            plot_data: Plot data dictionary.
        """
        self.latest_plots.append(plot_data)

    def get_latest_plots(self) -> List[Dict[str, Any]]:
        """Get the latest plots.

        Returns:
            List of plot data dictionaries.
        """
        return self.latest_plots.copy()

    def clear_plots(self) -> None:
        """Clear the latest plots list."""
        self.latest_plots.clear()

    # =========================================================================
    # Metadata Management
    # =========================================================================

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a modality.

        Args:
            name: Name of the modality.

        Returns:
            Metadata dict or None if not found.
        """
        return self.metadata_store.get(name)

    def set_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Set metadata for a modality.

        Args:
            name: Name of the modality.
            metadata: Metadata dictionary.
        """
        self.metadata_store[name] = metadata
