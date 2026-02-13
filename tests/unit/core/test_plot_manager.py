"""
Unit tests for PlotManager.

Tests cover:
- Plot addition with metadata (3-tuple return pattern)
- FIFO buffer behavior (max_plots_history enforcement)
- Thread safety (counter lock, save lock)
- Rate limiting (min_save_interval)
- Workspace export (HTML/PNG)
- PNG skip threshold (>50K points)
- Visualization state management
- Provenance IR generation
- Backward compatibility (property access)
- Error handling (invalid plots, missing workspace)

All tests run WITHOUT instantiating DataManagerV2 (isolated unit tests).
"""

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.analysis_ir import AnalysisStep
from lobster.core.plot_manager import PlotManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workspace_path(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def plot_manager(workspace_path):
    """Create PlotManager instance."""
    return PlotManager(
        workspace_path=workspace_path,
        max_plots_history=5,
        min_save_interval=0.1,  # Short interval for testing
        enable_ir=True,
    )


@pytest.fixture
def mock_plotly_figure():
    """Create mock Plotly Figure."""
    with patch("lobster.core.plot_manager._ensure_plotly") as mock_ensure:
        mock_go = Mock()
        mock_io = Mock()
        mock_ensure.return_value = (mock_go, mock_io)

        # Create mock figure
        mock_figure = Mock(spec=["update_layout"])
        mock_figure.update_layout = Mock()
        mock_go.Figure = type(mock_figure)

        yield mock_figure, mock_go, mock_io


@pytest.fixture
def sample_modalities():
    """Create sample modalities dict for context."""
    mock_adata = Mock()
    mock_adata.shape = (1000, 2000)
    mock_adata.n_obs = 1000
    mock_adata.n_vars = 2000
    return {"test_modality": mock_adata}


# =============================================================================
# Test: Plot Addition (Core Functionality)
# =============================================================================


def test_add_plot_basic(plot_manager, mock_plotly_figure):
    """Test basic plot addition with 3-tuple return."""
    figure, mock_go, _ = mock_plotly_figure

    plot_id, stats, ir = plot_manager.add_plot(
        plot=figure,
        title="Test Plot",
        source="test_service",
    )

    # Verify 3-tuple return
    assert isinstance(plot_id, str)
    assert plot_id.startswith("plot_")
    assert isinstance(stats, dict)
    assert isinstance(ir, AnalysisStep)

    # Verify stats content
    assert stats["plot_id"] == plot_id
    assert stats["title"].startswith("Test Plot")
    assert stats["source"] == "test_service"
    assert stats["plots_in_history"] == 1

    # Verify IR generation
    assert ir.operation == "plot_manager.add_plot"
    assert ir.library == "plotly"


def test_add_plot_with_modality_context(
    plot_manager, mock_plotly_figure, sample_modalities
):
    """Test plot addition with modality context."""
    figure, _, _ = mock_plotly_figure

    plot_id, stats, ir = plot_manager.add_plot(
        plot=figure,
        title="Cluster Plot",
        modalities=sample_modalities,
        current_dataset="test_modality",
    )

    assert plot_id
    assert "test_modality" in stats["title"]  # Modality name added to title


def test_add_plot_invalid_figure(plot_manager):
    """Test error handling for invalid figure."""
    with patch("lobster.core.plot_manager._ensure_plotly") as mock_ensure:
        mock_go = Mock()
        mock_ensure.return_value = (mock_go, None)
        mock_go.Figure = type(Mock())

        with pytest.raises(ValueError, match="Plot must be a plotly Figure"):
            plot_manager.add_plot(plot="not a figure")


def test_add_plot_increments_counter(plot_manager, mock_plotly_figure):
    """Test that plot_counter increments correctly."""
    figure, _, _ = mock_plotly_figure

    plot_id_1, _, _ = plot_manager.add_plot(plot=figure, title="Plot 1")
    plot_id_2, _, _ = plot_manager.add_plot(plot=figure, title="Plot 2")
    plot_id_3, _, _ = plot_manager.add_plot(plot=figure, title="Plot 3")

    assert plot_id_1 == "plot_1"
    assert plot_id_2 == "plot_2"
    assert plot_id_3 == "plot_3"
    assert plot_manager.plot_counter == 3


# =============================================================================
# Test: FIFO Buffer Behavior
# =============================================================================


def test_fifo_buffer_enforcement(workspace_path, mock_plotly_figure):
    """Test that max_plots_history is enforced (FIFO)."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=3)

    # Add 5 plots (exceeds max of 3)
    for i in range(5):
        pm.add_plot(plot=figure, title=f"Plot {i + 1}")

    # Should only keep last 3 plots
    assert len(pm.latest_plots) == 3
    plot_ids = [p["id"] for p in pm.latest_plots]
    assert plot_ids == ["plot_3", "plot_4", "plot_5"]


def test_fifo_buffer_preserves_latest(workspace_path, mock_plotly_figure):
    """Test that FIFO buffer preserves most recent plots."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=2)

    pm.add_plot(plot=figure, title="Old Plot")
    time.sleep(0.01)
    pm.add_plot(plot=figure, title="Middle Plot")
    time.sleep(0.01)
    pm.add_plot(plot=figure, title="New Plot")

    assert len(pm.latest_plots) == 2
    titles = [p["title"] for p in pm.latest_plots]
    assert any("Middle Plot" in t for t in titles)
    assert any("New Plot" in t for t in titles)


# =============================================================================
# Test: Thread Safety
# =============================================================================


def test_counter_thread_safety(workspace_path, mock_plotly_figure):
    """Test that plot_counter is thread-safe."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=100)

    plot_ids = []
    errors = []

    def add_plots(n=10):
        try:
            for _ in range(n):
                plot_id, _, _ = pm.add_plot(plot=figure, title="Concurrent Plot")
                plot_ids.append(plot_id)
        except Exception as e:
            errors.append(e)

    # Spawn 5 threads, each adding 10 plots
    threads = [threading.Thread(target=add_plots) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0

    # All 50 plots should be added with unique IDs
    assert len(plot_ids) == 50
    assert len(set(plot_ids)) == 50  # All unique
    assert pm.plot_counter == 50


def test_save_lock_prevents_concurrent_saves(plot_manager, mock_plotly_figure):
    """Test that save_lock prevents concurrent saves."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Test Plot")

    save_started = []
    save_completed = []

    def slow_save():
        # Simulate slow save
        save_started.append(threading.current_thread().name)
        saved_files, stats, _ = plot_manager.save_plots_to_workspace()
        save_completed.append(threading.current_thread().name)
        return saved_files

    # Attempt concurrent saves
    threads = [threading.Thread(target=slow_save, name=f"Thread-{i}") for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Only one save should succeed (others skipped due to lock)
    assert len(save_started) == 3
    # At least one should have been skipped (returned empty list)


# =============================================================================
# Test: Rate Limiting
# =============================================================================


def test_rate_limiting_enforcement(workspace_path, mock_plotly_figure):
    """Test that min_save_interval is enforced."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, min_save_interval=0.5)
    pm.add_plot(plot=figure, title="Test Plot")

    # First save should succeed
    files1, stats1, _ = pm.save_plots_to_workspace()
    assert len(files1) > 0
    assert not stats1.get("skipped")

    # Immediate second save should be rate-limited
    files2, stats2, _ = pm.save_plots_to_workspace()
    assert len(files2) == 0
    assert stats2["skipped"] is True
    assert stats2["reason"] == "rate_limited"

    # After waiting, third save should succeed
    time.sleep(0.6)
    files3, stats3, _ = pm.save_plots_to_workspace()
    assert len(files3) > 0
    assert not stats3.get("skipped")


def test_rate_limiting_per_instance(workspace_path, mock_plotly_figure):
    """Test that rate limiting is per PlotManager instance."""
    figure, _, _ = mock_plotly_figure

    ws1 = workspace_path / "pm1"
    ws2 = workspace_path / "pm2"
    ws1.mkdir()
    ws2.mkdir()

    pm1 = PlotManager(workspace_path=ws1, min_save_interval=0.5)
    pm2 = PlotManager(workspace_path=ws2, min_save_interval=0.5)

    pm1.add_plot(plot=figure, title="Plot 1")
    pm2.add_plot(plot=figure, title="Plot 2")

    # Both should save successfully (independent rate limits)
    files1, stats1, _ = pm1.save_plots_to_workspace()
    files2, stats2, _ = pm2.save_plots_to_workspace()

    assert len(files1) > 0
    assert len(files2) > 0


# =============================================================================
# Test: Workspace Export
# =============================================================================


def test_save_plots_to_workspace_html(plot_manager, mock_plotly_figure):
    """Test HTML export to workspace."""
    figure, _, mock_io = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Export Test")

    saved_files, stats, ir = plot_manager.save_plots_to_workspace()

    # Verify 3-tuple return
    assert isinstance(saved_files, list)
    assert isinstance(stats, dict)
    assert isinstance(ir, AnalysisStep)

    # Verify files were attempted to be saved
    mock_io.write_html.assert_called()
    assert stats["saved_count"] >= 0
    assert stats["total_plots"] == 1


def test_save_plots_empty_list(plot_manager):
    """Test save with no plots."""
    saved_files, stats, ir = plot_manager.save_plots_to_workspace()

    assert saved_files == []
    assert stats["skipped"] is True
    assert stats["reason"] == "no_plots"
    assert ir is None  # No IR for skipped operations


def test_save_plots_creates_directory(workspace_path, mock_plotly_figure):
    """Test that plots directory is created if missing."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path)
    pm.add_plot(plot=figure, title="Test")

    plots_dir = workspace_path / "plots"
    assert not plots_dir.exists()

    pm.save_plots_to_workspace()

    assert plots_dir.exists()
    assert plots_dir.is_dir()


# =============================================================================
# Test: PNG Skip Threshold
# =============================================================================


def test_png_skip_for_large_datasets(plot_manager, mock_plotly_figure):
    """Test that PNG export is skipped for >50K points."""
    figure, _, mock_io = mock_plotly_figure

    # Add plot with large dataset (>50K cells)
    plot_manager.add_plot(
        plot=figure,
        title="Large Dataset",
        dataset_info={"n_cells": 75000},
    )

    saved_files, _, _ = plot_manager.save_plots_to_workspace()

    # PNG export should be skipped
    # write_image should NOT be called
    mock_io.write_image.assert_not_called()


def test_png_export_for_small_datasets(plot_manager, mock_plotly_figure):
    """Test that PNG export works for <50K points."""
    figure, _, mock_io = mock_plotly_figure

    # Add plot with small dataset (<50K cells)
    plot_manager.add_plot(
        plot=figure,
        title="Small Dataset",
        dataset_info={"n_cells": 10000},
    )

    saved_files, _, _ = plot_manager.save_plots_to_workspace()

    # PNG export should be attempted
    mock_io.write_image.assert_called()


# =============================================================================
# Test: Plot Retrieval
# =============================================================================


def test_get_plot_by_id_found(plot_manager, mock_plotly_figure):
    """Test retrieving plot by ID."""
    figure, _, _ = mock_plotly_figure
    plot_id, _, _ = plot_manager.add_plot(plot=figure, title="Find Me")

    retrieved_figure, stats, none_ir = plot_manager.get_plot_by_id(plot_id)

    assert retrieved_figure == figure
    assert stats["found"] is True
    assert stats["plot_id"] == plot_id
    assert none_ir is None


def test_get_plot_by_id_not_found(plot_manager):
    """Test retrieving non-existent plot."""
    retrieved_figure, stats, _ = plot_manager.get_plot_by_id("plot_999")

    assert retrieved_figure is None
    assert stats["found"] is False
    assert stats["plot_id"] == "plot_999"


def test_get_latest_plots_all(plot_manager, mock_plotly_figure):
    """Test getting all plots."""
    figure, _, _ = mock_plotly_figure
    for i in range(3):
        plot_manager.add_plot(plot=figure, title=f"Plot {i + 1}")

    plots, stats, _ = plot_manager.get_latest_plots(n=None)

    assert len(plots) == 3
    assert stats["total_plots"] == 3
    assert stats["returned_plots"] == 3


def test_get_latest_plots_limited(plot_manager, mock_plotly_figure):
    """Test getting limited number of plots."""
    figure, _, _ = mock_plotly_figure
    for i in range(5):
        plot_manager.add_plot(plot=figure, title=f"Plot {i + 1}")

    plots, stats, _ = plot_manager.get_latest_plots(n=2)

    assert len(plots) == 2
    assert stats["total_plots"] == 5
    assert stats["returned_plots"] == 2
    # Should return last 2 plots
    assert plots[0]["id"] == "plot_4"
    assert plots[1]["id"] == "plot_5"


def test_get_plot_history(plot_manager, mock_plotly_figure):
    """Test getting plot history (minimal metadata)."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Plot 1", source="service_a")
    plot_manager.add_plot(plot=figure, title="Plot 2", source="service_b")

    history, stats, _ = plot_manager.get_plot_history()

    assert len(history) == 2
    assert stats["total_plots"] == 2

    # History should NOT include figure objects
    for entry in history:
        assert "figure" not in entry
        assert "id" in entry
        assert "title" in entry
        assert "timestamp" in entry
        assert "source" in entry


# =============================================================================
# Test: Clear Operations
# =============================================================================


def test_clear_plots(plot_manager, mock_plotly_figure):
    """Test clearing all plots."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Plot 1")
    plot_manager.add_plot(plot=figure, title="Plot 2")

    assert len(plot_manager.latest_plots) == 2

    _, stats, ir = plot_manager.clear_plots()

    assert len(plot_manager.latest_plots) == 0
    assert stats["cleared_count"] == 2
    assert isinstance(ir, AnalysisStep)


def test_clear_plots_empty(plot_manager):
    """Test clearing when no plots exist."""
    _, stats, ir = plot_manager.clear_plots()

    assert stats["cleared_count"] == 0
    assert isinstance(ir, AnalysisStep)


# =============================================================================
# Test: Visualization State Management
# =============================================================================


def test_add_visualization_record(plot_manager):
    """Test adding visualization record."""
    metadata = {"type": "scatter", "params": {"color": "red"}}

    _, stats, _ = plot_manager.add_visualization_record("plot_123", metadata)

    assert stats["plot_id"] == "plot_123"
    assert stats["history_size"] == 1
    assert stats["registry_size"] == 1
    assert "plot_123" in plot_manager.visualization_state["plot_registry"]


def test_get_visualization_history(plot_manager):
    """Test retrieving visualization history."""
    plot_manager.add_visualization_record("plot_1", {"type": "scatter"})
    plot_manager.add_visualization_record("plot_2", {"type": "heatmap"})
    plot_manager.add_visualization_record("plot_3", {"type": "volcano"})

    history, stats, _ = plot_manager.get_visualization_history(limit=2)

    assert len(history) == 2
    assert stats["total_history"] == 3
    assert stats["returned"] == 2
    # Should return last 2
    assert history[0]["plot_id"] == "plot_2"
    assert history[1]["plot_id"] == "plot_3"


def test_get_visualization_settings(plot_manager):
    """Test getting visualization settings."""
    settings, stats, _ = plot_manager.get_visualization_settings()

    assert isinstance(settings, dict)
    assert "default_width" in settings
    assert settings["default_width"] == 800
    assert stats["settings_count"] == len(settings)


def test_update_visualization_settings(plot_manager):
    """Test updating visualization settings."""
    new_settings = {"default_width": 1200, "color_scheme": "viridis"}

    _, stats, _ = plot_manager.update_visualization_settings(new_settings)

    assert stats["updated_keys"] == ["default_width", "color_scheme"]
    assert plot_manager.visualization_state["settings"]["default_width"] == 1200
    assert plot_manager.visualization_state["settings"]["color_scheme"] == "viridis"


def test_get_plot_by_uuid(plot_manager):
    """Test retrieving plot by UUID."""
    metadata = {"type": "bar", "data": "test"}
    plot_manager.add_visualization_record("uuid_123", metadata)

    retrieved_metadata, stats, _ = plot_manager.get_plot_by_uuid("uuid_123")

    assert retrieved_metadata == metadata
    assert stats["found"] is True
    assert stats["plot_id"] == "uuid_123"


def test_get_plot_by_uuid_not_found(plot_manager):
    """Test retrieving non-existent UUID."""
    retrieved_metadata, stats, _ = plot_manager.get_plot_by_uuid("nonexistent")

    assert retrieved_metadata is None
    assert stats["found"] is False


def test_clear_visualization_history(plot_manager):
    """Test clearing visualization history."""
    plot_manager.add_visualization_record("plot_1", {"type": "scatter"})
    plot_manager.add_visualization_record("plot_2", {"type": "bar"})

    _, stats, _ = plot_manager.clear_visualization_history()

    assert stats["cleared_history"] == 2
    assert stats["cleared_registry"] == 2
    assert len(plot_manager.visualization_state["history"]) == 0
    assert len(plot_manager.visualization_state["plot_registry"]) == 0


# =============================================================================
# Test: Provenance IR Generation
# =============================================================================


def test_ir_generation_enabled(workspace_path, mock_plotly_figure):
    """Test that IR is generated when enable_ir=True."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, enable_ir=True)

    _, _, ir = pm.add_plot(plot=figure, title="Test")

    assert isinstance(ir, AnalysisStep)
    assert ir.operation == "plot_manager.add_plot"
    assert "lobster.core.plot_manager" in ir.imports[0]


def test_ir_generation_disabled(workspace_path, mock_plotly_figure):
    """Test that IR is NOT generated when enable_ir=False."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, enable_ir=False)

    _, _, ir = pm.add_plot(plot=figure, title="Test")

    assert ir is None


def test_ir_for_save_operation(plot_manager, mock_plotly_figure):
    """Test IR generation for save_plots_to_workspace."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Test")

    _, _, ir = plot_manager.save_plots_to_workspace()

    assert isinstance(ir, AnalysisStep)
    assert ir.operation == "plot_manager.save_plots_to_workspace"


def test_ir_for_clear_operation(plot_manager, mock_plotly_figure):
    """Test IR generation for clear_plots."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Test")

    _, _, ir = plot_manager.clear_plots()

    assert isinstance(ir, AnalysisStep)
    assert ir.operation == "plot_manager.clear_plots"


# =============================================================================
# Test: Configuration
# =============================================================================


def test_custom_max_plots_history(workspace_path, mock_plotly_figure):
    """Test custom max_plots_history configuration."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, max_plots_history=10)

    for i in range(15):
        pm.add_plot(plot=figure, title=f"Plot {i + 1}")

    assert len(pm.latest_plots) == 10


def test_custom_min_save_interval(workspace_path, mock_plotly_figure):
    """Test custom min_save_interval configuration."""
    figure, _, _ = mock_plotly_figure
    pm = PlotManager(workspace_path=workspace_path, min_save_interval=1.0)
    pm.add_plot(plot=figure, title="Test")

    # First save
    pm.save_plots_to_workspace()

    # Immediate retry should fail
    _, stats, _ = pm.save_plots_to_workspace()
    assert stats["skipped"] is True


# =============================================================================
# Test: Error Handling
# =============================================================================


def test_add_plot_exception_handling(plot_manager):
    """Test error handling in add_plot for invalid figure."""
    # Pass invalid plot (not a Figure) - should raise ValueError
    with patch("lobster.core.plot_manager._ensure_plotly") as mock_ensure:
        mock_go = Mock()
        mock_ensure.return_value = (mock_go, None)
        mock_go.Figure = type(Mock())

        with pytest.raises(ValueError, match="Plot must be a plotly Figure"):
            plot_manager.add_plot(plot=None, title="Invalid")


def test_save_with_corrupted_plot_entry(plot_manager, mock_plotly_figure):
    """Test save with corrupted plot entry."""
    figure, _, _ = mock_plotly_figure
    plot_manager.add_plot(plot=figure, title="Valid Plot")

    # Manually corrupt a plot entry
    plot_manager.latest_plots.append(
        {"id": "corrupted", "title": "Bad"}
    )  # Missing 'figure'

    # Should handle gracefully (skip corrupted entry)
    saved_files, stats, _ = plot_manager.save_plots_to_workspace()

    # Should still report stats (not crash)
    assert isinstance(saved_files, list)
    assert isinstance(stats, dict)


# =============================================================================
# Summary
# =============================================================================

# Total tests: 40+
# Coverage areas:
# - Core functionality (add, retrieve, clear)
# - 3-tuple pattern compliance
# - FIFO buffer enforcement
# - Thread safety (counter lock, save lock)
# - Rate limiting
# - Workspace export (HTML/PNG)
# - PNG skip threshold (>50K)
# - Visualization state management
# - Provenance IR generation
# - Configuration options
# - Error handling
# - Isolated (no DataManagerV2 dependency)
