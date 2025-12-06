"""Plot preview widget showing latest visualizations."""

from typing import Optional, Dict, Any
from pathlib import Path

from textual.widgets import Static, ListView, ListItem, Label
from textual.reactive import reactive
from textual.containers import Vertical


class PlotItem(ListItem):
    """Single plot list item."""

    def __init__(self, plot_info: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_info = plot_info

    def compose(self):
        """Render plot item."""
        name = self.plot_info.get("name", "Unknown")
        file_path = self.plot_info.get("file_path", "")
        yield Label(f"📊 {name}")
        if file_path:
            yield Label(f"[dim]{Path(file_path).name}[/dim]")


class PlotPreview(Vertical):
    """
    Plot preview panel showing recently generated plots.

    Features:
    - Lists latest plots with metadata
    - Press Enter to open in browser
    - Auto-updates when new plots generated
    - Phase 5: Terminal thumbnails (kitty/sixel)
    """

    plot_count = reactive(0)

    def __init__(self, client=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.plot_list = ListView()

    def compose(self):
        """Render plot preview."""
        yield self.plot_list

    def on_mount(self) -> None:
        """Initialize plot preview."""
        self.border_title = "Plot Preview"
        self.refresh_plots()

    def refresh_plots(self) -> None:
        """Refresh the list of plots."""
        self.plot_list.clear()

        if not self.client:
            self.plot_list.append(ListItem(Label("No client loaded")))
            return

        # Get latest plots from data manager
        if not self.client.data_manager.has_data():
            self.plot_list.append(ListItem(Label("No plots yet")))
            self.plot_count = 0
            return

        plots = self.client.data_manager.get_latest_plots(5)

        if not plots:
            self.plot_list.append(ListItem(Label("No plots yet")))
            self.plot_count = 0
            return

        # Add plot items
        for plot in plots:
            self.plot_list.append(PlotItem(plot))

        self.plot_count = len(plots)
        self.border_title = f"Plots ({self.plot_count})"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle plot selection - open in browser."""
        if isinstance(event.item, PlotItem):
            file_path = event.item.plot_info.get("file_path")
            if file_path and Path(file_path).exists():
                # Open in browser
                import webbrowser

                webbrowser.open(f"file://{file_path}")
                self.notify(f"Opened plot in browser", timeout=3)
            else:
                self.notify("Plot file not found", severity="error")

    def watch_plot_count(self, count: int) -> None:
        """Update border title when plot count changes."""
        self.border_title = f"Plots ({count})" if count > 0 else "Plot Preview"
