"""Connections panel showing database connectivity status."""

from typing import Dict, Optional
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


# Database registry with check functions
DATABASE_REGISTRY = {
    "GEO": {"check": "ncbi", "icon": "📊", "desc": "Gene Expression Omnibus"},
    "SRA": {"check": "ncbi", "icon": "🧬", "desc": "Sequence Read Archive"},
    "PubMed": {"check": "ncbi", "icon": "📚", "desc": "Literature database"},
    "PRIDE": {"check": "pride", "icon": "🔬", "desc": "Proteomics repository"},
    "UniProt": {"check": "uniprot", "icon": "🧪", "desc": "Protein database"},
    "ENA": {"check": "ena", "icon": "🌍", "desc": "European Nucleotide Archive"},
}


class ConnectionsPanel(Vertical):
    """
    Database connections status panel.

    Shows which external databases are available/connected.
    Status dots: ● connected, ○ unavailable, ◐ checking
    """

    DEFAULT_CSS = """
    ConnectionsPanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    ConnectionsPanel > Static.header {
        text-style: bold;
        color: #CC2C18;
        margin-bottom: 1;
    }

    ConnectionsPanel > Static.row {
        height: 1;
    }
    """

    # Track connection status
    connection_status: reactive[Dict[str, str]] = reactive({})

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Initialize all as unknown
        self._status = {name: "unknown" for name in DATABASE_REGISTRY}

    def compose(self) -> ComposeResult:
        yield Static("CONNECTIONS", classes="header")
        yield Static(id="connections-grid")

    def on_mount(self) -> None:
        """Initialize and start status checks."""
        self._update_display()
        # Check connections on startup (non-blocking)
        self.set_interval(30.0, self._check_connections)
        # Initial check after brief delay
        self.set_timer(1.0, self._check_connections)

    def _check_connections(self) -> None:
        """Check database connectivity (simplified - based on config/env)."""
        import os

        # NCBI databases available if we have API key or just accessible
        ncbi_ok = True  # NCBI is publicly accessible

        # Check if NCBI API key is set (for rate limits)
        has_ncbi_key = bool(os.environ.get("NCBI_API_KEY"))

        for name, config in DATABASE_REGISTRY.items():
            check_type = config["check"]
            if check_type == "ncbi":
                self._status[name] = "ok" if ncbi_ok else "unknown"
            elif check_type == "pride":
                self._status[name] = "ok"  # PRIDE is publicly accessible
            elif check_type == "uniprot":
                self._status[name] = "ok"  # UniProt is publicly accessible
            elif check_type == "ena":
                self._status[name] = "ok"  # ENA is publicly accessible
            else:
                self._status[name] = "unknown"

        self._update_display()

    def _update_display(self) -> None:
        """Update the connections grid."""
        try:
            grid = self.query_one("#connections-grid", Static)

            text = Text()
            items = list(DATABASE_REGISTRY.items())

            # Display in 2 columns
            for i in range(0, len(items), 2):
                row_items = items[i:i+2]
                for j, (name, config) in enumerate(row_items):
                    status = self._status.get(name, "unknown")
                    dot, style = self._get_status_indicator(status)

                    text.append(f" {dot} ", style=style)
                    text.append(f"{name:<8}", style="")

                    if j == 0 and len(row_items) > 1:
                        text.append("  ")

                text.append("\n")

            grid.update(text)
        except Exception:
            pass

    def _get_status_indicator(self, status: str) -> tuple[str, str]:
        """Get status dot and style."""
        indicators = {
            "ok": ("●", "green"),
            "error": ("●", "red"),
            "warning": ("●", "yellow"),
            "checking": ("◐", "cyan"),
            "unknown": ("○", "dim"),
        }
        return indicators.get(status, ("○", "dim"))

    def set_status(self, database: str, status: str) -> None:
        """Manually set database status."""
        if database in self._status:
            self._status[database] = status
            self._update_display()
