"""
LobsterOS - Minimal terminal UI for Lobster bioinformatics platform.

Design: Terminal-native, transparent backgrounds, minimal color palette.
"""

from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from lobster.core.client import AgentClient
from lobster.core.workspace import resolve_workspace
from lobster.ui.screens import AnalysisScreen


class LobsterOS(App):
    """
    Minimal terminal-native bioinformatics workspace.

    Design principles:
    - Transparent backgrounds (inherit terminal theme)
    - Single accent color (orange) for active/important elements
    - Clean borders, no visual noise
    """

    # Minimal design - transparent, terminal-native
    CSS = """
    * {
        scrollbar-size: 1;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }

    Screen {
        background: transparent;
    }

    Header {
        background: transparent;
        color: $text;
    }

    Footer {
        background: transparent;
    }

    /* Accent color for active elements only */
    .active, .processing {
        color: #e45c47;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", key_display="Q"),
        Binding("f5", "refresh", "Refresh", key_display="F5"),
    ]

    TITLE = "lobster"

    def __init__(self, workspace_path: Optional[Path] = None):
        super().__init__()
        self.workspace_path = resolve_workspace(workspace_path, create=True)
        self.client: Optional[AgentClient] = None

    def on_mount(self) -> None:
        """Initialize the application."""
        self.sub_title = str(self.workspace_path.name)

        try:
            self.client = AgentClient(
                workspace_path=self.workspace_path,
                enable_reasoning=True,
            )
            self.push_screen(AnalysisScreen(self.client))

        except Exception as e:
            self.notify(f"Init failed: {e}", severity="error", timeout=30)

    def action_refresh(self) -> None:
        """Refresh all panels."""
        screen = self.screen
        if hasattr(screen, "refresh_all"):
            screen.refresh_all()


def run_lobster_os(workspace_path: Optional[Path] = None):
    """Entry point for lobster os command."""
    app = LobsterOS(workspace_path)
    app.run()
