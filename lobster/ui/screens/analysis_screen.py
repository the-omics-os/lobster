"""Main analysis screen with cockpit-style layout."""

from typing import Optional
from functools import partial

from textual.screen import Screen
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer
from textual.worker import Worker
from textual.binding import Binding
from textual import on

from lobster.core.client import AgentClient
from lobster.core.license_manager import get_current_tier
from lobster.config.llm_factory import LLMFactory
from lobster.config.settings import get_settings
from lobster.ui.widgets import (
    QueryPrompt,
    ModalityList,
    ResultsDisplay,
    PlotPreview,
    StatusBar,
    SystemInfoPanel,
    QueuePanel,
    ConnectionsPanel,
    AgentsPanel,
    AdaptersPanel,
)
from lobster.ui.widgets.status_bar import get_friendly_model_name
from lobster.ui.widgets.modality_list import ModalitySelected


class AnalysisScreen(Screen):
    """
    Cockpit-style analysis screen - information-dense dashboard.

    Layout (NASA mission control inspired):
    ┌─────────────────────────────────────────────────────────────┐
    │ Header: Lobster OS                                          │
    ├─────────────────────────────────────────────────────────────┤
    │ [tier] │ [provider/model] │ [agent status]      [status bar]│
    ├────────────────┬─────────────────────────┬──────────────────┤
    │ SYSTEM         │                         │ AGENTS           │
    │ ● CPU 45%      │     Results             │ ● supervisor     │
    │ ● MEM 8.2/16GB │     (conversation)      │ ▶ research (act) │
    │ ● GPU CUDA     │                         │ ● data_expert    │
    ├────────────────┤                         ├──────────────────┤
    │ CONNECTIONS    │                         │ MODALITIES       │
    │ ● GEO  ● SRA   │                         │ ● geo_gse12345   │
    │ ● PubMed ○PRIDE│                         │ ● rna_filtered   │
    ├────────────────┤                         ├──────────────────┤
    │ QUEUES         │                         │ ADAPTERS         │
    │ Downloads: 2   │                         │ Trans: ●●●●      │
    │ Papers: 15     ├─────────────────────────┤ Prot:  ●●●       │
    ├────────────────┤    [Query Prompt]       ├──────────────────┤
    │                │                         │ PLOTS            │
    │                │                         │ ○ No plots yet   │
    ├────────────────┴─────────────────────────┴──────────────────┤
    │ Footer: ESC Quit │ ^P Commands │ ^L Clear │ F5 Refresh      │
    └─────────────────────────────────────────────────────────────┘
    """

    CSS = """
    AnalysisScreen {
        background: transparent;
    }

    /* Main 3-column layout */
    #main-panels {
        height: 1fr;
    }

    /* Left panel - system telemetry */
    #left-panel {
        width: 26;
        border-right: solid $primary 30%;
        padding: 0 1;
    }

    #left-panel > * {
        margin-bottom: 1;
    }

    /* Center panel - conversation */
    #center-panel {
        width: 1fr;
        padding: 0 1;
    }

    #results-display {
        height: 1fr;
        border: round $primary 20%;
        overflow-y: auto;
        padding: 0 1;
    }

    #query-prompt {
        height: 6;
        border: round $primary 40%;
        margin-top: 1;
    }

    #query-prompt:focus {
        border: round #ffaa00 80%;
    }

    /* Right panel - data & agents */
    #right-panel {
        width: 26;
        border-left: solid $primary 30%;
        padding: 0 1;
    }

    #right-panel > * {
        margin-bottom: 1;
    }

    /* Cockpit panel styling - compact */
    SystemInfoPanel, ConnectionsPanel, QueuePanel,
    AgentsPanel, AdaptersPanel {
        height: auto;
        padding: 0 1;
        border: round $primary 30%;
    }

    /* Data panels */
    ModalityList {
        height: 1fr;
        min-height: 6;
        border: round $primary 30%;
    }

    PlotPreview {
        height: auto;
        max-height: 8;
        border: round $primary 30%;
    }

    /* Chat messages - minimal styling */
    ChatMessage {
        height: auto;
        width: 1fr;
        margin: 0 0 1 0;
        padding: 1;
    }

    .user-message {
        border: round #4a9eff 60%;
        background: #4a9eff 10%;
    }

    .user-message Markdown {
        color: #4a9eff;
    }

    .agent-message {
        border: round #e45c47 40%;
        background: #e45c47 5%;
    }

    .agent-message.streaming {
        border: round #ffaa00 70%;
        background: #ffaa00 8%;
    }

    .system-message {
        border: round #44cc44 40%;
        background: #44cc44 5%;
    }

    .error-message {
        border: round #ff4444 60%;
        background: #ff4444 10%;
    }

    /* Status bar */
    StatusBar {
        height: 1;
        background: transparent;
        border-bottom: solid $primary 20%;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", key_display="ESC", priority=True),
        Binding("ctrl+q", "quit", "Quit", key_display="^Q"),
        Binding("ctrl+p", "command_palette", "Commands", key_display="^P"),
        Binding("ctrl+c", "cancel_query", "Cancel", key_display="^C"),
        Binding("ctrl+l", "clear_results", "Clear", key_display="^L"),
        Binding("f5", "refresh_data", "Refresh", key_display="F5"),
    ]

    def __init__(self, client: AgentClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.current_worker: Optional[Worker] = None

    def compose(self):
        """Create cockpit-style 3-column layout."""
        yield Header()

        # Status bar - top telemetry strip
        yield StatusBar(id="status-bar")

        # Main content area - 3 columns
        with Horizontal(id="main-panels"):
            # Left panel: System telemetry
            with Vertical(id="left-panel"):
                yield SystemInfoPanel()
                yield ConnectionsPanel()
                yield QueuePanel(self.client)

            # Center panel: Conversation
            with Vertical(id="center-panel"):
                yield ResultsDisplay(id="results-display")
                yield QueryPrompt(id="query-prompt")

            # Right panel: Agents & Data
            with Vertical(id="right-panel"):
                yield AgentsPanel(client=self.client)
                yield ModalityList(self.client)
                yield AdaptersPanel()
                yield PlotPreview(self.client)

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen."""
        self.sub_title = f"Session: {self.client.session_id}"
        self._init_status_bar()

    def _init_status_bar(self) -> None:
        """Initialize status bar with client data."""
        status_bar = self.query_one(StatusBar)

        # Subscription tier
        status_bar.subscription_tier = get_current_tier()

        # Provider + Model
        try:
            provider = LLMFactory.get_current_provider()
            if provider:
                status_bar.provider_name = provider
                model_params = get_settings().get_agent_llm_params("supervisor")
                model_id = model_params.get("model_id", "unknown")
                status_bar.model_name = get_friendly_model_name(model_id, provider)
        except Exception:
            status_bar.provider_name = "unknown"
            status_bar.model_name = "unknown"

    @on(QueryPrompt.QuerySubmitted)
    def handle_query_submission(self, event: QueryPrompt.QuerySubmitted) -> None:
        """Handle query submission."""
        query_text = event.text

        # Show user message
        results = self.query_one(ResultsDisplay)
        results.append_user_message(query_text)

        # Update status bar
        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "processing"

        # Mark supervisor agent as active (cockpit telemetry)
        try:
            agents_panel = self.query_one(AgentsPanel)
            agents_panel.set_agent_active("supervisor")
        except Exception:
            pass

        # Lock input
        event.prompt_input.submit_ready = False

        # Run query in background
        self.run_worker(
            partial(self.execute_streaming_query, query_text),
            name=f"query_{query_text[:20]}",
            group="agent_query",
            exclusive=True,
            thread=True,
        )

    def execute_streaming_query(self, query: str) -> None:
        """Execute query with streaming."""
        results = self.query_one(ResultsDisplay)

        try:
            self.app.call_from_thread(results.start_agent_message)

            for event in self.client.query(query, stream=True):
                event_type = event.get("type")

                if event_type == "stream":
                    content = event.get("content", "")
                    if content:
                        self.app.call_from_thread(
                            results.append_to_agent_message, content
                        )
                elif event_type == "complete":
                    self.app.call_from_thread(results.complete_agent_message)
                elif event_type == "error":
                    error = event.get("error", "Unknown error")
                    self.app.call_from_thread(results.show_error, error)

        except Exception as e:
            self.app.call_from_thread(results.show_error, str(e))
        finally:
            self.app.call_from_thread(self._unlock_input)

    def _unlock_input(self) -> None:
        """Unlock input and refresh views."""
        prompt = self.query_one(QueryPrompt)
        prompt.submit_ready = True

        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "idle"

        # Mark all agents as idle (cockpit telemetry)
        try:
            agents_panel = self.query_one(AgentsPanel)
            agents_panel.set_agent_idle("supervisor")
            agents_panel.set_agent_idle("research_agent")
            agents_panel.set_agent_idle("data_expert")
        except Exception:
            pass

        # Refresh data panels
        self.query_one(ModalityList).refresh_modalities()
        self.query_one(PlotPreview).refresh_plots()

    def on_modality_list_modality_selected(self, event: ModalitySelected) -> None:
        """Handle modality selection."""
        self.notify(f"Selected: {event.modality_name}", timeout=2)

    def refresh_all(self) -> None:
        """Refresh all panels (called by F5)."""
        self.query_one(ModalityList).refresh_modalities()
        self.query_one(PlotPreview).refresh_plots()

    def action_cancel_query(self) -> None:
        """Cancel running query."""
        if self.current_worker and not self.current_worker.is_finished:
            self.current_worker.cancel()
            self.notify("Cancelled", severity="warning")
            self._unlock_input()

    def action_clear_results(self) -> None:
        """Clear results display."""
        self.query_one(ResultsDisplay).clear_display()

    def action_refresh_data(self) -> None:
        """Refresh all data panels."""
        self.refresh_all()
        self.notify("Refreshed", timeout=1)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
