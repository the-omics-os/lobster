"""Main analysis screen with multi-panel layout."""

from typing import Optional
from functools import partial

from textual.screen import Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer
from textual.worker import Worker
from textual.binding import Binding

from lobster.core.client import AgentClient
from lobster.core.license_manager import get_current_tier
from lobster.config.llm_factory import LLMFactory
from lobster.config.settings import get_settings
from lobster.ui.widgets import QueryPrompt, ModalityList, ResultsDisplay, PlotPreview, StatusBar
from lobster.ui.widgets.status_bar import get_friendly_model_name
from textual import on
from lobster.ui.widgets.modality_list import ModalitySelected


class AnalysisScreen(Screen):
    """
    Main analysis screen with multi-panel layout.

    Layout:
    ┌─────────────────────────────────────────────┐
    │ Header                                      │
    ├──────────┬──────────────────────┬───────────┤
    │ Modality │ Results              │ Plot      │
    │ List     │ (streaming)          │ Preview   │
    │          ├──────────────────────┤           │
    │          │ Query Prompt         │           │
    ├──────────┴──────────────────────┴───────────┤
    │ Footer (keybindings)                        │
    └─────────────────────────────────────────────┘

    Features:
    - Non-blocking query execution (@work)
    - Live streaming responses
    - Concurrent operations
    - Keyboard navigation (Tab to cycle focus)
    """

    CSS = """
    AnalysisScreen {
        background: transparent;
        scrollbar-size-vertical: 1;
    }

    #main-content {
        height: 1fr;
    }

    #main-panels {
        height: 1fr;
    }

    /* StatusBar uses DEFAULT_CSS - no overrides needed */

    #left-panel {
        width: 30;
        border: solid #333333;
    }

    #center-panel {
        width: 1fr;
    }

    #results-display {
        height: 1fr;
        border: solid #333333;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #query-prompt {
        height: 10;
        border: solid #444444;
    }

    #query-prompt:focus {
        border: solid $accent;
    }

    #query-prompt.-submit-blocked {
        border: solid #333333 30%;
    }

    #right-panel {
        width: 30;
        border: solid #333333;
    }

    ModalityList {
        height: 1fr;
    }

    PlotPreview {
        height: 1fr;
    }

    /* ChatMessage styling (Elia pattern) */
    ChatMessage {
        height: auto;
        width: 1fr;
        margin: 0 1;
        padding: 0 2;
    }

    .user-message {
        border: round $primary 50%;
    }

    .agent-message {
        border: round $accent 60%;
    }

    .agent-message.streaming {
        background: $accent 3%;
    }

    .error-message {
        border: round red 70%;
    }

    /* Ensure Markdown wraps (no horizontal scroll) */
    ChatMessage Markdown {
        width: 100%;
        max-width: 100%;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_query", "Cancel Query", key_display="^C"),
        Binding("ctrl+l", "clear_results", "Clear Results", key_display="^L"),
        Binding("f5", "refresh_data", "Refresh", key_display="F5"),
    ]

    def __init__(self, client: AgentClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self.current_worker: Optional[Worker] = None

    def compose(self):
        """Create the multi-panel layout."""
        yield Header()

        status_bar = StatusBar(id="status-bar")

        left_panel = Vertical(
            ModalityList(self.client),
            id="left-panel",
        )

        center_panel = Vertical(
            ResultsDisplay(id="results-display"),
            QueryPrompt(id="query-prompt"),
            id="center-panel",
        )

        right_panel = Vertical(
            PlotPreview(self.client),
            id="right-panel",
        )

        main_panels = Horizontal(
            left_panel,
            center_panel,
            right_panel,
            id="main-panels",
        )

        yield Vertical(
            status_bar,
            main_panels,
            id="main-content",
        )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen."""
        self.sub_title = f"Session: {self.client.session_id}"

        # Initialize status bar with current state
        self._init_status_bar()

        # Start polling for queue updates (every 2 seconds)
        self.set_interval(2.0, self._update_queue_status)

    def _init_status_bar(self) -> None:
        """Initialize status bar with client data."""
        status_bar = self.query_one(StatusBar)

        # 1. Subscription tier
        status_bar.subscription_tier = get_current_tier()

        # 2. Provider + Model
        try:
            provider = LLMFactory.get_current_provider()
            if provider:
                status_bar.provider_name = provider

                # Get model name for supervisor agent
                model_params = get_settings().get_agent_llm_params("supervisor")
                model_id = model_params.get("model_id", "unknown")
                status_bar.model_name = get_friendly_model_name(model_id, provider)
        except Exception as e:
            # Fallback if provider/model detection fails
            status_bar.provider_name = "unknown"
            status_bar.model_name = "unknown"

        # 3. Initial queue counts
        self._update_queue_status()

    def _update_queue_status(self) -> None:
        """Poll queues for status (runs every 2s via set_interval)."""
        status_bar = self.query_one(StatusBar)

        try:
            # Download queue (always available)
            download_queue = self.client.data_manager.download_queue
            active_downloads = len([
                e for e in download_queue.list_entries()
                if e.status in ["pending", "in_progress"]
            ])
            status_bar.download_count = active_downloads

            # Publication queue (premium - may be None)
            if self.client.publication_queue:
                pub_queue = self.client.publication_queue
                active_pubs = len([
                    e for e in pub_queue.list_entries()
                    if e.status not in ["completed", "failed"]
                ])
                status_bar.publication_count = active_pubs
            else:
                status_bar.publication_count = 0

        except Exception:
            # Silently handle queue access errors
            pass

    @on(QueryPrompt.QuerySubmitted)
    def handle_query_submission(self, event: QueryPrompt.QuerySubmitted) -> None:
        """Handle query submission from prompt (Elia pattern)."""
        query_text = event.text

        # 1. Show user's query in conversation
        results = self.query_one(ResultsDisplay)
        results.append_user_message(query_text)

        # 2. Update status bar - agent is now processing
        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "processing"

        # 3. Lock input via widget reference (Elia pattern)
        event.prompt_input.submit_ready = False

        # 4. Start streaming query in background worker (pass as callable!)
        self.run_worker(
            partial(self.execute_streaming_query, query_text),
            name=f"query_{query_text[:20]}",
            group="agent_query",
            exclusive=True,
            thread=True,
        )

    def execute_streaming_query(self, query: str) -> None:
        """
        Execute LangGraph query with live streaming updates.

        This runs in a background thread, updating the UI via post_message().
        The @work decorator with thread=True ensures non-blocking execution.
        """
        results = self.query_one(ResultsDisplay)

        try:
            # Create agent message bubble for streaming
            self.app.call_from_thread(results.start_agent_message)

            # Stream events from LangGraph
            for event in self.client.query(query, stream=True):
                event_type = event.get("type")

                if event_type == "stream":
                    # Streaming content from agent
                    content = event.get("content", "")

                    if content:
                        # Append chunk to agent message bubble
                        self.app.call_from_thread(
                            results.append_to_agent_message, content
                        )

                elif event_type == "complete":
                    # Mark agent message as complete
                    self.app.call_from_thread(results.complete_agent_message)

                elif event_type == "error":
                    # Error occurred
                    error = event.get("error", "Unknown error")
                    self.app.call_from_thread(results.show_error, error)

        except Exception as e:
            # Handle unexpected errors
            self.app.call_from_thread(results.show_error, str(e))

        finally:
            # Always unlock input and refresh data
            self.app.call_from_thread(self._unlock_input)

    def _unlock_input(self) -> None:
        """Unlock query prompt and refresh data views."""
        prompt = self.query_one(QueryPrompt)
        prompt.submit_ready = True

        # Reset agent status to idle
        status_bar = self.query_one(StatusBar)
        status_bar.agent_status = "idle"

        # Refresh data views
        self.query_one(ModalityList).refresh_modalities()
        self.query_one(PlotPreview).refresh_plots()

    def on_modality_list_modality_selected(
        self, event: ModalitySelected
    ) -> None:
        """Handle modality selection."""
        self.notify(f"Selected: {event.modality_name}", timeout=2)

    def action_cancel_query(self) -> None:
        """Cancel running query (Ctrl+C)."""
        if self.current_worker and not self.current_worker.is_finished:
            self.current_worker.cancel()
            self.notify("Query cancelled", severity="warning")
            self._unlock_input()
        else:
            self.notify("No query running", severity="information")

    def action_clear_results(self) -> None:
        """Clear results display (Ctrl+L)."""
        results = self.query_one(ResultsDisplay)
        results.clear_display()

    def action_refresh_data(self) -> None:
        """Refresh modality list and plots (F5)."""
        self.query_one(ModalityList).refresh_modalities()
        self.query_one(PlotPreview).refresh_plots()
        self.notify("Data refreshed", timeout=2)
