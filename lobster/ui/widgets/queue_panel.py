"""Queue panel showing download and publication queue status."""

from typing import Optional, List, Dict, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, ListView, ListItem, Label
from textual.reactive import reactive

from rich.text import Text


class QueueItem(ListItem):
    """Single queue item display."""

    def __init__(self, entry_id: str, status: str, label: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry_id = entry_id
        self.status = status
        self.label = label

    def compose(self):
        # Status indicator + label
        status_char = {
            "pending": "○",
            "in_progress": "●",
            "completed": "✓",
            "failed": "✗",
            "handoff_ready": "→",
        }.get(self.status, "?")

        status_style = {
            "pending": "dim",
            "in_progress": "bold yellow",
            "completed": "green",
            "failed": "red",
            "handoff_ready": "cyan",
        }.get(self.status, "")

        text = Text()
        text.append(f"{status_char} ", style=status_style)
        text.append(self.label[:25], style="")  # Truncate long labels
        yield Label(text)


class QueueSection(Vertical):
    """A single queue section (download or publication)."""

    DEFAULT_CSS = """
    QueueSection {
        height: auto;
        max-height: 12;
        padding: 0;
        margin: 0;
    }

    QueueSection > Static {
        height: 1;
        padding: 0 1;
    }

    QueueSection > ListView {
        height: auto;
        max-height: 8;
        padding: 0;
        margin: 0;
    }

    QueueSection ListItem {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, title: str, queue_type: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.queue_type = queue_type
        self._entries: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="header")
        yield ListView(id=f"{self.queue_type}-list")

    def update_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Update the queue entries."""
        self._entries = entries
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the list view."""
        try:
            list_view = self.query_one(f"#{self.queue_type}-list", ListView)
            list_view.clear()

            if not self._entries:
                list_view.append(ListItem(Label(Text("empty", style="dim italic"))))
                return

            # Show most recent entries (up to 5)
            for entry in self._entries[:5]:
                entry_id = entry.get("entry_id", entry.get("id", "?"))
                status = entry.get("status", "pending")
                # Try different label fields
                label = (
                    entry.get("dataset_id")
                    or entry.get("accession")
                    or entry.get("title", "")[:20]
                    or entry_id
                )
                list_view.append(QueueItem(entry_id, status, label))

            # Show count if more entries
            remaining = len(self._entries) - 5
            if remaining > 0:
                list_view.append(
                    ListItem(Label(Text(f"  +{remaining} more...", style="dim")))
                )

        except Exception:
            pass  # Widget not ready


class QueuePanel(Vertical):
    """
    Combined queue panel for downloads and publications.

    Shows:
    - Download queue (pending/active downloads)
    - Publication queue (papers being processed)
    """

    DEFAULT_CSS = """
    QueuePanel {
        height: auto;
        padding: 0;
        border: round #CC2C18 30%;
    }

    QueuePanel > Static.panel-header {
        text-style: bold;
        color: #CC2C18;
        padding: 0 1;
        margin-bottom: 1;
    }

    QueuePanel QueueSection .header {
        color: $text 70%;
        text-style: italic;
    }
    """

    download_count: reactive[int] = reactive(0)
    publication_count: reactive[int] = reactive(0)

    def __init__(self, client=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = client

    def compose(self) -> ComposeResult:
        yield Static("Queues", classes="panel-header")
        yield QueueSection("Downloads", "download")
        yield QueueSection("Publications", "publication")

    def on_mount(self) -> None:
        """Start refresh timer."""
        self._refresh_queues()
        self.set_interval(2.0, self._refresh_queues)

    def _refresh_queues(self) -> None:
        """Refresh both queue displays."""
        if not self.client:
            return

        try:
            # Download queue
            download_section = self.query_one(
                "QueueSection", QueueSection
            )  # First one
            download_entries = []

            if hasattr(self.client, "data_manager"):
                queue = self.client.data_manager.download_queue
                entries = queue.list_entries()
                # Filter to active entries
                download_entries = [
                    {
                        "entry_id": e.entry_id,
                        "status": e.status,
                        "dataset_id": e.dataset_id,
                    }
                    for e in entries
                    if e.status in ["pending", "in_progress"]
                ]
                self.download_count = len(download_entries)

            # Get all QueueSections
            sections = list(self.query("QueueSection").results())
            if len(sections) >= 1:
                sections[0].update_entries(download_entries)

            # Publication queue
            pub_entries = []
            if hasattr(self.client, "publication_queue") and self.client.publication_queue:
                entries = self.client.publication_queue.list_entries()
                pub_entries = [
                    {
                        "entry_id": e.entry_id,
                        "status": e.status,
                        "title": getattr(e, "title", ""),
                        "accession": getattr(e, "accession", ""),
                    }
                    for e in entries
                    if e.status not in ["completed", "failed"]
                ]
                self.publication_count = len(pub_entries)

            if len(sections) >= 2:
                sections[1].update_entries(pub_entries)

        except Exception:
            pass  # Queues not available
