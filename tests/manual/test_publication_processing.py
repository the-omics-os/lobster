#!/usr/bin/env python3
"""Manual runner for PublicationProcessingService queue enrichment.

This script keeps the Rich-based CLI experience but focuses entirely on the
current publication processing pipeline. It can optionally load entries from a
RIS file into the publication queue, drive the publication processing service,
and emit detailed metrics about dataset coverage, workspace artifacts, and
handoff readiness.

Examples:
    # Inspect queue entries without running extraction
    python tests/manual/test_publication_processing.py --dry-run --skip-ris-load --status-filter any

    # Load a RIS file and process the first three pending entries
    python tests/manual/test_publication_processing.py --ris-file data/example.ris --max-entries 3

    # Show service responses and persist run metrics to JSON
    python tests/manual/test_publication_processing.py --show-response --output-file results/publication_run.json

    # Show service responses and persist run metrics to JSON
    python tests/manual/test_publication_processing.py --ris-file kevin_notes/databiomix/CRC_microbiome_2.ris --show-response --max-entries 10 --output-file results/publication_run.json
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Ensure project root is importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.publication_queue import PublicationQueueError
from lobster.core.ris_parser import RISParser
from lobster.core.schemas.publication_queue import HandoffStatus, PublicationStatus
from lobster.services.orchestration.publication_processing_service import (
    PublicationProcessingService,
)

console = Console()

# Keep manual output focused on Rich status messages - suppress logs that disrupt progress bars
logging.basicConfig(level=logging.WARNING, force=True)
# Suppress all lobster loggers to ERROR (avoids INFO/WARNING pollution during parallel progress)
logging.getLogger("lobster").setLevel(logging.ERROR)
logging.getLogger("lobster.services").setLevel(logging.ERROR)
logging.getLogger("lobster.tools").setLevel(logging.ERROR)
logging.getLogger("lobster.tools.providers").setLevel(logging.ERROR)
logging.getLogger("lobster.tools.providers.pmc_provider").setLevel(logging.ERROR)
logging.getLogger("lobster.tools.providers.pubmed_provider").setLevel(logging.ERROR)
logging.getLogger("lobster.tools.rate_limiter").setLevel(logging.CRITICAL)
logging.getLogger("lobster.services.orchestration.publication_processing_service").setLevel(logging.ERROR)
logging.getLogger("lobster.services.data_access.content_access_service").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


@dataclass
class EntryProcessingResult:
    """Structured summary for a processed queue entry."""

    entry_id: str
    title: str
    publisher: str
    status: str
    handoff_status: str
    dataset_ids: List[str]
    workspace_keys: List[str]
    extracted_identifiers: Dict[str, List[str]]
    elapsed_seconds: float
    response: str
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "title": self.title,
            "publisher": self.publisher,
            "status": self.status,
            "handoff_status": self.handoff_status,
            "dataset_ids": self.dataset_ids,
            "workspace_keys": self.workspace_keys,
            "extracted_identifiers": self.extracted_identifiers,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "response": self.response,
            "timings": self.timings,
        }


DOI_PREFIX_PUBLISHERS = {
    "10.1016/": "cell.com",
    "10.1038/": "nature.com",
    "10.3389/": "frontiersin.org",
    "10.1007/": "springer.com",
    "10.1002/": "wiley.com",
}

DATASET_PREFIX_TYPES = {
    "GSE": "GEO Series",
    "GDS": "GEO Series",
    "GSM": "GEO Sample",
    "GPL": "GEO Platform",
    "SRP": "SRA Project",
    "SRR": "SRA Run",
    "SRX": "SRA Experiment",
    "SRS": "SRA Sample",
    "PRJ": "BioProject",
    "SAM": "BioSample",
    "ERP": "ENA Project",
    "ERR": "ENA Run",
    "ERS": "ENA Sample",
    "EGA": "EGA",
}


def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None
    except Exception:
        return None


def detect_publisher(entry) -> str:
    """Best-effort publisher detection from entry URLs or DOI prefix."""

    for url in (
        entry.fulltext_url,
        entry.metadata_url,
        entry.pdf_url,
        entry.pubmed_url,
    ):
        domain = _extract_domain(url)
        if domain:
            return domain

    if entry.doi:
        doi_lower = entry.doi.lower()
        for prefix, domain in DOI_PREFIX_PUBLISHERS.items():
            if doi_lower.startswith(prefix):
                return domain

    return "unknown"


def classify_dataset_id(dataset_id: str) -> str:
    """Categorize dataset identifiers using accession prefixes."""

    if not dataset_id:
        return "unknown"

    upper = dataset_id.upper()
    for prefix, label in DATASET_PREFIX_TYPES.items():
        if upper.startswith(prefix):
            return label
    return "other"


def summarize_list(values: List[str], limit: int = 4) -> str:
    if not values:
        return "-"
    shown = values[:limit]
    extra = len(values) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f" (+{extra} more)"
    return text


def summarize_collection(values, limit: int = 4) -> str:
    if not values:
        return "-"
    ordered = list(values)
    ordered.sort()
    return summarize_list(ordered, limit)


def identifier_summary(extracted_identifiers: Dict[str, List[str]]) -> str:
    parts = []
    for name, items in extracted_identifiers.items():
        if items:
            parts.append(f"{name}:{len(items)}")
    return ", ".join(parts) if parts else "-"


def render_entry_detail(result: EntryProcessingResult) -> None:
    table = Table(
        title=f"{result.entry_id} — {result.status.upper()}",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Title", result.title or "Untitled")
    table.add_row("Publisher", result.publisher)
    table.add_row("Status", result.status)
    table.add_row("Handoff", result.handoff_status)
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")
    table.add_row("Datasets", summarize_list(result.dataset_ids))
    table.add_row("Workspace Keys", summarize_list(result.workspace_keys))
    table.add_row("Identifiers", identifier_summary(result.extracted_identifiers))

    console.print(table)
    if result.timings:
        timing_table = Table(title="Step Timings", box=box.SIMPLE)
        timing_table.add_column("Step", style="cyan")
        timing_table.add_column("Seconds", justify="right")
        for step, value in sorted(
            result.timings.items(), key=lambda item: item[1], reverse=True
        ):
            timing_table.add_row(step, f"{value:.2f}")
        console.print(timing_table)

    console.print(
        Panel(
            result.response.strip() or "(no response)",
            title="Service Response",
            expand=False,
        )
    )


def render_overview_table(results: List[EntryProcessingResult]) -> None:
    if not results:
        return

    table = Table(title="Entry Overview", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Entry ID", style="cyan")
    table.add_column("Publisher", style="magenta")
    table.add_column("Status")
    table.add_column("Handoff")
    table.add_column("Datasets", justify="right")
    table.add_column("Workspace", justify="right")
    table.add_column("Identifiers", style="green")
    table.add_column("Elapsed", justify="right")

    for result in results:
        table.add_row(
            result.entry_id,
            result.publisher,
            result.status,
            result.handoff_status,
            str(len(result.dataset_ids)),
            str(len(result.workspace_keys)),
            identifier_summary(result.extracted_identifiers),
            f"{result.elapsed_seconds:.1f}s",
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual publication queue processing harness",
    )
    parser.add_argument(
        "--ris-file",
        type=Path,
        default=Path("kevin_notes/databiomix/CRC_microbiome.ris"),
        help="Path to RIS file used to seed the publication queue",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional workspace path. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--skip-ris-load",
        action="store_true",
        help="Skip loading the RIS file and operate on the existing queue only.",
    )
    parser.add_argument(
        "--load-limit",
        type=int,
        default=None,
        help="Maximum number of RIS entries to add to the queue (default: all).",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="Queue priority assigned to newly loaded entries (1=highest, 10=lowest).",
    )
    parser.add_argument(
        "--status-filter",
        type=str,
        default="pending",
        help="Queue status to target (pending, extracting, completed, handoff_ready, etc.).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=(
            "resolve_identifiers,ncbi_enrich,fetch_sra_metadata,metadata,methods,identifiers"
        ),
        help="Comma-separated extraction tasks passed to PublicationProcessingService.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of queue entries to process (default: all matching).",
    )
    parser.add_argument(
        "--entry",
        type=int,
        default=None,
        help="Process a single entry by index after queue filtering (0-based).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the queue plan without running extraction tasks.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional JSON path for saving detailed run results.",
    )
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Print the full PublicationProcessingService response for each entry.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect per-step timings for each entry (adds light instrumentation).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="WORKERS",
        help="Process entries in parallel with N workers (default: sequential). Recommended: 2-3.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show ERROR-level logs during parallel processing (for debugging rate limits, etc.).",
    )
    return parser.parse_args()


def init_data_manager(workspace: Optional[Path]) -> DataManagerV2:
    """Create (or reuse) a workspace for manual testing."""

    if workspace:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]Workspace[/bold green]: {workspace}")
        return DataManagerV2(workspace_path=str(workspace))

    temp_dir = Path(tempfile.mkdtemp(prefix="lobster_publication_queue_"))
    console.print(f"[bold yellow]Temporary workspace[/bold yellow]: {temp_dir}")
    return DataManagerV2(workspace_path=str(temp_dir))


def load_ris_into_queue(
    ris_file: Path,
    data_manager: DataManagerV2,
    limit: Optional[int],
    priority: int,
) -> Dict[str, int]:
    """Parse a RIS file and add entries to the publication queue."""

    parser = RISParser()
    entries = parser.parse_file(ris_file)

    if not entries:
        return {"added": 0, "skipped": 0}

    if limit and limit > 0:
        entries = entries[:limit]

    added = 0
    skipped = 0

    for entry in entries:
        entry.priority = priority
        try:
            data_manager.publication_queue.add_entry(entry)
            added += 1
        except PublicationQueueError:
            skipped += 1

    return {"added": added, "skipped": skipped}


def select_queue_entries(
    data_manager: DataManagerV2,
    status_filter: str,
    max_entries: Optional[int],
    entry_index: Optional[int],
) -> List:
    """Return queue entries to process based on CLI filters."""

    queue = data_manager.publication_queue

    status_enum = None
    if status_filter and status_filter.lower() not in {"any", "all"}:
        try:
            status_enum = PublicationStatus(status_filter.lower())
        except ValueError:
            raise SystemExit(
                f"Invalid status filter '{status_filter}'. Valid values: {[s.value for s in PublicationStatus]}"
            )

    entries = sorted(queue.list_entries(status=status_enum), key=lambda e: e.created_at)

    if entry_index is not None:
        if entry_index < 0 or entry_index >= len(entries):
            raise SystemExit(
                f"Entry index {entry_index} is out of range for {len(entries)} entries"
            )
        entries = [entries[entry_index]]

    if max_entries and max_entries > 0:
        entries = entries[:max_entries]

    return entries


def preview_entries(entries: List) -> None:
    """Display a quick table describing queued entries."""

    table = Table(title="Queued Entries", box=box.ROUNDED)
    table.add_column("#", justify="right")
    table.add_column("Entry ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("DOI", overflow="fold")
    table.add_column("PMID")
    table.add_column("Title", overflow="fold")

    for idx, entry in enumerate(entries):
        table.add_row(
            str(idx),
            entry.entry_id,
            entry.status.value,
            entry.doi or "-",
            entry.pmid or "-",
            (entry.title or "Untitled")[:80],
        )

    console.print(table)


def process_entries(
    service: PublicationProcessingService,
    entries: List,
    extraction_tasks: str,
    show_response: bool,
) -> List[EntryProcessingResult]:
    """Run extraction tasks for each entry with progress feedback."""

    results: List[EntryProcessingResult] = []
    if not entries:
        console.print("[yellow]No queue entries found for the selected filters.[/yellow]")
        return results

    progress_columns = [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns, console=console, transient=True) as progress:
        task = progress.add_task("Processing queue", total=len(entries))

        for idx, entry in enumerate(entries, 1):
            title = (entry.title or entry.entry_id)[:42]
            progress.update(task, description=f"[{idx}/{len(entries)}] {title}")
            start = time.time()
            response = service.process_entry(
                entry_id=entry.entry_id,
                extraction_tasks=extraction_tasks,
            )
            elapsed = time.time() - start
            updated_entry = service.data_manager.publication_queue.get_entry(entry.entry_id)
            step_timings = {}
            if hasattr(service, "get_latest_timings"):
                step_timings = service.get_latest_timings()

            status_value = (
                updated_entry.status.value
                if hasattr(updated_entry.status, "value")
                else str(updated_entry.status)
            )
            handoff_value = (
                updated_entry.handoff_status.value
                if hasattr(updated_entry.handoff_status, "value")
                else str(updated_entry.handoff_status)
            )
            publisher = detect_publisher(updated_entry)

            result = EntryProcessingResult(
                entry_id=updated_entry.entry_id,
                title=updated_entry.title or updated_entry.entry_id,
                publisher=publisher,
                status=status_value,
                handoff_status=handoff_value,
                dataset_ids=list(updated_entry.dataset_ids or []),
                workspace_keys=list(updated_entry.workspace_metadata_keys or []),
                extracted_identifiers=dict(updated_entry.extracted_identifiers or {}),
                elapsed_seconds=elapsed,
                response=response,
                timings=step_timings,
            )
            results.append(result)

            if show_response:
                render_entry_detail(result)

            progress.advance(task)
            time.sleep(0.2)

    return results


def process_entries_parallel(
    service: PublicationProcessingService,
    entries: List,
    extraction_tasks: str,
    max_workers: int,
    show_response: bool,
    debug: bool = False,
) -> List[EntryProcessingResult]:
    """
    Run extraction tasks in parallel - delegates to service layer.

    This is a thin wrapper around PublicationProcessingService.process_entries_parallel()
    that converts results to the test harness EntryProcessingResult format for summary tables.

    The actual parallel processing with Rich progress bars is handled by the service layer,
    ensuring a single implementation that's shared between the CLI agent tool and test harness.
    """
    results: List[EntryProcessingResult] = []
    if not entries:
        console.print("[yellow]No queue entries found for the selected filters.[/yellow]")
        return results

    entry_ids = [e.entry_id for e in entries]

    # Delegate to service layer (handles Rich progress display internally)
    parallel_result = service.process_entries_parallel(
        entry_ids=entry_ids,
        extraction_tasks=extraction_tasks,
        max_workers=max_workers,
        show_progress=True,
        debug=debug,
    )

    console.print(f"\n[bold green]Parallel processing complete:[/bold green] "
                  f"{parallel_result.successful}/{parallel_result.total_entries} successful "
                  f"in {parallel_result.total_time:.1f}s "
                  f"({parallel_result.entries_per_minute:.1f} entries/min)")

    # Convert service results to test harness EntryProcessingResult format for summary tables
    for pr in parallel_result.entry_results:
        updated_entry = service.data_manager.publication_queue.get_entry(pr.entry_id)
        status_value = (
            updated_entry.status.value
            if hasattr(updated_entry.status, "value")
            else str(updated_entry.status)
        )
        handoff_value = (
            updated_entry.handoff_status.value
            if hasattr(updated_entry.handoff_status, "value")
            else str(updated_entry.handoff_status)
        )
        publisher = detect_publisher(updated_entry)

        result = EntryProcessingResult(
            entry_id=updated_entry.entry_id,
            title=updated_entry.title or updated_entry.entry_id,
            publisher=publisher,
            status=status_value,
            handoff_status=handoff_value,
            dataset_ids=list(updated_entry.dataset_ids or []),
            workspace_keys=list(updated_entry.workspace_metadata_keys or []),
            extracted_identifiers=dict(updated_entry.extracted_identifiers or {}),
            elapsed_seconds=pr.elapsed_seconds,
            response=pr.response,
            timings=pr.timings,
        )
        results.append(result)

        if show_response:
            render_entry_detail(result)

    return results


def summarize(results: List[EntryProcessingResult]) -> Dict[str, object]:
    """Aggregate metrics useful for estimating enrichment readiness."""

    total = len(results)
    status_counts = Counter(r.status for r in results)
    handoff_counts = Counter(r.handoff_status for r in results)
    publisher_counts = Counter(r.publisher for r in results)

    dataset_entries = sum(1 for r in results if r.dataset_ids)
    dataset_total = sum(len(r.dataset_ids) for r in results)
    dataset_unique_ids = set()
    dataset_type_counts: Counter = Counter()
    dataset_type_unique: Dict[str, set] = {}

    workspace_entries = sum(1 for r in results if r.workspace_keys)
    workspace_total = sum(len(r.workspace_keys) for r in results)

    identifier_totals: Counter = Counter()
    identifier_entry_counts: Counter = Counter()
    identifier_unique: Dict[str, set] = {}
    timing_buckets: Dict[str, List[float]] = {}
    for r in results:
        dataset_unique_ids.update(r.dataset_ids)
        for ds in r.dataset_ids:
            ds_type = classify_dataset_id(ds)
            dataset_type_counts[ds_type] += 1
            dataset_type_unique.setdefault(ds_type, set()).add(ds)

        for name, ids in r.extracted_identifiers.items():
            if ids:
                identifier_totals[name] += len(ids)
                identifier_entry_counts[name] += 1
                identifier_unique.setdefault(name, set()).update(ids)

        for step, value in r.timings.items():
            timing_buckets.setdefault(step, []).append(value)

    avg_elapsed = sum(r.elapsed_seconds for r in results) / total if total else 0.0

    timing_details = {}
    for step, values in timing_buckets.items():
        if values:
            timing_details[step] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "max": max(values),
            }

    return {
        "processed": total,
        "status_counts": status_counts,
        "handoff_counts": handoff_counts,
        "publisher_counts": publisher_counts,
        "dataset_entries": dataset_entries,
        "dataset_total": dataset_total,
        "dataset_unique": dataset_unique_ids,
        "dataset_type_counts": dataset_type_counts,
        "dataset_type_unique": dataset_type_unique,
        "workspace_entries": workspace_entries,
        "workspace_total": workspace_total,
        "identifier_totals": identifier_totals,
        "identifier_entry_counts": identifier_entry_counts,
        "identifier_unique": identifier_unique,
        "timing_details": timing_details,
        "avg_elapsed": avg_elapsed,
    }


def render_summary(summary: Dict[str, object], results: List[EntryProcessingResult]) -> None:
    """Pretty-print aggregated metrics with Rich tables."""

    processed = summary["processed"]
    if not processed:
        console.print("[yellow]No entries were processed. Nothing to summarize.[/yellow]")
        return

    status_table = Table(title="Final Status Distribution", box=box.ROUNDED)
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Count", justify="right")
    status_table.add_column("Percent", justify="right")

    for status, count in summary["status_counts"].items():
        percent = (count / processed) * 100 if processed else 0
        status_table.add_row(status.upper(), str(count), f"{percent:.1f}%")

    handoff_table = Table(title="Handoff Status", box=box.ROUNDED)
    handoff_table.add_column("Handoff", style="magenta")
    handoff_table.add_column("Count", justify="right")
    handoff_table.add_column("Percent", justify="right")

    for status, count in summary["handoff_counts"].items():
        percent = (count / processed) * 100 if processed else 0
        handoff_table.add_row(status, str(count), f"{percent:.1f}%")

    enrichment_table = Table(title="Enrichment Coverage", box=box.ROUNDED)
    enrichment_table.add_column("Metric", style="green")
    enrichment_table.add_column("Value", justify="right")
    enrichment_table.add_column("Note")

    dataset_entries = summary["dataset_entries"]
    dataset_total = summary["dataset_total"]
    workspace_entries = summary["workspace_entries"]
    workspace_total = summary["workspace_total"]
    ready = summary["handoff_counts"].get(HandoffStatus.READY_FOR_METADATA.value, 0)

    enrichment_table.add_row(
        "Handoff Ready",
        f"{ready}/{processed}",
        f"{(ready / processed * 100):.1f}% of processed entries",
    )
    enrichment_table.add_row(
        "Dataset IDs",
        f"{dataset_total}",
        f"{dataset_entries} entries with dataset coverage",
    )
    enrichment_table.add_row(
        "Unique Dataset IDs",
        str(len(summary["dataset_unique"])),
        "Across all accession types",
    )
    enrichment_table.add_row(
        "Workspace Artifacts",
        f"{workspace_total}",
        f"{workspace_entries} entries with metadata files",
    )
    enrichment_table.add_row(
        "Avg Duration",
        f"{summary['avg_elapsed']:.1f}s",
        "Mean processing time per entry",
    )

    console.print(status_table)
    console.print(handoff_table)
    console.print(enrichment_table)

    publisher_counts: Counter = summary["publisher_counts"]
    if publisher_counts:
        publisher_table = Table(title="Publisher Distribution", box=box.ROUNDED)
        publisher_table.add_column("Publisher", style="cyan")
        publisher_table.add_column("Count", justify="right")
        publisher_table.add_column("Percent", justify="right")
        for publisher, count in publisher_counts.most_common():
            percent = (count / processed) * 100
            publisher_table.add_row(publisher, str(count), f"{percent:.1f}%")
        console.print(publisher_table)

    dataset_type_counts: Counter = summary["dataset_type_counts"]
    if dataset_type_counts:
        dataset_table = Table(title="Dataset Type Coverage", box=box.ROUNDED)
        dataset_table.add_column("Type", style="magenta")
        dataset_table.add_column("Total IDs", justify="right")
        dataset_table.add_column("Unique IDs", justify="right")
        dataset_table.add_column("Sample IDs")

        for ds_type, count in dataset_type_counts.most_common():
            unique_ids = summary["dataset_type_unique"].get(ds_type, set())
            dataset_table.add_row(
                ds_type,
                str(count),
                str(len(unique_ids)),
                summarize_collection(unique_ids),
            )

        console.print(dataset_table)

    identifier_totals: Counter = summary["identifier_totals"]
    if identifier_totals:
        identifier_table = Table(title="Identifier Breakdown", box=box.ROUNDED)
        identifier_table.add_column("Type", style="cyan")
        identifier_table.add_column("Total IDs", justify="right")
        identifier_table.add_column("Unique IDs", justify="right")
        identifier_table.add_column("Entries", justify="right")
        identifier_table.add_column("Avg / Entry", justify="right")

        for name, total in identifier_totals.items():
            entries = summary["identifier_entry_counts"].get(name, 0)
            unique_ids = summary["identifier_unique"].get(name, set())
            avg = total / entries if entries else 0
            identifier_table.add_row(
                name,
                str(total),
                str(len(unique_ids)),
                str(entries),
                f"{avg:.1f}",
            )

        console.print(identifier_table)

    timing_details = summary.get("timing_details", {})
    if timing_details:
        timing_table = Table(title="Step Timing Profile", box=box.ROUNDED)
        timing_table.add_column("Step", style="cyan")
        timing_table.add_column("Avg (s)", justify="right")
        timing_table.add_column("Max (s)", justify="right")
        timing_table.add_column("Samples", justify="right")

        for step, stats in sorted(
            timing_details.items(), key=lambda item: item[1]["avg"], reverse=True
        ):
            timing_table.add_row(
                step,
                f"{stats['avg']:.2f}",
                f"{stats['max']:.2f}",
                str(stats["count"]),
            )

        console.print(timing_table)

    render_overview_table(results)


def save_results(
    output_file: Path,
    args: argparse.Namespace,
    results: List[EntryProcessingResult],
    summary: Dict[str, object],
) -> None:
    """Persist run details for later inspection."""

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ris_file": str(args.ris_file) if args.ris_file else None,
        "tasks": args.tasks,
        "status_filter": args.status_filter,
        "max_entries": args.max_entries,
        "summary": {
            "processed": summary["processed"],
            "status_counts": dict(summary["status_counts"]),
            "handoff_counts": dict(summary["handoff_counts"]),
            "publisher_counts": dict(summary["publisher_counts"]),
            "dataset_entries": summary["dataset_entries"],
            "dataset_total": summary["dataset_total"],
            "dataset_unique_total": len(summary["dataset_unique"]),
            "dataset_type_counts": dict(summary["dataset_type_counts"]),
            "dataset_type_unique_counts": {
                k: len(v) for k, v in summary["dataset_type_unique"].items()
            },
            "workspace_entries": summary["workspace_entries"],
            "workspace_total": summary["workspace_total"],
            "identifier_totals": dict(summary["identifier_totals"]),
            "identifier_entry_counts": dict(summary["identifier_entry_counts"]),
            "identifier_unique_counts": {
                k: len(v) for k, v in summary["identifier_unique"].items()
            },
            "timing_details": summary.get("timing_details", {}),
            "avg_elapsed": summary["avg_elapsed"],
        },
        "entries": [result.to_dict() for result in results],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Saved results to[/green] {output_file}")


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if not args.skip_ris_load and not args.ris_file.exists():
        raise SystemExit(f"RIS file not found: {args.ris_file}")

    data_manager = init_data_manager(args.workspace)
    service = PublicationProcessingService(data_manager)
    if args.profile and hasattr(service, "enable_timing"):
        service.enable_timing(True)

    if not args.skip_ris_load:
        stats = load_ris_into_queue(
            ris_file=args.ris_file,
            data_manager=data_manager,
            limit=args.load_limit,
            priority=args.priority,
        )
        console.print(
            f"Loaded entries from RIS → added: [green]{stats['added']}[/green], "
            f"skipped: [yellow]{stats['skipped']}[/yellow]"
        )

    queue_entries = select_queue_entries(
        data_manager=data_manager,
        status_filter=args.status_filter,
        max_entries=args.max_entries,
        entry_index=args.entry,
    )

    if args.dry_run:
        preview_entries(queue_entries)
        return

    # Choose sequential or parallel processing
    if args.parallel:
        results = process_entries_parallel(
            service=service,
            entries=queue_entries,
            extraction_tasks=args.tasks,
            max_workers=args.parallel,
            show_response=args.show_response,
            debug=args.debug,
        )
    else:
        results = process_entries(
            service=service,
            entries=queue_entries,
            extraction_tasks=args.tasks,
            show_response=args.show_response,
        )

    summary = summarize(results)
    render_summary(summary, results)

    if args.output_file and results:
        save_results(args.output_file, args, results, summary)
    
    # Display total execution time
    total_elapsed = time.time() - start_time
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]Total execution time: {total_elapsed:.2f}s[/bold green]")
    console.print("=" * 60)


if __name__ == "__main__":
    main()
