#!/usr/bin/env python3
"""
DataBioMix Publication Processing Pipeline - Programmatic Interface

Complete non-interactive pipeline for processing microbiome publications:
1. Load RIS file to publication queue
2. Resolve identifiers (DOI → PMID)
3. Enrich with NCBI E-Link (find GEO/SRA datasets)
4. Fetch SRA sample metadata
5. Filter samples (16S, human, fecal, disease conditions)
6. Export to CSV

Usage:
    # Basic usage with default settings
    python scripts/databiomix_pipeline.py --ris-file path/to/publications.ris

    # Full workflow with filtering and CSV export
    python scripts/databiomix_pipeline.py \
        --ris-file kevin_notes/databiomix/CRC_microbiome.ris \
        --filter-criteria "16S human fecal CRC UC CD healthy" \
        --output-csv results/filtered_samples.csv \
        --max-entries 10

    # Dry run (parse RIS, show what would be processed)
    python scripts/databiomix_pipeline.py --ris-file path/to/file.ris --dry-run

    # Resume processing (skip already processed entries)
    python scripts/databiomix_pipeline.py --ris-file path/to/file.ris --skip-processed

    # Custom extraction tasks
    python scripts/databiomix_pipeline.py \
        --ris-file path/to/file.ris \
        --tasks resolve_identifiers,ncbi_enrich,fetch_sra_metadata

Example Python API usage:
    >>> from scripts.databiomix_pipeline import DataBioMixPipeline
    >>> pipeline = DataBioMixPipeline(workspace_path="results/my_workspace")
    >>> pipeline.load_ris("publications.ris")
    >>> pipeline.process_queue()
    >>> pipeline.export_results("output.csv")
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.ris_parser import RISParser
from lobster.core.schemas.publication_queue import PublicationStatus
from lobster.services.orchestration.publication_processing_service import PublicationProcessingService
from lobster.services.data_access.workspace_content_service import (
    WorkspaceContentService,
    ContentType,
    MetadataContent,
)


class DataBioMixPipeline:
    """
    Programmatic interface for DataBioMix publication processing workflow.

    This class provides a clean API for non-interactive execution of the
    complete pipeline from RIS file to filtered CSV export.

    Example:
        >>> pipeline = DataBioMixPipeline(workspace_path="results/my_workspace")
        >>>
        >>> # Step 1: Load publications
        >>> stats = pipeline.load_ris("publications.ris")
        >>> print(f"Loaded {stats['added_count']} publications")
        >>>
        >>> # Step 2: Process queue (resolve identifiers, fetch SRA metadata)
        >>> results = pipeline.process_queue(
        ...     tasks="resolve_identifiers,ncbi_enrich,fetch_sra_metadata",
        ...     max_entries=10
        ... )
        >>>
        >>> # Step 3: Get SRA metadata from workspace
        >>> sra_data = pipeline.get_sra_metadata()
        >>>
        >>> # Step 4: Export to CSV
        >>> csv_path = pipeline.export_to_csv(sra_data, "filtered_samples.csv")
    """

    DEFAULT_TASKS = "resolve_identifiers,ncbi_enrich,fetch_sra_metadata"

    def __init__(self, workspace_path: str = "results/databiomix_workspace"):
        """
        Initialize the pipeline with a workspace directory.

        Args:
            workspace_path: Directory for workspace files (queue, metadata, exports)
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self.data_manager = DataManagerV2(workspace_path=str(self.workspace_path))
        self.processing_service = PublicationProcessingService(self.data_manager)
        self.workspace_service = WorkspaceContentService(self.data_manager)
        self.ris_parser = RISParser()

        # Track pipeline state
        self.loaded_entries: List[str] = []
        self.processed_results: List[Dict] = []

    def load_ris(
        self,
        ris_file: str,
        priority: int = 5,
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Load publications from a RIS file into the queue.

        Args:
            ris_file: Path to RIS file
            priority: Queue priority (1=highest, 10=lowest)
            skip_duplicates: Skip entries already in queue

        Returns:
            Statistics dict with added_count, skipped_count, entry_ids
        """
        ris_path = Path(ris_file)
        if not ris_path.exists():
            raise FileNotFoundError(f"RIS file not found: {ris_file}")

        # Parse RIS file
        entries = self.ris_parser.parse_file(ris_path)

        added_count = 0
        skipped_count = 0
        entry_ids = []
        errors = []

        # Get existing entry IDs for duplicate detection
        existing_dois = set()
        existing_pmids = set()
        if skip_duplicates:
            for existing in self.data_manager.publication_queue.list_entries():
                if existing.doi:
                    existing_dois.add(existing.doi.lower())
                if existing.pmid:
                    existing_pmids.add(existing.pmid)

        for entry in entries:
            try:
                # Check for duplicates
                if skip_duplicates:
                    if entry.doi and entry.doi.lower() in existing_dois:
                        skipped_count += 1
                        continue
                    if entry.pmid and entry.pmid in existing_pmids:
                        skipped_count += 1
                        continue

                # Set priority
                entry.priority = priority

                # Add to queue
                self.data_manager.publication_queue.add_entry(entry)
                entry_ids.append(entry.entry_id)
                added_count += 1

                # Track for duplicate detection
                if entry.doi:
                    existing_dois.add(entry.doi.lower())
                if entry.pmid:
                    existing_pmids.add(entry.pmid)

            except Exception as e:
                errors.append({"entry": entry.title or entry.doi, "error": str(e)})

        self.loaded_entries = entry_ids

        return {
            "added_count": added_count,
            "skipped_count": skipped_count,
            "entry_ids": entry_ids,
            "errors": errors,
            "ris_file": str(ris_path.absolute()),
        }

    def process_queue(
        self,
        tasks: str = None,
        max_entries: int = None,
        status_filter: str = "pending",
        skip_processed: bool = False
    ) -> Dict[str, Any]:
        """
        Process publication queue entries.

        Args:
            tasks: Comma-separated extraction tasks (default: resolve_identifiers,ncbi_enrich,fetch_sra_metadata)
            max_entries: Maximum entries to process (None = all)
            status_filter: Filter by status (pending, extracting, completed, failed)
            skip_processed: Skip entries that have already been processed

        Returns:
            Processing results with statistics
        """
        tasks = tasks or self.DEFAULT_TASKS

        # Get entries to process
        try:
            status_enum = PublicationStatus(status_filter.lower())
        except ValueError:
            status_enum = None

        entries = self.data_manager.publication_queue.list_entries(status=status_enum)

        if skip_processed:
            entries = [e for e in entries if e.status == PublicationStatus.PENDING]

        if max_entries:
            entries = entries[:max_entries]

        if not entries:
            return {
                "processed_count": 0,
                "message": f"No entries found with status '{status_filter}'",
                "results": []
            }

        # Process each entry
        results = []
        success_count = 0
        failed_count = 0

        for i, entry in enumerate(entries, 1):
            print(f"[{i}/{len(entries)}] Processing: {entry.title or entry.doi or entry.entry_id}")

            try:
                result = self.processing_service.process_entry(
                    entry_id=entry.entry_id,
                    extraction_tasks=tasks
                )

                # Check result status
                if "COMPLETED" in result or "✓" in result:
                    success_count += 1
                else:
                    failed_count += 1

                results.append({
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "doi": entry.doi,
                    "pmid": entry.pmid,
                    "status": "success" if "COMPLETED" in result else "partial",
                    "result": result[:500] + "..." if len(result) > 500 else result
                })

            except Exception as e:
                failed_count += 1
                results.append({
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "status": "failed",
                    "error": str(e)
                })

        self.processed_results = results

        return {
            "processed_count": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "tasks": tasks,
            "results": results
        }

    def get_sra_metadata(self) -> List[Dict]:
        """
        Retrieve all SRA metadata from workspace.

        Returns:
            List of SRA sample metadata dictionaries
        """
        sra_data = []
        metadata_dir = self.workspace_path / "metadata"

        if not metadata_dir.exists():
            return sra_data

        # Find all SRA metadata files
        for json_file in metadata_dir.glob("sra_*_samples.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "data" in data:
                        samples = data["data"].get("samples", [])
                        if isinstance(samples, list):
                            sra_data.extend(samples)
                        elif isinstance(samples, dict):
                            sra_data.append(samples)
            except Exception as e:
                print(f"Warning: Failed to read {json_file}: {e}")

        return sra_data

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        stats = self.data_manager.publication_queue.get_statistics()
        return stats

    def export_to_csv(
        self,
        data: List[Dict],
        output_file: str,
        columns: List[str] = None
    ) -> str:
        """
        Export data to CSV file.

        Args:
            data: List of dictionaries to export
            output_file: Output CSV file path
            columns: Optional list of columns to include (None = all)

        Returns:
            Path to exported CSV file
        """
        import pandas as pd

        if not data:
            raise ValueError("No data to export")

        df = pd.DataFrame(data)

        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        return str(output_path.absolute())

    def export_results_json(self, output_file: str) -> str:
        """
        Export full pipeline results to JSON.

        Args:
            output_file: Output JSON file path

        Returns:
            Path to exported JSON file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "workspace": str(self.workspace_path),
                "pipeline_version": "1.0.0"
            },
            "queue_statistics": self.get_queue_statistics(),
            "processed_results": self.processed_results
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return str(output_path.absolute())


def main():
    """Command-line interface for DataBioMix pipeline."""
    parser = argparse.ArgumentParser(
        description="DataBioMix Publication Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--ris-file",
        required=True,
        help="Path to RIS file with publications"
    )

    # Processing options
    parser.add_argument(
        "--tasks",
        default="resolve_identifiers,ncbi_enrich,fetch_sra_metadata",
        help="Comma-separated extraction tasks (default: resolve_identifiers,ncbi_enrich,fetch_sra_metadata)"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum entries to process (default: all)"
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip already processed entries"
    )

    # Output options
    parser.add_argument(
        "--workspace",
        default="results/databiomix_workspace",
        help="Workspace directory (default: results/databiomix_workspace)"
    )
    parser.add_argument(
        "--output-json",
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--output-csv",
        help="Export SRA samples to CSV file"
    )

    # Filtering options (for CSV export)
    parser.add_argument(
        "--filter-criteria",
        help="Filter criteria for samples (e.g., '16S human fecal CRC UC CD healthy')"
    )

    # Modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse RIS and show queue, but don't process"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show queue statistics only"
    )

    args = parser.parse_args()

    # Initialize pipeline
    print(f"Initializing DataBioMix Pipeline...")
    print(f"  Workspace: {args.workspace}")
    pipeline = DataBioMixPipeline(workspace_path=args.workspace)

    # Stats only mode
    if args.stats_only:
        stats = pipeline.get_queue_statistics()
        print("\n=== Queue Statistics ===")
        print(json.dumps(stats, indent=2))
        return

    # Load RIS file
    print(f"\n=== Loading RIS File ===")
    print(f"  File: {args.ris_file}")
    load_stats = pipeline.load_ris(args.ris_file)
    print(f"  Added: {load_stats['added_count']} entries")
    print(f"  Skipped: {load_stats['skipped_count']} duplicates")

    if load_stats['errors']:
        print(f"  Errors: {len(load_stats['errors'])}")
        for err in load_stats['errors'][:3]:
            print(f"    - {err}")

    # Dry run mode
    if args.dry_run:
        print("\n=== Dry Run - Queue Preview ===")
        entries = pipeline.data_manager.publication_queue.list_entries()
        for i, entry in enumerate(entries[:10], 1):
            print(f"  {i}. {entry.title or entry.doi or 'No title'}")
            if entry.doi:
                print(f"     DOI: {entry.doi}")
            if entry.pmid:
                print(f"     PMID: {entry.pmid}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")
        return

    # Process queue
    print(f"\n=== Processing Queue ===")
    print(f"  Tasks: {args.tasks}")
    if args.max_entries:
        print(f"  Max entries: {args.max_entries}")

    process_results = pipeline.process_queue(
        tasks=args.tasks,
        max_entries=args.max_entries,
        skip_processed=args.skip_processed
    )

    print(f"\n=== Processing Complete ===")
    print(f"  Processed: {process_results['processed_count']} entries")
    print(f"  Success: {process_results['success_count']}")
    print(f"  Failed: {process_results['failed_count']}")

    # Export results
    if args.output_json:
        json_path = pipeline.export_results_json(args.output_json)
        print(f"\n  Results JSON: {json_path}")

    if args.output_csv:
        print(f"\n=== Exporting to CSV ===")
        sra_data = pipeline.get_sra_metadata()
        if sra_data:
            csv_path = pipeline.export_to_csv(sra_data, args.output_csv)
            print(f"  Exported {len(sra_data)} samples to: {csv_path}")
        else:
            print("  No SRA metadata found to export")

    # Final statistics
    print(f"\n=== Final Queue Statistics ===")
    stats = pipeline.get_queue_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
