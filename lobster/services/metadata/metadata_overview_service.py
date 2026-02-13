"""
Metadata overview service for aggregating statistics across all metadata sources.

Provides unified access to metadata state for the enhanced /metadata command:
- Publication queue (status, identifiers, extracted datasets)
- Sample metadata (counts, disease coverage, filter stats)
- Workspace files (categorized inventory)
- Export files (with usage guidance)
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from lobster.services.data_access.workspace_content_service import (
    WorkspaceContentService,
)
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


class MetadataOverviewService:
    """
    Centralized metadata statistics aggregation service.

    Provides unified access to metadata state across:
    - Publication queue (status, identifiers, extracted datasets)
    - Sample metadata (counts, disease coverage, filter stats)
    - Workspace files (categorized inventory)
    - Export files (with usage guidance)
    """

    def __init__(self, data_manager: "DataManagerV2"):
        self.data_manager = data_manager
        self.workspace_service = WorkspaceContentService(data_manager)
        self._workspace_path = Path(data_manager.workspace_path)

    def get_publication_queue_summary(
        self, status_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get publication queue status breakdown with identifier coverage.

        Args:
            status_filter: Optional status to filter entries by

        Returns:
            Dict with:
            - total: Total entries
            - status_breakdown: Counter of statuses
            - identifier_coverage: Dict of identifier type -> count/percentage
            - extracted_datasets: Counter of extracted database identifiers
            - workspace_ready: Count of entries with workspace files
            - recent_errors: List of recent error messages
            - entries: List of entries (if status_filter provided, limited to 20)
        """
        queue_path = self._workspace_path / "publication_queue.jsonl"

        if not queue_path.exists():
            return {
                "total": 0,
                "status_breakdown": {},
                "identifier_coverage": {},
                "extracted_datasets": {},
                "workspace_ready": 0,
                "recent_errors": [],
                "message": "No publication queue found. Use research_agent to process publications.",
            }

        try:
            entries = []
            with open(queue_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not entries:
                return {
                    "total": 0,
                    "status_breakdown": {},
                    "message": "Publication queue is empty.",
                }

            # Status breakdown
            status_counts = Counter(e.get("status", "unknown") for e in entries)

            # Identifier coverage
            total = len(entries)
            pmid_count = sum(1 for e in entries if e.get("pmid"))
            doi_count = sum(1 for e in entries if e.get("doi"))
            pmc_count = sum(1 for e in entries if e.get("pmc_id"))

            # Extracted identifiers (aggregated across all entries)
            extracted = Counter()
            for e in entries:
                ids = e.get("extracted_identifiers", {})
                for db_type, id_list in ids.items():
                    if id_list:
                        extracted[db_type] += len(id_list)

            # Workspace-ready entries
            workspace_ready = sum(
                1 for e in entries if e.get("workspace_metadata_keys")
            )

            # Recent errors
            recent_errors = []
            for e in entries:
                if e.get("status") == "failed" and e.get("error"):
                    recent_errors.append(
                        {
                            "entry_id": e.get("entry_id", "unknown"),
                            "title": (e.get("title") or "Untitled")[:60],
                            "error": str(e.get("error", ""))[:100],
                        }
                    )

            result = {
                "total": total,
                "status_breakdown": dict(status_counts),
                "identifier_coverage": {
                    "pmid": {
                        "count": pmid_count,
                        "pct": round(pmid_count / total * 100, 1) if total else 0,
                    },
                    "doi": {
                        "count": doi_count,
                        "pct": round(doi_count / total * 100, 1) if total else 0,
                    },
                    "pmc_id": {
                        "count": pmc_count,
                        "pct": round(pmc_count / total * 100, 1) if total else 0,
                    },
                },
                "extracted_datasets": dict(extracted),
                "workspace_ready": workspace_ready,
                "recent_errors": recent_errors[:5],
            }

            # If status filter provided, include filtered entries
            if status_filter:
                filtered_entries = [
                    e for e in entries if e.get("status") == status_filter
                ]
                result["filtered_entries"] = filtered_entries[:20]
                result["filtered_total"] = len(filtered_entries)

            return result

        except Exception as e:
            logger.error(f"Error reading publication queue: {e}")
            return {
                "total": 0,
                "status_breakdown": {},
                "error": str(e),
            }

    def get_sample_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated sample statistics across all processed metadata.

        Returns:
            Dict with:
            - total_samples: Total samples across all BioProjects
            - filtered_samples: Samples after filtering (if filtering was run)
            - bioproject_count: Number of BioProjects with samples
            - disease_coverage: Percentage with disease annotation
            - filter_criteria: Active filter string
            - filter_breakdown: Per-filter retention stats
            - has_aggregated: Whether aggregated data exists
            - sources: List of sample source identifiers
        """
        # Check for aggregated_filtered_samples in metadata_store
        aggregated = self.data_manager.metadata_store.get(
            "aggregated_filtered_samples", {}
        )

        if aggregated and aggregated.get("samples"):
            samples = aggregated.get("samples", [])
            stats = aggregated.get("stats", {})

            # Calculate disease coverage from samples if not in stats
            disease_coverage = stats.get("disease_coverage", 0)
            if not disease_coverage and samples:
                with_disease = sum(
                    1
                    for s in samples
                    if s.get("disease") or s.get("host_disease") or s.get("phenotype")
                )
                disease_coverage = round(with_disease / len(samples) * 100, 1)

            return {
                "total_samples": stats.get("total_extracted", len(samples)),
                "filtered_samples": stats.get("total_after_filter", len(samples)),
                "disease_coverage": disease_coverage,
                "filter_criteria": aggregated.get("filter_criteria", ""),
                "filter_breakdown": stats.get("filter_breakdown", {}),
                "bioproject_count": len(
                    set(s.get("bioproject", "") for s in samples if s.get("bioproject"))
                ),
                "has_aggregated": True,
                "retention_rate": round(
                    stats.get("total_after_filter", len(samples))
                    / stats.get("total_extracted", len(samples))
                    * 100,
                    1,
                )
                if stats.get("total_extracted")
                else 0,
            }

        # Fallback: count samples from sra_*_samples files in metadata_store
        total_samples = 0
        bioproject_count = 0
        sources = []

        for key, data in self.data_manager.metadata_store.items():
            if key.startswith("sra_") and key.endswith("_samples"):
                samples = data.get("samples", [])
                if isinstance(data, dict) and "data" in data:
                    # Handle nested structure from workspace
                    samples = data.get("data", {}).get("samples", [])
                total_samples += len(samples) if isinstance(samples, list) else 0
                bioproject_count += 1
                sources.append(key)

        # Also check workspace/metadata for sra_*_samples.json files
        metadata_dir = self._workspace_path / "metadata"
        if metadata_dir.exists():
            for f in metadata_dir.glob("sra_*_samples.json"):
                if f.stem not in sources:
                    try:
                        with open(f, "r") as fp:
                            data = json.load(fp)
                            samples = data.get("data", {}).get("samples", [])
                            total_samples += (
                                len(samples) if isinstance(samples, list) else 0
                            )
                            bioproject_count += 1
                            sources.append(f.stem)
                    except (json.JSONDecodeError, IOError):
                        continue

        return {
            "total_samples": total_samples,
            "filtered_samples": 0,
            "disease_coverage": 0,
            "filter_criteria": "",
            "bioproject_count": bioproject_count,
            "has_aggregated": False,
            "sources": sources[:20],  # Limit for display
            "message": "Run metadata filtering to generate aggregated statistics"
            if total_samples > 0
            else "No sample metadata found. Process publications first.",
        }

    def get_workspace_inventory(self) -> Dict[str, Any]:
        """
        Get categorized file inventory across all storage locations.

        Returns:
            Dict with:
            - metadata_store_count: In-memory metadata entries
            - workspace_files: Dict by category
            - exports: Export file info
            - total_size_mb: Total workspace size
            - deprecated_warnings: Files in old locations
        """
        result = {
            "metadata_store_count": len(self.data_manager.metadata_store),
            "workspace_files": {},
            "exports": [],
            "total_size_mb": 0,
            "deprecated_warnings": [],
        }

        # Categorize metadata_store entries
        store_categories = Counter()
        for key in self.data_manager.metadata_store.keys():
            if key.startswith("sra_") and key.endswith("_samples"):
                store_categories["sra_samples"] += 1
            elif key.startswith("pub_queue_"):
                store_categories["publication_metadata"] += 1
            elif key.startswith("geo_") or key.startswith("GSE"):
                store_categories["geo_datasets"] += 1
            elif key == "aggregated_filtered_samples":
                store_categories["aggregated"] += 1
            else:
                store_categories["other"] += 1
        result["metadata_store_categories"] = dict(store_categories)

        # Scan workspace/metadata directory
        metadata_dir = self._workspace_path / "metadata"
        if metadata_dir.exists():
            file_categories = Counter()
            total_size = 0
            for f in metadata_dir.glob("*.json"):
                size = f.stat().st_size
                total_size += size
                if f.stem.startswith("sra_") and f.stem.endswith("_samples"):
                    file_categories["sra_samples"] += 1
                elif "_metadata" in f.stem:
                    file_categories["publication_metadata"] += 1
                elif "_methods" in f.stem:
                    file_categories["methods"] += 1
                elif "_identifiers" in f.stem:
                    file_categories["identifiers"] += 1
                elif f.stem.startswith("aggregated"):
                    file_categories["aggregated"] += 1
                else:
                    file_categories["other"] += 1

            result["workspace_files"] = dict(file_categories)
            result["workspace_files_total"] = sum(file_categories.values())
            result["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        # Scan exports directory
        exports_dir = self._workspace_path / "exports"
        if exports_dir.exists():
            export_files = []
            for f in sorted(
                exports_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
            ):
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx", ".json"}:
                    size_kb = f.stat().st_size / 1024
                    export_files.append(
                        {
                            "name": f.name,
                            "size_kb": round(size_kb, 1),
                            "modified": datetime.fromtimestamp(
                                f.stat().st_mtime
                            ).strftime("%Y-%m-%d %H:%M"),
                        }
                    )
            result["exports"] = export_files[:15]
            result["exports_total"] = len(export_files)

        # Check for deprecated location
        old_exports = self._workspace_path / "metadata" / "exports"
        if old_exports.exists():
            old_files = list(old_exports.glob("*"))
            if old_files:
                result["deprecated_warnings"].append(
                    f"Found {len(old_files)} file(s) in deprecated location: metadata/exports/"
                )

        return result

    def get_export_summary(self) -> Dict[str, Any]:
        """
        Get export files with categories and usage guidance.

        Returns:
            Dict with:
            - files: List of export file info
            - categories: Counter of file categories
            - total_count: Total export files
            - usage_hints: Usage commands for accessing exports
        """
        exports_dir = self._workspace_path / "exports"

        if not exports_dir.exists():
            return {
                "files": [],
                "categories": {},
                "total_count": 0,
                "message": "No exports directory found. Use write_to_workspace() to export data.",
            }

        files = []
        categories = Counter()

        for f in sorted(
            exports_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx", ".json"}:
                size_kb = f.stat().st_size / 1024
                modified = datetime.fromtimestamp(f.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )

                # Categorize by filename pattern
                category = "other"
                name_lower = f.name.lower()
                if "_rich" in name_lower:
                    category = "rich_export"
                elif "_strict" in name_lower:
                    category = "strict_mimarks"
                elif "_harmonization_log" in name_lower:
                    category = "provenance_log"
                elif "samples" in name_lower or "filtered" in name_lower:
                    category = "sample_data"
                elif "de_" in name_lower or "differential" in name_lower:
                    category = "analysis_results"
                categories[category] += 1

                files.append(
                    {
                        "name": f.name,
                        "size_kb": round(size_kb, 1),
                        "modified": modified,
                        "category": category,
                        "path": str(f),
                    }
                )

        return {
            "files": files[:20],
            "categories": dict(categories),
            "total_count": len(files),
            "usage_hints": {
                "list": "get_content_from_workspace(workspace='exports')",
                "access": "execute_custom_code(load_workspace_files=True)",
                "cli": "/data exports",
            },
        }

    def get_quick_overview(self) -> Dict[str, Any]:
        """
        Get compact summary for default /metadata command.

        Combines key metrics from all summaries into single overview.
        """
        pub_summary = self.get_publication_queue_summary()
        sample_stats = self.get_sample_statistics()
        workspace_inv = self.get_workspace_inventory()

        # Build next steps based on current state
        next_steps = []

        # Publication queue actions
        status = pub_summary.get("status_breakdown", {})
        handoff_ready = status.get("handoff_ready", 0)
        failed = status.get("failed", 0)
        pending = status.get("pending", 0)

        if handoff_ready > 0:
            next_steps.append(
                f"{handoff_ready} entries ready for filtering → /metadata publications --status=handoff_ready"
            )
        if failed > 0:
            next_steps.append(
                f"{failed} failed entries need attention → /metadata publications --status=failed"
            )
        if pending > 0 and pub_summary.get("total", 0) < 10:
            next_steps.append(
                f"{pending} entries pending extraction → wait for processing to complete"
            )

        # Sample actions
        if sample_stats.get("total_samples", 0) > 0 and not sample_stats.get(
            "has_aggregated"
        ):
            next_steps.append(
                "Samples available but not filtered → use metadata_assistant to filter"
            )
        elif (
            sample_stats.get("has_aggregated")
            and sample_stats.get("filtered_samples", 0) > 0
        ):
            next_steps.append(
                f"{sample_stats['filtered_samples']:,} filtered samples ready → /metadata exports"
            )

        # Export actions
        exports_count = workspace_inv.get("exports_total", 0)
        if exports_count > 0:
            next_steps.append(
                f"{exports_count} export files available → /metadata exports"
            )

        return {
            "publication_queue": {
                "total": pub_summary.get("total", 0),
                "status_breakdown": pub_summary.get("status_breakdown", {}),
                "extracted_datasets": sum(
                    pub_summary.get("extracted_datasets", {}).values()
                ),
                "workspace_ready": pub_summary.get("workspace_ready", 0),
            },
            "samples": {
                "total_samples": sample_stats.get("total_samples", 0),
                "filtered_samples": sample_stats.get("filtered_samples", 0),
                "disease_coverage": sample_stats.get("disease_coverage", 0),
                "bioproject_count": sample_stats.get("bioproject_count", 0),
                "has_aggregated": sample_stats.get("has_aggregated", False),
                "retention_rate": sample_stats.get("retention_rate", 0),
            },
            "workspace": {
                "metadata_files": workspace_inv.get("workspace_files_total", 0),
                "export_files": workspace_inv.get("exports_total", 0),
                "total_size_mb": workspace_inv.get("total_size_mb", 0),
                "in_memory_entries": workspace_inv.get("metadata_store_count", 0),
            },
            "next_steps": next_steps[:5],
            "has_deprecated": len(workspace_inv.get("deprecated_warnings", [])) > 0,
        }
