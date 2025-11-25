"""Services for orchestrating publication queue extraction workflows."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.publication_queue import PublicationStatus
from lobster.tools.content_access_service import ContentAccessService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PublicationProcessingService:
    """High-level orchestration for publication queue extraction."""

    def __init__(self, data_manager: DataManagerV2) -> None:
        self.data_manager = data_manager
        self.content_service = ContentAccessService(data_manager=data_manager)

    def process_entry(
        self, entry_id: str, extraction_tasks: str = "metadata,methods,identifiers"
    ) -> str:
        """Process a single publication queue entry."""

        data_manager = self.data_manager

        try:
            try:
                entry = data_manager.publication_queue.get_entry(entry_id)
            except Exception as e:  # pragma: no cover - defensive
                return (
                    "## Error: Publication Queue Entry Not Found\n\n"
                    f"Entry ID '{entry_id}' not found in publication queue.\n\n"
                    f"**Error**: {str(e)}\n\n"
                    "**Tip**: Use get_content_from_workspace(workspace='publication_queue') "
                    "to list available entries."
                )

            tasks = [task.strip().lower() for task in extraction_tasks.split(",")]

            data_manager.publication_queue.update_status(
                entry_id=entry_id,
                status=PublicationStatus.EXTRACTING,
                processed_by="research_agent",
            )

            response_parts = [
                f"## Processing Publication: {entry.title or entry.entry_id}",
                "",
                f"**Entry ID**: {entry_id}",
                "**Status**: EXTRACTING → Processing",
                "",
            ]

            extracted_data = {}
            identifiers_found = None
            content_result = None

            # Metadata extraction
            if "metadata" in tasks or "full_text" in tasks:
                try:
                    identifier = entry.pmid or entry.doi or entry.pmc_id
                    if identifier:
                        content_result = self.content_service.get_full_content(
                            source=identifier,
                            prefer_webpage=True,
                            keywords=["abstract", "introduction", "methods"],
                            max_paragraphs=100,
                        )
                        content = content_result.get("content", "") if content_result else ""
                        extracted_data["metadata_extracted"] = bool(content)
                        response_parts.append("✓ Metadata extracted successfully")
                    else:
                        response_parts.append(
                            "⚠ No identifier available for metadata extraction"
                        )
                except Exception as e:  # pragma: no cover - provider errors
                    response_parts.append(f"✗ Metadata extraction failed: {str(e)}")

            # Methods extraction
            if "methods" in tasks or "full_text" in tasks:
                try:
                    identifier = entry.pmid or entry.doi or entry.pmc_id
                    if identifier:
                        if not content_result:
                            content_result = self.content_service.get_full_content(
                                source=identifier
                            )

                        if content_result and content_result.get("content"):
                            methods_dict = self.content_service.extract_methods(
                                content_result
                            )
                            methods_content = methods_dict.get("methods_text", "")
                        else:
                            methods_content = ""

                        extracted_data["methods_extracted"] = bool(methods_content)
                        response_parts.append("✓ Methods section extracted successfully")
                    else:
                        response_parts.append(
                            "⚠ No identifier available for methods extraction"
                        )
                except Exception as e:  # pragma: no cover - provider errors
                    response_parts.append(f"✗ Methods extraction failed: {str(e)}")

            # Identifier extraction
            if "identifiers" in tasks or "full_text" in tasks:
                try:
                    identifier = entry.pmid or entry.doi or entry.pmc_id
                    if identifier:
                        if not content_result:
                            content_result = self.content_service.get_full_content(
                                source=identifier
                            )

                        full_content = (
                            content_result.get("content", "") if content_result else ""
                        )
                        extracted_data["identifiers_extracted"] = bool(full_content)

                        import re

                        identifiers_found = {
                            "geo": re.findall(r"GSE\d+", full_content),
                            "sra": re.findall(r"SRP\d+|SRX\d+|SRR\d+", full_content),
                            "bioproject": re.findall(r"PRJNA\d+", full_content),
                            "biosample": re.findall(r"SAMN\d+", full_content),
                            "ena": re.findall(r"E-[A-Z]+-\d+", full_content),
                        }

                        for key in identifiers_found:
                            identifiers_found[key] = list(set(identifiers_found[key]))

                        data_manager.publication_queue.update_status(
                            entry_id=entry_id,
                            status=PublicationStatus.METADATA_EXTRACTED,
                            extracted_identifiers=identifiers_found,
                            processed_by="research_agent",
                        )

                        extracted_data["identifiers"] = identifiers_found

                        total_ids = sum(len(v) for v in identifiers_found.values())
                        if total_ids > 0:
                            response_parts.append(
                                f"✓ Found {total_ids} dataset identifiers:"
                            )
                            for id_type, id_list in identifiers_found.items():
                                if id_list:
                                    response_parts.append(
                                        f"  - {id_type.upper()}: {', '.join(id_list[:5])}"
                                    )
                                    if len(id_list) > 5:
                                        response_parts.append(
                                            f"    (+{len(id_list) - 5} more)"
                                        )
                        else:
                            response_parts.append(
                                "⚠ No dataset identifiers found in publication"
                            )
                    else:
                        response_parts.append(
                            "⚠ No identifier available for identifier extraction"
                        )
                except Exception as e:  # pragma: no cover - provider errors
                    response_parts.append(f"✗ Identifier extraction failed: {str(e)}")

            # Persist extracted data to workspace
            workspace_keys: List[str] = []
            try:
                from pathlib import Path

                metadata_dir = Path(
                    data_manager.workspace.get_workspace_dir("metadata")
                )
                metadata_dir.mkdir(parents=True, exist_ok=True)

                if extracted_data.get("metadata_extracted") and content_result:
                    metadata_file = metadata_dir / f"{entry_id}_metadata.json"
                    metadata_content = {
                        "content": content_result.get("content", ""),
                        "summary": content_result.get("summary"),
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "authors": entry.authors,
                        "year": entry.year,
                        "journal": entry.journal,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "metadata",
                    }
                    metadata_file.write_text(json.dumps(metadata_content, indent=2))
                    workspace_keys.append(f"{entry_id}_metadata.json")
                    logger.info("Saved metadata to %s", metadata_file)

                if extracted_data.get("methods_extracted") and "methods_content" in locals():
                    methods_file = metadata_dir / f"{entry_id}_methods.json"
                    methods_data = {
                        "methods_text": methods_content,
                        "methods_dict": methods_dict if "methods_dict" in locals() else {},
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "methods",
                    }
                    methods_file.write_text(json.dumps(methods_data, indent=2))
                    workspace_keys.append(f"{entry_id}_methods.json")
                    logger.info("Saved methods to %s", methods_file)

                if extracted_data.get("identifiers_extracted") and identifiers_found:
                    identifiers_file = metadata_dir / f"{entry_id}_identifiers.json"
                    identifiers_data = {
                        "identifiers": identifiers_found,
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "full_content_length": len(full_content)
                        if "full_content" in locals()
                        else 0,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "identifiers",
                    }
                    identifiers_file.write_text(json.dumps(identifiers_data, indent=2))
                    workspace_keys.append(f"{entry_id}_identifiers.json")
                    logger.info("Saved identifiers to %s", identifiers_file)

                if workspace_keys:
                    response_parts.append(
                        f"✓ Saved {len(workspace_keys)} files to workspace/metadata/"
                    )

            except Exception as e:  # pragma: no cover - workspace errors
                logger.error("Failed to save extracted data to workspace: %s", e)
                response_parts.append(
                    f"⚠ Warning: Workspace persistence failed: {str(e)}"
                )

            final_status = (
                PublicationStatus.COMPLETED.value
                if extracted_data
                else PublicationStatus.FAILED.value
            )
            data_manager.publication_queue.update_status(
                entry_id=entry_id,
                status=final_status,
                processed_by="research_agent",
                workspace_metadata_keys=workspace_keys,
            )

            data_manager.log_tool_usage(
                tool_name="process_publication_entry",
                parameters={
                    "entry_id": entry_id,
                    "extraction_tasks": extraction_tasks,
                    "tasks": tasks,
                    "final_status": final_status,
                    "extracted_identifiers": identifiers_found
                    if "identifiers" in tasks
                    else None,
                    "title": entry.title or "N/A",
                    "pmid": entry.pmid,
                    "doi": entry.doi,
                },
                description=(
                    f"Processed publication entry {entry_id}: {entry.title or 'N/A'} "
                    f"[{final_status}]"
                ),
            )

            response_parts.extend(
                [
                    "",
                    f"**Final Status**: {final_status.upper()}",
                    "",
                    "**Next Steps**:",
                    "- View extracted content: get_content_from_workspace("
                    + f"identifier='{entry_id}', workspace='publication_queue')",
                    "- Process more entries: process_publication_entry('next_entry_id')",
                ]
            )

            return "\n".join(response_parts)

        except Exception as e:  # pragma: no cover - outer safety net
            logger.error("Failed to process publication entry %s: %s", entry_id, e)
            return (
                "## Error Processing Publication Entry\n\n"
                f"Entry ID: {entry_id}\n\n"
                f"**Error**: {str(e)}\n\n"
                "**Tip**: Ensure the entry exists in the publication queue and try again."
            )

    def process_queue_entries(
        self,
        status_filter: str = "pending",
        max_entries: Optional[int] = None,
        extraction_tasks: str = "metadata,methods,identifiers",
    ) -> str:
        """Process multiple publication queue entries in sequence."""

        queue = getattr(self.data_manager, "publication_queue", None)
        if queue is None:
            return "Error: Publication queue is not initialized in DataManagerV2."

        status_enum = None
        if status_filter:
            try:
                status_enum = PublicationStatus(status_filter.lower())
            except ValueError:
                return (
                    "Error: Invalid status filter '"
                    + status_filter
                    + "'."
                )

        entries = queue.list_entries(status=status_enum)
        if not entries:
            scope = status_filter or "any"
            return f"No publication queue entries found with status '{scope}'."

        entries = sorted(entries, key=lambda e: e.created_at)
        if max_entries:
            entries = entries[: max(0, max_entries)]

        summary = [
            "## Publication Queue Processing",
            "",
            f"**Status Filter**: {status_filter or 'any'}",
            f"**Entries Selected**: {len(entries)}",
            f"**Extraction Tasks**: {extraction_tasks}",
            "",
        ]

        for idx, entry in enumerate(entries, start=1):
            summary.append(f"### Entry {idx}: {entry.title or entry.entry_id}")
            summary.append(f"- Entry ID: {entry.entry_id}")
            summary.append(f"- Current Status: {entry.status}")
            result = self.process_entry(
                entry.entry_id, extraction_tasks=extraction_tasks
            )
            summary.append(result)
            summary.append("")

        summary.append(
            f"Processed {len(entries)} publication entries from the queue successfully."
        )
        return "\n".join(summary)
