"""
Workspace content retrieval tool for accessing cached research content.

This module provides factory functions for creating shared tools that can be used
by multiple agents (research_agent, data_expert, supervisor):
- get_content_from_workspace: Access cached research content
- list_available_modalities: List loaded modalities with optional filtering
"""

import json
from typing import Optional

from langchain_core.tools import tool

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_management.modality_management_service import ModalityManagementService
from lobster.services.data_access.workspace_content_service import (
    ContentType,
    RetrievalLevel,
    WorkspaceContentService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_get_content_from_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create get_content_from_workspace tool with data_manager closure.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for retrieving workspace content
    """

    @tool
    def get_content_from_workspace(
        identifier: str = None,
        workspace: str = None,
        level: str = "summary",
        status_filter: str = None,
    ) -> str:
        """
        Retrieve cached research content from workspace with flexible detail levels.

        Reads previously cached publications, datasets, and metadata from workspace
        directories. Supports listing all content, filtering by workspace, and
        extracting specific details (summary, methods, samples, platform, full metadata).

        Workspace categories:
        - "literature": Publications, papers, abstracts
        - "data": Dataset metadata, GEO records
        - "metadata": Validation results, sample mappings
        - "download_queue": Pending/completed download tasks
        - "publication_queue": Pending/completed publication extraction tasks

        Detail Levels:
        - "summary": Key-value pairs, high-level overview (default)
        - "methods": Methods section (for publications)
        - "samples": Sample IDs list (for datasets)
        - "platform": Platform information (for datasets)
        - "metadata": Full metadata (for any content)
        - "github": GitHub repositories (for publications)
        - "validation": Validation results (for download_queue)
        - "strategy": Download strategy (for download_queue)

        For download_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
        - level: "summary" (basic info) | "metadata" (full entry details)

        For publication_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "pending" | "extracting" | "metadata_extracted" | "metadata_enriched" | "handoff_ready" | "completed" | "failed"
        - level: "summary" (basic info) | "metadata" (full entry details)

        Args:
            identifier: Content identifier to retrieve (None = list all)
            workspace: Filter by workspace category (None = all workspaces)
            level: Detail level to extract (default: "summary")
            status_filter: Status filter for download_queue (optional)

        Returns:
            Formatted content based on detail level or list of cached items

        Examples:
            # List all cached content
            get_content_from_workspace()

            # List content in specific workspace
            get_content_from_workspace(workspace="literature")

            # Read publication methods section
            get_content_from_workspace(
                identifier="publication_PMID12345",
                workspace="literature",
                level="methods"
            )

            # Get dataset sample IDs
            get_content_from_workspace(
                identifier="dataset_GSE12345",
                workspace="data",
                level="samples"
            )

            # Get full metadata
            get_content_from_workspace(
                identifier="metadata_GSE12345_samples",
                workspace="metadata",
                level="metadata"
            )

            # List pending downloads
            get_content_from_workspace(
                workspace="download_queue",
                status_filter="PENDING"
            )

            # Get download queue entry details
            get_content_from_workspace(
                identifier="queue_entry_123",
                workspace="download_queue",
                level="metadata"
            )

            # List pending publication extractions
            get_content_from_workspace(
                workspace="publication_queue",
                status_filter="pending"
            )

            # Get publication queue entry details
            get_content_from_workspace(
                identifier="pub_queue_456",
                workspace="publication_queue",
                level="metadata"
            )
        """
        try:
            # Initialize workspace service
            workspace_service = WorkspaceContentService(data_manager=data_manager)

            # Map workspace strings to ContentType enum
            workspace_to_content_type = {
                "literature": ContentType.PUBLICATION,
                "data": ContentType.DATASET,
                "metadata": ContentType.METADATA,
                "download_queue": ContentType.DOWNLOAD_QUEUE,
                "publication_queue": ContentType.PUBLICATION_QUEUE,
            }

            # Map level strings to RetrievalLevel enum
            level_to_retrieval = {
                "summary": RetrievalLevel.SUMMARY,
                "methods": RetrievalLevel.METHODS,
                "samples": RetrievalLevel.SAMPLES,
                "platform": RetrievalLevel.PLATFORM,
                "metadata": RetrievalLevel.FULL,
                "github": None,  # Special case, handle separately
                "validation": None,  # Special case for download_queue
                "strategy": None,  # Special case for download_queue
            }

            # Validate detail level using enum keys
            if level not in level_to_retrieval:
                valid = list(level_to_retrieval.keys())
                return f"Error: Invalid detail level '{level}'. Valid options: {', '.join(valid)}"

            # Validate workspace if provided
            if workspace and workspace not in workspace_to_content_type:
                valid_ws = list(workspace_to_content_type.keys())
                return f"Error: Invalid workspace '{workspace}'. Valid options: {', '.join(valid_ws)}"

            # List mode: Use service instead of manual scanning
            if identifier is None:
                # Special handling for download_queue workspace
                if workspace == "download_queue":
                    entries = workspace_service.list_download_queue_entries(
                        status_filter=status_filter
                    )

                    if not entries:
                        if status_filter:
                            return f"No download queue entries found with status '{status_filter}'."
                        return "Download queue is empty."

                    # Format response based on level
                    if level == "summary":
                        response = f"## Download Queue ({len(entries)} entries)\n\n"
                        for entry in entries:
                            response += (
                                f"- **{entry['entry_id']}**: {entry['dataset_id']} "
                            )
                            response += (
                                f"({entry['status']}, priority {entry['priority']})\n"
                            )
                            if entry.get("modality_name"):
                                response += (
                                    f"  └─> Loaded as: {entry['modality_name']}\n"
                                )
                        return response

                    elif level == "metadata":
                        # Full details for all entries
                        response = f"## Download Queue Entries ({len(entries)})\n\n"
                        for entry in entries:
                            response += _format_queue_entry_full(entry) + "\n\n"
                        return response

                # Special handling for publication_queue workspace
                if workspace == "publication_queue":
                    entries = workspace_service.list_publication_queue_entries(
                        status_filter=status_filter
                    )

                    if not entries:
                        if status_filter:
                            return f"No publication queue entries found with status '{status_filter}'."
                        return "Publication queue is empty."

                    # Format response based on level
                    if level == "summary":
                        # Token-efficient aggregated summary (avoid listing all entries)
                        from collections import Counter
                        from datetime import datetime

                        # Compute statistics
                        status_counts = Counter(entry['status'] for entry in entries)

                        # Priority distribution (1-3=high, 4-7=medium, 8-10=low)
                        priority_high = sum(1 for e in entries if e['priority'] <= 3)
                        priority_medium = sum(1 for e in entries if 4 <= e['priority'] <= 7)
                        priority_low = sum(1 for e in entries if e['priority'] >= 8)

                        # Failed entries count
                        failed_count = status_counts.get('failed', 0)

                        # Sort by updated_at (most recent first), show top 5
                        sorted_entries = sorted(
                            entries,
                            key=lambda e: e.get('updated_at', ''),
                            reverse=True
                        )[:5]

                        # Build response
                        response = f"## Publication Queue Summary\n\n"
                        response += f"**Total Entries**: {len(entries)}\n\n"

                        # Status breakdown
                        response += "**Status Breakdown**:\n"
                        for status in ['pending', 'extracting', 'metadata_extracted', 'metadata_enriched', 'handoff_ready', 'completed', 'failed', 'paywalled']:
                            count = status_counts.get(status, 0)
                            if count > 0:
                                response += f"- {status}: {count} entries\n"
                        response += "\n"

                        # Priority distribution
                        response += "**Priority Distribution**:\n"
                        response += f"- High priority (1-3): {priority_high} entries\n"
                        response += f"- Medium priority (4-7): {priority_medium} entries\n"
                        response += f"- Low priority (8-10): {priority_low} entries\n\n"

                        # Recent activity
                        response += "**Recent Activity** (last 5 updates):\n"
                        for entry in sorted_entries:
                            title = entry.get('title', 'Untitled')
                            title_short = title[:50] + "..." if len(title) > 50 else title
                            updated = entry.get('updated_at', 'unknown')

                            # Format time ago if possible
                            try:
                                if isinstance(updated, str):
                                    updated_dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                                else:
                                    updated_dt = updated
                                time_diff = datetime.now() - updated_dt.replace(tzinfo=None)

                                if time_diff.days > 0:
                                    time_ago = f"{time_diff.days}d ago"
                                elif time_diff.seconds >= 3600:
                                    time_ago = f"{time_diff.seconds // 3600}h ago"
                                elif time_diff.seconds >= 60:
                                    time_ago = f"{time_diff.seconds // 60}m ago"
                                else:
                                    time_ago = "just now"
                            except:
                                time_ago = "unknown"

                            response += f"- **{entry['entry_id']}**: \"{title_short}\" ({entry['status']}) - Updated {time_ago}\n"
                        response += "\n"

                        # Problem indicators
                        if failed_count > 0:
                            response += f"**Failed Entries**: {failed_count} (use status_filter='failed' to inspect)\n\n"

                        # Actionable guidance
                        response += "**Tip**: Use `status_filter` parameter to focus on specific statuses:\n"
                        response += "- `status_filter='handoff_ready'` - Ready for metadata processing\n"
                        response += "- `status_filter='failed'` - Entries needing attention\n"
                        response += "- `status_filter='pending'` - Not yet started\n"
                        response += "\nUse `level='metadata'` for detailed inspection of all entries.\n"

                        return response

                    elif level == "metadata":
                        # Full details for all entries
                        response = f"## Publication Queue Entries ({len(entries)})\n\n"
                        for entry in entries:
                            response += _format_pub_queue_entry_full(entry) + "\n\n"
                        return response

                logger.info("Listing all cached workspace content")

                # Determine content type filter
                content_type_filter = None
                if workspace:
                    content_type_filter = workspace_to_content_type[workspace]

                # Use service to list content (replaces manual glob + JSON reading)
                all_cached = workspace_service.list_content(
                    content_type=content_type_filter
                )

                if not all_cached:
                    filter_msg = f" in workspace '{workspace}'" if workspace else ""
                    return f"No cached content found{filter_msg}. Use write_to_workspace() to cache content first."

                # Format list response (same output format)
                response = f"## Cached Workspace Content ({len(all_cached)} items)\n\n"
                for item in all_cached:
                    response += f"- **{item['identifier']}**\n"
                    response += (
                        f"  - Workspace: {item.get('_content_type', 'unknown')}\n"
                    )
                    response += f"  - Type: {item.get('content_type', 'unknown')}\n"
                    response += f"  - Cached: {item.get('cached_at', 'unknown')}\n\n"
                return response

            # Handle download_queue retrieval mode
            if workspace == "download_queue":
                try:
                    entry = workspace_service.read_download_queue_entry(identifier)
                except FileNotFoundError as e:
                    return f"Error: {str(e)}"

                if level == "summary":
                    return _format_queue_entry_summary(entry)
                elif level == "metadata":
                    return json.dumps(entry, indent=2, default=str)
                elif level == "validation":
                    if entry.get("validation_result"):
                        return json.dumps(entry["validation_result"], indent=2)
                    return "No validation result available for this entry."
                elif level == "strategy":
                    if entry.get("recommended_strategy"):
                        return json.dumps(entry["recommended_strategy"], indent=2)
                    return "No recommended strategy available for this entry."

            # Handle publication_queue retrieval mode
            if workspace == "publication_queue":
                try:
                    entry = workspace_service.read_publication_queue_entry(identifier)
                except FileNotFoundError as e:
                    return f"Error: {str(e)}"

                if level == "summary":
                    return _format_pub_queue_entry_summary(entry)
                elif level == "metadata":
                    return json.dumps(entry, indent=2, default=str)

            # Retrieve mode: Handle "github" level specially (not in RetrievalLevel enum)
            if level == "github":
                # Try each content type if workspace not specified
                if workspace:
                    content_types_to_try = [workspace_to_content_type[workspace]]
                else:
                    content_types_to_try = list(ContentType)

                cached_content = None
                for content_type in content_types_to_try:
                    try:
                        # GitHub requires full content retrieval
                        cached_content = workspace_service.read_content(
                            identifier=identifier,
                            content_type=content_type,
                            level=RetrievalLevel.FULL,
                        )
                        break
                    except FileNotFoundError:
                        continue

                if not cached_content:
                    workspace_filter = (
                        f" in workspace '{workspace}'" if workspace else ""
                    )
                    return f"Error: Identifier '{identifier}' not found{workspace_filter}. Available content:\n{get_content_from_workspace(workspace=workspace)}"

                # Extract GitHub repos from data
                data = cached_content.get("data", {})
                if "github_repos" in data:
                    repos = data["github_repos"]
                    response = f"## GitHub Repositories for {identifier}\n\n"
                    response += f"**Found**: {len(repos)} repositories\n\n"
                    for repo in repos:
                        response += f"- {repo}\n"
                    return response
                else:
                    return f"No GitHub repositories found for '{identifier}'. This detail level is typically for publications with code."

            # Standard level retrieval using service with automatic filtering
            retrieval_level = level_to_retrieval[level]

            # Try each content type if workspace not specified
            if workspace:
                content_types_to_try = [workspace_to_content_type[workspace]]
            else:
                content_types_to_try = list(ContentType)

            cached_content = None
            found_content_type = None

            for content_type in content_types_to_try:
                try:
                    # Use service with level-based filtering (replaces manual if/elif)
                    cached_content = workspace_service.read_content(
                        identifier=identifier,
                        content_type=content_type,
                        level=retrieval_level,  # Service handles filtering automatically
                    )
                    found_content_type = content_type
                    break
                except FileNotFoundError:
                    continue

            if not cached_content:
                workspace_filter = f" in workspace '{workspace}'" if workspace else ""
                return f"Error: Identifier '{identifier}' not found{workspace_filter}. Available content:\n{get_content_from_workspace(workspace=workspace)}"

            # Format response based on level (service already filtered content)
            if level == "summary":
                data = cached_content.get("data", {})
                response = f"""## Summary: {identifier}

**Workspace**: {found_content_type.value if found_content_type else 'unknown'}
**Content Type**: {cached_content.get('content_type', 'unknown')}
**Cached At**: {cached_content.get('cached_at', 'unknown')}

**Data Overview**:
"""
                # Format data as key-value pairs
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            value_str = (
                                f"{type(value).__name__} with {len(value)} items"
                            )
                        else:
                            value_str = str(value)[:100]
                            if len(str(value)) > 100:
                                value_str += "..."
                        response += f"- **{key}**: {value_str}\n"
                return response

            elif level == "methods":
                # Service already filtered to methods fields
                if "methods" in cached_content:
                    return f"## Methods Section\n\n{cached_content['methods']}"
                else:
                    return f"No methods section found for '{identifier}'. This detail level is typically for publications."

            elif level == "samples":
                # Service already filtered to samples fields
                if "samples" in cached_content:
                    sample_list = cached_content["samples"]
                    response = f"## Sample IDs for {identifier}\n\n"
                    response += f"**Total Samples**: {len(sample_list)}\n\n"
                    response += "\n".join(f"- {sample}" for sample in sample_list[:50])
                    if len(sample_list) > 50:
                        response += f"\n\n... and {len(sample_list) - 50} more samples"
                    return response
                elif "obs_columns" in cached_content:
                    # For modalities, show obs_columns
                    return f"## Sample Information for {identifier}\n\n**N Observations**: {cached_content.get('n_obs')}\n**Obs Columns**: {', '.join(cached_content['obs_columns'])}"
                else:
                    return f"No sample information found for '{identifier}'. This detail level is typically for datasets."

            elif level == "platform":
                # Service already filtered to platform fields
                if "platform" in cached_content:
                    return f"## Platform Information\n\n{json.dumps(cached_content['platform'], indent=2)}"
                elif "var_columns" in cached_content:
                    # For modalities, show var_columns
                    return f"## Platform/Feature Information for {identifier}\n\n**N Variables**: {cached_content.get('n_vars')}\n**Var Columns**: {', '.join(cached_content['var_columns'])}"
                else:
                    return f"No platform information found for '{identifier}'. This detail level is typically for datasets."

            elif level == "metadata":
                # Service already filtered (full metadata)
                return f"## Full Metadata for {identifier}\n\n```json\n{json.dumps(cached_content, indent=2)}\n```"

            else:
                return f"Unsupported detail level: {level}"

        except Exception as e:
            logger.error(f"Error retrieving workspace content: {e}")
            return f"Error retrieving content from workspace: {str(e)}"

    def _format_queue_entry_summary(entry: dict) -> str:
        """Format queue entry as summary."""

        summary = f"## Download Queue Entry: {entry['entry_id']}\n\n"
        summary += f"**Dataset**: {entry['dataset_id']}\n"
        summary += f"**Database**: {entry['database']}\n"
        summary += f"**Status**: {entry['status']}\n"
        summary += f"**Priority**: {entry['priority']}/10\n"
        summary += f"**Created**: {entry['created_at']}\n"

        if entry.get("modality_name"):
            summary += f"**Modality**: {entry['modality_name']}\n"

        if entry.get("validation_result"):
            summary += "\n**Validation**: Available (use level='validation' to view)\n"

        if entry.get("recommended_strategy"):
            strategy = entry["recommended_strategy"]
            summary += f"\n**Recommended Strategy**: {strategy['strategy_name']}\n"
            summary += f"  - Confidence: {strategy['confidence']:.2%}\n"

        if entry.get("error_log"):
            summary += f"\n**Errors**: {len(entry['error_log'])} error(s) logged\n"

        return summary

    def _format_queue_entry_full(entry: dict) -> str:
        """Format queue entry with full details."""
        full = _format_queue_entry_summary(entry)

        # Add URLs if present
        urls = []
        if entry.get("matrix_url"):
            urls.append(f"- Matrix: {entry['matrix_url']}")
        if entry.get("h5_url"):
            urls.append(f"- H5: {entry['h5_url']}")
        if entry.get("raw_urls"):
            urls.append(f"- Raw: {len(entry['raw_urls'])} file(s)")

        if urls:
            full += "\n**URLs**:\n" + "\n".join(urls) + "\n"

        return full

    def _format_pub_queue_entry_summary(entry: dict) -> str:
        """Format publication queue entry as summary."""

        summary = f"## Publication Queue Entry: {entry['entry_id']}\n\n"

        if entry.get("title"):
            summary += f"**Title**: {entry['title']}\n"

        if entry.get("authors"):
            authors = entry["authors"]
            if len(authors) > 3:
                author_str = ", ".join(authors[:3]) + f" et al. ({len(authors)} total)"
            else:
                author_str = ", ".join(authors)
            summary += f"**Authors**: {author_str}\n"

        if entry.get("journal"):
            summary += f"**Journal**: {entry['journal']}\n"

        if entry.get("year"):
            summary += f"**Year**: {entry['year']}\n"

        # Identifiers
        identifiers = []
        if entry.get("pmid"):
            identifiers.append(f"PMID: {entry['pmid']}")
        if entry.get("doi"):
            identifiers.append(f"DOI: {entry['doi']}")
        if entry.get("pmc_id"):
            identifiers.append(f"PMC: {entry['pmc_id']}")
        if identifiers:
            summary += f"**Identifiers**: {', '.join(identifiers)}\n"

        summary += f"\n**Status**: {entry['status']}\n"
        summary += f"**Priority**: {entry['priority']}/10\n"
        summary += f"**Schema Type**: {entry.get('schema_type', 'general')}\n"
        summary += f"**Extraction Level**: {entry.get('extraction_level', 'methods')}\n"
        summary += f"**Created**: {entry.get('created_at', 'unknown')}\n"

        if entry.get("cached_content_path"):
            summary += f"\n**Cached Content**: {entry['cached_content_path']}\n"

        if entry.get("extracted_identifiers"):
            ids = entry["extracted_identifiers"]
            if any(ids.values()):
                summary += "\n**Extracted Identifiers**:\n"
                for id_type, id_list in ids.items():
                    if id_list:
                        summary += f"  - {id_type}: {', '.join(id_list[:5])}"
                        if len(id_list) > 5:
                            summary += f" (+{len(id_list) - 5} more)"
                        summary += "\n"

        if entry.get("error"):
            summary += f"\n**Error**: {entry['error']}\n"

        return summary

    def _format_pub_queue_entry_full(entry: dict) -> str:
        """Format publication queue entry with full details."""
        full = _format_pub_queue_entry_summary(entry)

        # Add URLs if present
        if entry.get("metadata_url"):
            full += f"\n**Metadata URL**: {entry['metadata_url']}\n"

        if entry.get("supplementary_files"):
            files = entry["supplementary_files"]
            full += f"\n**Supplementary Files**: {len(files)} file(s)\n"
            for file_url in files[:3]:
                full += f"  - {file_url}\n"
            if len(files) > 3:
                full += f"  - ... and {len(files) - 3} more\n"

        if entry.get("github_url"):
            full += f"\n**GitHub URL**: {entry['github_url']}\n"

        return full

    return get_content_from_workspace


def create_write_to_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create write_to_workspace tool with data_manager closure.

    Shared between research_agent and metadata_assistant for workspace export
    with JSON and CSV format support.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for writing content to workspace
    """
    from datetime import datetime

    from lobster.services.data_access.workspace_content_service import (
        ContentType,
        MetadataContent,
        WorkspaceContentService,
    )

    @tool
    def write_to_workspace(
        identifier: str, workspace: str, content_type: str = None, output_format: str = "json"
    ) -> str:
        """
        Cache research content to workspace for later retrieval and specialist handoff.

        Stores publications, datasets, and metadata in organized workspace directories
        for persistent access. Validates naming conventions and content standardization.

        Workspace Categories:
        - "literature": Publications, abstracts, methods sections
        - "data": Dataset metadata, sample information
        - "metadata": Standardized metadata schemas

        Output Formats:
        - "json": Structured JSON format (default)
        - "csv": Tabular CSV format (best for sample metadata tables)

        Args:
            identifier: Content identifier to cache (must exist in current session)
            workspace: Target workspace category ("literature", "data", "metadata")
            content_type: Type of content ("publication", "dataset", "metadata")
            output_format: Output format ("json" or "csv"). Default: "json"

        Returns:
            Confirmation message with storage location and next steps
        """
        try:
            workspace_service = WorkspaceContentService(data_manager=data_manager)

            workspace_to_content_type = {
                "literature": ContentType.PUBLICATION,
                "data": ContentType.DATASET,
                "metadata": ContentType.METADATA,
            }

            if workspace not in workspace_to_content_type:
                return f"Error: Invalid workspace '{workspace}'. Valid: {', '.join(workspace_to_content_type.keys())}"

            if content_type and content_type not in {"publication", "dataset", "metadata"}:
                return f"Error: Invalid content_type '{content_type}'. Valid: publication, dataset, metadata"

            if output_format not in {"json", "csv"}:
                return f"Error: Invalid output_format '{output_format}'. Valid: json, csv"

            # Check if identifier exists in session
            exists = False
            content_data = None
            source_location = None

            if identifier in data_manager.metadata_store:
                exists = True
                content_data = data_manager.metadata_store[identifier]
                source_location = "metadata_store"
                logger.info(f"Found '{identifier}' in metadata_store")
            elif identifier in data_manager.list_modalities():
                exists = True
                adata = data_manager.get_modality(identifier)
                content_data = {
                    "n_obs": adata.n_obs,
                    "n_vars": adata.n_vars,
                    "obs_columns": list(adata.obs.columns),
                    "var_columns": list(adata.var.columns),
                }
                source_location = "modalities"
                logger.info(f"Found '{identifier}' in modalities")

            if not exists:
                return f"Error: Identifier '{identifier}' not found in current session."

            content_model = MetadataContent(
                identifier=identifier,
                content_type=content_type or "unknown",
                description=f"Cached from {source_location}",
                data=content_data,
                related_datasets=[],
                source=f"DataManager.{source_location}",
                cached_at=datetime.now().isoformat(),
            )

            cache_file_path = workspace_service.write_content(
                content=content_model,
                content_type=workspace_to_content_type[workspace],
                output_format=output_format,
            )

            response = f"""## Content Cached Successfully

**Identifier**: {identifier}
**Workspace**: {workspace}
**Output Format**: {output_format.upper()}
**Location**: {cache_file_path}

**Next Steps**:
- Use `get_content_from_workspace()` to retrieve cached content
"""
            if output_format == "csv":
                response += "**Note**: CSV format ideal for spreadsheet import.\n"

            return response

        except Exception as e:
            logger.error(f"Error caching to workspace: {e}")
            return f"Error caching content to workspace: {str(e)}"

    return write_to_workspace


def create_list_modalities_tool(data_manager: DataManagerV2):
    """
    Factory function to create list_available_modalities tool with data_manager closure.

    Shared between supervisor and data_expert agents for consistent modality listing
    with provenance tracking and optional filtering.

    Args:
        data_manager: DataManagerV2 instance for modality access

    Returns:
        LangChain tool for listing modalities with optional filtering
    """

    # Initialize service once (closure captures this)
    modality_service = ModalityManagementService(data_manager)

    @tool
    def list_available_modalities(filter_pattern: Optional[str] = None) -> str:
        """
        List all available modalities with optional filtering.

        Args:
            filter_pattern: Optional glob-style pattern to filter modality names
                          (e.g., "geo_gse*", "*clustered", "bulk_*")
                          If None, lists all modalities.

        Returns:
            str: Formatted list of modalities with details
        """
        try:
            modality_info, stats, ir = modality_service.list_modalities(
                filter_pattern=filter_pattern
            )

            # Log to provenance (W3C-PROV compliant)
            data_manager.log_tool_usage(
                tool_name="list_available_modalities",
                parameters={"filter_pattern": filter_pattern},
                description=stats,
                ir=ir,
            )

            # Format response
            if not modality_info:
                return "No modalities found matching the criteria."

            response = f"## Available Modalities ({stats['matched_modalities']}/{stats['total_modalities']})\n\n"
            if filter_pattern:
                response += f"**Filter**: `{filter_pattern}`\n\n"

            for info in modality_info:
                if "error" in info:
                    response += f"- **{info['name']}**: Error - {info['error']}\n"
                else:
                    response += f"- **{info['name']}**: {info['n_obs']} obs × {info['n_vars']} vars\n"
                    if info["obs_columns"]:
                        response += f"  - Obs: {', '.join(info['obs_columns'][:3])}\n"
                    if info["var_columns"]:
                        response += f"  - Var: {', '.join(info['var_columns'][:3])}\n"

            # Add workspace info (useful for supervisor context)
            workspace_status = data_manager.get_workspace_status()
            response += f"\n**Workspace**: {workspace_status['workspace_path']}\n"

            return response

        except Exception as e:
            logger.error(f"Error listing modalities: {e}")
            return f"Error listing modalities: {str(e)}"

    return list_available_modalities
