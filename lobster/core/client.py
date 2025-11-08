"""
Clean Agent Client Interface for LangGraph Multi-Agent System.
Provides a simple, extensible interface for both CLI and future UI implementations.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langfuse.langchain import CallbackHandler as LangfuseCallback
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from lobster.agents.graph import create_bioinformatics_graph
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.base_client import BaseClient

# Import shared archive handling utilities
from lobster.core.archive_utils import (
    ArchiveExtractor,
    ArchiveInspector,
    ContentDetector,
    ArchiveContentType,
    NestedArchiveInfo,
)

# Import extraction cache manager
from lobster.core.extraction_cache import ExtractionCacheManager

# Configure logging
logger = logging.getLogger(__name__)


class AgentClient(BaseClient):
    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        session_id: str = None,
        enable_reasoning: bool = True,
        enable_langfuse: bool = False,
        workspace_path: Optional[Path] = None,
        custom_callbacks: Optional[List] = None,
        manual_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent client with DataManagerV2.

        Args:
            data_manager: DataManagerV2 instance (creates new if None)
            session_id: Unique session identifier
            enable_reasoning: Show agent reasoning/thinking process
            enable_langfuse: Enable Langfuse debugging callback
            workspace_path: Path to workspace for file operations
            custom_callbacks: Additional callback handlers
            manual_model_params: Manual model parameter overrides
        """
        # Set up session
        self.session_id = (
            session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.enable_reasoning = enable_reasoning

        # Set up workspace
        self.workspace_path = workspace_path or Path.cwd()
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize DataManagerV2
        if data_manager is None:
            from rich.console import Console

            console = Console() if custom_callbacks else None
            self.data_manager = DataManagerV2(
                workspace_path=self.workspace_path, console=console
            )
            logger.info("Initialized with DataManagerV2 (modular multi-omics)")
        else:
            self.data_manager = data_manager

        # Set up callbacks
        self.callbacks = []
        if enable_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY"):
            self.callbacks.append(LangfuseCallback())
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)

        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        # Initialize graph - pass all callbacks
        self.graph = create_bioinformatics_graph(
            data_manager=self.data_manager,
            checkpointer=self.checkpointer,
            store=self.store,
            callback_handler=self.callbacks,  # Pass the list of callbacks
            manual_model_params=manual_model_params,  # Placeholder for future manual model params
        )

        # Conversation state
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "workspace": str(self.workspace_path),
        }

    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the agent system.

        Args:
            user_input: User's input text
            stream: Whether to stream the response

        Returns:
            Dictionary with response and metadata
        """
        # Add user message
        self.messages.append(HumanMessage(content=user_input))

        # Prepare graph input
        graph_input = {"messages": [HumanMessage(content=user_input)]}

        config = {
            "configurable": {"thread_id": self.session_id},
            "callbacks": self.callbacks,
            "recursion_limit": 100,  # Prevent hitting default limit of 25
        }

        if stream:
            return self._stream_query(graph_input, config)
        else:
            return self._run_query(graph_input, config)

    def _run_query(self, graph_input: Dict, config: Dict) -> Dict[str, Any]:
        """Run a query and return the complete response."""
        try:
            # Track execution
            start_time = datetime.now()
            events = []
            last_agent = None

            # Execute graph
            for event in self.graph.stream(
                input=graph_input, config=config, stream_mode="updates"
            ):
                events.append(event)

                # Track which agent is responding
                if event:
                    for node_name in event.keys():
                        if node_name and node_name != "__end__":
                            last_agent = node_name

            # Extract final response from the last event
            final_response = self._extract_response(events)

            # Update messages with the final response (not the raw events)
            if final_response:
                self.messages.append(AIMessage(content=final_response))

            return {
                "success": True,
                "response": final_response,
                "duration": (datetime.now() - start_time).total_seconds(),
                "events_count": len(events),
                "session_id": self.session_id,
                "has_data": self.data_manager.has_data(),
                "plots": (
                    self.data_manager.get_latest_plots(5)
                    if self.data_manager.has_data()
                    else []
                ),
                "last_agent": last_agent,  # Include which agent provided the response
            }

        except Exception as e:
            # Enhanced error information for better debugging
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "response": f"I encountered an error: {str(e)}",
                "session_id": self.session_id,
                "hint": "Run with --debug flag for detailed error information",
            }

    def _stream_query(
        self, graph_input: Dict, config: Dict
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream query execution with intermediate results."""
        try:
            start_time = datetime.now()

            for event in self.graph.stream(graph_input, config):
                # Process each event
                for node_name, node_output in event.items():
                    # Extract meaningful content
                    content = self._extract_event_content(node_output)

                    if content:
                        yield {
                            "type": "stream",
                            "node": node_name,
                            "content": content,
                            "timestamp": datetime.now().isoformat(),
                        }

            # Final response
            yield {
                "type": "complete",
                "duration": (datetime.now() - start_time).total_seconds(),
                "session_id": self.session_id,
            }

        except Exception as e:
            yield {"type": "error", "error": str(e), "session_id": self.session_id}

    def _extract_response(self, events: List[Dict]) -> str:
        """Extract the final response from events, expecting supervisor responses."""
        if not events:
            return "No response generated."

        # Process events in reverse chronological order to find the last supervisor response
        for event in reversed(events):
            # Check for supervisor key first
            if "supervisor" not in event:
                # Log any unexpected keys
                unexpected_keys = [key for key in event.keys() if key != "supervisor"]
                if unexpected_keys:
                    logger.warning(
                        f"Unexpected event keys found (expected 'supervisor'): {unexpected_keys}"
                    )
                continue

            supervisor_data = event["supervisor"]
            if (
                not isinstance(supervisor_data, dict)
                or "messages" not in supervisor_data
            ):
                continue

            messages = supervisor_data["messages"]
            if not isinstance(messages, list):
                continue

            # Find the last AIMessage in the supervisor's messages
            for msg in reversed(messages):
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    content = self._extract_content_from_message(msg.content)
                    if content:
                        return content

        return "No response generated."

    def _extract_content_from_message(self, content) -> str:
        """Extract text content from a message, handling both string and list formats."""
        # Handle backward compatibility - if content is still a string
        if isinstance(content, str):
            return content.strip()

        # Handle new list format with content blocks
        if isinstance(content, list):
            text_parts = []
            reasoning_parts = []

            for block in content:
                if isinstance(block, dict):
                    # Extract text content
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(block["text"])

                    # Extract reasoning content if enabled
                    elif (
                        block.get("type") == "reasoning_content"
                        and self.enable_reasoning
                    ):
                        if "reasoning_content" in block and isinstance(
                            block["reasoning_content"], dict
                        ):
                            reasoning_text = block["reasoning_content"].get("text", "")
                            if reasoning_text:
                                reasoning_parts.append(f"[Thinking: {reasoning_text}]")

            # Combine parts - show reasoning first if enabled, then the main text
            result_parts = []
            if reasoning_parts and self.enable_reasoning:
                result_parts.extend(reasoning_parts)
            if text_parts:
                result_parts.extend(text_parts)

            if result_parts:
                return "\n\n".join(result_parts).strip()

        # Fallback for any other format
        return str(content).strip() if content else ""

    def _extract_event_content(self, node_output: Dict) -> Optional[str]:
        """Extract displayable content from a node output."""
        if not isinstance(node_output, dict):
            return None

        # Check for messages - only return content from AI messages
        if "messages" in node_output and node_output["messages"]:
            # Look for the last AI message in this event
            for msg in reversed(node_output["messages"]):
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    return self._extract_content_from_message(msg.content)

        # Check for other relevant fields
        for key in ["analysis_results", "next", "data_context"]:
            if key in node_output and node_output[key]:
                return f"{key}: {node_output[key]}"

        return None

    # Enhanced file operations
    def detect_file_type(self, file_path: Path) -> Dict[str, Any]:
        """
        Detect file type with comprehensive format identification.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file type information
        """
        import mimetypes

        # File extension mapping for bioinformatics and common formats
        extension_map = {
            # Bioinformatics data formats
            ".h5ad": {
                "category": "bioinformatics",
                "type": "single_cell_data",
                "description": "Single-cell RNA-seq data (H5AD format)",
                "binary": True,
            },
            ".h5mu": {
                "category": "bioinformatics",
                "type": "multimodal_data",
                "description": "Multi-modal omics data (H5MU format)",
                "binary": True,
            },
            ".loom": {
                "category": "bioinformatics",
                "type": "genomics_data",
                "description": "Genomics data (Loom format)",
                "binary": True,
            },
            ".h5": {
                "category": "bioinformatics",
                "type": "hdf5_data",
                "description": "HDF5 data file",
                "binary": True,
            },
            ".mtx": {
                "category": "bioinformatics",
                "type": "matrix_data",
                "description": "Matrix Market sparse matrix",
                "binary": False,
            },
            ".mex": {
                "category": "bioinformatics",
                "type": "matrix_data",
                "description": "Matrix Exchange format",
                "binary": False,
            },
            # Tabular data formats
            ".csv": {
                "category": "tabular",
                "type": "delimited_data",
                "description": "Comma-separated values",
                "binary": False,
            },
            ".tsv": {
                "category": "tabular",
                "type": "delimited_data",
                "description": "Tab-separated values",
                "binary": False,
            },
            ".txt": {
                "category": "tabular",
                "type": "delimited_data",
                "description": "Plain text data",
                "binary": False,
            },
            ".xlsx": {
                "category": "tabular",
                "type": "spreadsheet_data",
                "description": "Excel spreadsheet",
                "binary": True,
            },
            ".xls": {
                "category": "tabular",
                "type": "spreadsheet_data",
                "description": "Excel spreadsheet (legacy)",
                "binary": True,
            },
            # Configuration and metadata
            ".json": {
                "category": "metadata",
                "type": "structured_data",
                "description": "JSON metadata",
                "binary": False,
            },
            ".yaml": {
                "category": "metadata",
                "type": "structured_data",
                "description": "YAML configuration",
                "binary": False,
            },
            ".yml": {
                "category": "metadata",
                "type": "structured_data",
                "description": "YAML configuration",
                "binary": False,
            },
            ".xml": {
                "category": "metadata",
                "type": "structured_data",
                "description": "XML data",
                "binary": False,
            },
            # Code and scripts
            ".py": {
                "category": "code",
                "type": "python_script",
                "description": "Python script",
                "binary": False,
            },
            ".r": {
                "category": "code",
                "type": "r_script",
                "description": "R script",
                "binary": False,
            },
            ".sh": {
                "category": "code",
                "type": "shell_script",
                "description": "Shell script",
                "binary": False,
            },
            ".bash": {
                "category": "code",
                "type": "shell_script",
                "description": "Bash script",
                "binary": False,
            },
            # Documentation
            ".md": {
                "category": "documentation",
                "type": "markdown",
                "description": "Markdown document",
                "binary": False,
            },
            ".rst": {
                "category": "documentation",
                "type": "restructured_text",
                "description": "reStructuredText document",
                "binary": False,
            },
            # Archives
            ".gz": {
                "category": "archive",
                "type": "compressed",
                "description": "Gzip compressed file",
                "binary": True,
            },
            ".zip": {
                "category": "archive",
                "type": "compressed",
                "description": "ZIP archive",
                "binary": True,
            },
            ".tar": {
                "category": "archive",
                "type": "compressed",
                "description": "TAR archive",
                "binary": True,
            },
        }

        ext = file_path.suffix.lower()

        # Handle compound extensions like .csv.gz
        if file_path.name.endswith(".gz"):
            # Check the extension before .gz
            name_without_gz = file_path.name[:-3]  # Remove .gz
            inner_ext = Path(name_without_gz).suffix.lower()
            if inner_ext in extension_map:
                info = extension_map[inner_ext].copy()
                info["compressed"] = True
                info["description"] += " (gzip compressed)"
                return info
            ext = ".gz"

        # Direct extension match
        if ext in extension_map:
            return extension_map[ext].copy()

        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith("text/"):
                return {
                    "category": "text",
                    "type": "plain_text",
                    "description": f"Text file ({mime_type})",
                    "binary": False,
                }
            elif mime_type.startswith("image/"):
                return {
                    "category": "image",
                    "type": "image_file",
                    "description": f"Image file ({mime_type})",
                    "binary": True,
                }

        # Unknown file type
        return {
            "category": "unknown",
            "type": "unknown",
            "description": f'Unknown file type ({ext or "no extension"})',
            "binary": True,  # Assume binary for safety
        }

    def locate_file(self, filename: str) -> Dict[str, Any]:
        """
        Locate file with comprehensive search and validation.

        Args:
            filename: Filename or path to search for

        Returns:
            Dictionary with file location and metadata
        """
        file_path = Path(filename)

        # If it's an absolute path, check directly
        if file_path.is_absolute():
            if file_path.exists():
                if file_path.is_file():
                    file_info = self.detect_file_type(file_path)
                    return {
                        "found": True,
                        "path": file_path,
                        "relative_to_workspace": (
                            file_path.relative_to(self.workspace_path)
                            if file_path.is_relative_to(self.workspace_path)
                            else None
                        ),
                        "size_bytes": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                        "readable": os.access(file_path, os.R_OK),
                        **file_info,
                    }
                else:
                    return {
                        "found": False,
                        "error": f"Path exists but is not a file: {file_path}",
                    }
            else:
                return {"found": False, "error": f"File not found: {file_path}"}

        # For relative paths, search in workspace directories
        search_paths = [
            self.workspace_path / filename,
            self.workspace_path / "data" / filename,
            self.data_manager.data_dir / filename,
            self.data_manager.workspace_path / "plots" / filename,
            self.data_manager.exports_dir / filename,
            self.data_manager.cache_dir / filename,
            Path.cwd() / filename,  # Current working directory
        ]

        # Remove duplicates while preserving order
        unique_search_paths = []
        seen = set()
        for path in search_paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_search_paths.append(path)

        for search_path in unique_search_paths:
            if search_path.exists() and search_path.is_file():
                try:
                    file_info = self.detect_file_type(search_path)
                    return {
                        "found": True,
                        "path": search_path.resolve(),
                        "relative_to_workspace": (
                            search_path.relative_to(self.workspace_path)
                            if search_path.is_relative_to(self.workspace_path)
                            else None
                        ),
                        "size_bytes": search_path.stat().st_size,
                        "modified": datetime.fromtimestamp(search_path.stat().st_mtime),
                        "readable": os.access(search_path, os.R_OK),
                        "searched_paths": [str(p) for p in unique_search_paths],
                        **file_info,
                    }
                except (OSError, PermissionError):
                    continue

        return {
            "found": False,
            "error": f"File '{filename}' not found in any search location",
            "searched_paths": [str(p) for p in unique_search_paths],
        }

    def load_data_file(self, filename: str) -> Dict[str, Any]:
        """
        Smart data loading into DataManagerV2 based on file type.

        Args:
            filename: File to load

        Returns:
            Dictionary with loading results and metadata
        """
        # First, locate the file
        file_info = self.locate_file(filename)

        if not file_info["found"]:
            return {
                "success": False,
                "error": file_info["error"],
                "searched_paths": file_info.get("searched_paths", []),
            }

        file_path = file_info["path"]
        file_type = file_info["type"]

        # Check if file is readable
        if not file_info.get("readable", True):
            return {
                "success": False,
                "error": f"Permission denied: Cannot read {file_path}",
            }

        # Generate modality name from filename
        modality_name = file_path.stem  # Filename without extension

        try:
            # Check if this modality already exists
            if modality_name in self.data_manager.list_modalities():
                # Generate unique name
                counter = 1
                original_name = modality_name
                while modality_name in self.data_manager.list_modalities():
                    modality_name = f"{original_name}_{counter}"
                    counter += 1

            # Load based on file type
            if file_type in [
                "single_cell_data",
                "multimodal_data",
                "genomics_data",
                "hdf5_data",
            ]:
                # Use DataManager's load_modality method for bioinformatics formats
                # Try to auto-detect if it's single-cell or bulk based on file
                adapter_name = "transcriptomics_single_cell"  # Default assumption

                adata = self.data_manager.load_modality(
                    name=modality_name,
                    source=str(file_path),
                    adapter=adapter_name,
                    validate=False,  # Skip validation for now to be more permissive
                )

                return {
                    "success": True,
                    "modality_name": modality_name,
                    "file_path": str(file_path),
                    "file_type": file_info["description"],
                    "data_shape": (adata.n_obs, adata.n_vars),
                    "size_bytes": file_info["size_bytes"],
                    "message": f"Data loaded successfully as modality '{modality_name}'",
                }

            elif file_type in ["delimited_data", "spreadsheet_data"]:
                # For tabular data, load as DataFrame and convert using transcriptomics adapter
                try:
                    if file_path.suffix.lower() in [".csv"]:
                        import pandas as pd

                        df = pd.read_csv(file_path)
                    elif file_path.suffix.lower() in [".tsv", ".txt"]:
                        import pandas as pd

                        df = pd.read_csv(file_path, sep="\t")
                    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                        import pandas as pd

                        df = pd.read_excel(file_path)
                    else:
                        return {
                            "success": False,
                            "error": f"Unsupported tabular format: {file_path.suffix}",
                        }

                    # Use transcriptomics adapter for tabular data (genes x samples or samples x genes)
                    adapter_name = (
                        "transcriptomics_bulk"  # Bulk is more generic for tabular data
                    )

                    adata = self.data_manager.load_modality(
                        name=modality_name,
                        source=df,
                        adapter=adapter_name,
                        validate=False,  # Skip validation to be more permissive
                    )

                    return {
                        "success": True,
                        "modality_name": modality_name,
                        "file_path": str(file_path),
                        "file_type": file_info["description"],
                        "data_shape": (adata.n_obs, adata.n_vars),
                        "size_bytes": file_info["size_bytes"],
                        "message": f"Tabular data loaded successfully as modality '{modality_name}'",
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to load tabular data: {str(e)}",
                    }

            else:
                return {
                    "success": False,
                    "error": f"File type '{file_info['description']}' is not a supported data format for loading into workspace",
                    "suggestion": "Use '/read' for text files or ensure file is in a supported bioinformatics format (.h5ad, .csv, .tsv, etc.)",
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load file: {str(e)}",
                "file_path": str(file_path),
                "file_type": file_info["description"],
            }

    def load_quantification_directory(
        self, directory_path: str, tool_type: str
    ) -> Dict[str, Any]:
        """
        Load Kallisto or Salmon quantification files from a directory.

        This method merges per-sample quantification files, creates a proper AnnData
        object with correct orientation (samples × genes), and stores it in the
        DataManagerV2 system.

        Args:
            directory_path: Path to directory containing per-sample subdirectories
                          with quantification files (abundance.tsv for Kallisto,
                          quant.sf for Salmon)
            tool_type: Quantification tool type ('kallisto' or 'salmon')

        Returns:
            Dictionary with loading results:
                - success: bool
                - modality_name: str (if successful)
                - file_path: str
                - data_shape: tuple (if successful)
                - message: str
                - error: str (if failed)
        """
        try:
            # Convert to Path object
            dir_path = Path(directory_path)

            # Validate directory exists
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}",
                }

            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}",
                }

            # Generate modality name from directory name
            modality_name = dir_path.name
            if modality_name in self.data_manager.list_modalities():
                # Generate unique name
                counter = 1
                original_name = modality_name
                while modality_name in self.data_manager.list_modalities():
                    modality_name = f"{original_name}_{counter}"
                    counter += 1

            # Step 1: Merge quantification files using BulkRNASeqService
            from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

            bulk_service = BulkRNASeqService()

            logger.info(
                f"Merging {tool_type} quantification files from: {directory_path}"
            )

            df, metadata = bulk_service.load_from_quantification_files(
                quantification_dir=dir_path,
                tool=tool_type,
            )

            logger.info(
                f"Successfully merged {metadata['n_samples']} samples × {metadata['n_genes']} genes"
            )

            # Step 2: Create AnnData using TranscriptomicsAdapter
            from lobster.core.adapters.transcriptomics_adapter import (
                TranscriptomicsAdapter,
            )

            adapter = TranscriptomicsAdapter(data_type="bulk")
            adata = adapter.from_quantification_dataframe(
                df=df,
                data_type="bulk_rnaseq",
                metadata=metadata,
            )

            logger.info(
                f"Created AnnData: {adata.n_obs} samples × {adata.n_vars} genes"
            )

            # Step 3: Store in DataManagerV2
            self.data_manager.modalities[modality_name] = adata

            # Add quantification metadata to AnnData.uns
            adata.uns["quantification_metadata"] = {
                "tool": tool_type,
                "n_samples": metadata["n_samples"],
                "n_genes": metadata["n_genes"],
                "source_directory": str(directory_path),
                "data_type": "bulk_rnaseq",
            }

            # Step 4: Log operation for provenance
            self.data_manager.log_tool_usage(
                tool_name="load_quantification_directory",
                parameters={
                    "directory_path": str(directory_path),
                    "tool_type": tool_type,
                },
                description=f"Loaded {tool_type} quantification files from {directory_path}",
            )

            # Return formatted result
            return {
                "success": True,
                "modality_name": modality_name,
                "file_path": str(directory_path),
                "tool_type": tool_type,
                "data_shape": (adata.n_obs, adata.n_vars),
                "n_samples": metadata["n_samples"],
                "n_genes": metadata["n_genes"],
                "message": f"{tool_type.title()} quantification files loaded successfully as modality '{modality_name}'",
            }

        except Exception as e:
            logger.error(f"Failed to load quantification directory: {e}")
            return {
                "success": False,
                "error": f"Failed to load {tool_type} quantification files: {str(e)}",
                "directory_path": str(directory_path),
                "tool_type": tool_type,
            }

    def extract_and_load_archive(self, filename: str) -> Dict[str, Any]:
        """
        Extract and load local TAR/ZIP archive intelligently.

        Uses shared archive_utils for:
        - Secure extraction (path traversal protection)
        - Content type detection
        - Format-specific loading strategies

        Args:
            filename: Path to archive file (TAR, TAR.GZ, ZIP)

        Returns:
            Dictionary with loading results and metadata
        """
        # 1. Locate archive
        file_info = self.locate_file(filename)
        if not file_info["found"]:
            return {"success": False, "error": file_info["error"]}

        archive_path = file_info["path"]

        # 2. Inspect manifest (fast, no extraction)
        inspector = ArchiveInspector()
        manifest = inspector.inspect_manifest(archive_path)
        content_type = inspector.detect_content_type_from_manifest(manifest)

        logger.info(
            f"Archive inspection: {manifest['file_count']} files, "
            f"detected as {content_type.value}"
        )

        # 3. Extract safely with security checks
        extractor = ArchiveExtractor()
        try:
            extract_dir = extractor.extract_to_temp(
                archive_path=archive_path, prefix=f"lobster_local_{archive_path.stem}_"
            )

            # 4. Route based on detected content type
            if content_type == ArchiveContentType.KALLISTO_QUANT:
                result = self.load_quantification_directory(str(extract_dir), "kallisto")

            elif content_type == ArchiveContentType.SALMON_QUANT:
                result = self.load_quantification_directory(str(extract_dir), "salmon")

            elif content_type == ArchiveContentType.GEO_RAW:
                result = self._load_geo_raw_directory(extract_dir, archive_path.stem)

            elif content_type == ArchiveContentType.TEN_X_MTX:
                result = self._load_10x_from_directory(extract_dir, archive_path.stem)

            elif content_type == ArchiveContentType.GENERIC_EXPRESSION:
                result = self._load_generic_expression_from_directory(
                    extract_dir, archive_path.stem
                )

            else:
                result = {
                    "success": False,
                    "error": f"Unknown archive content type: {content_type.value}",
                    "manifest": manifest,
                    "suggestion": "Extract manually and load files individually",
                }

            return result

        finally:
            # Cleanup temporary extraction directory
            extractor.cleanup()

    def _load_geo_raw_directory(
        self, directory: Path, modality_base: str
    ) -> Dict[str, Any]:
        """Load GEO RAW files (GSM*.txt.gz) from extracted directory."""
        from lobster.tools.concatenation_service import ConcatenationService

        # Find GEO sample files
        geo_files = []
        for file_path in directory.rglob("GSM*"):
            if file_path.is_file() and any(
                ext in file_path.name for ext in [".txt", ".txt.gz", ".cel", ".CEL"]
            ):
                geo_files.append(file_path)

        if not geo_files:
            return {
                "success": False,
                "error": "No GEO sample files (GSM*.txt*) found in archive",
            }

        logger.info(f"Found {len(geo_files)} GEO sample files")

        try:
            # Use ConcatenationService for merging
            concat_service = ConcatenationService()
            merged_adata, stats = concat_service.concatenate_samples(
                file_paths=[str(f) for f in geo_files],
                axis="obs",  # Samples as observations
            )

            modality_name = f"{modality_base}_merged"
            self.data_manager.modalities[modality_name] = merged_adata

            self.data_manager.log_tool_usage(
                tool_name="load_geo_raw_archive",
                parameters={"n_samples": len(geo_files)},
                description=f"Loaded {len(geo_files)} GEO samples from {modality_base}",
            )

            return {
                "success": True,
                "modality_name": modality_name,
                "n_samples": len(geo_files),
                "data_shape": (merged_adata.n_obs, merged_adata.n_vars),
                "message": f"Successfully loaded {len(geo_files)} GEO RAW samples",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to merge GEO RAW files: {str(e)}",
            }

    def _load_10x_from_directory(
        self, directory: Path, modality_base: str
    ) -> Dict[str, Any]:
        """Load 10X Genomics MEX format from extracted directory."""
        try:
            import scanpy as sc

            # Find 10X directory (may be nested)
            mtx_dirs = list(directory.rglob("matrix.mtx*"))
            if not mtx_dirs:
                return {"success": False, "error": "No matrix.mtx file found"}

            # Use first valid 10X directory
            mtx_dir = mtx_dirs[0].parent
            logger.info(f"Loading 10X data from {mtx_dir}")

            # Load with scanpy
            adata = sc.read_10x_mtx(mtx_dir)

            # Validate AnnData before storing
            if adata.n_obs == 0 or adata.n_vars == 0:
                return {
                    "success": False,
                    "error": f"Invalid data shape: {adata.n_obs} obs × {adata.n_vars} vars (empty dataset)",
                }

            modality_name = f"{modality_base}_10x"
            self.data_manager.modalities[modality_name] = adata

            self.data_manager.log_tool_usage(
                tool_name="load_10x_archive",
                parameters={"source_dir": str(mtx_dir)},
                description=f"Loaded 10X Genomics data from {modality_base}",
            )

            return {
                "success": True,
                "modality_name": modality_name,
                "data_shape": (adata.n_obs, adata.n_vars),
                "message": "Successfully loaded 10X Genomics data",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load 10X data: {str(e)}",
            }

    def _load_generic_expression_from_directory(
        self, directory: Path, modality_base: str
    ) -> Dict[str, Any]:
        """Load generic expression matrix from extracted directory."""
        # Find largest file (likely the main expression matrix)
        expression_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and any(
                ext in file_path.suffix for ext in [".csv", ".tsv", ".txt", ".h5ad"]
            ):
                if file_path.stat().st_size > 100000:  # > 100KB
                    expression_files.append(file_path)

        if not expression_files:
            return {"success": False, "error": "No expression files found"}

        # Sort by size, try largest first
        expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

        for file_path in expression_files[:3]:  # Try top 3
            try:
                logger.info(f"Attempting to load {file_path.name}")

                # Try loading as standard data file
                load_result = self.load_data_file(str(file_path))
                if load_result["success"]:
                    return load_result

            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                continue

        return {
            "success": False,
            "error": "Could not parse any expression files from archive",
        }

    def inspect_archive(self, filename: str) -> Dict[str, Any]:
        """
        Inspect archive without loading - enables selective loading workflow.

        This method detects nested archives (e.g., 10X samples in GEO RAW.tar)
        and caches the extraction for selective loading by pattern/condition.

        Args:
            filename: Archive file to inspect

        Returns:
            Dictionary with inspection results:
            - success: bool
            - type: "nested_archive" or "regular_archive"
            - nested_info: NestedArchiveInfo (if nested)
            - cache_id: str (if nested, for subsequent loading)
            - content_type: ArchiveContentType (if regular)
            - manifest: Dict (if regular)
            - error: str (if failed)
        """
        try:
            # 1. Locate archive file
            file_info = self.locate_file(filename)
            if not file_info["found"]:
                return {"success": False, "error": file_info["error"]}

            archive_path = file_info["path"]

            # 2. Fast manifest inspection (no extraction yet)
            inspector = ArchiveInspector()
            manifest = inspector.inspect_manifest(archive_path)

            # 3. Check for nested archives
            nested_info = inspector.detect_nested_archives(
                manifest, str(archive_path)
            )

            if nested_info:
                # This is a nested archive - extract and cache for selective loading
                logger.info(
                    f"Detected nested archive with {nested_info.total_count} samples"
                )

                extractor = ArchiveExtractor()
                extract_dir = extractor.extract_to_temp(
                    archive_path, prefix=f"lobster_nested_{archive_path.stem}_"
                )

                # Cache extraction for selective loading
                cache_manager = ExtractionCacheManager(self.workspace_path)
                cache_id = cache_manager.cache_extraction(
                    archive_path, extract_dir, nested_info
                )

                # Cleanup temporary extractor (cache manager took ownership)
                extractor.temp_dirs.clear()

                return {
                    "success": True,
                    "type": "nested_archive",
                    "nested_info": nested_info,
                    "cache_id": cache_id,
                    "message": f"Inspected nested archive: {nested_info.total_count} samples",
                }

            else:
                # Regular archive - can auto-load as before
                content_type = inspector.detect_content_type_from_manifest(manifest)
                return {
                    "success": True,
                    "type": "regular_archive",
                    "content_type": content_type.value,
                    "manifest": manifest,
                    "message": f"Detected {content_type.value} archive",
                }

        except Exception as e:
            logger.error(f"Failed to inspect archive: {e}")
            return {
                "success": False,
                "error": f"Archive inspection failed: {str(e)}",
            }

    @staticmethod
    def _clean_archive_name(file_path: Path) -> str:
        """
        Remove compound archive extensions from filename.

        Handles: .tar.gz, .tar.bz2, .tgz, .tar

        Args:
            file_path: Path to archive file

        Returns:
            Clean filename without archive extensions

        Example:
            GSM4710689_PDAC_TISSUE_1.tar.gz -> GSM4710689_PDAC_TISSUE_1
        """
        filename = file_path.name

        # Strip compound extensions in order of specificity
        for ext in ['.tar.gz', '.tar.bz2', '.tgz', '.tar']:
            if filename.endswith(ext):
                return filename[:-len(ext)]

        # Fallback to stem for other extensions
        return file_path.stem

    def load_from_cache(
        self,
        cache_id: str,
        pattern: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load specific samples from cached extraction by pattern.

        Args:
            cache_id: Cached extraction identifier
            pattern: GSM ID, condition name, or glob pattern
            limit: Maximum samples to load (None = no limit)

        Returns:
            Dictionary with loading results:
            - success: bool
            - loaded_count: int
            - modalities: List[str] (loaded modality names)
            - failed: List[str] (failed files)
            - message: str
            - error: str (if failed)
        """
        try:
            # 1. Get matching files from cache
            cache_manager = ExtractionCacheManager(self.workspace_path)
            matching_files = cache_manager.load_from_cache(cache_id, pattern, limit)

            if not matching_files:
                return {
                    "success": False,
                    "error": f"No files matched pattern '{pattern}'",
                    "suggestion": "Try /archive list to see available samples",
                }

            logger.info(f"Loading {len(matching_files)} samples matching '{pattern}'")

            # 2. Load each matching nested archive
            results = []
            failed = []

            for nested_archive in matching_files:
                try:
                    # Clean archive name (removes .tar.gz, .tar.bz2, etc.)
                    clean_name = self._clean_archive_name(nested_archive)

                    # Extract nested archive
                    extractor = ArchiveExtractor()
                    nested_extract = extractor.extract_to_temp(
                        nested_archive, prefix=f"lobster_sample_{clean_name}_"
                    )

                    # Detect content type and load appropriately
                    content_detector = ContentDetector()
                    content_type = content_detector.detect_content_type(nested_extract)

                    result = None
                    if content_type == ArchiveContentType.TEN_X_MTX:
                        result = self._load_10x_from_directory(
                            nested_extract, clean_name
                        )
                    elif content_type == ArchiveContentType.KALLISTO_QUANT:
                        result = self.load_quantification_directory(
                            str(nested_extract), "kallisto"
                        )
                    elif content_type == ArchiveContentType.SALMON_QUANT:
                        result = self.load_quantification_directory(
                            str(nested_extract), "salmon"
                        )
                    elif content_type == ArchiveContentType.GENERIC_EXPRESSION:
                        result = self._load_generic_expression_from_directory(
                            nested_extract, clean_name
                        )
                    else:
                        logger.warning(
                            f"Unknown content type in {nested_archive.name}: {content_type}"
                        )
                        failed.append(nested_archive.name)

                    if result and result.get("success"):
                        results.append(result)
                    else:
                        failed.append(nested_archive.name)

                    # Cleanup temp extraction
                    extractor.cleanup()

                except Exception as e:
                    logger.error(f"Failed to load {nested_archive.name}: {e}")
                    failed.append(nested_archive.name)

            # 3. Return summary
            loaded_modalities = [r["modality_name"] for r in results]

            # 4. Auto-concatenate if multiple samples loaded
            merged_modality = None
            if len(results) > 1:
                try:
                    from lobster.tools.concatenation_service import ConcatenationService

                    concat_service = ConcatenationService(self.data_manager)

                    # Generate merged modality name from pattern
                    # Extract base pattern (e.g., "TISSUE" from "PDAC_TISSUE" or "TISSUE")
                    pattern_parts = pattern.split('_')
                    # Use last part if multiple underscores (e.g., "TISSUE" from "PDAC_TISSUE")
                    pattern_base = pattern_parts[-1] if len(pattern_parts) > 1 else pattern
                    merged_modality = f"{cache_id}_{pattern_base}_merged"

                    logger.info(
                        f"Auto-concatenating {len(results)} samples into '{merged_modality}'"
                    )

                    # Concatenate using existing service
                    merged_adata, stats, ir = concat_service.concatenate_from_modalities(
                        modality_names=loaded_modalities,
                        output_name=merged_modality,
                        batch_key="sample_id",
                        use_intersecting_genes_only=True,
                    )

                    # Store merged result
                    self.data_manager.modalities[merged_modality] = merged_adata

                    # Log provenance for notebook export
                    self.data_manager.log_tool_usage(
                        tool_name="auto_concatenate_nested_archives",
                        parameters={
                            "pattern": pattern,
                            "n_samples": len(results),
                            "cache_id": cache_id,
                        },
                        description=f"Auto-concatenated {len(results)} samples matching '{pattern}'",
                        ir=ir,
                    )

                    logger.info(
                        f"✓ Merged {len(results)} samples: {merged_adata.n_obs} obs × {merged_adata.n_vars} vars"
                    )

                except Exception as e:
                    logger.warning(
                        f"Auto-concatenation failed: {e}, returning individual samples"
                    )
                    # Continue with individual samples if concatenation fails
                    merged_modality = None

            # Return summary with merged modality if available
            result_dict = {
                "success": len(results) > 0,
                "loaded_count": len(results),
                "modalities": loaded_modalities,
                "failed": failed,
            }

            if merged_modality:
                result_dict["merged_modality"] = merged_modality
                result_dict["message"] = (
                    f"Loaded and merged {len(results)} samples into '{merged_modality}'"
                )
            else:
                result_dict["message"] = (
                    f"Loaded {len(results)} of {len(matching_files)} samples matching '{pattern}'"
                )

            return result_dict

        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return {
                "success": False,
                "error": f"Failed to load samples: {str(e)}",
            }

    # Workspace operations
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in the workspace."""
        files = []
        for path in self.workspace_path.glob(pattern):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "size": path.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).isoformat(),
                    }
                )
        return files

    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from the workspace or absolute path.

        Args:
            filename: Either a relative filename (searched in workspace) or absolute path

        Returns:
            File content as string, or None if not found
        """
        file_path = Path(filename)

        # If it's an absolute path, try to read directly
        if file_path.is_absolute():
            if file_path.exists() and file_path.is_file():
                try:
                    return file_path.read_text()
                except Exception as e:
                    return f"Error reading file {file_path}: {e}"
            else:
                return f"File not found: {file_path}"

        # For relative paths, search in workspace and data directories
        search_paths = [
            self.workspace_path / filename,
            self.data_manager.data_dir / filename,
            self.data_manager.workspace_path / "plots" / filename,
            self.data_manager.exports_dir / filename,
            self.data_manager.cache_dir / filename,
        ]

        for search_path in search_paths:
            if search_path.exists() and search_path.is_file():
                try:
                    return search_path.read_text()
                except Exception as e:
                    return f"Error reading file: {e}"

        return f"File not found in workspace: {filename}"

    def write_file(self, filename: str, content: str) -> bool:
        """Write a file to the workspace."""
        try:
            file_path = self.workspace_path / filename
            file_path.write_text(content)
            return True
        except Exception:
            return False

    # State management
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        history = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"

            history.append(
                {
                    "role": role,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                }
            )
        return history

    def _check_redis_health(self) -> Dict[str, Any]:
        """
        Check Redis connection health for rate limiting.

        Returns:
            Dictionary with Redis status, metrics, and availability info
        """
        try:
            from lobster.tools.rate_limiter import get_redis_client

            client = get_redis_client()

            if client is None:
                return {
                    "status": "unavailable",
                    "message": "Redis connection failed - rate limiting disabled",
                    "critical": True,
                }

            try:
                # Get Redis server info
                info = client.info("stats")
                memory_info = client.info("memory")

                return {
                    "status": "healthy",
                    "connected": True,
                    "total_commands": info.get("total_commands_processed", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "used_memory_human": memory_info.get(
                        "used_memory_human", "unknown"
                    ),
                    "critical": False,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Redis health check failed: {str(e)}",
                    "connected": False,
                    "critical": True,
                }

        except ImportError:
            return {
                "status": "not_configured",
                "message": "Redis rate limiter not imported",
                "critical": False,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error checking Redis: {str(e)}",
                "critical": True,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current client status."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "has_data": self.data_manager.has_data(),
            "data_summary": (
                self.data_manager.get_data_summary()
                if self.data_manager.has_data()
                else None
            ),
            "workspace": str(self.workspace_path),
            "reasoning_enabled": self.enable_reasoning,
            "callbacks_count": len(self.callbacks),
            "redis_health": self._check_redis_health(),
        }

    def reset(self):
        """Reset the conversation state."""
        self.messages = []
        self.metadata["reset_at"] = datetime.now().isoformat()

    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """Export the current session data."""
        if self.data_manager.has_data():
            export_path = self.data_manager.create_data_package(
                output_dir=str(self.data_manager.exports_dir)
            )
            return Path(export_path)

        export_path = (
            export_path or self.workspace_path / f"session_{self.session_id}.json"
        )

        session_data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "conversation": self.get_conversation_history(),
            "status": self.get_status(),
            "workspace_status": self.data_manager.get_workspace_status(),
            "exported_at": datetime.now().isoformat(),
        }

        with open(export_path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        return export_path
