"""
Shared pipeline commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    OutputBlock,
    alert_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)

logger = logging.getLogger(__name__)


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def pipeline_export(
    client: "AgentClient",
    output: OutputAdapter,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[str]:
    """
    Export current session as Jupyter notebook.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        name: Notebook name (without extension). If None, prompts interactively.
        description: Optional notebook description. If None, prompts interactively.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            _render_blocks(
                output,
                [
                    alert_block(
                        "Notebook export not available for cloud client",
                        level="error",
                    )
                ],
            )
            return "Notebook export only available for local client"

        if not hasattr(client.data_manager, "export_notebook"):
            _render_blocks(
                output,
                [
                    alert_block(
                        "Notebook export not available - update Lobster",
                        level="error",
                    )
                ],
            )
            return "Notebook export not available"

        _render_blocks(
            output, [section_block(title="Export Session as Jupyter Notebook")]
        )

        if name is None:
            name = output.prompt(
                "Notebook name (no extension)", default="analysis_workflow"
            )
        if not name:
            _render_blocks(output, [alert_block("Name required", level="error")])
            return "Export cancelled - no name provided"

        if description is None:
            description = output.prompt("Description (optional)", default="")

        _render_blocks(output, [section_block(body="Exporting notebook...")])
        path = client.data_manager.export_notebook(name, description)

        _render_blocks(
            output,
            [
                alert_block(f"Notebook exported: {path}", level="success"),
                list_block(
                    [
                        f"Review: jupyter notebook {path}",
                        f"Commit: git add {path} && git commit -m 'Add {name}'",
                        f"Run: /pipeline run {path.name} <modality>",
                    ],
                    title="Next Steps",
                    ordered=True,
                ),
            ],
        )

        return f"Exported notebook: {path}"

    except ValueError as e:
        _render_blocks(output, [alert_block(f"Export failed: {e}", level="error")])
        return f"Export failed: {e}"
    except Exception as e:
        _render_blocks(output, [alert_block(f"Export error: {e}", level="error")])
        logger.exception("Notebook export error")
        return f"Export error: {e}"


def pipeline_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List available notebooks.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            _render_blocks(
                output,
                [
                    alert_block(
                        "Notebook listing not available for cloud client",
                        level="error",
                    )
                ],
            )
            return "Notebook listing only available for local client"

        notebooks = client.data_manager.list_notebooks()

        if not notebooks:
            _render_blocks(
                output,
                [
                    alert_block(
                        "No notebooks found in workspace notebooks directory",
                        level="warning",
                    ),
                    hint_block("Export one with: /pipeline export"),
                ],
            )
            return "No notebooks found"

        rows = []

        for nb in notebooks:
            created_date = (
                nb["created_at"].split("T")[0] if nb["created_at"] else "unknown"
            )
            rows.append(
                [
                    nb["name"],
                    str(nb["n_steps"]),
                    nb["created_by"],
                    created_date,
                    f"{nb['size_kb']:.1f} KB",
                ]
            )

        _render_blocks(
            output,
            [
                table_block(
                    title="Available Notebooks",
                    columns=[
                        {"name": "Name"},
                        {"name": "Steps"},
                        {"name": "Created By"},
                        {"name": "Created"},
                        {"name": "Size"},
                    ],
                    rows=rows,
                )
            ],
        )
        return f"Found {len(notebooks)} notebooks"

    except Exception as e:
        _render_blocks(output, [alert_block(f"List error: {e}", level="error")])
        logger.exception("Notebook list error")
        return f"List error: {e}"


def pipeline_run(
    client: "AgentClient",
    output: OutputAdapter,
    notebook_name: Optional[str] = None,
    input_modality: Optional[str] = None,
) -> Optional[str]:
    """
    Run saved notebook with new data.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        notebook_name: Notebook filename. If None, prompts interactively.
        input_modality: Input modality name. If None, prompts interactively.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            _render_blocks(
                output,
                [
                    alert_block(
                        "Notebook execution not available for cloud client",
                        level="error",
                    )
                ],
            )
            return "Notebook execution only available for local client"

        if notebook_name is None:
            notebooks = client.data_manager.list_notebooks()
            if not notebooks:
                _render_blocks(
                    output, [alert_block("No notebooks available", level="error")]
                )
                return "No notebooks available"

            _render_blocks(
                output,
                [
                    list_block(
                        [f"{nb['name']} ({nb['n_steps']} steps)" for nb in notebooks],
                        title="Available Notebooks",
                        ordered=True,
                    )
                ],
            )

            selection = output.prompt("Select notebook number", default="1")
            try:
                idx = int(selection) - 1
                notebook_name = notebooks[idx]["filename"]
            except (ValueError, IndexError):
                _render_blocks(
                    output, [alert_block("Invalid selection", level="error")]
                )
                return "Invalid notebook selection"

        if input_modality is None:
            modalities = client.data_manager.list_modalities()
            if not modalities:
                _render_blocks(
                    output,
                    [alert_block("No data loaded. Use /read first.", level="error")],
                )
                return "No data loaded"

            _render_blocks(
                output,
                [
                    list_block(
                        [
                            f"{mod} ({client.data_manager.modalities[mod].n_obs} obs x {client.data_manager.modalities[mod].n_vars} vars)"
                            for mod in modalities
                        ],
                        title="Available Modalities",
                        ordered=True,
                    )
                ],
            )

            selection = output.prompt("Select modality number", default="1")
            try:
                idx = int(selection) - 1
                input_modality = modalities[idx]
            except (ValueError, IndexError):
                _render_blocks(
                    output, [alert_block("Invalid selection", level="error")]
                )
                return "Invalid modality selection"

        _render_blocks(output, [section_block(body="Running validation...")])
        dry_result = client.data_manager.run_notebook(
            notebook_name, input_modality, dry_run=True
        )

        validation = dry_result.get("validation")
        if validation and hasattr(validation, "has_errors") and validation.has_errors:
            _render_blocks(
                output,
                [
                    alert_block("Validation failed", level="error"),
                    list_block(list(validation.errors), title="Errors"),
                ],
            )
            return "Validation failed"

        blocks: list[OutputBlock] = []
        if (
            validation
            and hasattr(validation, "has_warnings")
            and validation.has_warnings
        ):
            blocks.append(alert_block("Warnings", level="warning"))
            blocks.append(list_block(list(validation.warnings), title="Warnings"))

        blocks.append(alert_block("Validation passed", level="success"))
        blocks.append(
            kv_block(
                [
                    ("Steps to execute", dry_result["steps_to_execute"]),
                    (
                        "Estimated time",
                        f"{dry_result['estimated_duration_minutes']} min",
                    ),
                ],
                title="Validation Summary",
            )
        )
        _render_blocks(output, blocks)

        if not output.confirm("Execute notebook?"):
            _render_blocks(output, [section_block(body="Cancelled")])
            return "Execution cancelled"

        _render_blocks(output, [section_block(body="Executing notebook...")])
        result = client.data_manager.run_notebook(notebook_name, input_modality)

        if result["status"] == "success":
            _render_blocks(
                output,
                [
                    alert_block("Execution complete", level="success"),
                    kv_block(
                        [
                            ("Output", result["output_notebook"]),
                            ("Duration", f"{result['execution_time']:.1f}s"),
                        ],
                        title="Execution Result",
                    ),
                ],
            )
            return f"Notebook executed successfully in {result['execution_time']:.1f}s"

        _render_blocks(
            output,
            [
                alert_block("Execution failed", level="error"),
                kv_block(
                    [
                        ("Error", result.get("error", "Unknown")),
                        ("Partial output", result.get("output_notebook", "N/A")),
                    ],
                    title="Execution Result",
                ),
            ],
        )
        return f"Execution failed: {result.get('error', 'Unknown')}"

    except FileNotFoundError as e:
        _render_blocks(output, [alert_block(f"File not found: {e}", level="error")])
        return f"Notebook not found: {e}"
    except Exception as e:
        _render_blocks(output, [alert_block(f"Execution error: {e}", level="error")])
        logger.exception("Notebook execution error")
        return f"Execution error: {e}"


def pipeline_info(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show notebook details.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            _render_blocks(
                output,
                [
                    alert_block(
                        "Notebook info not available for cloud client",
                        level="error",
                    )
                ],
            )
            return "Notebook info only available for local client"

        notebooks = client.data_manager.list_notebooks()
        if not notebooks:
            _render_blocks(output, [alert_block("No notebooks found", level="error")])
            return "No notebooks found"

        _render_blocks(
            output,
            [
                list_block(
                    [nb["name"] for nb in notebooks],
                    title="Select Notebook",
                    ordered=True,
                )
            ],
        )

        selection = output.prompt("Selection", default="1")
        try:
            idx = int(selection) - 1
            nb = notebooks[idx]
        except (ValueError, IndexError):
            _render_blocks(output, [alert_block("Invalid selection", level="error")])
            return "Invalid selection"

        import nbformat

        nb_path = Path(nb["path"])
        with open(nb_path) as f:
            notebook = nbformat.read(f, as_version=4)

        metadata = notebook.metadata.get("lobster", {})

        blocks: list[OutputBlock] = [
            section_block(title=nb["name"]),
            kv_block(
                [
                    ("Created by", metadata.get("created_by", "unknown")),
                    ("Date", metadata.get("created_at", "unknown")),
                    (
                        "Lobster version",
                        metadata.get("lobster_version", "unknown"),
                    ),
                    ("Steps", nb["n_steps"]),
                    ("Size", f"{nb['size_kb']:.1f} KB"),
                ],
                title="Notebook Details",
            ),
        ]
        dependencies = metadata.get("dependencies", {})
        if dependencies:
            blocks.append(
                kv_block(
                    list(dependencies.items()),
                    title="Dependencies",
                    key_label="Package",
                    value_label="Version",
                )
            )
        _render_blocks(output, blocks)

        return f"Notebook info: {nb['name']}"

    except Exception as e:
        _render_blocks(output, [alert_block(f"Info error: {e}", level="error")])
        logger.exception("Notebook info error")
        return f"Info error: {e}"
