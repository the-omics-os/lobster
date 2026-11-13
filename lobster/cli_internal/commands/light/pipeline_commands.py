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
from lobster.i18n import h

logger = logging.getLogger(__name__)


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


EXPORT_USAGE = "pipeline export [name] [description] [--replay] [--force]"


def parse_export_args(
    args: list[str],
) -> tuple[Optional[str], Optional[str], str, bool]:
    """Parse ``pipeline export`` arguments into (name, description, mode, force).

    Raises:
        ValueError: If an argument looks like an option but is not one, so a
            mistyped flag is never used as the notebook name.
    """
    flags = {"--replay", "--force"}
    unknown = [a for a in args if a.startswith("-") and a != "-" and a not in flags]
    if unknown:
        raise ValueError(f"Unknown option: {', '.join(unknown)}. Usage: {EXPORT_USAGE}")
    mode = "replay" if "--replay" in args else "successful"
    force = "--force" in args
    rest = [a for a in args if a not in flags]
    name = rest[0] if rest else None
    description = " ".join(rest[1:]) if len(rest) > 1 else None
    return name, description, mode, force


def pipeline_export(
    client: "AgentClient",
    output: OutputAdapter,
    name: Optional[str] = None,
    description: Optional[str] = None,
    mode: str = "successful",
    force: bool = False,
) -> Optional[str]:
    """
    Export current session as Jupyter notebook.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        name: Notebook name (without extension). If None, prompts interactively.
        description: Optional notebook description. If None, prompts interactively.
        mode: "successful" (curated, the default) or "replay" (every recorded
            step marked for replay, in order).
        force: Replace an existing notebook of the same name. Without it the
            export refuses, because the file may hold the user's own edits.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Notebook export not available for cloud client"),
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
                        h("Notebook export not available - update Lobster"),
                        level="error",
                    )
                ],
            )
            return "Notebook export not available"

        _render_blocks(
            output, [section_block(title=h("Export Session as Jupyter Notebook"))]
        )

        if name is None:
            name = output.prompt(
                h("Notebook name (no extension)"), default="analysis_workflow"
            )
        if not name:
            _render_blocks(output, [alert_block(h("Name required"), level="error")])
            return "Export cancelled - no name provided"

        existing = Path(client.data_manager.workspace_path) / "notebooks"
        existing = existing / f"{name}.ipynb"
        if existing.exists() and not force:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Notebook already exists: {path}").format(path=existing),
                        level="error",
                    ),
                    hint_block(h("Choose another name, or add --force to replace it.")),
                ],
            )
            return f"Export failed: notebook already exists: {existing}"

        if description is None:
            description = output.prompt(h("Description (optional)"), default="")

        _render_blocks(
            output,
            [
                section_block(
                    body=h("Exporting notebook ({mode} mode)...").format(mode=mode)
                )
            ],
        )
        path = client.data_manager.export_notebook(
            name, description, filter_strategy=mode
        )

        _render_blocks(
            output,
            [
                alert_block(
                    h("Notebook exported: {path}").format(path=path), level="success"
                ),
                list_block(
                    [
                        f"{h('Review:')} jupyter notebook {path}",
                        f"{h('Commit:')} git add {path} && git commit -m 'Add {name}'",
                        f"{h('Run:')} /pipeline run {path.name} <modality>",
                    ],
                    title=h("Next Steps"),
                    ordered=True,
                ),
            ],
        )

        return f"Exported notebook: {path}"

    except ValueError as e:
        _render_blocks(
            output,
            [alert_block(h("Export failed: {error}").format(error=e), level="error")],
        )
        return f"Export failed: {e}"
    except Exception as e:
        _render_blocks(
            output,
            [alert_block(h("Export error: {error}").format(error=e), level="error")],
        )
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
                        h("Notebook listing not available for cloud client"),
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
                        h("No notebooks found in workspace notebooks directory"),
                        level="warning",
                    ),
                    hint_block(h("Export one with: /pipeline export")),
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
                    title=h("Available Notebooks"),
                    columns=[
                        {"name": h("Name")},
                        {"name": h("Steps")},
                        {"name": h("Created By")},
                        {"name": h("Created")},
                        {"name": h("Size")},
                    ],
                    rows=rows,
                )
            ],
        )
        return f"Found {len(notebooks)} notebooks"

    except Exception as e:
        _render_blocks(
            output,
            [alert_block(h("List error: {error}").format(error=e), level="error")],
        )
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
                        h("Notebook execution not available for cloud client"),
                        level="error",
                    )
                ],
            )
            return "Notebook execution only available for local client"

        if notebook_name is None:
            notebooks = client.data_manager.list_notebooks()
            if not notebooks:
                _render_blocks(
                    output, [alert_block(h("No notebooks available"), level="error")]
                )
                return "No notebooks available"

            _render_blocks(
                output,
                [
                    list_block(
                        [f"{nb['name']} ({nb['n_steps']} steps)" for nb in notebooks],
                        title=h("Available Notebooks"),
                        ordered=True,
                    )
                ],
            )

            selection = output.prompt(h("Select notebook number"), default="1")
            try:
                idx = int(selection) - 1
                notebook_name = notebooks[idx]["filename"]
            except (ValueError, IndexError):
                _render_blocks(
                    output, [alert_block(h("Invalid selection"), level="error")]
                )
                return "Invalid notebook selection"

        if input_modality is None:
            modalities = client.data_manager.list_modalities()
            if not modalities:
                _render_blocks(
                    output,
                    [alert_block(h("No data loaded. Use /read first."), level="error")],
                )
                return "No data loaded"

            _render_blocks(
                output,
                [
                    list_block(
                        [
                            h("{name} ({obs} obs x {vars} vars)").format(
                                name=mod,
                                obs=client.data_manager.modalities[mod].n_obs,
                                vars=client.data_manager.modalities[mod].n_vars,
                            )
                            for mod in modalities
                        ],
                        title=h("Available Modalities"),
                        ordered=True,
                    )
                ],
            )

            selection = output.prompt(h("Select modality number"), default="1")
            try:
                idx = int(selection) - 1
                input_modality = modalities[idx]
            except (ValueError, IndexError):
                _render_blocks(
                    output, [alert_block(h("Invalid selection"), level="error")]
                )
                return "Invalid modality selection"

        _render_blocks(output, [section_block(body=h("Running validation..."))])
        dry_result = client.data_manager.run_notebook(
            notebook_name, input_modality, dry_run=True
        )

        validation = dry_result.get("validation")
        if validation and hasattr(validation, "has_errors") and validation.has_errors:
            _render_blocks(
                output,
                [
                    alert_block(h("Validation failed"), level="error"),
                    list_block(list(validation.errors), title=h("Errors")),
                ],
            )
            return "Validation failed"

        blocks: list[OutputBlock] = []
        if (
            validation
            and hasattr(validation, "has_warnings")
            and validation.has_warnings
        ):
            blocks.append(alert_block(h("Warnings"), level="warning"))
            blocks.append(list_block(list(validation.warnings), title=h("Warnings")))

        blocks.append(alert_block(h("Validation passed"), level="success"))
        blocks.append(
            kv_block(
                [
                    (h("Steps to execute"), dry_result["steps_to_execute"]),
                    (
                        h("Estimated time"),
                        f"{dry_result['estimated_duration_minutes']} min",
                    ),
                ],
                title=h("Validation Summary"),
            )
        )
        _render_blocks(output, blocks)

        if not output.confirm(h("Execute notebook?")):
            _render_blocks(output, [section_block(body=h("Cancelled"))])
            return "Execution cancelled"

        _render_blocks(output, [section_block(body=h("Executing notebook..."))])
        result = client.data_manager.run_notebook(notebook_name, input_modality)

        if result["status"] == "success":
            _render_blocks(
                output,
                [
                    alert_block(h("Execution complete"), level="success"),
                    kv_block(
                        [
                            (h("Output"), result["output_notebook"]),
                            (h("Duration"), f"{result['execution_time']:.1f}s"),
                        ],
                        title=h("Execution Result"),
                    ),
                ],
            )
            return f"Notebook executed successfully in {result['execution_time']:.1f}s"

        _render_blocks(
            output,
            [
                alert_block(h("Execution failed"), level="error"),
                kv_block(
                    [
                        (h("Error"), result.get("error", "Unknown")),
                        (h("Partial output"), result.get("output_notebook", "N/A")),
                    ],
                    title=h("Execution Result"),
                ),
            ],
        )
        return f"Execution failed: {result.get('error', 'Unknown')}"

    except FileNotFoundError as e:
        _render_blocks(
            output,
            [alert_block(h("File not found: {error}").format(error=e), level="error")],
        )
        return f"Notebook not found: {e}"
    except Exception as e:
        _render_blocks(
            output,
            [alert_block(h("Execution error: {error}").format(error=e), level="error")],
        )
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
                        h("Notebook info not available for cloud client"),
                        level="error",
                    )
                ],
            )
            return "Notebook info only available for local client"

        notebooks = client.data_manager.list_notebooks()
        if not notebooks:
            _render_blocks(
                output, [alert_block(h("No notebooks found"), level="error")]
            )
            return "No notebooks found"

        _render_blocks(
            output,
            [
                list_block(
                    [nb["name"] for nb in notebooks],
                    title=h("Select Notebook"),
                    ordered=True,
                )
            ],
        )

        selection = output.prompt(h("Selection"), default="1")
        try:
            idx = int(selection) - 1
            nb = notebooks[idx]
        except (ValueError, IndexError):
            _render_blocks(output, [alert_block(h("Invalid selection"), level="error")])
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
                    (h("Created by"), metadata.get("created_by", "unknown")),
                    (h("Date"), metadata.get("created_at", "unknown")),
                    (
                        h("Lobster version"),
                        metadata.get("lobster_version", "unknown"),
                    ),
                    (h("Steps"), nb["n_steps"]),
                    (h("Size"), f"{nb['size_kb']:.1f} KB"),
                ],
                title=h("Notebook Details"),
            ),
        ]
        dependencies = metadata.get("dependencies", {})
        if dependencies:
            blocks.append(
                kv_block(
                    list(dependencies.items()),
                    title=h("Dependencies"),
                    key_label=h("Package"),
                    value_label=h("Version"),
                )
            )
        _render_blocks(output, blocks)

        return f"Notebook info: {nb['name']}"

    except Exception as e:
        _render_blocks(
            output,
            [alert_block(h("Info error: {error}").format(error=e), level="error")],
        )
        logger.exception("Notebook info error")
        return f"Info error: {e}"
