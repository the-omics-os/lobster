"""
Shared custom code execution tool factory.

Creates agent-specific execute_custom_code tools with unified signature
and optional post-processing hooks. Follows workspace_tool.py pattern.

Architecture:
    Factory Pattern: create_execute_custom_code_tool() returns @tool function
    Post-Processor: Optional callback for agent-specific behavior
    Single Source: Eliminates duplication across data_expert and metadata_assistant

Usage:
    # data_expert (no post-processing)
    tool = create_execute_custom_code_tool(data_manager, service, "data_expert")

    # metadata_assistant (with metadata_store persistence)
    tool = create_execute_custom_code_tool(
        data_manager, service, "metadata_assistant",
        post_processor=metadata_store_post_processor
    )

See Also:
    - workspace_tool.py: Established factory pattern for shared tools
    - CustomCodeExecutionService: Underlying execution service
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Optional

from langchain_core.tools import InjectedToolCallId, tool

from lobster.utils.logger import get_logger

# TYPE_CHECKING imports - only evaluated by type checkers, not at runtime
# This prevents circular imports when component_registry loads entry points
if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )

logger = get_logger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

# Post-processor callback signature
# Args: (result, stats, data_manager, workspace_key, modality_name)
# Returns: Optional message to append to response
# Note: Using string annotation to avoid runtime import of DataManagerV2
PostProcessor = Callable[
    [Any, Dict[str, Any], "DataManagerV2", Optional[str], Optional[str]],
    Optional[str],
]


# =============================================================================
# Deferred Help Text (returned on empty code or errors, NOT in tool schema)
# =============================================================================

_EXECUTE_CUSTOM_CODE_HELP = """EXECUTE_CUSTOM_CODE — Detailed Usage Guide

PARAMETER SELECTION (choose ONE):
- modality_name: Load AnnData as `adata` variable
- workspace_key: Load specific JSON/CSV file (token-efficient)
- Neither: Auto-load all workspace files

NAMESPACE AVAILABLE:
- WORKSPACE: Path to workspace directory (pathlib.Path)
- OUTPUT_DIR: Recommended export path (workspace/exports/)
- adata: Loaded modality (if modality_name provided)
- Auto-loaded CSV/JSON files as variables (if load_workspace_files=True)
- publication_queue, download_queue (if they exist in workspace)

RETURNING RESULTS:
Assign to `result` variable: `result = my_computation()`

LARGE OUTPUT HANDLING:
stdout capped at 500 chars. result capped at 8,000 chars.
Save large data to file instead:
    import json
    output_path = OUTPUT_DIR / 'my_results.json'
    with open(output_path, 'w') as f:
        json.dump(large_data, f)
    result = {"saved_to": str(output_path), "count": len(large_data), "preview": large_data[:5]}

DEFENSIVE CODING:
- Use data.get('key', default) instead of data['key']
- Check `if value is not None` before .lower()/.upper()
- Convert numpy types: int(val), float(val), list(arr)
- Check isinstance(data, dict) before dict operations

PERSIST PARAMETER:
- persist=False (default): Ephemeral — logged, but not included in the default notebook export
- persist=True: recorded and eligible for the default notebook export; omitted steps are listed in the notebook.

STATE PERSISTENCE:
Modified adata is saved as NEW modality with _custom suffix.
Example: modality_name="geo_gse12345_clustered" → "geo_gse12345_clustered_custom"

METADATA PERSISTENCE:
Return dict with samples and output_key to persist:
  result = {'samples': [...], 'output_key': 'filtered_samples'}
"""


# =============================================================================
# Factory Function
# =============================================================================


def create_execute_custom_code_tool(
    data_manager: DataManagerV2,
    custom_code_service: CustomCodeExecutionService,
    agent_name: str = "generic",
    post_processor: Optional[PostProcessor] = None,
):
    """
    Factory function to create execute_custom_code tool with configurable behavior.

    Follows workspace_tool.py pattern for shared tools across agents.
    The underlying CustomCodeExecutionService already supports both modality_name
    (for AnnData) and workspace_keys (for metadata), so NO service changes needed.

    Args:
        data_manager: DataManagerV2 instance for provenance logging
        custom_code_service: Service instance for code execution
        agent_name: Agent identifier for logging (default: "generic")
        post_processor: Optional callback for agent-specific post-processing.
                       Signature: (result, stats, data_manager, workspace_key, modality_name) -> Optional[str]
                       Return value (if any) is appended to response.

    Returns:
        LangChain tool for executing custom Python code

    Example:
        # data_expert (no post-processing)
        execute_custom_code = create_execute_custom_code_tool(
            data_manager=data_manager,
            custom_code_service=custom_code_service,
            agent_name="data_expert",
        )

        # metadata_assistant (with metadata_store persistence)
        execute_custom_code = create_execute_custom_code_tool(
            data_manager=data_manager,
            custom_code_service=custom_code_service,
            agent_name="metadata_assistant",
            post_processor=metadata_store_post_processor,
        )
    """

    @tool
    async def execute_custom_code(
        python_code: str,
        modality_name: Optional[str] = None,
        workspace_key: Optional[str] = None,
        load_workspace_files: bool = True,
        persist: bool = False,
        description: str = "Custom code execution",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Execute custom Python code with workspace context.

        Use ONLY when existing specialized tools don't cover your need.

        Data sources: pass modality_name to get AnnData as `adata`, workspace_key
        for a specific JSON/CSV file, or neither to auto-load all workspace files.

        Namespace: WORKSPACE, OUTPUT_DIR, adata (if modality_name), auto-loaded files.
        Assign results to `result` variable. Large output (>8K chars) is truncated
        — save to OUTPUT_DIR instead.

        Figures: call submit_figure(fig, source=df, intent="what it shows", label="...")
        for every Plotly figure to keep; it is saved only if the code succeeds.
        publish_plot(fig, title) is an alias. Figures are NOT captured automatically
        (changed in 1.2.0): an unsubmitted Figure variable is named in the reply and not
        saved. Saved figures of this session: list_figures(), load_figure("f_..@r_..")
        (write refs literally to load non-latest revisions), load_figure_config(ref),
        load_source(source_id). Ready-made figure templates and the house style are
        available: list_templates(), get_template(name), show_template_source(name);
        prefer a matching template over hand-written plotting code. A request about a
        figure that already exists ("make the title bigger", "use log scale") is an
        EDIT: fig = load_figure("f_..@r_.."),
        change it, then submit_figure(fig, parent="f_..@r_..") with that same ref. This
        saves a new revision of the same figure; the original is kept and the parent's
        config is kept unless you pass config={...} ({} clears it). Make a new figure
        (no parent) only for a genuinely separate plot or a different underlying
        dataset, and say so in the reply; filtering or restyling an existing plot (e.g.
        "show only the top 20 genes") is still an edit.

        Args:
            python_code: Python code to execute (multi-line supported)
            modality_name: AnnData modality to load as 'adata'
            workspace_key: Specific workspace JSON/CSV file to load
            load_workspace_files: Auto-inject workspace files (default: True)
            persist: False=ephemeral (default), True=included in /notebook export
            description: Human-readable description of the operation

        DEFENSIVE PATTERNS (apply these BEFORE running analysis code):

        1. Backed AnnData — If adata is in backed mode, matrix operations will fail.
           Always check and convert first:
               if adata.isbacked:
                   adata = adata.to_memory()

        2. Zero-count observations — Normalization (sc.pp.normalize_total, sc.pp.log1p)
           will produce NaN/Inf on cells with zero total counts. Filter first:
               sc.pp.filter_cells(adata, min_counts=1)
           Or manually:
               adata = adata[adata.obs['total_counts'] > 0]

        3. Sparse matrix compatibility — Matrix reduce operations (.sum(), .mean(),
           .max()) may fail or return unexpected shapes on sparse matrices.
           Wrap in try/except:
               try:
                   counts = adata.X.sum(axis=1)
               except (TypeError, AttributeError):
                   import numpy as np
                   counts = np.asarray(adata.X.todense()).sum(axis=1)
        """
        # Lazy import to prevent circular imports when component_registry loads entry points
        # These exceptions are needed at runtime for except clauses, but importing the
        # service module at module level triggers DataManagerV2 → component_registry → agents
        from lobster.services.execution.custom_code_execution_service import (
            CodeExecutionError,
            CodeValidationError,
        )

        # Validation: mutual exclusivity
        if modality_name and workspace_key:
            return (
                "Error: Cannot specify both `modality_name` and `workspace_key`. "
                "Choose ONE based on your data type:\n"
                "- `modality_name`: For AnnData/H5AD operations\n"
                "- `workspace_key`: For JSON/CSV metadata operations"
            )

        # Empty code guard
        if not python_code or not python_code.strip():
            return _EXECUTE_CUSTOM_CODE_HELP

        try:
            # Convert workspace_key to list (service expects Optional[List[str]])
            workspace_keys_list = [workspace_key] if workspace_key else None

            # Execute via service in a thread so the event loop stays free for
            # SSE heartbeats during long-running subprocess calls.
            submitted: List[Any] = []
            # Born set (the attempt fence): captured BEFORE execute() materializes any input, so a
            # session purge during this attempt fences its figures (generation_fenced).
            attempt_fence = _attempt_fence(data_manager)
            result, stats, ir = await asyncio.to_thread(
                custom_code_service.execute,
                code=python_code,
                modality_name=modality_name,
                load_workspace_files=load_workspace_files,
                workspace_keys=workspace_keys_list,
                persist=persist,
                description=description,
                figure_sink=submitted,
            )

            # Log provenance FIRST: the figure execution record links this exact activity.
            activity = data_manager.log_tool_usage(
                tool_name="execute_custom_code",
                parameters={
                    "description": description,
                    "modality_name": modality_name,
                    "workspace_key": workspace_key,
                    "persist": persist,
                    "agent": agent_name,
                    "duration_seconds": stats.get("duration_seconds", 0),
                    "success": stats.get("success", False),
                },
                description=f"{description} ({'success' if stats['success'] else 'failed'})",
                ir=ir,
                # Executing agent as a first-class field, not buried in `parameters`.
                # `agent_name` was already in scope and already written into
                # `parameters["agent"]` above — but a consumer had to know this tool
                # specifically to find it there, while the activity's own `agent` field said
                # `"data_manager"`. Keep `parameters["agent"]` for backward
                # compatibility with consumers already reading it.
                agent=agent_name,
            )

            # Trusted save of explicitly submitted figures (successful attempt only).
            figure_report = _commit_submitted_figures(
                data_manager,
                submitted,
                stats,
                activity=activity,
                tool_call_id=tool_call_id,
                agent_name=agent_name,
                code=python_code,
                modality_name=modality_name,
                workspace_key=workspace_key,
                persist=persist,
                attempt_fence=attempt_fence,
            )

            # Agent-specific post-processing (only on success)
            post_processor_msg = None
            if post_processor and stats.get("success", False):
                try:
                    post_processor_msg = post_processor(
                        result, stats, data_manager, workspace_key, modality_name
                    )
                except Exception as e:
                    logger.warning(f"Post-processor error (non-fatal): {e}")
                    post_processor_msg = f"Warning: Post-processing failed: {e}"

            # Format response
            return _format_response(
                result, stats, persist, post_processor_msg, figure_report
            )

        except CodeValidationError as e:
            logger.error(f"Code validation failed: {e}")
            import json

            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "error_type": "validation_error",
                    "stderr": str(e),
                },
                indent=2,
            )
        except CodeExecutionError as e:
            logger.error(f"Code execution failed: {e}")
            import json

            # Include defensive coding hints on runtime errors (just-in-time guidance)
            error_str = str(e)
            hints = ""
            if "KeyError" in error_str or "'NoneType'" in error_str:
                hints = (
                    "\n\nHINT — Defensive coding required:\n"
                    "- Use data.get('key', default) instead of data['key']\n"
                    "- Check `if value is not None` before .lower()/.upper()\n"
                    "- Check isinstance(data, dict) before dict operations"
                )
            elif (
                "numpy" in error_str.lower()
                or "int64" in error_str
                or "float64" in error_str
            ):
                hints = (
                    "\n\nHINT — Convert numpy types:\n"
                    "- int(val), float(val), list(arr), str(val)"
                )
            elif "truncat" in error_str.lower() or len(error_str) > 2000:
                hints = (
                    "\n\nHINT — Large output handling:\n"
                    "- Save to OUTPUT_DIR / 'results.json' instead of returning large data\n"
                    "- result = {'saved_to': str(path), 'count': N, 'preview': data[:5]}"
                )

            failed = {
                "success": False,
                "error": error_str + hints,
                "error_type": "execution_error",
                "stderr": error_str,
            }
            discarded = getattr(e, "figures_submitted", 0)
            if discarded:
                failed["figures"] = (
                    f"No figure saved: the code failed, so {discarded} submitted "
                    "figure(s) were discarded."
                )
            return json.dumps(failed, indent=2)
        except Exception as e:
            logger.error(f"Unexpected error in execute_custom_code: {e}", exc_info=True)
            import json

            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "error_type": "unexpected_error",
                    "stderr": str(e),
                },
                indent=2,
            )

    from lobster.tools._async_compat import enable_sync_fallback

    return enable_sync_fallback(execute_custom_code)


# =============================================================================
# Response Formatting
# =============================================================================

# Maximum characters for result field to prevent LLM context overflow
# Large results (e.g., DataFrames, sample lists) can exceed 100K+ chars
MAX_RESULT_LENGTH = 8000


def _truncate_result(result: Any, max_length: int = MAX_RESULT_LENGTH) -> tuple:
    """
    Truncate result to prevent LLM context overflow.

    Handles various result types (dict, list, str, primitives) by converting
    to JSON string, checking length, and truncating if needed.

    Args:
        result: Execution result (any JSON-serializable type)
        max_length: Maximum character length for serialized result

    Returns:
        Tuple of (truncated_result, was_truncated, original_length)
        - truncated_result: Original or truncated value
        - was_truncated: Boolean indicating if truncation occurred
        - original_length: Character count of original serialized result
    """
    import json

    if result is None:
        return None, False, 0

    # Serialize to check length
    try:
        serialized = json.dumps(result, default=str)
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        serialized = str(result)

    original_length = len(serialized)

    if original_length <= max_length:
        return result, False, original_length

    # Truncation needed - strategy depends on type
    if isinstance(result, str):
        # Simple string truncation
        truncated = result[: max_length - 100]  # Leave room for suffix
        return (
            f"{truncated}\n\n... [TRUNCATED: {original_length:,} chars total, "
            f"showing first {len(truncated):,}. "
            f"Save large data to OUTPUT_DIR instead of returning it. Do NOT re-execute.]",
            True,
            original_length,
        )

    elif isinstance(result, list):
        # For lists (e.g., samples), truncate items and show count
        truncated_list = []
        current_length = 2  # Account for []
        for item in result:
            item_str = json.dumps(item, default=str)
            if current_length + len(item_str) + 2 > max_length - 200:
                break
            truncated_list.append(item)
            current_length += len(item_str) + 2  # +2 for comma and space

        return (
            {
                "_truncated": True,
                "_total_items": len(result),
                "_shown_items": len(truncated_list),
                "_original_chars": original_length,
                "_action": "Result too large for context. Save to OUTPUT_DIR / 'results.json' and assign a summary dict to result instead. Do NOT re-execute.",
                "items": truncated_list,
            },
            True,
            original_length,
        )

    elif isinstance(result, dict):
        # For dicts, try to preserve structure but truncate large values
        truncated_dict = {
            "_truncated": True,
            "_original_chars": original_length,
            "_action": "Dict too large for context. Save full dict to OUTPUT_DIR / 'results.json' and return summary. Do NOT re-execute.",
        }
        current_length = 50  # Account for metadata

        for key, value in result.items():
            value_str = json.dumps(value, default=str)
            if current_length + len(key) + len(value_str) + 10 > max_length - 200:
                truncated_dict["_remaining_keys"] = list(
                    set(result.keys()) - set(truncated_dict.keys())
                )
                break

            # Recursively truncate large nested values
            if len(value_str) > max_length // 4:
                if isinstance(value, list) and len(value) > 5:
                    truncated_dict[key] = {
                        "_list_preview": value[:5],
                        "_total_items": len(value),
                    }
                elif isinstance(value, str) and len(value) > 500:
                    truncated_dict[key] = value[:500] + f"... [{len(value)} chars]"
                else:
                    truncated_dict[key] = value
            else:
                truncated_dict[key] = value

            current_length += len(key) + len(
                json.dumps(truncated_dict[key], default=str)
            )

        return truncated_dict, True, original_length

    else:
        # Primitives or other types - convert to string and truncate
        str_result = str(result)
        if len(str_result) > max_length:
            return (
                f"{str_result[: max_length - 100]}\n\n... [TRUNCATED: {original_length:,} chars. "
                f"Save large data to OUTPUT_DIR instead of returning it. Do NOT re-execute.]",
                True,
                original_length,
            )
        return result, False, original_length


def _format_response(
    result: Any,
    stats: Dict[str, Any],
    persist: bool,
    post_processor_msg: Optional[str] = None,
    figure_report: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format unified JSON response for all agents (DeepAgents pattern).

    Returns structured JSON instead of markdown for programmatic parsing by LLMs.
    Enables retry logic, error pattern matching, and conditional branching.

    Includes automatic result truncation to prevent LLM context overflow.
    Large results (>8000 chars) are intelligently truncated while preserving
    structure and indicating total size.

    Args:
        result: Execution result value
        stats: Execution statistics from service
        persist: Whether execution was persisted to provenance
        post_processor_msg: Optional message from agent-specific post-processor

    Returns:
        JSON string with structured execution result
    """
    import json

    # Truncate result to prevent context overflow
    truncated_result, was_truncated, original_length = _truncate_result(result)

    # Build typed response object
    response_obj = {
        "success": True,
        "duration_seconds": stats.get("duration_seconds", 0),
        "result": truncated_result,
        "result_type": stats.get(
            "result_type", type(result).__name__ if result is not None else None
        ),
        "result_truncated": was_truncated,
        "result_original_chars": original_length if was_truncated else None,
        "stdout": stats.get("stdout_preview", ""),
        "stderr": stats.get("stderr_preview", ""),
        "stdout_full_path": stats.get("stdout_full_path"),
        "stderr_full_path": stats.get("stderr_full_path"),
        "warnings": stats.get("warnings", []),
        "persisted": persist,
        "new_modality_name": stats.get("new_modality_name"),
        "write_back_error": stats.get("write_back_error"),
        "replay_excluded": stats.get("replay_excluded"),
        "post_processor_message": post_processor_msg,
    }
    if figure_report:
        response_obj.update(figure_report)

    return json.dumps(response_obj, indent=2)


# =============================================================================
# Trusted save of explicitly submitted figures
# =============================================================================


_FENCED_COMPLETION_STATES = frozenset(
    {"completion_unconfirmed", "result_missing", "result_unreadable"}
)


def _safe_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _unsubmitted_warning(names: List[str]) -> Optional[str]:
    if not names:
        return None
    return (
        f"Not saved: {', '.join(names)}. "
        "Call submit_figure(fig, source=…, intent=…) to save a figure."
    )


def _attempt_fence(data_manager) -> Any:
    """The store's born-set fence for one attempt, or None for a store without one (Cloud)."""
    store = data_manager.figure_store
    capture = getattr(store, "attempt_fence", None)
    return capture() if callable(capture) else None


def _commit_submitted_figures(
    data_manager,
    submitted: List[Any],
    stats: Dict[str, Any],
    *,
    activity: Optional[Dict[str, Any]],
    tool_call_id: str,
    agent_name: str,
    code: str,
    modality_name: Optional[str],
    workspace_key: Optional[str],
    persist: bool,
    attempt_fence: Any = None,
) -> Dict[str, Any]:
    """Commit the successful attempt's submissions; return reply fields (codes only).

    ``attempt_fence``: the born set captured before the attempt's first input read (the tool
    always passes it; None only for direct calls and stores without one)."""
    import uuid

    from lobster.core.figures.records import (
        KIND_CREATED,
        KIND_EDITED,
        Candidate,
        ExecutionInfo,
        InputEdge,
    )
    from lobster.core.figures.saver import TrustedSaver, outcome_state, summarize

    report: Dict[str, Any] = {}
    warning = _unsubmitted_warning(stats.get("unsubmitted_figures") or [])
    completion = stats.get("completion")
    if not (isinstance(completion, str) and completion == "complete") and stats.get(
        "figures_submitted"
    ):
        # Exit 0 but the run did not finish normally (early exit, missing/unreadable result).
        # Closed vocabulary only: an unknown state is reported as completion_unconfirmed.
        state = (
            completion
            if isinstance(completion, str) and completion in _FENCED_COMPLETION_STATES
            else "completion_unconfirmed"
        )
        report["figures"] = (
            "No figure saved: the run did not finish normally, so "
            f"{_safe_count(stats.get('figures_submitted'))} submitted figure(s) were "
            "discarded "
            f"({state})."
        )
        if warning:
            report["unsubmitted_figures"] = warning
        return report
    if stats.get("figure_submission_error"):
        report["figures"] = (
            "No figure saved: the submitted figures were malformed "
            f"({stats['figure_submission_error']})."
        )
        if warning:
            report["unsubmitted_figures"] = warning
        return report
    if not submitted:
        if warning:
            report["figures"] = "No figure saved."
            report["unsubmitted_figures"] = warning
        return report

    # Inside a graph run LangChain always supplies the tool call id; a direct call gets a
    # one-off id with NO repeat-safety.
    call_id = tool_call_id or f"direct_{uuid.uuid4().hex[:12]}"
    session = str(getattr(data_manager, "session_id", None) or "local")
    base = f"{session}:{call_id}"
    # Input edges were captured at materialization, before the child ran, from the object
    # actually given to it. No lookup here: a later store/eviction cannot rebind them.
    inputs = tuple(InputEdge.from_dict(d) for d in stats.get("input_edges") or ())
    # Config: explicit -> exactly those bytes ({} = clear); omitted on an edit ->
    # the parent's EXACT saved config via a full sha-verified read (current lifecycle);
    # omitted on a create -> {}. A failed parent read is a typed refusal, never {}.
    from lobster.core.figures.errors import FigureError

    configs: Dict[int, bytes] = {}
    pre_refused: Dict[int, FigureError] = {}
    for fig in submitted:
        explicit = getattr(fig, "config_bytes", None)
        if explicit is not None:
            configs[fig.index] = explicit
        elif fig.parent:
            try:
                configs[fig.index] = data_manager.figure_store.read(fig.parent)[2]
            except FigureError as exc:
                pre_refused[fig.index] = exc
                configs[fig.index] = (
                    b"{}"  # placeholder: never validated/digested/stored
                )
        else:
            configs[fig.index] = b"{}"
    candidates = [
        Candidate(
            attempt_key=f"{base}:{fig.index}",
            candidate_index=fig.index,
            figure_bytes=fig.figure_bytes,
            config_bytes=configs[fig.index],
            kind=KIND_EDITED if fig.parent else KIND_CREATED,
            actor=f"expert:{agent_name}",
            label=fig.label,
            request=fig.request,
            resolved_intent=fig.resolved_intent,
            parent_ref=fig.parent,
            inputs=inputs,
            source=fig.source,
        )
        for fig in submitted
    ]
    try:
        import plotly

        plotly_version = plotly.__version__
    except Exception:  # pragma: no cover
        plotly_version = "unknown"
    try:
        from lobster.version import __version__ as engine_version
    except Exception:  # pragma: no cover
        engine_version = "unknown"
    info = ExecutionInfo(
        attempt_base=base,
        tool="execute_custom_code",
        agent=agent_name,
        parameters={
            "modality_name": modality_name,
            "workspace_key": workspace_key,
            "persist": bool(persist),
        },
        persist=bool(persist),
        code_bytes=code.encode("utf-8"),
        activity=activity,
        engine_version=engine_version,
        plotly_version=plotly_version,
    )
    results = TrustedSaver(data_manager.figure_store).commit_attempt(
        info,
        candidates,
        succeeded=True,
        pre_refused=pre_refused,
        fence=attempt_fence,
    )
    report["figures"] = summarize(results)
    report["figure_refs"] = [r.ref for r in results if r.status == "available"]
    report["figure_outcome"] = outcome_state(results)
    if warning:
        report["unsubmitted_figures"] = warning
    truncated = (stats.get("figure_snapshot") or {}).get("snapshot_truncated")
    if truncated:
        report["figure_snapshot"] = f"snapshot_truncated: {truncated} not listed"
    unknown = stats.get("input_identity_unknown") or []
    if unknown and report["figure_refs"]:
        report["input_identity"] = (
            "Input identity unknown (recorded as unverified): " + ", ".join(unknown)
        )
    return report


# =============================================================================
# Pre-built Post-Processors
# =============================================================================


def metadata_store_post_processor(
    result: Any,
    stats: Dict[str, Any],
    data_manager: DataManagerV2,
    workspace_key: Optional[str],
    modality_name: Optional[str],
) -> Optional[str]:
    """
    Post-processor for metadata_assistant: Persist results to metadata_store.

    Handles two patterns:
    1. In-place update: If workspace_key exists in metadata_store, update its 'samples'
    2. New key creation: If result contains {'samples': [...], 'output_key': 'name'}

    This enables the metadata_assistant workflow where users filter/transform samples
    and persist them for later export via write_to_workspace.

    Args:
        result: Execution result (should be dict with 'samples' key for persistence)
        stats: Execution statistics (unused, but required by signature)
        data_manager: DataManagerV2 instance with metadata_store
        workspace_key: Original workspace key that was loaded (for in-place updates)
        modality_name: Modality name (unused for metadata operations)

    Returns:
        Message describing what was persisted, or None if nothing persisted

    Example Result Formats:
        # Pattern 1: In-place update (workspace_key="existing_key")
        result = {"samples": [{"id": 1}, {"id": 2}]}

        # Pattern 2: New key creation
        result = {"samples": [...], "output_key": "filtered_samples"}
    """
    if not isinstance(result, dict):
        return None

    # Pattern 1: In-place update of existing workspace_key
    if workspace_key and workspace_key in data_manager.metadata_store:
        if "samples" in result:
            data_manager.metadata_store[workspace_key]["samples"] = result["samples"]
            count = len(result["samples"])
            logger.debug(
                f"Persisted {count} samples to metadata_store['{workspace_key}']"
            )
            return f"Persisted {count} samples to metadata_store['{workspace_key}']"

    # Pattern 2: Create new key with output_key
    elif "samples" in result and "output_key" in result:
        output_key = result["output_key"]
        data_manager.metadata_store[output_key] = {
            "samples": result["samples"],
            "filter_criteria": result.get("filter_criteria", "custom"),
            "stats": result.get("stats", {}),
        }
        count = len(result["samples"])
        logger.debug(f"Created metadata_store['{output_key}'] with {count} samples")
        return (
            f"Created metadata_store['{output_key}'] with {count} samples\n"
            f"Export with: write_to_workspace(identifier='{output_key}', "
            f"workspace='metadata', output_format='csv')"
        )

    return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "create_execute_custom_code_tool",
    "metadata_store_post_processor",
    "PostProcessor",
    "MAX_RESULT_LENGTH",
    "_truncate_result",
]
