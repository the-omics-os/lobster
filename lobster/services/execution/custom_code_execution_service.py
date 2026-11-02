"""
Custom Code Execution Service for ad-hoc Python code execution.

This service provides a fallback mechanism for agents to execute arbitrary
Python code when existing specialized tools don't cover edge cases. It records
analysis steps for notebook export; this does not establish standards compliance.

EXECUTION LIMITS:
- Runs a generated Python script with a timeout (300s default).
- Child and parent run as the same OS user: this is not a sandbox.
- AST screening and a socket.connect patch are cooperative restrictions, not
  filesystem or network confinement.
- Child failures are reported to the caller; no host-isolation guarantee is made.
- Completion files are cooperative signals, not authentication. Managed-figure
  saves through the parent path still require validation, records and fences.
"""

import ast
import json
import os
import stat as _stat
import subprocess  # nosec B404 # Only runs the generated script with the current interpreter; never a shell.
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.execution.execution_context_builder import ExecutionContextBuilder
from lobster.utils.ipc_read import RefusedRead, read_regular, regular_identity
from lobster.utils.logger import get_logger
from lobster.utils.subprocess_env import minimal_child_env

logger = get_logger(__name__)

# Maximum captured output length (10,000 chars ~= 200 lines)
MAX_OUTPUT_LENGTH = 10000

# Default execution timeout (5 minutes)
DEFAULT_TIMEOUT = 300


# Completion states of a run that exited 0. Closed set.
COMPLETION_COMPLETE = "complete"
COMPLETION_UNCONFIRMED = (
    "completion_unconfirmed"  # child never reached its last statement
)
COMPLETION_RESULT_MISSING = "result_missing"  # marker present, result envelope absent
COMPLETION_RESULT_UNREADABLE = "result_unreadable"  # envelope not JSON / wrong shape
COMPLETION_STATES = frozenset(
    {
        COMPLETION_COMPLETE,
        COMPLETION_UNCONFIRMED,
        COMPLETION_RESULT_MISSING,
        COMPLETION_RESULT_UNREADABLE,
    }
)
RESULT_SCHEMA = "lobster.execution_result/1"

# Post-child reads (the child has exited; it controls what is at these paths). Each is a
# bounded read of a regular file only (lobster.utils.ipc_read): a FIFO, device or symlink
# left by the child is refused instead of blocking the parent after the child's timeout.
MAX_RESULT_BYTES = 16 * 1024 * 1024  # result envelope; larger = result_unreadable
MAX_MARKER_BYTES = 64  # completion marker (the nonce is 32 hex characters)
MODIFIED_DATA_NOT_REGULAR = "the modified data file is not a regular file"
MODIFIED_DATA_UNREADABLE = "the modified data file could not be read"

# Names that only the attempt's runtime setup defines: the figure helpers (child_api.install,
# including the publish_plot alias and the template helpers) and the attempt cache path. The
# exported setup defines none of them, so code that uses one cannot run as recorded in an
# exported or replayed notebook: its IR keeps the code but is neither exportable nor
# replayable. Detection bound: a static scan for these names as loaded variables; a helper
# reached only dynamically (through globals(), getattr or a string) is not detected.
FIGURE_HELPER_NAMES = frozenset(
    {
        "submit_figure",
        "publish_plot",
        "list_figures",
        "load_figure",
        "load_figure_config",
        "load_source",
        "FigureUnavailable",
        "list_templates",
        "get_template",
        "show_template_source",
        "apply_house_style",
        "initialize_house_style",
        "EXEC_CACHE",
    }
)
REPLAY_EXCLUDED_FIGURE_HELPERS = "uses_attempt_figure_helpers"


def _uses_figure_helpers(code: str) -> bool:
    """True if ``code`` loads one of :data:`FIGURE_HELPER_NAMES` (unparseable: True)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return True
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in FIGURE_HELPER_NAMES
        for node in ast.walk(tree)
    )


def _read_result_envelope(path: Path) -> "tuple[Any, str]":
    """Return (result, completion state) from the child's result envelope. Never raises."""
    try:
        raw = read_regular(path, MAX_RESULT_BYTES)
    except RefusedRead:
        return None, COMPLETION_RESULT_UNREADABLE
    if raw is None:
        return None, COMPLETION_RESULT_MISSING
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None, COMPLETION_RESULT_UNREADABLE
    if not (
        isinstance(data, dict)
        and data.get("schema") == RESULT_SCHEMA
        and isinstance(data.get("has_result"), bool)
        and "result" in data
        and "type" in data
        and (
            (isinstance(data["type"], str) and data["type"])
            if data["has_result"]
            else (data["result"] is None and data["type"] is None)
        )
    ):
        return None, COMPLETION_RESULT_UNREADABLE
    return data["result"], COMPLETION_COMPLETE


def _child_api_path() -> Path:
    """File path of the child-side figure helpers (loaded by path; no lobster import)."""
    return Path(__file__).resolve().parents[2] / "core" / "figures" / "child_api.py"


class CodeExecutionError(Exception):
    """Base exception for code execution failures."""

    pass


class CodeValidationError(CodeExecutionError):
    """Raised when code fails safety validation."""

    pass


class CustomCodeExecutionService:
    """
    Stateless service for executing arbitrary Python code with safety checks.

    This service follows Lobster's 3-tuple pattern: (result, stats, AnalysisStep).
    It provides a high-trust execution model (Jupyter-like) with package restrictions
    and opt-in provenance tracking.

    Design Philosophy:
    - High trust: Execute user code as-is (like Jupyter)
    - Safety warnings: AST validation, forbidden import detection
    - Ephemeral by default: Only persist to provenance if persist=True
    - Full workspace access: Auto-inject modalities, CSV, JSON, queues

    Injected Variables (available in custom code):
    - WORKSPACE (Path): Full workspace path for reading inputs
    - OUTPUT_DIR (Path): Recommended directory for exports (workspace/exports/)
    - modalities (list): Available modality names
    - adata (AnnData): Loaded modality if modality_name specified
    - Workspace data (dict/DataFrame): If load_workspace_files=True

    File Output Convention:
        Use OUTPUT_DIR for CSV/TSV/Excel exports to ensure predictable file locations:

        >>> import pandas as pd
        >>> df = pd.DataFrame(...)
        >>> df.to_csv(OUTPUT_DIR / "my_results.csv")

        This saves to workspace/exports/my_results.csv for easy user discovery.
        Note: OUTPUT_DIR is a convention, not enforced. Code can write anywhere.

    Example:
        >>> service = CustomCodeExecutionService(data_manager)
        >>> result, stats, ir = service.execute(
        ...     code="result = 2 + 2",
        ...     persist=False
        ... )
        >>> result
        4
    """

    # Forbidden imports (security) - these modules are blocked entirely
    FORBIDDEN_MODULES = {
        # Process execution
        "subprocess",
        "multiprocessing",
        "concurrent",
        # Import mechanisms (bypass vectors)
        "importlib",
        "imp",
        # Serialization (RCE vectors via pickle deserialization)
        "pickle",
        "marshal",
        "shelve",
        "dill",
        # Low-level (escape vectors)
        "ctypes",
        "cffi",
        # Network access (blocked for security - use Lobster's tools for downloads)
        "socket",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "websocket",
        "websockets",
    }

    # Forbidden specific imports (from X import Y)
    # os module is allowed for os.path, but dangerous functions are blocked
    FORBIDDEN_FROM_IMPORTS = {
        # os dangerous functions (keep os module for os.path)
        ("os", "system"),
        ("os", "popen"),
        ("os", "exec"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "spawnl"),
        ("os", "spawnle"),
        ("os", "spawnlp"),
        ("os", "spawnlpe"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "kill"),
        ("os", "killpg"),
        # signal manipulation (can kill parent process)
        ("signal", "signal"),
        ("signal", "kill"),
        ("signal", "alarm"),
        # destructive file ops
        ("shutil", "rmtree"),
        ("shutil", "move"),
        # Network access (urllib.parse is OK for URL string manipulation,
        # urllib.request is not - it makes actual HTTP calls)
        ("urllib", "request"),
        ("urllib", "urlopen"),
    }

    # Allowed imports (standard Lobster stack from pyproject.toml)
    ALLOWED_MODULES = {
        # Core scientific computing
        "numpy",
        "np",
        "pandas",
        "pd",
        "scipy",
        "sklearn",
        # Bioinformatics
        "scanpy",
        "sc",
        "anndata",
        "ad",
        "mudata",
        "mdata",
        "pydeseq2",
        "biopython",
        "Bio",
        # Visualization
        "plotly",
        "matplotlib",
        "plt",
        "seaborn",
        "sns",
        # Standard library (safe subset)
        "math",
        "statistics",
        "re",
        "json",
        "csv",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "typing",
        "pathlib",
        "Path",
        # Lobster internal (for advanced users)
        "lobster",
    }

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize the custom code execution service.

        Args:
            data_manager: DataManagerV2 instance for workspace access
        """
        self.data_manager = data_manager
        self.context_builder = ExecutionContextBuilder(data_manager)
        logger.debug("Initialized CustomCodeExecutionService")

    def _resolve_workspace_key_path(self, key: str) -> Optional[Path]:
        """
        Resolve workspace_key to actual file path in workspace subdirectories.

        Searches in priority order: metadata/ → data/ → literature/ → root (backward compat)
        Follows DRY principle by using workspace structure knowledge.

        Args:
            key: Workspace key identifier

        Returns:
            Path to file if found, None otherwise

        Example:
            >>> service._resolve_workspace_key_path("harmonized_publication_metadata")
            Path(".lobster_workspace/metadata/harmonized_publication_metadata.json")
        """
        workspace_path = Path(self.data_manager.workspace_path)

        # Search workspace subdirectories in priority order
        search_dirs = [
            "metadata",
            "data",
            "literature",
            "",
        ]  # "" = root (backward compat)

        for subdir in search_dirs:
            base_path = workspace_path / subdir if subdir else workspace_path

            # Try both .json and .csv extensions
            for ext in [".json", ".csv"]:
                file_path = base_path / f"{key}{ext}"
                if file_path.exists():
                    logger.debug(
                        f"Resolved workspace_key '{key}' → {file_path.relative_to(workspace_path)}"
                    )
                    return file_path

        logger.warning(f"workspace_key '{key}' not found in any subdirectory")
        return None

    def execute(
        self,
        code: str,
        modality_name: Optional[str] = None,
        load_workspace_files: bool = True,
        workspace_keys: Optional[List[str]] = None,
        persist: bool = False,
        description: str = "Custom code execution",
        timeout: int = DEFAULT_TIMEOUT,
        figure_sink: Optional[List[Any]] = None,
    ) -> Tuple[Any, Dict[str, Any], AnalysisStep]:
        """
        Execute arbitrary Python code with workspace context injection.

        Args:
            code: Python code to execute (can be multi-line)
            modality_name: Optional specific modality to load as 'adata'
            load_workspace_files: Auto-inject CSV/JSON files from workspace
            workspace_keys: Optional list of specific workspace keys to load (reduces token usage)
                          If provided, only these files are loaded. If None, all files are loaded.
                          Example: ["sra_prjna834801_samples"] loads only that file
            persist: If True, save this execution to provenance/notebook export
            description: Human-readable description of what this code does
            timeout: Maximum execution time in seconds (default: 300s)
            figure_sink: Optional caller-owned list. On a SUCCESSFUL attempt only, the figures
                the code submitted (``submit_figure``) are appended as
                :class:`~lobster.core.figures.records.SubmittedFigure`; the caller commits them
                through the trusted saver after logging provenance. Nothing is saved here.

        Returns:
            Tuple containing:
            - result: Execution result (from 'result' variable or last expression)
            - stats: Execution statistics (success, duration, warnings, etc.)
            - ir: AnalysisStep for provenance tracking

        Raises:
            CodeValidationError: If code fails safety checks
            CodeExecutionError: If execution fails

        Example:
            >>> result, stats, ir = service.execute(
            ...     code="import numpy as np\\nresult = np.mean([1, 2, 3])",
            ...     persist=False
            ... )
            >>> result
            2.0
            >>> stats['success']
            True

            >>> # Selective loading to avoid token overflow
            >>> result, stats, ir = service.execute(
            ...     code="samples = sra_prjna834801_samples['samples']; result = len(samples)",
            ...     workspace_keys=["sra_prjna834801_samples"],
            ...     persist=False
            ... )
        """
        start_time = time.time()

        logger.debug(f"Executing custom code: {description}")
        logger.debug(f"Code ({len(code)} chars):\n{code[:500]}...")

        # Step 1: Validate code safety
        validation_warnings = self._validate_code_safety(code)

        # Step 2: Materialize modalities to local disk for subprocess IPC.
        # Inputs are materialized FRESH for every attempt into <exec_cache>/inputs and reused
        # only within this attempt (freshness): an in-place change or a
        # rewritten local file keeps its lineage id, so nothing is reused across attempts.
        # The per-execution UUID dir is removed after the run (inputs included).
        from lobster.core.interfaces.locality import _safe_filename

        # Preflight BEFORE the attempt directory exists, so a refusal leaves nothing behind:
        # detect all modalities referenced in code (fails fast on unknown paths), and refuse
        # names that share one file name (they would alias one file).
        required_modalities = self._detect_required_modalities(code, modality_name)
        by_file: Dict[str, List[str]] = {}
        for name in required_modalities:
            by_file.setdefault(_safe_filename(name), []).append(name)
        shared = sorted(sorted(v) for v in by_file.values() if len(v) > 1)
        if shared:
            raise CodeValidationError(
                "These inputs map to the same file name, so they cannot be told apart: "
                + "; ".join(", ".join(repr(n) for n in group) for group in shared)
                + ". Rename one of the modalities, then run again."
            )

        exec_id = uuid.uuid4().hex[:12]
        exec_cache = self.data_manager.workspace_path / "cache" / "execution" / exec_id
        exec_cache.mkdir(parents=True, exist_ok=True)
        materialized_dir = exec_cache / "inputs"
        materialized_dir.mkdir(parents=True, exist_ok=True)

        alias_map = {}
        materialized_inputs: List[Dict[str, Any]] = []  # delivery cost only
        # Input edges captured HERE, before the child and before any plan/digest, from the
        # object (or provider file) actually materialized. Never a later
        # lookup by name, so a store or eviction after this point cannot rebind an edge.
        input_edges: List[Dict[str, Any]] = []
        identity_unknown: List[str] = []

        for name in required_modalities:
            t0 = time.time()
            identity: Dict[str, Any] = {}
            try:
                local_path = self._materialize_modality(
                    name, materialized_dir, identity
                )
                size = local_path.stat().st_size  # inside the cleanup boundary
            except Exception as e:
                import shutil

                shutil.rmtree(exec_cache, ignore_errors=True)  # no copy left behind
                if isinstance(e, FileNotFoundError):
                    raise CodeValidationError(str(e)) from e
                raise CodeExecutionError(
                    f"Failed to materialize modality '{name}': {e}"
                ) from e
            materialized_inputs.append(
                {
                    "name": name,
                    "bytes": size,
                    "seconds": round(time.time() - t0, 3),
                }
            )
            edge = identity.get("edge")
            if (
                edge is None
            ):  # known required name, identity unresolved: kept, unverified
                from lobster.core.figures.records import InputEdge

                identity_unknown.append(name)
                edge = InputEdge(name, None, "declared", False)
            input_edges.append(edge.to_dict())
            safe = _safe_filename(name)
            real = str(local_path)
            ws = str(self.data_manager.workspace_path)
            alias_map[f"{ws}/modalities/{safe}.h5ad"] = real
            alias_map[f"{ws}/data/{safe}.h5ad"] = real
            # Alias keys rewrite paths in model-written code; these are never
            # opened as temporary files.
            alias_map[f"/tmp/workspace/modalities/{safe}.h5ad"] = real  # nosec B108
            alias_map[f"/tmp/workspace/data/{safe}.h5ad"] = real  # nosec B108
            alias_map[f"data/{safe}.h5ad"] = real
            alias_map[f"{safe}.h5ad"] = real

        # Step 2.5: Resolve workspace_key paths (Bug fix - Gemini Option C)
        resolved_paths = {}
        if workspace_keys:
            for key in workspace_keys:
                resolved_path = self._resolve_workspace_key_path(key)
                if resolved_path:
                    resolved_paths[key] = str(resolved_path)
                else:
                    logger.warning(
                        f"workspace_key '{key}' not found, will be unavailable in code"
                    )

        # Step 3: Build execution context (now just metadata for subprocess)
        exec_context = {
            "modality_name": modality_name,
            "workspace_path": self.data_manager.workspace_path,
            "load_workspace_files": load_workspace_files,
            "workspace_keys": workspace_keys,
            "resolved_paths": resolved_paths,
            "exec_cache_path": exec_cache,
            "materialized_dir": materialized_dir,
            "alias_map": alias_map,
            "timeout": timeout,
        }

        # Attempt-start figure snapshot, read-only for the child.
        snapshot_info: Dict[str, Any] = {}
        try:
            from lobster.core.figures.snapshot import build_snapshot

            index = build_snapshot(
                self.data_manager.figure_store, code, exec_cache / "figures"
            )
            snapshot_info = {
                "catalog_generation": index["catalog_generation"],
                "revisions": len(index["revisions"]),
                "snapshot_truncated": index["snapshot_truncated"],
            }
        except (
            Exception
        ) as e:  # the attempt still runs; the child sees an empty snapshot
            from lobster.core.figures.errors import safe_reason

            snapshot_info = {"error": safe_reason(e)}
            logger.warning("Figure snapshot unavailable: %s", snapshot_info["error"])

        # Step 4: Execute code in subprocess
        result, stdout_output, stderr_output, exec_error = self._execute_in_namespace(
            code, exec_context
        )

        # Step 4a: Detect and read back modified adata (snapshot write-back)
        new_modality_name = None
        write_back_error = None
        if modality_name and exec_error is None:
            modified_path = exec_cache / ".modified_adata.h5ad"
            # Only a regular file is read: a FIFO, device or link left by the code is
            # refused. The file is then opened again by path. Custom code runs as your own
            # user and is not sandboxed: another process of that user can still disrupt
            # the session.
            try:
                modified_present = regular_identity(modified_path) is not None
            except RefusedRead:
                modified_present = False
                write_back_error = MODIFIED_DATA_NOT_REGULAR
                logger.warning("Adata write-back refused: %s", write_back_error)
            modified_adata = None
            if modified_present:
                try:
                    import anndata

                    modified_adata = anndata.read_h5ad(modified_path)
                except Exception:  # a closed code: library/OS messages can carry paths
                    write_back_error = MODIFIED_DATA_UNREADABLE
                    logger.warning("Adata write-back refused: %s", write_back_error)
            if modified_adata is not None:
                try:
                    # Deduplicate: _custom, _custom_2, _custom_3...
                    new_modality_name = f"{modality_name}_custom"
                    counter = 2
                    while new_modality_name in self.data_manager.list_modalities():
                        new_modality_name = f"{modality_name}_custom_{counter}"
                        counter += 1

                    # Extract change description from stdout sentinel
                    step_summary = description
                    if stdout_output and "__LOBSTER_CHANGES__:" in stdout_output:
                        for line in stdout_output.splitlines():
                            if "__LOBSTER_CHANGES__:" in line:
                                step_summary = f"{description} ({line.split('__LOBSTER_CHANGES__:', 1)[1].strip()})"
                                break

                    self.data_manager.store_modality(
                        name=new_modality_name,
                        adata=modified_adata,
                        parent_name=modality_name,
                        step="custom",
                        step_summary=step_summary,
                    )
                    logger.info(
                        f"Stored modified adata as '{new_modality_name}' (parent: '{modality_name}')"
                    )
                except Exception as e:
                    write_back_error = str(e)
                    new_modality_name = None
                    logger.warning(f"Adata write-back failed (non-fatal): {e}")
                finally:
                    try:
                        modified_path.unlink(missing_ok=True)
                    except Exception:  # nosec B110 # Best-effort removal of the temporary write-back file; failure must not mask the result.
                        pass

        # Step 4b: read explicitly submitted figures (successful attempts only; the caller
        # commits them via the trusted saver after logging provenance).
        from lobster.core.figures.records import read_submitted_figures

        submitted, unsubmitted, submission_error = read_submitted_figures(exec_cache)
        completion = exec_context.get("completion", COMPLETION_UNCONFIRMED)
        if (
            exec_error is None
            and completion == COMPLETION_COMPLETE
            and figure_sink is not None
            and submission_error is None
        ):
            figure_sink.extend(submitted)

        # Strip sentinel lines from stdout (LLM shouldn't see internal signals)
        if stdout_output:
            stdout_output = "\n".join(
                line
                for line in stdout_output.splitlines()
                if "__LOBSTER_ADATA_MODIFIED__" not in line
                and "__LOBSTER_CHANGES__:" not in line
            )

        # Clean up UUID-scoped exec_cache now that all IPC reads are done
        try:
            import shutil

            shutil.rmtree(exec_cache, ignore_errors=True)
        except Exception:  # nosec B110 # Best-effort cleanup of the per-execution cache once all reads are done.
            pass

        # Step 4b: Save large outputs to files (DeepAgents pattern)
        workspace_path = Path(self.data_manager.workspace_path)
        MAX_STDOUT_PREVIEW = 500
        MAX_STDERR_PREVIEW = 2000
        stdout_file_path = None
        stderr_file_path = None

        if stdout_output and len(stdout_output) > MAX_STDOUT_PREVIEW:
            output_dir = workspace_path / ".execution_outputs"
            output_dir.mkdir(exist_ok=True)
            stdout_file = output_dir / f"stdout_{int(time.time() * 1000)}.txt"
            stdout_file.write_text(stdout_output)
            stdout_file_path = str(stdout_file)
            stdout_preview = (
                stdout_output[:MAX_STDOUT_PREVIEW]
                + f"\n\n... [OUTPUT TRUNCATED at {MAX_STDOUT_PREVIEW} chars. "
                f"Full stdout saved to: {stdout_file_path}. "
                "DO NOT re-execute to see the rest. "
                "Save large outputs to a file and return a summary instead.]"
            )
        else:
            stdout_preview = stdout_output if stdout_output else ""

        if stderr_output and len(stderr_output) > MAX_STDERR_PREVIEW:
            output_dir = workspace_path / ".execution_outputs"
            output_dir.mkdir(exist_ok=True)
            stderr_file = output_dir / f"stderr_{int(time.time() * 1000)}.txt"
            stderr_file.write_text(stderr_output)
            stderr_file_path = str(stderr_file)
            stderr_preview = (
                stderr_output[:MAX_STDERR_PREVIEW]
                + f"\n\n... Output truncated at {MAX_STDERR_PREVIEW} chars. Full output: {stderr_file_path}"
            )
        else:
            stderr_preview = stderr_output if stderr_output else ""

        # Step 5: Compute statistics
        duration = time.time() - start_time
        stats = {
            "success": exec_error is None,
            "duration_seconds": round(duration, 3),
            "warnings": validation_warnings,
            "stdout_lines": len(stdout_output.splitlines()) if stdout_output else 0,
            "stderr_lines": len(stderr_output.splitlines()) if stderr_output else 0,
            "stdout_preview": stdout_preview,
            "stderr_preview": stderr_preview,
            "stdout_full_path": stdout_file_path,
            "stderr_full_path": stderr_file_path,
            "result_type": type(result).__name__ if result is not None else None,
            "modality_loaded": modality_name,
            "workspace_files_loaded": load_workspace_files,
            "workspace_keys": workspace_keys,
            "selective_loading": workspace_keys is not None,
            "persisted": persist,
            "notebook_export_requested": persist,
            "modality_saved": (
                (
                    None
                    if getattr(self.data_manager, "_modality_backend", None) is not None
                    else False
                )
                if new_modality_name
                else None
            ),
            "modality_storage": (
                (
                    "in memory (disk persistence not confirmed)"
                    if getattr(self.data_manager, "_modality_backend", None) is not None
                    else "in memory (not saved to disk)"
                )
                if new_modality_name
                else None
            ),
            "error": str(exec_error) if exec_error else None,
            "new_modality_name": new_modality_name,
            "write_back_error": write_back_error,
            "figures_submitted": len(submitted),
            "completion": completion if exec_error is None else None,
            "unsubmitted_figures": unsubmitted,
            "figure_submission_error": submission_error,
            "figure_snapshot": snapshot_info,
            "materialized_inputs": materialized_inputs,
            # one edge per required name, sorted by name (deterministic on retry)
            "input_edges": sorted(input_edges, key=lambda e: e["name"]),
            "input_identity_unknown": sorted(identity_unknown),
        }

        # Step 5: Generate IR (always, but mark as non-exportable if persist=False)
        ir = self._create_ir(
            code=code,
            description=description,
            modality_name=modality_name,
            load_workspace_files=load_workspace_files,
            workspace_keys=workspace_keys,
            persist=persist,
            stats=stats,
        )

        # Update IR output_entities if a new modality was created
        if new_modality_name:
            ir.output_entities = [new_modality_name]
        stats["replay_excluded"] = ir.execution_context.get("replay_excluded")
        stats["replay_incomplete"] = bool(new_modality_name) and not ir.replayable
        if stats["replay_incomplete"]:  # the data changed, but no replay can redo it
            ir.execution_context["replay_incomplete"] = True

        logger.debug(
            f"Code execution {'succeeded' if stats['success'] else 'failed'} "
            f"in {duration:.2f}s"
        )

        if exec_error:
            logger.error(f"Execution error: {exec_error}")
            err = CodeExecutionError(
                f"Code execution failed: {exec_error}\n\n"
                f"Stderr output:\n{stderr_output}"
            )
            err.figures_submitted = len(submitted)  # fence: none of them is saved
            raise err

        return result, stats, ir

    def _validate_code_safety(self, code: str) -> List[str]:
        """
        Validate code for syntax errors and forbidden imports.

        Args:
            code: Python code to validate

        Returns:
            List of warning messages (empty if no warnings)

        Raises:
            CodeValidationError: If code has syntax errors or forbidden imports

        Example:
            >>> warnings = service._validate_code_safety("import pandas as pd")
            >>> warnings
            []
            >>> service._validate_code_safety("import subprocess")
            Traceback (most recent call last):
            ...
            CodeValidationError: Forbidden import: 'subprocess'
        """
        warnings = []

        # Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeValidationError(
                f"Syntax error in code at line {e.lineno}: {e.msg}\nCode: {e.text}"
            )

        # Import validation
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_MODULES:
                        raise CodeValidationError(
                            f"Forbidden import: '{alias.name}'. "
                            f"Subprocess execution and destructive file operations are restricted."
                        )
                    if alias.name not in self.ALLOWED_MODULES:
                        warnings.append(
                            f"Import '{alias.name}' not in standard Lobster stack. "
                            f"Execution may fail if module not installed."
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                # Check for forbidden module imports
                if module in self.FORBIDDEN_MODULES:
                    raise CodeValidationError(
                        f"Forbidden import: '{module}'. "
                        f"Subprocess execution and file system operations are restricted."
                    )

                # Check for forbidden specific imports (from X import Y)
                for alias in node.names:
                    if (module, alias.name) in self.FORBIDDEN_FROM_IMPORTS:
                        raise CodeValidationError(
                            f"Forbidden import: 'from {module} import {alias.name}'. "
                            f"System command execution and destructive operations are restricted."
                        )

                if module and module.split(".")[0] not in self.ALLOWED_MODULES:
                    warnings.append(
                        f"Import from '{module}' not in standard Lobster stack."
                    )

            # SECURITY: Block dangerous code execution functions
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile"]:
                        raise CodeValidationError(
                            f"Function '{node.func.id}()' is forbidden. "
                            f"This function can execute arbitrary code and bypass safety checks. "
                            f"Use explicit operations instead."
                        )
                    if node.func.id == "__import__":
                        raise CodeValidationError(
                            "Function '__import__()' is forbidden. "
                            "Use standard 'import' statements instead."
                        )

        return warnings

    def _detect_required_modalities(
        self, code: str, explicit_modality: Optional[str]
    ) -> set:
        """Parse code for h5ad path references and match to known modalities.

        Raises CodeValidationError if unresolvable h5ad paths are found (fail-fast).
        """
        import re

        required = set()
        available = set(self.data_manager.locality_provider.list_available())
        # Also include in-memory modalities (may not be in provider's list yet)
        available |= set(self.data_manager.list_modalities())

        if explicit_modality:
            if explicit_modality not in available:
                raise CodeValidationError(
                    f"Modality '{explicit_modality}' not found. "
                    f"Available: {sorted(available)}"
                )
            required.add(explicit_modality)
        unresolved = []

        # Regex patterns for h5ad file access
        patterns = [
            r"""(?:read_h5ad|\.read)\s*\(\s*['"](.*?\.h5ad)['"]\s*""",
            r"""Path\s*\(\s*['"](.*?\.h5ad)['"]\s*\)""",
            r"""h5py\.File\s*\(\s*['"](.*?\.h5ad)['"]\s*""",
        ]
        h5ad_refs = []
        for pattern in patterns:
            h5ad_refs.extend(re.findall(pattern, code))

        for ref in h5ad_refs:
            stem = Path(ref).stem
            if stem in available:
                required.add(stem)
            else:
                unresolved.append(ref)

        if unresolved:
            raise CodeValidationError(
                "Code references h5ad files that don't match any known modality:\n"
                + "\n".join(f"  - {p}" for p in unresolved)
                + f"\n\nAvailable modalities: {sorted(available)}"
                + "\n\nUse the modality_name parameter instead of hardcoded paths."
            )

        return required

    def _capture_identity(self, name: str, adata: Any):
        """InputEdge of the materialized object (or provider file when ``adata`` is None)."""
        from lobster.core.figures.records import InputEdge

        resolver = getattr(self.data_manager, "input_identity_of", None)
        if resolver is None:
            return None
        try:
            edge = resolver(name, adata)
        except Exception:  # an identity failure must never invent an identity
            return None
        return edge if isinstance(edge, InputEdge) else None

    def _materialize_modality(
        self,
        name: str,
        materialized_dir: Path,
        identity_out: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Materialize a single modality for ONE attempt (fresh per attempt).

        ``materialized_dir`` is per attempt (``<exec_cache>/inputs``). When ``identity_out``
        is given, ``identity_out["edge"]`` is set to the identity of the SAME object (or
        provider file) written for the child, or None when unknown. A file already there was
        written earlier in THIS attempt and is reused; nothing is reused across attempts, so
        every attempt reads the current data.

        Step 1: A DM-held modality → serialize the object the DM returns via H5ADBackend
                (a backed one is converted first with ``materialize_modality``: RAM cost as
                on the resident path, and the memory budget may spill another modality).
                A failure is an error, never a fallback to a possibly older provider file.
        Step 2: A name the DM does not hold → delegate to the locality provider (S3 download, disk lookup, etc.); a file that
                lives elsewhere (e.g. data_dir) is copied, never moved
        """
        import os
        import shutil

        from lobster.core.interfaces.locality import _safe_filename

        safe_name = _safe_filename(name)
        target = materialized_dir / f"{safe_name}.h5ad"

        # within-attempt reuse only (the directory is per attempt)
        if target.exists() and target.stat().st_size > 0:
            return target

        # Step 1: a modality the data manager holds (resident, spilled or backend) is
        # serialized from the object it returns. A backed object (e.g. a spilled modality
        # reloaded backed) is first converted with the EXISTING materialize_modality, so a
        # DM-held input never falls to a provider file that may be older.
        if name in self.data_manager.list_modalities():
            adata = self.data_manager.get_modality(name)
            if getattr(adata, "isbacked", False):
                try:
                    adata = self.data_manager.materialize_modality(name)
                except Exception as e:
                    raise RuntimeError(
                        f"could not load the backed modality '{name}' into memory "
                        f"({type(e).__name__})"
                    ) from e
                if getattr(adata, "isbacked", False):
                    raise RuntimeError(
                        f"could not load the backed modality '{name}' into memory"
                    )
            # always serialize the DM-held object; never the provider for this name
            tmp_dir = materialized_dir / f".tmp_{uuid.uuid4().hex[:8]}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                from lobster.core.backends.h5ad_backend import H5ADBackend

                local_backend = H5ADBackend(base_path=tmp_dir)
                local_backend.save(adata, f"{safe_name}.h5ad")
                tmp_file = tmp_dir / f"{safe_name}.h5ad"
                os.replace(str(tmp_file), str(target))
                if identity_out is not None:
                    identity_out["edge"] = self._capture_identity(name, adata)
                return target
            except Exception as e:
                # Never fall back to a provider file: it may be older than this
                # in-memory object.
                raise RuntimeError(
                    f"could not serialize the in-memory modality '{name}' "
                    f"({type(e).__name__})"
                ) from e
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Step 2: delegate to locality provider (S3 download, disk check, etc.)
        provider = self.data_manager.locality_provider
        tmp_dir = materialized_dir / f".tmp_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            provider_path = Path(provider.ensure_local(name, tmp_dir))
            if not provider_path.exists() or provider_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Provider returned invalid h5ad for '{name}': {provider_path}"
                )
            # If provider returned a file inside our temp dir, move it (atomic).
            # If it returned an existing file elsewhere (e.g. data_dir), COPY to
            # avoid moving the original and breaking the local CLI.
            try:
                provider_path.relative_to(tmp_dir)
                os.replace(str(provider_path), str(target))
            except ValueError:
                shutil.copy2(str(provider_path), str(target))
            if (
                identity_out is not None
            ):  # provider identity only; never the DM's lineage
                identity_out["edge"] = self._capture_identity(name, None)
            return target
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _generate_context_setup_code(
        self,
        modality_name: Optional[str],
        workspace_path: Path,
        load_workspace_files: bool,
        workspace_keys: Optional[List[str]] = None,
        resolved_paths: Optional[Dict[str, str]] = None,
        exec_cache_path: Optional[Path] = None,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate Python code to set up execution context in subprocess.

        Since subprocess runs in isolation, we can't pass objects directly.
        Instead, we generate code that loads data from workspace files.

        Args:
            modality_name: Optional modality to load as 'adata'
            workspace_path: Path to workspace
            load_workspace_files: Whether to load CSV/JSON files
            workspace_keys: Optional list of specific keys to load (selective loading)

        Returns:
            Python code string that sets up context
        """
        exec_cache_str = (
            str(exec_cache_path)
            if exec_cache_path
            else str(workspace_path / "cache" / "execution")
        )
        alias_map_repr = repr(alias_map or {})
        setup_code = f"""
# Auto-generated context setup for Lobster custom code execution

# === NETWORK ISOLATION (defense-in-depth) ===
# Patch socket module to block all outbound network I/O from subprocess.
# This catches transitive network calls from allowed libraries
# (e.g., scanpy internally trying to download reference data).
# We patch connect/connect_ex instead of socket() constructor because
# patching the constructor breaks internal socket usage (e.g., multiprocessing IPC).
import socket as _orig_socket_module

_orig_connect = _orig_socket_module.socket.connect
_orig_connect_ex = _orig_socket_module.socket.connect_ex

def _blocked_connect(self, *args, **kwargs):
    raise PermissionError(
        "Network access is disabled in custom code execution. "
        "Use Lobster's built-in tools for data download."
    )

_orig_socket_module.socket.connect = _blocked_connect
_orig_socket_module.socket.connect_ex = _blocked_connect
_orig_socket_module.create_connection = lambda *a, **kw: (_ for _ in ()).throw(
    PermissionError("Network access is disabled in custom code execution.")
)
# === END NETWORK ISOLATION ===

import sys
import builtins
from pathlib import Path
import json

# ==============================================================================
# SECURITY NOTES:
# - Dangerous builtins (eval, exec, compile, __import__) are blocked at AST level
# - We do NOT remove them at runtime because Python's import system needs exec()
# - Network access blocked via socket.connect patch (AST + runtime defense-in-depth)
# - File access is unrestricted in subprocess, but env vars are filtered
# ==============================================================================

# Disable Arrow-backed strings (H5AD cannot serialize ArrowStringArray)
import pandas as _pd
_pd.options.future.infer_string = False

# ==============================================================================
# Execution Locality Resolver — redirects hardcoded h5ad paths to materialized files
# ==============================================================================
import os as _os

_LOBSTER_ALIAS_MAP = {alias_map_repr}

def _resolve_alias(path):
    str_path = str(path)
    norm_path = _os.path.normpath(str_path)
    basename = _os.path.basename(str_path)
    for candidate in [str_path, norm_path, basename]:
        if candidate in _LOBSTER_ALIAS_MAP:
            real = _LOBSTER_ALIAS_MAP[candidate]
            print(f"[lobster] Redirected {{str_path}} -> {{real}}", file=__import__('sys').stderr)
            return real
    return path

try:
    import anndata as _ad
    _orig_read_h5ad = _ad.read_h5ad
    def _aliased_read_h5ad(path, *args, **kwargs):
        return _orig_read_h5ad(_resolve_alias(path), *args, **kwargs)
    _ad.read_h5ad = _aliased_read_h5ad
except ImportError:
    pass

try:
    import scanpy as _sc
    _sc.read_h5ad = _aliased_read_h5ad
    _orig_sc_read = _sc.read
    def _aliased_sc_read(path, *args, **kwargs):
        return _orig_sc_read(_resolve_alias(path), *args, **kwargs)
    _sc.read = _aliased_sc_read
except (ImportError, NameError):
    pass

try:
    import h5py as _h5py
    _orig_h5py_File_init = _h5py.File.__init__
    def _patched_h5py_init(self, path, mode='r', *args, **kwargs):
        if mode in ('r', 'r+'):
            path = _resolve_alias(path)
        return _orig_h5py_File_init(self, path, mode, *args, **kwargs)
    _h5py.File.__init__ = _patched_h5py_init
except (ImportError, AttributeError):
    pass
# === End Locality Resolver ===

# ==============================================================================
# Workspace configuration
# ==============================================================================
WORKSPACE = Path('{workspace_path}')
sys.path.append(str(WORKSPACE))  # SECURITY: Lower priority - cannot shadow stdlib

# EXEC_CACHE: Per-execution temp directory for subprocess IPC (modality h5ad files)
EXEC_CACHE = Path('{exec_cache_str}')
EXEC_CACHE.mkdir(parents=True, exist_ok=True)

# OUTPUT_DIR: Recommended directory for all CSV/TSV/Excel exports (v1.0+)
# Using this ensures files go to workspace/exports/ for easy user discovery
# Convention: df.to_csv(f"{{OUTPUT_DIR}}/my_results.csv")
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize context
modalities = []
"""

        # Add modality loading if specified
        if modality_name:
            from lobster.core.interfaces.locality import _safe_filename

            safe_mod_name = _safe_filename(modality_name)
            materialized_dir_str = (
                str(Path(exec_cache_str) / "inputs")  # per attempt (freshness)
                if exec_cache_path
                else str(workspace_path / "cache" / "execution" / "materialized")
            )
            setup_code += f"""
# Load specified modality (check materialized cache first, then exec cache)
try:
    import anndata
    anndata.settings.allow_write_nullable_strings = True
    _materialized_path = Path('{materialized_dir_str}') / '{safe_mod_name}.h5ad'
    _exec_cache_path = EXEC_CACHE / '{safe_mod_name}.h5ad'
    if _materialized_path.exists():
        adata = anndata.read_h5ad(_materialized_path)
        print(f"Loaded modality '{modality_name}': {{adata.n_obs}} obs x {{adata.n_vars}} vars")
    elif _exec_cache_path.exists():
        adata = anndata.read_h5ad(_exec_cache_path)
        print(f"Loaded modality '{modality_name}': {{adata.n_obs}} obs x {{adata.n_vars}} vars")
    else:
        print(f"Warning: Modality file not found at {{_materialized_path}} or {{_exec_cache_path}}")
        adata = None
except Exception as e:
    print(f"Error loading modality: {{e}}")
    adata = None

# === ADATA FINGERPRINT (pre-user-code) ===
_lobster_fp = None
if adata is not None:
    try:
        _lobster_fp = {{
            "shape": adata.shape,
            "obs_cols": sorted(adata.obs.columns.tolist()),
            "var_cols": sorted(adata.var.columns.tolist()),
            "obsm_keys": sorted(adata.obsm.keys()),
            "obsp_keys": sorted(adata.obsp.keys()),
            "uns_keys": sorted(adata.uns.keys()),
        }}
    except Exception:
        _lobster_fp = None
"""

        # Add workspace file loading
        if load_workspace_files:
            # Determine if selective loading is enabled
            if workspace_keys:
                # v2.0: Use pre-resolved paths from main process (Gemini Option C - DRY principle)
                resolved_paths_dict = resolved_paths or {}
                paths_str = repr(
                    resolved_paths_dict
                )  # {'key': '/full/path/to/file.json', ...}

                setup_code += f"""
# Selective workspace file loading with pre-resolved paths (v2.0)
import pandas as pd
import json

resolved_paths = {paths_str}
loaded_files = []

for key, file_path_str in resolved_paths.items():
    file_path = Path(file_path_str)

    if not file_path.exists():
        print(f"Warning: Resolved path does not exist: {{file_path}}")
        continue

    try:
        if file_path.suffix == '.json':
            with open(file_path) as f:
                content = json.load(f)

                # Always preserve full data structure (fixed: was breaking dict→list)
                file_size = file_path.stat().st_size
                globals()[key] = content  # ALWAYS preserve full structure
                if file_size > 100_000_000:
                    # Large file: log warning but preserve structure
                    sample_count = len(content.get('samples', [])) if isinstance(content, dict) else 'N/A'
                    print(f"Loaded {{key}} (large file: {{file_size // 1_000_000}}MB, samples: {{sample_count}})")
                else:
                    print(f"Loaded {{key}} from {{file_path.parent.name}}/{{file_path.name}}")

                loaded_files.append(f"{{key}} ({{file_path.parent.name}}/{{file_path.name}})")

        elif file_path.suffix == '.csv':
            globals()[key] = pd.read_csv(file_path)
            loaded_files.append(f"{{key}} ({{file_path.parent.name}}/{{file_path.name}})")
            print(f"Loaded {{key}}: {{len(globals()[key])}} rows from CSV")

    except Exception as e:
        print(f"Error loading {{key}} from {{file_path}}: {{e}}")

if loaded_files:
    print(f"✓ Loaded {{len(loaded_files)}} workspace keys successfully")
else:
    print(f"Warning: No workspace keys loaded (resolved_paths={{list(resolved_paths.keys())}})")

# Convenience imports
workspace_path = WORKSPACE
Path = Path

"""
            else:
                # Full loading: load all files (backward compatible)
                setup_code += """
# Auto-load data files from workspace with size guards
import os
import pandas as pd

# Size guard constants (configurable via env)
_AUTOLOAD_MAX_FILE_BYTES = int(os.getenv("LOBSTER_AUTOLOAD_MAX_FILE_BYTES", str(50 * 1024**2)))  # 50MB
_AUTOLOAD_MAX_TOTAL_BYTES = int(os.getenv("LOBSTER_AUTOLOAD_TOTAL_BYTES", str(200 * 1024**2)))  # 200MB
_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".txt", ".flatprottable"}
_SCAN_DIRS = ["data", "exports", "."]  # data first (uploads), exports, then root

def _infer_separator(path):
    ext = path.suffix.lower()
    if ext in (".tsv", ".flatprottable"):
        return "\\t"
    return ","

# Build workspace_files manifest + selective auto-load
workspace_files = []
autoloaded_bytes = 0
csv_data = {}
json_data = {}

for dir_name in _SCAN_DIRS:
    scan_dir = WORKSPACE if dir_name == "." else WORKSPACE / dir_name
    if not scan_dir.exists():
        continue
    for path in sorted(scan_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        if path.name.startswith('.'):
            continue
        rel = path.relative_to(WORKSPACE).as_posix()
        size = path.stat().st_size
        entry = {"path": rel, "name": path.stem, "ext": path.suffix.lower(),
                 "size": size, "loaded": False}

        if size <= _AUTOLOAD_MAX_FILE_BYTES and autoloaded_bytes + size <= _AUTOLOAD_MAX_TOTAL_BYTES:
            # Dir-prefixed variable name to prevent collision
            prefix = dir_name.replace(".", "") + "_" if dir_name != "." else ""
            var_name = prefix + path.stem.replace('-', '_').replace(' ', '_')
            var_name = ''.join(c for c in var_name if c.isalnum() or c == '_') or 'data'
            if var_name[0].isdigit():
                var_name = 'f_' + var_name

            try:
                if path.suffix.lower() == ".json":
                    with open(path) as f:
                        globals()[var_name] = json.load(f)
                    json_data[var_name] = rel
                else:
                    sep = _infer_separator(path)
                    globals()[var_name] = pd.read_csv(path, sep=sep)
                    csv_data[var_name] = rel

                autoloaded_bytes += size
                entry["loaded"] = True
                entry["variable"] = var_name
            except Exception as e:
                print(f"Warning: Could not load {rel}: {e}")
        else:
            if size > _AUTOLOAD_MAX_FILE_BYTES:
                print(f"Skipped {rel}: {size / (1024**2):.1f} MB exceeds per-file limit")

        workspace_files.append(entry)

# Load JSONL queue files
for queue_file in ['download_queue.jsonl', 'publication_queue.jsonl']:
    queue_path = WORKSPACE / queue_file
    if queue_path.exists():
        var_name = queue_file.replace('.jsonl', '').replace('-', '_')
        try:
            entries = []
            with open(queue_path) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            globals()[var_name] = entries
        except Exception as e:
            print(f"Warning: Could not load {queue_file}: {e}")

loaded_count = len(csv_data) + len(json_data)
if loaded_count:
    print(f"Loaded {loaded_count} workspace files: {len(csv_data)} tabular, {len(json_data)} JSON")
if workspace_files:
    skipped = sum(1 for f in workspace_files if not f.get("loaded"))
    if skipped:
        print(f"Manifest: {len(workspace_files)} files found, {skipped} skipped (size limits)")

# Convenience imports
workspace_path = WORKSPACE
Path = Path

"""

        setup_code += """
# === FIGURE SUBMISSION API (explicit submit only) ===
# submit_figure(fig, source=..., intent=..., parent=..., label=..., config=...) saves a figure
# AFTER this code finishes successfully. publish_plot(fig, title) is an alias. Nothing is
# captured automatically. list_figures()/load_figure(ref)/load_figure_config(ref)/
# load_source(id) read this attempt's snapshot.
import importlib.util as _lobster_ilu
_lobster_spec = _lobster_ilu.spec_from_file_location("_lobster_fig_child", __LOBSTER_CHILD_API_PATH__)
_lobster_child = _lobster_ilu.module_from_spec(_lobster_spec)
_lobster_spec.loader.exec_module(_lobster_child)
_lobster_child.install(globals(), EXEC_CACHE)
del _lobster_ilu, _lobster_spec, _lobster_child
# === END FIGURE SUBMISSION API ===

# User code starts here
""".replace("__LOBSTER_CHILD_API_PATH__", repr(str(_child_api_path())))
        return setup_code

    def _execute_in_namespace(
        self, code: str, context: Dict[str, Any]
    ) -> Tuple[Any, str, str, Optional[Exception]]:
        """
        Run a generated Python script in a subprocess with a timeout.

        Child and parent run as the same OS user; this is not a sandbox or a
        host-isolation boundary. Timeout and exit status are reported to the
        caller; screening and child-written completion files are cooperative.

        Args:
            code: Python code to execute
            context: Execution context (modality_name, workspace_path, etc. for setup)

        Returns:
            Tuple of (result, stdout, stderr, error)

        Example:
            >>> context = {'modality_name': 'test', 'workspace_path': Path('...'), ...}
            >>> result, stdout, stderr, error = service._execute_in_namespace(
            ...     "result = 2 + 2", context
            ... )
            >>> result
            4
        """
        # Generate context setup code
        modality_name = context.get("modality_name")
        workspace_path = context.get("workspace_path", self.data_manager.workspace_path)
        load_workspace_files = context.get("load_workspace_files", True)
        workspace_keys = context.get("workspace_keys")
        resolved_paths = context.get("resolved_paths", {})

        exec_cache_path = context.get("exec_cache_path")
        alias_map = context.get("alias_map", {})

        setup_code = self._generate_context_setup_code(
            modality_name=modality_name,
            workspace_path=workspace_path,
            load_workspace_files=load_workspace_files,
            workspace_keys=workspace_keys,
            resolved_paths=resolved_paths,
            exec_cache_path=exec_cache_path,
            alias_map=alias_map,
        )

        # Combine setup + user code + write-back + result extraction
        full_script = setup_code + "\n" + code + "\n\n"

        # Add adata write-back epilogue (detects modifications and saves to temp file)
        if modality_name:
            full_script += """
# === ADATA WRITE-BACK ===
if _lobster_fp is not None and adata is not None:
    try:
        _post_fp = {
            "shape": adata.shape,
            "obs_cols": sorted(adata.obs.columns.tolist()),
            "var_cols": sorted(adata.var.columns.tolist()),
            "obsm_keys": sorted(adata.obsm.keys()),
            "obsp_keys": sorted(adata.obsp.keys()),
            "uns_keys": sorted(adata.uns.keys()),
        }
        if _lobster_fp != _post_fp:
            import pandas as _wb_pd
            _wb_pd.options.future.infer_string = False
            adata.write_h5ad(EXEC_CACHE / '.modified_adata.h5ad')
            print("__LOBSTER_ADATA_MODIFIED__")
            # Report changes
            _ch = []
            if _lobster_fp['shape'] != _post_fp['shape']:
                _ch.append(f"shape {_lobster_fp['shape']}->{_post_fp['shape']}")
            for _k, _label in [('obs_cols','obs'), ('obsm_keys','obsm'), ('obsp_keys','obsp'), ('uns_keys','uns')]:
                _new = set(_post_fp[_k]) - set(_lobster_fp[_k])
                if _new: _ch.append(f"+{_label}: {_new}")
            if _ch: print(f"__LOBSTER_CHANGES__:{'; '.join(_ch)}")
    except Exception as _e:
        print(f"Warning: adata write-back failed: {_e}")

"""

        # Figure epilogue: NO automatic capture and no writes to plots/.
        # It only names unsubmitted Figure variables so the reply can say what was not saved.
        full_script += """
# === FIGURE EPILOGUE (warning only) ===
try:
    _lobster_figure_epilogue(dict(locals()))
except Exception as _fig_err:
    print(f"Warning: figure epilogue failed: {type(_fig_err).__name__}")
# === END FIGURE EPILOGUE ===

# Extract result and write the result envelope for the parent process. It is ALWAYS
# written (has_result=false for runs without a result), so a missing file is detectable.
import json
_lobster_env = {'schema': __LOBSTER_RESULT_SCHEMA__, 'has_result': False, 'result': None, 'type': None}
if 'result' in dir() and result is not None:
    try:
        json.dumps(result)
        _lobster_payload = result
    except (TypeError, ValueError):
        _lobster_payload = str(result)  # Result not JSON-serializable, save as string
    _lobster_env.update(has_result=True, result=_lobster_payload, type=type(result).__name__)
with open(EXEC_CACHE / '.execution_result.json', 'w') as _lobster_rf:
    json.dump(_lobster_env, _lobster_rf)

# === COMPLETION MARKER (figure fence; must stay the LAST statement) ===
(EXEC_CACHE / '.execution_complete').write_text(__LOBSTER_RUN_NONCE__)
"""

        # IPC dir must match EXEC_CACHE in the subprocess template.
        # When exec_cache_path is provided, subprocess uses it directly.
        # When not provided, subprocess defaults to WORKSPACE / 'cache' / 'execution'.
        if exec_cache_path:
            ipc_dir = exec_cache_path
        else:
            ipc_dir = workspace_path / "cache" / "execution"
        ipc_dir.mkdir(parents=True, exist_ok=True)
        script_path = ipc_dir / f".script_{uuid.uuid4().hex}.py"
        result_path = ipc_dir / ".execution_result.json"
        # Completion signal for cooperative early-exit detection,
        # not authentication: child and parent use the same OS user, and the child
        # can read/copy this nonce. Missing/mismatched markers or unreadable results
        # prevent figure ingestion, but matching files do not prove the normal
        # epilogue ran. A forged signal with a valid figure may save through the
        # parent path; validation, record-first ordering, digests and lifecycle/
        # attempt fences still apply there. This is not a tamper-resistance boundary.
        complete_path = ipc_dir / ".execution_complete"
        run_nonce = uuid.uuid4().hex
        head, _, tail = full_script.rpartition("__LOBSTER_RUN_NONCE__")  # marker only
        full_script = head + repr(run_nonce) + tail
        head, _, tail = full_script.rpartition("__LOBSTER_RESULT_SCHEMA__")
        full_script = head + repr(RESULT_SCHEMA) + tail
        context["completion"] = COMPLETION_UNCONFIRMED
        for stale in (result_path, complete_path):
            stale.unlink(missing_ok=True)

        try:
            script_path.write_text(full_script)

            # Execute in subprocess with timeout (from context)
            timeout_seconds = context.get("timeout", DEFAULT_TIMEOUT)
            logger.debug(f"Executing code in subprocess (timeout={timeout_seconds}s)")

            # Preserve the established child environment policy shared by tools.
            safe_env = minimal_child_env()

            proc_result = subprocess.run(  # nosec B603 # Fixed argv (current interpreter + generated script path); no shell, no caller-supplied command.
                [sys.executable, str(script_path)],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=safe_env,  # SECURITY: Filtered environment only
            )

            stdout_output = proc_result.stdout
            stderr_output = proc_result.stderr
            return_code = proc_result.returncode

            # Truncate output if needed
            if len(stdout_output) > MAX_OUTPUT_LENGTH:
                stdout_output = (
                    stdout_output[:MAX_OUTPUT_LENGTH]
                    + "\n\n... [STDOUT TRUNCATED at 10,000 chars. "
                    "DO NOT re-execute to see the rest — it will truncate again. "
                    "Save large outputs to OUTPUT_DIR / 'results.json' instead of printing them.]"
                )
                logger.warning(f"Stdout truncated at {MAX_OUTPUT_LENGTH} characters")

            if len(stderr_output) > MAX_OUTPUT_LENGTH:
                stderr_output = (
                    stderr_output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
                )
                logger.warning(f"Stderr truncated at {MAX_OUTPUT_LENGTH} characters")

            # Extract result if available
            result = None
            error = None

            if return_code != 0:
                # Execution failed - append exit code to stderr for LLM visibility
                stderr_output = f"{stderr_output.rstrip()}\n\nExit code: {return_code}"
                error = Exception(
                    f"Code execution failed with return code {return_code}"
                )
            else:
                # Load the result envelope (always written by a normally completed run).
                result, completion = _read_result_envelope(result_path)
                try:
                    marker = read_regular(complete_path, MAX_MARKER_BYTES)
                except RefusedRead:
                    marker = None
                marker_ok = marker == run_nonce.encode("ascii")
                if not marker_ok:
                    completion = COMPLETION_UNCONFIRMED  # early exit takes precedence
                context["completion"] = completion

            return result, stdout_output, stderr_output, error

        except subprocess.TimeoutExpired:
            error = Exception(f"Code execution exceeded {DEFAULT_TIMEOUT}s timeout")
            return None, "", f"Execution timeout after {DEFAULT_TIMEOUT}s", error

        except Exception as e:
            logger.error(f"Subprocess execution error: {e}")
            error = e
            return None, "", f"Subprocess error: {e}", error

        finally:
            # Clean up temporary files. Best effort: whatever the code left at these paths,
            # cleanup never replaces the outcome decided above.
            for leftover in (result_path, complete_path):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            if error is not None and os.path.lexists(script_path):
                # Archive the failed script for debugging: a rename into a real directory
                # only, never a copy of a file the code could have changed.
                archive_dir = workspace_path / ".failed_scripts"
                archive_path = archive_dir / script_path.name
                try:
                    archive_dir.mkdir(exist_ok=True)
                    if not _stat.S_ISDIR(os.lstat(archive_dir).st_mode):
                        raise NotADirectoryError(archive_dir.name)
                    os.rename(script_path, archive_path)
                    logger.debug(f"Archived failed script to {archive_path}")
                except OSError:
                    logger.debug("Failed script not archived")
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _create_ir(
        self,
        code: str,
        description: str,
        modality_name: Optional[str],
        load_workspace_files: bool,
        workspace_keys: Optional[List[str]],
        persist: bool,
        stats: Dict[str, Any],
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance and notebook export.

        Args:
            code: Executed Python code
            description: Human-readable description
            modality_name: Loaded modality name (if any)
            load_workspace_files: Whether workspace files were loaded
            workspace_keys: Specific workspace keys loaded (if selective loading)
            persist: Whether to include in notebook export
            stats: Execution statistics

        Returns:
            AnalysisStep object

        Example:
            >>> ir = service._create_ir(
            ...     code="result = 2 + 2",
            ...     description="Simple addition",
            ...     modality_name=None,
            ...     load_workspace_files=False,
            ...     workspace_keys=None,
            ...     persist=False,
            ...     stats={'success': True, 'duration_seconds': 0.001}
            ... )
            >>> ir.operation
            'custom_code_execution'
            >>> ir.exportable
            False
        """
        # Extract imports from code
        imports = self._extract_imports(code)
        uses_helpers = _uses_figure_helpers(code)

        # Build workspace setup code for notebook export
        # This ensures WORKSPACE and OUTPUT_DIR are defined when notebook runs
        workspace_setup = """# Workspace setup (for notebook execution)
from pathlib import Path
import sys

WORKSPACE = Path({{ workspace_path | string | repr }})
sys.path.append(str(WORKSPACE))

# OUTPUT_DIR: Recommended directory for CSV/TSV/Excel exports
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""

        # Combine workspace setup with user code for notebook export
        # Uses Jinja2 template syntax for workspace_path parameter injection
        code_template = f"{workspace_setup}\n# User code\n{code}"

        # Get workspace path for parameter default
        workspace_path_str = str(self.data_manager.workspace_path)

        # Parameter schema (for Papermill injection if exported)
        parameter_schema = {
            "workspace_path": ParameterSpec(
                param_type="str",
                papermill_injectable=True,  # Can override workspace location
                default_value=workspace_path_str,
                required=True,
                description="Path to Lobster workspace directory",
            ),
            "code": ParameterSpec(
                param_type="str",
                papermill_injectable=False,  # Code should not be overridden
                default_value=code,
                required=True,
                description="Python code to execute",
            ),
            "description": ParameterSpec(
                param_type="str",
                papermill_injectable=False,
                default_value=description,
                required=False,
                description="Human-readable description",
            ),
        }

        # Add modality_name parameter if used
        if modality_name:
            parameter_schema["modality_name"] = ParameterSpec(
                param_type="str",
                papermill_injectable=True,  # Can override which modality to use
                default_value=modality_name,
                required=False,
                description="Modality to load as adata",
            )

        return AnalysisStep(
            operation="custom_code_execution",
            tool_name="execute_custom_code",
            description=description,
            library="custom",
            code_template=code_template,
            imports=imports,
            parameters={
                "workspace_path": workspace_path_str,  # For Jinja2 template rendering
                "code": code,
                "description": description,
                "modality_name": modality_name,
                "load_workspace_files": load_workspace_files,
                "workspace_keys": workspace_keys,
                "selective_loading": workspace_keys is not None,
            },
            parameter_schema=parameter_schema,
            input_entities=[modality_name] if modality_name else [],
            output_entities=[],  # Custom code may not produce named outputs
            execution_context={
                "persist": persist,
                "success": stats["success"],
                "duration": stats["duration_seconds"],
                "warnings": stats["warnings"],
                "selective_loading": workspace_keys is not None,
                "workspace_keys_count": len(workspace_keys) if workspace_keys else 0,
                **(
                    {"replay_excluded": REPLAY_EXCLUDED_FIGURE_HELPERS}
                    if uses_helpers
                    else {}
                ),
            },
            validates_on_export=True,
            requires_validation=True,
            # Only include in notebook if persist=True, and never a cell that needs the
            # attempt-only figure helpers (see FIGURE_HELPER_NAMES).
            exportable=persist and not uses_helpers,
            replayable=not uses_helpers,
        )

    def _extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from code using AST.

        Args:
            code: Python code

        Returns:
            List of import statements as strings

        Example:
            >>> imports = service._extract_imports("import numpy as np\\nfrom pandas import DataFrame")
            >>> imports
            ['import numpy as np', 'from pandas import DataFrame']
        """
        imports = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            imports.append(f"import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"import {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in node.names
                    )
                    imports.append(f"from {module} import {names}")

        except SyntaxError:
            # If code has syntax errors, return empty (validation will catch it)
            pass

        return imports
