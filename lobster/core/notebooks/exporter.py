"""
Jupyter Notebook Exporter for Pipeline Replay.

This module converts Lobster provenance records into executable Jupyter notebooks,
so an analysis workflow can be read and re-run with standard tools.

The exporter uses Service-Emitted Intermediate Representation (IR) to automatically
generate code without manual mapping registries, achieving 95%+ executable notebooks.
"""

import ast
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance.analysis_ir import (
    AnalysisStep,
    extract_unique_helper_code,
    extract_unique_imports,
)
from lobster.core.provenance.provenance import ProvenanceTracker
from lobster.core.session_identity import display_client_session_id
from lobster.i18n import h

logger = logging.getLogger(__name__)

# Default for helpers called outside export(): take their own identity reading.
_OBSERVE: Any = object()


# Printed by the save cell when it cannot import the session writer; the replay check
# looks for exactly this line in the executed notebook.
SESSION_WRITER_FALLBACK_LINE = (
    "Written without the session writer (lobster not importable); "
    "the saved file may differ from the session's."
)


MIGRATED_RECORD_LINE = (
    "- `migrated`: an earlier version of Lobster wrote the session record. `lobster` and "
    "`software_versions` are that version's; `system` is the version that exported this "
    "notebook. The input file the session read was not recorded.\n"
)


_COMMIT = re.compile(r"[0-9a-f]{40}")


def _installed_commit() -> str:
    """Short commit of the installed lobster-ai, or "unavailable".

    Only a git install records one (PEP 610 ``direct_url.json``,
    ``vcs_info.commit_id``); a wheel, an editable or folder install, or an
    unreadable file has none. Never the current directory's repository, which may
    be the user's own project. Never raises: an export must not fail on this.
    """
    try:
        direct_url = metadata.distribution("lobster-ai").read_text("direct_url.json")
        commit = json.loads(direct_url)["vcs_info"]["commit_id"]
    except Exception:
        return "unavailable"
    if isinstance(commit, str) and _COMMIT.fullmatch(commit):
        return commit[:8]
    return "unavailable"


def _inside(path: Path, folder: Path) -> bool:
    """Whether ``path`` is in ``folder``, by its name and where it really is."""
    try:
        path.relative_to(folder)
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def _guarded_save_code(path_line: str, compression: Optional[str] = None) -> str:
    """Code for the final save cell.

    The cell writes through the function the session saves with
    (``write_session_h5ad``), so the notebook writes what the session wrote. Only if
    Lobster cannot be imported does it fall back to a plain write; it says so, and
    if h5ad cannot write some uns entries it writes a copy without them, names them
    in ``uns['lobster_unsaved_uns_keys']`` and leaves the in-memory adata as is.
    """
    kwargs = f", compression={compression!r}" if compression else ""
    return f"""# Save processed data
{path_line}
try:
    from lobster.core.backends.h5ad_backend import write_session_h5ad
except ImportError:
    print({SESSION_WRITER_FALLBACK_LINE!r})
    try:
        adata.write_h5ad(output_path{kwargs})
    except Exception as exc:
        # Some uns entries cannot be written to h5ad. Write a copy without them
        # and record their names in that copy; the in-memory adata keeps them.
        import tempfile
        from pathlib import Path as _Path

        _unwritable = []
        with tempfile.TemporaryDirectory() as _tmp:
            for _key in list(adata.uns):
                _probe = adata.copy()
                _probe.uns = {{_key: adata.uns[_key]}}
                try:
                    _probe.write_h5ad(_Path(_tmp) / "probe.h5ad")
                except Exception:
                    _unwritable.append(_key)
        _to_write = adata.copy()
        for _key in _unwritable:
            del _to_write.uns[_key]
        _to_write.uns["lobster_unsaved_uns_keys"] = list(_unwritable)
        print(f"WARNING: these uns keys could not be written and are NOT in {{output_path}}: {{_unwritable}}")
        print(f"         original error: {{type(exc).__name__}}: {{exc}}")
        _to_write.write_h5ad(output_path{kwargs})
else:
    write_session_h5ad(adata, output_path{kwargs})
print(f"Saved processed data to: {{output_path}}")
"""


def _keep_cell(
    names: List[str], first: bool, sources: Optional[Dict[str, str]] = None
) -> NotebookNode:
    """Keep a copy of each named modality: a later step starts from it, or the
    session saved it before later steps changed it. ``sources`` names the
    variable a step left a modality in, when not ``adata``."""
    sources = sources or {}
    lines = ["# Keep a copy for a later step that starts from it, or for the save."]
    if first:
        lines.append("_kept_modalities = {}")
    lines += [
        f"_kept_modalities[{name!r}] = {sources.get(name, 'adata')}.copy()"
        for name in names
    ]
    return new_code_cell("\n".join(lines))


def _rebind_cell(name: str) -> NotebookNode:
    """Start the next step from the modality the session gave it."""
    return new_code_cell(
        f"# The session ran this step on {name!r}, not on the result above.\n"
        f"adata = _kept_modalities[{name!r}].copy()"
    )


# Raised in a custom-code cell for a read the notebook cannot serve from its own
# results; replay names its message.
CUSTOM_READ_ERROR = "LobsterReplayReadError"

# Defined once, before the first custom-code cell.
CUSTOM_READS_HELPER = f'''# Custom code reads this notebook's own results, as the session gave it the
# modalities in memory. In each custom cell the anndata, scanpy and h5py readers
# are pointed at this notebook's object for each modality the cell reads, written
# to a fresh file of that cell on its first read (one write per name per cell). Run
# anywhere but the session's workspace, a read of any other file in that workspace
# stops the run. Not guarded: pandas, open() or numpy on a typed path, and readers
# other than anndata, scanpy and h5py.
import os as _lobster_os
from pathlib import Path as _LobsterPath


class {CUSTOM_READ_ERROR}(RuntimeError):
    pass


_lobster_reading = {{}}


def _lobster_target(path):
    """The file a custom cell's read of ``path`` opens."""
    if not _lobster_reading or not isinstance(path, (str, _lobster_os.PathLike)):
        return path
    text = _lobster_os.fspath(path)
    aliases = _lobster_reading["aliases"]
    for key in (text, _lobster_os.path.normpath(text), _lobster_os.path.basename(text)):
        if key in aliases:
            return _lobster_serve(*aliases[key])
    real = _LobsterPath(text).resolve()
    here = _lobster_reading["here"]  # the run's own folder may be inside the session's
    if real.suffix.lower() == ".h5ad" and real.parent.name == "data" and not real.exists():
        if real.parent.parent in (here, _lobster_reading["session"]):
            _lobster_serve("missing", real.stem)
    guarded = _lobster_reading["guarded"]
    if guarded is not None:
        if (
            str(real) != _lobster_reading["input"]
            and (real == guarded or guarded in real.parents)
            and not (real == here or here in real.parents)
        ):
            raise {CUSTOM_READ_ERROR}(
                f"custom code reads {{real.relative_to(guarded).as_posix()}} of the "
                "session's workspace; a replay reads only this notebook's results"
            )
    return path


def _lobster_serve(kind, value, name=None):
    """The file for one aliased read: the input's path, or this notebook's object
    for the name written to this cell's folder on its first read."""
    if kind == "missing":
        raise {CUSTOM_READ_ERROR}(
            f"custom code reads {{value!r}}, which this notebook does not compute"
        )
    if kind == "file":
        return value
    target = _lobster_reading["folder"] / f"{{name}}.h5ad"
    if not target.exists():
        _lobster_reading["folder"].mkdir(parents=True, exist_ok=True)
        try:
            from lobster.core.backends.h5ad_backend import write_session_h5ad
        except ImportError:
            value.write_h5ad(target)
        else:
            write_session_h5ad(value, target)
    return str(target)


def _lobster_patch_readers():
    try:
        import anndata as _ad

        _read_h5ad = _ad.read_h5ad

        def _aliased_read_h5ad(path, *args, **kwargs):
            return _read_h5ad(_lobster_target(path), *args, **kwargs)

        _ad.read_h5ad = _aliased_read_h5ad
    except ImportError:
        _aliased_read_h5ad = None
    try:
        import scanpy as _sc

        if _aliased_read_h5ad is not None:
            _sc.read_h5ad = _aliased_read_h5ad
        _sc_read = _sc.read

        def _aliased_sc_read(path, *args, **kwargs):
            return _sc_read(_lobster_target(path), *args, **kwargs)

        _sc.read = _aliased_sc_read
    except ImportError:
        pass
    try:
        import h5py as _h5py

        _h5py_init = _h5py.File.__init__

        def _aliased_h5py_init(self, name, mode="r", *args, **kwargs):
            if mode in ("r", "r+"):
                name = _lobster_target(name)
            return _h5py_init(self, name, mode, *args, **kwargs)

        _h5py.File.__init__ = _aliased_h5py_init
    except (ImportError, AttributeError):
        pass


_lobster_patch_readers()


def _lobster_reads(cell, session_workspace, sources, missing, runs_on=None):
    """Point the next custom cell's reads at this notebook's results.

    ``sources`` maps each modality the cell reads to this notebook's object for
    it, or to the input file's path; ``missing`` names those it cannot compute.
    An object is written to this cell's own folder when first read.
    """
    if runs_on is not None:
        raise {CUSTOM_READ_ERROR}(
            f"custom code runs on {{runs_on!r}}, which this notebook does not compute"
        )
    folder = WORKSPACE / ".lobster_replay" / str(cell)
    roots = {{str(WORKSPACE), str(session_workspace), "/tmp/workspace"}}

    def keys(name):
        found = {{f"data/{{name}}.h5ad", f"{{name}}.h5ad"}}
        for root in roots:
            found |= {{f"{{root}}/data/{{name}}.h5ad", f"{{root}}/modalities/{{name}}.h5ad"}}
        return found

    aliases = {{}}
    for name, source in sources.items():
        if isinstance(source, (str, _lobster_os.PathLike)):
            alias = ("file", str(_LobsterPath(source).resolve()))
        else:
            alias = ("object", source, name)
        aliases.update((key, alias) for key in keys(name))
    for name in missing:
        aliases.update((key, ("missing", name)) for key in keys(name))
    elsewhere = _LobsterPath(WORKSPACE).resolve() != _LobsterPath(session_workspace).resolve()
    _lobster_reading.update(
        aliases=aliases,
        folder=folder,
        guarded=_LobsterPath(session_workspace).resolve() if elsewhere else None,
        here=_LobsterPath(WORKSPACE).resolve(),
        session=_LobsterPath(session_workspace).resolve(),
        input=str(_LobsterPath(input_data).resolve()),
    )


def _lobster_reads_done():
    """After a custom cell: its readers are plain again, its copies removed (a
    replay counts any data file left in its folder)."""
    import shutil

    folder = _lobster_reading.get("folder")
    _lobster_reading.clear()
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)
        try:
            folder.parent.rmdir()  # only when no other cell's copies are left
        except OSError:
            pass
'''


def _custom_code_cell(
    ir: AnalysisStep,
    position: int,
    plan: Dict[str, str],
    runs_on: Optional[str],
    written_back: Optional[str],
) -> NotebookNode:
    """A custom-code cell that runs on this notebook's results.

    ``WORKSPACE`` is the ``workspace_path`` parameter (a replay sets its own
    folder), and each modality the code reads is this notebook's object for it
    (``plan``: name -> ``adata``, ``kept``, ``input`` or ``missing``). As in the
    session, the code runs on a copy of ``adata``: the changed copy is kept as
    ``written_back`` (the name the session stored it under), ``adata`` is left as
    it was.
    """
    session_workspace = str(ir.parameters.get("workspace_path") or "")
    sources = {
        name: {
            "adata": "adata",
            "kept": f"_kept_modalities[{name!r}]",
            "input": "input_data",
        }[kind]
        for name, kind in plan.items()
        if kind != "missing"
    }
    missing = [name for name, kind in plan.items() if kind == "missing"]
    source_map = "{" + ", ".join(f"{k!r}: {v}" for k, v in sources.items()) + "}"
    keep_result = (
        "if '_kept_modalities' not in globals():\n"
        "    _kept_modalities = {}\n"
        f"_kept_modalities[{written_back!r}] = adata  # the session stored it so\n"
        if written_back
        else ""
    )
    code = f"""# Workspace setup (a replay runs in its own folder)
from pathlib import Path
import sys

WORKSPACE = Path(workspace_path)
sys.path.append(str(WORKSPACE))

# OUTPUT_DIR: Recommended directory for CSV/TSV/Excel exports
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(WORKSPACE / 'data').mkdir(exist_ok=True)
_lobster_reads({position}, {session_workspace!r}, {source_map}, {missing!r}, runs_on={runs_on!r})

_lobster_held, adata = adata, (adata.copy() if adata is not None else None)

# User code
{ir.parameters.get("code", "")}

_lobster_reads_done()
{keep_result}adata = _lobster_held
"""
    return new_code_cell(code)


# Cell metadata (under "lobster") naming the step of a cell that draws a figure.
FIGURE_CELL_KEY = "figure"
# Cell metadata (under "lobster") on the cell that stands in for custom code left
# out: {"step": <step number>, "reason": "unrecognised_record"}. Replay refuses it.
OMITTED_CUSTOM_CODE_KEY = "omitted_custom_code"
# The same mark's reason on the cell that stands in for a step recorded with a
# template this Lobster does not write for its tool (written_templates); the
# export record lists those steps under OUTDATED_STEPS_KEY.
OUTDATED_TEMPLATE_REASON = "outdated_template"
OUTDATED_STEPS_KEY = "outdated_steps"
UNRECORDED_INPUT_NOTE = (
    "*Input not recorded: this step runs on the result of the cells above.*"
)


class NotebookExporter:
    """
    Convert Lobster provenance to executable Jupyter notebook using IR.

    This class transforms provenance tracking data with embedded Intermediate
    Representation (IR) into analysis pipelines using standard
    Jupyter notebook format compatible with Papermill for parameterized execution.

    The key innovation is that services emit AnalysisStep IR objects that contain
    complete code generation instructions, eliminating the need for manual mapping
    registries like TOOL_TO_CODE.

    Attributes:
        provenance: ProvenanceTracker instance with analysis history
        data_manager: DataManagerV2 instance for data context
    """

    def __init__(
        self, provenance: ProvenanceTracker, data_manager: DataManagerV2
    ) -> None:
        """
        Initialize notebook exporter.

        Args:
            provenance: ProvenanceTracker with recorded activities
            data_manager: DataManagerV2 for workspace context
        """
        self.provenance = provenance
        self.data_manager = data_manager
        # Set by each export of a migrated session: links in data/ that point nowhere
        self.dangling_links: List[str] = []
        # Set by each export: the steps whose custom code was left out
        self.omitted_custom_code: List[int] = []
        # Set by each export: the steps left out for a template it does not write
        self.outdated_steps: List[int] = []
        logger.debug("Initialized NotebookExporter with IR support")

    def _migrated_origin(self) -> Optional[Dict[str, Any]]:
        """Where the session record came from when an earlier version wrote it."""
        origin = getattr(self.provenance, "migrated_from", lambda: None)()
        return origin if isinstance(origin, dict) else None

    @staticmethod
    def _source_words(origin: Dict[str, Any]) -> str:
        version = str(origin.get("lobster") or "").split("+")[0]
        if not version:
            return h("Lobster session of unknown version")
        return h("Lobster {version} session").format(version=version)

    def _find_dangling_links(self) -> List[str]:
        """Links in the workspace's ``data/`` folder whose target is missing."""
        workspace = Path(self.data_manager.workspace_path)
        found = []
        try:
            for path in sorted((workspace / "data").rglob("*")):
                if path.is_symlink() and not path.exists():
                    found.append(path.relative_to(workspace).as_posix())
        except OSError:
            pass
        return found

    @staticmethod
    def _is_custom_code(activity: Dict[str, Any], ir: AnalysisStep) -> bool:
        return (
            activity.get("type") in NotebookExporter._EXCLUDED_STEP_TYPES
            or ir.operation in NotebookExporter._EXCLUDED_STEP_TYPES
        )

    def _unrun_cell(
        self, ir: AnalysisStep, origin: Dict[str, Any], step: int
    ) -> NotebookNode:
        """Custom code of an earlier version: shown, never a code cell."""
        from lobster.services.execution.custom_code_execution_service import (
            recorded_custom_code,
        )

        code = recorded_custom_code(ir)
        if code is None:
            label = h(
                "Not shown: custom code from a {source} whose record has a form "
                "Lobster did not write"
            ).format(source=self._source_words(origin))
            cell = new_markdown_cell(f"**{label}**\n")
            cell.metadata["lobster"] = {
                OMITTED_CUSTOM_CODE_KEY: {"step": step, "reason": "unrecognised_record"}
            }
            return cell
        longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
        fence = "`" * max(3, longest + 1)
        label = h(
            "Not executed: custom code from a {source} "
            "(it may use paths of the machine it ran on)"
        ).format(source=self._source_words(origin))
        return new_markdown_cell(f"**{label}**\n\n{fence}python\n{code}\n{fence}\n")

    @staticmethod
    def _outdated_cell(tool: str, step: int) -> NotebookNode:
        """In place of a step whose stored template could run its values: named,
        never shown."""
        label = h(
            "Not shown: a {tool} step recorded with code this Lobster no longer "
            "writes (run the step again to include it)"
        ).format(tool=tool)
        cell = new_markdown_cell(f"**{label}**\n")
        cell.metadata["lobster"] = {
            OMITTED_CUSTOM_CODE_KEY: {"step": step, "reason": OUTDATED_TEMPLATE_REASON}
        }
        return cell

    def export(
        self,
        name: str,
        description: str = "",
        filter_strategy: str = "successful",
        validate_syntax: bool = True,
    ) -> Path:
        """
        Generate Jupyter notebook from current session using IR.

        Args:
            name: Notebook filename (without extension)
            description: Human-readable description for header
            filter_strategy: Activity filter
                ("successful" | "all" | "manual" | "replay").
                The first three curate — they select IR marked ``exportable``,
                which services commonly derive from a ``persist`` flag, so the
                result is a notebook a user could publish. ``"replay"`` instead
                selects IR marked ``replayable``: the recorded steps marked
                replayable, including ad-hoc computations and state mutations
                that curation drops; steps that cannot be replayed are listed,
                and a step that changed data but cannot be replayed makes it
                refuse. Use it when the notebook is meant to re-run the
                session's steps rather than to be published. The default is
                unchanged.
            validate_syntax: Whether to validate generated Python syntax

        Returns:
            Path to generated .ipynb file

        Raises:
            ValueError: If no activities to export or name is invalid
            RuntimeError: If IR extraction fails or code generation invalid
        """
        if not name or not name.strip():
            raise ValueError("Notebook name cannot be empty")

        if not self.provenance.activities:
            raise ValueError("No activities recorded - nothing to export")

        # One reading of the client session identity for the whole notebook, so the
        # export record, its hash, the summary and the metadata always agree.
        session_identity = self._client_session_identity()

        logger.info(f"Exporting notebook with IR: {name}")

        # A failed step may have changed data before it failed. A replay
        # cannot assume it did not, so "replay" refuses instead of dropping it.
        failed_types = [
            self._omitted_label(a)
            for a in self.provenance.activities
            if self._is_failed(a)
        ]
        if filter_strategy == "replay" and failed_types:
            raise ValueError(
                "Cannot export a replay notebook: the session has failed steps "
                f"({', '.join(dict.fromkeys(failed_types))}) that may have changed "
                "data before failing. Re-run the analysis without them, or export "
                "with the default mode."
            )

        # A step that changed data but cannot be replayed says so in its IR
        # (``replay_incomplete``). A replay would silently miss that change, so
        # "replay" refuses instead of dropping it.
        # The message is literal: step metadata never reaches it.
        incomplete = any(
            not self._ir_field(a, "replayable", True)
            and self._ir_context(a).get("replay_incomplete") is True
            for a in self.provenance.activities
        )
        if filter_strategy == "replay" and incomplete:
            raise ValueError(
                "Cannot export a replay notebook: some steps changed data but cannot "
                "be replayed. Export with the default mode, or re-run the analysis "
                "without them."
            )

        # Filter activities by success status
        activities = self._filter_activities(filter_strategy)
        logger.debug(f"Filtered {len(activities)} activities for export")

        # Extract the (activity, IR) pairs this strategy selects.
        # Under the default strategies this selects on `exportable`, filtering out
        # orchestration activities (publication processing, etc.) that belong in
        # provenance but not in a curated notebook. Under "replay" it selects on
        # `replayable` instead, keeping the steps marked replayable.
        select_on = "replayable" if filter_strategy == "replay" else "exportable"
        exportable_pairs = self._get_exportable_activity_ir_pairs(
            activities, select_on=select_on
        )
        exportable_count = len(exportable_pairs)
        filtered_count = len(activities) - exportable_count
        origin = self._migrated_origin()
        self.dangling_links = self._find_dangling_links() if origin else []
        unrun = {
            id(activity)
            for activity, ir in exportable_pairs
            if origin and self._is_custom_code(activity, ir)
        }
        from lobster.services.execution.custom_code_execution_service import (
            recorded_custom_code,
        )

        self.omitted_custom_code = [
            idx
            for idx, (activity, ir) in enumerate(exportable_pairs, start=1)
            if id(activity) in unrun and recorded_custom_code(ir) is None
        ]
        from lobster.core.notebooks.written_templates import outdated_tool

        outdated = {
            id(activity): tool
            for activity, ir in exportable_pairs
            if (tool := outdated_tool(ir)) is not None
        }
        self.outdated_steps = [
            idx
            for idx, (activity, _) in enumerate(exportable_pairs, start=1)
            if id(activity) in outdated
        ]
        exportable_activity_ids = {id(activity) for activity, _ in exportable_pairs}
        excluded_activity_types = [
            self._omitted_label(activity)
            for activity in activities
            if id(activity) not in exportable_activity_ids
        ]

        logger.info(
            f"Extracted {exportable_count} {select_on} IR objects "
            f"from {len(activities)} activities "
            f"({filtered_count} provenance-only activities filtered)"
        )

        # Extract IRs for imports and parameters collection
        irs = [
            ir
            for activity, ir in exportable_pairs
            if id(activity) not in unrun and id(activity) not in outdated
        ]

        if exportable_count == 0:
            raise ValueError(
                f"No {select_on} IR objects found in provenance. "
                f"Found {len(activities)} activities, but none have "
                f"{select_on}=True IR. "
                f"Services need to emit AnalysisStep objects with {select_on}=True."
            )

        # Create new notebook
        notebook = new_notebook()

        # Add header
        notebook.cells.append(
            self._create_header_cell(
                name,
                description,
                exportable_count,
                mode=filter_strategy,
                omitted_types=failed_types + excluded_activity_types,
            )
        )

        # Add the export record (hashes taken at export time; not verified at run time)
        notebook.cells.append(self._create_integrity_manifest_cell(session_identity))

        # Collect and add imports from all exportable IRs
        imports_cell = self._create_imports_cell(irs)
        notebook.cells.append(imports_cell)

        # Add helper functions cell (shared across analysis steps)
        helper_cell = self._create_helper_code_cell(irs)
        if helper_cell:
            notebook.cells.append(helper_cell)

        # Parameters cell (Papermill-tagged). Filled in once every code cell
        # exists, so it offers only names the notebook code actually reads.
        parameters_index = len(notebook.cells)
        notebook.cells.append(new_code_cell(""))

        # Initialize ML bindings once per run, on mutually exclusive input paths.
        # Self-loading roots supply their own load but still need a run boundary.
        if self._has_download_activity(exportable_pairs):
            notebook.cells.append(self._create_ml_run_reset_cell())
        else:
            notebook.cells.append(self._create_data_loading_cell())

        # Convert exportable (activity, IR) pairs to code cells
        # This iterates over ONLY exportable IRs, not all activities
        # Skip load_dataset and save_dataset since they have dedicated cells above/below
        steps = [
            (idx, activity, ir)
            for idx, (activity, ir) in enumerate(exportable_pairs, start=1)
            if activity.get("type", "") not in ("load_dataset", "save_dataset")
        ]
        saves = self._saves(irs)
        custom = {
            position: self._custom_reads(activity, ir)
            for position, (_, activity, ir) in enumerate(steps)
            if self._is_custom_code(activity, ir)
            and id(activity) not in unrun
            and isinstance(ir.parameters.get("code"), str)  # else rendered as recorded
        }
        keep, rebind, save_at, read_plans, keep_vars = self._lineage(
            [activity for _, activity, _ in steps], list(saves), custom
        )
        placed = set()
        kept_any = False
        custom_helper = False

        def after(position: int) -> None:
            """Write each modality the session saved where it was made, then keep
            the copies later steps start from."""
            nonlocal kept_any
            written_back = custom[position][2] if position in custom else None
            for name in save_at.get(position, []):
                cell = self._save_cell(saves[name])
                if cell is not None and name == written_back:
                    # the save cell writes adata: the custom cell's result, here
                    notebook.cells.append(
                        new_code_cell(
                            "_lobster_saving = adata\n"
                            f"adata = _kept_modalities[{name!r}]"
                        )
                    )
                    notebook.cells.append(cell)
                    notebook.cells.append(new_code_cell("adata = _lobster_saving"))
                    placed.add(name)
                elif cell is not None:
                    notebook.cells.append(cell)
                    placed.add(name)
            if position in keep:
                notebook.cells.append(
                    _keep_cell(
                        keep[position],
                        first=not kept_any,
                        sources=keep_vars.get(position),
                    )
                )
                kept_any = True

        after(-1)
        for position, (idx, activity, ir) in enumerate(steps):
            # Add documentation cell
            doc_cell = self._create_doc_cell(activity, idx)
            if (keep or rebind) and self._reads_data_unnamed(activity, ir):
                doc_cell.source += f"\n\n{UNRECORDED_INPUT_NOTE}"
            notebook.cells.append(doc_cell)

            if position in rebind:
                notebook.cells.append(_rebind_cell(rebind[position]))

            # Add executable code cell from IR (no placeholder fallback)
            if id(activity) in outdated:
                notebook.cells.append(self._outdated_cell(outdated[id(activity)], idx))
                after(position)
                continue
            if id(activity) in unrun:
                notebook.cells.append(self._unrun_cell(ir, origin, idx))
                after(position)
                continue
            if position in custom:
                if not custom_helper:
                    notebook.cells.append(new_code_cell(CUSTOM_READS_HELPER))
                    custom_helper = True
                runs_on, _, written_back = custom[position]
                code_cell = _custom_code_cell(
                    ir,
                    position,
                    read_plans[position],
                    runs_on if runs_on in read_plans[-1] else None,
                    written_back,
                )
                kept_any = kept_any or written_back is not None
            else:
                code_cell = self._ir_to_code_cell(ir, validate=validate_syntax)
            if "fig" in (ir.output_entities or []):
                # replay names each figure it drew without comparing it
                label = f"Step {idx}: {self._step_label(activity)}"
                code_cell.metadata["lobster"] = {FIGURE_CELL_KEY: label}
            notebook.cells.append(code_cell)

            after(position)

        # The closing save cell, unless its modality was written where it was
        # made. A save the steps never made stays unwritten unless it is this
        # one, and replay reports it as not produced.
        closing = self._closing_save(irs)
        if closing is None or closing.get("parameters", {}).get("name") not in placed:
            notebook.cells.append(self._create_data_saving_cell(irs))

        read_names = self._names_read(
            cell
            for index, cell in enumerate(notebook.cells)
            if index != parameters_index
        )
        notebook.cells[parameters_index] = self._create_parameters_cell(
            irs, read_names=read_names
        )

        # Add provenance summary cell (explains what's in notebook vs provenance)
        notebook.cells.append(
            self._create_provenance_summary_cell(
                len(activities),
                exportable_count,
                excluded_activity_types,
                select_on=select_on,
                session_identity=session_identity,
            )
        )

        # Add footer cell
        notebook.cells.append(
            self._create_footer_cell(
                excluded_activity_types,
                select_on=select_on,
                failed_types=failed_types,
            )
        )

        # Add notebook metadata
        notebook.metadata["lobster"] = self._create_metadata(
            exportable_count, len(activities), session_identity=session_identity
        )
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        notebook.metadata["language_info"] = {
            "name": "python",
            "pygments_lexer": "ipython3",
        }

        # Save to workspace notebooks directory
        output_dir = self.data_manager.workspace_path / "notebooks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.ipynb"

        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        logger.info(f"Notebook exported: {output_path}")
        logger.info(
            f"Executable steps: {exportable_count} | "
            f"Provenance-only: {filtered_count} | "
            f"Total activities: {len(activities)}"
        )

        return output_path

    def _filter_activities(self, strategy: str) -> List[Dict[str, Any]]:
        """
        Filter provenance activities based on strategy.

        Args:
            strategy: Filter strategy ("successful" | "all" | "manual" | "replay")

        Returns:
            List of filtered activity dictionaries
        """
        if strategy in ("successful", "replay"):
            # Exclude activities with error markers. export() refuses "replay"
            # when any exist, and lists them as omitted for "successful".
            return [a for a in self.provenance.activities if not self._is_failed(a)]
        elif strategy == "all":
            return self.provenance.activities.copy()
        else:
            # Manual filtering not implemented yet
            logger.warning("Manual filter strategy not implemented, using 'successful'")
            return self._filter_activities("successful")

    @staticmethod
    def _ir_field(activity: Dict[str, Any], name: str, default: Any = None) -> Any:
        """One field of an activity's IR (a dict or an AnalysisStep); ``default`` if none."""
        ir = activity.get("ir")
        if isinstance(ir, dict):
            return ir.get(name, default)
        return getattr(ir, name, default) if ir is not None else default

    @classmethod
    def _ir_context(cls, activity: Dict[str, Any]) -> Dict[str, Any]:
        """The IR's execution context ({} if absent or not a mapping)."""
        context = cls._ir_field(activity, "execution_context")
        return context if isinstance(context, dict) else {}

    # Fixed labels for an excluded step (never the raw metadata strings).
    _EXCLUDED_STEP_TYPES = frozenset({"execute_custom_code", "custom_code_execution"})
    _EXCLUSION_REASONS = {"uses_attempt_figure_helpers": "uses_attempt_figure_helpers"}

    @classmethod
    def _step_label(cls, activity: Dict[str, Any]) -> str:
        """An activity's type; for a step whose IR gives an exclusion reason, a known
        type or ``custom step``."""
        label = str(activity.get("type") or "unknown")
        if "replay_excluded" not in cls._ir_context(activity):
            return label
        return label if label in cls._EXCLUDED_STEP_TYPES else "custom step"

    @classmethod
    def _omitted_label(cls, activity: Dict[str, Any]) -> str:
        """An activity's type; for a step whose IR gives an exclusion reason, a fixed
        label: a known type (else ``custom step``) and a known reason (else
        ``excluded``)."""
        label = cls._step_label(activity)
        context = cls._ir_context(activity)
        if "replay_excluded" not in context:
            return label
        reason = context.get("replay_excluded")
        known = reason if isinstance(reason, str) else None
        return f"{label} ({cls._EXCLUSION_REASONS.get(known, 'excluded')})"

    @staticmethod
    def _is_failed(activity: Dict[str, Any]) -> bool:
        """Whether a provenance activity records a failed step."""
        return bool(activity.get("error")) or activity.get("type") == "failed_operation"

    @staticmethod
    def _names_read(cells) -> set:
        """Names the given code cells read (Python ``Name`` loads)."""
        names = set()
        for cell in cells:
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            try:
                tree = ast.parse(cell.source)
            except SyntaxError:
                continue  # Only cells exempt from validation get here.
            names.update(
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
        return names

    def _extract_irs(self, activities: List[Dict[str, Any]]) -> List[AnalysisStep]:
        """
        Extract AnalysisStep IR objects from provenance activities.

        Args:
            activities: List of activity dictionaries from provenance

        Returns:
            List of AnalysisStep objects (may be empty if no IRs present)
        """
        irs = []

        for activity in activities:
            ir_dict = activity.get("ir")

            if ir_dict is not None:
                try:
                    if isinstance(ir_dict, AnalysisStep):
                        ir = ir_dict
                    elif isinstance(ir_dict, dict):
                        ir = AnalysisStep.from_dict(ir_dict)
                    else:
                        continue
                    # Check if this IR should be included in notebook export
                    if ir.exportable:
                        irs.append(ir)
                        logger.debug(f"Extracted IR for operation: {ir.operation}")
                    else:
                        logger.debug(
                            f"Skipping non-exportable IR for operation: {ir.operation}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize IR for activity "
                        f"{activity.get('type')}: {e}"
                    )

        return irs

    def _get_exportable_activity_ir_pairs(
        self, activities: List[Dict[str, Any]], select_on: str = "exportable"
    ) -> List[Tuple[Dict[str, Any], AnalysisStep]]:
        """
        Extract (activity, IR) pairs where IR exists and is selected for output.

        Two independent selections are available, because "should this appear in
        a notebook a user might publish?" and "is this step needed to re-run the
        session?" are unrelated questions:

        - ``select_on="exportable"`` (default, unchanged) — curation. Filters out
          orchestration activities (e.g. publication processing, metadata
          extraction) that log provenance but don't need notebook
          representation, keeping a curated notebook publishable while the
          complete audit trail stays in provenance.
        - ``select_on="replayable"`` — replay. Keeps the steps marked as needed to
          re-run the session, including ad-hoc computations and
          state mutations that curation deliberately drops. Selecting on
          ``exportable`` here would silently omit the step that produced the
          reported result and any mutation that deleted a modality mid-session,
          so a replay would diverge without saying so.

        Args:
            activities: List of activity dictionaries from provenance
            select_on: IR flag to select on ("exportable" | "replayable")

        Returns:
            List of (activity, AnalysisStep) tuples for selected steps only

        Raises:
            ValueError: If select_on is not a recognized flag
        """
        if select_on not in ("exportable", "replayable"):
            raise ValueError(
                f"select_on must be 'exportable' or 'replayable', got {select_on!r}"
            )

        pairs = []

        for activity in activities:
            ir_obj = activity.get("ir")

            if ir_obj is not None and isinstance(ir_obj, dict):
                # IR is a dict - deserialize to AnalysisStep (in-memory provenance)
                try:
                    ir = AnalysisStep.from_dict(ir_obj)
                    if getattr(ir, select_on):
                        pairs.append((activity, ir))
                        logger.debug(f"Included {select_on} IR: {ir.operation}")
                    else:
                        logger.debug(f"Filtered non-{select_on} IR: {ir.operation}")
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize IR for activity "
                        f"{activity.get('type')}: {e}"
                    )
            elif ir_obj is not None and isinstance(ir_obj, AnalysisStep):
                # IR is already an AnalysisStep (loaded from disk)
                if getattr(ir_obj, select_on):
                    pairs.append((activity, ir_obj))
                    logger.debug(f"Included {select_on} IR: {ir_obj.operation}")
                else:
                    logger.debug(f"Filtered non-{select_on} IR: {ir_obj.operation}")
            else:
                # Activity has no IR - provenance-only (e.g., orchestration)
                logger.debug(
                    f"Filtered provenance-only activity: {activity.get('type')}"
                )

        return pairs

    def _has_download_activity(
        self, exportable_pairs: List[Tuple[Dict[str, Any], AnalysisStep]]
    ) -> bool:
        """
        Check if any exportable IR represents a download operation.

        Download workflows produce their own data-loading code in the IR cell,
        so the generic ``_create_data_loading_cell()`` should be skipped to
        avoid loading a non-existent ``input_data`` parameter.

        Args:
            exportable_pairs: List of (activity, AnalysisStep) tuples

        Returns:
            True if a download activity is present
        """
        download_ops = {
            "geo_service.download_dataset",
            "sra_download",
            "pride.download.download_dataset",
            "metabolights.download.download_dataset",
        }
        if any(ir.operation in download_ops for _, ir in exportable_pairs):
            return True
        # Steps that build adata from a file themselves replace the generic
        # load only when they come first; a later one must not remove the
        # input load that the earlier cells depend on.
        self_loading_ops = {"peptidomics.import_sequences"}
        return bool(exportable_pairs) and (
            exportable_pairs[0][1].operation in self_loading_ops
        )

    def _create_header_cell(
        self,
        name: str,
        description: str,
        n_irs: int,
        mode: str = "successful",
        omitted_types: Optional[List[str]] = None,
    ) -> NotebookNode:
        """
        Create markdown header cell for notebook.

        Args:
            name: Notebook name
            description: Optional description
            n_irs: Number of IR objects extracted
            mode: Export mode the notebook was built with
            omitted_types: Types of recorded activities left out of the notebook

        Returns:
            Markdown cell with header content
        """
        coverage_pct = (
            n_irs / len(self.provenance.activities) * 100
            if self.provenance.activities
            else 0
        )

        if mode == "replay":
            mode_text = (
                "`replay`: the recorded steps marked replayable; "
                "steps that cannot be replayed are listed below"
            )
        else:
            mode_text = (
                f"`{mode}`: a curated subset of steps "
                "(see `/pipeline export --replay` for the replayable steps)"
            )
        omitted = list(dict.fromkeys(omitted_types or []))
        omitted_text = ", ".join(f"`{t}`" for t in omitted) if omitted else "none"

        header_content = f"""# {name}

**Generated from Lobster AI Session**

{description if description else "Bioinformatics analysis workflow"}

{self._migrated_header()}{self._outdated_header()}**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Lobster Version:** {self._get_lobster_version()}
**Export mode:** {mode_text}
**Steps included:** {n_irs} of {len(self.provenance.activities)} recorded activities
**Omitted activity types:** {omitted_text}
**Verification:** not run. This notebook has not been executed or compared with the session.
**IR Coverage:** {n_irs}/{len(self.provenance.activities)} activities ({coverage_pct:.1f}%)

---

## Workflow Summary

This notebook was automatically generated from a Lobster analysis session using
Service-Emitted Intermediate Representation (IR). Each included step carries the
code recorded for it.

**Session Statistics:**
- Activities: {len(self.provenance.activities)}
- Entities: {len(self.provenance.entities)}
- Data Modalities: {len(self.data_manager.modalities)}
- IR Objects: {n_irs}

---
"""
        return new_markdown_cell(header_content)

    def _migrated_header(self) -> str:
        """The header lines of a migrated session; empty for any other."""
        origin = self._migrated_origin()
        if origin is None:
            return ""
        text = h(
            "Migrated from a {source}: the cells were rebuilt from its record; "
            "the input file the session read was not recorded."
        ).format(source=self._source_words(origin))
        lines = f"**{text}**\n\n"
        if self.dangling_links:
            note = h(
                "Links in the workspace that point to missing files: {names} "
                "(restore or remove before replay)."
            ).format(names=", ".join(self.dangling_links))
            lines += f"**{note}**\n\n"
        if self.omitted_custom_code:
            note = h(
                "Custom code in a form Lobster did not write is left out "
                "(steps: {steps}); replay cannot check this notebook."
            ).format(steps=", ".join(map(str, self.omitted_custom_code)))
            lines += f"**{note}**\n\n"
        return lines

    def _outdated_header(self) -> str:
        """The header line naming the steps left out for their template."""
        if not self.outdated_steps:
            return ""
        note = h(
            "Steps recorded with code this Lobster no longer writes are left out "
            "(steps: {steps}); replay cannot check this notebook."
        ).format(steps=", ".join(map(str, self.outdated_steps)))
        return f"**{note}**\n\n"

    def _client_session_identity(self) -> Optional[str]:
        """The client session id bound on the data manager, or None when unknown.

        Never the provenance namespace, the data manager's timestamp or a
        directory name. Any failure to read it is reported as unknown. The
        result is a plain ``str`` copy of the checked text, so rendering it
        cannot run code from a ``str`` subclass.
        """
        try:
            value = getattr(self.data_manager, "client_session_id", None)
        except Exception:
            logger.warning(
                "Notebook export: client session identity is unknown (lookup_failed)"
            )
            return None
        if value is None:
            return None  # unbound: the ordinary case, not worth a warning
        identity = display_client_session_id(value)
        if identity is None:
            logger.warning(
                "Notebook export: client session identity is unknown (invalid_value)"
            )
        return identity

    def _create_integrity_manifest_cell(
        self, session_identity: Optional[str] = _OBSERVE
    ) -> NotebookNode:
        """
        Create the export-record cell: hashes and versions taken at export time.

        The record is informational. Nothing in the notebook checks it when the
        notebook runs, the provenance hash covers only the session ID, the
        provenance namespace and the activity/entity counts, and the record is
        not tamper evidence or proof of regulatory compliance. The wording must
        say so.

        Returns:
            Markdown cell with the export record
        """
        input_files = self._get_input_file_hashes()
        record = {
            "data_integrity_manifest": {
                "generated_at": datetime.now().isoformat(),
                "provenance": self._get_provenance_hash(session_identity),
                "input_files": input_files,
                "input_file_paths": self._get_input_file_paths(),
                "input_roles": self._get_input_roles(input_files),
                "system": self._get_system_info(),
            }
        }
        if self.outdated_steps:
            record["data_integrity_manifest"][OUTDATED_STEPS_KEY] = list(
                self.outdated_steps
            )
        origin = self._migrated_origin()
        if origin is not None:
            record["data_integrity_manifest"]["migrated"] = {
                **origin,
                "input": "not recorded by the source version",
                "dangling_links": list(self.dangling_links),
                "omitted_custom_code": list(self.omitted_custom_code),
            }

        content = f"""## Export record (hashes)

Recorded when this notebook was exported, for your own checks.

```json
{json.dumps(record, indent=2)}
```

**What is recorded**
- `input_files`: SHA-256 of each modality's saved `.h5ad` file at export time: the file Lobster last saved it to, else `data/<name>.h5ad`, else its auto-save `data/<name>_autosave.h5ad`, else `data/<name>_raw.h5ad`, else `<name>.h5ad` in the workspace folder. `input_file_paths` names the file that was hashed, relative to the workspace. `not_saved`: the session never saved that modality; `file_not_found`: it saved it and the file is gone.
- `input_roles`: `loaded` lists the name, SHA-256 and size of each file a modality was loaded from, read when it was loaded (no path); `saved` repeats `input_files`.
- `provenance.sha256`: SHA-256 of the session ID (`null` when the client session identity was unavailable or ambiguous at export time), the provenance namespace and the activity and entity counts only. It does not cover the content of the provenance record.
{MIGRATED_RECORD_LINE if origin is not None else ""}- `system`: Lobster and Python versions, the platform, and the short Git commit of the directory the export ran in (if any), as plain text.

**Limits**
- Nothing in this notebook checks these hashes when it runs, so changed input files are not detected automatically. Compare the hashes yourself if you need to.
- This record is not tamper evidence and not proof of regulatory compliance.
- It does not show that running this notebook reproduces the original session.

---
"""
        return new_markdown_cell(content)

    def _get_provenance_hash(
        self, session_identity: Optional[str] = _OBSERVE
    ) -> Dict[str, Any]:
        """SHA-256 of the session ID, the provenance namespace and the counts."""
        if session_identity is _OBSERVE:
            session_identity = self._client_session_identity()
        namespace = self.provenance.namespace
        try:
            prov_data = {
                "session_id": session_identity,
                "namespace": namespace,
                "activities_count": len(self.provenance.activities),
                "entities_count": len(self.provenance.entities),
            }
            prov_json = json.dumps(prov_data, sort_keys=True).encode("utf-8")
            prov_hash = hashlib.sha256(prov_json).hexdigest()

            return {
                "session_id": session_identity,
                "namespace": namespace,
                "sha256": prov_hash,
                "activities": len(self.provenance.activities),
                "entities": len(self.provenance.entities),
            }
        except Exception as e:
            logger.warning(f"Failed to hash provenance: {e}")
            return {
                "session_id": session_identity,
                "namespace": namespace,
                "sha256": "unavailable",
            }

    def _get_input_roles(self, saved: Dict[str, str]) -> Dict[str, Any]:
        """Which files the session loaded and which it saved, for a later check.

        ``loaded``: name, SHA-256 and size of each file a modality was loaded from,
        as read at load time (never a path). ``saved``: the hashes above.
        """
        loaded: List[Dict[str, Any]] = []
        try:
            for entity in self.data_manager.provenance.entities.values():
                meta = entity.get("metadata") or {}
                modality = meta.get("modality_name")
                if meta.get("loaded_file"):
                    entry = {
                        "modality": modality,
                        **meta["loaded_file"],
                        "adapter": meta.get("adapter"),
                    }
                elif meta.get("loaded_object"):
                    entry = {
                        "modality": modality,
                        "name": f"{modality} (in memory)",
                        **meta["loaded_object"],
                    }
                else:
                    continue
                same = (
                    entry["name"],
                    entry.get("sha256") or entry.get("content_sha256"),
                )
                if not any(
                    (e["name"], e.get("sha256") or e.get("content_sha256")) == same
                    for e in loaded
                ):
                    loaded.append(entry)
        except Exception as e:
            logger.warning(f"Failed to read the loaded files: {e}")
        return {"version": 2, "loaded": loaded, "saved": saved}

    def _saved_paths(self) -> Dict[str, Path]:
        """Where the data manager last saved each modality, by modality name.

        ``save_modality`` records a ``modality_data`` entity with the path it wrote;
        this is the only place that path is kept. Later saves win.
        """
        saved: Dict[str, Path] = {}
        for entity in list(getattr(self.provenance, "entities", {}).values()):
            if not isinstance(entity, dict) or entity.get("type") != "modality_data":
                continue
            name = (entity.get("metadata") or {}).get("modality_name")
            if name is not None and entity.get("uri"):
                saved[name] = Path(str(entity["uri"]))
        return saved

    def _modality_files(self) -> Dict[str, Optional[Path]]:
        """Each modality's saved file (``None`` if there is none), by ``<name>.h5ad``.

        The file the data manager saved the modality to comes first, if it is still
        there inside the workspace (a download names its file after the dataset, not
        the modality). Otherwise the names the data manager saves with, in this
        order: the save, its auto-save, the raw download in ``data/``, then
        ``<name>.h5ad`` in the workspace folder, where older versions looked. A name
        that leads out of the workspace is never looked for.
        """
        from lobster.core.runtime.data_manager import _autosave_file_name

        workspace = Path(self.data_manager.workspace_path)
        # Modalities are saved in the workspace's data/ folder.
        data_dir = Path(
            getattr(self.data_manager, "data_dir", None) or workspace / "data"
        )
        saved = self._saved_paths()
        files: Dict[str, Optional[Path]] = {}
        for modality_name in self.data_manager.modalities:
            found = None
            candidates = []
            if modality_name in saved:
                candidates.append(saved[modality_name])
            # A separator is what would lead out of data/ (".." alone is a file name).
            if not any(mark in modality_name for mark in ("/", "\\")):
                candidates += [
                    data_dir / f"{modality_name}.h5ad",
                    data_dir / _autosave_file_name(modality_name),
                    data_dir / f"{modality_name}_raw.h5ad",
                    workspace / f"{modality_name}.h5ad",
                ]
            for candidate in candidates:
                try:
                    if candidate.is_file() and _inside(candidate, workspace):
                        found = candidate
                        break
                except (OSError, ValueError):
                    continue
            files[f"{modality_name}.h5ad"] = found
        return files

    def _saved_modality_names(self) -> Optional[set]:
        """Modalities the session saved to a file (its save record); None if unknown."""
        if not getattr(self, "provenance", None):
            return None
        return set(self._saved_paths())

    def _get_input_file_hashes(self) -> Dict[str, str]:
        """Calculate SHA-256 hashes of input data files."""
        file_hashes = {}

        try:
            saved = self._saved_modality_names()
            for name, path in self._modality_files().items():
                if path is not None:
                    file_hashes[name] = self._calculate_file_hash(path)
                else:
                    # a file the session saved and lost, or a modality it never saved
                    file_hashes[name] = (
                        "file_not_found"
                        if saved is None or name[: -len(".h5ad")] in saved
                        else "not_saved"
                    )

        except Exception as e:
            logger.warning(f"Failed to hash input files: {e}")
            file_hashes["error"] = str(e)

        return file_hashes if file_hashes else {"note": "No input files hashed"}

    def _get_input_file_paths(self) -> Dict[str, str]:
        """The file each hashed modality was read from, relative to the workspace."""
        workspace = Path(self.data_manager.workspace_path)
        try:
            return {
                name: path.relative_to(workspace).as_posix()
                for name, path in self._modality_files().items()
                if path is not None
            }
        except Exception as e:
            logger.warning(f"Failed to name input files: {e}")
            return {}

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks for memory efficiency
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return "hash_failed"

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information including Lobster version and Git commit."""
        system_info = {
            "lobster_version": self._get_lobster_version(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "git_commit": _installed_commit(),
        }
        return system_info

    def _create_imports_cell(self, irs: List[AnalysisStep]) -> NotebookNode:
        """
        Create imports cell with deduplicated imports from all IRs.

        Args:
            irs: List of AnalysisStep objects

        Returns:
            Code cell with all required imports
        """
        if not irs:
            # Default imports if no IRs present
            imports = [
                "import numpy as np",
                "import pandas as pd",
                "import scanpy as sc",
                "import anndata",
            ]
        else:
            imports = extract_unique_imports(irs)
        if self._migrated_origin() is not None:
            # an earlier version recorded some entries that are not statements; the
            # record is not rewritten, only this cell leaves them out, visibly
            imports = [
                (
                    line
                    if line.lstrip().startswith(("import ", "from ", "#"))
                    else "# Not an import statement in the 1.2.x record, left out: "
                    + " ".join(line.split())
                )
                for line in imports
            ]

        imports_code = "# Required imports\n" + "\n".join(imports)

        logger.debug(f"Generated {len(imports)} import statements")

        return new_code_cell(imports_code)

    def _create_helper_code_cell(
        self, irs: List[AnalysisStep]
    ) -> Optional[NotebookNode]:
        """
        Create helper functions cell from IR helper_code fields.

        Deduplicates helper code across all IRs and combines them into
        a single cell. Returns None if no helpers are needed.

        Args:
            irs: List of AnalysisStep objects

        Returns:
            Code cell with helper functions, or None if no helpers
        """
        helpers = extract_unique_helper_code(irs)
        if not helpers:
            return None

        code = "# Helper functions (shared across analysis steps)\n\n"
        code += "\n\n".join(helpers)

        logger.debug(f"Generated helper code cell with {len(helpers)} helper(s)")
        return new_code_cell(code)

    def _create_parameters_cell(
        self, irs: List[AnalysisStep], read_names: Optional[set] = None
    ) -> NotebookNode:
        """
        Create the Papermill parameters cell.

        A parameter is offered only if the notebook code reads it. Recorded
        step settings are written into each step's own code, so offering them
        here would suggest an override that changes nothing.

        Args:
            irs: List of AnalysisStep objects
            read_names: Names the notebook's other code cells read

        Returns:
            Code cell tagged for Papermill parameter injection
        """
        read_names = read_names or set()

        code = """# Parameters (tagged for Papermill parameter injection)
# Override with: papermill notebook.ipynb output.ipynb -p param_name value
# Only names this notebook reads are listed. Each step's recorded settings are
# written into that step's code cell; edit the cell to change them.

"""
        # The runner always supplies these two, so they are always defined.
        standard = {
            "input_data": ("dataset.h5ad", "Path to the input data file"),
            "output_prefix": ("results", "Prefix for output files"),
        }
        for param_name, (value, comment) in standard.items():
            if param_name not in read_names:
                comment += " (not read by this notebook)"
            code += f"{param_name} = {value!r}  # {comment}\n"

        offered = {}
        for ir in irs:
            for param_name, value in ir.get_papermill_parameters().items():
                if param_name in standard or param_name not in read_names:
                    continue
                literal = repr(value)
                try:
                    ok = ast.literal_eval(literal) == value
                except (ValueError, SyntaxError, TypeError):
                    ok = False
                if ok:
                    offered.setdefault(param_name, literal)

        if offered:
            code += "\n# Analysis parameters read by the code below\n"
            for param_name, literal in sorted(offered.items()):
                code += f"{param_name} = {literal}\n"

        cell = new_code_cell(code)
        cell.metadata["tags"] = ["parameters"]

        logger.debug(f"Generated parameters cell with {len(offered)} IR parameters")

        return cell

    def _create_ml_run_reset_cell(self) -> NotebookNode:
        """Reset the ML-only named-input stopgap at the initial input boundary."""
        return new_code_cell(
            "# Reset ML notebook bindings for this run's initial input\n"
            "lobster_modalities = {}\n"
            "lobster_registered = {}\n"
            "lobster_epoch = 0\n"
            "globals().pop('lobster_last_output', None)\n"
        )

    def _create_data_loading_cell(self) -> NotebookNode:
        """
        Create cell for loading input data using IR if available.

        Returns:
            Code cell with data loading code
        """
        # Only the initial input is a run boundary. Individual analysis cells
        # and later in-run imports must retain their ambiguity protection.
        ml_reset = "\n" + self._create_ml_run_reset_cell().source

        # Try to find data loading IR from provenance
        for activity in self.provenance.activities:
            tool_name = activity.get("type", "")
            if tool_name == "load_dataset":
                ir_obj = activity.get("ir")
                if ir_obj:
                    try:
                        if isinstance(ir_obj, AnalysisStep):
                            ir = ir_obj
                        else:
                            ir = AnalysisStep.from_dict(ir_obj)
                    except Exception as e:
                        logger.warning(f"Failed to use IR for data loading: {e}")
                        continue
                    # A template that does not render fails the export, as for
                    # any analysis step: the stock loader would not be this load.
                    code = self._ir_to_code_cell(ir, validate=False).source
                    logger.debug("Using IR for data loading cell")
                    return new_code_cell(code + ml_reset)

        # Fallback to hardcoded cell if no IR found
        logger.debug("Using fallback hardcoded data loading cell")
        code = """# Load input data
import anndata as ad

adata = ad.read_h5ad(input_data)
print(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")
"""
        return new_code_cell(code + ml_reset)

    @staticmethod
    def _adata_name(entries: Any) -> Optional[str]:
        """The modality an activity read or wrote as ``adata``, if recorded."""
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("ir_name") == "adata":
                return str(entry["name"]) if entry.get("name") else None
        return None

    @staticmethod
    def _adata_names(entries: Any) -> List[str]:
        """Every modality an activity wrote as ``adata`` (one object holds them all)."""
        return [
            str(entry["name"])
            for entry in entries or []
            if isinstance(entry, dict)
            and entry.get("ir_name") == "adata"
            and entry.get("name")
        ]

    @classmethod
    def _reads_data_unnamed(cls, activity: Dict[str, Any], ir: AnalysisStep) -> bool:
        """The step reads ``adata`` but its record does not name the modality."""
        return "adata" in (ir.input_entities or []) and not cls._adata_name(
            activity.get("inputs")
        )

    @staticmethod
    def _custom_reads(
        activity: Dict[str, Any], ir: AnalysisStep
    ) -> Tuple[Optional[str], List[str], Optional[str]]:
        """A custom-code step's ``modality_name``, the modalities its code reads by
        a literal ``.h5ad`` path (the names the session gave it from memory), and
        the name the session stored its changed ``adata`` under, if it did."""
        from lobster.services.execution.custom_code_execution_service import (
            h5ad_references,
        )

        code = str(ir.parameters.get("code") or "")
        refs = h5ad_references(code)
        try:  # any .h5ad name in the code, e.g. WORKSPACE / "data" / "x.h5ad"
            refs += [
                node.value
                for node in ast.walk(ast.parse(code))
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.lower().endswith(".h5ad")
            ]
        except SyntaxError:
            pass
        names = dict.fromkeys(Path(ref).stem for ref in refs)
        written_back = next(
            (
                str(entry["name"])
                for entry in activity.get("outputs") or []
                if isinstance(entry, dict) and entry.get("name")
            ),
            None,
        )
        return ir.parameters.get("modality_name") or None, list(names), written_back

    @classmethod
    def _lineage(
        cls,
        steps: List[Dict[str, Any]],
        saved: List[str],
        custom: Optional[
            Dict[int, Tuple[Optional[str], List[str], Optional[str]]]
        ] = None,
    ) -> Tuple[
        Dict[int, List[str]],
        Dict[int, str],
        Dict[int, List[str]],
        Dict[int, Dict[str, str]],
        Dict[int, Dict[str, str]],
    ]:
        """Where the step cells must keep, restore and save modalities.

        Every step cell works on one ``adata``. A step whose recorded input is
        not the modality ``adata`` holds then starts from a copy kept when that
        modality was made. Each modality in *saved* is written right after the
        step that made it. Returns, per step position (-1: before the first
        step), the names to keep after it, the name to restore before it, and
        the saved names to write after it. A step without a recorded ``adata``
        input runs on what the step above left, as before; a saved name no step
        made is in none of them.

        A custom-code step (*custom*: position -> its ``modality_name``, the
        modalities its code reads and the name the session stored its changed
        ``adata`` under) runs on a copy of ``modality_name`` and leaves ``adata``
        unchanged, as the session left that modality; the cell itself keeps the
        changed copy under the stored name, and a save of that name is written
        after it. The fourth result gives, per custom step, where each name it
        reads comes from as of that step: ``input`` (the loaded input, not remade
        since), ``adata``, ``kept`` (a copy kept where it was last made) or
        ``missing``; under -1, the custom steps' ``modality_name`` the notebook
        does not compute. For a custom step only, a step whose recorded output
        is not ``adata`` made the modality its parameters name as the result
        (``result_modality_name``, ...) or else the one modality the session
        stored during it; it is kept from the variable the step's code assigns
        that output to, else from ``adata`` (the fifth result: position ->
        name -> variable). A step neither names stays unknown: the custom step
        stops on a read of it.
        """
        custom = custom or {}
        held: set = set()  # the modalities adata holds
        made_at: Dict[str, int] = {}  # where adata last became each modality
        saved_made_at: Dict[str, int] = {}  # the same, custom steps left out
        keep: Dict[int, List[str]] = {}
        rebind: Dict[int, str] = {}
        plans: Dict[int, Dict[str, str]] = {-1: {}}
        loaded: Optional[str] = None
        self_kept: set = set()  # kept by the custom cell that made them
        named_at: Dict[str, Tuple[int, str]] = {}  # non-adata results: (where, var)
        keep_vars: Dict[int, Dict[str, str]] = {}

        def keep_at(where: int, name: str, var: str = "adata") -> None:
            if name in self_kept:
                return
            names = keep.setdefault(where, [])
            if name not in names:
                names.append(name)
            if var != "adata":
                keep_vars.setdefault(where, {})[name] = var

        def latest(name: str) -> Optional[Tuple[int, str]]:
            """Where the notebook last made ``name`` and in which variable."""
            found = [(made_at[name], "adata")] if name in made_at else []
            if name in named_at:
                found.append(named_at[name])
            return max(found) if found else None

        for position, activity in enumerate(steps):
            if position in custom:
                source, results = custom[position][0], []
            else:
                source = cls._adata_name(activity.get("inputs"))
                results = cls._adata_names(activity.get("outputs"))
            if not held and source:  # the first step that says what it read
                held, made_at[source] = {source}, position - 1
                saved_made_at[source] = position - 1
                loaded = source
            if source and source not in held and source in made_at:
                keep_at(made_at[source], source)
                rebind[position] = source
                held = {source}
            elif (
                position in custom and source and source not in held and latest(source)
            ):
                where, var = latest(source)
                keep_at(where, source, var)
                rebind[position] = source
                held = {source}
            if position in custom:
                if source and source not in held:  # not computed by this notebook
                    plans[-1][source] = "missing"
                plan = plans[position] = {}
                for name in custom[position][1]:
                    if name == loaded and made_at.get(name) == -1:
                        plan[name] = "input"
                    elif name in held:
                        plan[name] = "adata"
                    elif latest(name):
                        where, var = latest(name)
                        keep_at(where, name, var)
                        plan[name] = "kept"
                    else:
                        plan[name] = "missing"
                for name in sorted(held):  # read by a path the code builds: free
                    plan.setdefault(name, "adata")
                if loaded and made_at.get(loaded) == -1:
                    plan.setdefault(loaded, "input")
                written_back = custom[position][2]
                if written_back:
                    made_at[written_back] = saved_made_at[written_back] = position
                    self_kept.add(written_back)
            else:
                named = cls._named_result(activity, source)
                if named:
                    named_at[named] = (position, cls._result_var(activity))
            if results:
                # A step may write several modalities (its input in place and a
                # saved copy): adata is each of them after it.
                held = set(results)
                for result in results:
                    self_kept.discard(result)
                    made_at[result] = position
                    if position not in custom:
                        # shortcut: a save is placed after the last non-custom
                        # step that made the name; a session that saved a name a
                        # custom step changed after that save would differ.
                        saved_made_at[result] = position
        save_at: Dict[int, List[str]] = {}
        for name in saved:
            if name in saved_made_at:
                save_at.setdefault(saved_made_at[name], []).append(name)
        return keep, rebind, save_at, plans, keep_vars

    @staticmethod
    def _named_result(activity: Dict[str, Any], source: Optional[str]) -> Optional[str]:
        """The modality a step made under its own name: the result its parameters
        name, else the one modality the session stored during the step as made
        from the step's input (``source``, else its ``modality_name`` parameter;
        ``stored_modalities``, recorded since 1.3.0 b7). None when neither says,
        or more than one was made."""
        from lobster.core.provenance.provenance import ProvenanceTracker

        params = activity.get("parameters") or {}
        for key in ProvenanceTracker._OUTPUT_PARAM_KEYS:
            if isinstance(params.get(key), str) and params[key]:
                return params[key]
        if not source and isinstance(params.get("modality_name"), str):
            source = params["modality_name"]
        made = {
            str(entry.get("name"))
            for entry in activity.get("stored_modalities") or []
            if isinstance(entry, dict)
            and source
            and entry.get("parent") == source
            and entry.get("name") not in (None, source)
        }
        return made.pop() if len(made) == 1 else None

    @classmethod
    def _result_var(cls, activity: Dict[str, Any]) -> str:
        """The variable a step's code leaves its result in: its recorded output
        when the code assigns it (``adata_x, stats, _ = ...``), else ``adata``."""
        outputs = cls._ir_field(activity, "output_entities") or []
        template = str(cls._ir_field(activity, "code_template") or "")
        for output in outputs[:1]:
            if isinstance(output, str) and output.isidentifier():
                target = (
                    rf"^[ \t]*(?:[\w ]+,\s*)*{re.escape(output)}\s*(?:,[\w ,]*)?=[^=]"
                )
                if re.search(target, template, re.M):
                    return output
        return "adata"

    @staticmethod
    def _split_saves(emitted_irs: Optional[List[AnalysisStep]]) -> set:
        """(name, file) of the saves create_ml_splits writes itself."""
        split_saves = set()
        for emitted_ir in emitted_irs or []:
            if (
                emitted_ir.tool_name == "create_ml_splits"
                and emitted_ir.parameters.get("save_result") is True
            ):
                parent = emitted_ir.parameters["modality_name"]
                names = [f"{parent}_train", f"{parent}_test"]
                if emitted_ir.parameters["validation_size"] > 0:
                    names.append(f"{parent}_validation")
                split_saves.update((name, f"{name}.h5ad") for name in names)
        return split_saves

    @staticmethod
    def _is_split_save(activity: Dict[str, Any], split_saves: set) -> bool:
        saved = activity.get("parameters", {})
        return (saved.get("name"), Path(saved.get("path") or "").name) in split_saves

    def _saves(
        self, emitted_irs: Optional[List[AnalysisStep]]
    ) -> Dict[str, Dict[str, Any]]:
        """The last save of each modality, by name in save order; the saves
        create_ml_splits writes itself are left out."""
        split_saves = self._split_saves(emitted_irs)
        saves: Dict[str, Dict[str, Any]] = {}
        for activity in self.provenance.activities:
            name = (activity.get("parameters") or {}).get("name")
            if (
                activity.get("type") == "save_dataset"
                and activity.get("ir")
                and name
                and not self._is_split_save(activity, split_saves)
            ):
                saves.pop(name, None)
                saves[name] = activity
        return saves

    def _closing_save(
        self, emitted_irs: Optional[List[AnalysisStep]]
    ) -> Optional[Dict[str, Any]]:
        """The save the closing save cell is built from: the most recent one
        with IR, none when that is a split create_ml_splits writes itself."""
        split_saves = self._split_saves(emitted_irs)
        for activity in reversed(self.provenance.activities):
            if activity.get("type", "") == "save_dataset":
                if self._is_split_save(activity, split_saves):
                    return None
                if activity.get("ir"):
                    return activity
        return None

    def _save_cell(self, activity: Dict[str, Any]) -> Optional[NotebookNode]:
        """The cell that writes what *activity* saved, from its IR."""
        ir_obj = activity.get("ir")
        try:
            if isinstance(ir_obj, AnalysisStep):
                ir = ir_obj
            else:
                ir = AnalysisStep.from_dict(ir_obj)
        except Exception as e:
            logger.warning(f"Failed to use IR for data saving: {e}")
            return None
        # A template that does not render fails the export, as for any analysis
        # step: the stock save would not write what the session saved.
        code = self._ir_to_code_cell(ir, validate=False).source
        path_lines = [
            line for line in code.splitlines() if line.startswith("output_path = ")
        ]
        if not path_lines:
            # Not the standard saver; keep its code as is.
            return new_code_cell(code)
        # Keep the session's file name and compression, but write through
        # the same guarded save as the fallback.
        compression = ir.parameters.get("compression")
        if compression in (None, "", "none"):
            compression = None
        logger.debug("Using IR for data saving cell")
        return new_code_cell(_guarded_save_code(path_lines[0], compression))

    def _create_data_saving_cell(
        self, emitted_irs: Optional[List[AnalysisStep]] = None
    ) -> NotebookNode:
        """
        Create the closing save cell, from the most recent save's IR if possible.

        Returns:
            Code cell with data saving code
        """
        # create_ml_splits explicitly writes its split objects. Its save_dataset
        # records precede the split activity at runtime; rendering their generic
        # `adata` saver here would overwrite the last split with the parent (or
        # with a later analysis output). Then the final adata is saved only
        # under the honest generic `_processed` filename.
        closing = self._closing_save(emitted_irs)
        cell = self._save_cell(closing) if closing is not None else None
        if cell is not None:
            return cell

        # Fallback to hardcoded cell if no IR found
        logger.debug("Using fallback hardcoded data saving cell")
        # The write is guarded because a pipeline this same exporter generates can
        # produce an object h5ad cannot serialise: sc.tl.filter_rank_genes_groups
        # sets filtered-out genes to NaN, so uns['rank_genes_groups_filtered']
        # holds non-string entries in a string field and h5py raises. An
        # unconditional write aborts the replay on the very last step, after every
        # analysis cell has already succeeded. Offending uns keys are dropped and
        # NAMED, never silently, and the retry preserves everything else.
        #
        # The except is deliberately broad. It was TypeError first, matched to the
        # one failure seen; a later template added uns entries that raise
        # AttributeError instead and the replay aborted on the last cell again.
        # Any serialisation failure should sanitise and retry, because reaching
        # this cell means every analysis step already succeeded.
        return new_code_cell(
            _guarded_save_code('output_path = f"{output_prefix}_processed.h5ad"')
        )

    def _create_doc_cell(
        self, activity: Dict[str, Any], step_number: int
    ) -> NotebookNode:
        """
        Create markdown documentation cell for activity.

        Args:
            activity: Activity dictionary from provenance
            step_number: Sequential step number

        Returns:
            Markdown cell documenting the analysis step
        """
        activity_type = (
            self._step_label(activity)
            if "replay_excluded" in self._ir_context(activity)
            else activity.get("type", "unknown")
        )
        params = activity.get("parameters", {})
        timestamp = activity.get("timestamp", "unknown")

        # Check if IR is present
        has_ir = activity.get("ir") is not None
        ir_status = "✓ IR available" if has_ir else "⚠ No IR (manual review needed)"

        doc_content = f"""## Step {step_number}: {activity_type}

**Timestamp:** {timestamp}
**Agent:** {activity.get("agent", "unknown")}
**Status:** {ir_status}

"""
        if activity.get("description"):
            doc_content += f"**Description:** {activity['description']}\n\n"

        if params:
            doc_content += "**Parameters:**\n"
            for key, value in params.items():
                # Format parameter value for display
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    value_str = (
                        f"[{', '.join(str(x) for x in value[:3])}...] "
                        f"(length: {len(value)})"
                    )
                else:
                    value_str = str(value)
                doc_content += f"- `{key}`: {value_str}\n"

        return new_markdown_cell(doc_content)

    def _ir_to_code_cell(self, ir: AnalysisStep, validate: bool = True) -> NotebookNode:
        """
        Convert AnalysisStep IR directly to executable code cell.

        Requires a valid IR object and never writes a placeholder or a cell
        that does not parse.

        Args:
            ir: AnalysisStep IR object with code template
            validate: Whether to validate generated Python syntax

        Returns:
            Code cell with executable Python code

        Raises:
            ValueError: If IR rendering fails
        """
        try:
            code = ir.render()
        except Exception as e:
            logger.error(f"Failed to render code for {ir.operation}: {e}")
            raise ValueError(f"Failed to render IR for {ir.operation}: {e}") from e

        if validate and ir.validates_on_export:
            try:
                ast.parse(code)
            except SyntaxError as e:
                # Fail the export rather than write a cell that cannot run.
                raise ValueError(
                    f"Generated code for step {ir.operation!r} is not valid "
                    f"Python ({e}); the notebook was not exported"
                ) from e
            logger.debug(f"Validated syntax for {ir.operation}")

        return new_code_cell(code)

    def _create_footer_cell(
        self,
        excluded_activity_types: List[str],
        select_on: str = "exportable",
        failed_types: Optional[List[str]] = None,
    ) -> NotebookNode:
        """
        Create markdown footer cell.

        Args:
            excluded_activity_types: Activity types excluded from the selected IR count
            select_on: IR flag used to select activities
            failed_types: Types of failed activities left out of the notebook

        Returns:
            Markdown cell with export instructions
        """
        if excluded_activity_types:
            activity_names = ", ".join(
                f"`{name}`" for name in dict.fromkeys(excluded_activity_types)
            )
            export_status = (
                f"The following activities are excluded from the {select_on} IR count: "
                f"{activity_names}. Review the IR coverage and provenance summary."
            )
        else:
            export_status = (
                "Notebook export is complete. "
                f"All selected activities have {select_on} IR."
            )

        if failed_types:
            failed_names = ", ".join(f"`{t}`" for t in dict.fromkeys(failed_types))
            export_status += (
                f"\n\n**Failed steps not in this notebook:** {failed_names}. A failed "
                "step may have changed data before it failed, so this notebook may "
                "not reproduce the session."
            )

        footer_content = f"""---

## Results Export

{export_status}

### Next Steps

1. **Review Results**: Check output files and visualizations
2. **Version Control**: Commit this notebook to Git
3. **Share**: Share the notebook with collaborators
4. **Re-run**: Use Papermill to execute on new data

### Usage Example

```bash
# Execute notebook with new data
papermill notebook.ipynb output.ipynb \\
    -p input_data "new_dataset.h5ad" \\
    -p output_prefix "new_analysis"
```

Only the names in the parameters cell can be overridden. To change a step's
settings, edit that step's code cell.

### IR Coverage Note

This notebook was generated using Service-Emitted IR. Review the IR coverage and
provenance summary for activity counts and excluded activity types.
IR coverage alone does not show that running it gives the same results.

---

*Generated with [Lobster AI](https://github.com/the-omics-os/lobster) - Multi-Agent Bioinformatics Platform*
*Using Service-Emitted IR Architecture for automatic code generation*
"""
        return new_markdown_cell(footer_content)

    def _create_provenance_summary_cell(
        self,
        total_activities: int,
        exportable_count: int,
        excluded_activity_types: List[str],
        select_on: str = "exportable",
        session_identity: Optional[str] = _OBSERVE,
    ) -> NotebookNode:
        """
        Create summary cell explaining provenance vs notebook content.

        This cell provides transparency about which activities are included
        in the executable notebook vs which are recorded only in provenance
        for audit trail purposes.

        Args:
            total_activities: Total number of activities in provenance
            exportable_count: Number of activities with the selected IR flag
            excluded_activity_types: Activity types excluded from the selected IR count
            select_on: IR flag used to select activities

        Returns:
            Markdown cell with provenance summary
        """
        if session_identity is _OBSERVE:
            session_identity = self._client_session_identity()
        session_line = (
            f"Session ID: `{session_identity}`"
            if session_identity is not None
            else "Session ID: unknown (the client session identity was unavailable "
            "or ambiguous at export time)"
        )
        filtered_count = total_activities - exportable_count
        activity_names = (
            ", ".join(f"`{name}`" for name in dict.fromkeys(excluded_activity_types))
            or "None"
        )

        content = f"""---

## Provenance

| Metric | Count |
|--------|-------|
| **Executable Steps** | {exportable_count} |
| **Provenance-Only Activities** | {filtered_count} |
| **Total Activities** | {total_activities} |

**What's Included:**
- **{exportable_count} activities** have {select_on} IR.
- **{filtered_count} activities** are excluded from the {select_on} IR count.
- **Excluded activity types:** {activity_names}

**Full Provenance Record:**
- {session_line}
- Activity records are tracked separately from the notebook; disk persistence is optional.

---
"""
        return new_markdown_cell(content)

    def _create_metadata(
        self,
        n_irs: int,
        n_activities: int,
        session_identity: Optional[str] = _OBSERVE,
    ) -> Dict[str, Any]:
        """
        Create notebook metadata with provenance link and IR stats.

        Args:
            n_irs: Number of IR objects
            n_activities: Number of activities

        Returns:
            Metadata dictionary for notebook
        """
        if session_identity is _OBSERVE:
            session_identity = self._client_session_identity()
        return {
            "source_session_id": session_identity,
            "provenance_namespace": self.provenance.namespace,
            "created_by": "lobster",
            "created_at": datetime.now().isoformat(),
            "lobster_version": self._get_lobster_version(),
            "dependencies": self._snapshot_dependencies(),
            "source_provenance_summary": {
                "n_activities": len(self.provenance.activities),
                "n_entities": len(self.provenance.entities),
                "n_agents": len(self.provenance.agents),
            },
            "ir_statistics": {
                "n_irs_extracted": n_irs,
                "n_activities": n_activities,
                "coverage_percent": (
                    n_irs / n_activities * 100 if n_activities > 0 else 0
                ),
            },
        }

    def _snapshot_dependencies(self) -> Dict[str, str]:
        """
        Snapshot current package versions.

        Returns:
            Dictionary of package versions
        """
        versions = {}

        # Core bioinformatics packages
        packages = ["scanpy", "anndata", "pandas", "numpy", "scipy", "scikit-learn"]

        for package in packages:
            try:
                module = __import__(package)
                versions[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                pass

        # Add Python version
        versions["python"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        return versions

    def _get_lobster_version(self) -> str:
        """Get Lobster version."""
        try:
            from lobster.version import __version__

            return __version__
        except ImportError:
            return "unknown"
