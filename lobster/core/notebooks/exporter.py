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
import subprocess  # nosec B404 # Only fixed metadata argv is executed below; no shell commands.
import sys
from datetime import datetime
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

logger = logging.getLogger(__name__)

# Default for helpers called outside export(): take their own identity reading.
_OBSERVE: Any = object()


def _guarded_save_code(path_line: str, compression: Optional[str] = None) -> str:
    """Code for the final save cell.

    ``path_line`` assigns ``output_path``. If h5ad cannot write some uns
    entries, the cell writes a copy without them, names them in
    ``uns['lobster_unsaved_uns_keys']`` and leaves the in-memory adata as is.
    """
    kwargs = f", compression={compression!r}" if compression else ""
    return f"""# Save processed data
{path_line}
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
print(f"Saved processed data to: {{output_path}}")
"""


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
        logger.debug("Initialized NotebookExporter with IR support")

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
        irs = [ir for _, ir in exportable_pairs]

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
        for idx, (activity, ir) in enumerate(exportable_pairs, start=1):
            activity_type = activity.get("type", "")
            if activity_type in ("load_dataset", "save_dataset"):
                continue

            # Add documentation cell
            notebook.cells.append(self._create_doc_cell(activity, idx))

            # Add executable code cell from IR (no placeholder fallback)
            code_cell = self._ir_to_code_cell(ir, validate=validate_syntax)
            notebook.cells.append(code_cell)

        # Add data saving cell
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

**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
        record = {
            "data_integrity_manifest": {
                "generated_at": datetime.now().isoformat(),
                "provenance": self._get_provenance_hash(session_identity),
                "input_files": self._get_input_file_hashes(),
                "system": self._get_system_info(),
            }
        }

        content = f"""## Export record (hashes)

Recorded when this notebook was exported, for your own checks.

```json
{json.dumps(record, indent=2)}
```

**What is recorded**
- `input_files`: SHA-256 of each modality's `.h5ad` file in the workspace at export time (`file_not_found` if the file was missing).
- `provenance.sha256`: SHA-256 of the session ID (`null` when the client session identity was unavailable or ambiguous at export time), the provenance namespace and the activity and entity counts only. It does not cover the content of the provenance record.
- `system`: Lobster and Python versions, the platform, and the short Git commit of the directory the export ran in (if any), as plain text.

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

    def _get_input_file_hashes(self) -> Dict[str, str]:
        """Calculate SHA-256 hashes of input data files."""
        file_hashes = {}

        try:
            # Hash modality files from workspace
            for modality_name in self.data_manager.modalities:
                modality_path = (
                    self.data_manager.workspace_path / f"{modality_name}.h5ad"
                )
                if modality_path.exists():
                    file_hash = self._calculate_file_hash(modality_path)
                    file_hashes[f"{modality_name}.h5ad"] = file_hash
                else:
                    file_hashes[f"{modality_name}.h5ad"] = "file_not_found"

        except Exception as e:
            logger.warning(f"Failed to hash input files: {e}")
            file_hashes["error"] = str(e)

        return file_hashes if file_hashes else {"note": "No input files hashed"}

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
        }

        # Try to get Git commit hash
        try:
            git_commit = (
                subprocess.check_output(  # nosec B603, B607 # Fixed git metadata argv, no shell or user command interpolation.
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
            system_info["git_commit"] = git_commit[:8]  # Short hash
        except Exception:
            system_info["git_commit"] = "unavailable"

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
                        code = ir.render()
                        logger.debug("Using IR for data loading cell")
                        return new_code_cell(code + ml_reset)
                    except Exception as e:
                        logger.warning(f"Failed to use IR for data loading: {e}")

        # Fallback to hardcoded cell if no IR found
        logger.debug("Using fallback hardcoded data loading cell")
        code = """# Load input data
import anndata as ad

adata = ad.read_h5ad(input_data)
print(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")
"""
        return new_code_cell(code + ml_reset)

    def _create_data_saving_cell(
        self, emitted_irs: Optional[List[AnalysisStep]] = None
    ) -> NotebookNode:
        """
        Create cell for saving output data using IR if available.

        Returns:
            Code cell with data saving code
        """
        # create_ml_splits explicitly writes its split objects. Its save_dataset
        # records precede the split activity at runtime; rendering their generic
        # `adata` saver here would overwrite the last split with the parent (or
        # with a later analysis output). Only selected, saving split IRs qualify.
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

        # Try to find data saving IR from provenance
        for activity in reversed(self.provenance.activities):  # Check most recent first
            tool_name = activity.get("type", "")
            if tool_name == "save_dataset":
                saved = activity.get("parameters", {})
                if (
                    saved.get("name"),
                    Path(saved.get("path") or "").name,
                ) in split_saves:
                    # All requested split artifacts already exist. Save final
                    # adata only under the honest generic `_processed` filename.
                    break
                ir_obj = activity.get("ir")
                if ir_obj:
                    try:
                        if isinstance(ir_obj, AnalysisStep):
                            ir = ir_obj
                        else:
                            ir = AnalysisStep.from_dict(ir_obj)
                        code = ir.render()
                        path_lines = [
                            line
                            for line in code.splitlines()
                            if line.startswith("output_path = ")
                        ]
                        if not path_lines:
                            # Not the standard saver; keep its code as is.
                            return new_code_cell(code)
                        else:
                            # Keep the session's file name and compression, but
                            # write through the same guarded save as below.
                            compression = ir.parameters.get("compression")
                            if compression in (None, "", "none"):
                                compression = None
                            logger.debug("Using IR for data saving cell")
                            return new_code_cell(
                                _guarded_save_code(path_lines[0], compression)
                            )
                    except Exception as e:
                        logger.warning(f"Failed to use IR for data saving: {e}")

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

*Generated with [Lobster AI](https://github.com/OmicsOS/lobster) - Multi-Agent Bioinformatics Platform*
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
        import os

        if session_identity is _OBSERVE:
            session_identity = self._client_session_identity()
        return {
            "source_session_id": session_identity,
            "provenance_namespace": self.provenance.namespace,
            "created_by": os.getenv("USER", "unknown"),
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
